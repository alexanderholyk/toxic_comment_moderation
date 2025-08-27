import os, time, joblib, glob
from fastapi import FastAPI, Body, HTTPException
from sqlalchemy import text
from src.api.schemas import PredictRequest, PredictResponse
from src.api.db import log_prediction, fetch_cached, get_conn, engine
from src.utils.hashing import text_hash
import numpy as np
import wandb
import logging

logger = logging.getLogger(__name__)

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

WANDB_ENTITY = os.environ["WANDB_ENTITY"]
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "toxic-moderation")
MODEL_NAME = os.environ.get("WANDB_MODEL_NAME", "toxic-comment")
PROD_SPEC = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{MODEL_NAME}:production"

# Download latest production artifact to a cache dir (reused across restarts)
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/tmp/wandb_models")
os.makedirs(CACHE_DIR, exist_ok=True)
artifact_dir = wandb.Api().artifact(PROD_SPEC).download(root=CACHE_DIR)

# find model.pkl
candidates = glob.glob(os.path.join(artifact_dir, "**", "model.pkl"), recursive=True)
if not candidates:
    raise RuntimeError("model.pkl not found in downloaded artifact")
MODEL_PATH = candidates[0]
model = joblib.load(MODEL_PATH)

# Track the exact artifact version loaded
MODEL_VERSION = wandb.Api().artifact(PROD_SPEC).version

app = FastAPI(title="Toxic Moderation API", version="1.0")


@app.get("/health")
def health():
    db = "down"
    if engine is not None:
        try:
            with engine.connect() as c:
                c.execute(text("select 1"))
            db = "ok"
        except Exception:
            db = "down"
    return {"status": "ok", "model": MODEL_NAME, "version": MODEL_VERSION, "db": db}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    ih = text_hash(req.comment_text)

    # Try to reuse cached result, but still log this request with a real UUID
    cached = None
    try:
        cached = fetch_cached(ih)
    except Exception:
        logger.exception("DB cache fetch failed")
        cached = None

    if cached and cached["model_version"] == str(MODEL_VERSION):
        scores = cached["scores"]
        labels = cached["labels"]
        latency_ms = 0  # cache path

        req_id = "unlogged"
        try:
            req_id = log_prediction(
                req.comment_text, ih, scores, labels,
                MODEL_NAME, str(MODEL_VERSION), latency_ms
            )
        except Exception:
            logger.exception("DB logging failed (cache path)")

        return PredictResponse(
            labels=labels,
            scores=scores,
            model_version=str(MODEL_VERSION),
            request_id=req_id,  # real UUID now
        )

    # No suitable cache → run inference
    t0 = time.time()
    if hasattr(model, "decision_function"):
        y = model.decision_function([req.comment_text])
        y = 1 / (1 + np.exp(-y))
    else:
        y = model.predict_proba([req.comment_text])

    scores = {lab: float(y[0, i]) for i, lab in enumerate(LABELS)}
    labels = [lab for lab, s in scores.items() if s >= 0.5]
    latency_ms = int((time.time() - t0) * 1000)

    req_id = "unlogged"
    try:
        req_id = log_prediction(
            req.comment_text, ih, scores, labels, MODEL_NAME, str(MODEL_VERSION), latency_ms
        )
    except Exception:
        logger.exception("DB logging failed")

    return PredictResponse(
        labels=labels, scores=scores, model_version=str(MODEL_VERSION), request_id=req_id
    )


@app.post("/feedback")
def feedback(
    request_id: str = Body(..., embed=True),
    correct: bool = Body(..., embed=True),
    true_labels: list[str] | None = Body(None, embed=True),
    notes: str | None = Body(None, embed=True),
):
    with get_conn() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM prediction_logs WHERE request_id = :rid LIMIT 1"),
            {"rid": request_id},
        ).first()
        if not exists:
            raise HTTPException(status_code=404, detail="request_id not found")

        conn.execute(
            text("""
                INSERT INTO feedback (request_id, correct, true_labels, notes)
                VALUES (:rid, :corr, :tl, :nt)
            """),
            {"rid": request_id, "corr": correct, "tl": true_labels, "nt": notes},
        )

    return {"status": "ok"}