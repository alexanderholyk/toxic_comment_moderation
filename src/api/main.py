import os, time, joblib, glob
from fastapi import FastAPI
from src.api.schemas import PredictRequest, PredictResponse
from src.api.db import log_prediction, fetch_cached
from src.utils.hashing import text_hash
import numpy as np
import wandb

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

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
    return {"status":"ok", "model": MODEL_NAME, "version": MODEL_VERSION}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    ih = text_hash(req.comment_text)
    cached = fetch_cached(ih)
    if cached and cached["model_version"] == str(MODEL_VERSION):
        return PredictResponse(labels=cached["labels"], scores=cached["scores"], model_version=str(MODEL_VERSION))

    t0 = time.time()
    # get probabilities from sklearn pipeline
    if hasattr(model, "decision_function"):
        y = model.decision_function([req.comment_text])
        y = 1/(1+np.exp(-y))
    else:
        y = model.predict_proba([req.comment_text])

    scores = {lab: float(y[0, i]) for i, lab in enumerate(LABELS)}
    labels = [lab for lab, s in scores.items() if s >= 0.5]
    latency_ms = int((time.time() - t0)*1000)

    log_prediction(req.comment_text, ih, scores, labels, MODEL_NAME, str(MODEL_VERSION), latency_ms)
    return PredictResponse(labels=labels, scores=scores, model_version=str(MODEL_VERSION))