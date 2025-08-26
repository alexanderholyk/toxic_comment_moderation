import os, time
from fastapi import FastAPI
from src.api.schemas import PredictRequest, PredictResponse
from src.api.db import log_prediction, fetch_cached
from src.utils.hashing import text_hash
import mlflow
from mlflow.tracking import MlflowClient

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = os.environ.get("MODEL_NAME", "toxic-comment")
MODEL_URI = f"models:/{MODEL_NAME}/Production"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)
MODEL_VERSION = MlflowClient().get_registered_model(MODEL_NAME).latest_versions[0].version

app = FastAPI(title="Toxic Moderation API", version="1.0")

@app.get("/health")
def health():
    return {"status":"ok","model":MODEL_NAME,"version":MODEL_VERSION}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    ih = text_hash(req.comment_text)
    cached = fetch_cached(ih)
    if cached and cached["model_version"] == str(MODEL_VERSION):
        return PredictResponse(labels=cached["labels"], scores=cached["scores"], model_version=str(MODEL_VERSION))

    t0 = time.time()
    # model expects array-like of strings
    y = model.predict([req.comment_text])  # with our pipeline, this returns probabilities per class
    # If pipeline returns decision_function, convert to prob-like
    import numpy as np
    if y.ndim == 2 and (y.max() > 1 or y.min() < 0):
        y = 1/(1+np.exp(-y))
    scores = {lab: float(y[0, i]) for i, lab in enumerate(LABELS)}
    labels = [lab for lab, s in scores.items() if s >= 0.5]
    latency_ms = int((time.time() - t0)*1000)

    rid = log_prediction(req.comment_text, ih, scores, labels, MODEL_NAME, str(MODEL_VERSION), latency_ms)
    return PredictResponse(labels=labels, scores=scores, model_version=str(MODEL_VERSION))