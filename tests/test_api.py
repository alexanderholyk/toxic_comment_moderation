# tests/test_api.py
import os
import uuid
import joblib
import pytest
import numpy as np
import pandas as pd
import numpy as np
from starlette.testclient import TestClient
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ---- Move DummyModel to MODULE scope so joblib can pickle it ----
class DummyModel:
    def __init__(self):
        # small vectorizer so joblib load doesn't fail on attributes
        self._pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=10)),
        ])
        X = pd.Series(["ok", "great", "bad", "idiot", "disgusting"])
        y = np.array([0, 0, 0, 1, 1])
        # just fit to have a fitted vectorizer; clf isn’t actually used
        self._pipe.fit(X, y)

    # API under test expects either decision_function or predict_proba
    def decision_function(self, texts):
        out = []
        for t in texts:
            s = (t or "").lower()
            arr = np.zeros(6, dtype=float)  # 6 labels
            if any(k in s for k in ["idiot", "disgust"]):
                arr[:] = 2.0
            elif "bad" in s:
                arr[:] = 0.8
            else:
                arr[:] = -2.0
            out.append(arr)
        return np.vstack(out)


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture(scope="session")
def fake_artifact_dir(tmp_path_factory):
    """Create a tiny dummy 'model.pkl' that exposes decision_function -> (n,6)."""
    tmp = tmp_path_factory.mktemp("artifact")
    (tmp / "model.pkl").parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(DummyModel(), tmp / "model.pkl")
    return str(tmp)

class _FakeWandbArtifact:
    def __init__(self, artifact_dir, version="vTEST"):
        self._dir = artifact_dir
        self.version = version
    def download(self, root):
        return self._dir

class _FakeWandbApi:
    def __init__(self, artifact_dir):
        self._artifact_dir = artifact_dir
    def artifact(self, spec):
        return _FakeWandbArtifact(self._artifact_dir, version="vTEST")

@pytest.fixture
def app_client(monkeypatch, fake_artifact_dir):
    # Env the API expects
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_MODEL_NAME", "test-model")
    monkeypatch.setenv("MODEL_CACHE_DIR", "/tmp/wandb_models_test")
    # disable real DB writes
    monkeypatch.setenv("DEV_DB_SKIP", "1")

    # Stub wandb.Api()
    import wandb
    monkeypatch.setattr(wandb, "Api", lambda: _FakeWandbApi(fake_artifact_dir))

    # Import AFTER patching
    from src.api.main import app
    return TestClient(app)

# --- Tests ------------------------------------------------------------------

def test_health_ok(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body and "version" in body

def test_predict_returns_scores_and_labels(app_client):
    r = app_client.post("/predict", json={"comment_text": "You are disgusting."})
    assert r.status_code == 200
    body = r.json()
    assert "labels" in body and isinstance(body["labels"], list)
    assert "scores" in body and isinstance(body["scores"], dict)
    assert "model_version" in body

def test_cached_path_still_logs_request_id(app_client, monkeypatch):
    # Patch fetch_cached to simulate a cache hit for the same text hash
    from src.api import main as api_main
    cached = {
        "scores": {k: 0.1 for k in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]},
        "labels": [],
        "model_version": str(api_main.MODEL_VERSION),
    }
    monkeypatch.setattr(api_main, "fetch_cached", lambda _ih: cached)

    # Ensure log_prediction is called and returns a UUID
    calls = {}
    def _fake_log(*args, **kwargs):
        calls["called"] = True
        return str(uuid.uuid4())
    monkeypatch.setattr(api_main, "log_prediction", _fake_log)

    r = app_client.post("/predict", json={"comment_text": "This is fine."})
    assert r.status_code == 200
    body = r.json()
    assert body["scores"] == cached["scores"]
    assert body["labels"] == cached["labels"]
    uuid.UUID(body["request_id"])  # valid UUID string
    assert calls.get("called", False)