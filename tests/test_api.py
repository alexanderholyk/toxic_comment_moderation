# tests/test_api.py

import os
import tempfile
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _env_offline():
    # Keep W&B offline and provide dummy identifiers so imports don't fail
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_ENTITY", "dummy")
    os.environ.setdefault("WANDB_PROJECT", "dummy")
    os.environ.setdefault("WANDB_MODEL_NAME", "dummy")
    # Provide a writable cache dir for artifact download
    os.environ.setdefault("MODEL_CACHE_DIR", tempfile.gettempdir())


def _install_fakes(monkeypatch):
    """
    Install fakes for:
      - wandb.Api().artifact(...).download() and .version
      - joblib.load(model_path) -> returns a fake sklearn-like model
      - DB logging functions (no real Postgres needed)
    """
    # --- Fake W&B API / Artifact ---
    class _FakeArtifact:
        def __init__(self, version="test-1"):
            self._version = version

        @property
        def version(self):
            return self._version

        def download(self, root=None):
            # return an empty temp directory; joblib.load is mocked anyway
            return tempfile.mkdtemp()

    class _FakeApi:
        def artifact(self, spec: str):
            # spec like "entity/project/name:production"
            return _FakeArtifact(version="test-1")

    # Patch global wandb.Api() *before* importing the API module
    monkeypatch.setattr("wandb.Api", lambda: _FakeApi(), raising=True)

    # --- Fake model object ---
    class _FakeModel:
        # mimic sklearn decision_function -> ndarray [n_samples, 6]
        def decision_function(self, X):
            import numpy as np
            n = len(X)
            return np.zeros((n, 6), dtype=float)

    # Ensure joblib.load returns our fake model regardless of file path
    monkeypatch.setattr("joblib.load", lambda _path: _FakeModel(), raising=True)

    # --- Stub out DB I/O used by the API ---
    import src.api.db as db
    monkeypatch.setattr(db, "log_prediction", lambda *a, **k: "test-request-id", raising=True)
    monkeypatch.setattr(db, "fetch_cached", lambda *a, **k: None, raising=True)


def test_health(monkeypatch):
    _install_fakes(monkeypatch)

    # Import after fakes are installed so startup code uses them
    from src.api import main  # noqa: WPS433 (import inside function)
    importlib.reload(main)    # ensure patches are in effect on re-import

    client = TestClient(main.app)
    r = client.get("/health")
    assert r.status_code == 200
    js = r.json()
    assert js["status"] == "ok"
    assert "version" in js


def test_predict_smoke(monkeypatch):
    _install_fakes(monkeypatch)

    from src.api import main  # noqa: WPS433
    importlib.reload(main)

    client = TestClient(main.app)
    r = client.post("/predict", json={"comment_text": "You stink"})
    assert r.status_code == 200
    js = r.json()
    assert "labels" in js and isinstance(js["labels"], list)
    assert "scores" in js and isinstance(js["scores"], dict)
    assert len(js["scores"]) == 6  # six toxicity labels