from fastapi.testclient import TestClient
from src.api.main import app

def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200

def test_predict_empty():
    c = TestClient(app)
    r = c.post("/predict", json={"comment_text": "You are awful!"})
    assert r.status_code == 200
    js = r.json()
    assert "labels" in js and "scores" in js