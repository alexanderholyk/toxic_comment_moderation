# tests/test_metrics.py
import numpy as np
from src.training.train import evaluate, LABELS

def test_evaluate_keys_and_ranges():
    y_true = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [1,0,1,0,0,0],
        [0,0,0,0,1,0],
    ])
    y_prob = np.array([
        [0.9, 0.1, 0.2, 0.1, 0.2, 0.1],
        [0.2, 0.95,0.1, 0.1, 0.1, 0.1],
        [0.8, 0.2, 0.85,0.1, 0.1, 0.1],
        [0.2, 0.1, 0.1, 0.1, 0.8, 0.1],
    ])
    m = evaluate(y_true, y_prob, threshold=0.5)
    assert "f1_macro" in m and "f1_micro" in m
    for lab in LABELS:
        assert f"auc_{lab}" in m
    assert 0 <= m["f1_macro"] <= 1
    assert 0 <= m["f1_micro"] <= 1