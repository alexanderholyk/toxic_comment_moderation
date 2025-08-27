# train.py

# Loads train.csv, splits into train/test sets, 
# builds TF-IDF vectorizer and 
# Logistic Regression classifier pipeline,
# and logs parameters, metrics, and model artifacts into MLflow

# requires .env file with variables; see README.md

# Example run from project root:
# python -m src.training.train \
#   --train_csv data/train.csv \
#   --max_features 50000

import os
import json
import argparse
import time
import tempfile
import joblib
import hashlib
import pathlib
import subprocess
import wandb
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()  # read .env from project root

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }
    for i, lab in enumerate(LABELS):
        try:
            metrics[f"auc_{lab}"] = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
        except ValueError:
            metrics[f"auc_{lab}"] = float("nan")
    return metrics

def main(args):
    # ---- W&B init (single run) ----
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "toxic-moderation"),
        entity=os.environ.get("WANDB_ENTITY", None),
        job_type="train",
        config={
            "git_sha": _git_sha(),
            "random_state": 42,
            "test_size": 0.2,
            "vectorizer": {
                "max_features": args.max_features,
                "ngram_range": (1, 2),
                "min_df": 3,
                "max_df": 0.9,
                "sublinear_tf": True,
            },
            "classifier": {
                "family": "LogisticRegression",
                "solver": "liblinear",  # or "saga"
                "C": 1.0,
                "penalty": "l2",
                "max_iter": 1000,
                "class_weight": "balanced",
                "tol": 1e-4,
            },
            "labels": LABELS,
        },
    )
    run.log_code(root="src")

    # ---- Data & data versioning ----
    df = pd.read_csv(args.train_csv)
    train_hash = _sha256(args.train_csv)
    wandb.config.update(
        {"data": {"train_csv": str(pathlib.Path(args.train_csv).resolve()), "sha256": train_hash}},
        allow_val_change=True,
    )

    ds = wandb.Artifact(
        name="jigsaw-train",
        type="dataset",
        metadata={"source": "Jigsaw Toxic Comment", "sha256": train_hash},
    )
    ds.add_file(args.train_csv, name="train.csv")
    logged_ds = run.log_artifact(ds, aliases=["raw"])
    logged_ds.wait()
    run.use_artifact(f"{logged_ds.entity}/{logged_ds.project}/jigsaw-train:raw")

    X = df["comment_text"].astype(str).values
    y = df[LABELS].values.astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=(y.sum(axis=1) > 0)
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=args.max_features,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
            dtype=np.float32
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                penalty="l2",
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                tol=1e-4
            )
        ))
    ])

    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    wandb.log({"train_seconds": time.time() - t0})

    # decision_function -> sigmoid to [0,1]; else use predict_proba
    if hasattr(pipe, "decision_function"):
        y_prob = pipe.decision_function(X_te)
        y_prob = 1 / (1 + np.exp(-y_prob))
    else:
        y_prob = pipe.predict_proba(X_te)

    metrics = evaluate(y_te, y_prob, threshold=0.5)
    wandb.log(metrics)

    # ---- Save model & register in W&B Model Registry via Artifacts ----
    model_name = os.environ.get("WANDB_MODEL_NAME", args.registered_model_name)
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.pkl")
        joblib.dump(pipe, model_path)

        art = wandb.Artifact(
            name=model_name,
            type="model",
            metadata={
                "framework": "scikit-learn",
                "labels": LABELS,
                "max_features": args.max_features,
                "metrics": metrics,
            },
        )
        art.add_file(model_path, name="model.pkl")
        readme = os.path.join(tmp, "README.txt")
        with open(readme, "w") as f:
            f.write("TF-IDF + OneVsRest(LogReg) for Jigsaw toxic comments.")
        art.add_file(readme, name="README.txt")

        # Assign aliases here (no separate API call needed)
        run.log_artifact(art, aliases=["staging", "production"]).wait()

    print(json.dumps(metrics, indent=2))
    run.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/train.csv")
    p.add_argument("--registered_model_name", default="toxic-comment")
    p.add_argument("--max_features", type=int, default=200000)
    main(p.parse_args())