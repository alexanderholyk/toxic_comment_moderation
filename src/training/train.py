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

import os, json, argparse, time, tempfile, joblib
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
load_dotenv()  # will read .env from project root

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

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
    # --- W&B init ---
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "toxic-moderation"),
        entity=os.environ.get("WANDB_ENTITY", None),
        job_type="train",
        config={
            "max_features": args.max_features,
            "model_type": "tfidf_logreg_ovr",
            "labels": LABELS,
        },
    )

    df = pd.read_csv(args.train_csv)
    X = df["comment_text"].astype(str).values
    y = df[LABELS].values.astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=(y.sum(axis=1) > 0)
    )

    pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=3,          # was 2
        max_df=0.9,        # drop ultra-common tokens
        sublinear_tf=True, # log-scale term freq
        dtype=np.float32   # smaller, more stable mats
    )),
    ("clf", OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear",     # robust for OvR
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
    train_secs = time.time() - t0
    wandb.log({"train_seconds": train_secs})

    # decision_function -> convert to [0,1]
    y_raw = getattr(pipe, "decision_function", None)
    if callable(y_raw):
        y_prob = pipe.decision_function(X_te)
        y_prob = 1 / (1 + np.exp(-y_prob))
    else:
        # some sklearn models expose predict_proba
        y_prob = pipe.predict_proba(X_te)

    metrics = evaluate(y_te, y_prob, threshold=0.5)
    wandb.log(metrics)

    # --- Save model + log artifact ---
    model_name = os.environ.get("WANDB_MODEL_NAME", args.registered_model_name)
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.pkl")
        joblib.dump(pipe, model_path)
        art = wandb.Artifact(
            name=f"{model_name}",
            type="model",
            metadata={
                "framework": "scikit-learn",
                "labels": LABELS,
                "max_features": args.max_features,
                "metrics": metrics,
            },
        )
        art.add_file(model_path, name="model.pkl")
        # also include a small README for clarity
        readme = os.path.join(tmp, "README.txt")
        with open(readme, "w") as f:
            f.write("TF-IDF + OneVsRest(LogReg) for Jigsaw toxic comments.")
        art.add_file(readme, name="README.txt")

        # log the artifact
        res = run.log_artifact(art)
        res.wait()  # ensure upload completes

        # optionally add aliases (e.g., move best to 'production')
        # You can make this conditional on a metric threshold
        api = wandb.Api()
        # Fetch the just-logged version
        art_full = api.artifact(f"{run.entity}/{run.project}/{model_name}:{res.aliases[0] if res.aliases else 'latest'}")
        # set/update aliases
        # For first run, you may want to point 'staging' and 'production' here:
        art_full.aliases = list(set(art_full.aliases + ["staging", "production"]))
        art_full.save()

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/train.csv")
    p.add_argument("--experiment", default="toxicity-baselines")  # kept for CLI compatibility
    p.add_argument("--registered_model_name", default="toxic-comment")
    p.add_argument("--max_features", type=int, default=200000)
    main(p.parse_args())