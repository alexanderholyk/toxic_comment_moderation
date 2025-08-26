import os, json, argparse, time
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }
    # per-label AUC if possible
    per_auc = {}
    for i, lab in enumerate(LABELS):
        try:
            per_auc[f"auc_{lab}"] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            per_auc[f"auc_{lab}"] = np.nan
    metrics.update(per_auc)
    return metrics

def main(args):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(args.experiment)

    df = pd.read_csv(args.train_csv)
    X = df["comment_text"].astype(str).values
    y = df[LABELS].values.astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y.sum(axis=1) > 0)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=args.max_features, ngram_range=(1,2), min_df=2)),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced")))
    ])

    with mlflow.start_run() as run:
        mlflow.log_params({
            "max_features": args.max_features,
            "model_type": "tfidf_logreg_ovr",
            "code_version": os.popen("git rev-parse --short HEAD").read().strip() or "unknown",
        })
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        train_secs = time.time() - t0
        mlflow.log_metric("train_seconds", train_secs)

        y_prob = pipe.decision_function(X_te)
        # decision_function can be unbounded; map to pseudo-prob via logistic
        y_prob = 1 / (1 + np.exp(-y_prob))
        m = evaluate(y_te, y_prob, threshold=0.5)
        for k, v in m.items():
            if v == v:  # not NaN
                mlflow.log_metric(k, float(v))

        mlflow.sklearn.log_model(pipe, artifact_path="model", registered_model_name=args.registered_model_name)

        # optionally set description for the run
        mlflow.set_tag("data_version", os.popen(f"shasum {args.train_csv}").read().split()[0] if os.name != "nt" else "n/a")
        print(json.dumps(m, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/train.csv")
    p.add_argument("--experiment", default="toxicity-baselines")
    p.add_argument("--registered_model_name", default="toxic-comment")
    p.add_argument("--max_features", type=int, default=200000)
    main(p.parse_args())


# # example via mlflow CLI once run is logged
# mlflow models alias create --name "toxic-comment" --version 1 --alias Staging
# mlflow models alias create --name "toxic-comment" --version 1 --alias Production