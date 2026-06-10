"""Evaluation helpers for churn prediction models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_preprocessing import preprocess_and_split


def evaluate(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Compute standard classification metrics."""
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate churn model")
    parser.add_argument("--data", default="data/Telco-Customer-Churn.csv", help="Dataset path")
    parser.add_argument("--model", default="models/best_model.joblib", help="Model bundle path")
    parser.add_argument("--output", default="reports/metrics.json", help="Metrics JSON output path")
    return parser.parse_args()


def main() -> None:
    """Load model bundle and evaluate against test split."""
    args = parse_args()
    x_train, x_test, y_train, y_test = preprocess_and_split(args.data)

    bundle = joblib.load(args.model)
    feature_columns = bundle["feature_columns"]
    model = bundle["model"]

    x_test = x_test.reindex(columns=feature_columns, fill_value=0)
    metrics = evaluate(model, x_test, y_test)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
