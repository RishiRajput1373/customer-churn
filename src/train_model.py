"""Model training script for churn prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data_preprocessing import preprocess_and_split

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    include_xgboost: bool = False,
) -> dict[str, Any]:
    """Train multiple models and select best one by ROC-AUC on test data."""
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    if include_xgboost and XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

    best_name = ""
    best_model: Any = None
    best_score = -1.0

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_test)[:, 1]
        score = roc_auc_score(y_test, y_prob)
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("Failed to train any model.")

    return {
        "name": best_name,
        "model": best_model,
        "roc_auc": best_score,
    }


def save_model_bundle(model_bundle: dict[str, Any], feature_columns: list[str], output_path: str | Path) -> None:
    """Save model and metadata bundle with joblib."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_bundle["name"],
        "model": model_bundle["model"],
        "roc_auc": model_bundle["roc_auc"],
        "feature_columns": feature_columns,
    }
    joblib.dump(payload, output)


def parse_args() -> argparse.Namespace:
    """Parse script arguments."""
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument(
        "--data",
        default="data/Telco-Customer-Churn.csv",
        help="Path to Telco churn CSV dataset",
    )
    parser.add_argument(
        "--output",
        default="models/best_model.joblib",
        help="Output path for serialized model",
    )
    parser.add_argument(
        "--xgboost",
        action="store_true",
        help="Include XGBoost model training if package is installed",
    )
    return parser.parse_args()


def main() -> None:
    """Run training pipeline and persist best model bundle."""
    args = parse_args()
    x_train, x_test, y_train, y_test = preprocess_and_split(args.data)
    result = train_models(x_train, y_train, x_test, y_test, include_xgboost=args.xgboost)
    save_model_bundle(result, list(x_train.columns), args.output)
    print(f"Saved best model '{result['name']}' with ROC-AUC={result['roc_auc']:.4f} to {args.output}")


if __name__ == "__main__":
    main()
