"""Prediction script for single customer churn inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def prepare_features(input_record: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """Prepare a single record for model inference."""
    df = pd.DataFrame([input_record])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    encoded = pd.get_dummies(df, drop_first=True)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded


def predict_churn(model_bundle_path: str | Path, input_json_path: str | Path) -> dict[str, Any]:
    """Run churn prediction for an input JSON file."""
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    payload = json.loads(Path(input_json_path).read_text(encoding="utf-8"))
    features = prepare_features(payload, feature_columns)

    probability = float(model.predict_proba(features)[:, 1][0])
    prediction = int(model.predict(features)[0])

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": probability,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Predict customer churn")
    parser.add_argument("--model", default="models/best_model.joblib", help="Model bundle path")
    parser.add_argument("--input", required=True, help="Input JSON path with customer fields")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result = predict_churn(args.model, args.input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
