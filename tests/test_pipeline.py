"""Focused tests for churn pipeline modules."""

from __future__ import annotations

import pandas as pd

from src.data_preprocessing import preprocess_dataframe, split_data
from src.train_model import train_models


def sample_df() -> pd.DataFrame:
    """Create a small synthetic churn dataset for tests."""
    return pd.DataFrame(
        {
            "customerID": ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008"],
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 1, 0, 1, 0, 1],
            "tenure": [1, 2, 3, 4, 5, 6, 7, 8],
            "MonthlyCharges": [30.0, 90.0, 35.0, 95.0, 40.0, 100.0, 45.0, 110.0],
            "TotalCharges": ["30", "180", "", "380", "200", "600", "315", "880"],
            "Contract": ["Month-to-month", "Two year", "Month-to-month", "Two year", "Month-to-month", "Two year", "Month-to-month", "Two year"],
            "Churn": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        }
    )


def test_preprocess_dataframe_handles_total_charges_and_encoding() -> None:
    """Ensure preprocessing removes id, encodes features and maps target."""
    x, y = preprocess_dataframe(sample_df())

    assert "customerID" not in x.columns
    assert "TotalCharges" in x.columns
    assert y.tolist() == [1, 0, 1, 0, 1, 0, 1, 0]


def test_train_models_returns_best_model_bundle() -> None:
    """Ensure model training returns selected best model information."""
    x, y = preprocess_dataframe(sample_df())
    x_train, x_test, y_train, y_test = split_data(x, y, test_size=0.25, random_state=42)

    result = train_models(x_train, y_train, x_test, y_test)

    assert result["name"] in {"logistic_regression", "random_forest"}
    assert hasattr(result["model"], "predict")
    assert 0.0 <= result["roc_auc"] <= 1.0
