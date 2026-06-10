"""Data preprocessing utilities for churn prediction."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load dataset from CSV file path."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean and encode churn dataframe into model-ready features and labels.

    Numeric missing values are filled with the median, and categorical missing values are
    filled with the column mode (or ``"Unknown"`` when no mode is available).
    """
    data = df.copy()

    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].fillna(data[col].median())
        else:
            mode = data[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            data[col] = data[col].fillna(fill_value).replace("", fill_value)

    if "Churn" not in data.columns:
        raise ValueError("Input dataset must contain a 'Churn' column.")

    y = data["Churn"].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("'Churn' column must contain only 'Yes' and 'No' values.")

    x = data.drop(columns=["Churn"])
    x_encoded = pd.get_dummies(x, drop_first=True)

    return x_encoded, y.astype(int)


def split_data(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split processed data into train and test sets."""
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def preprocess_and_split(
    csv_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, preprocess, and split Telco churn dataset."""
    df = load_dataset(csv_path)
    x, y = preprocess_dataframe(df)
    return split_data(x, y, test_size=test_size, random_state=random_state)
