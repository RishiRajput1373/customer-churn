"""Streamlit app for customer churn prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.predict import prepare_features

MODEL_PATH = Path("models/best_model.joblib")
DATA_PATH = Path("data/Telco-Customer-Churn.csv")


def main() -> None:
    """Render streamlit churn prediction app."""
    st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")
    st.title("Customer Churn Prediction")

    if not MODEL_PATH.exists():
        st.error("Model not found. Train a model first: python -m src.train_model")
        return

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    if DATA_PATH.exists():
        raw_df = pd.read_csv(DATA_PATH)
        template_cols = [c for c in raw_df.columns if c not in {"Churn", "customerID"}]
    else:
        template_cols = []

    st.subheader("Enter customer information")
    record: dict[str, str | float | int] = {}

    if template_cols:
        for col in template_cols:
            if col in {"tenure", "SeniorCitizen"}:
                record[col] = st.number_input(col, min_value=0, value=0)
            elif col in {"MonthlyCharges", "TotalCharges"}:
                record[col] = st.number_input(col, min_value=0.0, value=0.0)
            else:
                options = sorted(raw_df[col].dropna().astype(str).unique().tolist())
                record[col] = st.selectbox(col, options=options)
    else:
        st.info("Dataset not found. Provide basic fields manually.")
        record["tenure"] = st.number_input("tenure", min_value=0, value=0)
        record["MonthlyCharges"] = st.number_input("MonthlyCharges", min_value=0.0, value=0.0)
        record["TotalCharges"] = st.number_input("TotalCharges", min_value=0.0, value=0.0)

    if st.button("Predict Churn"):
        features = prepare_features(record, feature_columns)
        churn_prob = float(model.predict_proba(features)[:, 1][0])
        churn_label = "Yes" if int(model.predict(features)[0]) == 1 else "No"

        st.success(f"Prediction: **{churn_label}**")
        st.metric("Churn Probability", f"{churn_prob:.2%}")


if __name__ == "__main__":
    main()
