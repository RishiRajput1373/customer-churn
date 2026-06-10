"""Exploratory data analysis routines for churn project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda(csv_path: str | Path, output_dir: str | Path = "reports") -> None:
    """Generate required EDA charts and save them to reports directory."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False)
    plt.title("Churn Distribution")
    plt.tight_layout()
    plt.savefig(output / "churn_distribution.png")
    plt.close()

    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output / "correlation_heatmap.png")
        plt.close()

    if "gender" in df.columns and "Churn" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="gender", hue="Churn")
        plt.title("Customer Demographics: Gender vs Churn")
        plt.tight_layout()
        plt.savefig(output / "demographics_gender_churn.png")
        plt.close()

    if "InternetService" in df.columns and "Churn" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.countplot(data=df, x="InternetService", hue="Churn")
        plt.title("Service Usage: Internet Service vs Churn")
        plt.tight_layout()
        plt.savefig(output / "service_usage_internet_churn.png")
        plt.close()

    if "MonthlyCharges" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
        plt.title("Feature Importance Proxy: Monthly Charges by Churn")
        plt.tight_layout()
        plt.savefig(output / "feature_importance_monthlycharges.png")
        plt.close()
