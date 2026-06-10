# Customer Churn Prediction (End-to-End)

Production-ready Customer Churn Prediction project using Python and the Telco Customer Churn dataset.

## Project Structure

```text
customer-churn/
├── app.py
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── eda.py
├── tests/
├── requirements.txt
└── README.md
```

## Overview

This project includes:
- Data preprocessing (cleaning, encoding, train/test split)
- Exploratory data analysis (EDA visualizations)
- Model training (Logistic Regression, Random Forest, optional XGBoost)
- Metrics-based evaluation
- Best model serialization with `joblib`
- Streamlit app for interactive churn prediction

## Dataset

Use the Telco Customer Churn dataset and place it at:

`data/Telco-Customer-Churn.csv`

Expected target column: `Churn` with values `Yes`/`No`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Usage

### 1) Train model

```bash
python -m src.train_model --data data/Telco-Customer-Churn.csv --output models/best_model.joblib
```

Optional XGBoost:

```bash
python -m src.train_model --xgboost
```

### 2) Evaluate model

```bash
python -m src.evaluate_model --data data/Telco-Customer-Churn.csv --model models/best_model.joblib --output reports/metrics.json
```

### 3) Predict with JSON input

```bash
python -m src.predict --model models/best_model.joblib --input sample_customer.json
```

Example `sample_customer.json`:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "tenure": 12,
  "MonthlyCharges": 70.35,
  "TotalCharges": 844.2,
  "Contract": "Month-to-month"
}
```

### 4) Run EDA report generation

```python
from src.eda import run_eda
run_eda("data/Telco-Customer-Churn.csv", output_dir="reports")
```

Generated charts include:
- Churn distribution
- Correlation heatmap
- Customer demographics analysis
- Service usage analysis
- Feature-importance proxy visualization

### 5) Launch Streamlit app

```bash
streamlit run app.py
```

## Results

- Metrics are saved to `reports/metrics.json`
- Best model is saved to `models/best_model.joblib`
- EDA figures are saved in `reports/`

Add screenshots of Streamlit UI and EDA plots in this section after running the project locally.

## Testing

```bash
python -m pytest tests/test_pipeline.py -q
```

## Error Handling & Best Practices

- Type hints in all major functions
- Docstrings for maintainability
- Input validation and clear exceptions for missing files/columns
- Modular scripts for preprocessing, training, evaluation, and prediction
