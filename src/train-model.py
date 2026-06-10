import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────
# STEP 1: Load Data
# ─────────────────────────────────────────────
data = pd.read_csv('../data/Telco_customer_churn.csv')
 
print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
 
print("\nMissing Values:")
print(data.isnull().sum())
 
print("\nChurn Distribution:")
print(data['Churn'].value_counts())
print(data['Churn'].value_counts(normalize=True) * 100)
 
# ─────────────────────────────────────────────
# STEP 2: EDA Plots
# ─────────────────────────────────────────────
 
# Plot 1: Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data, palette='Set2')
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.savefig('churn_distribution.png')
plt.show()
 
# Plot 2: Churn by Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=data, palette='Set2')
plt.title('Churn by Contract Type')
plt.tight_layout()
plt.savefig('churn_by_contract.png')
plt.show()
 
# Plot 3: Churn by Tenure
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x='tenure', hue='Churn', bins=30, palette='Set2')
plt.title('Churn by Tenure')
plt.tight_layout()
plt.savefig('churn_by_tenure.png')
plt.show()
 
# ─────────────────────────────────────────────
# STEP 3: Data Preprocessing
# ─────────────────────────────────────────────
 
# Convert TotalCharges to numeric (some values are blank spaces)
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())
 
# Drop customerID if it exists (not useful for prediction)
if 'customerID' in data.columns:
    data.drop('customerID', axis=1, inplace=True)
 
# Encode all categorical columns
le = LabelEncoder()
for col in data.select_dtypes(include="object").columns:
    data[col] = le.fit_transform(data[col].astype(str))
 
print("\nData after encoding:")
print(data.head())
 
# ─────────────────────────────────────────────
# STEP 4: Features & Target Split
# ─────────────────────────────────────────────
X = data.drop("Churn", axis=1)
Y = data["Churn"]
 
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y  # stratify keeps churn ratio balanced
)
 
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")
 
# ─────────────────────────────────────────────
# STEP 5: Apply SMOTE (Fix Class Imbalance)
# ─────────────────────────────────────────────
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, Y_train)
 
print("\nBefore SMOTE:", Y_train.value_counts().to_dict())
print("After SMOTE :", pd.Series(y_train_res).value_counts().to_dict())
 
# ─────────────────────────────────────────────
# STEP 6: Train Model 1 — Random Forest
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL 1: Random Forest")
print("="*50)
 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_res, y_train_res)
 
Y_pred_rf = rf_model.predict(X_test)
Y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
 
print(classification_report(Y_test, Y_pred_rf))
print("AUC-ROC Score:", roc_auc_score(Y_test, Y_prob_rf))
 
# ─────────────────────────────────────────────
# STEP 7: Train Model 2 — XGBoost (Better)
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL 2: XGBoost")
print("="*50)
 
xgb_model = XGBClassifier(
    scale_pos_weight=3,       # handles imbalance
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train_res, y_train_res)
 
Y_pred_xgb = xgb_model.predict(X_test)
Y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
 
print(classification_report(Y_test, Y_pred_xgb))
print("AUC-ROC Score:", roc_auc_score(Y_test, Y_prob_xgb))
 
# ─────────────────────────────────────────────
# STEP 8: Tune Threshold (Quick Win)
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("XGBoost with Threshold = 0.4")
print("="*50)
 
Y_pred_tuned = (Y_prob_xgb >= 0.4).astype(int)
print(classification_report(Y_test, Y_pred_tuned))
print("AUC-ROC Score:", roc_auc_score(Y_test, Y_prob_xgb))
 
# ─────────────────────────────────────────────
# STEP 9: Confusion Matrix (Best Model)
# ─────────────────────────────────────────────
cm = confusion_matrix(Y_test, Y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - XGBoost (Threshold 0.4)")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
 
# ─────────────────────────────────────────────
# STEP 10: Feature Importance
# ─────────────────────────────────────────────
feat_imp = pd.Series(xgb_model.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False).head(10)
 
plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title('Top 10 Important Features')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
 
print("\nTop 10 Features:")
print(feat_imp)
 
# ─────────────────────────────────────────────
# STEP 11: Save Best Model
# ─────────────────────────────────────────────

import os
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, '..', 'Model')

os.makedirs(model_dir, exist_ok=True)

joblib.dump(xgb_model, os.path.join(model_dir, 'churn_model.pkl'))
print("Model saved as 'Model/churn_model.pkl'")
# ─────────────────────────────────────────────
# STEP 12: Load & Predict on New Customer (Demo)
# ─────────────────────────────────────────────
loaded_model = joblib.load('churn_model.pkl')
 
# Example: predict on first row of test set
sample = X_test.iloc[[0]]
prediction = loaded_model.predict(sample)
probability = loaded_model.predict_proba(sample)[:, 1]
 
print("\n--- Sample Prediction ---")
print("Churn Prediction:", "YES - Will Churn" if prediction[0] == 1 else "NO - Will Stay")
print(f"Churn Probability: {probability[0]*100:.2f}%")
 
