import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("cs-training.csv")
df = df.rename(columns={"Unnamed: 0": "ID"})  # Rename ID column

# Features & target
X = df.drop(columns=["ID", "SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="auc",
    use_label_encoder=False
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test ROC-AUC: {auc:.4f}")

# Save model & scaler
joblib.dump(model, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
