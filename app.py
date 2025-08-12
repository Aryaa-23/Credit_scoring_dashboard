import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# Load model, scaler, and feature names
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Load training data for global SHAP
df = pd.read_csv("cs-training.csv")
df = df.rename(columns={"Unnamed: 0": "ID"})
X = df.drop(columns=["ID", "SeriousDlqin2yrs"])
X_scaled = scaler.transform(X)

# SHAP explainer (TreeExplainer for XGBoost)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

# Streamlit UI
st.title("💳 Credit Risk Prediction & Explainability Dashboard")
st.markdown("**Demo project for Explainable AI** — Predicts the probability of default and explains the decision using SHAP values.")

tab1, tab2 = st.tabs(["🔍 Single Prediction", "🌍 Global Insights"])

# =========================
# TAB 1: Single Prediction
# =========================
with tab1:
    st.sidebar.header("Applicant Information")
    user_data = {}
    for feature in feature_names:
        user_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

    input_df = pd.DataFrame([user_data])
    input_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict_proba(input_scaled)[0, 1]
    prediction = "High Risk" if prob > 0.5 else "Low Risk"

    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Probability of Default:** {prob*100:.2f}%")

    # Local SHAP explanation
    st.subheader("Feature Contribution (SHAP)")
    shap_single = explainer.shap_values(input_scaled)
    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_single[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_names
        )
    )
    st.pyplot(fig)

# =========================
# TAB 2: Global Insights
# =========================
with tab2:
    st.subheader("Global Feature Importance")
    fig_importance, ax = plt.subplots()
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    st.pyplot(fig_importance)

    st.subheader("Detailed Feature Impact")
    fig_summary, ax = plt.subplots()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    st.pyplot(fig_summary)

    st.markdown("""
    **Interpretation:**
    - The bar chart shows the average magnitude of impact each feature has on predictions (importance).
    - The dot plot shows both importance and whether high values push the prediction towards high or low risk.
    """)

