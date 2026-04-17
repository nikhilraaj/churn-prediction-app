from __future__ import annotations

import json
import joblib
import pandas as pd
import streamlit as st


# Page config
st.set_page_config(
    page_title="Smart Customer Churn Analyzer",
    page_icon="📊",
    layout="wide"
)


# 🎨 Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #22c55e 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background: #ecfdf5;
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
        border-left: 6px solid #22c55e;
    }
    .risk-high {
        color: #ef4444;
        font-weight: bold;
    }
    .risk-low {
        color: #22c55e;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 🎯 Header
st.markdown(
    """
    <div class="main-header">
        <h1>📊 Smart Churn Analyzer</h1>
        <p>AI-based system to predict customer retention risk</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# 📦 Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/model.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")

    with open("artifacts/feature_columns.json", "r") as f:
        feature_columns = json.load(f)

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# 📊 Input Section (CENTER UI)
st.subheader("📝 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.slider("💳 Credit Score", 300, 850, 600)
    age = st.slider("🎂 Age", 18, 100, 40)

with col2:
    gender = st.radio("👤 Gender", ["Female", "Male"])
    tenure = st.slider("📅 Tenure", 0, 10, 3)

with col3:
    balance = st.number_input("💰 Balance", 0.0, 250000.0, 60000.0)
    estimated_salary = st.number_input("💵 Salary", 0.0, 500000.0, 80000.0)


col4, col5, col6 = st.columns(3)

with col4:
    num_products = st.selectbox("📦 Products", [1, 2, 3, 4])

with col5:
    has_cr_card = st.checkbox("💳 Has Credit Card", value=True)

with col6:
    is_active_member = st.checkbox("⚡ Active Member", value=True)

geography = st.selectbox("🌍 Geography", ["France", "Germany", "Spain"])

analyze_button = st.button("🚀 Predict Churn")


# 🔮 Prediction
if analyze_button:
    raw_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card else 0,
        "IsActiveMember": 1 if is_active_member else 0,
        "EstimatedSalary": estimated_salary
    }])

    # Preprocessing
    processed = raw_data.copy()
    processed["Gender"] = processed["Gender"].map({"Female": 0, "Male": 1})
    processed = pd.get_dummies(processed, columns=["Geography"], drop_first=True)
    processed = processed.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(processed)

    # Prediction
    churn_probability = model.predict_proba(scaled)[0][1]
    stay_probability = 1 - churn_probability

    churn_label = "⚠️ HIGH RISK" if churn_probability >= 0.5 else "✅ LOW RISK"
    risk_class = "risk-high" if churn_probability >= 0.5 else "risk-low"

    st.markdown('<div class="prediction-result">', unsafe_allow_html=True)

    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("⚠️ Churn Probability", f"{churn_probability*100:.2f}%")

    with col2:
        st.metric("✅ Retention Probability", f"{stay_probability*100:.2f}%")

    with col3:
        st.markdown(f"<span class='{risk_class}'>{churn_label}</span>", unsafe_allow_html=True)

    st.progress(float(churn_probability))

    if churn_probability >= 0.5:
        st.warning("⚠️ Customer likely to leave. Take action!")
    else:
        st.success("🎉 Customer is likely to stay.")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👉 Enter details and click 'Predict Churn'")