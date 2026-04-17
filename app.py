from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "churn_ann_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"


# Page config
st.set_page_config(
    page_title="Smart Customer Churn Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #22c55e 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(34, 197, 94, 0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .main-header p {
        margin-top: 0.5rem;
        opacity: 0.85;
        font-size: 1.1rem;
    }

    /* Prediction Box */
    .prediction-result {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
        border-left: 6px solid #22c55e;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }

    /* Risk Colors */
    .risk-high {
        color: #ef4444;
        font-weight: bold;
        font-size: 1.1rem;
    }

    .risk-low {
        color: #22c55e;
        font-weight: bold;
        font-size: 1.1rem;
    }

    /* Sidebar Inputs */
    .sidebar-input {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }

    /* Button Style */
    .stButton>button {
        background: linear-gradient(90deg, #22c55e, #15803d);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.4);
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border-top: 4px solid #22c55e;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# 🎯 Main Header
st.markdown(
    """
    <div class="main-header">
        <h1>📊 Smart Churn Analyzer</h1>
        <p>AI-based system to predict customer retention risk</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Load artifacts
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. Run model.ipynb first."
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Missing scaler file: {SCALER_PATH}. Run model.ipynb first."
        )

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature columns file: {FEATURE_COLUMNS_PATH}. Run model.ipynb first."
        )

    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
except Exception as error:
    st.error(str(error))
    st.stop()


# 📊 Sidebar for Inputs
st.sidebar.title("📝 Customer Profile")

with st.sidebar:
    st.markdown('<div class="sidebar-input">', unsafe_allow_html=True)
    credit_score = st.slider("💳 Credit Score", 300, 850, 600, help="Customer's credit score")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-input">', unsafe_allow_html=True)
    gender = st.radio("👤 Gender", ["Female", "Male"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-input">', unsafe_allow_html=True)
    age = st.slider("🎂 Age", 18, 100, 40, help="Customer's age in years")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-input">', unsafe_allow_html=True)
    tenure = st.slider("📅 Tenure", 0, 10, 3, help="Years with the bank")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-input">', unsafe_allow_html=True)
    balance = st.number_input("💰 Balance ($)", 0.0, 250000.0, 60000.0, step=1000.0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-input">', unsafe_allow_html=True)
    num_products = st.selectbox("📦 Number of Products", [1, 2, 3, 4], help="Products held by customer")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-input">', unsafe_allow_html=True)
    has_cr_card = st.checkbox("💳 Has Credit Card", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-input">', unsafe_allow_html=True)
    is_active_member = st.checkbox("⚡ Active Member", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-input">', unsafe_allow_html=True)
    estimated_salary = st.number_input("💵 Estimated Salary ($)", 0.0, 500000.0, 80000.0, step=1000.0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-input">', unsafe_allow_html=True)
    geography = st.selectbox("🌍 Geography", ["France", "Germany", "Spain"])
    st.markdown('</div>', unsafe_allow_html=True)

    analyze_button = st.button("🔍 Analyze Customer Risk", use_container_width=True)


# 🔮 Prediction Section
if analyze_button:
    raw_row = pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": 1 if has_cr_card else 0,
                "IsActiveMember": 1 if is_active_member else 0,
                "EstimatedSalary": estimated_salary,
            }
        ]
    )

    processed_row = raw_row.copy()
    processed_row["Gender"] = processed_row["Gender"].map({"Female": 0, "Male": 1})
    processed_row = pd.get_dummies(processed_row, columns=["Geography"], drop_first=True)
    processed_row = processed_row.reindex(columns=feature_columns, fill_value=0)

    scaled_row = scaler.transform(processed_row)
    churn_probability = float(model.predict(scaled_row, verbose=0)[0][0])
    stay_probability = 1.0 - churn_probability
    churn_label = "⚠️ HIGH RISK" if churn_probability >= 0.5 else "✅ LOW RISK"
    risk_class = "risk-high" if churn_probability >= 0.5 else "risk-low"

    st.markdown('<div class="prediction-result">', unsafe_allow_html=True)

    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚠️ Churn Probability", f"{churn_probability * 100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("✅ Retention Probability", f"{stay_probability * 100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"**Risk Level:** <span class='{risk_class}'>{churn_label}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Progress bar
    st.progress(churn_probability)
    st.caption("Churn Risk Indicator")

    # Additional insights
    if churn_probability >= 0.5:
        st.warning("⚠️ Customer likely to leave. Immediate action recommended!")
    else:
        st.success("🎉 This customer shows low churn risk. Great job maintaining customer satisfaction!")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👈 Enter customer details in the sidebar and click 'Analyze Customer Risk' to get predictions.")