import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔁", layout="wide")

st.title("🔁 Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict churn probability.")
st.markdown("---")

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('Model/churn_model.pkl')

model = load_model()

# ─────────────────────────────────────────────
# Input Fields
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Info")
    gender         = st.selectbox("Gender",           [0, 1],        format_func=lambda x: ["Female", "Male"][x])
    senior         = st.selectbox("Senior Citizen",   [0, 1],        format_func=lambda x: ["No", "Yes"][x])
    partner        = st.selectbox("Partner",          [0, 1],        format_func=lambda x: ["No", "Yes"][x])
    dependents     = st.selectbox("Dependents",       [0, 1],        format_func=lambda x: ["No", "Yes"][x])
    tenure         = st.slider("Tenure (months)",     0, 72, 12)

    st.subheader("📞 Phone Services")
    phone_service  = st.selectbox("Phone Service",    [0, 1],        format_func=lambda x: ["No", "Yes"][x])
    multiple_lines = st.selectbox("Multiple Lines",   [0, 1, 2],     format_func=lambda x: ["No", "Yes", "No Phone Service"][x])

    st.subheader("🌐 Internet Services")
    internet       = st.selectbox("Internet Service", [0, 1, 2],     format_func=lambda x: ["DSL", "Fiber Optic", "No"][x])
    online_sec     = st.selectbox("Online Security",  [0, 1, 2],     format_func=lambda x: ["No", "Yes", "No Internet"][x])
    online_backup  = st.selectbox("Online Backup",    [0, 1, 2],     format_func=lambda x: ["No", "Yes", "No Internet"][x])

with col2:
    st.subheader("🛡️ Add-on Services")
    device_prot    = st.selectbox("Device Protection",[0, 1, 2],     format_func=lambda x: ["No", "Yes", "No Internet"][x])
    tech_support   = st.selectbox("Tech Support",     [0, 1, 2],     format_func=lambda x: ["No", "Yes", "No Internet"][x])
    streaming_tv   = st.selectbox("Streaming TV",     [0, 1, 2],     format_func=lambda x: ["No", "Yes", "No Internet"][x])
    streaming_mov  = st.selectbox("Streaming Movies", [0, 1, 2],     format_func=lambda x: ["No", "Yes", "No Internet"][x])

    st.subheader("💳 Billing & Contract")
    contract       = st.selectbox("Contract Type",    [0, 1, 2],     format_func=lambda x: ["Month-to-Month", "One Year", "Two Year"][x])
    paperless      = st.selectbox("Paperless Billing",[0, 1],        format_func=lambda x: ["No", "Yes"][x])
    payment        = st.selectbox("Payment Method",   [0, 1, 2, 3],  format_func=lambda x: ["Bank Transfer", "Credit Card", "Electronic Check", "Mailed Check"][x])
    monthly        = st.number_input("Monthly Charges ($)",  min_value=20.0,  max_value=120.0, value=65.0,   step=0.5)
    total          = st.number_input("Total Charges ($)",    min_value=0.0,   max_value=9000.0, value=1000.0, step=10.0)

# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
st.markdown("---")
predict_clicked = st.button("🔍 Predict Churn", use_container_width=True)

if predict_clicked:
    input_data = pd.DataFrame([[
        gender, senior, partner, dependents, tenure,
        phone_service, multiple_lines, internet,
        online_sec, online_backup, device_prot, tech_support,
        streaming_tv, streaming_mov, contract, paperless,
        payment, monthly, total
    ]], columns=[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ])

    prob = model.predict_proba(input_data)[:, 1][0]

    # ── Result ──
    res_col, gauge_col = st.columns([1, 1])

    with res_col:
        st.subheader("📊 Prediction Result")
        if prob >= 0.4:
            st.error("⚠️ Customer is likely to **CHURN**")
        else:
            st.success("✅ Customer is likely to **STAY**")

        st.metric(label="Churn Probability", value=f"{prob * 100:.1f}%")
        st.progress(float(prob))
        st.caption("Threshold: 0.40 — above this the customer is predicted to churn")

        # Risk label
        if prob < 0.4:
            risk, color = "🟢 Low Risk", "green"
        elif prob < 0.7:
            risk, color = "🟠 Medium Risk", "orange"
        else:
            risk, color = "🔴 High Risk", "red"
        st.markdown(f"**Risk Level:** :{color}[{risk}]")

    # ── Gauge Chart ──
    with gauge_col:
        st.subheader("🎯 Churn Risk Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={'suffix': "%"},
            title={'text': "Churn Risk"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "red" if prob >= 0.4 else "green"},
                'steps': [
                    {'range': [0,  40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "lightyellow"},
                    {'range': [70, 100],'color': "lightsalmon"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 40
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)