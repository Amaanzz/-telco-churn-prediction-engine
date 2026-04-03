import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from src.preprocess import preprocess_input, MODELS_DIR
from src.predict import predict_churn
from src.strategy import generate_retention_strategy

# 1. Page Configuration
st.set_page_config(
    page_title="Churn Prediction Engine",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔥 INSANE BOOST #1: The Tagline
st.title("Telco Customer Churn Engine")
st.markdown("### 🚀 AI-Powered Customer Retention Intelligence System")
st.markdown("Enter customer details below to generate real-time churn probability and actionable retention strategies.")
st.divider()

# 2. Main Dashboard Layout (2 Columns)
col1, col2 = st.columns([1.2, 1])

# --- COLUMN 1: User Inputs ---
with col1:
    st.subheader("Customer Profile")

    with st.form("customer_input_form"):
        # Grouped inputs for better UI
        st.markdown("**Demographics & Account History**")
        c1, c2, c3 = st.columns(3)
        tenure = c1.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        monthly_charges = c2.number_input("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=50.0, step=1.0)
        total_charges = c3.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=600.0, step=10.0)

        st.markdown("**Service & Contract Details**")
        c4, c5 = st.columns(2)
        contract = c4.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment = c5.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        internet = c4.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        paperless = c5.radio("Paperless Billing", ["Yes", "No"], horizontal=True)

        st.markdown("**Household Status**")
        c6, c7 = st.columns(2)
        partner = c6.radio("Has Partner", ["Yes", "No"], horizontal=True)
        dependents = c7.radio("Has Dependents", ["Yes", "No"], horizontal=True)

        submit_button = st.form_submit_button("Analyze Churn Risk", use_container_width=True)

# --- COLUMN 2: Output & Strategy ---
with col2:
    st.subheader("Predictive Analytics")

    if submit_button:
        # Construct the raw data dictionary
        raw_data = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Contract': contract,
            'PaymentMethod': payment,
            'InternetService': internet,
            'PaperlessBilling': paperless,
            'Partner': partner,
            'Dependents': dependents
        }

        with st.spinner("Running AI Engine..."):
            try:
                # Execute Pipeline
                processed_array = preprocess_input(raw_data)
                probability = predict_churn(processed_array)
                strategy = generate_retention_strategy(probability, monthly_charges, tenure)

                # 1. Display Probability
                st.markdown("### Churn Probability")
                st.progress(float(probability))
                st.markdown(f"<h2 style='text-align: center;'>{probability * 100:.1f}%</h2>", unsafe_allow_html=True)

                st.divider()

                # 2. Risk Level & Customer Value
                st.markdown("### Customer Assessment")
                m1, m2 = st.columns(2)

                with m1:
                    if probability > 0.70:
                        st.error("🔥 Critical Risk")
                    elif probability > 0.50:
                        st.warning("⚠️ High Risk")
                    elif probability > 0.30:
                        st.info("📊 Medium Risk")
                    else:
                        st.success("✅ Low Risk")

                with m2:
                    if monthly_charges > 70.0:
                        st.metric("Customer Value", "High 💎")
                    else:
                        st.metric("Customer Value", "Standard")

                st.divider()

                # 3. Display Strategy
                st.markdown("### Recommended Action")
                st.write(strategy['action'])

                st.divider()

                # 🔥 INSANE BOOST #2: Explainability Engine (Native SHAP Alternative)
                with st.expander("🔍 Why this prediction? (Explainability)"):
                    st.markdown(
                        "This chart shows which features are driving this specific customer toward or away from churning.")

                    # Load model and columns
                    with open(MODELS_DIR / 'model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    with open(MODELS_DIR / 'columns.pkl', 'rb') as f:
                        cols = pickle.load(f)

                    # Calculate local feature importance (Value * Coefficient)
                    coefficients = model.coef_[0]
                    feature_impact = processed_array[0] * coefficients

                    # Create a clean dataframe for plotting
                    impact_df = pd.DataFrame({
                        'Feature': cols,
                        'Impact': feature_impact
                    })

                    # Filter out zero-impact features (dummy columns not triggered)
                    impact_df = impact_df[impact_df['Impact'] != 0]

                    # Sort by absolute impact
                    impact_df['Absolute Impact'] = impact_df['Impact'].abs()
                    impact_df = impact_df.sort_values(by='Absolute Impact', ascending=False).head(8)  # Top 8 drivers

                    # Map to colors (Red = drives churn up, Green = drives churn down)
                    impact_df.set_index('Feature', inplace=True)

                    # Render Native Streamlit Bar Chart
                    st.bar_chart(impact_df['Impact'], color="#ff4b4b", height=300)
                    st.caption("Positive bars (Up) increase churn risk. Negative bars (Down) increase retention.")

            except Exception as e:
                st.error(f"Pipeline Error: {str(e)}")
    else:
        st.info("Awaiting customer data. Fill out the profile on the left and click 'Analyze Churn Risk'.")