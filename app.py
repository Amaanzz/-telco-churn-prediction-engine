import streamlit as st
import pandas as pd
import requests

from src.strategy import generate_retention_strategy_v2

# -----------------------------
# Configuration
# -----------------------------
import os

API_URL = os.getenv(
    "API_URL",
    "https://telco-churn-api-urtt.onrender.com/predict"
)
# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Churn Prediction Engine",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Telco Customer Churn Engine")
st.markdown("### 🚀 AI-Powered Customer Retention Intelligence System")
st.markdown(
    "Enter customer details below to generate real-time churn probability "
    "and actionable retention strategies."
)
st.divider()

# -----------------------------
# 2. Main Dashboard Layout
# -----------------------------
col1, col2 = st.columns([1.2, 1])

# -----------------------------
# COLUMN 1: User Inputs
# -----------------------------
with col1:
    st.subheader("Customer Profile")

    with st.form("customer_input_form"):

        st.markdown("**Demographics & Account History**")
        c1, c2, c3 = st.columns(3)
        tenure = c1.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = c2.number_input(
            "Monthly Charges ($)", 15.0, 120.0, 50.0, step=1.0
        )
        total_charges = c3.number_input(
            "Total Charges ($)", 0.0, 9000.0, 600.0, step=10.0
        )

        st.markdown("**Personal Details**")
        c1b, c2b, c3b = st.columns(3)
        gender = c1b.selectbox("Gender", ["Male", "Female"])
        senior_citizen = c2b.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
        phone_service = c3b.radio("Phone Service", ["Yes", "No"], horizontal=True)

        st.markdown("**Service & Contract Details**")
        c4, c5 = st.columns(2)
        contract = c4.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        payment = c5.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        internet = c4.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        paperless = c5.radio("Paperless Billing", ["Yes", "No"], horizontal=True)
        multiple_lines = c4.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        st.markdown(
            "**Add-on Services** *(select 'No internet service' if no internet plan)*"
        )
        c8, c9, c10 = st.columns(3)
        online_security = c8.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = c9.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        device_protection = c10.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        tech_support = c8.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        streaming_tv = c9.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = c10.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        st.markdown("**Household Status**")
        c6, c7 = st.columns(2)
        partner = c6.radio("Has Partner", ["Yes", "No"], horizontal=True)
        dependents = c7.radio("Has Dependents", ["Yes", "No"], horizontal=True)

        submit_button = st.form_submit_button(
            "Analyze Churn Risk",
            use_container_width=True
        )

# -----------------------------
# COLUMN 2: Output
# -----------------------------
with col2:
    st.subheader("Predictive Analytics")

    if submit_button:

        raw_data = {
            "gender": gender,
            "senior_citizen": 1 if senior_citizen == "Yes" else 0,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless,
            "payment_method": payment,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
        }

        with st.spinner("Running prediction and business intelligence engine..."):

            try:
                # -----------------------------
                # Call deployed FastAPI backend
                # -----------------------------
                response = requests.post(API_URL, json=raw_data, timeout=30)
                response.raise_for_status()

                result = response.json()

                # Adjust this key if your API returns a different field name
                probability = result["churn_probability"]

                # -----------------------------
                # Generate EV-based strategy
                # -----------------------------
                strategy = {
                    "tier": result["ev_tier"],
                    "expected_value": result["expected_value"],
                    "action": result["retention_action"],
                    "senior_warning": result["senior_citizen_alert"],
                }

                # -----------------------------
                # Display Probability
                # -----------------------------
                st.markdown("### Predicted Churn Probability")
                st.progress(float(probability))
                st.markdown(
                    f"<h2 style='text-align: center;'>{probability * 100:.1f}%</h2>",
                    unsafe_allow_html=True
                )

                st.divider()

                # -----------------------------
                # Expected Value Assessment
                # -----------------------------
                st.markdown("### Expected Value Assessment")
                m1, m2, m3 = st.columns(3)

                with m1:
                    tier = strategy["tier"]
                    ev = strategy["expected_value"]

                    if tier == "high_value":
                        st.success(
                            f"💎 High-Value Customer\n\nExpected Value\n\n${ev:.2f}"
                        )
                    elif tier == "standard":
                        st.info(
                            f"📊 Standard Intervention\n\nExpected Value\n\n${ev:.2f}"
                        )
                    else:
                        st.warning(
                            f"⚪ Monitor Only\n\nExpected Value\n\n${ev:.2f}"
                        )

                with m2:
                    st.metric(
                        "Churn Probability",
                        f"{probability * 100:.1f}%"
                    )

                with m3:
                    st.metric(
                        "Expected Value",
                        f"${strategy['expected_value']:.2f}"
                    )

                st.caption(
                    "Expected Value (EV) estimates the financial benefit of a "
                    "retention intervention using predicted churn probability, "
                    "estimated customer lifetime value, offer success rate, "
                    "and intervention cost."
                )

                if strategy["senior_warning"]:
                    st.warning(
                        "⚠️ **Senior Citizen Alert**: This customer is in a "
                        "high-churn-risk demographic (41.7% vs. 23.7%). "
                        "Prioritize empathetic communication and simplified "
                        "service packages."
                    )

                st.divider()

                # -----------------------------
                # Recommended Action
                # -----------------------------
                st.markdown("### Recommended Action")
                st.write(strategy["action"])

                st.divider()

                # -----------------------------
                # Explainability
                # -----------------------------
                if "explainability" in result and result["explainability"] is not None:
                    exp = result["explainability"]

                    impact_df = pd.DataFrame({
                        "Feature": exp["feature_names"],
                        "Impact": exp["shap_values"]
                    })

                    impact_df = impact_df[impact_df["Impact"] != 0]
                    impact_df["Absolute Impact"] = impact_df["Impact"].abs()
                    impact_df = impact_df.sort_values(
                        by="Absolute Impact",
                        ascending=False
                    ).head(8)
                    impact_df.set_index("Feature", inplace=True)

                    with st.expander("🔍 Why this prediction? (Explainability)"):
                        st.bar_chart(impact_df["Impact"], color="#ff4b4b", height=300)
                        st.caption(
                            "Feature contributions computed using SHAP. "
                            "Positive values increase churn risk, negative values reduce it."
                        )
                else:
                    st.info(
                        "Explainability data was not returned by the API for "
                        "this prediction."
                    )

            except requests.exceptions.RequestException as e:
                st.error(f"API Connection Error: {str(e)}")

            except Exception as e:
                st.error(f"Application Error: {str(e)}")

    else:
        st.info(
            "Awaiting customer data. Fill out the profile on the left and click "
            "'Analyze Churn Risk'."
        )