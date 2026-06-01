from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_model.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
META_PATH = ARTIFACT_DIR / "churn_preprocess_meta.pkl"


BINARY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "PaperlessBilling": ["No", "Yes"],
    "SeniorCitizen": [0, 1],
}

MULTI_LINE_OPTIONS = ["No phone service", "No", "Yes"]
INTERNET_OPTIONS = ["No", "DSL", "Fiber optic"]
SERVICE_OPTIONS = ["No internet service", "No", "Yes"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAYMENT_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


@st.cache_resource

def load_artifacts() -> tuple:
    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Missing artifacts. Run the notebook cells that save the model, scaler, and preprocessing metadata first."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta = joblib.load(META_PATH)
    return model, scaler, meta


def preprocess_customer_input(raw_input_df: pd.DataFrame, preprocess_meta: dict, scaler) -> pd.DataFrame:
    df_inp = raw_input_df.copy()

    df_inp["TotalCharges"] = pd.to_numeric(df_inp["TotalCharges"], errors="coerce")
    df_inp["TotalCharges"] = df_inp["TotalCharges"].fillna(preprocess_meta["total_charges_median"])
    df_inp["AvgMonthlySpend"] = df_inp["TotalCharges"] / (df_inp["tenure"] + 1)
    df_inp["TenureBucket"] = pd.cut(
        df_inp["tenure"],
        bins=[0, 12, 36, 72],
        labels=[0, 1, 2],
        include_lowest=True,
    ).astype(int)

    binary_maps = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
        "SeniorCitizen": {0: 0, 1: 1, "0": 0, "1": 1},
    }

    for col in preprocess_meta["binary_cols"]:
        if col in df_inp.columns and col in binary_maps:
            df_inp[col] = df_inp[col].map(binary_maps[col])

    df_inp = pd.get_dummies(df_inp, columns=preprocess_meta["ohe_cols"], drop_first=True)
    df_inp = df_inp.reindex(columns=preprocess_meta["feature_columns"], fill_value=0)
    df_inp[preprocess_meta["num_scale_cols"]] = scaler.transform(df_inp[preprocess_meta["num_scale_cols"]])
    return df_inp


def predict_churn(raw_input_df: pd.DataFrame, threshold: float | None = None) -> tuple[float, int, dict]:
    model, scaler, meta = load_artifacts()
    threshold_value = meta["default_threshold"] if threshold is None else threshold
    features = preprocess_customer_input(raw_input_df, meta, scaler)
    churn_probability = float(model.predict_proba(features)[0, 1])
    churn_prediction = int(churn_probability >= threshold_value)
    return churn_probability, churn_prediction, meta


st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, rgba(18, 52, 86, 0.10), transparent 30%),
                    radial-gradient(circle at top right, rgba(208, 82, 82, 0.10), transparent 28%),
                    linear-gradient(180deg, #f7f8fb 0%, #eef3f9 100%);
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #102a43 0%, #1d3557 55%, #2a6f97 100%);
        color: white;
        box-shadow: 0 18px 40px rgba(16, 42, 67, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        line-height: 1.1;
    }
    .hero p {
        margin: 0.4rem 0 0;
        opacity: 0.92;
        font-size: 1rem;
    }
    .result-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: white;
        border: 1px solid rgba(16, 42, 67, 0.08);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }
    .label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #5b6573;
        margin-bottom: 0.25rem;
    }
    .big {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .subtle {
        color: #5b6573;
        margin-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Telco Customer Churn Predictor</h1>
        <p>Enter a customer profile, run the saved Gradient Boosting model, and get a churn risk estimate instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col = st.container()

with left_col:
    st.subheader("Customer Profile")
    with st.form("churn_form"):
        form_left, form_right = st.columns(2)

        with form_left:
            gender = st.selectbox("Gender", BINARY_OPTIONS["gender"], index=0)
            senior = st.selectbox("Senior Citizen", BINARY_OPTIONS["SeniorCitizen"], index=0)
            partner = st.selectbox("Partner", BINARY_OPTIONS["Partner"], index=0)
            dependents = st.selectbox("Dependents", BINARY_OPTIONS["Dependents"], index=0)
            phone_service = st.selectbox("Phone Service", BINARY_OPTIONS["PhoneService"], index=1)
            multiple_lines = st.selectbox("Multiple Lines", MULTI_LINE_OPTIONS, index=1)
            internet_service = st.selectbox("Internet Service", INTERNET_OPTIONS, index=1)
            contract = st.selectbox("Contract", CONTRACT_OPTIONS, index=0)

        with form_right:
            paperless = st.selectbox("Paperless Billing", BINARY_OPTIONS["PaperlessBilling"], index=1)
            payment_method = st.selectbox("Payment Method", PAYMENT_OPTIONS, index=0)
            tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=0.05)
            total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=900.0, step=1.0)

            online_security = st.selectbox("Online Security", SERVICE_OPTIONS, index=0)
            online_backup = st.selectbox("Online Backup", SERVICE_OPTIONS, index=0)
            device_protection = st.selectbox("Device Protection", SERVICE_OPTIONS, index=0)
            tech_support = st.selectbox("Tech Support", SERVICE_OPTIONS, index=0)
            streaming_tv = st.selectbox("Streaming TV", SERVICE_OPTIONS, index=0)
            streaming_movies = st.selectbox("Streaming Movies", SERVICE_OPTIONS, index=0)

        submitted = st.form_submit_button("Predict churn risk")

    if submitted:
        if phone_service == "No":
            multiple_lines = "No phone service"
        if internet_service == "No":
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"

        input_row = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": senior,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }
            ]
        )

        try:
            churn_probability, churn_prediction, meta = predict_churn(input_row)
            status_label = "High risk" if churn_prediction == 1 else "Low risk"
            status_color = "#d64545" if churn_prediction == 1 else "#2f9e44"

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="label">Prediction</div>
                    <p class="big" style="color: {status_color};">{status_label}</p>
                    <p class="subtle">Churn probability: <strong>{churn_probability:.1%}</strong></p>
                    <p class="subtle">Decision threshold: <strong>{meta['default_threshold']:.2f}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(int(round(churn_probability * 100)))

            if churn_prediction == 1:
                st.warning("This customer profile is being classified as likely to churn. Consider a retention offer or proactive outreach.")
            else:
                st.success("This customer profile is being classified as low churn risk.")

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

