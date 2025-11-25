import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load("../ML_models/smartcare_model.pkl")

st.set_page_config(page_title="SmartCare Fraud Detection", layout="wide")
st.title("SmartCare Health Insurance Claim Fraud Detection")
st.write("Streamlit")

st.sidebar.header("Upload Claims Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Claim_Date", "Service_Date", "Policy_Expiration_Date"])
    df['Claim_Service_Delay'] = (df['Claim_Date'] - df['Service_Date']).dt.days
    df['Days_To_Expiration_Proximity'] = (df['Policy_Expiration_Date'] - df['Claim_Date']).dt.days
    df['Claim_Month'] = df['Claim_Date'].dt.month
    df['Claim_DOW'] = df['Claim_Date'].dt.dayofweek

    date_columns_to_drop = ['Claim_Date','Service_Date','Policy_Expiration_Date']
    df_model = df.drop(columns=date_columns_to_drop + ["Patient_ID","Claim_ID","Policy_Number","Hospital_ID"])

    X = df_model.drop('Is_Fraudulent', axis=1, errors='ignore')

    df["Fraud_Probability"] = pipeline.predict_proba(X)[:,1]
    df["Fraud_Prediction"] = pipeline.predict(X)

    st.subheader("Batch Prediction Results")
    st.dataframe(df[["Fraud_Probability", "Fraud_Prediction"]].head(20))


st.sidebar.header("Predict Single Claim")

if st.sidebar.button("Add Single Claim"):
    st.sidebar.info("Enter claim details for prediction:")

    patient_age = st.sidebar.number_input("Patient Age", min_value=0, max_value=120, value=30)
    patient_gender = st.sidebar.selectbox("Patient Gender", options=["Male", "Female", "Other"])
    claim_amount = st.sidebar.number_input("Claim Amount", min_value=0, value=1000)
    claim_service_delay = st.sidebar.number_input("Claim Service Delay (days)", value=2)
    days_to_expiration_proximity = st.sidebar.number_input("Days To Expiration Proximity", value=30)
    claim_month = st.sidebar.number_input("Claim Month", min_value=1, max_value=12, value=6)
    claim_dow = st.sidebar.number_input("Claim Day of Week", min_value=0, max_value=6, value=2)

    single_claim = pd.DataFrame({
        "Patient_Age": [patient_age],
        "Patient_Gender": [patient_gender],
        "Claim_Amount": [claim_amount],
        "Claim_Service_Delay": [claim_service_delay],
        "Days_To_Expiration_Proximity": [days_to_expiration_proximity],
        "Claim_Month": [claim_month],
        "Claim_DOW": [claim_dow]
    })

    prediction = pipeline.predict(single_claim)
    probability = pipeline.predict_proba(single_claim)[:,1]

    st.subheader("Single Claim Prediction")
    st.write(f"Fraud Prediction: {prediction[0]}")
    st.write(f"Fraud Probability: {probability[0]:.2f}")
