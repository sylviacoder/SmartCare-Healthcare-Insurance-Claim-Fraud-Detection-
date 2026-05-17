import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="SmartCare | Insurance Risk Engine",
    layout="centered"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 16px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #1E3A8A;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏥 SmartCare Insurance Risk Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Automated Predictive Underwriting Portal powered by Random Forest Machine Learning</div>', unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return joblib.load('models/health_insurance_model.pkl')
    except FileNotFoundError:
        st.error("⚠️ **Artifact Missing:** 'models/health_insurance_model.pkl' not found! Please ensure you have created a 'models' folder and generated your model into it.")
        return None

pipeline = load_model()

if pipeline is not None:

    st.subheader("📋 Applicant Demographic & Medical Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        sex = st.selectbox("Sex / Gender", options=['male', 'female'])
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=75, step=1)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.4, step=0.1)
        bloodpressure = st.number_input("Systolic Blood Pressure", min_value=80, max_value=220, value=120, step=1)
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1)

    with col2:
        smoker_input = st.selectbox("Tobacco Smoking Status", options=["No", "Yes"])
        diabetes_input = st.selectbox("Diagnosed with Diabetes?", options=["No", "Yes"])
        exercise_input = st.selectbox("Engages in Regular Exercise?", options=["No", "Yes"])
        hereditary_input = st.selectbox("Hereditary Disease Background", options=[
            "None / NoDisease", "Diabetes", "Alzheimer", "Obesity", "EyeDisease", "Cancer", "Arthritis", "HeartDisease", "Epilepsy", "High BP"
        ])
        job_title = st.text_input("Job Title / Occupation", value="Software Engineer")
        city = st.selectbox("Residential City", options=[
            "NewYork", "Boston", "Phildelphia", "Pittsburg", "Buffalo", "AtlanticCity", "Portland", "Cambridge", "Hartford"
        ])


    smoker = 1 if smoker_input == "Yes" else 0
    diabetes = 1 if diabetes_input == "Yes" else 0
    regular_ex = 1 if exercise_input == "Yes" else 0
    has_hereditary_disease = 0 if hereditary_input == "None / NoDisease" else 1

    if bmi < 18.5: bmi_grouping = 'Underweight'
    elif 18.5 <= bmi < 25: bmi_grouping = 'Healthy'
    elif 25 <= bmi < 30: bmi_grouping = 'Overweight'
    else: bmi_grouping = 'Obese'

    smoker_age_risk = smoker * age

    title_lower = str(job_title).lower()
    if any(k in title_lower for k in ['actor', 'singer', 'artist', 'dancer', 'fashion', 'film']): job_category = 'Creative'
    elif any(k in title_lower for k in ['engineer', 'it', 'software', 'data', 'technician']): job_category = 'Technical'
    elif any(k in title_lower for k in ['doctor', 'nurse', 'healthcare', 'medical']): job_category = 'Medical'
    elif any(k in title_lower for k in ['chef', 'cook', 'waiter', 'restaurant', 'service']): job_category = 'Service'
    elif any(k in title_lower for k in ['manager', 'admin', 'clerk', 'executive', 'office']): job_category = 'Office'
    else: job_category = 'Other'


    st.markdown("---")
    
    if st.button("🚀 Calculate Underwriting Valuation", use_container_width=True):
        
        # Build the exact mapping structure expected by your pipeline
        input_data = pd.DataFrame([{
            'age': age,
            'weight': weight,
            'bmi': bmi,
            'no_of_dependents': no_of_dependents,
            'smoker': smoker,
            'bloodpressure': bloodpressure,
            'diabetes': diabetes,
            'regular_ex': regular_ex,
            'smoker_age_risk': smoker_age_risk,
            'has_hereditary_disease': has_hereditary_disease,
            'sex': sex,
            'city': city,
            'bmi_grouping': bmi_grouping,
            'job_category': job_category
        }])
        
        # Execute estimation through the wrapped Scikit-Learn pipeline
        predicted_claim = pipeline.predict(input_data)[0]
        
        st.markdown(f"""
            <div class="metric-box">
                <h4 style="margin:0; color:#4B5563;">Estimated Annual Premium Claim Cost:</h4>
                <p style="font-size:36px; font-weight:800; color:#1E3A8A; margin:5px 0 0 0;">${predicted_claim:,.2f}</p>
                <small style="color:#6B7280;">Based on your 93.70% variance-confident Random Forest regression pipeline model.</small>
            </div>
        """, unsafe_allow_html=True)
        
        if smoker == 1 and bmi >= 30.0:
            st.warning("⚠️ **High Risk Interaction Warning:** The applicant is classified as both a tobacco user and clinically obese. The model heavily adjusts calculations upward due to non-linear cost curves for this interaction threshold.")
