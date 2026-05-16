import streamlit as st
import pandas as pd
import joblib
import os

# 1. CRISP PAGE CONFIGURATIONS
st.set_page_config(
    page_title="SmartCare Underwriting Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CLEAN UI STYLES (Avoids looking like a basic template)
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.25rem; }
    .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 2rem; }
    .card-container { background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 1rem; }
    .metric-box { background-color: #EFF6FF; border: 1px solid #BFDBFE; padding: 1.5rem; border-radius: 0.75rem; text-align: center; }
    .metric-label { font-size: 0.95rem; font-weight: 600; color: #1E40AF; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 2.4rem; font-weight: 800; color: #1E3A8A; margin-top: 0.25rem; }
    </style>
""", unsafe_allow_html=True)

# 3. PERFORMANCE OPTIMIZED MODEL LOADING
@st.cache_resource
def load_underwriting_model():
    model_path = 'health_insurance_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline = load_underwriting_model()

# 4. SIDEBAR INPUT CONTROLS (Clean, tactile parameters)
with st.sidebar:
    st.image("https://img.icons8.com/external-flatart-icons-flat-flatarticons/128/external-shield-medical-health-care-flatart-icons-flat-flatarticons.png", width=60)
    st.markdown("### **Risk Parameters**")
    st.caption("Adjust core biographical metrics to evaluate premium risk variance live.")
    st.markdown("---")
    
    age = st.slider("Primary Applicant Age", min_value=18, max_value=100, value=35, step=1)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=72.5, step=0.5)
    bmi = st.number_input("Calculated BMI", min_value=10.0, max_value=60.0, value=24.2, step=0.1)
    bloodpressure = st.slider("Systolic Blood Pressure", min_value=80, max_value=200, value=120, step=1)
    no_of_dependents = st.number_input("Dependents Scheduled", min_value=0, max_value=10, value=0, step=1)
    
    st.markdown("---")
    city = st.selectbox("Geographic Rating Area (City)", options=['NewYork', 'Boston', 'Phildelphia', 'Pittsburg', 'Buffalo', 'AtlanticCity', 'Portland', 'Cambridge', 'Hartford', 'Springfield', 'Brimingham', 'Charleston', 'Charlotte', 'Louisville', 'Memphis', 'Nashville', 'NewOrleans'])

# 5. MAIN WORKSPACE HEADER
st.markdown('<div class="main-header">SmartCare Underwriting Risk Interface</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated cost prediction engine built using historical health claim models.</div>', unsafe_allow_html=True)

if pipeline is None:
    st.error("🚨 Core engine pipeline file (`health_insurance_model.pkl`) not found in the current directory. Please make sure to run your training notebook/script to save the model artifact first.")
else:
    # 6. DUAL COLUMN WORKSPACE STRUCTURE
    col1, col2 = st.columns([5, 4], gap="large")
    
    with col1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown("#### **Clinical & Behavioral Profile**")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Horizontal toggle configuration for cleaner alignment
        sub_c1, sub_c2, sub_c3 = st.columns(3)
        with sub_c1:
            smoker_input = st.toggle("Nicotine/Smoker", value=False)
        with sub_c2:
            diabetes_input = st.toggle("Diagnosed Diabetes", value=False)
        with sub_c3:
            exercise_input = st.toggle("Regular Exercise", value=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        has_hereditary = st.selectbox("Genetic / Hereditary Status", options=["No Stated Hereditary History", "Known Hereditary Condition History"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown("#### **Occupational Demographics**")
        job_category = st.selectbox("Professional Domain Cluster", options=['Office', 'Technical', 'Medical', 'Service', 'Creative', 'Other'])
        
        # Under-the-hood feature derivations matching your notebook logic
        if bmi < 18.5: bmi_grouping = 'Underweight'
        elif bmi < 25: bmi_grouping = 'Healthy'
        elif bmi < 30: bmi_grouping = 'Overweight'
        else: bmi_grouping = 'Obese'
            
        st.markdown(f"**Automated Segment Mapping:** `{bmi_grouping}`")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### **Analysis & Premium Valuation**")
        st.caption("Click below to pass inputs into the Random Forest Regressor framework.")
        
        # Parse inputs into the precise structures expected by your scikit-learn ColumnTransformer
        smoker = 1.0 if smoker_input else 0.0
        diabetes = 1.0 if diabetes_input else 0.0
        regular_ex = 1.0 if exercise_input else 0.0
        has_hereditary_disease = 1.0 if has_hereditary == "Known Hereditary Condition History" else 0.0
        smoker_age_risk = smoker * age
        
        user_payload = pd.DataFrame([{
            'age': float(age), 'sex': 'male', 'weight': float(weight), 'bmi': float(bmi),
            'no_of_dependents': float(no_of_dependents), 'smoker': smoker, 'city': city,
            'bloodpressure': float(bloodpressure), 'diabetes': diabetes, 'regular_ex': regular_ex,
            'smoker_age_risk': smoker_age_risk, 'has_hereditary_disease': has_hereditary_disease,
            'bmi_grouping': bmi_grouping, 'job_category': job_category
        }])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Calculate Expected Insurance Claim", type="primary", use_container_width=True):
            # Pass the single-row dataframe cleanly directly into your pipeline predict function
            predicted_claim = pipeline.predict(user_payload)[0]
            
            # Render custom styled continuous output component
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Predicted Annual Cost Valuation</div>
                    <div class="metric-value">${predicted_claim:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Domain context text block
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 **Underwriting Note:** This output represents the model's localized mean regression baseline score. Actual policy quotes may vary based on structural market adjustments.")
