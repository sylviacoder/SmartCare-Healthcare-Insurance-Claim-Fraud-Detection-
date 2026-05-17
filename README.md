# SmartCare-Healthcare-Insurance-Claim-Fraud-Detection-

This project is a machine learning application designed to automate medical risk assessment and forecast premium claim costs. The model predicts individual annual insurance claim valuations using demographic indicators, lifestyle profiles, and clinical risk factors collected from historical insurance underwriting data.

The application was built using Scikit-learn and deployed with Streamlit as an interactive web app.

## Project Overview
Insurance claim decisions are influenced by several factors such as:
* applicant age and biological wear
* continuous body mass index (BMI) metrics
* tobacco use and lifestyle behaviors
* diagnosed chronic conditions (e.g., diabetes)
* hereditary disease histories
* occupational hazard domains

The goal of this project is to analyze these intersecting patterns and build a highly dependable predictive regression system capable of automating underwriting valuations. The project also explores structural feature engineering strategies to bypass data noise and handle severe non-linear threshold variations.

## Problem Statement
* average absolute errors tripled, pushing predictions off by an operational margin of $3,848.24
* standard linear models failed to map conditional dependencies (such as the compounding risk of aging smokers) without complex polynomial expansions

To improve model stability and prediction precision, the framework was transitioned to an ensemble non-parametric architecture using a Random Forest Regressor. This approach successfully isolated non-linear threshold cliffs and delivered significantly tighter error bounds across consumer profiles.

## Features Used
The model was trained using the following features:
* age
* weight
* bmi
* no_of_dependents
* bloodpressure
* smoker
* diabetes
* regular_ex
* sex
* city
* bmi_grouping *(Engineered)*
* smoker_age_risk *(Engineered)*
* has_hereditary_disease *(Engineered)*
* job_category *(Engineered)*

These features collectively capture physical metrics, behavioral inputs, and structural risk categories essential for accurate pricing estimation.

## Machine Learning Pipeline
The project uses a complete preprocessing and modeling pipeline built with Scikit-learn to guarantee deterministic behavior at runtime and eliminate data leakage.
* **Preprocessing**
  * Numerical scaling using StandardScaler
  * Categorical encoding using OneHotEncoder
  * Multi-channel transformation via ColumnTransformer
* **Modeling**
  * RandomForestRegressor (n_estimators=200, max_depth=10) for robust tree-based regression splitting

Using an encapsulated pipeline ensured that training-set scaling weights and matrix dimensions remained identical and deployment-safe within the production app.

## Model Evaluation

The model was evaluated using:
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R2 Score (Variance Explained)
* Train-Test Cross-Validation Splits

Validation execution output:
```bash
MODEL EVALUATION RESULTS       
Mean Absolute Error (MAE): $1463.58
Root Mean Squared Error (RMSE): $3067.40
R2 Score (Variance Explained): 0.9370
```

Given the structural performance of the ensemble pipeline, the focus of the project was placed on:
high-precision risk mapping (93.70% variance explanation)
thread-safe model serialization
processing unstructured end-user inputs
deployment-ready pipeline consistency

## Feature Importance
Feature importance analysis was used to understand which clinical and lifestyle variables exerted the strongest mathematical push on claim pricing.
Some of the most influential features included:

Tobacco smoking status
Smoker-to-age cross-product interaction (smoker_age_risk)
Continuous systolic blood pressure
Engineered obesity step-functions (bmi_grouping)
Binary genetic history (has_hereditary_disease)

Isolating these features provided strict transparency and clear medical justification for automated premium pricing adjustments.

## Streamlit Application
The project includes a Streamlit web application where users can:

Enter applicant demographic and lifestyle profiles
Receive an instant annual premium cost prediction
Clear automatic alerts for high-risk interaction thresholds
Type unstructured occupations that parse into clean domains on-the-fly

## Technologies Used
Python
Pandas
NumPy
Scikit-learn
Joblib
Matplotlib
Seaborn

## Streamlit

Project Structure
```Bash
SmartCare-Healthcare-Insurance-Claim-Fraud-Detection-/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── Health_Insurance.csv
│   └── processed/
│       └── processed_data.csv
│
├── models/
│   └── health_insurance_model.pkl
│
└── notebooks/
    └── Exploratory_Analysis.ipynb
    └── Model Training and Evaluation.ipynb
```
## Installation
Clone the repository:

```Bash
git clone https://github.com/sylviacoder/SmartCare-Healthcare-Insurance-Claim-Fraud-Detection-/tree/main
```
```
cd SmartCare-Healthcare-Insurance-Claim-Fraud-Detection-
```

Install dependencies:
```Bash
pip install -r requirements.txt
```
Run the application:
```Bash
streamlit run app.py
```

## Challenges Encountered
A major part of the project involved debugging feature imbalances and structural execution paths during production deployment.
Some of the challenges included:

High-Cardinality Sparsity: Raw text inputs for hereditary diseases and job titles contained rare, sparse records that destabilized cross-validation splits. This was resolved by designing heuristic keyword-parsers and a binary indicator (has_hereditary_disease) to consolidate classes.

Schema Alignment: Mismatch between the open text field inputs in Streamlit and the rigid data matrix structure expected by the model. This was solved by hardcoding matching transformation rules inside the UI script right before calling the model inference function.

Crash Resilience: Preventing application failure when encountering unobserved categories. This was fixed by establishing OneHotEncoder(handle_unknown='ignore') boundaries to safely ignore out-of-vocabulary inputs.

## Future Improvements
Possible improvements for future versions include:

Expanding the dataset across broader geographic regional models
Implementing SHAP or LIME value visualizers inside the user interface
Deploying the core pipeline via an asynchronous FastAPI microservice
Integrating automated anomaly detection flags for extreme medical claims
Benchmarking against advanced tree boosting architectures (e.g., XGBoost, LightGBM)

## Conclusion
This project demonstrates an end-to-end data engineering and predictive modeling workflow covering:
structured clinical feature extraction
targeted data preprocessing and leak protection
unified production pipeline compilation
residual-based model validation metrics
low-latency user deployment using Streamlit

It effectively showcases how software engineering design principles can turn a complex medical underwriting dataset into a reliable, automated software asset.
