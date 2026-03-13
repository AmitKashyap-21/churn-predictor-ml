import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/churn_model.pkl")

# -----------------------------
# Feature list (same as training)
# -----------------------------
FEATURES = [
'gender',
'SeniorCitizen',
'Partner',
'Dependents',
'tenure',
'PhoneService',
'MultipleLines',
'InternetService',
'OnlineSecurity',
'OnlineBackup',
'DeviceProtection',
'TechSupport',
'StreamingTV',
'StreamingMovies',
'Contract',
'PaperlessBilling',
'PaymentMethod',
'MonthlyCharges',
'TotalCharges',
'Feature20',
'Feature21',
'Feature22',
'Feature23',
'Feature24',
'Feature25',
'Feature26',
'Feature27',
'Feature28',
'Feature29',
'Feature30'
]

# -----------------------------
# UI
# -----------------------------
st.title("📊 Customer Churn Predictor")

st.write(
"""
Predict whether a telecom customer is likely to churn.
Enter customer information and click **Predict**.
"""
)

# -----------------------------
# User Inputs
# -----------------------------

tenure = st.slider("Tenure (months)", 0, 72, 12)

monthly_charges = st.slider(
    "Monthly Charges",
    0,
    150,
    70
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

# -----------------------------
# Encoding Inputs
# -----------------------------

gender_map = {"Male":0, "Female":1}
yes_no_map = {"No":0, "Yes":1}
contract_map = {"Month-to-month":0, "One year":1, "Two year":2}

gender_encoded = gender_map[gender]
partner_encoded = yes_no_map[partner]
dependents_encoded = yes_no_map[dependents]
tech_encoded = yes_no_map[tech_support]
contract_encoded = contract_map[contract]

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Churn"):

    # create dataframe with all features
    input_data = pd.DataFrame([[0]*len(FEATURES)], columns=FEATURES)

    # fill known inputs
    input_data["gender"] = gender_encoded
    input_data["Partner"] = partner_encoded
    input_data["Dependents"] = dependents_encoded
    input_data["tenure"] = tenure
    input_data["Contract"] = contract_encoded
    input_data["TechSupport"] = tech_encoded
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = tenure * monthly_charges

    # prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk ({probability:.2f})")
    else:
        st.success(f"✅ Customer Likely to Stay ({probability:.2f})")

    st.write("Churn Probability:", round(probability,3))