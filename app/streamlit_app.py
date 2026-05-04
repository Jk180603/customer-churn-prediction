import sys
import os

sys.path.append(os.path.abspath("."))

import streamlit as st
from src.predict import predict_churn


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered",
)

st.title("📉 Customer Churn Prediction Dashboard")
st.write("Predict customer churn risk using an XGBoost model.")

st.divider()

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [1, 0])
is_active_member = st.selectbox("Is Active Member", [1, 0])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)
complain = st.selectbox("Complain", [0, 1])
satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "PLATINUM", "SILVER"])
point_earned = st.number_input("Point Earned", min_value=0, value=500)

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
    "Complain": complain,
    "Satisfaction Score": satisfaction_score,
    "Card Type": card_type,
    "Point Earned": point_earned,
}

if st.button("Predict Churn"):
    result = predict_churn(input_data)

    probability = result["churn_probability"]
    status = result["status"]

    st.subheader("Prediction Result")

    if result["prediction"] == 1:
        st.error(f"⚠️ {status}")
    else:
        st.success(f"✅ {status}")

    st.metric("Churn Probability", f"{probability * 100:.2f}%")

    st.json(result)