import streamlit as st
import requests
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"

st.title("Fraud Detection Dashboard")

st.write("Enter 30 transaction feature values:")

features = []

for i in range(30):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(value)

if st.button("Predict"):

    response = requests.post(API_URL, json={"features": features})

    if response.status_code == 200:
        result = response.json()

        st.subheader("Prediction Result")
        st.write("Fraud Probability:", result["fraud_probability"])
        st.write("Is Fraud:", result["is_fraud"])
        st.write("Threshold Used:", result["threshold_used"])

        if result["is_fraud"] == 1:
            st.error("⚠️ Fraud Detected")
        else:
            st.success("✅ Transaction is Safe")

    else:
        st.error("API Error")