# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Configure page
st.set_page_config(page_title="Cancer Risk Predictor", page_icon="🧬", layout="centered")

# Load model and features
model, feature_names = joblib.load("cancer_risk_model.pkl")

# Define symptom fields
SYMPTOM_LIST = [f for f in feature_names if f not in ["race_encoded", "age_years"]]

# Header
st.title("🧬 Cancer Risk Prediction Demo")
st.caption("Estimate your risk based on symptoms, age, and race — powered by real-world clinical data.")

st.markdown("---")

# Input section
st.subheader("Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 10, 100, 40)
    race = st.radio("Race", options=["White", "Black or African American"])

with col2:
    symptom_text = st.text_area("Describe your symptoms", placeholder="e.g. headache, blurry vision", height=100)

# Convert text input to symptom flags
user_symptoms = [s.strip().lower() for s in symptom_text.split(",") if s.strip()]
symptom_flags = [1 if symptom in user_symptoms else 0 for symptom in SYMPTOM_LIST]
race_encoded = 0 if race.lower() == "white" else 1
input_data = [race_encoded, age] + symptom_flags
input_df = pd.DataFrame([input_data], columns=feature_names)

# Prediction
if st.button("Predict Cancer Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    risk_level = "High" if prediction == 1 else "Low"
    color = "🟥" if risk_level == "High" else "🟩"

    st.markdown(f"### {color} Risk Prediction: **{risk_level}**")
    st.metric(label="Confidence", value=f"{probability * 100:.2f}%")

    if risk_level == "High":
        st.warning("This patient may be at elevated risk based on their inputs.")
    else:
        st.success("This patient appears to be at low estimated risk.")

st.markdown("---")
st.info("This tool is powered by a real clinical dataset (NIH GDC, April 2025).")
