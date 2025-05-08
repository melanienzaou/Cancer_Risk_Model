# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and feature names
model, feature_names = joblib.load("cancer_risk_model.pkl")

SYMPTOM_LIST = [f for f in feature_names if f not in ["race_encoded", "age_years"]]

# App layout
st.title("🧬 Cancer Risk Prediction Demo")

st.markdown("Enter your information and symptoms to check your estimated cancer risk:")

# User inputs
age = st.slider("Age", 10, 100, 40)
race = st.radio("Race", options=["White", "Black or African American"])
symptom_text = st.text_input("Describe your symptoms (comma-separated):", placeholder="e.g. headache, fatigue")

# Parse symptoms into feature flags
user_symptoms = [s.strip().lower() for s in symptom_text.split(",") if s.strip()]
symptom_flags = [1 if symptom in user_symptoms else 0 for symptom in SYMPTOM_LIST]

# Encode race
race_encoded = 0 if race.lower() == "white" else 1

# Build feature vector in correct order
input_data = [race_encoded, age] + symptom_flags
input_df = pd.DataFrame([input_data], columns=feature_names)

# Predict and display
if st.button("Predict Cancer Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"Risk Prediction: {'High' if prediction == 1 else 'Low'}")
    st.write(f"Confidence: **{probability * 100:.2f}%**")
