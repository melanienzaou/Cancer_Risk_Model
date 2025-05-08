# app.py
# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Cancer Risk Predictor",
    page_icon="🧬",
    layout="centered"
)

# --- Load model and features ---
model, feature_names = joblib.load("cancer_risk_model.pkl")
SYMPTOM_LIST = [f for f in feature_names if f not in ["race_encoded", "age_years"]]

# --- Header ---
st.title("🧬 Cancer Risk Prediction Demo")
st.caption("Enter your symptoms, age, and race to assess possible brain cancer risk.")

st.markdown("---")

# --- Input Form ---
st.subheader("Patient Information")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Your age", 18, 90, 40)
    race = st.radio("Race", ["White", "Black or African American"])

with col2:
    symptom_text = st.text_area("Describe your symptoms (e.g. headache, fatigue, blurry vision):")

# --- Convert Inputs ---
user_symptoms = [s.strip().lower() for s in symptom_text.split(",") if s.strip()]
symptom_flags = [1 if symptom in user_symptoms else 0 for symptom in SYMPTOM_LIST]
race_encoded = 0 if race.lower() == "white" else 1
input_data = [race_encoded, age] + symptom_flags
input_df = pd.DataFrame([input_data], columns=feature_names)

# --- Prediction ---
if st.button("Predict Cancer Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    risk_level = "High" if prediction == 1 else "Low"
    color = "🟥" if prediction == 1 else "🟩"

    st.markdown("---")
    st.subheader("🔍 Risk Prediction:")
    st.markdown(f"### {color} **{risk_level} Risk**")
    st.metric(label="Model Confidence", value=f"{probability * 100:.2f}%")

    if prediction == 1:
        st.warning("⚠️ This patient may be at elevated risk. Please consult a medical professional.")
    else:
        st.success("✅ This patient appears to be at low estimated risk.")

    # --- Feature Breakdown ---
    st.markdown("---")
    st.subheader("🔬 Feature Breakdown")
    feature_contrib = pd.DataFrame(
        [symptom_flags],
        columns=SYMPTOM_LIST
    ).T.rename(columns={0: "Reported"}).astype(int)
    st.bar_chart(feature_contrib)

st.markdown("---")
st.info("This tool is powered by real clinical data from the NIH GDC (May 2025).")
st.caption("Developed by Melanie, Desiree and Jola • Final Presentation • Spring 2025")
