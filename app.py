# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and trained feature list
model, trained_columns = joblib.load("cancer_risk_model.pkl")

# Symptom keywords → model feature mapping
SYMPTOM_FEATURES = ["blurry_vision", "dizziness", "fatigue", "headache", "nausea", "pain", "seizure"]

SYMPTOM_KEYWORDS = {
    "headache": "headache",
    "vision": "blurry_vision",
    "blurry": "blurry_vision",
    "seizure": "seizure",
    "nausea": "nausea",
    "fatigue": "fatigue",
    "pain": "pain",
    "dizzy": "dizziness",
    "dizziness": "dizziness"
}

def extract_features(text, age, race):
    features = {sym: 0 for sym in SYMPTOM_FEATURES}
    for word, key in SYMPTOM_KEYWORDS.items():
        if word in text.lower():
            features[key] = 1
    features["age_years"] = age
    features["race_encoded"] = 0 if race == "White" else 1
    # Align with training columns
    df = pd.DataFrame([features])
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0
    return df[trained_columns]

# Streamlit UI
st.title("🧬 Cancer Risk Predictor Demo")
st.write("Enter your symptoms, age, and race to assess possible  brain cancer risk.")

user_text = st.text_area("Describe your symptoms (e.g. headache, fatigue, blurry vision):")
user_age = st.slider("Your age", 18, 90, 50)
user_race = st.selectbox("Race", ["White", "Black or African American"])

if st.button("Predict Risk"):
    if not user_text.strip():
        st.warning("Please describe your symptoms.")
    else:
        input_df = extract_features(user_text, user_age, user_race)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown(f"### 🔍 Risk Prediction: {'🟥 High' if prediction else '🟩 Low'}")
        st.markdown(f"**Model Confidence:** `{round(probability * 100, 2)}%`")

        with st.expander("🔬 Feature Breakdown"):
            st.write(input_df.T.rename(columns={0: "Value"}))
