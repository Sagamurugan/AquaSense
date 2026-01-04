import streamlit as st
import joblib
from pathlib import Path
import numpy as np
import os

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "tn_random_forest_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "tn_scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.title("🌊 AquaSense - TN Prediction System")

st.write("Enter water quality parameters to predict TN level")

tp = st.number_input("Total Phosphorus (TP)", format="%.3f")
nh3 = st.number_input("Ammonia (NH3)", format="%.3f")
no23 = st.number_input("Nitrate (NO23)", format="%.3f")
op = st.number_input("Orthophosphate (OP)", format="%.3f")

if st.button("Predict TN"):
    data = np.array([[tp, nh3, no23, op]])
    scaled = scaler.transform(data)
    prediction = model.predict(scaled)[0]
    prediction = float(prediction[0])


    st.success(f"Predicted TN Level: {prediction:.3f} mg/L")

    if prediction < 1:
        st.info("Water Quality: GOOD")
    elif prediction < 3:
        st.warning("Water Quality: MODERATE")
    else:
        st.error("Water Quality: POOR")
