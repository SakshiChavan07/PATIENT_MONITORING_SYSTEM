import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and label encoder
model = joblib.load("stress_model.pkl")
le = joblib.load("label_encoder_stress.pkl")

st.title("Patient Monitoring & Vital Prediction System")
st.write("Predict Stress Level from vitals (Manual input or Real-time)")

# --- User Input Section ---
st.header("Manual Input")
heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=180, value=75)
hrv = st.number_input("HRV (ms)", min_value=10, max_value=150, value=50)
spo2 = st.number_input("SpO₂ (%)", min_value=80, max_value=100, value=98)
resp_rate = st.number_input("Respiration Rate (breaths/min)", min_value=10, max_value=40, value=16)
temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)

# Combine input into array
user_data = np.array([[heart_rate, hrv, spo2, resp_rate, temp]])

if st.button("Predict Stress Level"):
    pred = model.predict(user_data)
    pred_label = le.inverse_transform(pred)
    st.success(f"Predicted Stress Level: {pred_label[0]}")

# --- Real-time input simulation ---
st.header("Real-Time Input Simulation")
uploaded_file = st.file_uploader("Upload CSV file from sensors (HeartRate, HRV, SpO2, Respiration, Temp)", type="csv")

if uploaded_file is not None:
    sensor_data = pd.read_csv(uploaded_file)
    predictions = model.predict(sensor_data[features])
    sensor_data['Predicted_Stress_Level'] = le.inverse_transform(predictions)
    st.dataframe(sensor_data)
