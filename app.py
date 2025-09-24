import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Patient Monitoring & Vital Prediction", layout="wide")

st.title("Patient Monitoring & Vital Prediction System")
st.write("Predict Stress Level, Cardio-Respiratory Risk, and Fever Risk from patient vitals")

# Load models and label encoders
stress_model = joblib.load("stress_model.pkl")
stress_le = joblib.load("label_encoder_stress.pkl")

cardio_model = joblib.load("cardio_model.pkl")
cardio_le = joblib.load("label_encoder_cardio.pkl")

fever_model = joblib.load("fever_model.pkl")
fever_le = joblib.load("label_encoder_fever.pkl")

# Features
features = ['Heart_Rate', 'HRV', 'SpO2', 'Respiration_Rate', 'Temperature']

# --- Patient Info ---
st.header("Patient Information")
name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])

# --- Manual Input Section ---
st.header("Manual Input")
heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=180, value=75)
hrv = st.number_input("HRV (ms)", min_value=10, max_value=150, value=50)
spo2 = st.number_input("SpO₂ (%)", min_value=80, max_value=100, value=98)
resp_rate = st.number_input("Respiration Rate (breaths/min)", min_value=10, max_value=40, value=16)
temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)

user_data = np.array([[heart_rate, hrv, spo2, resp_rate, temp]])

if st.button("Predict from Manual Input"):
    stress_pred = stress_model.predict(user_data)
    cardio_pred = cardio_model.predict(user_data)
    fever_pred = fever_model.predict(user_data)

    # Decode labels
    stress_label = stress_le.inverse_transform(stress_pred)[0]
    cardio_label = cardio_le.inverse_transform(cardio_pred)[0]
    fever_label = fever_le.inverse_transform(fever_pred)[0]

    st.success(f"Patient: {name}, Age: {age}, Gender: {gender}")
    st.write(f"**Predicted Stress Level:** {stress_label}")
    st.write(f"**Predicted Cardio-Respiratory Risk:** {cardio_label}")
    st.write(f"**Predicted Fever Risk:** {fever_label}")

    # Save report
    report_df = pd.DataFrame({
        "Patient_Name": [name],
        "Age": [age],
        "Gender": [gender],
        "Heart_Rate": [heart_rate],
        "HRV": [hrv],
        "SpO2": [spo2],
        "Respiration_Rate": [resp_rate],
        "Temperature": [temp],
        "Stress_Level": [stress_label],
        "Cardio_Resp_Risk": [cardio_label],
        "Fever_Risk": [fever_label]
    })
    report_csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Patient Report", data=report_csv, file_name=f"{name}_report.csv", mime="text/csv")

# --- Real-Time / CSV Input Section ---
st.header("Real-Time Input / Sensor CSV Upload")
uploaded_file = st.file_uploader("Upload CSV file from sensors (columns: Heart_Rate, HRV, SpO2, Respiration_Rate, Temperature)", type="csv")

if uploaded_file is not None:
    sensor_data = pd.read_csv(uploaded_file)

    # Predict
    stress_pred_rt = stress_model.predict(sensor_data[features])
    cardio_pred_rt = cardio_model.predict(sensor_data[features])
    fever_pred_rt = fever_model.predict(sensor_data[features])

    # Decode
    sensor_data['Stress_Level'] = stress_le.inverse_transform(stress_pred_rt)
    sensor_data['Cardio_Resp_Risk'] = cardio_le.inverse_transform(cardio_pred_rt)
    sensor_data['Fever_Risk'] = fever_le.inverse_transform(fever_pred_rt)

    st.dataframe(sensor_data)

    # Save report
    report_csv_rt = sensor_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Report", data=report_csv_rt, file_name=f"{name}_real_time_report.csv", mime="text/csv")



"""
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
    """
