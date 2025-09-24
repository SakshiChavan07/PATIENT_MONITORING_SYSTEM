
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="Patient Monitoring & Prediction", layout="wide")
st.title("Patient Monitoring & Vital Prediction System")

# Load models -
stress_model = joblib.load("stress_model.pkl")
stress_le = joblib.load("label_encoder_stress.pkl")
cardio_model = joblib.load("cardio_model.pkl")
cardio_le = joblib.load("label_encoder_cardio.pkl")
fever_model = joblib.load("fever_model.pkl")
fever_le = joblib.load("label_encoder_fever.pkl")

features = ['Heart_Rate', 'HRV', 'SpO2', 'Respiration_Rate', 'Temperature']

# Sidebar -
mode = st.sidebar.radio("Select Input Mode", ["Manual Input", "Live/CSV Input"])

st.sidebar.header("Patient Information")
name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

# Helper function -
def color_risk(value):
    """Return HTML span with color based on risk level."""
    value_lower = value.lower()
    if value_lower == "high":
        return f"<span style='color:red;font-weight:bold'>{value}</span>"
    elif value_lower == "medium":
        return f"<span style='color:orange;font-weight:bold'>{value}</span>"
    else:
        return f"<span style='color:green;font-weight:bold'>{value}</span>"

# Manual Input Mode -
if mode == "Manual Input":
    st.header("Enter Vitals Manually")
    heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=180, value=75)
    hrv = st.number_input("HRV (ms)", min_value=10, max_value=150, value=50)
    spo2 = st.number_input("SpO₂ (%)", min_value=80, max_value=100, value=98)
    resp_rate = st.number_input("Respiration Rate (breaths/min)", min_value=10, max_value=40, value=16)
    temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)

    user_data = np.array([[heart_rate, hrv, spo2, resp_rate, temp]])

    if st.button("Predict Parameters"):
        stress_pred = stress_model.predict(user_data)
        cardio_pred = cardio_model.predict(user_data)
        fever_pred = fever_model.predict(user_data)

        stress_label = stress_le.inverse_transform(stress_pred)[0]
        cardio_label = cardio_le.inverse_transform(cardio_pred)[0]
        fever_label = fever_le.inverse_transform(fever_pred)[0]

        st.markdown(f"### {name}, this is your health report. Stay Healthy! 🎯")
        st.markdown(f"**Stress Level:** {color_risk(stress_label)}", unsafe_allow_html=True)
        st.markdown(f"**Cardio-Respiratory Risk:** {color_risk(cardio_label)}", unsafe_allow_html=True)
        st.markdown(f"**Fever Risk:** {color_risk(fever_label)}", unsafe_allow_html=True)

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
        st.download_button("Download Full Report", data=report_csv,
                           file_name=f"{name}_report.csv", mime="text/csv")

# Live/CSV Input Mode -
elif mode == "Live/CSV Input":
    st.header("Upload CSV from Sensors / Real-Time Data")
    st.write("CSV must have columns: Heart_Rate, HRV, SpO2, Respiration_Rate, Temperature")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is None:
        st.warning("Arduino is not connected or CSV not uploaded. Please check and try again!")
    else:
        sensor_data = pd.read_csv(uploaded_file)

        stress_pred = stress_model.predict(sensor_data[features])
        cardio_pred = cardio_model.predict(sensor_data[features])
        fever_pred = fever_model.predict(sensor_data[features])

        sensor_data['Stress_Level'] = stress_le.inverse_transform(stress_pred)
        sensor_data['Cardio_Resp_Risk'] = cardio_le.inverse_transform(cardio_pred)
        sensor_data['Fever_Risk'] = fever_le.inverse_transform(fever_pred)

        sensor_data.insert(0, "Patient_Name", name)
        sensor_data.insert(1, "Age", age)
        sensor_data.insert(2, "Gender", gender)

        st.markdown(f"### {name}, this is your health report. Stay Healthy! 🎯")

        def highlight_risk(row):
            return ['','','','','','',
                    f'color:red' if row['Stress_Level'].lower()=='high' else 'color:orange' if row['Stress_Level'].lower()=='medium' else 'color:green',
                    f'color:red' if row['Cardio_Resp_Risk'].lower()=='high' else 'color:orange' if row['Cardio_Resp_Risk'].lower()=='medium' else 'color:green',
                    f'color:red' if row['Fever_Risk'].lower()=='high' else 'color:orange' if row['Fever_Risk'].lower()=='medium' else 'color:green']

        st.dataframe(sensor_data.style.apply(highlight_risk, axis=1))

        report_csv = sensor_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Report", data=report_csv,
                           file_name=f"{name}_live_report.csv", mime="text/csv")

