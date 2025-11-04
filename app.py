import streamlit as st
import joblib
import numpy as np

model = joblib.load("model_file.pkl")
scaler = joblib.load("stdmodel_file.pkl")
st.title("Student Performance Prediction App")
st.write("Fill in the details below to predict overall student performance:")
# Input fields
iq = st.number_input("IQ", min_value=0, max_value=200, step=1)
prev_sem = st.number_input("Previous Semester Result", min_value=0.0, max_value=100.0, step=0.1)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
academic = st.number_input("Academic Performance (1-10)", min_value=0, max_value=10, step=1)
internship_exp = st.selectbox("Internship Experience", ["Yes", "No"])
internship = 1 if internship_exp == "Yes" else 0
extra = st.number_input("Extracurricular Score (1-10)", min_value=0, max_value=10, step=1)
communication = st.number_input("Communication Skill (1-10)", min_value=0, max_value=10, step=1)
projected = st.number_input("Project Completed (1-10)", min_value=0, max_value=10, step=1)
# Prediction button
if st.button("Predict Performance"):
    features = np.array([[iq, prev_sem, cgpa, academic, internship, extra, communication, projected]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    result = "Yes you will get placement in a company as you have a great performance" if prediction[0] == 1 else "No you won't get a placement "
    proba = model.predict_proba(scaled_features)[0]
    confidence = round(np.max(proba) * 100, 2)
    # Display result
    st.success(f" Predicted Performance: {result}")
    st.info(f"🧠 Model Confidence: {confidence}%")
