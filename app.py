import streamlit as st
import pickle
import numpy as np

# Load model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Heart Attack Risk Prediction")

# Input fields
age = st.number_input("Age", 0, 120)
sex = st.selectbox("Sex (1 = male, 0 = female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", [1, 0])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
exang = st.selectbox("Exercise Induced Angina (1 = yes; 0 = no)", [1, 0])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1, 2, 3])

# Prediction
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

if st.button("Predict"):
    result = model.predict(features)
    st.success(f"Heart Attack Risk: {'Yes (1)' if result[0]==1 else 'No (0)'}")
