#print("Question no. 10")
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction")

age = st.number_input("Age", 1, 100)

sex = st.selectbox("Sex", ["M","F"])

cp = st.selectbox("Chest Pain Type",
                  ["ATA","NAP","ASY","TA"])

bp = st.number_input("Resting BP")

chol = st.number_input("Cholesterol")

fbs = st.selectbox("Fasting Blood Sugar",
                   [0,1])

ecg = st.selectbox("Resting ECG",
                   ["Normal","ST","LVH"])

hr = st.number_input("Maximum Heart Rate")

angina = st.selectbox("Exercise Angina",
                      ["Y","N"])

oldpeak = st.number_input("Old Peak")

slope = st.selectbox("ST Slope",
                     ["Up","Flat","Down"])

if st.button("Predict"):

    sample = {
        "Age":age,
        "RestingBP":bp,
        "Cholesterol":chol,
        "FastingBS":fbs,
        "MaxHR":hr,
        "Oldpeak":oldpeak,
        "Sex":sex,
        "ChestPainType":cp,
        "RestingECG":ecg,
        "ExerciseAngina":angina,
        "ST_Slope":slope
    }

    sample = pd.DataFrame([sample])

    sample = pd.get_dummies(sample)

    sample = sample.reindex(columns=columns, fill_value=0)

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("Heart Disease: YES")
    else:
        st.success("Heart Disease: NO")
