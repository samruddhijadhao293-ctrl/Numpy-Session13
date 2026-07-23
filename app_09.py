#print("Question no.9")
import streamlit as st

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

