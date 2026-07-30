import streamlit as st
import joblib


model = joblib.load(
    "diabetes_model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)



st.title("Diabetes Prediction System")


preg = st.number_input("Pregnancies")

glu = st.number_input("Glucose")

bp = st.number_input("Blood Pressure")

skin = st.number_input("Skin Thickness")

ins = st.number_input("Insulin")

bmi = st.number_input("BMI")

dpf = st.number_input("Diabetes Pedigree Function")

age = st.number_input("Age")



if st.button("Predict"):


    data = [[
        preg,
        glu,
        bp,
        skin,
        ins,
        bmi,
        dpf,
        age
    ]]


    data_scaled = scaler.transform(data)


    prediction = model.predict(
        data_scaled
    )


    if prediction[0] == 1:

        st.error(
            "Person has Diabetes"
        )


    else:

        st.success(
            "Person does not have Diabetes"
        )