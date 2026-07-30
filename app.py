#print("Question no 5")
import streamlit as st
import joblib
import numpy as np

# Title
st.title("Multi Model Prediction App")

# Select Problem Type
problem = st.selectbox(
    "Select Problem Type",
    ["Classification", "Regression"]
)

# Load Model According to Problem
if problem == "Classification":

    model = joblib.load("classification_model.pkl")
    scaler = joblib.load("classification_scaler.pkl")
    columns = joblib.load("classification_columns.pkl")

    st.subheader("Titanic Survival Prediction")

else:

    model = joblib.load("regression_model.pkl")
    scaler = joblib.load("regression_scaler.pkl")
    columns = joblib.load("regression_columns.pkl")

    st.subheader("House Price Prediction")

# User Input
values = []

st.write("Enter Feature Values:")

for col in columns:

    value = st.number_input(
        f"{col}",
        value=0.0
    )

    values.append(value)


# Prediction Button
if st.button("Predict"):

    # Convert input into numpy array
    data = np.array(values).reshape(1, -1)

    # Scaling
    data = scaler.transform(data)

    # Prediction
    pred = model.predict(data)


    # Classification Output
    if problem == "Classification":

        if pred[0] == 1:

            st.success(
                "Prediction: Passenger Survived"
            )

        else:

            st.error(
                "Prediction: Passenger Did Not Survive"
            )


    # Regression Output
    else:

        st.success(
            f"Predicted House Value: {pred[0]:.2f}"
        )

