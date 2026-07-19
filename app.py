# Q1. Setup and Libraries
# Streamlit is used to create the web application.
import streamlit as st

# Pandas is used to create and manipulate DataFrames.
import pandas as pd

# Joblib is used to load the trained model and preprocessing files.
import joblib


# Q2. Loading Model and Preprocessing Objects
model = joblib.load("LR_ford_car.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")


# Q3. Page Configuration
# Set page title and centered layout for a clean interface.
st.set_page_config(
    page_title="Ford Car Price Prediction",
    layout="centered"
)


# Q4. Title and Description
st.title("Ford Car Price Prediction")
st.write("Enter the car details below to predict its selling price.")


# Q5. Numerical Input Fields
year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2018)

mileage = st.number_input("Mileage", min_value=0, max_value=300000, value=50000)

tax = st.number_input("Road Tax", min_value=0, max_value=1000, value=150)

mpg = st.number_input("MPG", min_value=0.0, max_value=150.0, value=50.0)

engineSize = st.number_input("Engine Size", min_value=0.0, max_value=10.0, value=1.5)


# Q6. Categorical Input using Dropdowns
# Selectbox allows users to select only valid options.
transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic", "Semi-Auto"]
)

# Selectbox reduces input errors for fuel type.
fuelType = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
)


# Q7. Text Input and Predict Button
model_name = st.text_input("Car Model")
predict = st.button("Predict Price")


# Q8 and Q9. DataFrame Creation, Encoding, Scaling and Prediction
if predict:
    try:
        input_data = pd.DataFrame({
            "model": [model_name],
            "year": [year],
            "transmission": [transmission],
            "mileage": [mileage],
            "fuelType": [fuelType],
            "tax": [tax],
            "mpg": [mpg],
            "engineSize": [engineSize]
        })

        input_encoded = pd.get_dummies(input_data)

        input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)

        numerical_columns = ["year", "mileage", "tax", "mpg", "engineSize"]

        input_encoded[numerical_columns] = scaler.transform(
            input_encoded[numerical_columns]
        )

        prediction = model.predict(input_encoded)

        st.success(f"Predicted Car Price: £ {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
