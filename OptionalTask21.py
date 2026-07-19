# Setup and Libraries

# Streamlit is used to create the web application.
import streamlit as st

# Pandas is used to create and manipulate DataFrames.
import pandas as pd

# Joblib is used to load the trained model and preprocessing files.
import joblib

# Loading Model and Preprocessing Objects
model = joblib.load("HousePriceModel.pkl")
scaler = joblib.load("HousePriceScaler.pkl")
encoded_columns = joblib.load("HousePriceColumns.pkl")

# Page Configuration
# Set page title and centered layout for a clean interface.
st.set_page_config(
    page_title="House Price Prediction",
    layout="centered"
)

# Title and Description
st.title("House Price Prediction")
st.write("Enter the house details below to predict the selling price.")

# Numerical Input Fields
OverallQual = st.number_input(
    "Overall Quality",
    min_value=1,
    max_value=10,
    value=5
)

GrLivArea = st.number_input(
    "Ground Living Area",
    min_value=100,
    max_value=6000,
    value=1500
)

GarageArea = st.number_input(
    "Garage Area",
    min_value=0,
    max_value=1500,
    value=500
)

TotalBsmtSF = st.number_input(
    "Basement Area",
    min_value=0,
    max_value=7000,
    value=800
)

LotArea = st.number_input(
    "Lot Area",
    min_value=1000,
    max_value=250000,
    value=9000
)

# Categorical Input using Dropdowns
# Selectbox allows users to select only valid options.
Neighborhood = st.selectbox(
    "Neighborhood",
    ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"]
)

# Selectbox reduces input errors for house style.
HouseStyle = st.selectbox(
    "House Style",
    ["1Story", "2Story", "1.5Fin"]
)

# Predict Button
predict = st.button("Predict House Price")

# DataFrame Creation, Encoding, Scaling and Prediction
if predict:

    try:

        input_data = pd.DataFrame({
            "OverallQual": [OverallQual],
            "GrLivArea": [GrLivArea],
            "GarageArea": [GarageArea],
            "TotalBsmtSF": [TotalBsmtSF],
            "LotArea": [LotArea],
            "Neighborhood": [Neighborhood],
            "HouseStyle": [HouseStyle]
        })

        input_encoded = pd.get_dummies(input_data)

        input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)

        numerical_columns = [
            "OverallQual",
            "GrLivArea",
            "GarageArea",
            "TotalBsmtSF",
            "LotArea"
        ]

        input_encoded[numerical_columns] = scaler.transform(
            input_encoded[numerical_columns]
        )

        prediction = model.predict(input_encoded)

        st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")

