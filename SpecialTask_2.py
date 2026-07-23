# Import Libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset

df = pd.read_csv("salary_prediction.csv")

# Select Features

features = [
    'Rating',
    'age',
    'python_yn',
    'R_yn',
    'spark',
    'aws',
    'excel',
    'desc_len',
    'num_comp'
]

X = df[features]
y = df['avg_salary']

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model

model = LinearRegression()

model.fit(X_train_scaled, y_train)

# Prediction

y_pred = model.predict(X_test_scaled)

# Evaluation

print("MAE :", mean_absolute_error(y_test,y_pred))
print("MSE :", mean_squared_error(y_test,y_pred))
print("R2 Score :", r2_score(y_test,y_pred))

# Save Model

joblib.dump(model,"LinearReg_salary.pkl")

joblib.dump(scaler,"scaler_salary.pkl")

joblib.dump(features,"columns.pkl")

print("Model Saved Successfully")


# Import Libraries

import pandas as pd
import streamlit as st
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load Model Files

model = joblib.load("LinearReg_salary.pkl")
scaler = joblib.load("scaler_salary.pkl")
columns = joblib.load("columns.pkl")

# Page Configuration

st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰"
)

# Title

st.title("Glassdoor Salary Prediction")

st.write(
    "This app predicts the average salary based on job skills and company details."
)

# User Inputs

rating = st.number_input(
    "Company Rating",
    min_value=0.0,
    max_value=5.0,
    value=3.5
)

age = st.number_input(
    "Company Age",
    min_value=0,
    max_value=300,
    value=10
)

python_yn = st.selectbox(
    "Python Required",
    ["Yes", "No"]
)

R_yn = st.selectbox(
    "R Required",
    ["Yes", "No"]
)

spark = st.selectbox(
    "Spark Required",
    ["Yes", "No"]
)

aws = st.selectbox(
    "AWS Required",
    ["Yes", "No"]
)

excel = st.selectbox(
    "Excel Required",
    ["Yes", "No"]
)

desc_len = st.number_input(
    "Job Description Length",
    min_value=0,
    value=3000
)

num_comp = st.number_input(
    "Number of Competitors",
    min_value=0,
    value=2
)

# Prediction Button

if st.button("Predict Salary"):

    input_data = pd.DataFrame({

        'Rating':[rating],

        'age':[age],

        'python_yn':[1 if python_yn=="Yes" else 0],

        'R_yn':[1 if R_yn=="Yes" else 0],

        'spark':[1 if spark=="Yes" else 0],

        'aws':[1 if aws=="Yes" else 0],

        'excel':[1 if excel=="Yes" else 0],

        'desc_len':[desc_len],

        'num_comp':[num_comp]

    })

    # Arrange Columns Same As Training

    input_data = input_data.reindex(
        columns=columns,
        fill_value=0
    )

    # Scaling

    input_scaled = scaler.transform(input_data)

    # Prediction

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted Average Salary: ${prediction[0]*1000:,.2f}"
    )



