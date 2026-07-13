# ============================
# Import Libraries
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("ford_car_dataset.csv")

print("Question no.1")
# First 10 rows
print("First 10 Rows")
print(df.head(10))

# Last 5 rows
print("\nLast 5 Rows")
print(df.tail())

# Shape
print("\nShape of Dataset")
print(df.shape)

# Data Types
print("\nData Types")
print(df.dtypes)

# Dataset Information
print("\nDataset Info")
print(df.info())

# Comments:
# 1. Dataset contains both numerical and categorical columns.
# 2. Price is the target variable.
# 3. Model, fuelType, transmission are categorical features.


print("Question no.2")

# Missing Values
print("Missing Values")
print(df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove Duplicates
df = df.drop_duplicates()

print("\nShape after removing duplicates")
print(df.shape)

# Comments:
# Missing values checked using isnull().
# Duplicate rows removed using drop_duplicates().

print("Question no.3")
# Statistical Summary
print(df.describe())

# Price Statistics
print("\nPrice")
print("Minimum:", df["price"].min())
print("Maximum:", df["price"].max())
print("Mean:", df["price"].mean())
print("Median:", df["price"].median())

# Mileage Statistics
print("\nMileage")
print("Minimum:", df["mileage"].min())
print("Maximum:", df["mileage"].max())
print("Mean:", df["mileage"].mean())
print("Median:", df["mileage"].median())

# Year Statistics
print("\nYear")
print("Minimum:", df["year"].min())
print("Maximum:", df["year"].max())
print("Mean:", df["year"].mean())
print("Median:", df["year"].median())

print("Question no.4")
numeric_cols = ['price', 'mileage', 'year', 'engineSize', 'mpg']

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()

# Comments:
# Observe skewness, spread, outliers and distribution.

print("Question no.5")
# Categorical Columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns

for col in cat_cols:
    plt.figure(figsize=(10,5))
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=90)
    plt.title(f"Count Plot of {col}")
    plt.show()

# Comments:
# Most common fuel type, transmission and model can be identified.

print("Question no.6")
plt.figure(figsize=(10,8))

corr = df.corr(numeric_only=True)

sns.heatmap(corr,
            annot=True,
            cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()

# Correlation with Price
print(corr["price"].sort_values(ascending=False))

# Comments:
# Check which variables are positively or negatively correlated with price.


print("Question no.7")
# Independent Features
X = df.drop("price", axis=1)

# Dependent Feature
y = df["price"]

print("Independent Features")
print(X.columns)

print("\nDependent Feature")
print(y.name)

# Comments:
# price is target variable because we predict car price.
# Remaining columns are input features.

print("Question no.8")
print("Before Encoding")
print(df[['model','fuelType']].head())

encoded_df = pd.get_dummies(df, drop_first=True)

print("\nAfter Encoding")
print(encoded_df.head())

print("\nNew Shape")
print(encoded_df.shape)

# Comments:
# One Hot Encoding converts categorical variables into numeric columns.

print("Question no.9")
encoded_df = pd.get_dummies(df, drop_first=True)

X = encoded_df.drop("price", axis=1)
y = encoded_df["price"]

num_cols = ['year','mileage','tax','mpg','engineSize']

scaler = StandardScaler()

X[num_cols] = scaler.fit_transform(X[num_cols])

print("Scaled Data")
print(X.head())

print("Question no. 10")
# =========================
# Step 1: Load Dataset
# =========================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ford_car_dataset.csv")

# =========================
# Step 2: Data Cleaning
# =========================

print(df.isnull().sum())

df.drop_duplicates(inplace=True)

# =========================
# Step 3: EDA
# =========================

# Histograms

num_cols = ['price','mileage','year','engineSize','mpg']

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.show()

# Count Plots

cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    plt.figure(figsize=(10,5))
    sns.countplot(x=col,data=df)
    plt.xticks(rotation=90)
    plt.show()

# Heatmap

plt.figure(figsize=(10,8))

sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap='coolwarm')

plt.show()

# =========================
# Step 4: Feature Selection
# =========================

X = df.drop("price",axis=1)
y = df["price"]

# =========================
# Step 5: Encoding
# =========================

X = pd.get_dummies(X,drop_first=True)

# =========================
# Step 6: Scaling
# =========================

scaler = StandardScaler()

numeric_columns = ['year','mileage','tax','mpg','engineSize']

X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

print("Final Processed Data")
print(X.head())
print(y.head())

from sklearn.preprocessing import StandardScaler

# Independent Features
X = df.drop("price", axis=1)

# One Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Numeric Columns
numeric_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

# Standard Scaling
scaler = StandardScaler()

X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print("First 5 Rows of Scaled Data")
print(X.head())