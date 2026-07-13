# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load Dataset

df = pd.read_csv("insurance.csv")

print("First 5 Rows:")
print(df.head())


# 2. Basic Information

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# 3. Missing Values & Duplicates

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)


# 4. Exploratory Data Analysis

# Expense Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['expenses'], kde=True)
plt.title("Distribution of Expenses")
plt.show()


# Smoker vs Expenses
plt.figure(figsize=(8,5))
sns.boxplot(x="smoker", y="expenses", data=df)
plt.title("Smoker Effect on Expenses")
plt.show()


# Age vs Expenses
plt.figure(figsize=(8,5))
sns.scatterplot(x="age", y="expenses", data=df)
plt.title("Age vs Expenses")
plt.show()


# BMI vs Expenses
plt.figure(figsize=(8,5))
sns.scatterplot(x="bmi", y="expenses", data=df)
plt.title("BMI vs Expenses")
plt.show()


# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# Pairplot
sns.pairplot(df)
plt.show()


# 5. Data Preprocessing

# Convert categorical columns into numerical
df = pd.get_dummies(df, drop_first=True)

print("\nAfter Encoding:")
print(df.head())


# 6. Split Data

X = df.drop("expenses", axis=1)
y = df["expenses"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 7. Linear Regression Model

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)


print("\nLinear Regression Results")
print("MAE :", mean_absolute_error(y_test, lr_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, lr_pred)))
print("R2 Score :", r2_score(y_test, lr_pred))



# 8. Decision Tree Model

dt = DecisionTreeRegressor(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)


print("\nDecision Tree Results")
print("MAE :", mean_absolute_error(y_test, dt_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, dt_pred)))
print("R2 Score :", r2_score(y_test, dt_pred))


# 9. Random Forest Model

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)


print("\nRandom Forest Results")
print("MAE :", mean_absolute_error(y_test, rf_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("R2 Score :", r2_score(y_test, rf_pred))


# 10. Model Comparison

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "R2 Score": [
        r2_score(y_test, lr_pred),
        r2_score(y_test, dt_pred),
        r2_score(y_test, rf_pred)
    ]
})

print("\nModel Comparison:")
print(results)


# Visualization
plt.figure(figsize=(8,5))
sns.barplot(
    x="Model",
    y="R2 Score",
    data=results
)

plt.title("Model Performance Comparison")
plt.xticks(rotation=45)
plt.show()


# 11. Feature Importance

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)


plt.figure(figsize=(8,5))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Feature Importance")
plt.show()
