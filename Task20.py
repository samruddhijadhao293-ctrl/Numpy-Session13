print("Question no.1")
import pandas as pd

# Load Dataset
df = pd.read_csv("ford_cleaned.csv")

# Independent and Dependent Features
X = df.drop("price", axis=1)
Y = df["price"]

# Print Shapes
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

print("Question no.2")
# Identify categorical columns
cat_columns = X.select_dtypes(include=["object", "string"]).columns

print("Categorical Columns:")
print(cat_columns)

# One-Hot Encoding
X = pd.get_dummies(X, columns=cat_columns)

# Convert Boolean columns to Integer
X = X.astype(int)

# Display first 5 rows
print(X.head())

print("Question no.3")
from sklearn.preprocessing import StandardScaler

# Numerical Columns
num_columns = ["year", "mileage", "tax", "mpg", "engineSize"]

# Standard Scaling
scaler = StandardScaler()

X[num_columns] = scaler.fit_transform(X[num_columns])

# Display first 5 rows
print(X.head())

print("Question no.4")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.33,
    random_state=42
)

print("X_train Shape:", X_train.shape)
print("X_test Shape :", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape :", y_test.shape)

print("Question no. 5")
from sklearn.linear_model import LinearRegression

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Print Intercept
print("Intercept:")
print(model.intercept_)

# Print Coefficients
print("\nCoefficients:")
print(model.coef_)

print("Question no. 6")
# Predict
y_pred = model.predict(X_test)

print("First 10 Predicted Values:")
print(y_pred[:10])

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

print("Question no.7")
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)

print("Question no. 8")
import joblib

# Save Model
joblib.dump(model, "LR_ford_car.pkl")

# Save Scaler
joblib.dump(scaler, "scaler.pkl")

# Save Feature Columns
joblib.dump(X.columns.tolist(), "columns.pkl")

print("Model and preprocessing files saved successfully.")