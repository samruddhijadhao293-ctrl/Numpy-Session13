import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# CSV load
df = pd.read_csv("houseprice.csv")

# Features aur Target
X = df.drop("Price", axis=1)   # "Price" ko apne target column ke naam se replace karein
y = df["Price"]

# One-hot encoding (agar categorical columns hain)
X = pd.get_dummies(X)

# Column names save karne ke liye
encoded_columns = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model train
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save files
joblib.dump(model, "HousePriceModel.pkl")
joblib.dump(scaler, "HousePriceScaler.pkl")
joblib.dump(encoded_columns, "HousePriceColumns.pkl")

print("All .pkl files created successfully!")