print("Question no. 2")
import pandas as pd

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# First 10 Rows
print(df.head(10))

print("Question no 3")
# Missing Values
print(df.isnull().sum())

# Missing Percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

# Fill Numeric Missing Values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill Categorical Missing Values
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Duplicate Rows
print("Duplicate Rows:", df.duplicated().sum())

# Remove Duplicates
df = df.drop_duplicates()

print("New Shape:", df.shape)


print("Question no.4")
print(df.describe())

# Target Variable
print("Minimum Price:", df["Price"].min())
print("Maximum Price:", df["Price"].max())
print("Mean Price:", df["Price"].mean())
print("Median Price:", df["Price"].median())

print("Question no.5")
import matplotlib.pyplot as plt

numeric_columns = df.select_dtypes(include=['int64','float64']).columns

df[numeric_columns].hist(figsize=(15,10), bins=20)

plt.tight_layout()
plt.show()

print("Question no.6")
import seaborn as sns
import matplotlib.pyplot as plt

cat_columns = df.select_dtypes(include=["object", "string"]).columns

for col in cat_columns:
    plt.figure(figsize=(7,5))
    sns.countplot(data=df,x=col)
    plt.xticks(rotation=45)
    plt.title(col)
    plt.show()


print("Question no.7")
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")

plt.show()

print("Question no.8")
X = df.drop("Price", axis=1)

y = df["Price"]

print("Independent Features:")
print(X.columns)

print("\nDependent Feature:")
print(y.name)

print("Question no.9")
print("Before Encoding")

categorical_df = df.select_dtypes(include=["object", "string"])
print(categorical_df.head())

encoded_df = pd.get_dummies(df, drop_first=True)

print("\nAfter Encoding")

print(encoded_df.head())


print("Question no.10")
from sklearn.preprocessing import StandardScaler

X = encoded_df.drop("Price", axis=1)

y = encoded_df["Price"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print(scaled_df.head())
