import numpy as np
import pandas as pd

df = pd.read_csv("Titanic.csv")


print("First 5 Rows:")
print(df.head())

print("Last 5 Rows:")
print(df.tail())

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nInfo:")
df.info()

print("\nSummary:")
print(df.describe())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nUnique Values:")
print(df.nunique())

print("\nRandom 5 Rows:")
print(df.sample(5))

print("\nValue Counts:")
print(df["Pclass"].value_counts())

