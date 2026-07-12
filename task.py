import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
 
df = pd.read_csv("car_price_dataset.csv") 
 
# Basic EDA 
print(df.head()) 
print(df.tail())
print(df.shape)
df.info() 
print(df.describe()) 
print(df.columns)
print(df.dtypes) 
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique()) 
 
# Value Counts print(df["Fuel_Type"].value_counts()) print(df["Brand"].value_counts()) 
 
# Histogram sns.histplot(df["Price"]) plt.xlabel("Price") plt.ylabel("Count") 
 
sns.histplot(df["Mileage"]) 
plt.xlabel("Mileage") 
plt.ylabel("Count") 
 
# Count Plot 
sns.countplot(x=df["Fuel_Type"]) 
plt.xlabel("Fuel Type") 
plt.ylabel("Count") 
 
sns.countplot(x=df["Transmission"]) 
plt.xlabel("Transmission") 
plt.ylabel("Count") 
 
# Scatter Plot 
sns.scatterplot(x="Mileage", y="Price", data=df)
plt.xlabel("Mileage")
plt.ylabel("Price") 
 
# Bar Plot 
sns.barplot(x="Fuel_Type", y="Price", data=df) 
plt.xlabel("Fuel Type") 
plt.ylabel("Price") 
 
# Heatmap 
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm") 
plt.show()
