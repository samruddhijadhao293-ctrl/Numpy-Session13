# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("heart.csv")



print("Question 1")
print(df.head(10))
print(df.shape)
df.info()


print("Question 2")
print(df.isnull().sum())

print("Question 3")
print("Duplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("New Shape:", df.shape)



print("Question 4")
print("Cholesterol = 0 :", (df["Cholesterol"] == 0).sum())
print("RestingBP = 0 :", (df["RestingBP"] == 0).sum())


print("Question 5")
print("Before Cleaning")
print(df[["Cholesterol","RestingBP"]].describe())

chol_mean = df[df["Cholesterol"] != 0]["Cholesterol"].mean()
bp_mean = df[df["RestingBP"] != 0]["RestingBP"].mean()

df["Cholesterol"] = df["Cholesterol"].replace(0, chol_mean)
df["RestingBP"] = df["RestingBP"].replace(0, bp_mean)

df["Cholesterol"] = df["Cholesterol"].round(2)
df["RestingBP"] = df["RestingBP"].round(2)

print("After Cleaning")
print(df[["Cholesterol","RestingBP"]].describe())


print("Question 6")
def plot_hist(column, position):
    plt.subplot(2,2,position)
    sns.histplot(df[column])

plot_hist("Age",1)
plot_hist("RestingBP",2)
plot_hist("Cholesterol",3)
plot_hist("MaxHR",4)

plt.tight_layout()
plt.show()



print("Question 7")
num_cols = df.select_dtypes(include=["int64","float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

print(num_cols)
print(cat_cols)



print("Question 8")
df_encoded = pd.get_dummies(df)

print(df_encoded.shape)
print(df_encoded.head())


print("Question 9")
print(df_encoded.shape)
print(df_encoded.columns)


print("Question 10")
print("1. Invalid values found: Cholesterol = 0 and RestingBP = 0.")
print("2. They were replaced with the mean value.")
print("3. One-Hot Encoding converts categorical data into numerical data.")
print("4. Duplicate rows were removed and data was cleaned before encoding.")

