print("Question No. 1")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("insurance.csv")

print("First 10 Rows:")
print(df.head(10))

print("Question No. 2")
print("\nShape of Dataset:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("Question No. 3")
print("\nMissing Values:")
print(df.isnull().sum())

print("Question No. 4")
numerical_columns = df.select_dtypes(include=['number']).columns
categorical_columns = df.select_dtypes(include=['object', 'string']).columns

print("Numerical Columns:")
print(list(numerical_columns))

print("\nCategorical Columns:")
print(list(categorical_columns))

print("Question No. 5")
for column in numerical_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

print("Question No.6")
import matplotlib.pyplot as plt
import seaborn as sns

categorical_columns = ['sex', 'smoker', 'region']

for column in categorical_columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[column])
    plt.title(f"Count Plot of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()

print("Question No. 7")
plt.figure(figsize=(6,5))
sns.heatmap(df[['age','bmi','children','expenses']].corr(),
            annot=True,
            cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()

print("Question No.8")
print("Average Expenses:")
print(df['expenses'].mean())

print("\nMaximum Expenses:")
print(df['expenses'].max())

print("\nMinimum Expenses:")
print(df['expenses'].min())

print("\nAverage Expenses by Smoker:")
print(df.groupby('smoker')['expenses'].mean())

print("Question No. 9")
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot with Smoker
plt.figure(figsize=(6,4))
sns.boxplot(x='smoker', y='expenses', data=df)
plt.title("Expenses by Smoker")
plt.xlabel("Smoker")
plt.ylabel("Expenses")
plt.show()

# Boxplot with Sex
plt.figure(figsize=(6,4))
sns.boxplot(x='sex', y='expenses', data=df)
plt.title("Expenses by Sex")
plt.xlabel("Sex")
plt.ylabel("Expenses")
plt.show()

print("Question No 10")
print("Average Age:", df['age'].mean())
print("Average BMI:", df['bmi'].mean())

print("\nAverage Expenses by Smoker:")
print(df.groupby('smoker')['expenses'].mean())

print("\nRegion with Highest Customers:")
print(df['region'].value_counts())

print("\nSummary:")
print("1. Average age and BMI were calculated.")
print("2. Smokers have higher insurance expenses than non-smokers.")
print("3. The region with the highest number of customers is shown above.")
print("4. Boxplots and histograms show the distribution of the data and highlight outliers.")