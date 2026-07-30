print("Question no.1")
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("Titanic.csv")

df.head()
print(df.head())

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Fare"] = df["Fare"].fillna(df["Fare"].median())

df.fillna({
    "Age": df["Age"].median(),
    "Embarked": df["Embarked"].mode()[0],
    "Fare": df["Fare"].median()
}, inplace=True)


df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

print(df.isnull().sum())

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

X_class = df.drop("Survived", axis=1)
y_class = df["Survived"]

scaler_class = StandardScaler()

X_class = scaler_class.fit_transform(X_class)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42
)

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)

df_reg = housing.frame

df_reg.head()

X_reg = df_reg.drop("MedHouseVal", axis=1)
y_reg = df_reg["MedHouseVal"]

scaler_reg = StandardScaler()

X_reg = scaler_reg.fit_transform(X_reg)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)


print("Question no.2")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

models = {

    "Logistic Regression": LogisticRegression(),

    "Decision Tree": DecisionTreeClassifier(),

    "SVM": SVC(),

    "KNN": KNeighborsClassifier(),

    "Naive Bayes": GaussianNB()
}

results = []

for name, model in models.items():

    model.fit(X_train_c, y_train_c)

    pred = model.predict(X_test_c)

    acc = accuracy_score(y_test_c, pred)

    print("="*60)
    print(name)

    print("Accuracy:", acc)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test_c, pred))

    print("\nClassification Report")
    print(classification_report(y_test_c, pred))

    results.append([name, acc])

    comparison = pd.DataFrame(results,
                          columns=["Model","Accuracy"])

comparison.sort_values("Accuracy", ascending=False)


print("Question no.3")
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score

reg_models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(),

    "SVR": SVR(),

    "KNN": KNeighborsRegressor()
}

reg_results = []

for name, model in reg_models.items():

    model.fit(X_train_r, y_train_r)

    pred = model.predict(X_test_r)

    score = r2_score(y_test_r, pred)

    print("="*60)
    print(name)

    print("R2 Score:", score)

    reg_results.append([name, score])

    reg_comparison = pd.DataFrame(
    reg_results,
    columns=["Model","R2 Score"]
)

reg_comparison.sort_values(
    "R2 Score",
    ascending=False
)


#print("Question no.4")
best_class_model = LogisticRegression()

best_class_model.fit(X_class, y_class)

best_reg_model = LinearRegression()

best_reg_model.fit(X_reg, y_reg)

import joblib

joblib.dump(best_class_model,"classification_model.pkl")

joblib.dump(best_reg_model,"regression_model.pkl")

joblib.dump(scaler_class,"classification_scaler.pkl")

joblib.dump(scaler_reg,"regression_scaler.pkl")

joblib.dump(df.drop("Survived",axis=1).columns,
            "classification_columns.pkl")

joblib.dump(df_reg.drop("MedHouseVal",axis=1).columns,
            "regression_columns.pkl")

