# Import Libraries

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score


# Load Dataset

df = pd.read_csv("diabetes.csv")

print(df.head())


# Split Data

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# SVM GridSearch

parameter = {
    "C":[1,10,20],
    "kernel":["linear","rbf"]
}


grid = GridSearchCV(
    SVC(),
    parameter,
    cv=5
)


grid.fit(X_train,y_train)


svm_model = grid.best_estimator_


print("Best SVM Parameters:")
print(grid.best_params_)


# Other Models

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


ada = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)


gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)


xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="logloss"
)


# Model Comparison

models = {

    "SVM": svm_model,

    "Random Forest": rf,

    "AdaBoost": ada,

    "Gradient Boosting": gb,

    "XGBoost": xgb
}



result=[]


for name, model in models.items():

    model.fit(X_train,y_train)

    prediction = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        prediction
    )

    result.append(
        [name,acc]
    )



comparison = pd.DataFrame(
    result,
    columns=["Model","Accuracy"]
)


print("\nModel Comparison")
print(comparison)


# Save Best Model

joblib.dump(
    svm_model,
    "diabetes_model.pkl"
)


joblib.dump(
    scaler,
    "scaler.pkl"
)


print("\nModel Saved Successfully")