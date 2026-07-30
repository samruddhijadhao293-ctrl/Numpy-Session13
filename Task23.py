# Q1 Linear Regression

import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset

data = fetch_california_housing()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

# Train Test Split

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


# Linear Regression Model

lr = LinearRegression()

lr.fit(
    X_train,
    y_train
)

# Prediction

prediction = lr.predict(
    X_test
)

# Evaluation

score = r2_score(
    y_test,
    prediction
)

import streamlit as st

st.title("Linear Regression")

st.write("## R² Score")
st.write(score)

st.write("## First 10 Predictions")
st.write(prediction[:10])

# Q2 Logistic Regression

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


data = load_breast_cancer()

X = pd.DataFrame(
data.data,
columns=data.feature_names
)

y = data.target


X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)


model=LogisticRegression()

model.fit(
X_train,
y_train
)

y_pred=model.predict(
X_test
)

import streamlit as st

st.title("Logistic Regression Results")

st.write("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))
st.write("F1 Score:", f1_score(y_test, y_pred))




#Question 3 KKN  Classification


import pandas as pd
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Load Dataset
data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target


# Train Test Split
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


# KNN

accuracy = []

for k in [3,5,7]:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(
        X_train,
        y_train
    )

    pred = knn.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        pred
    )

    accuracy.append(
        [k, acc]
    )


result = pd.DataFrame(
    accuracy,
    columns=[
        "K Value",
        "Accuracy"
    ]
)


# Streamlit Output

st.title("KNN Classification")

st.dataframe(result)


best_k = result.loc[
    result["Accuracy"].idxmax()
]


st.write("Best K:")
st.write(best_k)


#Question 4 Native Bayes
import pandas as pd
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix


# Load Dataset

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target


# Train Test Split

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


# Naive Bayes Model

nb = GaussianNB()

nb.fit(
    X_train,
    y_train
)


# Prediction

nb_pred = nb.predict(
    X_test
)


# Streamlit Output

st.title("Naive Bayes Classification")


st.subheader("Classification Report")

st.text(
    classification_report(
        y_test,
        nb_pred
    )
)


st.subheader("Confusion Matrix")

st.write(
    confusion_matrix(
        y_test,
        nb_pred
    )
)

#Question5  Comprison

import pandas as pd
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Load Dataset

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target


# Train Test Split

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



# Models

models = {

    "Logistic Regression":
    LogisticRegression(),

    "KNN":
    KNeighborsClassifier(n_neighbors=5),

    "Naive Bayes":
    GaussianNB()

}



results = []


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )


    pred = model.predict(
        X_test
    )


    results.append(
        [
            name,
            accuracy_score(y_test, pred),
            precision_score(y_test, pred),
            recall_score(y_test, pred),
            f1_score(y_test, pred)
        ]
    )



comparison = pd.DataFrame(
    results,
    columns=[
        "Algorithm",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)


# Streamlit Output

st.title("Model Comparison")

st.dataframe(comparison)
