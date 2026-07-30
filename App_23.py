import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ------------------- PAGE CONFIG -------------------

st.set_page_config(
    page_title="Machine Learning Models",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Models Dashboard")
st.markdown("Select a model from the sidebar.")

# ------------------- SIDEBAR -------------------

option = st.sidebar.selectbox(
    "Choose Model",
    [
        "Linear Regression",
        "Logistic Regression",
        "KNN",
        "Naive Bayes",
        "Model Comparison"
    ]
)



# Linear Regression

if option == "Linear Regression":

    data = fetch_california_housing()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    score = r2_score(y_test,pred)

    st.header("📈 Linear Regression")

    c1,c2 = st.columns(2)

    c1.metric("R² Score", round(score,4))

    c2.metric("Predictions", len(pred))

    st.subheader("First 10 Predictions")

    st.dataframe(pd.DataFrame({
        "Prediction":pred[:10]
    }))


# Logistic Regression
elif option=="Logistic Regression":

    data = load_breast_cancer()

    X=data.data
    y=data.target

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    scaler=StandardScaler()

    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    model=LogisticRegression()

    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    st.header("📊 Logistic Regression")

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Accuracy",round(accuracy_score(y_test,pred),4))
    c2.metric("Precision",round(precision_score(y_test,pred),4))
    c3.metric("Recall",round(recall_score(y_test,pred),4))
    c4.metric("F1 Score",round(f1_score(y_test,pred),4))

    st.subheader("Confusion Matrix")

    st.dataframe(confusion_matrix(y_test,pred))

# KNN
elif option=="KNN":

    data=load_breast_cancer()

    X=data.data
    y=data.target

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    scaler=StandardScaler()

    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    result=[]

    for k in [3,5,7]:

        model=KNeighborsClassifier(n_neighbors=k)

        model.fit(X_train,y_train)

        pred=model.predict(X_test)

        result.append([k,accuracy_score(y_test,pred)])

    df=pd.DataFrame(result,columns=["K","Accuracy"])

    st.header("🎯 KNN Classification")

    st.dataframe(df)

    st.subheader("Accuracy Graph")

    st.line_chart(df.set_index("K"))

# Naive Bayes
elif option=="Naive Bayes":

    data=load_breast_cancer()

    X=data.data
    y=data.target

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    scaler=StandardScaler()

    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    model=GaussianNB()

    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    st.header("📚 Naive Bayes")

    st.subheader("Classification Report")

    st.text(classification_report(y_test,pred))

    st.subheader("Confusion Matrix")

    st.dataframe(confusion_matrix(y_test,pred))

# Model Comparison
else:

    data=load_breast_cancer()

    X=data.data
    y=data.target

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    scaler=StandardScaler()

    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    models={

        "Logistic Regression":LogisticRegression(),

        "KNN":KNeighborsClassifier(n_neighbors=5),

        "Naive Bayes":GaussianNB()

    }

    result=[]

    for name,model in models.items():

        model.fit(X_train,y_train)

        pred=model.predict(X_test)

        result.append([

            name,

            accuracy_score(y_test,pred),

            precision_score(y_test,pred),

            recall_score(y_test,pred),

            f1_score(y_test,pred)

        ])

    df=pd.DataFrame(result,columns=[
        "Algorithm",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ])

    st.header("🏆 Model Comparison")

    st.dataframe(df)

    st.subheader("Accuracy Comparison")

    st.bar_chart(df.set_index("Algorithm")["Accuracy"])

    st.subheader("Performance Metrics")

    st.bar_chart(df.set_index("Algorithm"))
