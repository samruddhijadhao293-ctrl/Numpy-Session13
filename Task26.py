
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


st.set_page_config(page_title="Task24 - KMeans Clustering", layout="wide")

st.title("Task 24 - K-Means Clustering Assignment")

# Q1. Understanding & Dataset Creation

st.header("Q1. Understanding & Dataset Creation")

X, y = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=0.70,
    random_state=42
)

df = pd.DataFrame(X, columns=["Feature1", "Feature2"])

st.subheader("First 10 Rows")

st.dataframe(df.head(10))

st.subheader("Shape of Dataset")

st.write(df.shape)

st.divider()

# Q2. Data Scaling

st.header("Q2. Data Scaling")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=["Feature1", "Feature2"]
)

st.subheader("First 5 Rows of Scaled Data")

st.dataframe(scaled_df.head())

st.divider()

# Q3. Basic K-Means Clustering

st.header("Q3. Basic K-Means Clustering")

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

st.subheader("Updated DataFrame")

st.dataframe(df.head(10))

st.divider()

# Q4. Visualizing Clusters

st.header("Q4. Visualizing Clusters")

fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Feature1",
    y="Feature2",
    hue="Cluster",
    palette="Set1",
    ax=ax
)

ax.set_title("Clusters Formed by KMeans")
ax.set_xlabel("Feature1")
ax.set_ylabel("Feature2")

st.pyplot(fig)

st.success("Q1 to Q4 Completed Successfully.")

# Q5. Elbow Method - Finding Optimal K

st.divider()
st.header("Q5. Elbow Method - Finding Optimal K")

wss = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    wss.append(model.inertia_)

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(range(1,11), wss, marker='o', color='blue')
ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia (WSS)")
ax.grid(True)

st.pyplot(fig)

st.success("Optimal K = 4 (Based on the Elbow Graph)")

# Q6. Final K-Means Model

st.divider()
st.header("Q6. Final K-Means Model")

final_model = KMeans(
    n_clusters=4,
    random_state=42
)

final_clusters = final_model.fit_predict(X_scaled)

df["Final_Cluster"] = final_clusters

st.subheader("Cluster Labels Added")

st.dataframe(df.head(10))

st.subheader("Number of Data Points in Each Cluster")

st.write(df["Final_Cluster"].value_counts())

# Q7. Visualizing Final Clusters

st.divider()
st.header("Q7. Visualizing Final Clusters")

fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Feature1",
    y="Feature2",
    hue="Final_Cluster",
    palette="tab10",
    s=60,
    ax=ax
)

centers = scaler.inverse_transform(final_model.cluster_centers_)

ax.scatter(
    centers[:,0],
    centers[:,1],
    color="black",
    marker="X",
    s=250,
    label="Centroids"
)

ax.set_title("K-Means Clustering Results")
ax.set_xlabel("Feature1")
ax.set_ylabel("Feature2")
ax.legend()

st.pyplot(fig)

st.success("Q5 to Q7 Completed Successfully.")

from sklearn.datasets import load_iris

# Q8. Real Dataset Application

st.divider()
st.header("Q8. Real Dataset Application")

iris = load_iris()

iris_df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

st.subheader("Original Iris Dataset")
st.dataframe(iris_df.head())

# Select Numerical Features
X = iris_df.select_dtypes(include="number")

# Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

iris_df["Cluster"] = kmeans.fit_predict(X_scaled)

st.subheader("Clustered Iris Dataset")
st.dataframe(iris_df.head())

st.subheader("Cluster Counts")
st.write(iris_df["Cluster"].value_counts())

# Q9. Elbow Method on Real Dataset

st.divider()
st.header("Q9. Elbow Method on Real Dataset")

wss = []

for k in range(1,11):
    model = KMeans(
        n_clusters=k,
        random_state=42
    )
    model.fit(X_scaled)
    wss.append(model.inertia_)

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(
    range(1,11),
    wss,
    marker="o",
    color="red"
)

ax.set_title("Elbow Method - Iris Dataset")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")
ax.grid(True)

st.pyplot(fig)

st.success("Optimal K = 3")

# Final Model
final_model = KMeans(
    n_clusters=3,
    random_state=42
)

iris_df["Final_Cluster"] = final_model.fit_predict(X_scaled)

st.subheader("Final Cluster Sizes")

st.write(iris_df["Final_Cluster"].value_counts())

fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(
    x=iris_df.iloc[:,0],
    y=iris_df.iloc[:,1],
    hue=iris_df["Final_Cluster"],
    palette="Set2",
    s=70,
    ax=ax
)

ax.set_xlabel(iris_df.columns[0])
ax.set_ylabel(iris_df.columns[1])
ax.set_title("Final Iris Dataset Clusters")

st.pyplot(fig)

# Q10. Mini Project

st.divider()
st.header("Q10. Mini Project - Complete Clustering Pipeline")

st.subheader("Step 1 : Load Dataset")
st.dataframe(iris_df.head())

st.subheader("Step 2 : Dataset Information")
st.write(iris_df.describe())

st.subheader("Step 3 : Elbow Method Result")
st.write("Optimal Number of Clusters = 3")

st.subheader("Step 4 : Final Cluster Sizes")
st.write(iris_df["Final_Cluster"].value_counts())

st.subheader("Step 5 : Final Visualization")

fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(
    x=iris_df.iloc[:,0],
    y=iris_df.iloc[:,1],
    hue=iris_df["Final_Cluster"],
    palette="tab10",
    s=80,
    ax=ax
)

centers = scaler.inverse_transform(final_model.cluster_centers_)

ax.scatter(
    centers[:,0],
    centers[:,1],
    marker="X",
    color="black",
    s=250,
    label="Centroids"
)

ax.set_title("K-Means Clustering Results")
ax.set_xlabel(iris_df.columns[0])
ax.set_ylabel(iris_df.columns[1])
ax.legend()

st.pyplot(fig)

st.success("🎉 Task 24 Completed Successfully!")

