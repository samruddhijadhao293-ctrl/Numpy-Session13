print("Question no.1")
import pandas as pd

# Load Dataset
df = pd.read_csv("heart.csv")

# Features and Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# One Hot Encoding
X = pd.get_dummies(X, drop_first=True)

print("X Shape:", X.shape)
print("y Shape:", y.shape)

print("Question no.2")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


print("Question no.3")
from sklearn.linear_model import LogisticRegression

# Create Logistic Regression Model
model = LogisticRegression(
    max_iter=2000,
    solver="liblinear",
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

print("Model Trained Successfully")

print("Question no.4")
y_pred = model.predict(X_test)

print("Actual Values:")
print(y_test.head(10).values)

print()

print("Predicted Values:")
print(y_pred[:10])


print("Question no.5")
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("\nTN =", TN)
print("FP =", FP)
print("FN =", FN)
print("TP =", TP)


print("Question no.6")
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))


print("Question no.7")
import joblib

# Save Model
joblib.dump(model, "heart_model.pkl")

# Save Column Names
joblib.dump(X.columns.tolist(), "columns.pkl")

print("Files Saved Successfully")

print("Question no.8")
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")

sample = {
    "Age":52,
    "RestingBP":130,
    "Cholesterol":250,
    "FastingBS":0,
    "MaxHR":150,
    "Oldpeak":1.2,
    "Sex":"M",
    "ChestPainType":"ATA",
    "RestingECG":"Normal",
    "ExerciseAngina":"N",
    "ST_Slope":"Up"
}

sample_df = pd.DataFrame([sample])

sample_df = pd.get_dummies(sample_df)

sample_df = sample_df.reindex(columns=columns, fill_value=0)

prediction = model.predict(sample_df)

if prediction[0] == 1:
    print("Heart Disease: YES")
else:
    print("Heart Disease: NO")