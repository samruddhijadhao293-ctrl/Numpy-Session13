print("Question no.1")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=42
)

neighbors = [3, 5, 7, 11, 13, 15]

best_score = 0
best_k = 0

print("KNN Test Accuracy")

for k in neighbors:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    print(f"n_neighbors={k} --> Accuracy={score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print("\nBest n_neighbors =", best_k)
print("Best Accuracy =", best_score)

print("Question no 2")
from sklearn.svm import SVC

C_values = [1, 10, 20]
kernels = ['linear', 'rbf']

best_score = 0
best_params = {}

for c in C_values:
    for k in kernels:
        model = SVC(C=c, kernel=k)
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        print(f"C={c}, Kernel={k} --> Accuracy={score:.4f}")

        if score > best_score:
            best_score = score
            best_params = {"C": c, "Kernel": k}

print("\nBest Parameters:", best_params)
print("Best Accuracy:", best_score)

print("Question no.3")
from sklearn.model_selection import GridSearchCV
import pandas as pd

param_grid = {
    'C': [1, 10, 20],
    'kernel': ['linear', 'rbf']
}

grid = GridSearchCV(
    SVC(),
    param_grid,
    cv=5
)

grid.fit(X_train, y_train)

results = pd.DataFrame(grid.cv_results_)

print(results[['param_C',
               'param_kernel',
               'mean_test_score']])

print("\nBest Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

print("Question no 4")
from sklearn.model_selection import RandomizedSearchCV

random = RandomizedSearchCV(
    SVC(),
    param_distributions=param_grid,
    n_iter=5,
    cv=5,
    random_state=42
)

random.fit(X_train, y_train)

results = pd.DataFrame(random.cv_results_)

print(results[['param_C',
               'param_kernel',
               'mean_test_score']])

print("\nBest Parameters:", random.best_params_)
print("Best Score:", random.best_score_)

print("Question no 5")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Random Forest Accuracy:", accuracy)

print("Question no. 6")
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

ada = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

ada.fit(X_train, y_train)
gb.fit(X_train, y_train)

ada_pred = ada.predict(X_test)
gb_pred = gb.predict(X_test)

print("AdaBoost Accuracy:",
      accuracy_score(y_test, ada_pred))

print("Gradient Boosting Accuracy:",
      accuracy_score(y_test, gb_pred))


print("Question no 7")
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

xgb.fit(X_train, y_train)

pred = xgb.predict(X_test)

print("XGBoost Accuracy:",
      accuracy_score(y_test, pred))

print("Question no.8")
param_grid = {
    'n_estimators': [50,100,150],
    'max_depth': [3,5,7]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5
)

grid_rf.fit(X_train, y_train)

print("Best Parameters:")
print(grid_rf.best_params_)

print("Best Score:")
print(grid_rf.best_score_)

print("Question no 9")
models = {
    "SVM": grid.best_estimator_,
    "Random Forest": rf,
    "AdaBoost": ada,
    "Gradient Boosting": gb,
    "XGBoost": xgb
}

comparison = []

for name, model in models.items():
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    comparison.append([name, acc])

comparison_df = pd.DataFrame(
    comparison,
    columns=["Model", "Accuracy"]
)

print(comparison_df)

best = comparison_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\nBest Model:")
print(best.iloc[0])

print("Question no 10")
import joblib

# Save Best Model
joblib.dump(grid.best_estimator_, "best_model.pkl")

print("Model Saved Successfully!")