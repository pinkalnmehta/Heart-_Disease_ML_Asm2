import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

os.makedirs("model", exist_ok=True)

# LOAD DATA
df = pd.read_csv("heart_disease_uci.csv")

# Rename target column
if "num" in df.columns:
    df = df.rename(columns={"num": "target"})

# Convert target values >0 to 1
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

# REMOVE MISSING VALUES  ⭐ IMPORTANT FIX
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

# Convert categorical → numeric
df = pd.get_dummies(df)

# Save columns
joblib.dump(df.columns, "model/columns.pkl")

# Split features
X = df.drop("target", axis=1)
y = df["target"]

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "model/scaler.pkl")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "bayes": GaussianNB(),
    "forest": RandomForestClassifier(),
    "xgb": XGBClassifier(eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}.pkl")

    preds = model.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
