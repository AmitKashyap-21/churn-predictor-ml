import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib

from preprocess import load_data, clean_data, encode_features, scale_features
from evaluate import plot_roc_curves

# Load data
df = load_data("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocess
df = clean_data(df)
df = encode_features(df)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Train models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1]

    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "model": model,
        "auc": auc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

    print(f"{name} AUC: {auc:.3f}")

best_name = max(results, key=lambda x: results[x]["auc"])
best_model = results[best_name]["model"]

print(f"\nBest model: {best_name}")

joblib.dump(best_model, "../models/churn_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("Model saved ✓")