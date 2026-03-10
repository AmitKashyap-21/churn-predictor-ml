import joblib
import pandas as pd

model = joblib.load("../models/churn_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

def predict_customer(data):
    df = pd.DataFrame([data])

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return prediction, probability