Customer Churn Predictor

A end-to-end ML project that predicts which customers are likely to cancel their subscription, before they actually do.

Built as part of an 8-week ML portfolio challenge.

---

## What this project does

Takes raw telecom customer data (tenure, charges, contract type, etc.) and trains a classifier to flag customers at risk of churning. The final model comes with SHAP explainability so you can actually see *why* it flagged someone — not just that it did.

---

## Project Structure
```
week01-churn-predictor/
├── data/
│   ├── raw/          # original CSV, never modified
│   └── processed/    # cleaned version, ready for modeling
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
├── models/           # saved .pkl model
├── outputs/
│   ├── figures/      # plots and charts
│   └── reports/      # metrics summary
├── src/
│   ├── preprocess.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

---

## Dataset

**Telco Customer Churn** — IBM sample dataset, ~7,000 customers, 21 features.

Download from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Drop the CSV into `data/raw/` before running anything.

---

## How to run it

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Launch Jupyter**
```bash
jupyter notebook
```

**3. Run notebooks in order**
```
01_eda.ipynb              → explore the data
02_preprocessing.ipynb    → clean and encode
03_modeling.ipynb         → train and evaluate
04_explainability.ipynb   → SHAP analysis
```

---

## Results

| Model | AUC Score |
|---|---|
| Logistic Regression | ~0.84 |
| Random Forest | ~0.86 |
| Gradient Boosting | ~0.87 |

*(Your actual numbers will show up after running notebook 03)*

---

## Key findings

- Customers on month-to-month contracts churn the most
- Higher monthly charges = higher churn risk
- New customers (low tenure) are more likely to leave than long-term ones

---

## Stack

Python · Pandas · Scikit-learn · SHAP · Matplotlib · Seaborn · Jupyter

---

## Part of

8-Week ML Portfolio Challenge
Week 01 of 08 — Beginner Friendly · Score: 7/10
```

---

## `requirements.txt`
```
# Core
pandas==2.1.0
numpy==1.24.0

# Machine Learning
scikit-learn==1.3.0
imbalanced-learn==0.11.0

# Explainability
shap==0.43.0

# Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Notebook
jupyter==1.0.0
ipykernel==6.25.0

# Utilities
joblib==1.3.2
warnings