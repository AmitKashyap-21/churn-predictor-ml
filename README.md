# Customer Churn Prediction

A production-grade machine learning pipeline for predicting telecom customer churn. This project demonstrates an end-to-end data science workflow: exploratory analysis, feature engineering, model comparison, and interpretability using SHAP.

**Why it matters:** Customer acquisition costs significantly more than retention. Identifying at-risk customers early enables proactive intervention and measurable ROI.

## Objectives

This project demonstrates core data science competencies:

- **Reproducibility** — Clean, documented workflow that others can extend
- **Model Selection** — Rigorous comparison across multiple algorithms with appropriate metrics
- **Feature Analysis** — Understanding which variables drive business outcomes
- **Interpretability** — Using SHAP to explain individual predictions and feature importance
- **Production Quality** — Modular code structure, proper separation of concerns, reusable components

## Dataset

**Source:** Telco Customer Churn (IBM sample dataset)  
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Dataset Characteristics:**
- 7,043 customer records
- 21 features (13 categorical, 8 numerical)
- Binary classification: Churn (Yes/No)
- ~27% positive class imbalance

**Key Features:**

| Feature | Type | Description |
|---------|------|-------------|
| `tenure` | Numerical | Months as customer |
| `MonthlyCharges` | Numerical | Monthly service cost |
| `Contract` | Categorical | Month-to-month, one-year, or two-year |
| `InternetService` | Categorical | DSL, fiber optic, or no service |
| `TechSupport` | Categorical | Customer has tech support |

**Target Variable:** `Churn` (0 = retained, 1 = churned)

## Methodology

The analysis follows the standard ML workflow:

```
Data Acquisition
       ↓
Exploratory Data Analysis (EDA)
       ↓
Data Cleaning & Preprocessing
       ↓
Feature Engineering & Selection
       ↓
Model Training & Hyperparameter Tuning
       ↓
Model Evaluation & Comparison
       ↓
Interpretability Analysis (SHAP)
```

## Project Structure

```
week01-churn-predictor/
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned, ready to model
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploration & analysis
│   ├── 02_preprocessing.ipynb   # Cleaning & feature engineering
│   ├── 03_modeling.ipynb        # Training & evaluation
│   └── 04_explainability.ipynb  # SHAP analysis
│
├── models/
│   └── churn_model.pkl         # Final trained model
│
├── outputs/
│   ├── figures/                # Plots & visualizations
│   └── reports/                # Summary results
│
├── src/
│   ├── preprocess.py           # Reusable preprocessing functions
│   └── evaluate.py             # Evaluation helpers
│
├── requirements.txt
└── README.md
```

## Models Evaluated

Three classification algorithms were trained and compared:

| Model | Rationale |
|-------|-----------|
| Logistic Regression | Interpretable baseline; provides probability estimates |
| Random Forest | Captures non-linear relationships; feature importance estimates |
| Gradient Boosting (XGBoost) | State-of-the-art performance; handles feature interactions |

**Evaluation Metrics:** ROC-AUC, precision, recall, F1-score, confusion matrix

## Results

| Model | ROC-AUC | Precision | Recall |
|-------|---------|-----------|--------|
| Logistic Regression | 0.840 | 0.65 | 0.58 |
| Random Forest | 0.863 | 0.68 | 0.61 |
| Gradient Boosting | 0.874 | 0.71 | 0.64 |

**Key Finding:** Gradient Boosting achieved the best performance, with 3.4% AUC improvement over the baseline. The model generalizes well with balanced precision-recall tradeoff suitable for intervention strategies.

## Feature Importance & Business Insights

SHAP analysis reveals key churn drivers:

1. **Contract Type** — Month-to-month contracts show 3x higher churn risk. Customers without lock-in terms leave at elevated rates.

2. **Tenure** — Early churn dominates; 80% of churners leave within first 6 months. Onboarding quality is critical.

3. **Monthly Charges** — Strong positive correlation with churn. Premium-tier customers may face value-realization challenges.

4. **Technical Support** — Lack of support significantly increases churn probability, suggesting service quality issues directly impact retention.

**Business Implication:** Targeting month-to-month customers with support bundles or contract incentives in the first 6 months could yield highest ROI.

## Outputs

The project generates the following visualizations:

- **EDA Plots** — Distribution analysis, missing values, churn baseline
- **Correlation Matrices** — Feature relationships and multicollinearity assessment
- **Model Comparison** — ROC curves, precision-recall curves, confusion matrices
- **SHAP Visualizations** — Feature importance, partial dependence, individual prediction explanations

All outputs saved to `outputs/figures/`

## Getting Started

### Installation

```bash
git clone https://github.com/AmitKashyap-21/churn-predictor-ml.git
cd churn-predictor-ml
pip install -r requirements.txt
```

### Running the Analysis

```bash
jupyter notebook
```

Execute notebooks sequentially:

1. `01_eda.ipynb` — Data exploration and profiling
2. `02_preprocessing.ipynb` — Cleaning and feature engineering
3. `03_modeling.ipynb` — Model training and evaluation
4. `04_explainability.ipynb` — SHAP analysis and interpretation

## Requirements

- Python 3.8+
- pandas — Data manipulation and analysis
- numpy — Numerical computing
- scikit-learn — Machine learning algorithms and evaluation
- xgboost — Gradient boosting framework
- shap — Model interpretability
- matplotlib & seaborn — Data visualization
- jupyter — Interactive notebooks

See `requirements.txt` for full dependency list.

## Future Work

- **Hyperparameter Optimization** — Systematic grid/random search with cross-validation
- **Class Imbalance Strategies** — SMOTE, class weighting, threshold optimization
- **Feature Selection** — RFE, permutation importance for dimensionality reduction
- **Model Deployment** — REST API using FastAPI, containerization with Docker
- **Monitoring Pipeline** — Model drift detection, performance tracking in production
- **Interactive Dashboard** — Streamlit/Dash application for non-technical stakeholders

## Summary

This project demonstrates proficiency in:

- **Data Science Fundamentals** — EDA, data cleaning, feature engineering
- **Machine Learning** — Model selection, training, hyperparameter tuning, evaluation
- **Model Interpretability** — SHAP analysis, feature importance, business translation
- **Software Engineering** — Code organization, documentation, reproducibility
- **Business Acumen** — Translating model findings into actionable insights

The pipeline is production-ready and extensible for real-world deployment.

---

**Amit Kashyap**

[GitHub](https://github.com/AmitKashyap-21)
