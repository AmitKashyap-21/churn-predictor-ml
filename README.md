# Customer Churn Prediction – End-to-End Machine Learning Project

Predicting customer churn is critical for subscription-based businesses such as telecom providers. Losing a customer is significantly more expensive than retaining one, so identifying high-risk customers early can help companies take proactive retention actions.

This project builds a **complete machine learning pipeline** that predicts telecom customer churn and provides an **interactive frontend interface** for testing predictions.

The workflow covers:

* Data exploration
* Data preprocessing
* Feature engineering
* Model training and evaluation
* Model explainability
* Interactive prediction UI using Streamlit

---

# Project Overview

The goal of this project is to predict whether a telecom customer is likely to cancel their service based on customer attributes such as:

* tenure
* contract type
* monthly charges
* technical support availability
* payment method

The trained model helps identify **customers at risk of leaving** so businesses can intervene early.

---

# Dataset

Dataset used: **Telco Customer Churn Dataset**

Source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Dataset characteristics:

* ~7,000 customer records
* 21 features
* Mix of categorical and numerical variables
* Binary classification problem

Target variable:

```
Churn
0 → Customer stays
1 → Customer leaves
```

---

# Machine Learning Pipeline

The project follows a standard data science workflow:

```
Raw Data
↓
Exploratory Data Analysis
↓
Data Cleaning
↓
Feature Encoding
↓
Model Training
↓
Model Evaluation
↓
Model Explainability
↓
Interactive Prediction UI
```

---

# Models Used

Three machine learning models were trained and compared.

| Model               | Purpose                            |
| ------------------- | ---------------------------------- |
| Logistic Regression | Baseline interpretable model       |
| Random Forest       | Handles nonlinear patterns         |
| Gradient Boosting   | Strong performance on tabular data |

Evaluation metrics used:

* ROC-AUC
* Precision
* Recall
* Confusion Matrix
* Classification Report

---

# Model Performance

| Model               | ROC-AUC |
| ------------------- | ------- |
| Logistic Regression | ~0.84   |
| Random Forest       | ~0.86   |
| Gradient Boosting   | ~0.87   |

Gradient Boosting achieved the best performance on the dataset.

---

# Model Explainability

The project uses **SHAP (SHapley Additive Explanations)** to interpret predictions.

Key churn drivers identified:

* Month-to-month contracts
* Higher monthly charges
* Low customer tenure
* Lack of technical support

These insights align with real telecom business patterns.

---

# Interactive Frontend (Streamlit)

An interactive **Streamlit UI** allows users to input customer details and instantly predict churn risk.

Users can:

* Enter customer attributes
* Predict churn probability
* Identify high-risk customers

Run the app locally:

```
streamlit run app.py
```

The application will launch at:

```
http://localhost:8501
```
<img width="1155" height="859" alt="image" src="https://github.com/user-attachments/assets/2757100b-778e-45f3-a88d-d5d9e7f87769" />

---

# Project Structure

```
churn-predictor-ml
│
├── data
│   ├── raw
│   └── processed
│
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
│
├── models
│   └── churn_model.pkl
│
├── outputs
│   ├── figures
│   └── reports
│
├── src
│   ├── preprocess.py
│   └── evaluate.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/AmitKashyap-21/churn-predictor-ml.git
cd churn-predictor-ml
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the Project

Run the notebooks in order:

```
01_eda.ipynb
02_preprocessing.ipynb
03_modeling.ipynb
04_explainability.ipynb
```

To launch the interactive UI:

```
streamlit run app.py
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SHAP
* Matplotlib
* Seaborn
* Streamlit
* Jupyter Notebook

---

# Future Improvements

Potential improvements for the project:

* Hyperparameter optimization
* Model deployment using FastAPI
* Real-time churn prediction pipeline
* Interactive churn dashboard
* Monitoring model drift in production

---

# Author

Amit Kashyap

GitHub
https://github.com/AmitKashyap-21
