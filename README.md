# Customer Churn Predictor ML

End-to-end machine learning project predicting telecom customer churn using
Scikit-Learn and SHAP explainability.# Customer Churn Prediction (Machine Learning Project)

## Overview

Customer churn is a critical problem for subscription-based businesses such as telecom companies. Predicting which customers are likely to leave helps companies take proactive actions to retain them.

This project builds an **end-to-end machine learning pipeline** to predict customer churn using the **Telco Customer Churn dataset**. The project includes **data exploration, preprocessing, model training, evaluation, and model explainability using SHAP**.

---

## Project Goals

* Predict whether a telecom customer will churn.
* Identify key factors contributing to customer churn.
* Build a complete machine learning workflow suitable for real-world projects.

---

## Dataset

Dataset used in this project:

**Telco Customer Churn Dataset**

Source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains information about **7043 telecom customers** with features such as:

* Customer demographics
* Contract type
* Internet services
* Payment method
* Monthly charges
* Customer tenure

Target variable:

* **Churn**

  * `0` → Customer stayed
  * `1` → Customer left

---

## Project Structure

```
churn-predictor-ml
│
├── data
│   ├── raw
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed
│
├── models
│   └── churn_model.pkl
│
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
│
├── outputs
│   └── figures
│
├── src
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

## Machine Learning Workflow

### 1. Exploratory Data Analysis (EDA)

* Data inspection
* Distribution analysis
* Churn imbalance visualization
* Feature relationship exploration

### 2. Data Preprocessing

* Handling missing values
* Converting `TotalCharges` to numeric
* Encoding categorical variables
* Feature scaling

### 3. Model Training

Multiple machine learning models were trained and compared:

* Logistic Regression
* Random Forest
* Gradient Boosting

---

## Model Evaluation

Models were evaluated using:

* **Classification Report**
* **Confusion Matrix**
* **ROC Curve**
* **ROC-AUC Score**

The best performing model was selected based on **ROC-AUC performance**.

---

## Model Explainability

To understand how the model makes predictions, **SHAP (SHapley Additive Explanations)** was used.

SHAP identifies the most influential features affecting churn predictions.

Key churn drivers identified:

* Contract type
* Customer tenure
* Monthly charges
* Availability of technical support

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SHAP
* Jupyter Notebook

---

## Installation

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

## Running the Project

Run Jupyter notebooks:

```
jupyter notebook
```

Open the notebooks inside the `notebooks` folder to explore each step of the machine learning pipeline.

---

## Future Improvements

Potential improvements for this project include:

* Hyperparameter tuning
* Model deployment using FastAPI
* Building an interactive dashboard
* Real-time churn prediction system

---

## Author

**Amit Kashyap**

GitHub:
https://github.com/AmitKashyap-21
