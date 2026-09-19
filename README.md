# 💳 AI-Based Financial Fraud Detection and Transaction Risk Scoring System

## 📌 Project Overview

The AI-Based Financial Fraud Detection and Transaction Risk Scoring System is a machine learning based application designed to identify potentially fraudulent financial transactions.

The system analyzes transaction-related features and predicts whether a transaction is normal or potentially fraudulent. It also calculates a risk score from 0 to 100 and classifies the transaction into Low, Medium, or High Risk.

The project provides both single transaction analysis and bulk CSV transaction analysis through an interactive Streamlit dashboard.

---

## 🎯 Objectives

- Detect potentially fraudulent financial transactions.
- Use Machine Learning for fraud classification.
- Handle imbalanced transaction data using SMOTE.
- Generate a transaction risk score from 0 to 100.
- Classify transactions into different risk levels.
- Analyze multiple transactions using CSV upload.
- Provide an interactive web dashboard.
- Allow users to download bulk analysis results.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Streamlit
- Joblib
- VS Code

---

## 🤖 Machine Learning Algorithm

### Random Forest Classifier

Random Forest is used as the main classification algorithm.

It combines multiple decision trees to make predictions and classify transactions as:

- Normal
- Potential Fraud

The model uses multiple transaction characteristics to identify suspicious patterns.

---

## ⚖️ Handling Imbalanced Data

Financial fraud datasets generally contain fewer fraudulent transactions compared to normal transactions.

To handle this class imbalance, the project uses:

### SMOTE

SMOTE stands for:

**Synthetic Minority Over-sampling Technique**

It generates synthetic samples for the minority class to improve the model's ability to detect fraudulent transactions.

---

## 📊 Dataset Features

The dataset contains the following important features:

| Feature | Description |
|---|---|
| Transaction_ID | Unique transaction identifier |
| Amount | Transaction amount |
| Transaction_Hour | Hour at which transaction occurred |
| Previous_Transaction_Amount | Amount of previous transaction |
| Transaction_Frequency | Number of transactions |
| Account_Age | Age of the account |
| Is_New_Device | Indicates whether a new device was used |
| Is_International | Indicates an international transaction |
| Fraud | Target variable |

---

## 🔍 Risk Scoring

The trained model generates a fraud probability.

This probability is converted into a risk score between 0 and 100.

### Risk Classification

| Risk Score | Risk Level |
|---|---|
| 0–30 | Low Risk |
| 31–70 | Medium Risk |
| 71–100 | High Risk |

---

## 🔎 Risk Factors

The system checks several transaction characteristics that may indicate suspicious activity:

- High transaction amount
- Unusual transaction hours
- New device
- International transaction
- High transaction frequency
- Very new account
- Large increase compared with previous transaction

---

## 🖥️ Application Features

### 1. Transaction Dashboard

Displays:

- Total transactions
- Fraud transactions
- Normal transactions
- Fraud percentage

### 2. Transaction Visualization

The dashboard provides:

- Fraud vs Normal transaction chart
- Transactions by hour chart
- Risk level distribution chart

### 3. Model Performance

The dashboard displays:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 4. Single Transaction Analysis

Users can enter transaction information manually and receive:

- Fraud probability
- Risk score
- Risk level
- Model prediction
- Risk factors

### 5. Bulk CSV Analysis

Users can upload a CSV file containing multiple transactions.

The system generates:

- Fraud probability
- Risk score
- Risk level
- Prediction

The results can also be downloaded as a CSV file.

---

## 📈 Model Performance

The model was trained and evaluated using a train-test split.

The current test results are approximately:

- Accuracy: 70.10%
- Precision: 34.32%
- Recall: 51.35%
- F1 Score: 41.14%

The dataset used in this academic project is synthetically generated, so these results should not be interpreted as production-level fraud detection performance.

---

## 🔄 Project Workflow

```text
Transaction Dataset
        ↓
Data Generation
        ↓
Data Preprocessing
        ↓
Train-Test Split
        ↓
SMOTE
        ↓
Random Forest Model
        ↓
Model Evaluation
        ↓
Fraud Probability
        ↓
Risk Score
        ↓
Risk Level
        ↓
Streamlit Dashboard