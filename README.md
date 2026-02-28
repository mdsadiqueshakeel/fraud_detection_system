![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![ML](https://img.shields.io/badge/MachineLearning-RandomForest-orange)
# 🛡️ Real-Time Fraud Detection System

### Cost-Sensitive ML Pipeline + API Deployment

------------------------------------------------------------------------

## 🚀 Overview

This project implements an end-to-end **cost-sensitive fraud detection
system** designed to simulate real-world financial fraud detection
scenarios.

Unlike typical ML projects that focus only on accuracy, this system:

-   Handles extreme class imbalance\
-   Optimizes business loss instead of accuracy\
-   Uses proper cross-validation\
-   Applies threshold tuning\
-   Compares multiple models\
-   Deploys the final model via FastAPI\
-   Includes monitoring dashboard\
-   Implements versioning & structured logging

------------------------------------------------------------------------

# 🎯 Problem Statement

Fraud detection is a **highly imbalanced binary classification
problem**.

In real-world systems:

-   Fraud rate ≈ 1%\
-   Predicting all transactions as non-fraud gives \~99% accuracy\
-   But results in zero fraud prevention

Therefore, accuracy is misleading.

------------------------------------------------------------------------

# 📊 Dataset Used

Public Credit Card Fraud Detection dataset.

-   284,807 transactions\
-   492 fraud cases (\~0.17%)\
-   Highly imbalanced

------------------------------------------------------------------------

# 🧠 What Is PCA (Principal Component Analysis)?

The dataset uses PCA-transformed features:

V1, V2, V3, ..., V28

These are **principal components**, not raw transaction attributes.

## Why PCA Was Used

Original transaction features were anonymized for privacy reasons.

PCA:

-   Reduces dimensionality\
-   Converts correlated variables into orthogonal components\
-   Preserves maximum variance\
-   Removes interpretability

Mathematically:

Z = XW

Where:\
W = eigenvectors of covariance matrix\
Z = principal components

------------------------------------------------------------------------

# 🏗️ ML Pipeline

1.  Loaded PCA dataset\
2.  Stratified train-test split\
3.  Handled class imbalance\
4.  Trained Logistic Regression, Random Forest, XGBoost\
5.  Cross-validation\
6.  Hyperparameter tuning (RandomizedSearchCV)\
7.  Threshold tuning\
8.  Business loss minimization\
9.  Final model selection

------------------------------------------------------------------------

# 💰 Cost-Sensitive Optimization

Business Loss = (False Negatives × Fraud Cost) + (False Positives ×
False Alarm Cost)

Fraud Cost = ₹10,000\
False Alarm Cost = ₹50

------------------------------------------------------------------------

## Key Engineering Learnings

- Importance of threshold tuning in imbalanced classification
- Business-loss optimization over accuracy maximization
- Preventing data leakage during cross-validation
- Difference between CV score and real test performance
- Model serving with stateless API design

------------------------------------------------------------------------

# 🏆 Final Model Selection

  Model                   Test ROC-AUC   Best Threshold   Business Loss
  ----------------------- -------------- ---------------- ---------------
  Logistic Regression     0.9720         0.70             ₹122,250
  Random Forest (Tuned)   **0.9784**     0.20             **₹107,000**
  XGBoost (Tuned)         0.9776         0.75             ₹118,850

Final Production Model: **Random Forest (Tuned)**

------------------------------------------------------------------------

# 🔌 API Architecture

Client → FastAPI → Random Forest → Threshold 0.20 → Fraud Decision →
Logging

------------------------------------------------------------------------

# 🖥️ API Endpoints

POST /predict\
POST /predict_batch\
GET /health

------------------------------------------------------------------------

# 📊 Dashboard

Built using Streamlit for real-time monitoring and manual testing.

------------------------------------------------------------------------

# 🔮 Future Improvements

-   Behavioral feature engineering\
-   Synthetic transaction simulator\
-   Drift detection\
-   Automated retraining pipeline

------------------------------------------------------------------------

Author: Sadique\
Engineering Student \| Full Stack Developer \| ML Enthusiast
