# 🛡️ Real-Time Fraud Detection System

### Cost-Sensitive ML Pipeline + API Deployment

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-API-green) ![MachineLearning](https://img.shields.io/badge/MachineLearning-RandomForest-orange)

------------------------------------------------------------------------

## Overview

This repository implements a cost-sensitive fraud detection pipeline and a simple API for serving the final model. The emphasis is on minimizing business loss (not just accuracy) in the presence of extreme class imbalance.

Key features:
- Stratified cross-validation and randomized hyperparameter search
- Cost-aware evaluation and threshold tuning
- Trained model export to `models/` for deployment
- Minimal FastAPI endpoint for predictions and a Streamlit dashboard for monitoring

------------------------------------------------------------------------

## Dataset

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- Place `creditcard.csv` in the repository `data/` folder: `data/creditcard.csv`.
- File is not included due to size and licensing.

Directory layout expected (root of repo):

- `data/creditcard.csv` (raw CSV)
- `models/` (output models will be saved here)
- `src/` (scripts and modules)

------------------------------------------------------------------------

## Reproducible Model Tuning (Random Forest)

The tuning script for the Random Forest is `src/tune_rf.py`. It:

- Loads the CSV from `../data/creditcard.csv` (so run it from `src/`),
- Splits the data using `split_data()` from `src/data_loader.py`,
- Runs a `RandomizedSearchCV` with `StratifiedKFold`,
- Prints best parameters and CV ROC-AUC, evaluates on the hold-out test set,
- Saves the best model to `../models/rf_tuned_clean.pkl` (relative to `src/`).

How to run (from repository root):

```bash
cd src
python tune_rf.py
```

Notes:
- Ensure `data/creditcard.csv` exists at `../data/creditcard.csv` when running from `src/`.
- After the script finishes, the tuned model will be saved to `models/rf_tuned_clean.pkl` at repository root.

------------------------------------------------------------------------

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

Typical packages used in this project (example):
- numpy, pandas, scikit-learn, xgboost, joblib, fastapi, uvicorn, streamlit

If you encounter version issues, create a virtual environment and reinstall.

------------------------------------------------------------------------

## Quick Start — From data to a saved tuned RF model

1. Download `creditcard.csv` from Kaggle and put it into `data/`.
2. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Run the tuner:

```bash
cd src
python tune_rf.py
```

4. Inspect output in the console. The final model is written to `models/rf_tuned_clean.pkl`.

------------------------------------------------------------------------

## Serving the Model (FastAPI)

An example FastAPI app is available at `src/app.py`. To run the API (after saving a model):

```bash
cd src
uvicorn app:app --reload
```

- Open Swagger at `http://127.0.0.1:8000/docs` to test endpoints.

The API expects the model file in the path where the app loads it; if you need the app to point to a specific model file, update the model path in `src/app.py`.

------------------------------------------------------------------------

## Additional scripts

- `src/train_rf.py`, `src/train_xgb.py`: training scripts for non-tuned training.
- `src/tune_xgb.py`, `src/tune_xgb_clean.py`: XGBoost tuning scripts.
- `src/final_model_comparison.py`: compares final models on business loss.
- `src/dashboard.py`: Streamlit dashboard for monitoring and visualizing results.

------------------------------------------------------------------------

## Model Comparison

Model                   Test ROC-AUC   Best Threshold   Business Loss
----------------------- -------------- ---------------- ---------------
Logistic Regression     0.9720         0.70             ₹122,250
Random Forest (Tuned)   0.9784         0.20             ₹107,000
XGBoost (Tuned)         0.9776         0.75             ₹118,850

Why Random Forest?

- Random Forest achieved the highest Test ROC-AUC (0.9784) and the lowest business loss (₹107,000) at the tuned threshold (0.20).
- In addition to measured metrics, Random Forest provided a strong practical trade-off: robust performance across folds, straightforward hyperparameter tuning, and fast inference for this dataset, making it a good production choice.


## Business Loss

We use a simple business loss function:

Business Loss = (False Negatives × Fraud Cost) + (False Positives × False Alarm Cost)

Default example costs used in experiments:

- Fraud Cost = ₹10,000
- False Alarm Cost = ₹50

Tune thresholds on validation folds to minimize this loss instead of optimizing pure accuracy.

------------------------------------------------------------------------

## Reproducibility & Notes

- All experiments use fixed random seeds in the scripts to ensure reproducible results.
- If you want to change RandomizedSearch ranges or CV splits, edit `src/tune_rf.py`.

------------------------------------------------------------------------

## Author

Sadique — Engineering Student | Full Stack Developer | ML Enthusiast
