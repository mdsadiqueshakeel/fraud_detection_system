from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path

import joblib
import numpy as np
import logging
import time


# ===== Config =====

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "rf_tuned_clean.pkl"

THRESHOLD = 0.20
MODEL_VERSION = "1.0.0"


# ===== Logging Setup =====

logging.basicConfig(
    filename="predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ===== Load Model Once =====

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")


app = FastAPI(title="Fraud Detection API", version=MODEL_VERSION)


# ===== Input Schemas =====

class Transaction(BaseModel):
    features: List[float] = Field(..., min_items=30, max_items=30)


class BatchTransaction(BaseModel):
    transactions: List[List[float]]


# ===== Utility: Risk Level =====

def get_risk_level(probability: float) -> str:
    if probability < 0.10:
        return "Low"
    elif probability < 0.30:
        return "Medium"
    else:
        return "High"


# ===== Single Prediction =====

@app.post("/predict")
def predict(transaction: Transaction):

    start_time = time.time()

    try:
        input_array = np.array(transaction.features).reshape(1, -1)

        probability = model.predict_proba(input_array)[0][1]
        prediction = int(probability >= THRESHOLD)

        risk_level = get_risk_level(probability)

        response_time = round((time.time() - start_time) * 1000, 2)

        logging.info({
            "model_version": MODEL_VERSION,
            "probability": float(probability),
            "prediction": prediction,
            "response_time_ms": response_time
        })

        return {
            "model_version": MODEL_VERSION,
            "fraud_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "is_fraud": prediction,
            "threshold_used": THRESHOLD,
            "response_time_ms": response_time
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


# ===== Batch Prediction =====

@app.post("/predict_batch")
def predict_batch(batch: BatchTransaction):

    start_time = time.time()

    try:
        input_array = np.array(batch.transactions)

        if input_array.shape[1] != 30:
            raise HTTPException(status_code=400, detail="Each transaction must contain 30 features.")

        probabilities = model.predict_proba(input_array)[:, 1]

        results = []

        for prob in probabilities:
            prediction = int(prob >= THRESHOLD)
            risk_level = get_risk_level(prob)

            results.append({
                "fraud_probability": round(float(prob), 4),
                "risk_level": risk_level,
                "is_fraud": prediction
            })

        response_time = round((time.time() - start_time) * 1000, 2)

        logging.info({
            "model_version": MODEL_VERSION,
            "batch_size": len(results),
            "response_time_ms": response_time
        })

        return {
            "model_version": MODEL_VERSION,
            "total_transactions": len(results),
            "results": results,
            "response_time_ms": response_time
        }

    except Exception as e:
        logging.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


# ===== Health Check =====

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD
    }