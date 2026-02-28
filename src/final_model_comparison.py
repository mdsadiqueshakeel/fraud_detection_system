import numpy as np 
import joblib

from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from data_loader import load_data, split_data

fraud_cost= 10000
fake_alarm_cost= 50

def evaluate_model(model_path,model_name):
    df = load_data("../data/creditcard.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    model = joblib.load(model_path)

    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n============================")
    print(f"{model_name}")
    print(f"============================")

    print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))

    best_loss = float("inf")
    best_threshold = None

    thresholds = np.linspace(0.1,0.9,17)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        loss = (fn * fraud_cost) + (fp * fake_alarm_cost)

        if loss < best_loss:
            best_loss = loss
            best_threshold = t

    print(f"Best Threshold: {best_threshold}")
    print(f"Minimum Business Loss: ₹{best_loss}")

def main():
    evaluate_model("../models/logistic_model.pkl", "Logistic Regression")
    evaluate_model("../models/rf_tuned_clean.pkl", "Random Forest Tuned")
    evaluate_model("../models/xgb_tuned_clean.pkl", "XGBoost Tuned")


if __name__ == "__main__":
    main()