import joblib
import numpy as np
from sklearn.metrics import(
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score
)

from data_loader import load_data, split_data


fraud_cost=10000
false_alarm_cost=50


def main():
    df = load_data("../data/creditcard.csv")
    X_train,X_test,y_train,y_test = split_data(df)

    model = joblib.load("../models/random_forest.pkl")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]


    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC Score:")
    print(roc_auc_score(y_test, y_prob))

    print("\n--- Threshold Tuning ---")
    thresholds = np.linspace(0.1,0.9,9)

    for t in thresholds:
        y_pred_custom = (y_prob >= t).astype(int)

        precision = precision_score(y_test, y_pred_custom)
        recall = recall_score(y_test, y_pred_custom)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_custom).ravel()

        loss = (fn * fraud_cost)+(fp * false_alarm_cost)

        print(f"\nThreshold: {t:.2f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Business Loss: ₹{loss}")


if __name__ == "__main__":
    main()