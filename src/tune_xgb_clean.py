import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from data_loader import load_data, split_data


def main():
    # 1️⃣ Load + Split FIRST
    df = load_data("../data/creditcard.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

    param_dist = {
        "n_estimators": [100, 150],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=8,
        scoring="roc_auc",
        cv=skf,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    # 2️⃣ Tune ONLY on training data
    random_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(random_search.best_params_)

    print("\nBest CV ROC-AUC:")
    print(random_search.best_score_)

    # 3️⃣ Evaluate ONLY on untouched test data
    best_model = random_search.best_estimator_
    y_prob = best_model.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_prob)
    print("\nTest ROC-AUC (Real Performance):")
    print(test_auc)

    # Save tuned model
    joblib.dump(best_model, "../models/xgb_tuned_clean.pkl")
    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()