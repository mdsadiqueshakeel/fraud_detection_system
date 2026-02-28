import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from data_loader import load_data, split_data


def main():
    df = load_data("../data/creditcard.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    param_dist = {
        "n_estimators": [100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    model = RandomForestClassifier(
        class_weight="balanced",
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

    # Tune only on training data
    random_search.fit(X_train, y_train)

    print("Best RF Parameters:")
    print(random_search.best_params_)

    print("Best CV ROC-AUC:")
    print(random_search.best_score_)

    # Evaluate on untouched test set
    best_model = random_search.best_estimator_
    y_prob = best_model.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_prob)
    print("Test ROC-AUC:")
    print(test_auc)

    joblib.dump(best_model, "../models/rf_tuned_clean.pkl")
    print("Model saved successfully.")


if __name__ == "__main__":
    main()