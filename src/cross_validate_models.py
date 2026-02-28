import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loader import load_data

def main():
    df = load_data("../data/creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            ))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=len(y[y==0]) / len(y[y==1]),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    for name, model in models.items():
        auc_scores = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, y_prob)
            auc_scores.append(auc)

        print(f"\n{name}")
        print(f"Mean ROC-AUC: {np.mean(auc_scores):.4f}")
        print(f"Std ROC-AUC: {np.std(auc_scores):.4f}")

if __name__ == "__main__":
    main()