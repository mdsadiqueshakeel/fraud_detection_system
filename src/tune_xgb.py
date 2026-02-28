# import numpy as np
# import joblib

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
# from sklearn.metrics import roc_auc_score

# from data_loader import load_data

# def main():
#     df = load_data("../data/creditcard.csv")
#     X = df.drop("Class", axis=1)
#     y = df["Class"]

#     param_dist = {
#         "n_estimators": [100, 200, 300],
#         "max_depth": [None, 5, 10, 20],
#         "min_samples_split": [2, 5, 10],
#         "min_samples_leaf": [1, 2, 4]
#     }

#     model = RandomForestClassifier(
#         class_weight="balanced",
#         random_state=42,
#         n_jobs=-1
#     )

#     skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

#     random_search = RandomizedSearchCV(
#         model,
#         param_distributions=param_dist,
#         n_iter=15,
#         scoring="roc_auc",
#         cv=skf,
#         verbose=1,
#         n_jobs=-1,
#         random_state=42
#     )

#     random_search.fit(X, y)

#     print("Best RF Parameters:")
#     print(random_search.best_params_)
#     print("Best RF ROC-AUC:")
#     print(random_search.best_score_)

#     joblib.dump(random_search.best_estimator_, "../models/rf_tuned.pkl")

# if __name__ == "__main__":
#     main()

import joblib
from sklearn.metrics import roc_auc_score
from data_loader import load_data, split_data

df = load_data("../data/creditcard.csv")
X_train, X_test, y_train, y_test = split_data(df)

model = joblib.load("../models/xgb_tuned.pkl")

y_prob = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_prob))