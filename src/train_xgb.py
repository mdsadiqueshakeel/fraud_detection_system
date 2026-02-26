import joblib
from xgboost import XGBClassifier
from data_loader import load_data, split_data

def main():
    df = load_data("../data/creditcard.csv")
    X_train,X_test,y_train,y_test = split_data(df)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=1,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        eval_metric ="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train,y_train)

    joblib.dump(model, "../models/xgb_model.pkl")
    print("XGBoost trained and saved.")

if __name__ == "__main__":
    main()