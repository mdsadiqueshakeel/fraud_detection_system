import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from data_loader import load_data, split_data

def build_pipleine():
    pipleline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])
    return pipleline

def main():
    df = load_data("../data/creditcard.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_pipleine()

    pipeline.fit(X_train,y_train)

    joblib.dump(pipeline, "../models/logistic_model.pkl")

    print("Model trained and saved successfully")

if __name__ == "__main__":
    main() 