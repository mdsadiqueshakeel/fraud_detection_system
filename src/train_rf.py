import joblib
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_data, split_data

def main():
    df = load_data("../data/creditcard.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "../models/random_forest.pkl")

    print("Random Forest trained and saved")

if __name__ == "__main__":
    main()