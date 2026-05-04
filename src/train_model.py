import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


def load_train_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    return X_train, y_train


def train_model() -> None:
    X_train, y_train = load_train_data()

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train_resampled, y_train_resampled)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/best_model.pkl")

    print("Model training completed.")
    print("Model saved to models/best_model.pkl")


if __name__ == "__main__":
    train_model()