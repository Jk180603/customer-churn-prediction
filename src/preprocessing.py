import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


RAW_PATH = "data/raw/Customer-Churn-Records.csv"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    if "Exited" not in df.columns:
        raise ValueError("Target column 'Exited' not found.")

    df = pd.get_dummies(df, drop_first=True)

    return df


def split_and_scale(df: pd.DataFrame) -> None:
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
        f"{PROCESSED_DIR}/X_train.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
        f"{PROCESSED_DIR}/X_test.csv", index=False
    )

    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_columns.pkl")

    print("Preprocessing completed.")
    print(f"Train shape: {X_train_scaled.shape}")
    print(f"Test shape: {X_test_scaled.shape}")


def run_preprocessing() -> None:
    df = load_data()
    clean_df = clean_data(df)
    split_and_scale(clean_df)


if __name__ == "__main__":
    run_preprocessing()