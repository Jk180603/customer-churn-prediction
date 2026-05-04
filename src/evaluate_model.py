import os
import json
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
REPORTS_DIR = "reports"


def evaluate_model():
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()

    model = joblib.load(f"{MODEL_DIR}/best_model.pkl")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(f"{REPORTS_DIR}/metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    report = classification_report(y_test, y_pred)

    with open(f"{REPORTS_DIR}/classification_report.txt", "w") as file:
        file.write(report)

    print("Evaluation completed.")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    evaluate_model()