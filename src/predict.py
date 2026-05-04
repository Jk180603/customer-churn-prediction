import joblib
import pandas as pd


MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, scaler, feature_columns


def predict_churn(input_data: dict):
    model, scaler, feature_columns = load_artifacts()

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, drop_first=True)

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    scaled_input = scaler.transform(input_df)

    prediction = int(model.predict(scaled_input)[0])
    probability = float(model.predict_proba(scaled_input)[0][1])

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "status": "Churn Risk" if prediction == 1 else "No Churn Risk",
    }


if __name__ == "__main__":
    sample_customer = {
        "CreditScore": 650,
        "Geography": "Germany",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 60000,
        "Complain": 0,
        "Satisfaction Score": 3,
        "Card Type": "DIAMOND",
        "Point Earned": 500,
    }

    print(predict_churn(sample_customer))