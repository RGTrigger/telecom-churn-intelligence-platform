import pickle
import pandas as pd

from utils import (
    get_risk_level,
    explain_churn,
    retention_action,
    simulate_what_if
)
from schema_mapper import map_schema

# Load trained components
model = pickle.load(open("churn_model.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))
encoders = pickle.load(open("label_encoders.pkl", "rb"))


def predict_customer(customer_dict):
    # Convert to DataFrame
    df = pd.DataFrame([customer_dict])

    # Apply schema mapping
    df = map_schema(df)

    # Convert mapped row back to dict (VERY IMPORTANT)
    mapped_customer = df.iloc[0].to_dict()

    # Encode categorical columns
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    df = df[feature_names]

    prob = model.predict_proba(df)[0][1]
    risk = get_risk_level(prob)

    return {
        "Churn Probability": round(prob, 2),
        "Risk Level": risk,
        "Reasons": explain_churn(mapped_customer),
        "Recommended Action": retention_action(mapped_customer, risk),
        "What-If Simulation": simulate_what_if(
            mapped_customer, model, encoders, feature_names
        )
    }

if __name__ == "__main__":

    # Company-specific schema example
    customer_data = {
        "sex": "Male",
        "months_active": 5,
        "bill_amount": 85.0,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_type": "Fiber optic",
        "paperless": "Yes",
        "partner": "No",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "total_charges": 450.0,
        "senior": 0
    }

    result = predict_customer(customer_data)

    print("\n--- Telecom Churn Intelligence Report ---")
    for k, v in result.items():
        print(f"{k}: {v}")
