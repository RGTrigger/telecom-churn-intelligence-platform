"""
Schema Mapping Module
Maps external company dataset columns to the model-required schema.
"""

REQUIRED_SCHEMA = {
    "gender": ["gender", "sex"],
    "SeniorCitizen": ["senior", "is_senior"],
    "Partner": ["partner", "has_partner"],
    "Dependents": ["dependents"],
    "tenure": ["tenure", "months_active", "years_active"],
    "PhoneService": ["phone_service"],
    "MultipleLines": ["multiple_lines"],
    "InternetService": ["internet_service", "internet_type"],
    "OnlineSecurity": ["online_security"],
    "OnlineBackup": ["online_backup"],
    "DeviceProtection": ["device_protection"],
    "TechSupport": ["tech_support"],
    "StreamingTV": ["streaming_tv"],
    "StreamingMovies": ["streaming_movies"],
    "Contract": ["contract", "contract_type", "plan_type"],
    "PaperlessBilling": ["paperless", "paperless_billing"],
    "PaymentMethod": ["payment_method"],
    "MonthlyCharges": ["monthly_charges", "bill_amount"],
    "TotalCharges": ["total_charges"]
}


def map_schema(df):
    column_mapping = {}

    for required_col, alternatives in REQUIRED_SCHEMA.items():
        for alt in alternatives:
            if alt in df.columns:
                column_mapping[alt] = required_col
                break

    df = df.rename(columns=column_mapping)

    missing_cols = [
        col for col in REQUIRED_SCHEMA
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Dataset missing required columns: {missing_cols}"
        )

    return df
