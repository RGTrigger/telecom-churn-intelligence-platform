def get_risk_level(prob):
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.65:
        return "Medium Risk"
    else:
        return "High Risk"


def explain_churn(customer):
    reasons = []

    if customer.get("Contract") == "Month-to-month":
        reasons.append("Short-term contract")

    if customer.get("MonthlyCharges", 0) > 70:
        reasons.append("High monthly charges")

    if customer.get("tenure", 0) < 12:
        reasons.append("Low customer tenure")

    return reasons if reasons else ["Stable usage pattern"]


def retention_action(customer, risk):
    if risk == "High Risk":
        return "Offer discount + long-term contract"
    elif risk == "Medium Risk":
        return "Provide loyalty benefits"
    else:
        return "No action required"


def simulate_what_if(customer, model, encoders, feature_names):
    import pandas as pd

    simulations = {}

    # Scenario 1: Price reduction
    price_cut = customer.copy()
    price_cut["MonthlyCharges"] *= 0.9

    df1 = pd.DataFrame([price_cut])
    for col, le in encoders.items():
        if col in df1.columns:
            df1[col] = le.transform(df1[col])
    df1 = df1[feature_names]

    simulations["10% Price Reduction"] = round(
        model.predict_proba(df1)[0][1], 2
    )

    # Scenario 2: Contract change
    contract_change = customer.copy()
    contract_change["Contract"] = "One year"

    df2 = pd.DataFrame([contract_change])
    for col, le in encoders.items():
        if col in df2.columns:
            df2[col] = le.transform(df2[col])
    df2 = df2[feature_names]

    simulations["Switch to Yearly Contract"] = round(
        model.predict_proba(df2)[0][1], 2
    )

    return simulations
