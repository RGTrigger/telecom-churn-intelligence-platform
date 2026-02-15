import pickle
import pandas as pd

from utils import get_risk_level, explain_churn, retention_action
from schema_mapper import map_schema

# Load trained components
model = pickle.load(open("churn_model.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))
encoders = pickle.load(open("label_encoders.pkl", "rb"))

# -----------------------------
# Load company dataset
# -----------------------------
df = pd.read_csv("data/telco_churn.csv")

# ✅ SCHEMA MAPPING (KEY ADDITION)
df = map_schema(df)

df_original = df.copy()

# -----------------------------
# Preprocessing
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

for col, le in encoders.items():
    if col in df.columns:
        df[col] = le.transform(df[col])

X = df[feature_names]
probs = model.predict_proba(X)[:, 1]

# -----------------------------
# Build output
# -----------------------------
results = []

for i, prob in enumerate(probs):
    risk = get_risk_level(prob)
    revenue_risk = prob * df_original.iloc[i]["MonthlyCharges"]

    results.append({
        "Churn_Probability": round(prob, 2),
        "Risk_Level": risk,
        "Revenue_Risk": round(revenue_risk, 2),
        "Recommended_Action": retention_action(df_original.iloc[i], risk),
        "Churn_Reasons": ", ".join(explain_churn(df_original.iloc[i]))
    })

output = pd.concat(
    [df_original.reset_index(drop=True), pd.DataFrame(results)],
    axis=1
)

output.sort_values("Revenue_Risk", ascending=False, inplace=True)
output.to_csv("churn_analysis_output.csv", index=False)

print("✅ Batch churn analysis completed.")
print("📄 Output saved as churn_analysis_output.csv")
