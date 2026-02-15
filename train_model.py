import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/telco_churn.csv")
df.columns = df.columns.str.strip()

TARGET = "Churn"

df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(TARGET, axis=1)
y = df[TARGET]

with open("feature_names.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=120,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {acc:.2f}")

with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)
