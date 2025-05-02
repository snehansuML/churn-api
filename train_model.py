import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("🔥 Training model with 2 categorical variables: contract_type & payment_method...")

n = 200
df = pd.DataFrame({
    "tenure_months": np.random.randint(1, 72, n),
    "monthly_charges": np.round(np.random.uniform(20, 120, n), 2),
    "total_charges": np.round(np.random.uniform(100, 9000, n), 2),
    "complaints": np.random.randint(0, 5, n),
    "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n),
    "payment_method": np.random.choice(["Electronic", "Bank Transfer", "Mailed"], n),
    "churned": np.random.choice([0, 1], n)
})

X = df.drop("churned", axis=1)
y = df["churned"]

encoders = {}
for col in ["contract_type", "payment_method"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/churn_model.pkl")
joblib.dump(encoders, "model/label_encoders.pkl")

print("✅ Model and encoders saved.")
