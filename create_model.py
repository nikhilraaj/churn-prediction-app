import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create full feature dataset (same as app)
data = pd.DataFrame({
    "CreditScore": [600, 700, 500, 650],
    "Gender": [0, 1, 0, 1],
    "Age": [40, 35, 50, 30],
    "Tenure": [3, 5, 2, 4],
    "Balance": [60000, 50000, 70000, 40000],
    "NumOfProducts": [2, 1, 2, 1],
    "HasCrCard": [1, 1, 0, 1],
    "IsActiveMember": [1, 0, 0, 1],
    "EstimatedSalary": [80000, 90000, 60000, 70000],
    "Geography_Germany": [0, 1, 0, 0],
    "Geography_Spain": [0, 0, 1, 0],
    "Exited": [0, 0, 1, 0]
})

X = data.drop("Exited", axis=1)
y = data["Exited"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# Save everything
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")

# Save feature columns
import json
with open("artifacts/feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)