import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Simulated dataset
df = pd.DataFrame({
    "url_length": [20, 75, 60, 90],
    "has_https": [1, 0, 1, 0],
    "has_at_symbol": [0, 1, 0, 1],
    "has_ip_address": [0, 1, 0, 1],
    "count_dots": [2, 4, 3, 5],
    "has_hyphen": [0, 1, 0, 1],
    "has_subdomain": [0, 1, 0, 1],
    "count_digits": [3, 6, 2, 8],
    "label": [0, 1, 0, 1]
})

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
