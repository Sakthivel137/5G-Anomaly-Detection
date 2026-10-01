import pandas as pd
import joblib

# Load saved model
model = joblib.load("models/xgboost_model.pkl")

# Load dataset
df = pd.read_csv("data/features/all_runs_features.csv")

# Load feature names
with open("models/feature_columns.txt", "r") as f:
    feature_columns = [line.strip() for line in f]

# Take one sample
X_sample = df[feature_columns].iloc[[0]]

# Make prediction
prediction = model.predict(X_sample)[0]
probability = model.predict_proba(X_sample)[0][1]

print("Model loaded successfully.")
print("Prediction:", "Abnormal" if prediction == 1 else "Normal")
print(f"Abnormal probability: {probability:.4f}")
