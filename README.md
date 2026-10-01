# Member 3 - 5G Anomaly Detection Dashboard

## Purpose

Streamlit dashboard for visualizing 5G network telemetry and testing the trained XGBoost anomaly detection model.

## Dataset

The dashboard uses:

data/features/all_runs_features.csv

Dataset:
- 100 samples
- 19 ML features
- 55 normal samples
- 45 abnormal samples

## Model

Trained model:

models/xgboost_model.pkl

Feature list:

models/feature_columns.txt

Model type:
XGBoost Classifier

## Run the Dashboard

Create and activate a Python virtual environment, install the requirements, then run:

python -m streamlit run app.py

The dashboard normally opens at:

http://localhost:8501

If that port is already in use, Streamlit may use another port such as 8502.

## Dashboard Features

1. Dataset overview
2. Normal vs Abnormal distribution
3. XGBoost feature importance
4. Manual telemetry prediction
5. Existing dataset sample prediction
6. Abnormal probability

## Verification

Normal sample:
- Dataset row: 0
- Actual: Normal
- Prediction: Normal
- Abnormal probability: 5.11%

Abnormal sample:
- Dataset row: 55
- Actual: Abnormal
- Prediction: Abnormal
- Abnormal probability: 61.08%

## Project Structure

member3_5G_anomaly_detection/

+-- app.py
+-- README.md
+-- requirements.txt
+-- data/
¦   +-- features/
¦       +-- all_runs_features.csv
+-- models/
    +-- xgboost_model.pkl
    +-- feature_columns.txt

The .venv folder is intentionally excluded from the handoff ZIP.
