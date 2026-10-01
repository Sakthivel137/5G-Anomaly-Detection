import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================
# 1. Load dataset
# ============================================

input_file = "data/features/all_runs_features.csv"
df = pd.read_csv(input_file)

# ============================================
# 2. Define train/test runs
# ============================================

train_runs = [
    "normal_01",
    "normal_02",
    "abnormal_01",
    "abnormal_02"
]

test_runs = [
    "normal_03",
    "abnormal_03"
]

# ============================================
# 3. Split by complete experimental runs
# ============================================

train_df = df[df["run_id"].isin(train_runs)].copy()
test_df = df[df["run_id"].isin(test_runs)].copy()

# ============================================
# 4. Select ML features
# ============================================

feature_columns = [
    col for col in df.columns
    if col not in ["run_id", "window", "label"]
]

X_train = train_df[feature_columns]
y_train = train_df["label"]

X_test = test_df[feature_columns]
y_test = test_df["label"]

# ============================================
# 5. Create XGBoost model
# ============================================

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# ============================================
# 6. Train
# ============================================

print("Training XGBoost...")
model.fit(X_train, y_train)

print("Training completed.")

# ============================================
# 7. Predict test data
# ============================================

y_pred = model.predict(X_test)

# ============================================
# 8. Evaluation
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Abnormal"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# ============================================
# 9. Feature importance
# ============================================

importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance.to_string(index=False))
# ============================================
# 10. Save model and feature list
# ============================================

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/xgboost_model.pkl")

with open("models/feature_columns.txt", "w") as f:
    for feature in feature_columns:
        f.write(feature + "\n")

print("\nModel saved to: models/xgboost_model.pkl")
print("Feature list saved to: models/feature_columns.txt")