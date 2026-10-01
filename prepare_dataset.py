
import pandas as pd

# Load Member 1 feature dataset
input_file = "data/features/all_runs_features.csv"

df = pd.read_csv(input_file)

print("Original shape:", df.shape)

# Separate identifiers and target
run_ids = df["run_id"]
windows = df["window"]
y = df["label"]

# Remove identifiers and label from ML features
X = df.drop(columns=["run_id", "window", "label"])

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nLabels:")
print(y.value_counts().sort_index())

print("\nData types:")
print(X.dtypes)

print("\nMissing values:", X.isnull().sum().sum())

print("\nDataset preparation successful.")