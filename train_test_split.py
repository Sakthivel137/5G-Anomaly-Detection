import pandas as pd

# Load dataset
input_file = "data/features/all_runs_features.csv"
df = pd.read_csv(input_file)

# Define runs for training and testing
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

# Split by complete experimental runs
train_df = df[df["run_id"].isin(train_runs)].copy()
test_df = df[df["run_id"].isin(test_runs)].copy()

# Separate features and labels
feature_columns = [
    col for col in df.columns
    if col not in ["run_id", "window", "label"]
]

X_train = train_df[feature_columns]
y_train = train_df["label"]

X_test = test_df[feature_columns]
y_test = test_df["label"]

# Print results
print("TRAINING DATA")
print("Rows:", len(train_df))
print("Runs:", train_df["run_id"].unique().tolist())
print("Labels:")
print(y_train.value_counts().sort_index())

print("\nTEST DATA")
print("Rows:", len(test_df))
print("Runs:", test_df["run_id"].unique().tolist())
print("Labels:")
print(y_test.value_counts().sort_index())

print("\nFeature count:", len(feature_columns))

print("\nTrain/Test preparation successful.")