import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="5G Anomaly Detection",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "data/features/all_runs_features.csv"
MODEL_PATH = "models/xgboost_model.pkl"
FEATURE_PATH = "models/feature_columns.txt"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD FEATURE LIST
# ============================================================

@st.cache_data
def load_features():
    with open(FEATURE_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]


df = load_data()
model = load_model()
feature_columns = load_features()


# ============================================================
# TITLE
# ============================================================

st.title("📡 5G Network Anomaly Detection")
st.subheader("XGBoost-based Telemetry Analysis Dashboard")

st.write(
    "This dashboard analyzes 5G network telemetry features and "
    "classifies traffic windows as Normal or Abnormal."
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Samples", len(df))

with col2:
    st.metric("Features", len(feature_columns))

with col3:
    st.metric("Abnormal Samples", int(df["label"].sum()))


st.dataframe(df, use_container_width=True)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

st.header("2. Normal vs Abnormal Distribution")

label_counts = df["label"].value_counts().sort_index()

label_names = {
    0: "Normal",
    1: "Abnormal"
}

display_labels = [
    label_names.get(index, str(index))
    for index in label_counts.index
]

fig, ax = plt.subplots()

ax.bar(display_labels, label_counts.values)

ax.set_xlabel("Class")
ax.set_ylabel("Number of Samples")
ax.set_title("Dataset Class Distribution")

st.pyplot(fig)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header("3. XGBoost Feature Importance")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

st.dataframe(
    importance_df,
    use_container_width=True
)

fig, ax = plt.subplots()

top_features = importance_df.head(10).sort_values("Importance")

ax.barh(
    top_features["Feature"],
    top_features["Importance"]
)

ax.set_xlabel("Importance")
ax.set_title("Top 10 Features")

st.pyplot(fig)


# ============================================================
# MANUAL PREDICTION
# ============================================================

st.header("4. Anomaly Prediction")

st.write(
    "Enter telemetry values below and the trained XGBoost model "
    "will classify the sample."
)


input_values = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(feature_columns):

    default_value = float(df[feature].median())

    if i % 2 == 0:
        with col1:
            input_values[feature] = st.number_input(
                feature,
                value=default_value
            )
    else:
        with col2:
            input_values[feature] = st.number_input(
                feature,
                value=default_value
            )


if st.button("🔍 Detect Anomaly"):

    input_df = pd.DataFrame(
        [input_values],
        columns=feature_columns
    )

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    abnormal_probability = probability[1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("🚨 ABNORMAL TRAFFIC DETECTED")

    else:

        st.success("✅ NORMAL TRAFFIC")


    st.metric(
        "Abnormal Probability",
        f"{abnormal_probability:.2%}"
    )


# ============================================================
# SAMPLE PREDICTION
# ============================================================

st.header("5. Test Existing Dataset Sample")

sample_index = st.number_input(
    "Select dataset row",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

if st.button("Test Selected Row"):

    sample = df.iloc[[int(sample_index)]]

    X_sample = sample[feature_columns]

    prediction = model.predict(X_sample)[0]

    probability = model.predict_proba(X_sample)[0]

    abnormal_probability = probability[1]

    actual_label = sample["label"].iloc[0]

    st.write(
        f"Actual label: "
        f"{'Abnormal' if actual_label == 1 else 'Normal'}"
    )

    if prediction == 1:

        st.error("Model Prediction: ABNORMAL")

    else:

        st.success("Model Prediction: NORMAL")

    st.metric(
        "Abnormal Probability",
        f"{abnormal_probability:.2%}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "5G Anomaly Detection Project | Member 3 Streamlit Dashboard"
)