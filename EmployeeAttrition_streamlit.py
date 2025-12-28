import streamlit as st
import pandas as pd
import joblib
import os

from preprocessing import clean_data, encode_data
from feature_engineering import add_features

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    layout="wide"
)

st.title("📊 Employee Attrition Prediction Dashboard")
st.write("Upload employee data to predict attrition risk.")

# --------------------------------------------------
# MODEL PATH (CHANGE ONLY IF YOUR PATH IS DIFFERENT)
# --------------------------------------------------
MODEL_PATH = r"D:\Nivi\Python\EmployeeAttrition\models\attrition_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Run Training.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# FILE UPLOADER (THIS PART IS CRITICAL)
# --------------------------------------------------
st.sidebar.header("📂 Upload Employee CSV")
uploaded_file = st.sidebar.file_uploader(
    "Upload Employee CSV File", type=["csv"]
)

# ⛔ STOP EVERYTHING UNTIL FILE IS UPLOADED
if uploaded_file is None:
    st.info("👈 Please upload a CSV file to proceed.")
    st.stop()

# --------------------------------------------------
# SAFE CSV READ (NO ERROR POSSIBLE HERE)
# --------------------------------------------------
raw_df = pd.read_csv(uploaded_file)

st.subheader("🔍 Raw Data Preview")
st.dataframe(raw_df.head())

# Keep readable copy
display_df = raw_df.copy()

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
df = clean_data(raw_df)
df = encode_data(df)
df = add_features(df)

# --------------------------------------------------
# ALIGN FEATURES WITH TRAINED MODEL
# --------------------------------------------------
model_features = model.feature_names_in_

for col in model_features:
    if col not in df.columns:
        df[col] = 0

X = df[model_features]

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
df["Attrition_Probability"] = model.predict_proba(X)[:, 1]
df["Attrition_Prediction"] = (df["Attrition_Probability"] >= 0.5).astype(int)
df["Attrition_Label"] = df["Attrition_Prediction"].map({0: "Stay", 1: "Leave"})

display_df["Attrition_Label"] = df["Attrition_Label"]
display_df["Attrition_Probability"] = df["Attrition_Probability"]

# --------------------------------------------------
# FILTERS
# --------------------------------------------------
st.sidebar.header("🔎 Filters")

dept_filter = st.sidebar.selectbox(
    "Department",
    ["All"] + sorted(display_df["Department"].astype(str).unique())
)

job_filter = st.sidebar.selectbox(
    "Job Role",
    ["All"] + sorted(display_df["JobRole"].astype(str).unique())
)

filtered_df = display_df.copy()

if dept_filter != "All":
    filtered_df = filtered_df[
        filtered_df["Department"].astype(str) == dept_filter
    ]

if job_filter != "All":
    filtered_df = filtered_df[
        filtered_df["JobRole"].astype(str) == job_filter
    ]

# --------------------------------------------------
# RESULTS
# --------------------------------------------------
st.subheader("📌 Attrition Predictions")
st.dataframe(
    filtered_df.sort_values(
        "Attrition_Probability", ascending=False
    )
)

# --------------------------------------------------
# METRICS
# --------------------------------------------------
st.subheader("📈 Summary Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", len(filtered_df))
col2.metric(
    "High Risk (Leave)",
    int((filtered_df["Attrition_Label"] == "Leave").sum())
)
col3.metric(
    "Low Risk (Stay)",
    int((filtered_df["Attrition_Label"] == "Stay").sum())
)

# --------------------------------------------------
# TOP 10 RISKY EMPLOYEES
# --------------------------------------------------
st.subheader("🚨 Top 10 At-Risk Employees")
st.dataframe(
    filtered_df.sort_values(
        "Attrition_Probability", ascending=False
    ).head(10)
)

# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------
st.subheader("⬇️ Download Predictions")
csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="employee_attrition_predictions.csv",
    mime="text/csv"
)
