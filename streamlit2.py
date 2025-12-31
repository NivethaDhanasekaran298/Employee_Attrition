import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Attrition Predictor")
st.title("Employee Attrition Prediction")

# -----------------------------
# Load trained model bundle
# -----------------------------
bundle = joblib.load("models/attrition_model.pkl")
model = bundle["model"]
features = bundle["features"]

# -----------------------------
# User Input Form
# -----------------------------
with st.form("attrition_form"):
    Age = st.number_input("Age", min_value=18, max_value=60, value=30)
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000, value=30000)
    YearsAtCompany = st.number_input("Years at Company", min_value=0, value=3)
    JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
    WorkLifeBalance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
    OverTime_ui = st.selectbox("OverTime", ["Yes", "No"])

    submit = st.form_submit_button("Predict")

# -----------------------------
# Prediction
# -----------------------------
if submit:

    # ✅ HARD CONVERSION (NO CHANCE OF STRING)
    if OverTime_ui == "Yes":
        OverTime = 1
    else:
        OverTime = 0

    # Build dataframe
    input_df = pd.DataFrame([{
        "Age": Age,
        "MonthlyIncome": MonthlyIncome,
        "YearsAtCompany": YearsAtCompany,
        "JobSatisfaction": JobSatisfaction,
        "WorkLifeBalance": WorkLifeBalance,
        "OverTime": OverTime
    }])

    # 🔒 Force numeric dtype (extra safety)
    input_df = input_df.astype(float)

    # Align columns exactly
    input_df = input_df[features]

    # DEBUG (you can remove later)
    st.write("Final input to model:")
    st.write(input_df)
    st.write(input_df.dtypes)

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Employee likely to LEAVE ({probability:.2%})")
    else:
        st.success(f"✅ Employee likely to STAY ({probability:.2%})")
