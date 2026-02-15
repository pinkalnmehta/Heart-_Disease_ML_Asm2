import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Cardio Risk Analyzer", layout="wide")

# Hide icons
st.markdown("""
<style>
header {visibility:hidden;}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# Background color
st.markdown("""
<style>
.stApp {
    background-color: #eaf4ff;
}
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("<h1 style='text-align:center;color:#8e44ad;'>🫀 Cardio Risk Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI system to estimate heart disease risk using patient medical attributes</p>", unsafe_allow_html=True)

st.markdown("---")

# -------- FILE UPLOAD --------
uploaded_file = st.file_uploader("📂 Upload Patient CSV File")

if uploaded_file:

    patient_df = pd.read_csv(uploaded_file)

    # -------- MODEL SELECT --------
    st.subheader("⚙ Select Prediction Model")
    model_choice = st.selectbox(
        "",
        ["logistic","tree","knn","bayes","forest","xgb"]
    )

    # Clean missing values
    patient_df.replace("?", pd.NA, inplace=True)
    patient_df.dropna(inplace=True)

    # Encoding
    encoded_df = pd.get_dummies(patient_df)

    # Load training columns
    trained_cols = joblib.load("model/columns.pkl")
    trained_cols = [c for c in trained_cols if c != "target"]

    encoded_df = encoded_df.reindex(columns=trained_cols, fill_value=0)

    # Load model
    scaler = joblib.load("model/scaler.pkl")
    predictor = joblib.load(f"model/{model_choice}.pkl")

    scaled_data = scaler.transform(encoded_df)
    results = predictor.predict(scaled_data)

    # Labels
    outcome = ["🔴 High Risk" if r == 1 else "🟢 Low Risk" for r in results]

    st.markdown("---")

    # -------- SUMMARY --------
    st.subheader("📊 Risk Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(results))
    c2.metric("High Risk", sum(results))
    c3.metric("Low Risk", len(results) - sum(results))

    st.markdown("---")

    # -------- FINAL REPORT --------
    report_df = pd.DataFrame({
        "Patient No": range(1, len(results)+1),
        "Prediction": outcome
    })

    st.subheader("🧾 Prediction Report")
    st.dataframe(report_df)
