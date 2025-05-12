import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# ---------------------------
# 🔽 File mapping from repo
# ---------------------------
model_file_map = {
    "Logistic Regression": "logistic_model.pkl",
    "Naive Bayes": "naive.pkl",
    "SVM": "svm.pkl",
    "KNN": "knn.pkl",
    "Decision Tree": "tree.pkl",
    "Random Forest": "rf.pkl",
    "XGBoost": "xgb.pkl",
}
SCALER_FILENAME = "scaler.pkl"
REPO = "Dharmesh234/Diebates23"

# ---------------------------
# 🔽 Load model from HF
# ---------------------------
@st.cache_resource
def load_model(file_name):
    url = f"https://huggingface.co/{REPO}/resolve/main/{file_name}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.get(url, headers=headers)
    return joblib.load(BytesIO(response.content))

# ---------------------------
# 🔽 Main app
# ---------------------------
def main():
    st.title("💓 Heart Disease Prediction App")

    st.sidebar.header("Choose Model")
    selected_model = st.sidebar.selectbox("Select ML Model", list(model_file_map.keys()))

    # Load selected model and scaler
    model = load_model(model_file_map[selected_model])
    scaler = load_model(SCALER_FILENAME)

    st.header("Enter Your Health Details:")

    age = st.slider("Age", 20, 90, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.slider("Resting BP (trestbps)", 90, 200, 120)
    chol = st.slider("Cholesterol (chol)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0, 1])
    restecg = st.selectbox("RestECG", [0, 1, 2])
    thalach = st.slider("Max Heart Rate (thalach)", 60, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    # Prepare input
    input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)

    if st.button("Predict"):
        result = model.predict(input_scaled)[0]
        st.success("✅ No Heart Disease" if result == 0 else "⚠️ High Risk of Heart Disease")

if __name__ == '__main__':
    main()
