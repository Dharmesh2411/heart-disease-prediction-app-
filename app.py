import streamlit as st
import pandas as pd
import pickle
import requests
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score
from io import BytesIO
from fpdf import FPDF
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF for PDF parsing
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Repo and model filenames
repo_id = "Dharmesh234/Diebates23"
model_files = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "SVM": "svm_model.pkl",
    "KNN": "knn_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

# Load all models from Hugging Face
@st.cache_resource
def load_models():
    models = {}
    for name, file in model_files.items():
        model_path = hf_hub_download(repo_id=repo_id, filename=file, repo_type="model")
        with open(model_path, "rb") as f:
            models[name] = pickle.load(f)
    return models

models = load_models()

# App Interface
st.title("💖 Heart Disease Predictor")
st.write("Enter the following details to predict heart disease using multiple models.")

# Input Fields
with st.form("input_form"):
    name = st.text_input("Patient Name", max_chars=50)
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", [0, 1])  # 0: Female, 1: Male
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=250)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])
    submit = st.form_submit_button("Predict")

if submit:
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal]]
    
    results = {}
    for model_name, model in models.items():
        prediction = model.predict(input_data)[0]
        results[model_name] = prediction

    st.subheader(f"Prediction Results for {name}")
    for model, result in results.items():
        status = "✅ Likely Healthy" if result == 0 else "⚠️ Risk of Heart Disease"
        st.write(f"**{model}**: {status}")

    # Graph of model predictions
    st.subheader("Model Comparison")
    labels = list(results.keys())
    values = list(results.values())

    fig, ax = plt.subplots()
    ax.barh(labels, values, color=['green' if v == 0 else 'red' for v in values])
    ax.set_xlabel("Prediction (0=Healthy, 1=Disease)")
    st.pyplot(fig)

    # Generate report
    report_df = pd.DataFrame({
        "Name": [name],
        "Age": [age],
        "Sex": [sex],
        "Chest Pain": [cp],
        "BP": [trestbps],
        "Cholesterol": [chol],
        "FBS": [fbs],
        "Rest ECG": [restecg],
        "Thalach": [thalach],
        "Exang": [exang],
        "Oldpeak": [oldpeak],
        "Slope": [slope],
        "CA": [ca],
        "Thal": [thal],
        **results
    })

    def convert_df(df):
        return df.to_csv(index=False).encode("utf-8")

    st.download_button("📄 Download Report", convert_df(report_df), file_name=f"{name}_heart_report.csv")

# Upload Report and Predict
st.subheader("📤 Upload Existing Report (CSV or PDF)")
uploaded_file = st.file_uploader("Upload report", type=["csv", "pdf"])

if uploaded_file:
    def extract_pdf_text(pdf_file):
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = " ".join([page.get_text() for page in doc])
        return text

    try:
        if uploaded_file.name.endswith(".csv"):
            user_data = pd.read_csv(uploaded_file).iloc[0].to_dict()
        else:
            text = extract_pdf_text(uploaded_file)
            prompt = f"""
            Extract structured data from this patient report for heart disease prediction.
            Required fields: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.
            Text: {text}
            Return JSON.
            """
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}]
            )
            user_data = eval(completion.choices[0].message.content)

        missing_fields = [key for key in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                                          "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
                          if key not in user_data]

        for field in missing_fields:
            user_data[field] = st.number_input(f"Missing value for {field}:", key=field)

        input_data = [[user_data['age'], user_data['sex'], user_data['cp'], user_data['trestbps'],
                       user_data['chol'], user_data['fbs'], user_data['restecg'], user_data['thalach'],
                       user_data['exang'], user_data['oldpeak'], user_data['slope'],
                       user_data['ca'], user_data['thal']]]

        results = {}
        for model_name, model in models.items():
            prediction = model.predict(input_data)[0]
            results[model_name] = prediction

        st.success("✅ Predictions based on uploaded report:")
        for model, result in results.items():
            st.write(f"**{model}**: {'Healthy' if result == 0 else 'At Risk'}")

    except Exception as e:
        st.error(f"Error processing report: {e}")
