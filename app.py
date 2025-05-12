import streamlit as st
import os
import joblib
import requests
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import json
import fitz  # PyMuPDF
import datetime
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------------- Load models --------------------------
@st.cache_resource
def load_models():
    # Download the models from Hugging Face
    model_names = [
        "logistic_regression_model.pkl",
        "naive_bayes_model.pkl",
        "svm_model.pkl",
        "knn_model.pkl",
        "decision_tree_model.pkl",
        "random_forest_model.pkl",
        "xgboost_model.pkl"
    ]
    
    models = {}
    for model_name in model_names:
        model_path = hf_hub_download(repo_id="jaik256/heartDiseasePredictor", filename=model_name)
        models[model_name.split(".")[0]] = joblib.load(model_path)
    
    return models

models = load_models()

# ---------------------- Groq API --------------------------
def extract_features_from_report(report_text):
    prompt = f"""Extract the following values as numbers from the medical report below:
    - Pregnancies
    - Glucose
    - BloodPressure
    - SkinThickness
    - Insulin
    - BMI
    - DiabetesPedigreeFunction
    - Age

    Report:
    {report_text}

    Instructions:
    - Return a valid JSON object with those keys.
    - All values must be numbers or null.
    - No text outside JSON.
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, data=json.dumps(data))
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        extracted_data = json.loads(content)

        required_keys = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        for key in required_keys:
            if key not in extracted_data or not isinstance(extracted_data[key], (int, float, type(None))):
                st.warning(f"Missing or invalid value for '{key}'")
        return extracted_data
    except Exception as e:
        st.error(f"Groq extraction error: {e}")
        return None

# ---------------------- PDF Report Generator --------------------------
def generate_pdf_with_fitz(patient_name, input_data, prediction, probability):
    pdf_doc = fitz.open()
    page = pdf_doc.new_page()

    y = 50
    page.insert_text((50, y), "🩺 Diabetes Prediction Report", fontsize=16)
    y += 30
    page.insert_text((50, y), f"Patient: {patient_name}")
    y += 20
    page.insert_text((50, y), f"Date: {datetime.date.today().strftime('%Y-%m-%d')}")
    y += 30
    page.insert_text((50, y), "Input Data:", fontsize=12)
    y += 20
    for k, v in input_data.items():
        page.insert_text((60, y), f"{k}: {v}")
        y += 18
    y += 20
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    page.insert_text((50, y), f"Prediction: {result} ({probability:.2f}%)", fontsize=12)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_doc.save(tmp.name)
        return tmp.name

# ---------------------- Streamlit UI --------------------------
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.write("Upload a medical report or enter health details to get your diabetes risk.")

patient_name = st.text_input("Patient Name")

input_method = st.radio("Select Input Method", ["Enter Manually", "Upload Report"])

input_data = {}

# Upload & Extract
if input_method == "Upload Report":
    file = st.file_uploader("Upload TXT or PDF file", type=["txt", "pdf"])
    if file:
        if file.name.endswith(".txt"):
            report_text = file.read().decode()
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                path = tmp.name
            doc = fitz.open(path)
            report_text = "\n".join([page.get_text() for page in doc])
            doc.close()

        st.text_area("Extracted Report", report_text, height=200)
        st.info("🔍 Extracting features using Groq API...")
        input_data = extract_features_from_report(report_text)

        # Fill missing fields
        for key in ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]:
            if input_data.get(key) is None:
                default = 0.0 if key != "Age" else 30
                input_data[key] = st.number_input(f"{key} (Enter manually)", value=default)

else:
    input_data = {
        "Pregnancies": st.number_input("Pregnancies", 0, 20, 2),
        "Glucose": st.number_input("Glucose (mg/dL)", 0.0, 300.0, 120.0),
        "BloodPressure": st.number_input("Blood Pressure (mmHg)", 0.0, 200.0, 70.0),
        "SkinThickness": st.number_input("Skin Thickness (mm)", 0.0, 100.0, 20.0),
        "Insulin": st.number_input("Insulin (mu U/ml)", 0.0, 1000.0, 80.0),
        "BMI": st.number_input("BMI", 0.0, 70.0, 25.0),
        "DiabetesPedigreeFunction": st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5),
        "Age": st.number_input("Age", 0, 120, 30)
    }

# Model Selection
model_choice = st.selectbox("Select Model", ["Logistic Regression", "Naive Bayes", "SVM", "KNN", "Decision Tree", "Random Forest", "XGBoost"])

# ---------------------- Prediction --------------------------
if st.button("Predict"):
    if not patient_name:
        st.warning("Please enter patient name.")
        st.stop()

    model = models[model_choice.replace(" ", "_").lower()]
    features_df = pd.DataFrame([input_data])
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0][1] * 100

    result_text = "🟥 Diabetic" if prediction == 1 else "🟩 Non-Diabetic"
    st.subheader(f"Prediction Result: {result_text}")
    st.write(f"Probability of being diabetic: **{probability:.2f}%**")

    # PDF Report Download
    pdf_path = generate_pdf_with_fitz(patient_name, input_data, prediction, probability)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name="diabetes_report.pdf")
