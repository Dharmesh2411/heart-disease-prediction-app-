# app.py

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
    model_filenames = {
        "Logistic Regression": "logistic_regression_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "KNN": "knn_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
        "SVM": "svm_model.pkl",
        "Naive Bayes": "naive_bayes_model.pkl"
    }
    models = {}
    for name, filename in model_filenames.items():
        model_path = hf_hub_download(repo_id="jaik256/heartDiseasePredictor", filename=filename)
        with open(model_path, "rb") as f:
            models[name] = joblib.load(f)
    return models

models = load_models()

# ---------------------- Groq API Feature Extractor --------------------------
def extract_features_from_report(report_text):
    prompt = f"""Extract numeric health features for heart disease prediction from this report:
    {report_text}
    Return only this JSON format:
    {{
        "age": ..., "sex": ..., "cp": ..., "trestbps": ..., "chol": ..., "fbs": ..., "restecg": ...,
        "thalach": ..., "exang": ..., "oldpeak": ..., "slope": ..., "ca": ..., "thal": ...
    }}"""

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
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        st.error(f"Failed to extract features: {e}")
        return None

# ---------------------- Prediction Chart --------------------------
def plot_prediction_chart(probabilities):
    fig, ax = plt.subplots()
    ax.barh(list(probabilities.keys()), [p * 100 for p in probabilities.values()], color='teal')
    ax.set_xlabel('Probability (%)')
    ax.set_title('Heart Disease Risk by Model')
    chart_path = os.path.join(tempfile.gettempdir(), "risk_chart.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    return chart_path

# ---------------------- PDF Report Generator --------------------------
def generate_pdf_with_fitz(patient_name, input_data, predictions, probabilities, chart_path):
    pdf_doc = fitz.open()
    page = pdf_doc.new_page()
    y = 50
    page.insert_text((50, y), "Heart Disease Prediction Report", fontsize=16)
    y += 30
    page.insert_text((50, y), f"Patient: {patient_name}", fontsize=12)
    y += 20
    page.insert_text((50, y), f"Date: {datetime.date.today()}", fontsize=12)
    y += 30
    page.insert_text((50, y), "Input Features:", fontsize=12)
    for k, v in input_data.items():
        y += 15
        page.insert_text((60, y), f"{k}: {v}", fontsize=11)
    y += 30
    page.insert_text((50, y), "Model Predictions:", fontsize=12)
    for model, pred in predictions.items():
        prob = probabilities[model] * 100
        y += 15
        label = "High Risk" if pred == 1 else "Low Risk"
        page.insert_text((60, y), f"{model}: {label} ({prob:.2f}%)", fontsize=11)
    y += 30
    img_rect = fitz.Rect(50, y, 400, y + 250)
    page.insert_image(img_rect, filename=chart_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_doc.save(tmp.name)
        return tmp.name

# ---------------------- Streamlit UI --------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction")
st.markdown("Upload a medical report or enter values to predict heart disease risk using ML models.")

patient_name = st.text_input("Enter Patient Name")

option = st.radio("Choose Input Method", ["Enter Manually", "Upload Report"])
input_data = {}

if option == "Upload Report":
    uploaded_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
    if uploaded_file:
        if uploaded_file.name.endswith(".txt"):
            report_text = uploaded_file.read().decode()
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            with fitz.open(tmp_path) as doc:
                report_text = "\n".join([page.get_text() for page in doc])
        st.text_area("Extracted Report Text", report_text, height=200)
        st.info("Extracting features from report...")
        input_data = extract_features_from_report(report_text)

if option == "Enter Manually" or (option == "Upload Report" and not input_data):
    input_data = {
        "age": st.number_input("Age", 20, 100, 50),
        "sex": st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female"),
        "cp": st.slider("Chest Pain Type (0–3)", 0, 3, 1),
        "trestbps": st.number_input("Resting BP", 80, 200, 120),
        "chol": st.number_input("Cholesterol", 100, 600, 240),
        "fbs": st.selectbox("Fasting Blood Sugar > 120", [1, 0]),
        "restecg": st.slider("Resting ECG (0–2)", 0, 2, 1),
        "thalach": st.number_input("Max Heart Rate", 60, 220, 150),
        "exang": st.selectbox("Exercise Angina", [1, 0]),
        "oldpeak": st.number_input("Oldpeak", 0.0, 6.0, 1.0),
        "slope": st.slider("Slope (0–2)", 0, 2, 1),
        "ca": st.slider("Major Vessels (0–3)", 0, 3, 0),
        "thal": st.slider("Thal (1=Normal, 2=Fixed, 3=Reversible)", 1, 3, 2)
    }

if st.button("Predict"):
    df = pd.DataFrame([input_data])
    predictions = {}
    probabilities = {}

    for model_name, model in models.items():
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1] if hasattr(model, "predict_proba") else 0.0
        predictions[model_name] = pred
        probabilities[model_name] = prob

    st.success("✅ Prediction complete!")
    for model in predictions:
        label = "🛑 High Risk" if predictions[model] == 1 else "✅ Low Risk"
        st.write(f"**{model}**: {label} — Probability: {probabilities[model]*100:.2f}%")

    chart_path = plot_prediction_chart(probabilities)
    st.image(chart_path, caption="Risk Probabilities by Model")

    if patient_name:
        pdf_path = generate_pdf_with_fitz(patient_name, input_data, predictions, probabilities, chart_path)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name=f"{patient_name}_Heart_Report.pdf", mime="application/pdf")
