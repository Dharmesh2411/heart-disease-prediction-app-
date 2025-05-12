```python
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
def load_models():
    model_filenames = {
        "Logistic Regression": "logistic_regression_diabetes_model.pkl",
        "Random Forest": "random_forest_diabetes_model.pkl",
        "SVM": "svm_diabetes_model.pkl",
        "KNN": "knn_diabetes_model.pkl",
        "Naive Bayes": "naive_bayes_diabetes_model.pkl",
        "Decision Tree": "decision_tree_diabetes_model.pkl"
    }
    models = {}
    for name, filename in model_filenames.items():
        try:
            # Replace with your Hugging Face repo or local path
            model_path = hf_hub_download(repo_id="jaik256/heart-disease-predictor", filename="heart_model.joblib")
            if os.path.getsize(model_path) == 0:
                raise ValueError(f"Model file {filename} is empty")
            with open(model_path, "rb") as f:
                models[name] = joblib.load(f)
        except Exception as e:
            st.warning(f"Failed to load {name} model: {str(e)}")
    return models

models = load_models()

# ---------------------- Groq API --------------------------
def extract_features_from_report(report_text):
    prompt = f"""Extract the following values as numbers from the medical report below:
    - Pregnancies (number of times pregnant)
    - Glucose (plasma glucose concentration in mg/dL)
    - BloodPressure (diastolic blood pressure in mmHg)
    - SkinThickness (triceps skin fold thickness in mm)
    - Insulin (2-hour serum insulin in mu U/ml)
    - BMI (body mass index in kg/m²)
    - DiabetesPedigreeFunction (diabetes pedigree function score)
    - Age (age in years)

    Report:
    {report_text}

    Instructions:
    - Return a valid JSON object with keys: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
    - All values must be numbers (integers or floats). If a value is missing or unclear, return null for that key.
    - Do not include any explanations or additional text outside the JSON object.
    """

    url = "https://api.groq.com/openai/v1/chat/completions"
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
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        extracted_data = json.loads(content)

        # Validate extracted data
        required_keys = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        for key in required_keys:
            if key not in extracted_data or extracted_data[key] is None:
                st.warning(f"Missing or invalid value for '{key}' in the report.")
                return None
            if not isinstance(extracted_data[key], (int, float)):
                st.error(f"Invalid value for '{key}': {extracted_data[key]}. Must be a number.")
                return None

        return extracted_data
    except Exception as e:
        st.error(f"❌ Failed to extract features from the report: {str(e)}")
        return None

# ---------------------- PDF Report Generator --------------------------
def generate_pdf_with_fitz(patient_name, input_data, predictions, probabilities, chart_path=None):
    pdf_doc = fitz.open()
    page = pdf_doc.new_page()

    y = 50
    line_spacing = 20

    page.insert_text((50, y), "Diabetes Prediction Report", fontsize=16, fontname="helv")
    y += line_spacing * 2

    page.insert_text((50, y), f"Patient Name: {patient_name}", fontsize=12, fontname="helv")
    y += line_spacing
    page.insert_text((50, y), f"Date: {datetime.date.today().strftime('%B %d, %Y')}", fontsize=12, fontname="helv")
    y += line_spacing

    page.insert_text((50, y), "Input Features:", fontsize=12, fontname="helv")
    y += line_spacing
    for key, value in input_data.items():
        page.insert_text((60, y), f"{key}: {value}", fontsize=11, fontname="helv")
        y += line_spacing

    y += line_spacing
    page.insert_text((50, y), "Prediction Results:", fontsize=12, fontname="helv")
    y += line_spacing
    for model_name in predictions:
        result = "Diabetic" if predictions[model_name] == 1 else "Non-Diabetic"
        prob = probabilities[model_name] * 100
        page.insert_text((60, y), f"{model_name}: {result} ({prob:.2f}%)", fontsize=11, fontname="helv")
        y += line_spacing

    if chart_path:
        y += line_spacing
        page.insert_text((50, y), "Model Accuracy Comparison:", fontsize=12, fontname="helv")
        img_rect = fitz.Rect(50, y + 10, 400, y + 310)
        page.insert_image(img_rect, filename=chart_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf_doc.save(tmpfile.name)
        return tmpfile.name

# ---------------------- UI --------------------------
st.title("🩺 Diabetes Prediction App")
st.markdown("Upload a medical report or enter health data to predict diabetes risk using multiple ML models!")

patient_name = st.text_input("Enter Patient Name", "")

option = st.radio("Choose Input Method", ["Enter Manually", "Upload Health Report"])

input_data = {}
is_report_upload = option == "Upload Health Report"

if option == "Upload Health Report":
    uploaded_file = st.file_uploader("Upload Report (TXT or PDF)", type=["txt", "pdf"])
    if uploaded_file:
        if uploaded_file.name.endswith(".txt"):
            report_text = uploaded_file.read().decode()
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            doc = fitz.open(tmp_path)
            report_text = "\n".join([page.get_text() for page in doc])
            doc.close()
        st.subheader("Extracted Report Text:")
        st.text(report_text)
        st.info("Calling Groq API to extract features...")
        input_data = extract_features_from_report(report_text)

        if input_data:
            # Check for missing or None values
            required_keys = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            missing_keys = [key for key in required_keys if key not in input_data or input_data[key] is None]
            if missing_keys:
                st.warning(f"The report is missing values for: {', '.join(missing_keys)}. Please provide these values manually.")
                for key in missing_keys:
                    if key == "Pregnancies":
                        input_data[key] = st.number_input(f"{key} (number of pregnancies)", 0, 20, 0, key=key)
                    elif key == "Glucose":
                        input_data[key] = st.number_input(f"{key} (mg/dL)", 0.0, 300.0, 100.0, key=key)
                    elif key == "BloodPressure":
                        input_data[key] = st.number_input(f"{key} (mmHg)", 0.0, 200.0, 80.0, key=key)
                    elif key == "SkinThickness":
                        input_data[key] = st.number_input(f"{key} (mm)", 0.0, 100.0, 20.0, key=key)
                    elif key == "Insulin":
                        input_data[key] = st.number_input(f"{key} (mu U/ml)", 0.0, 1000.0, 100.0, key=key)
                    elif key == "BMI":
                        input_data[key] = st.number_input(f"{key} (kg/m²)", 0.0, 70.0, 25.0, key=key)
                    elif key == "DiabetesPedigreeFunction":
                        input_data[key] = st.number_input(f"{key} (score)", 0.0, 3.0, 0.5, key=key)
                    elif key == "Age":
                        input_data[key] = st.number_input(f"{key} (years)", 0, 120, 30, key=key)

elif option == "Enter Manually":
    input_data = {
        "Pregnancies": st.number_input("Pregnancies (number of times pregnant)", 0, 20, 0),
        "Glucose": st.number_input("Glucose (mg/dL)", 0.0, 300.0, 100.0),
        "BloodPressure": st.number_input("BloodPressure (mmHg)", 0.0, 200.0, 80.0),
        "SkinThickness": st.number_input("SkinThickness (mm)", 0.0, 100.0, 20.0),
        "Insulin": st.number_input("Insulin (mu U/ml)", 0.0, 1000.0, 100.0),
        "BMI": st.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0),
        "DiabetesPedigreeFunction": st.number_input("DiabetesPedigreeFunction (score)", 0.0, 3.0, 0.5),
        "Age": st.number_input("Age (years)", 0, 120, 30)
    }

if st.button("Predict Diabetes"):
    if not patient_name.strip():
        st.warning("Please enter the patient's name before prediction.")
        st.stop()

    if not input_data:
        st.error("No valid input data provided. Please check the uploaded report or enter data manually.")
        st.stop()

    # Validate input data
    required_keys = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    missing_keys = [key for key in required_keys if key not in input_data or input_data[key] is None]
    if missing_keys:
        st.error(f"Missing required features: {', '.join(missing_keys)}. Please provide these values.")
        st.stop()

    # Debug: Show input data
    st.write("Input data for prediction:", input_data)

    try:
        features = pd.DataFrame([input_data])
        # Ensure all columns are numeric
        features = features.astype(float)
    except ValueError as e:
        st.error(f"Invalid input data: {str(e)}. Please ensure all values are numeric.")
        st.stop()

    predictions = {}
    probabilities = {}

    for name, model in models.items():
        try:
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][1]
            predictions[name] = pred
            probabilities[name] = prob
        except Exception as e:
            st.warning(f"Prediction failed for {name}: {str(e)}")
            continue

    if not predictions:
        st.error("No models were able to make predictions. Please check the input data.")
        st.stop()

    st.subheader("🩺 Prediction Results from Multiple Models:")
    for name in predictions:
        st.write(f"**{name}:** {'Diabetic' if predictions[name] == 1 else 'Non-Diabetic'} | Probability: {probabilities[name] * 100:.2f}%")

    best_model = max(probabilities, key=probabilities.get)
    st.success(f"⭐ **Most Confident Model: {best_model} ({probabilities[best_model] * 100:.2f}% probability of diabetes)**")

    # Accuracy chart for report uploading scenario
    if is_report_upload:
        st.subheader("📊 Accuracy Comparison of Models (Report Uploading)")
        model_accuracies = {
            "Logistic Regression": 0.886,  # From[](https://www.sciencedirect.com/science/article/pii/S2405959521000205)
            "Random Forest": 0.820,       # From[](https://www.sciencedirect.com/science/article/pii/S2772442522000582)
            "SVM": 0.890,                 # From[](https://www.emerald.com/insight/content/doi/10.1016/j.aci.2018.12.004/full/html)
            "KNN": 0.788,                 # From[](https://pmc.ncbi.nlm.nih.gov/articles/PMC10378239/)
            "Naive Bayes": 0.823,         # From[](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0175-6)
            "Decision Tree": 0.761        # From[](https://pmc.ncbi.nlm.nih.gov/articles/PMC10378239/)
        }

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(model_accuracies.keys(), [v * 100 for v in model_accuracies.values()], color='skyblue')
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Model Accuracy for Diabetes Prediction (Report Uploading)")
        plt.xticks(rotation=45)

        chart_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig.savefig(chart_path)
        st.pyplot(fig)
    else:
        chart_path = None

    # Generate PDF report
    pdf_path = generate_pdf_with_fitz(patient_name, input_data, predictions, probabilities, chart_path)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download Prediction Report", f, file_name="Diabetes_Prediction_Report.pdf", mime="application/pdf")
```
