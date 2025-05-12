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
        if os.path.getsize(model_path) == 0:
            raise ValueError(f"Model file {filename} is empty")
        with open(model_path, "rb") as f:
            models[name] = joblib.load(f)
    return models

models = load_models()

# ---------------------- Groq API --------------------------
def extract_features_from_report(report_text):
    prompt = f"""Extract the following values as numbers from the medical report below:
    - age (patient's age in years)
    - sex (0 = female, 1 = male)
    - cp (chest pain type: 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)
    - trestbps (resting blood pressure in mmHg)
    - chol (serum cholesterol in mg/dL)
    - fbs (fasting blood sugar > 120 mg/dL: 1 = true, 0 = false)
    - restecg (resting ECG results: 0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy)
    - thalach (maximum heart rate achieved)
    - exang (exercise-induced angina: 1 = yes, 0 = no)
    - oldpeak (ST depression induced by exercise relative to rest)
    - slope (slope of the peak exercise ST segment: 0 = upsloping, 1 = flat, 2 = downsloping)
    - ca (number of major vessels colored by fluoroscopy: 0–3)
    - thal (thalassemia: 1 = normal, 2 = fixed defect, 3 = reversible defect)

    Report:
    {report_text}

    Instructions:
    - Return a valid JSON object with keys: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.
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
        required_keys = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
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
def generate_pdf_with_fitz(patient_name, input_data, predictions, probabilities, chart_path):
    pdf_doc = fitz.open()
    page = pdf_doc.new_page()

    y = 50
    line_spacing = 20

    page.insert_text((50, y), "Heart Disease Prediction Report", fontsize=16, fontname="helv")
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
        result = "High Risk" if predictions[model_name] == 1 else "Low Risk"
        prob = probabilities[model_name] * 100
        page.insert_text((60, y), f"{model_name}: {result} ({prob:.2f}%)", fontsize=11, fontname="helv")
        y += line_spacing

    img_rect = fitz.Rect(50, y + 10, 400, y + 310)
    page.insert_image(img_rect, filename=chart_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf_doc.save(tmpfile.name)
        return tmpfile.name

# ---------------------- UI --------------------------
st.title("❤️ Heart Disease Prediction App")
st.markdown("Upload a report or enter health data to predict heart disease risk using multiple ML models!")

patient_name = st.text_input("Enter Patient Name", "")

option = st.radio("Choose Input Method", ["Enter Manually", "Upload Health Report"])

input_data = {}

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
            required_keys = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
            missing_keys = [key for key in required_keys if key not in input_data or input_data[key] is None]
            if missing_keys:
                st.warning(f"The report is missing values for: {', '.join(missing_keys)}. Please provide these values manually.")
                for key in missing_keys:
                    if key == "sex":
                        input_data[key] = st.selectbox(f"{key} (0 = female, 1 = male)", [1, 0], key=key)
                    elif key in ["cp", "restecg", "slope", "ca", "thal"]:
                        max_val = 3 if key in ["cp", "ca"] else 2 if key in ["restecg", "slope"] else 3
                        input_data[key] = st.slider(f"{key}", 0, max_val, 0, key=key)
                    elif key == "fbs" or key == "exang":
                        input_data[key] = st.selectbox(f"{key} (0 = no, 1 = yes)", [0, 1], key=key)
                    elif key == "oldpeak":
                        input_data[key] = st.number_input(f"{key} (ST depression)", 0.0, 6.0, 0.0, key=key)
                    else:
                        input_data[key] = st.number_input(f"{key}", value=0.0, key=key)

elif option == "Enter Manually":
    input_data = {
        "age": st.number_input("Age", 20, 100, 50),
        "sex": st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female"),
        "cp": st.slider("Chest Pain Type (0–3)", 0, 3, 1),
        "trestbps": st.number_input("Resting Blood Pressure", 80, 200, 120),
        "chol": st.number_input("Cholesterol", 100, 600, 240),
        "fbs": st.selectbox("Fasting Blood Sugar > 120", [1, 0]),
        "restecg": st.slider("Resting ECG (0–2)", 0, 2, 1),
        "thalach": st.number_input("Max Heart Rate", 60, 220, 150),
        "exang": st.selectbox("Exercise Induced Angina", [1, 0]),
        "oldpeak": st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0),
        "slope": st.slider("Slope (0–2)", 0, 2, 1),
        "ca": st.slider("Major Vessels Colored (0–3)", 0, 3, 0),
        "thal": st.slider("Thal (1=Normal, 2=Fixed, 3=Reversible)", 1, 3, 2)
    }

if st.button("Predict Heart Disease"):
    if not patient_name.strip():
        st.warning("Please enter the patient's name before prediction.")
        st.stop()

    if not input_data:
        st.error("No valid input data provided. Please check the uploaded report or enter data manually.")
        st.stop()

    # Validate input data
    required_keys = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
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
        st.write(f"**{name}:** {'High Risk' if predictions[name] == 1 else 'Low Risk'} | Probability: {probabilities[name] * 100:.2f}%")

    best_model = max(probabilities, key=probabilities.get)
    st.success(f"⭐ **Most Confident Model: {best_model} ({probabilities[best_model] * 100:.2f}% probability of heart disease)**")

    # Accuracy chart
    model_accuracies = {
        "Logistic Regression": 0.83,
        "Random Forest": 0.88,
        "KNN": 0.79,
        "Decision Tree": 0.76,
        "SVM": 0.82,
        "Naive Bayes": 0.75
    }

    st.subheader("📊 Accuracy Comparison of Models")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(model_accuracies.keys(), [v * 100 for v in model_accuracies.values()], color='salmon')
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model Accuracy Comparison")
    plt.xticks(rotation=45)

    chart_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig.savefig(chart_path)
    st.pyplot(fig)

    # Generate PDF report
    pdf_path = generate_pdf_with_fitz(patient_name, input_data, predictions, probabilities, chart_path)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download Prediction Report", f, file_name="Heart_Disease_Report.pdf", mime="application/pdf")
