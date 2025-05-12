import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from datetime import datetime
import base64
import fitz  # PyMuPDF for report extraction
from groq import Groq  # Assuming Groq API is set
import os

# Set page
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("🫀 Heart Disease Predictor App")

# Load models from Hugging Face
@st.cache_data
def load_models():
    models = {}
    model_names = [
        "logistic_regression_model.joblib",
        "naive_bayes_model.joblib",
        "svm_model.joblib",
        "knn_model.joblib",
        "decision_tree_model.joblib",
        "random_forest_model.joblib",
        "xgboost_model.joblib"
    ]
    base_url = "https://huggingface.co/Dharmesh234/Diebates23/resolve/main/"

    for name in model_names:
        url = base_url + name
        r = requests.get(url)
        r.raise_for_status()
        models[name.replace("_model.joblib", "")] = joblib.load(BytesIO(r.content))

    return models

models = load_models()

# Input features
st.sidebar.header("User Input")
name = st.sidebar.text_input("Name", "John Doe")

features = {
    'age': st.sidebar.number_input("Age", 1, 120, 50),
    'sex': st.sidebar.selectbox("Sex", [0, 1]),
    'cp': st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3]),
    'trestbps': st.sidebar.slider("Resting Blood Pressure (trestbps)", 90, 200, 120),
    'chol': st.sidebar.slider("Cholesterol (chol)", 100, 600, 240),
    'fbs': st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1]),
    'restecg': st.sidebar.selectbox("Rest ECG", [0, 1, 2]),
    'thalach': st.sidebar.slider("Max Heart Rate (thalach)", 60, 220, 150),
    'exang': st.sidebar.selectbox("Exercise Induced Angina", [0, 1]),
    'oldpeak': st.sidebar.slider("Oldpeak", 0.0, 6.2, 1.0),
    'slope': st.sidebar.selectbox("Slope", [0, 1, 2]),
    'ca': st.sidebar.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3]),
    'thal': st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])
}

input_df = pd.DataFrame([features])

# Prediction
if st.button("Predict Heart Disease"):
    st.subheader(f"Prediction Results for {name}")
    results = {}
    for model_name, model in models.items():
        try:
            results[model_name] = model.predict_proba(input_df)[0][1] * 100  # Probability of disease
        except:
            results[model_name] = model.predict(input_df)[0] * 100  # Fallback

    # Show results table
    result_df = pd.DataFrame(list(results.items()), columns=["Model", "Disease Probability (%)"])
    st.dataframe(result_df)

    # Plot results
    st.subheader("Model-wise Prediction Chart")
    fig, ax = plt.subplots()
    ax.barh(result_df['Model'], result_df['Disease Probability (%)'], color='coral')
    plt.xlabel("Probability (%)")
    st.pyplot(fig)

    # Downloadable report
    report = f"Report for {name}\nDate: {datetime.now()}\n\nUser Input:\n{input_df.to_string(index=False)}\n\nModel Results:\n{result_df.to_string(index=False)}"
    b64 = base64.b64encode(report.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{name}_heart_disease_report.txt">📄 Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# Upload Report and Extract
st.subheader("📤 Upload Existing Report")
uploaded_file = st.file_uploader("Upload Previous Report (PDF)", type=["pdf"])
if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "".join([page.get_text() for page in doc])

    # Send to Groq for data extraction
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not set in environment variables.")
    else:
        client = Groq(api_key=groq_api_key)
        prompt = f"""
        Extract the following from this report text:
        - Name
        - All 13 features required for heart disease prediction: age, sex, cp, trestbps, chol, fbs, restecg,
          thalach, exang, oldpeak, slope, ca, thal.
        If any field is missing, say "Missing".

        Report Text:
        {text}
        """
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        st.markdown("### Extracted Data from Report")
        st.text(answer)
        # Optionally, parse and allow user to edit missing fields here
