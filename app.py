import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import base64
import tempfile
import json
from io import BytesIO

# Load models
@st.cache_data
def load_models():
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

    models = {}
    for name, filename in model_files.items():
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
        with open(model_path, "rb") as f:
            models[name] = pickle.load(f)
    return models

models = load_models()

# Input form
def user_input():
    st.header("Heart Disease Predictor")
    name = st.text_input("Full Name", "John Doe")

    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG results (0-2)", [0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)", [0, 1, 2])

    input_data = {
        "name": name,
        "features": [
            age, 1 if sex == "Male" else 0, cp, trestbps, chol,
            fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        ]
    }
    return input_data

# Predict function
def predict_all(models, input_features):
    results = {}
    for name, model in models.items():
        pred = model.predict([input_features])[0]
        prob = model.predict_proba([input_features])[0][1] if hasattr(model, "predict_proba") else 0.5
        results[name] = {"prediction": pred, "probability": prob}
    return results

# Generate download report
def generate_report(input_data, predictions):
    report = {
        "name": input_data["name"],
        "input_features": input_data["features"],
        "predictions": predictions
    }
    json_str = json.dumps(report, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="heart_disease_report.json">📄 Download Report (JSON)</a>'
    st.markdown(href, unsafe_allow_html=True)

# Upload report and auto-complete
def upload_report():
    uploaded = st.file_uploader("📤 Upload Report File (JSON)", type="json")
    if uploaded:
        data = json.load(uploaded)
        name = data.get("name", "")
        features = data.get("input_features", [])
        if not name or len(features) != 13:
            st.warning("⚠️ Incomplete report data. Please complete manually.")
            return user_input()
        return {"name": name, "features": features}
    return None

# Main app
st.title("🫀 AI-Based Heart Disease Prediction App")
option = st.radio("Choose Input Method", ["Manual Entry", "Upload Report"])

if option == "Manual Entry":
    input_data = user_input()
elif option == "Upload Report":
    input_data = upload_report()

if input_data and st.button("🔍 Predict Heart Disease"):
    results = predict_all(models, input_data["features"])

    st.subheader(f"Prediction Results for {input_data['name']}")
    for model, result in results.items():
        st.write(f"**{model}**: {'💔 Disease Detected' if result['prediction'] == 1 else '❤️ No Disease'} (Chance: {result['probability']*100:.2f}%)")

    # Accuracy Graph
    st.subheader("📊 Model Probabilities Comparison")
    fig, ax = plt.subplots()
    model_names = list(results.keys())
    probs = [results[m]['probability'] for m in model_names]
    ax.barh(model_names, probs, color='skyblue')
    ax.set_xlabel('Probability of Heart Disease')
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    # Report Download
    generate_report(input_data, results)
