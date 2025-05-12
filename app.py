import streamlit as st
import requests
import joblib
from io import BytesIO
import numpy as np
from PIL import Image

# ---------------------------
# Load model from Hugging Face
# ---------------------------
@st.cache_resource
def load_model(file_name):
    REPO = "Dharmesh234/Diebates23"
    url = f"https://huggingface.co/{REPO}/resolve/main/{file_name}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"❌ Failed to load model file: {file_name}")
        st.stop()
    try:
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        st.error(f"❌ Error loading model: {file_name}\n\n{e}")
        st.stop()

# ---------------------------
# Model mapping
# ---------------------------
model_file_map = {
    "Logistic Regression": "logistic_regression_model.joblib",
    "Naive Bayes": "naive_bayes_model.joblib",
    "SVM": "svm_model.joblib",
    "KNN": "knn_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "XGBoost": "xgboost_model.joblib"
}

# ---------------------------
# Main App
# ---------------------------
def main():
    st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

    with st.sidebar:
        try:
            logo = Image.open("heart_logo.png")
            st.image(logo, use_column_width=True)
        except:
            st.warning("Logo not found. Please upload `heart_logo.png` to the app folder.")
        
        st.title("🧠 Select Model")
        selected_model = st.selectbox("Choose ML model:", list(model_file_map.keys()))
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit and Hugging Face.")

    st.title("❤️ Heart Disease Prediction App")
    st.markdown("Enter patient metrics to predict **heart disease risk**.")

    model = load_model(model_file_map[selected_model])

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=130)
        chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=240)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)

    # Encoding categorical inputs
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak]])

    if st.button("🔍 Predict"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("🔴 The person is at **risk of heart disease**.")
        else:
            st.success("🟢 The person is **not likely** to have heart disease.")

if __name__ == "__main__":
    main()
