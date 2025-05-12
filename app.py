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
    st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

    with st.sidebar:
        try:
            logo = Image.open("diabetes_logo.png")
            st.image(logo, use_column_width=True)
        except:
            st.warning("Logo not found. Please upload `diabetes_logo.png` to root folder.")
        
        st.title("🔍 Select Model")
        selected_model = st.selectbox("Choose ML model:", list(model_file_map.keys()))
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit and Hugging Face.")

    st.title("🩺 Diabetes Prediction App")
    st.markdown("Enter patient health metrics to predict likelihood of diabetes.")

    model = load_model(model_file_map[selected_model])

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=180, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    if st.button("🔍 Predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, diabetes_pedigree, age]])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🔴 The person is likely to have **Diabetes**.")
        else:
            st.success("🟢 The person is **not likely** to have Diabetes.")

if __name__ == "__main__":
    main()
