import streamlit as st
import pandas as pd
import joblib
import requests
import datetime
from io import BytesIO
from fpdf import FPDF

# -------------------------------
# Function to load model from Hugging Face
def load_model_from_hf(model_name):
    url = f"https://huggingface.co/Dharmesh234/Diebates23/resolve/main/{model_name}"
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    return model

# -------------------------------
# Prediction function
def predict_heart_disease(model, input_data):
    prediction = model.predict(input_data)
    return prediction[0]

# -------------------------------
# PDF Report Generator
def generate_pdf(patient_name, input_data, model_name, prediction):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt=f"Heart Disease Prediction Report", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(200, 10, txt=f"Model Used: {model_name}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {'Positive' if prediction else 'Negative'}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    
    pdf.ln(10)
    pdf.cell(200, 10, txt="Input Features:", ln=True)
    for col, val in input_data.to_dict(orient='records')[0].items():
        pdf.cell(200, 10, txt=f"{col}: {val}", ln=True)
    
    # Create the PDF and save to a file in memory
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    
    return buffer

# -------------------------------
# Streamlit UI
st.title("💓 Heart Disease Prediction App")

patient_name = st.text_input("Enter Patient Name")
model_option = st.selectbox("Select Prediction Model", [
    "logistic_regression_model.joblib",
    "knn_model.joblib",
    "decision_tree_model.joblib",
    "random_forest_model.joblib",
    "svm_model.joblib",
    "xgboost_model.joblib",
    "naive_bayes_model.joblib"
])

# Input fields (example: 13 features used in UCI Heart Disease dataset)
age = st.number_input("Age", 0, 120)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 50, 200)
chol = st.number_input("Serum Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 50, 250)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
slope = st.selectbox("Slope of ST segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

input_df = pd.DataFrame([{
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}])

if st.button("Predict"):
    # Load the selected model from Hugging Face
    model = load_model_from_hf(model_option)
    
    # Make the prediction
    result = predict_heart_disease(model, input_df)
    
    # Show prediction result
    st.success(f"Prediction: {'Heart Disease Detected' if result else 'No Heart Disease'}")
    
    # Generate the PDF report
    report = generate_pdf(patient_name, input_df, model_option, result)
    
    # Provide a button for the user to download the report
    st.download_button(label="📄 Download Patient Report", data=report, file_name=f"{patient_name}_heart_report.pdf")
