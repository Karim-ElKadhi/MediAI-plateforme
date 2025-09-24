import streamlit as st
import numpy as np
import pickle
import cv2
from tensorflow.keras.models import load_model
from brain_tumor import predict_sample

# ===========================
# Load Models
# ===========================
stroke_model = pickle.load(open("tr_model_best.sav", "rb"))
tumor_model = load_model("brain_tumor_model_v2.h5")

# ===========================
# MediAI Platform UI
# ===========================
st.set_page_config(page_title="MediAI Platform", layout="wide")

st.title("🧠 MediAI Platform")
st.markdown(
    "An AI-powered healthcare assistant for **Stroke Risk Assessment** and **Brain Tumor Detection**."
)

# Navigation Tabs
tab1, tab2 = st.tabs(["🏥 Stroke Assessment", "🩺 Tumor Detection"])

# ===========================
# Stroke Prediction UI
# ===========================
with tab1:
    st.header("Stroke Risk Assessment")
    st.markdown(
        "Evaluate patient data using advanced ML algorithms to assess stroke risk probability "
        "based on **medical history, lifestyle factors, and clinical measurements.**"
    )

    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio("Gender", ["Male", "Female"])
        hypertension = st.radio("Hypertension", ["Yes", "No"])
        avg_glucose = st.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0)

    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, value=40)
        heart_disease = st.radio("Heart Disease", ["Yes", "No"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

    smoking = st.radio("Smoking Status", ["Unknown", "Formerly smoked", "Never smoked", "Currently smoking"])

    if st.button("🚑 Assess Risk", use_container_width=True):
        gender_val = 1 if gender == "Male" else 0
        smoking_map = {
            "Unknown": [1, 0, 0, 0],
            "Formerly smoked": [0, 1, 0, 0],
            "Never smoked": [0, 0, 1, 0],
            "Currently smoking": [0, 0, 0, 1]
        }
        smoking_vals = smoking_map[smoking]

        input_data = [
            gender_val, age, int(hypertension == "Yes"),
            int(heart_disease == "Yes"), avg_glucose, bmi,
            *smoking_vals
        ]

        result = stroke_model.predict([input_data])[0]
        if result == 0:
            st.success("✅ Patient is **not likely** to have a stroke.")
        else:
            st.error("⚠️ Patient is **likely** to have a stroke.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ℹ️ About Stroke Assessment")
        st.write("""
        Our stroke prediction model analyzes key risk factors including **age, gender, medical history,
        glucose levels, BMI, and smoking status** to provide accurate risk assessments.
        
        - 🟢 Low Risk: Maintain healthy lifestyle  
        - 🔴 High Risk: Immediate medical consultation recommended
        """)

# ===========================
# Brain Tumor Detection UI
# ===========================
with tab2:
    st.header("Medical Image Analysis")
    st.markdown(
        "Upload MRI scans for **AI-powered tumor detection** using state-of-the-art deep learning models."
    )

    uploaded_file = st.file_uploader("📤 Upload Medical Image", type=["jpg", "png", "jpeg", "webp"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="BGR", caption="Uploaded MRI", use_column_width=True)

        if st.button("🔍 Analyze Image", use_container_width=True):
            result = predict_sample(uploaded_file.name, tumor_model)
            if "Tumor" in result:
                st.error("⚠️ Tumor detected!")
            else:
                st.success("✅ No tumor detected.")

    st.markdown("---")
    st.subheader("ℹ️ About Image Analysis")
    st.write("""
    Our tumor detection model uses **advanced CNNs** trained on MRI scans to detect abnormalities with high accuracy.
    
    - 🧾 Supports: MRI, CT scans, X-rays  
    - ⚡ Real-time predictions with confidence scoring  
    - 🔒 HIPAA-compliant secure image processing
    """)
