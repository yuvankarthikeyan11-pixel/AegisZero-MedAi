import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from fpdf import FPDF
import requests
from tensorflow.keras.applications.densenet import preprocess_input

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI PneumoScan", layout="wide")

# =========================
# MODERN MEDICAL UI STYLE
# =========================
st.markdown("""
<style>

.main {
    background-color: #eef3f8;
}

.header {
    background: linear-gradient(90deg,#0f4c75,#3282b8);
    padding: 22px;
    border-radius: 14px;
    color: white;
    text-align: center;
    font-size: 30px;
    font-weight: 600;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}

.section-title {
    font-size:18px;
    font-weight:600;
    color:#0f4c75;
    margin-bottom:10px;
}

.status-normal {
    color:#1b8a3a;
    font-weight:600;
    font-size:16px;
}

.status-alert {
    color:#d62828;
    font-weight:600;
    font-size:16px;
}

.footer {
    text-align:center;
    color:gray;
    font-size:13px;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">AI PneumoScan — Clinical Decision Support</div>', unsafe_allow_html=True)

# =========================
# SETTINGS
# =========================
IMG_SIZE = 224
VALIDATOR_MODEL_PATH = "models/xray_validator.h5"
MULTICLASS_MODEL_PATH = "models/best_multiclass.h5"

CLASS_NAMES = ["Bacterial Pneumonia", "Normal", "Viral Pneumonia"]

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    validator = tf.keras.models.load_model(VALIDATOR_MODEL_PATH, compile=False)
    multiclass = tf.keras.models.load_model(MULTICLASS_MODEL_PATH, compile=False)
    return validator, multiclass

validator_model, multiclass_model = load_models()

# =========================
# PREPROCESS IMAGE (FIXED)
# =========================
def preprocess_image(img):
    # Convert RGB → BGR (important if trained with OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32")

    # DenseNet normalization
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)
    return img

# =========================
# SIMPLE MEDICAL INFO
# =========================
def get_info(p_type):
    if p_type == "Bacterial Pneumonia":
        return ("Antibiotics (amoxicillin, azithromycin)",
                "Hospital care & supportive therapy")
    elif p_type == "Viral Pneumonia":
        return ("Antivirals if required + supportive care",
                "Rest, hydration & oxygen support")
    else:
        return ("Not required", "Maintain lung health")

# =========================
# LLM REPORT (UNCHANGED)
# =========================
def generate_llm_report(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi3", "prompt": prompt, "stream": False},
            timeout=60
        )
        return response.json()["response"]
    except:
        return "LLM not available. Ensure Ollama is running."

# =========================
# CLEAN TEXT FOR PDF
# =========================
def clean_text(text):
    return (
        text.replace("–", "-")
            .replace("—", "-")
            .replace("•", "-")
            .encode("latin-1", "ignore")
            .decode("latin-1")
    )

# =========================
# PDF GENERATION (UNCHANGED)
# =========================
def create_pdf(report):
    pdf = FPDF()
    pdf.add_page()
    pdf.rect(5, 5, 200, 287)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI CLINICAL REPORT", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    report = clean_text(report)

    for line in report.split("\n"):
        pdf.multi_cell(0, 8, line)
        pdf.ln(1)

    return pdf.output(dest="S").encode("latin-1")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    img = preprocess_image(img_array)

    # ===== VALIDATOR =====
    val_pred = validator_model.predict(img, verbose=0)[0][0]

    if val_pred < 0.4:   # improved threshold
        st.error("Not a chest X-ray image")
        st.stop()

    # ===== MULTICLASS DETECTION =====
    preds = multiclass_model.predict(img, verbose=0)[0]

    # normalize probabilities (safety)
    preds = preds / np.sum(preds)

    class_index = np.argmax(preds)
    confidence = float(preds[class_index]) * 100

    diagnosis = CLASS_NAMES[class_index]
    medication, treatment = get_info(diagnosis)

    # =========================
    # DASHBOARD LAYOUT
    # =========================
    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Uploaded X-ray</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Detection Summary</div>', unsafe_allow_html=True)

        if diagnosis == "Normal":
            st.markdown('<p class="status-normal">✔ Lungs appear normal</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-alert">⚠ Pneumonia Detected</p>', unsafe_allow_html=True)

        st.write(f"**Diagnosis:** {diagnosis}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        if diagnosis != "Normal":
            st.write(f"**Type:** {diagnosis}")
            st.write(f"**Medication:** {medication}")
            st.write(f"**Treatment:** {treatment}")

        st.progress(int(confidence))
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # LLM REPORT (UNCHANGED)
    # =========================
    prompt = f"""
Write a concise clinical report for physician review.

Diagnosis: {diagnosis}

Include sections:
Diagnosis
Findings
Treatment
Medication
Prevention

Each section must contain exactly 2 short lines.
"""

    report = generate_llm_report(prompt)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Clinical Report</div>', unsafe_allow_html=True)
    st.text(report)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # PDF DOWNLOAD
    # =========================
    pdf = create_pdf(report)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.download_button(
        label="📄 Download Clinical Report",
        data=pdf,
        file_name="AI_Clinical_Report.pdf",
        mime="application/pdf"
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">AI-assisted clinical screening tool. Professional medical evaluation required.</div>', unsafe_allow_html=True)