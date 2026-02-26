import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# ======================
# SETTINGS
# ======================
IMG_SIZE = 224

VALIDATOR_PATH = "models/xray_validator.h5"
PNEUMONIA_PATH = "models/densenet_best.h5"

# ======================
# LOAD MODELS
# ======================
@st.cache_resource(show_spinner=False)
def load_models():
    validator = tf.keras.models.load_model(VALIDATOR_PATH, compile=False)
    pneumonia = tf.keras.models.load_model(PNEUMONIA_PATH, compile=False)
    return validator, pneumonia

validator_model, pneumonia_model = load_models()

# ======================
# PAGE DESIGN
# ======================
st.set_page_config(page_title="AI PneumoScan", layout="centered")

st.title("🫁 AI Pneumonia Detection")
st.write("Upload a chest X-ray to detect pneumonia using AI.")

st.markdown("---")

# ======================
# FILE UPLOAD
# ======================
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    st.write("🔍 Analyzing image...")

    # STEP 1: Validate X-ray
    val_pred = validator_model.predict(img)[0][0]

    if val_pred < 0.5:
        st.error("❌ Not a Chest X-ray. Please upload a valid X-ray.")
    else:
        st.success("✅ Chest X-ray detected")

        # STEP 2: Pneumonia detection
        pneu_pred = pneumonia_model.predict(img)[0][0]
        confidence = pneu_pred * 100

        if pneu_pred > 0.5:
            st.error(f"🦠 Pneumonia Detected ({confidence:.2f}%)")
        else:
            st.success(f"🫁 Normal ({100 - confidence:.2f}%)")

        st.warning("⚠ This AI tool assists screening only. Consult a doctor.")