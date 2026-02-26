import numpy as np
import cv2
import tensorflow as tf

# ==============================
# SETTINGS
# ==============================
IMG_SIZE = 224

VALIDATOR_MODEL_PATH = "models/xray_validator.h5"
PNEUMONIA_MODEL_PATH = "models/densenet_best.h5"

# ==============================
# LOAD MODELS
# ==============================
print("\nLoading models...\n")

validator_model = tf.keras.models.load_model(
    VALIDATOR_MODEL_PATH,
    compile=False
)

pneumonia_model = tf.keras.models.load_model(
    PNEUMONIA_MODEL_PATH,
    compile=False
)

print("✅ Models loaded successfully\n")

# ==============================
# IMAGE PREPROCESS FUNCTION
# ==============================
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not found")
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# ==============================
# FINAL PREDICTION PIPELINE
# ==============================
def predict(image_path):

    print(f"\n📷 Processing: {image_path}\n")

    img = preprocess_image(image_path)
    if img is None:
        return

    # STEP 1️⃣ Validate X-ray
    val_pred = validator_model.predict(img)[0][0]

    if val_pred < 0.5:
        print("❌ Not a Chest X-ray")
        print("⚠ Please upload a valid chest X-ray image\n")
        return

    print("✅ Chest X-ray detected")

    # STEP 2️⃣ Pneumonia detection
    pneu_pred = pneumonia_model.predict(img)[0][0]
    confidence = pneu_pred * 100

    print("\n🩺 Diagnosis Result:\n")

    if pneu_pred > 0.5:
        print(f"🦠 PNEUMONIA DETECTED")
        print(f"Confidence: {confidence:.2f}%")
    else:
        print(f"🫁 NORMAL")
        print(f"Confidence: {100 - confidence:.2f}%")

    print("\n⚠ This AI tool assists screening only.")
    print("⚠ Always consult a medical professional.\n")

# ==============================
# TEST
# ==============================# ## # predict("test.jpg")

predict("sanjeevi.jpeg")
