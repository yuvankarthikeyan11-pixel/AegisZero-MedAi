import cv2
import numpy as np
import tensorflow as tf

# load trained validator model
model = tf.keras.models.load_model("models/xray_validator.h5")

IMG_SIZE = 224


def check_image(path):
    print("\nChecking:", path)

    img = cv2.imread(path)

    if img is None:
        print("❌ Image not found or corrupted")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        print("✅ Chest X-ray detected")
        print(f"Confidence: {prediction*100:.2f}%")
    else:
        print("❌ Not a chest X-ray")
        print(f"Confidence: {(1-prediction)*100:.2f}%")


# ===== TEST IMAGES =====

# ✔ should PASS
check_image(r"A:\aegishack\dataset\test\NORMAL\IM-0001-0001.jpeg")

# ❌ should FAIL (use wallpaper/selfie)
check_image(r"A:\aegishack\sample.jpg")