from utils.xray_validator import is_chest_xray
import cv2

# 👉 provide test image path here
image_path = "dataset/test/NORMAL/IM-0001-0001.jpeg"

# read image
image = cv2.imread(image_path)

# check if image loaded
if image is None:
    print("❌ Image not found. Check path.")
    exit()

# Validate X-ray
if not is_chest_xray(image):
    print("❌ Invalid Image: Please upload a chest X-ray")
else:
    print("✅ Valid Chest X-ray")