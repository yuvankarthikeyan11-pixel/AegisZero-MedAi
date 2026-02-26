import cv2
import numpy as np

def is_chest_xray(image):
    """
    Returns True if image resembles a chest X-ray.
    """

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check if image is mostly grayscale (low color variance)
    if len(image.shape) == 3:
        b, g, r = cv2.split(image)
        color_variance = np.mean([
            np.std(b - g),
            np.std(b - r),
            np.std(g - r)
        ])
        if color_variance > 8:   # colorful image → reject
            return False

    # check brightness distribution
    mean_intensity = np.mean(gray)

    if mean_intensity < 40 or mean_intensity > 220:
        return False

    # check contrast (bones & lungs create contrast)
    contrast = gray.std()
    if contrast < 30:
        return False

    return True