import cv2
import numpy as np


def preprocess_image(image):
    """
    CPU-based image preprocessing for OCR.

    Steps:
    1. Convert to grayscale
    2. Denoise
    3. Improve contrast
    4. Apply adaptive thresholding
    """

    # PIL/NumPy RGB image -> OpenCV BGR
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    # 1. Grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # 2. Denoising
    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        10,
        7,
        21
    )

    # 3. Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(denoised)

    # 4. Adaptive thresholding
    thresholded = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return thresholded