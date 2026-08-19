import time
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image


PDF_DIR = Path("data/raw/gazettes")


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


def benchmark_preprocessing(pdf_name):

    pdf_path = PDF_DIR / pdf_name
    doc = fitz.open(pdf_path)

    total_render_time = 0
    total_preprocessing_time = 0
    total_ocr_time = 0

    all_text = []

    print("=" * 70)
    print(f"CPU PREPROCESSING BENCHMARK: {pdf_name}")
    print("=" * 70)
    print(f"Pages: {len(doc)}\n")

    for page_number, page in enumerate(doc, start=1):

        # -------------------------
        # Rendering
        # -------------------------

        start = time.perf_counter()

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        render_time = time.perf_counter() - start
        total_render_time += render_time

        # Convert PIL image to NumPy
        image_array = np.array(image)

        # -------------------------
        # CPU preprocessing
        # -------------------------

        start = time.perf_counter()

        processed = preprocess_image(
            image_array
        )

        preprocessing_time = time.perf_counter() - start
        total_preprocessing_time += preprocessing_time

        # -------------------------
        # OCR
        # -------------------------

        start = time.perf_counter()

        text = pytesseract.image_to_string(
            processed,
            lang="hin+eng"
        )

        ocr_time = time.perf_counter() - start
        total_ocr_time += ocr_time

        all_text.append(
            f"\n===== PAGE {page_number} =====\n\n{text}"
        )

        print(
            f"Page {page_number:>3}/{len(doc)}: "
            f"render={render_time:.3f}s | "
            f"preprocess={preprocessing_time:.3f}s | "
            f"ocr={ocr_time:.3f}s | "
            f"chars={len(text)}"
        )

    # -------------------------
    # Save OCR output
    # -------------------------

    output_dir = Path("data/raw/ocr")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir /
        f"{Path(pdf_name).stem}_preprocessed.txt"
    )

    output_path.write_text(
        "\n".join(all_text),
        encoding="utf-8"
    )

    # -------------------------
    # Summary
    # -------------------------

    total_time = (
        total_render_time
        + total_preprocessing_time
        + total_ocr_time
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total pages:          {len(doc)}")
    print(f"Rendering time:       {total_render_time:.3f}s")
    print(f"Preprocessing time:   {total_preprocessing_time:.3f}s")
    print(f"OCR time:             {total_ocr_time:.3f}s")
    print(f"Total processing:     {total_time:.3f}s")
    print(f"Average/page:         {total_time / len(doc):.3f}s")
    print(f"\nSaved OCR output to:  {output_path}")


if __name__ == "__main__":
    benchmark_preprocessing("275564.pdf")