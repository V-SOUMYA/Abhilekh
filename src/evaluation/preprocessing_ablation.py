import time
import difflib
from pathlib import Path

import fitz
import cv2
import numpy as np
import pytesseract


PDF_DIR = Path("data/raw/gazettes")
TEXT_DIR = Path("data/raw/text")
OCR_DIR = Path("data/raw/ocr")
ABLATION_DIR = OCR_DIR / "ablation"

PDF_ID = "275564"


def render_page(page):
    pix = page.get_pixmap(
        matrix=fitz.Matrix(2, 2),
        alpha=False
    )

    return np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(pix.height, pix.width, 3)


# ---------------------------------------------------------
# PREPROCESSING METHODS
# ---------------------------------------------------------

def baseline(image):
    return image


def grayscale(image):
    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )


def grayscale_upscale(image):
    gray = grayscale(image)

    return cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )


def grayscale_denoise(image):
    gray = grayscale(image)

    return cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )


def grayscale_binarize(image):
    gray = grayscale(image)

    return cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]


def full_preprocessing(image):
    gray = grayscale(image)

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    binary = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return cv2.resize(
        binary,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )


METHODS = {
    "baseline": baseline,
    "grayscale": grayscale,
    "grayscale_upscale": grayscale_upscale,
    "grayscale_denoise": grayscale_denoise,
    "grayscale_binarize": grayscale_binarize,
    "full_preprocessing": full_preprocessing,
}


# ---------------------------------------------------------
# SIMILARITY METRICS
# ---------------------------------------------------------

def character_similarity(reference, candidate):
    return difflib.SequenceMatcher(
        None,
        reference,
        candidate
    ).ratio()


def word_similarity(reference, candidate):
    reference_words = reference.split()
    candidate_words = candidate.split()

    return difflib.SequenceMatcher(
        None,
        reference_words,
        candidate_words
    ).ratio()


# ---------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------

def run_experiment():

    pdf_path = PDF_DIR / f"{PDF_ID}.pdf"
    reference_path = TEXT_DIR / f"{PDF_ID}.txt"

    reference = reference_path.read_text(
        encoding="utf-8"
    )

    doc = fitz.open(pdf_path)

    ABLATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print(f"PREPROCESSING ABLATION: {PDF_ID}")
    print("=" * 70)
    print(f"Pages: {len(doc)}")
    print()

    results = {}

    for method_name, method in METHODS.items():

        total_preprocessing = 0
        total_ocr = 0
        all_text = []

        print("-" * 70)
        print(f"METHOD: {method_name}")
        print("-" * 70)

        for page_number, page in enumerate(
            doc,
            start=1
        ):

            image = render_page(page)

            # -------------------------
            # Preprocessing
            # -------------------------

            start = time.perf_counter()

            processed = method(image)

            preprocessing_time = (
                time.perf_counter() - start
            )

            total_preprocessing += preprocessing_time

            # -------------------------
            # OCR
            # -------------------------

            start = time.perf_counter()

            text = pytesseract.image_to_string(
                processed,
                lang="hin+eng"
            )

            ocr_time = time.perf_counter() - start

            total_ocr += ocr_time

            all_text.append(
                f"\n===== PAGE {page_number} =====\n\n{text}"
            )

            print(
                f"Page {page_number}/{len(doc)}: "
                f"prep={preprocessing_time:.3f}s | "
                f"ocr={ocr_time:.3f}s | "
                f"chars={len(text)}"
            )

        # -------------------------
        # Save OCR output
        # -------------------------

        output_path = (
            ABLATION_DIR
            / f"{PDF_ID}_{method_name}.txt"
        )

        candidate = "\n".join(all_text)

        output_path.write_text(
            candidate,
            encoding="utf-8"
        )

        # -------------------------
        # Accuracy
        # -------------------------

        char_similarity = character_similarity(
            reference,
            candidate
        )

        word_similarity_value = word_similarity(
            reference,
            candidate
        )

        total_time = (
            total_preprocessing
            + total_ocr
        )

        results[method_name] = {
            "preprocessing": total_preprocessing,
            "ocr": total_ocr,
            "total": total_time,
            "chars": len(candidate),
            "char_similarity": char_similarity,
            "word_similarity": word_similarity_value,
        }

        print(
            f"Total preprocessing: {total_preprocessing:.3f}s"
        )

        print(
            f"Total OCR:           {total_ocr:.3f}s"
        )

        print(
            f"Total time:          {total_time:.3f}s"
        )

        print(
            f"Character similarity: "
            f"{char_similarity * 100:.2f}%"
        )

        print(
            f"Word similarity:      "
            f"{word_similarity_value * 100:.2f}%"
        )

        print(
            f"Saved to:             {output_path}"
        )

        print()

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print(
        f"{'Method':<25}"
        f"{'Prep(s)':>10}"
        f"{'OCR(s)':>10}"
        f"{'Total(s)':>11}"
        f"{'Char Sim':>12}"
        f"{'Word Sim':>12}"
    )

    print("-" * 100)

    for method_name, result in results.items():

        print(
            f"{method_name:<25}"
            f"{result['preprocessing']:>10.3f}"
            f"{result['ocr']:>10.3f}"
            f"{result['total']:>11.3f}"
            f"{result['char_similarity'] * 100:>11.2f}%"
            f"{result['word_similarity'] * 100:>11.2f}%"
        )


if __name__ == "__main__":
    run_experiment()