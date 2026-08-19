import time
from pathlib import Path

import fitz
import pytesseract
from PIL import Image


PDF_DIR = Path("data/raw/gazettes")


def benchmark_ocr(pdf_name):
    pdf_path = PDF_DIR / pdf_name

    doc = fitz.open(pdf_path)

    total_render_time = 0
    total_ocr_time = 0
    total_pages = len(doc)

    print("=" * 70)
    print(f"OCR BENCHMARK: {pdf_name}")
    print("=" * 70)
    print(f"Pages: {total_pages}\n")

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

        # -------------------------
        # OCR
        # -------------------------
        start = time.perf_counter()

        text = pytesseract.image_to_string(
            image,
            lang="hin+eng"
        )

        ocr_time = time.perf_counter() - start
        total_ocr_time += ocr_time

        print(
            f"Page {page_number:>3}/{total_pages}: "
            f"render={render_time:.3f}s | "
            f"ocr={ocr_time:.3f}s | "
            f"chars={len(text)}"
        )

    total_time = total_render_time + total_ocr_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total pages:          {total_pages}")
    print(f"Rendering time:       {total_render_time:.3f}s")
    print(f"OCR time:             {total_ocr_time:.3f}s")
    print(f"Total processing:     {total_time:.3f}s")
    print(f"Average/page:         {total_time / total_pages:.3f}s")
    print(f"OCR/page:             {total_ocr_time / total_pages:.3f}s")


if __name__ == "__main__":
    benchmark_ocr("275564.pdf")