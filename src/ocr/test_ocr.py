import fitz
import pytesseract
from PIL import Image
from pathlib import Path


PDF_PATH = Path("data/raw/gazettes/275564.pdf")


def ocr_first_page():
    doc = fitz.open(PDF_PATH)

    page = doc[0]

    # Render PDF page at high resolution
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    # OCR Hindi + English
    text = pytesseract.image_to_string(
        image,
        lang="hin+eng"
    )

    print("=" * 70)
    print("OCR RESULT")
    print("=" * 70)
    print(text)


if __name__ == "__main__":
    ocr_first_page()