import fitz
import pytesseract
from PIL import Image
from pathlib import Path


PDF_DIR = Path("data/raw/gazettes")
OCR_DIR = Path("data/raw/ocr")

OCR_DIR.mkdir(parents=True, exist_ok=True)


def ocr_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        print(
            f"OCR processing: {pdf_path.name} "
            f"- page {page_number}/{len(doc)}"
        )

        # Render PDF page at 2x resolution
        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        # Hindi + English OCR
        text = pytesseract.image_to_string(
            image,
            lang="hin+eng"
        )

        pages.append(
            f"\n===== PAGE {page_number} =====\n\n{text}"
        )

    return "\n".join(pages)


def main():

    pdf_path = PDF_DIR / "275564.pdf"

    text = ocr_pdf(pdf_path)

    output_path = OCR_DIR / "275564.txt"

    output_path.write_text(
        text,
        encoding="utf-8"
    )

    print("\nOCR complete.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()