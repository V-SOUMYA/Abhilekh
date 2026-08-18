import fitz
from pathlib import Path


PDF_DIR = Path("data/raw/gazettes")
OUTPUT_DIR = Path("data/raw/text")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")

        pages.append(
            f"\n===== PAGE {page_number} =====\n\n{text}"
        )

    return "\n".join(pages)


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        text = extract_pdf_text(pdf_path)

        output_file = OUTPUT_DIR / f"{pdf_path.stem}.txt"

        output_file.write_text(
            text,
            encoding="utf-8"
        )

        print(f"Extracted: {pdf_path.name}")
        print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()