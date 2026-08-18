import fitz
from pathlib import Path


PDF_DIR = Path("data/raw/gazettes")


def inspect_pdf(pdf_path):
    """Inspect a Gazette PDF and report basic text information."""

    doc = fitz.open(pdf_path)

    print("=" * 70)
    print(f"FILE: {pdf_path.name}")
    print(f"PAGES: {len(doc)}")

    total_chars = 0

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text("text").strip()
        char_count = len(text)
        total_chars += char_count

        print(f"\nPage {page_number}")
        print(f"Extracted characters: {char_count}")

        if char_count > 100:
            print("Status: Text layer detected")
        else:
            print("Status: Image-heavy / possible scanned page")

        if text:
            preview = text[:300].replace("\n", " ")
            print(f"Preview: {preview}")

    print(f"\nTOTAL EXTRACTED CHARACTERS: {total_chars}")
    print("=" * 70)


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {PDF_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s).\n")

    for pdf_path in pdf_files:
        inspect_pdf(pdf_path)


if __name__ == "__main__":
    main()