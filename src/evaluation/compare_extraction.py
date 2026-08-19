from pathlib import Path
import difflib


TEXT_DIR = Path("data/raw/text")
OCR_DIR = Path("data/raw/ocr")


def compare_files(pdf_id):
    text_path = TEXT_DIR / f"{pdf_id}.txt"
    ocr_path = OCR_DIR / f"{pdf_id}.txt"

    if not text_path.exists():
        print(f"Native text not found: {text_path}")
        return

    if not ocr_path.exists():
        print(f"OCR text not found: {ocr_path}")
        return

    native_text = text_path.read_text(encoding="utf-8", errors="replace")
    ocr_text = ocr_path.read_text(encoding="utf-8", errors="replace")

    print("=" * 70)
    print(f"COMPARISON: {pdf_id}")
    print("=" * 70)

    print(f"Native extraction characters: {len(native_text)}")
    print(f"OCR characters:              {len(ocr_text)}")

    print("\n" + "-" * 70)
    print("NATIVE EXTRACTION")
    print("-" * 70)
    print(native_text[:2000])

    print("\n" + "-" * 70)
    print("OCR EXTRACTION")
    print("-" * 70)
    print(ocr_text[:2000])

    print("\n" + "-" * 70)
    print("DIFFERENCES")
    print("-" * 70)

    diff = difflib.unified_diff(
        native_text.splitlines(),
        ocr_text.splitlines(),
        fromfile="native",
        tofile="ocr",
        n=2
    )

    for line in list(diff)[:100]:
        print(line)


if __name__ == "__main__":
    compare_files("275564")