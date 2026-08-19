from pathlib import Path
import difflib


TEXT_DIR = Path("data/raw/text")
OCR_DIR = Path("data/raw/ocr")

PDF_ID = "275564"


def load_text(path):
    return path.read_text(encoding="utf-8")


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


def main():

    reference_path = TEXT_DIR / f"{PDF_ID}.txt"
    baseline_path = OCR_DIR / f"{PDF_ID}.txt"
    preprocessed_path = OCR_DIR / f"{PDF_ID}_preprocessed.txt"

    reference = load_text(reference_path)
    baseline = load_text(baseline_path)
    preprocessed = load_text(preprocessed_path)

    baseline_char_sim = character_similarity(
        reference,
        baseline
    )

    preprocessed_char_sim = character_similarity(
        reference,
        preprocessed
    )

    baseline_word_sim = word_similarity(
        reference,
        baseline
    )

    preprocessed_word_sim = word_similarity(
        reference,
        preprocessed
    )

    print("=" * 70)
    print(f"OCR ACCURACY COMPARISON: {PDF_ID}")
    print("=" * 70)

    print(f"\nReference characters:       {len(reference)}")
    print(f"Baseline OCR characters:   {len(baseline)}")
    print(f"Preprocessed characters:   {len(preprocessed)}")

    print("\n" + "-" * 70)
    print("BASELINE OCR")
    print("-" * 70)

    print(
        f"Character similarity:      "
        f"{baseline_char_sim * 100:.2f}%"
    )

    print(
        f"Word similarity:            "
        f"{baseline_word_sim * 100:.2f}%"
    )

    print("\n" + "-" * 70)
    print("PREPROCESSED OCR")
    print("-" * 70)

    print(
        f"Character similarity:      "
        f"{preprocessed_char_sim * 100:.2f}%"
    )

    print(
        f"Word similarity:            "
        f"{preprocessed_word_sim * 100:.2f}%"
    )

    char_improvement = (
        preprocessed_char_sim - baseline_char_sim
    ) * 100

    word_improvement = (
        preprocessed_word_sim - baseline_word_sim
    ) * 100

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Character similarity change: "
        f"{char_improvement:+.2f} percentage points"
    )

    print(
        f"Word similarity change:       "
        f"{word_improvement:+.2f} percentage points"
    )


if __name__ == "__main__":
    main()