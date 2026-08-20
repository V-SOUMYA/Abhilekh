from pathlib import Path
import difflib


TEXT_DIR = Path("data/raw/text")
OCR_DIR = Path("data/raw/ocr")
METAL_DIR = OCR_DIR / "metal_comparison"

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


def evaluate_method(reference, candidate):
    return {
        "characters": len(candidate),
        "char_similarity": character_similarity(
            reference,
            candidate
        ),
        "word_similarity": word_similarity(
            reference,
            candidate
        )
    }


def main():

    print("=" * 70)
    print(f"FINAL OCR EVALUATION: {PDF_ID}")
    print("=" * 70)

    # --------------------------------------------------
    # Load reference
    # --------------------------------------------------

    reference_path = (
        TEXT_DIR / f"{PDF_ID}.txt"
    )

    reference = load_text(
        reference_path
    )

    # --------------------------------------------------
    # Load OCR outputs
    # --------------------------------------------------

    baseline_path = (
        OCR_DIR / f"{PDF_ID}.txt"
    )

    cpu_path = (
        OCR_DIR / f"{PDF_ID}_preprocessed.txt"
    )

    metal_path = (
        METAL_DIR / f"{PDF_ID}_metal.txt"
    )

    baseline = load_text(
        baseline_path
    )

    cpu = load_text(
        cpu_path
    )

    metal = load_text(
        metal_path
    )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    results = {
        "Baseline": evaluate_method(
            reference,
            baseline
        ),
        "CPU preprocessing": evaluate_method(
            reference,
            cpu
        ),
        "Metal preprocessing": evaluate_method(
            reference,
            metal
        )
    }

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(
        f"\nReference characters: "
        f"{len(reference)}"
    )

    print("\n" + "-" * 70)

    print(
        f"{'METHOD':<24}"
        f"{'CHAR SIM':>15}"
        f"{'WORD SIM':>15}"
        f"{'CHARACTERS':>15}"
    )

    print("-" * 70)

    for method, result in results.items():

        print(
            f"{method:<24}"
            f"{result['char_similarity'] * 100:>14.2f}%"
            f"{result['word_similarity'] * 100:>14.2f}%"
            f"{result['characters']:>15}"
        )

    # --------------------------------------------------
    # Compare against baseline
    # --------------------------------------------------

    baseline_result = results["Baseline"]

    print("\n" + "=" * 70)
    print("CHANGE FROM BASELINE")
    print("=" * 70)

    for method in [
        "CPU preprocessing",
        "Metal preprocessing"
    ]:

        result = results[method]

        char_change = (
            result["char_similarity"]
            - baseline_result["char_similarity"]
        ) * 100

        word_change = (
            result["word_similarity"]
            - baseline_result["word_similarity"]
        ) * 100

        print(f"\n{method}")

        print(
            f"Character similarity change: "
            f"{char_change:+.2f} percentage points"
        )

        print(
            f"Word similarity change:       "
            f"{word_change:+.2f} percentage points"
        )

    # --------------------------------------------------
    # CPU preprocessing vs Metal
    # --------------------------------------------------

    cpu_result = results["CPU preprocessing"]
    metal_result = results["Metal preprocessing"]

    print("\n" + "=" * 70)
    print("CPU PREPROCESSING vs METAL")
    print("=" * 70)

    print(
        f"Character similarity difference: "
        f"{(metal_result['char_similarity'] - cpu_result['char_similarity']) * 100:+.2f} "
        f"percentage points"
    )

    print(
        f"Word similarity difference:       "
        f"{(metal_result['word_similarity'] - cpu_result['word_similarity']) * 100:+.2f} "
        f"percentage points"
    )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
