from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path("results/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Verified final benchmark results
methods = ["CPU", "Metal"]

preprocessing_time = [0.0530, 0.0082]

character_similarity = [69.86, 69.80]
word_similarity = [67.59, 67.55]

cpu_preprocessing = 0.0530
metal_preprocessing = 0.0082

cpu_ocr = 3.8253
metal_ocr = 3.8836


# ============================================================
# 1. PREPROCESSING SPEED
# ============================================================

plt.figure(figsize=(7, 5))

plt.bar(
    methods,
    preprocessing_time
)

plt.ylabel("Time (seconds)")
plt.title("CPU vs Metal Preprocessing Time")

for i, value in enumerate(preprocessing_time):
    plt.text(
        i,
        value,
        f"{value:.4f}s",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "preprocessing_speed.png",
    dpi=200
)

plt.close()


# ============================================================
# 2. OCR ACCURACY
# ============================================================

x = [0, 1]
width = 0.35

plt.figure(figsize=(7, 5))

plt.bar(
    [i - width / 2 for i in x],
    character_similarity,
    width,
    label="Character similarity"
)

plt.bar(
    [i + width / 2 for i in x],
    word_similarity,
    width,
    label="Word similarity"
)

plt.xticks(
    x,
    methods
)

plt.ylabel("Similarity (%)")
plt.title("CPU vs Metal OCR Accuracy")
plt.legend()

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ocr_accuracy.png",
    dpi=200
)

plt.close()


# ============================================================
# 3. END-TO-END RUNTIME BREAKDOWN
# ============================================================

plt.figure(figsize=(8, 5))

x = [0, 1]

plt.bar(
    x,
    [cpu_preprocessing, metal_preprocessing],
    label="Preprocessing"
)

plt.bar(
    x,
    [cpu_ocr, metal_ocr],
    bottom=[
        cpu_preprocessing,
        metal_preprocessing
    ],
    label="OCR"
)

plt.xticks(
    x,
    ["CPU", "Metal"]
)

plt.ylabel("Time (seconds)")
plt.title("End-to-End OCR Runtime Breakdown")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "runtime_breakdown.png",
    dpi=200
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("VISUALIZATIONS CREATED")
print("=" * 60)

print(
    f"Saved: "
    f"{OUTPUT_DIR / 'preprocessing_speed.png'}"
)

print(
    f"Saved: "
    f"{OUTPUT_DIR / 'ocr_accuracy.png'}"
)

print(
    f"Saved: "
    f"{OUTPUT_DIR / 'runtime_breakdown.png'}"
)

speedup = cpu_preprocessing / metal_preprocessing

print(
    f"\nMetal preprocessing speedup: "
    f"{speedup:.2f}x"
)

print("=" * 60)
