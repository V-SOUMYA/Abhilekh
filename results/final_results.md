# Final Experimental Results

## Dataset

- PDF: `275564.pdf`
- Pages: 3
- Platform: Apple M4 MacBook Air
- GPU backend: Apple Metal
- OCR engine: Tesseract
- OCR languages: `hin+eng`

## Baseline OCR

| Metric | Result |
|---|---:|
| Character similarity | 75.04% |
| Word similarity | 70.22% |
| OCR characters | 5751 |

## CPU vs Metal: Same GPU-Friendly Preprocessing

| Metric | CPU | Metal |
|---|---:|---:|
| Preprocessing time | 0.0530s | **0.0082s** |
| OCR time | 3.8253s | 3.8836s |
| Total time | **3.8784s** | 3.8918s |
| Character similarity | 69.86% | 69.80% |
| Word similarity | 67.59% | 67.55% |

## GPU Preprocessing Speedup

CPU preprocessing:

`0.0530s`

Metal preprocessing:

`0.0082s`

Approximate preprocessing speedup:

**6.46x**

## Accuracy Difference

Metal compared with CPU:

- Character similarity: **-0.06 percentage points**
- Word similarity: **-0.04 percentage points**

The preprocessing outputs were also nearly identical:

- Maximum pixel difference: **2**
- Mean pixel difference: **0.060346**

## End-to-End Performance

CPU:

`3.8784s`

Metal:

`3.8918s`

Overall speedup:

**1.00x**

## Conclusion

Metal substantially accelerates the preprocessing stage, achieving approximately 6.5x faster preprocessing than the equivalent CPU implementation.

The Metal implementation produces nearly identical image output and maintains essentially the same OCR accuracy.

However, the overall OCR pipeline does not become faster because Tesseract OCR dominates the runtime. Therefore, GPU acceleration of preprocessing alone does not provide meaningful end-to-end speedup for this workload.

This demonstrates that GPU acceleration can be highly effective for the preprocessing stage while still providing limited overall benefit when the downstream OCR engine remains the primary bottleneck.
