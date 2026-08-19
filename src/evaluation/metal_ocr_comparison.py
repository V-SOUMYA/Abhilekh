import time
from pathlib import Path
from difflib import SequenceMatcher

import fitz
import Metal
import numpy as np
import pytesseract


PDF_DIR = Path("data/raw/gazettes")
REFERENCE_DIR = Path("data/raw/ocr")
OUTPUT_DIR = Path("data/raw/ocr/metal_comparison")


def similarity(reference, prediction):
    """
    Calculate similarity between reference and OCR output.
    """
    return SequenceMatcher(
        None,
        reference,
        prediction
    ).ratio() * 100


def word_similarity(reference, prediction):
    """
    Calculate word-level similarity.
    """
    reference_words = reference.split()
    prediction_words = prediction.split()

    return SequenceMatcher(
        None,
        reference_words,
        prediction_words
    ).ratio() * 100


def cpu_preprocess(image):
    """
    GPU-friendly CPU preprocessing:
    RGB -> grayscale -> 3x3 denoise -> contrast.
    """

    gray = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    ).astype(np.uint8)

    padded = np.pad(
        gray,
        1,
        mode="edge"
    )

    denoised = (
        padded[:-2, :-2].astype(np.uint16)
        + padded[:-2, 1:-1].astype(np.uint16)
        + padded[:-2, 2:].astype(np.uint16)
        + padded[1:-1, :-2].astype(np.uint16)
        + padded[1:-1, 1:-1].astype(np.uint16)
        + padded[1:-1, 2:].astype(np.uint16)
        + padded[2:, :-2].astype(np.uint16)
        + padded[2:, 1:-1].astype(np.uint16)
        + padded[2:, 2:].astype(np.uint16)
    ) // 9

    denoised = denoised.astype(np.uint8)

    enhanced = (
        (denoised.astype(np.float32) - 128.0) * 1.2
        + 128.0
    )

    return np.clip(
        enhanced,
        0,
        255
    ).astype(np.uint8)


def load_metal_shader(device):

    shader_path = (
        Path(__file__).parent.parent
        / "benchmark"
        / "metal"
        / "full_preprocessing.metal"
    )

    source = shader_path.read_text()

    library, error = (
        device.newLibraryWithSource_options_error_(
            source,
            None,
            None
        )
    )

    if error:
        raise RuntimeError(
            f"Metal shader compilation failed: {error}"
        )

    function = library.newFunctionWithName_(
        "full_preprocessing"
    )

    pipeline, error = (
        device.newComputePipelineStateWithFunction_error_(
            function,
            None
        )
    )

    if error:
        raise RuntimeError(
            f"Pipeline creation failed: {error}"
        )

    return pipeline


def metal_preprocess(
    device,
    pipeline,
    image
):

    height, width, _ = image.shape

    image = np.ascontiguousarray(
        image,
        dtype=np.uint8
    )

    output = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    input_buffer = (
        device.newBufferWithBytes_length_options_(
            image.tobytes(),
            image.nbytes,
            Metal.MTLResourceStorageModeShared
        )
    )

    output_buffer = (
        device.newBufferWithLength_options_(
            output.nbytes,
            Metal.MTLResourceStorageModeShared
        )
    )

    width_buffer = (
        device.newBufferWithBytes_length_options_(
            np.uint32(width).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared
        )
    )

    height_buffer = (
        device.newBufferWithBytes_length_options_(
            np.uint32(height).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared
        )
    )

    command_queue = device.newCommandQueue()

    start = time.perf_counter()

    command_buffer = command_queue.commandBuffer()

    encoder = command_buffer.computeCommandEncoder()

    encoder.setComputePipelineState_(pipeline)

    encoder.setBuffer_offset_atIndex_(
        input_buffer,
        0,
        0
    )

    encoder.setBuffer_offset_atIndex_(
        output_buffer,
        0,
        1
    )

    encoder.setBuffer_offset_atIndex_(
        width_buffer,
        0,
        2
    )

    encoder.setBuffer_offset_atIndex_(
        height_buffer,
        0,
        3
    )

    threads = Metal.MTLSizeMake(
        16,
        16,
        1
    )

    groups = Metal.MTLSizeMake(
        (width + 15) // 16,
        (height + 15) // 16,
        1
    )

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        groups,
        threads
    )

    encoder.endEncoding()

    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    elapsed = time.perf_counter() - start

    contents = output_buffer.contents()

    result = np.frombuffer(
        contents.as_buffer(output.nbytes),
        dtype=np.uint8
    ).copy()

    result = result.reshape(
        height,
        width
    )

    return result, elapsed


def main():

    pdf_name = "275564.pdf"

    print("=" * 70)
    print("CPU vs METAL: OCR ACCURACY COMPARISON")
    print("=" * 70)

    device = Metal.MTLCreateSystemDefaultDevice()

    if device is None:
        raise RuntimeError(
            "Metal GPU unavailable"
        )

    print(f"GPU: {device.name()}")

    doc = fitz.open(
        PDF_DIR / pdf_name
    )

    print(f"Pages: {len(doc)}")

    pipeline = load_metal_shader(device)

    print("Metal shader: READY\n")

    # --------------------------------------------------
    # Load reference text
    # --------------------------------------------------

    reference_path = (
        REFERENCE_DIR / "275564.txt"
    )

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference OCR not found: {reference_path}"
        )

    reference_text = reference_path.read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cpu_text = []
    metal_text = []

    total_cpu_preprocess = 0.0
    total_metal_preprocess = 0.0

    total_cpu_ocr = 0.0
    total_metal_ocr = 0.0

    for page_number, page in enumerate(
        doc,
        start=1
    ):

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False
        )

        image = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        ).reshape(
            pix.height,
            pix.width,
            3
        )

        # -----------------------------------------
        # CPU preprocessing
        # -----------------------------------------

        cpu_start = time.perf_counter()

        cpu_image = cpu_preprocess(image)

        cpu_preprocess_time = (
            time.perf_counter() - cpu_start
        )

        # -----------------------------------------
        # Metal preprocessing
        # -----------------------------------------

        metal_image, metal_preprocess_time = (
            metal_preprocess(
                device,
                pipeline,
                image
            )
        )

        # -----------------------------------------
        # CPU OCR
        # -----------------------------------------

        cpu_ocr_start = time.perf_counter()

        cpu_page_text = pytesseract.image_to_string(
            cpu_image,
            lang="hin+eng"
        )

        cpu_ocr_time = (
            time.perf_counter() - cpu_ocr_start
        )

        # -----------------------------------------
        # Metal OCR
        # -----------------------------------------

        metal_ocr_start = time.perf_counter()

        metal_page_text = pytesseract.image_to_string(
            metal_image,
            lang="hin+eng"
        )

        metal_ocr_time = (
            time.perf_counter() - metal_ocr_start
        )

        cpu_text.append(cpu_page_text)
        metal_text.append(metal_page_text)

        total_cpu_preprocess += cpu_preprocess_time
        total_metal_preprocess += metal_preprocess_time

        total_cpu_ocr += cpu_ocr_time
        total_metal_ocr += metal_ocr_time

        difference = np.abs(
            cpu_image.astype(np.int16)
            - metal_image.astype(np.int16)
        )

        print(
            f"Page {page_number}/{len(doc)}: "
            f"CPU prep={cpu_preprocess_time:.4f}s | "
            f"Metal prep={metal_preprocess_time:.4f}s | "
            f"CPU OCR={cpu_ocr_time:.4f}s | "
            f"Metal OCR={metal_ocr_time:.4f}s | "
            f"Max diff={difference.max()}"
        )

    # --------------------------------------------------
    # Combine OCR text
    # --------------------------------------------------

    cpu_text = "\n".join(cpu_text)
    metal_text = "\n".join(metal_text)

    # Save OCR outputs
    cpu_output_path = (
        OUTPUT_DIR / "275564_cpu.txt"
    )

    metal_output_path = (
        OUTPUT_DIR / "275564_metal.txt"
    )

    cpu_output_path.write_text(
        cpu_text,
        encoding="utf-8"
    )

    metal_output_path.write_text(
        metal_text,
        encoding="utf-8"
    )

    # --------------------------------------------------
    # Timing summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)

    print(
        f"CPU preprocessing:     "
        f"{total_cpu_preprocess:.4f}s"
    )

    print(
        f"Metal preprocessing:   "
        f"{total_metal_preprocess:.4f}s"
    )

    print(
        f"CPU OCR:               "
        f"{total_cpu_ocr:.4f}s"
    )

    print(
        f"Metal OCR:             "
        f"{total_metal_ocr:.4f}s"
    )

    cpu_total = (
        total_cpu_preprocess
        + total_cpu_ocr
    )

    metal_total = (
        total_metal_preprocess
        + total_metal_ocr
    )

    print(
        f"\nCPU total:             "
        f"{cpu_total:.4f}s"
    )

    print(
        f"Metal total:           "
        f"{metal_total:.4f}s"
    )

    if metal_total > 0:
        print(
            f"Overall speedup:      "
            f"{cpu_total / metal_total:.2f}x"
        )

    # --------------------------------------------------
    # Accuracy comparison
    # --------------------------------------------------

    cpu_char_similarity = similarity(
        reference_text,
        cpu_text
    )

    metal_char_similarity = similarity(
        reference_text,
        metal_text
    )

    cpu_word_similarity = word_similarity(
        reference_text,
        cpu_text
    )

    metal_word_similarity = word_similarity(
        reference_text,
        metal_text
    )

    print("\n" + "=" * 70)
    print("OCR ACCURACY")
    print("=" * 70)

    print(
        f"CPU character similarity:   "
        f"{cpu_char_similarity:.2f}%"
    )

    print(
        f"Metal character similarity: "
        f"{metal_char_similarity:.2f}%"
    )

    print(
        f"\nCPU word similarity:        "
        f"{cpu_word_similarity:.2f}%"
    )

    print(
        f"Metal word similarity:      "
        f"{metal_word_similarity:.2f}%"
    )

    print("\n" + "=" * 70)
    print("ACCURACY CHANGE")
    print("=" * 70)

    print(
        f"Character similarity change: "
        f"{metal_char_similarity - cpu_char_similarity:+.2f} "
        f"percentage points"
    )

    print(
        f"Word similarity change:       "
        f"{metal_word_similarity - cpu_word_similarity:+.2f} "
        f"percentage points"
    )

    print("\n" + "=" * 70)
    print("OCR OUTPUT")
    print("=" * 70)

    print(
        f"Reference characters: "
        f"{len(reference_text)}"
    )

    print(
        f"CPU OCR characters:   "
        f"{len(cpu_text)}"
    )

    print(
        f"Metal OCR characters: "
        f"{len(metal_text)}"
    )

    print("\nSaved outputs:")

    print(
        f"CPU:   {cpu_output_path}"
    )

    print(
        f"Metal: {metal_output_path}"
    )


if __name__ == "__main__":
    main()
