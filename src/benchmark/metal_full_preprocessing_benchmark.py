import time
from pathlib import Path

import fitz
import Metal
import numpy as np


PDF_DIR = Path("data/raw/gazettes")


def cpu_full_preprocessing(image):
    """
    CPU version matching the Metal shader:

    RGB
    -> grayscale
    -> 3x3 box denoising
    -> contrast stretch
    """

    start = time.perf_counter()

    # -----------------------------------------
    # 1. RGB -> Grayscale
    # -----------------------------------------

    gray = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    ).astype(np.uint8)

    # -----------------------------------------
    # 2. 3x3 box denoising
    # -----------------------------------------

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

    # -----------------------------------------
    # 3. Contrast enhancement
    # -----------------------------------------

    enhanced = (
        (denoised.astype(np.float32) - 128.0) * 1.2
        + 128.0
    )

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(np.uint8)

    elapsed = time.perf_counter() - start

    return enhanced, elapsed


def load_shader(device):

    shader_path = (
        Path(__file__).parent
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

    if function is None:
        raise RuntimeError(
            "full_preprocessing kernel not found"
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


def metal_full_preprocessing(
    device,
    pipeline,
    image
):

    height, width, channels = image.shape

    image = np.ascontiguousarray(
        image,
        dtype=np.uint8
    )

    output = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # -----------------------------------------
    # GPU buffers
    # -----------------------------------------

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

    # -----------------------------------------
    # GPU execution
    # -----------------------------------------

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

    threads_per_group = Metal.MTLSizeMake(
        16,
        16,
        1
    )

    threadgroups = Metal.MTLSizeMake(
        (width + 15) // 16,
        (height + 15) // 16,
        1
    )

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        threadgroups,
        threads_per_group
    )

    encoder.endEncoding()

    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    elapsed = time.perf_counter() - start

    # -----------------------------------------
    # Read GPU output
    # -----------------------------------------

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

    print("=" * 64)
    print("CPU vs METAL: FULL GPU-FRIENDLY PREPROCESSING")
    print("=" * 64)

    device = Metal.MTLCreateSystemDefaultDevice()

    if device is None:
        raise RuntimeError(
            "Metal GPU not available"
        )

    print(f"GPU: {device.name()}")

    doc = fitz.open(
        PDF_DIR / pdf_name
    )

    print(f"Pages: {len(doc)}")

    pipeline = load_shader(device)

    print("Metal full preprocessing shader: READY\n")

    total_cpu = 0.0
    total_gpu = 0.0

    for page_number, page in enumerate(
        doc,
        start=1
    ):

        # -----------------------------------------
        # Render PDF page
        # -----------------------------------------

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
        # CPU
        # -----------------------------------------

        cpu_result, cpu_time = (
            cpu_full_preprocessing(image)
        )

        # -----------------------------------------
        # Metal
        # -----------------------------------------

        gpu_result, gpu_time = (
            metal_full_preprocessing(
                device,
                pipeline,
                image
            )
        )

        # -----------------------------------------
        # Verification
        # -----------------------------------------

        difference = np.abs(
            cpu_result.astype(np.int16)
            - gpu_result.astype(np.int16)
        )

        max_difference = difference.max()
        mean_difference = difference.mean()

        total_cpu += cpu_time
        total_gpu += gpu_time

        print(
            f"Page {page_number}/{len(doc)}: "
            f"CPU={cpu_time:.6f}s | "
            f"Metal={gpu_time:.6f}s | "
            f"Max diff={max_difference} | "
            f"Mean diff={mean_difference:.6f}"
        )

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)

    print(
        f"Total CPU:      {total_cpu:.6f}s"
    )

    print(
        f"Total Metal:    {total_gpu:.6f}s"
    )

    if total_gpu > 0:
        print(
            f"GPU speedup:    {total_cpu / total_gpu:.2f}x"
        )


if __name__ == "__main__":
    main()
