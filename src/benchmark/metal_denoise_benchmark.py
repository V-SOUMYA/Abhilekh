import time
from pathlib import Path

import fitz
import Metal
import numpy as np


PDF_DIR = Path("data/raw/gazettes")


def cpu_grayscale_denoise(image):
    """
    CPU baseline.

    Grayscale conversion followed by a 3x3 box blur.
    """

    start = time.perf_counter()

    gray = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    ).astype(np.uint8)

    # Simple 3x3 box blur implemented with NumPy.
    padded = np.pad(
        gray,
        1,
        mode="edge"
    )

    result = (
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

    result = result.astype(np.uint8)

    elapsed = time.perf_counter() - start

    return result, elapsed


def load_shader(device):
    shader_path = (
        Path(__file__).parent
        / "metal"
        / "denoise.metal"
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
        "box_blur"
    )

    if function is None:
        raise RuntimeError(
            "Could not find box_blur kernel"
        )

    pipeline, error = (
        device.newComputePipelineStateWithFunction_error_(
            function,
            None
        )
    )

    if error:
        raise RuntimeError(
            f"Could not create compute pipeline: {error}"
        )

    return pipeline


def metal_grayscale_denoise(
    device,
    pipeline,
    image
):
    height, width, _ = image.shape

    # -----------------------------------------
    # CPU grayscale first
    # -----------------------------------------

    gray = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    ).astype(np.uint8)

    gray = np.ascontiguousarray(gray)

    output = np.zeros_like(gray)

    # -----------------------------------------
    # GPU buffers
    # -----------------------------------------

    input_buffer = (
        device.newBufferWithBytes_length_options_(
            gray.tobytes(),
            gray.nbytes,
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

    threadgroup_width = 16
    threadgroup_height = 16

    threadgroups = Metal.MTLSizeMake(
        (
            width + threadgroup_width - 1
        ) // threadgroup_width,
        (
            height + threadgroup_height - 1
        ) // threadgroup_height,
        1
    )

    threads_per_group = Metal.MTLSizeMake(
        threadgroup_width,
        threadgroup_height,
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
    # Read result
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

    print("=" * 60)
    print("CPU vs METAL: GRAYSCALE + DENOISING")
    print("=" * 60)

    device = Metal.MTLCreateSystemDefaultDevice()

    if device is None:
        raise RuntimeError(
            "Metal GPU not available"
        )

    print(f"GPU: {device.name()}")

    pdf_path = PDF_DIR / pdf_name

    doc = fitz.open(pdf_path)

    print(f"Pages: {len(doc)}")

    pipeline = load_shader(device)

    print("Metal denoising shader: READY\n")

    total_cpu = 0
    total_gpu = 0

    for page_number, page in enumerate(
        doc,
        start=1
    ):

        # -----------------------------------------
        # Render page
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
            cpu_grayscale_denoise(image)
        )

        # -----------------------------------------
        # Metal
        # -----------------------------------------

        gpu_result, gpu_time = (
            metal_grayscale_denoise(
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

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"Total CPU:   {total_cpu:.6f}s"
    )

    print(
        f"Total Metal: {total_gpu:.6f}s"
    )

    if total_gpu > 0:
        print(
            f"GPU speedup: {total_cpu / total_gpu:.2f}x"
        )


if __name__ == "__main__":
    main()
