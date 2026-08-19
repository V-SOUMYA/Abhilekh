import time
from pathlib import Path

import fitz
import Metal
import numpy as np
from PIL import Image


PDF_DIR = Path("data/raw/gazettes")


def cpu_grayscale(image):
    start = time.perf_counter()

    gray = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    ).astype(np.uint8)

    elapsed = time.perf_counter() - start

    return gray, elapsed


def load_metal_shader(device):
    shader_path = (
        Path(__file__).parent
        / "metal"
        / "grayscale.metal"
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
            f"Shader compilation failed: {error}"
        )

    function = library.newFunctionWithName_(
        "rgb_to_grayscale"
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


def metal_grayscale(
    device,
    pipeline,
    image
):
    height, width, _ = image.shape
    pixels = height * width

    input_data = np.ascontiguousarray(
        image,
        dtype=np.uint8
    )

    output_data = np.zeros(
        pixels,
        dtype=np.uint8
    )

    input_buffer = (
        device.newBufferWithBytes_length_options_(
            input_data.tobytes(),
            input_data.nbytes,
            Metal.MTLResourceStorageModeShared
        )
    )

    output_buffer = (
        device.newBufferWithLength_options_(
            output_data.nbytes,
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

    total_threads = pixels

    threadgroup_size = min(
        pipeline.maxTotalThreadsPerThreadgroup(),
        256
    )

    threadgroups = (
        total_threads + threadgroup_size - 1
    ) // threadgroup_size

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(
            threadgroups,
            1,
            1
        ),
        Metal.MTLSizeMake(
            threadgroup_size,
            1,
            1
        )
    )

    encoder.endEncoding()

    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    elapsed = time.perf_counter() - start

    contents = output_buffer.contents()

    result = np.frombuffer(
        contents.as_buffer(output_data.nbytes),
        dtype=np.uint8
    ).copy()

    result = result.reshape(
        height,
        width
    )

    return result, elapsed


def main():
    pdf_name = "275564.pdf"

    pdf_path = PDF_DIR / pdf_name

    print("=" * 60)
    print("CPU vs METAL: REAL ABHILEKH PDF")
    print("=" * 60)

    device = Metal.MTLCreateSystemDefaultDevice()

    if device is None:
        raise RuntimeError(
            "Metal GPU not available"
        )

    print(f"GPU: {device.name()}")

    doc = fitz.open(pdf_path)

    print(f"Pages: {len(doc)}")

    pipeline = load_metal_shader(device)

    print("Metal shader: READY\n")

    total_cpu = 0
    total_gpu = 0

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

        cpu_result, cpu_time = cpu_grayscale(
            image
        )

        # -----------------------------------------
        # Metal
        # -----------------------------------------

        gpu_result, gpu_time = metal_grayscale(
            device,
            pipeline,
            image
        )

        # -----------------------------------------
        # Verify
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
        f"Total CPU grayscale:   "
        f"{total_cpu:.6f}s"
    )

    print(
        f"Total Metal grayscale: "
        f"{total_gpu:.6f}s"
    )

    if total_gpu > 0:
        print(
            f"GPU speedup:           "
            f"{total_cpu / total_gpu:.2f}x"
        )


if __name__ == "__main__":
    main()
