import time
from pathlib import Path

import Metal
import numpy as np


def load_shader(device):
    shader_path = Path(__file__).parent / "metal" / "grayscale.metal"

    source = shader_path.read_text()

    library, error = device.newLibraryWithSource_options_error_(
        source,
        None,
        None
    )

    if error:
        raise RuntimeError(f"Metal shader compilation failed: {error}")

    return library


def main():
    print("=" * 50)
    print("CPU vs METAL GPU GRAYSCALE TEST")
    print("=" * 50)

    # --------------------------------------------------
    # GPU
    # --------------------------------------------------

    device = Metal.MTLCreateSystemDefaultDevice()

    if device is None:
        raise RuntimeError("Metal GPU not available")

    print(f"GPU: {device.name()}")

    # --------------------------------------------------
    # Test image
    # --------------------------------------------------

    height = 1000
    width = 1000

    image = np.random.randint(
        0,
        256,
        (height, width, 3),
        dtype=np.uint8
    )

    print(f"Test image: {image.shape}")

    # --------------------------------------------------
    # CPU grayscale
    # --------------------------------------------------

    start = time.perf_counter()

    cpu_gray = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    ).astype(np.uint8)

    cpu_time = time.perf_counter() - start

    print(f"\nCPU grayscale:   {cpu_time:.6f}s")

    # --------------------------------------------------
    # Load Metal shader
    # --------------------------------------------------

    print("Compiling Metal shader...")

    library = load_shader(device)

    function = library.newFunctionWithName_("rgb_to_grayscale")

    if function is None:
        raise RuntimeError("Could not find Metal kernel")

    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        function,
        None
    )

    if error:
        raise RuntimeError(
            f"Could not create compute pipeline: {error}"
        )

    print("Metal shader compiled successfully.")

    # --------------------------------------------------
    # GPU buffers
    # --------------------------------------------------

    input_data = np.ascontiguousarray(image)
    output_data = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    input_buffer = device.newBufferWithBytes_length_options_(
        input_data.tobytes(),
        input_data.nbytes,
        Metal.MTLResourceStorageModeShared
    )

    output_buffer = device.newBufferWithLength_options_(
        output_data.nbytes,
        Metal.MTLResourceStorageModeShared
    )

    # --------------------------------------------------
    # Command queue
    # --------------------------------------------------

    command_queue = device.newCommandQueue()

    # --------------------------------------------------
    # GPU execution
    # --------------------------------------------------

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

    threads = Metal.MTLSizeMake(
        width * height,
        1,
        1
    )

    threadgroup_size = min(
        pipeline.maxTotalThreadsPerThreadgroup(),
        256
    )

    threadgroups = Metal.MTLSizeMake(
        (width * height + threadgroup_size - 1)
        // threadgroup_size,
        1,
        1
    )

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        threadgroups,
        Metal.MTLSizeMake(
            threadgroup_size,
            1,
            1
        )
    )

    encoder.endEncoding()

    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    gpu_time = time.perf_counter() - start

    # --------------------------------------------------
    # Read GPU result
    # --------------------------------------------------

    contents = output_buffer.contents()

    gpu_result = np.frombuffer(
        contents.as_buffer(output_data.nbytes),
        dtype=np.uint8
    ).reshape(height, width).copy()

    print(f"Metal grayscale:  {gpu_time:.6f}s")

    # --------------------------------------------------
    # Verify correctness
    # --------------------------------------------------

    difference = np.abs(
        cpu_gray.astype(np.int16)
        - gpu_result.astype(np.int16)
    )

    max_difference = difference.max()
    mean_difference = difference.mean()

    print("\nVerification:")
    print(f"Maximum pixel difference: {max_difference}")
    print(f"Mean pixel difference:    {mean_difference:.6f}")

    if max_difference <= 1:
        print("RESULT: CPU and GPU outputs match.")
    else:
        print("RESULT: CPU and GPU outputs differ.")

    # --------------------------------------------------
    # Speedup
    # --------------------------------------------------

    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"\nGPU/CPU speed ratio: {speedup:.2f}x")


if __name__ == "__main__":
    main()
