import time

import Metal
import numpy as np


def cpu_grayscale(image):
    """RGB -> grayscale using NumPy."""
    return (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    ).astype(np.uint8)


def main():
    device = Metal.MTLCreateSystemDefaultDevice()

    if device is None:
        print("No Metal GPU found.")
        return

    print("=" * 50)
    print("CPU vs METAL GPU TEST")
    print("=" * 50)
    print(f"GPU: {device.name()}")

    # Same image for both CPU and GPU
    image = np.random.randint(
        0,
        256,
        size=(1000, 1000, 3),
        dtype=np.uint8
    )

    print(f"Test image: {image.shape}")

    # -------------------------
    # CPU grayscale
    # -------------------------
    start = time.perf_counter()

    cpu_result = cpu_grayscale(image)

    cpu_time = time.perf_counter() - start

    print(f"\nCPU grayscale:   {cpu_time:.6f}s")

    # -------------------------
    # Metal GPU setup
    # -------------------------
    gpu_buffer = device.newBufferWithBytes_length_options_(
        image.tobytes(),
        image.nbytes,
        0
    )

    if gpu_buffer is None:
        print("GPU buffer allocation failed.")
        return

    print("GPU buffer:       OK")

    print("\nMetal GPU is ready for image processing.")
    print("Next step: execute the grayscale operation on the GPU.")


if __name__ == "__main__":
    main()
