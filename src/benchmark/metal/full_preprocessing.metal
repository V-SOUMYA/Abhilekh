#include <metal_stdlib>
using namespace metal;

kernel void full_preprocessing(
    device const uchar *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant uint &width [[buffer(2)]],
    constant uint &height [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    int x = int(gid.x);
    int y = int(gid.y);

    // --------------------------------------------------
    // 1. RGB -> Grayscale
    // --------------------------------------------------

    uint rgb_index = (y * width + gid.x) * 3;

    float r = float(input[rgb_index]);
    float g = float(input[rgb_index + 1]);
    float b = float(input[rgb_index + 2]);

    float gray_value =
        0.299 * r +
        0.587 * g +
        0.114 * b;

    // --------------------------------------------------
    // 2. 3x3 Box Denoising
    // --------------------------------------------------

    float sum = 0.0;
    int count = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {

            int nx = x + dx;
            int ny = y + dy;

            if (
                nx >= 0 &&
                nx < int(width) &&
                ny >= 0 &&
                ny < int(height)
            ) {

                uint neighbor_index =
                    (ny * int(width) + nx) * 3;

                float nr = float(input[neighbor_index]);
                float ng = float(input[neighbor_index + 1]);
                float nb = float(input[neighbor_index + 2]);

                float neighbor_gray =
                    0.299 * nr +
                    0.587 * ng +
                    0.114 * nb;

                sum += neighbor_gray;
                count++;
            }
        }
    }

    float denoised = sum / float(count);

    // --------------------------------------------------
    // 3. Contrast enhancement
    // --------------------------------------------------

    float enhanced =
        (denoised - 128.0) * 1.2 + 128.0;

    enhanced = clamp(
        enhanced,
        0.0,
        255.0
    );

    output[y * width + gid.x] =
        uchar(enhanced);
}
