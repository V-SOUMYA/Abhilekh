#include <metal_stdlib>
using namespace metal;

kernel void contrast_stretch(
    device const uchar *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant uint &width [[buffer(2)]],
    constant uint &height [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    uint index = gid.y * width + gid.x;

    float pixel = float(input[index]);

    // Simple contrast stretch around the midpoint.
    float enhanced = (pixel - 128.0) * 1.2 + 128.0;

    enhanced = clamp(enhanced, 0.0, 255.0);

    output[index] = uchar(enhanced);
}
