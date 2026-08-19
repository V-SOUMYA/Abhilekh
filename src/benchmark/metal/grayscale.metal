#include <metal_stdlib>
using namespace metal;

kernel void rgb_to_grayscale(
    device const uchar3 *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    uchar3 pixel = input[id];

    float gray =
        0.299 * float(pixel.x) +
        0.587 * float(pixel.y) +
        0.114 * float(pixel.z);

    output[id] = uchar(gray);
}
