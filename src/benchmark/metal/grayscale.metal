#include <metal_stdlib>
using namespace metal;

kernel void rgb_to_grayscale(
    device const uchar *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    uint base = id * 3;

    uchar r = input[base];
    uchar g = input[base + 1];
    uchar b = input[base + 2];

    float gray =
        0.299 * float(r) +
        0.587 * float(g) +
        0.114 * float(b);

    output[id] = uchar(gray);
}
