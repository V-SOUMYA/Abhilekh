#include <metal_stdlib>
using namespace metal;

kernel void box_blur(
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

    int sum = 0;
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
                sum += input[ny * int(width) + nx];
                count++;
            }
        }
    }

    output[y * int(width) + x] =
        uchar(sum / count);
}
