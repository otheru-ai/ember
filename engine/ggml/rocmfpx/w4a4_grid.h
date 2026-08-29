#pragma once

// Shared scalar pieces of the experimental W4A4 activation grid. The warp
// shuffle remains in quantize_mmq_q8_1; these helpers pin the surrounding
// arithmetic for host tests and device code.
#if defined(__HIPCC__) || defined(__CUDACC__)
#define ROCMI4_W4A4_HD __host__ __device__
#else
#define ROCMI4_W4A4_HD
#endif

#include <math.h>
#include <stdint.h>

static ROCMI4_W4A4_HD inline int rocmi4_w4a4_quantize_value(float value,
                                                            float d_inv) {
    const float scaled = roundf(value * d_inv);
    const float clamped = fminf(fmaxf(scaled, -8.0f), 7.0f);
    return (int)clamped;
}

static ROCMI4_W4A4_HD inline uint32_t rocmi4_w4a4_pack4(
        float x0, float x1, float x2, float x3, float d_inv) {
    const uint32_t c0 = (uint32_t)(rocmi4_w4a4_quantize_value(x0, d_inv) & 15);
    const uint32_t c1 = (uint32_t)(rocmi4_w4a4_quantize_value(x1, d_inv) & 15);
    const uint32_t c2 = (uint32_t)(rocmi4_w4a4_quantize_value(x2, d_inv) & 15);
    const uint32_t c3 = (uint32_t)(rocmi4_w4a4_quantize_value(x3, d_inv) & 15);
    return c0 | (c1 << 8) | (c2 << 16) | (c3 << 24);
}

static ROCMI4_W4A4_HD inline float rocmi4_w4a4_scale(float amax) {
    return amax > 0.0f ? amax / (7.0f * 16.0f) : 0.0f;
}

static ROCMI4_W4A4_HD inline uint32_t rocmi4_w4a4_fold(uint32_t lo4,
                                                        uint32_t hi4) {
    return lo4 | (hi4 << 4);
}

#undef ROCMI4_W4A4_HD
