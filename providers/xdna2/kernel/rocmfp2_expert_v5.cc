// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Exact scalar activation for the spatially fused Gen5 expert.  Projection
// partials remain FP32; the hidden vector is rounded once, RNE, to BF16 before
// the down projection.  The exp implementation was exhaustively compared on
// hardware against Ember's host expression over the 16,384-value probe corpus.

#define NOCPP

#include <aie_api/aie.hpp>

#include <cstdint>
#include <cstring>

namespace {

float exp_approx(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 87.0f) x = 87.0f;
    constexpr float kInvLn2 = 1.4426950408889634f;
    constexpr float kLn2 = 0.6931471805599453f;
    const float scaled = x * kInvLn2;
    const int exponent = static_cast<int>(
        scaled + (scaled < 0.0f ? -0.5f : 0.5f));
    const float r = x - static_cast<float>(exponent) * kLn2;
    float p = 1.0f / 39916800.0f;
    p = 1.0f / 3628800.0f + r * p;
    p = 1.0f / 362880.0f + r * p;
    p = 1.0f / 40320.0f + r * p;
    p = 1.0f / 5040.0f + r * p;
    p = 1.0f / 720.0f + r * p;
    p = 1.0f / 120.0f + r * p;
    p = 1.0f / 24.0f + r * p;
    p = 1.0f / 6.0f + r * p;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;
    const uint32_t bits = static_cast<uint32_t>(exponent + 127) << 23;
    float power;
    std::memcpy(&power, &bits, sizeof(power));
    return power * p;
}

uint16_t bf16_rne(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

float raw_float(const bfloat16 * values, unsigned index) {
    const uint16_t * raw = reinterpret_cast<const uint16_t *>(values);
    const uint32_t bits = static_cast<uint32_t>(raw[index * 2]) |
                          (static_cast<uint32_t>(raw[index * 2 + 1]) << 16);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace

extern "C" void swiglu_rocmfp2_v5(const float * __restrict gate,
                                   const float * __restrict up,
                                   const bfloat16 * __restrict params,
                                   bfloat16 * __restrict output) {
    const float gate_scale = raw_float(params, 0);
    const float up_scale = raw_float(params, 1);
    const float clamp = raw_float(params, 2);
    for (unsigned lane = 0; lane < 64; ++lane) {
        float g = gate[lane] * gate_scale;
        float u = up[lane] * up_scale;
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        const float value = (g / (1.0f + exp_approx(-g))) * u;
        reinterpret_cast<uint16_t *>(output)[lane] = bf16_rne(value);
    }
}

extern "C" void store_down_rocmfp2_v5(const float * __restrict first,
                                       const float * __restrict second,
                                       bfloat16 * __restrict packet) {
    float * output = reinterpret_cast<float *>(packet);
    for (unsigned lane = 0; lane < 64; lane += 16) {
        aie::store_v(output + lane, aie::load_v<16>(first + lane));
        aie::store_v(output + 64 + lane, aie::load_v<16>(second + lane));
    }
}
