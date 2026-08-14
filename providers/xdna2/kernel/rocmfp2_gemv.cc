// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The 4x8 streaming topology is adapted from TileFuse's XDNA2 W4A16 GEMV
// example (mlir-aie commit 8c3d2be63161c255c4e290e500702571c69a7112).
// Unlike TileFuse's INT4 microkernel, this bring-up kernel decodes GGML's
// affine ROCMFP2 blocks directly. It is deliberately scalar until hardware
// equivalence establishes the byte layout and arithmetic contract.

#include <aie_api/aie.hpp>

#include <cstdint>

namespace {

constexpr int kInput = 128;
constexpr int kOutput = 64;
constexpr int kBlockWeights = 32;
constexpr int kBlockBytes = 10;

inline float ue4m3(uint8_t value) {
    if (value > 0x7e) return 0.0f;
    const unsigned exponent = value >> 3;
    const unsigned mantissa = value & 7u;
    if (exponent == 0) return static_cast<float>(mantissa) / 1024.0f;
    const int shift = static_cast<int>(exponent) - 11;
    return static_cast<float>(8u + mantissa) *
           static_cast<float>(1u << (shift >= 0 ? shift : 0)) /
           static_cast<float>(1u << (shift < 0 ? -shift : 0));
}

}  // namespace

extern "C" {

void zero_rocmfp2_bf16(bfloat16 * output) {
    for (int lane = 0; lane < kOutput; ++lane) output[lane] = bfloat16(0.0f);
}

void gemv_rocmfp2_bf16(const bfloat16 * input,
                       const uint8_t * weights,
                       bfloat16 * output) {
    for (int lane = 0; lane < kOutput; ++lane) {
        float sum = static_cast<float>(output[lane]);
        const uint8_t * row = weights + lane * 4 * kBlockBytes;
        for (int block = 0; block < kInput / kBlockWeights; ++block) {
            const uint8_t * q = row + block * kBlockBytes;
            const float scale = ue4m3(q[8]);
            const float offset = ue4m3(q[9]);
            for (int i = 0; i < kBlockWeights; ++i) {
                const uint8_t code =
                    static_cast<uint8_t>((q[i >> 2] >> (2 * (i & 3))) & 3u);
                const float value = static_cast<float>(code) * scale - offset;
                sum += static_cast<float>(input[block * kBlockWeights + i]) * value;
            }
        }
        output[lane] = bfloat16(sum);
    }
}

}  // extern "C"
