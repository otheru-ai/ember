// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Compensated five-row Q8_0 projection. The host splits each dequantized
// weight into a BF16 high term and BF16 residual. Both planes MAC directly
// into the same FP32 accumulator. This avoids Peano's AIE2P lowering of vector
// FP32 multiplication through BF16 operands, which invalidated the compact
// int8/F32-scale prototype on trained weights.

#define NOCPP

#include <aie_api/aie.hpp>

namespace {

constexpr unsigned kBatch = 5;
constexpr unsigned kInput = 128;
constexpr unsigned kOutput = 64;
constexpr unsigned kPlane = kInput * kOutput;

}  // namespace

extern "C" {

void zero_q8_v2_f32x5(float * output) {
    const aie::vector<float, 16> zeros = aie::zeros<float, 16>();
    for (unsigned lane = 0; lane < kBatch * kOutput; lane += 16)
        aie::store_v(output + lane, zeros);
}

void gemm_q8_v2_f32x5(const bfloat16 * __restrict input,
                      const bfloat16 * __restrict weights,
                      float * __restrict output) {
    event0();
    const bfloat16 * __restrict residual = weights + kPlane;
    for (unsigned token = 0; token < kBatch; ++token) {
        aie::accum<accfloat, kOutput> acc;
        acc.from_vector(aie::load_v<kOutput>(output + token * kOutput));
        const bfloat16 * __restrict token_input = input + token * kInput;
        for (unsigned row = 0; row < kInput; row += 8)
            chess_prepare_for_pipelining chess_loop_range(16, ) {
            const aie::vector<bfloat16, 8> activation =
                aie::load_v<8>(token_input + row);
            const bfloat16 * __restrict high = weights + row * kOutput;
            const bfloat16 * __restrict low = residual + row * kOutput;
            acc = aie::accumulate<kOutput>(
                acc, activation, 0,
                aie::load_v<kOutput>(high),
                aie::load_v<kOutput>(high + kOutput),
                aie::load_v<kOutput>(high + 2 * kOutput),
                aie::load_v<kOutput>(high + 3 * kOutput),
                aie::load_v<kOutput>(high + 4 * kOutput),
                aie::load_v<kOutput>(high + 5 * kOutput),
                aie::load_v<kOutput>(high + 6 * kOutput),
                aie::load_v<kOutput>(high + 7 * kOutput));
            acc = aie::accumulate<kOutput>(
                acc, activation, 0,
                aie::load_v<kOutput>(low),
                aie::load_v<kOutput>(low + kOutput),
                aie::load_v<kOutput>(low + 2 * kOutput),
                aie::load_v<kOutput>(low + 3 * kOutput),
                aie::load_v<kOutput>(low + 4 * kOutput),
                aie::load_v<kOutput>(low + 5 * kOutput),
                aie::load_v<kOutput>(low + 6 * kOutput),
                aie::load_v<kOutput>(low + 7 * kOutput));
        }
        aie::store_v(output + token * kOutput,
                     acc.template to_vector<float>());
    }
    event1();
}

}  // extern "C"
