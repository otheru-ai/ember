// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Five-row BF16 projection used after one-time Q8_0 dequantization. The packed
// 128x64 tile is already K-major, so each invocation spends its cycles only on
// the BF16/FP32 MAC. This is the shared-expert building block and the same
// arithmetic contract needed by the draft's Q8 dense projections.

#define NOCPP

#include <aie_api/aie.hpp>

namespace {

constexpr unsigned kBatch = 5;
constexpr unsigned kInput = 128;
constexpr unsigned kOutput = 64;

}  // namespace

extern "C" {

void zero_q8_v1_f32x5(float * output) {
    const aie::vector<float, 16> zeros = aie::zeros<float, 16>();
    for (unsigned lane = 0; lane < kBatch * kOutput; lane += 16)
        aie::store_v(output + lane, zeros);
}

void gemm_q8_v1_f32x5(const bfloat16 * __restrict input,
                      const bfloat16 * __restrict weights,
                      float * __restrict output) {
    event0();
    for (unsigned token = 0; token < kBatch; ++token) {
        aie::accum<accfloat, kOutput> acc;
        acc.from_vector(aie::load_v<kOutput>(output + token * kOutput));
        const bfloat16 * __restrict token_input = input + token * kInput;
        for (unsigned row = 0; row < kInput; row += 8)
            chess_prepare_for_pipelining chess_loop_range(16, ) {
            const aie::vector<bfloat16, 8> activation =
                aie::load_v<8>(token_input + row);
            const bfloat16 * __restrict b = weights + row * kOutput;
            const aie::vector<bfloat16, kOutput> b0 = aie::load_v<kOutput>(b);
            const aie::vector<bfloat16, kOutput> b1 = aie::load_v<kOutput>(b + kOutput);
            const aie::vector<bfloat16, kOutput> b2 = aie::load_v<kOutput>(b + 2 * kOutput);
            const aie::vector<bfloat16, kOutput> b3 = aie::load_v<kOutput>(b + 3 * kOutput);
            const aie::vector<bfloat16, kOutput> b4 = aie::load_v<kOutput>(b + 4 * kOutput);
            const aie::vector<bfloat16, kOutput> b5 = aie::load_v<kOutput>(b + 5 * kOutput);
            const aie::vector<bfloat16, kOutput> b6 = aie::load_v<kOutput>(b + 6 * kOutput);
            const aie::vector<bfloat16, kOutput> b7 = aie::load_v<kOutput>(b + 7 * kOutput);
            acc = aie::accumulate<kOutput>(acc, activation, 0,
                                          b0, b1, b2, b3, b4, b5, b6, b7);
        }
        aie::store_v(output + token * kOutput,
                     acc.template to_vector<float>());
    }
    event1();
}

}  // extern "C"
