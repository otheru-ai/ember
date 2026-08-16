// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Four-row ROCMFP2 projection microkernel for Gen6.  The expensive affine-FP2
// tile decode is performed once, then reused by four independent activation
// rows.  This is the weight-reuse property Gen5 runlists cannot provide.

#define NOCPP

#include <aie_api/aie.hpp>

#include <cstdint>

namespace {

constexpr unsigned kBatch = 4;
constexpr unsigned kInput = 128;
constexpr unsigned kOutput = 64;
constexpr unsigned kBlockWeights = 32;
constexpr unsigned kBlocks = kInput / kBlockWeights;
constexpr unsigned kCodesPerByte = 2;
constexpr unsigned kCodeBytes = kInput * kOutput / kCodesPerByte;
constexpr unsigned kMetadataValues = kBlocks * kOutput;

alignas(aie::vector_decl_align) static bfloat16 decoded_weights[kInput * kOutput];

}  // namespace

extern "C" {

void zero_rocmfp2_v6_f32x4(float * output) {
    const aie::vector<float, 16> zeros = aie::zeros<float, 16>();
    for (unsigned lane = 0; lane < kBatch * kOutput; lane += 16)
        aie::store_v(output + lane, zeros);
}

void gemm_rocmfp2_v6_f32x4(const bfloat16 * __restrict input,
                           const uint8_t * __restrict weights,
                           float * __restrict output) {
    event0();

    const uint8_t * __restrict codes = weights;
    const bfloat16 * __restrict scales =
        reinterpret_cast<const bfloat16 *>(weights + kCodeBytes);
    const bfloat16 * __restrict offsets = scales + kMetadataValues;

    for (unsigned block = 0; block < kBlocks; ++block)
        chess_prepare_for_pipelining chess_loop_range(4, ) {
        const aie::vector<bfloat16, kOutput> block_scale =
            aie::load_v<kOutput>(scales + block * kOutput);
        const aie::vector<bfloat16, kOutput> block_offset =
            aie::load_v<kOutput>(offsets + block * kOutput);
        for (unsigned i = 0; i < kBlockWeights; ++i)
            chess_prepare_for_pipelining chess_loop_range(32, ) {
            const unsigned input_lane = block * kBlockWeights + i;
            const aie::vector<uint8, kOutput / kCodesPerByte> packed =
                aie::load_v<kOutput / kCodesPerByte>(
                    codes + input_lane * (kOutput / kCodesPerByte));
            const aie::vector<uint4, kOutput> codes_u4 =
                packed.template cast_to<uint4>();
            const aie::vector<uint8, kOutput> codes_u8 = codes_u4.unpack();
            const aie::vector<int8, kOutput> codes_i8 =
                codes_u8.template cast_to<int8>();
            const aie::vector<bfloat16, kOutput> codes_bf16 =
                aie::to_float<bfloat16>(codes_i8);
            const auto scaled_acc = aie::mul(codes_bf16, block_scale);
            const aie::vector<bfloat16, kOutput> scaled =
                scaled_acc.template to_vector<bfloat16>();
            aie::store_v(decoded_weights + input_lane * kOutput,
                         aie::sub(scaled, block_offset));
        }
    }

    for (unsigned token = 0; token < kBatch; ++token) {
        aie::accum<accfloat, kOutput> acc;
        acc.from_vector(aie::load_v<kOutput>(output + token * kOutput));
        const bfloat16 * __restrict token_input = input + token * kInput;
        for (unsigned row = 0; row < kInput; row += 8)
            chess_prepare_for_pipelining chess_loop_range(16, ) {
            const aie::vector<bfloat16, 8> activation =
                aie::load_v<8>(token_input + row);
            const bfloat16 * __restrict b = decoded_weights + row * kOutput;
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
