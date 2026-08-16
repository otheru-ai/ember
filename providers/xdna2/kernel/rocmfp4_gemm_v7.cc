// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Five-row ROCMFP4_FAST projection microkernel for the DSpark draft block.
// The compact signed-codebook nibbles stay packed until they reach an AIE
// core.  One vector decode is then reused by all five speculative rows.

#define NOCPP

#include <aie_api/aie.hpp>

#include <cstdint>

namespace {

constexpr unsigned kBatch = 5;
constexpr unsigned kInput = 128;
constexpr unsigned kOutput = 64;
constexpr unsigned kBlockWeights = 32;
constexpr unsigned kBlocks = kInput / kBlockWeights;
constexpr unsigned kCodesPerByte = 2;
constexpr unsigned kCodeBytes = kInput * kOutput / kCodesPerByte;
constexpr unsigned kMetadataValues = kBlocks * kOutput;

alignas(aie::vector_decl_align) static bfloat16 decoded_weights[kInput * kOutput];

aie::vector<int8, kOutput> decode_signed_codebook(
        const aie::vector<uint8, kOutput> & codes) {
    const aie::vector<uint8, kOutput> low_three = aie::bit_and(
        codes, aie::broadcast<uint8, kOutput>(7));
    const aie::vector<uint8, kOutput> expanded = aie::sub(
        aie::add(low_three, low_three),
        aie::broadcast<uint8, kOutput>(4));
    const aie::mask<kOutput> direct = aie::lt(
        low_three, aie::broadcast<uint8, kOutput>(5));
    const aie::vector<uint8, kOutput> magnitude =
        aie::select(expanded, low_three, direct);
    const aie::vector<uint8, kOutput> sign = aie::bit_and(
        codes, aie::broadcast<uint8, kOutput>(8));
    const aie::mask<kOutput> negative = aie::lt(
        aie::zeros<uint8, kOutput>(), sign);
    const aie::vector<int8, kOutput> positive =
        magnitude.template cast_to<int8>();
    return aie::select(positive, aie::neg(positive), negative);
}

}  // namespace

extern "C" {

void zero_rocmfp4_v7_f32x5(float * output) {
    const aie::vector<float, 16> zeros = aie::zeros<float, 16>();
    for (unsigned lane = 0; lane < kBatch * kOutput; lane += 16)
        aie::store_v(output + lane, zeros);
}

void gemm_rocmfp4_v7_f32x5(const bfloat16 * __restrict input,
                           const uint8_t * __restrict weights,
                           float * __restrict output) {
    event0();

    const uint8_t * __restrict codes = weights;
    const bfloat16 * __restrict scales =
        reinterpret_cast<const bfloat16 *>(weights + kCodeBytes);

    for (unsigned block = 0; block < kBlocks; ++block)
        chess_prepare_for_pipelining chess_loop_range(4, ) {
        const aie::vector<bfloat16, kOutput> block_scale =
            aie::load_v<kOutput>(scales + block * kOutput);
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
                decode_signed_codebook(codes_u8);
            const aie::vector<bfloat16, kOutput> codes_bf16 =
                aie::to_float<bfloat16>(codes_i8);
            const auto scaled_acc = aie::mul(codes_bf16, block_scale);
            aie::store_v(decoded_weights + input_lane * kOutput,
                         scaled_acc.template to_vector<bfloat16>());
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
