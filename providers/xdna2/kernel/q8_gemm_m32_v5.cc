// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// BFP16-MAC-emulated variant of q8_gemm_m32_v4.cc. Current AMD IRON enables
// AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 by default for AIE2P BF16 GEMM,
// exposing an 8x8x8 mmul tile in place of the native 4x8x8 shape. Storage
// remains compensated BF16 high+residual and accumulation remains FP32.

#define NOCPP

#include <aie_api/aie.hpp>

namespace {

constexpr unsigned kM = 32;
constexpr unsigned kK = 128;
constexpr unsigned kN = 64;
constexpr unsigned kR = 8;
constexpr unsigned kS = 8;
constexpr unsigned kT = 8;
constexpr unsigned kMBlocks = kM / kR;
constexpr unsigned kKBlocks = kK / kS;
constexpr unsigned kNBlocks = kN / kT;
constexpr unsigned kBElements = kK * kN;
constexpr unsigned kCElements = kM * kN;

using Mmul = aie::mmul<kR, kS, kT, bfloat16, bfloat16, accauto>;

static_assert(Mmul::size_A == kR * kS);
static_assert(Mmul::size_B == kS * kT);
static_assert(Mmul::size_C == kR * kT);

}  // namespace

extern "C" {

void zero_q8_v5_f32_m32(float * output) {
    const aie::vector<float, 16> zeros = aie::zeros<float, 16>();
    for (unsigned lane = 0; lane < kCElements; lane += 16)
        aie::store_v(output + lane, zeros);
}

void gemm_q8_v5_f32_m32(const bfloat16 * __restrict input,
                        const bfloat16 * __restrict weights,
                        float * __restrict output) {
    event0();
    aie::set_rounding(aie::rounding_mode::conv_even);
    const bfloat16 * __restrict residual = weights + kBElements;

    for (unsigned mb = 0; mb < kMBlocks; mb += 2) {
        for (unsigned nb = 0; nb < kNBlocks; nb += 2) {
            float * __restrict c00_ptr =
                output + (mb * kNBlocks + nb) * Mmul::size_C;
            float * __restrict c01_ptr = c00_ptr + Mmul::size_C;
            float * __restrict c10_ptr =
                output + ((mb + 1) * kNBlocks + nb) * Mmul::size_C;
            float * __restrict c11_ptr = c10_ptr + Mmul::size_C;

            Mmul c00(aie::load_v<Mmul::size_C>(c00_ptr));
            Mmul c01(aie::load_v<Mmul::size_C>(c01_ptr));
            Mmul c10(aie::load_v<Mmul::size_C>(c10_ptr));
            Mmul c11(aie::load_v<Mmul::size_C>(c11_ptr));

            for (unsigned kb = 0; kb < kKBlocks; ++kb)
                chess_prepare_for_pipelining chess_loop_range(16, ) {
                const auto a0 = aie::load_v<Mmul::size_A>(
                    input + (mb * kKBlocks + kb) * Mmul::size_A);
                const auto a1 = aie::load_v<Mmul::size_A>(
                    input + ((mb + 1) * kKBlocks + kb) * Mmul::size_A);
                const unsigned b0_offset =
                    (kb * kNBlocks + nb) * Mmul::size_B;
                const auto b0 = aie::load_v<Mmul::size_B>(
                    weights + b0_offset);
                const auto b1 = aie::load_v<Mmul::size_B>(
                    weights + b0_offset + Mmul::size_B);
                c00.mac(a0, b0);
                c01.mac(a0, b1);
                c10.mac(a1, b0);
                c11.mac(a1, b1);

                const auto r0 = aie::load_v<Mmul::size_B>(
                    residual + b0_offset);
                const auto r1 = aie::load_v<Mmul::size_B>(
                    residual + b0_offset + Mmul::size_B);
                c00.mac(a0, r0);
                c01.mac(a0, r1);
                c10.mac(a1, r0);
                c11.mac(a1, r1);
            }

            aie::store_v(c00_ptr, c00.template to_vector<float>());
            aie::store_v(c01_ptr, c01.template to_vector<float>());
            aie::store_v(c10_ptr, c10.template to_vector<float>());
            aie::store_v(c11_ptr, c11.template to_vector<float>());
        }
    }
    event1();
}

}  // extern "C"
