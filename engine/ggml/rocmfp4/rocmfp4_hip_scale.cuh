#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

static __device__ __forceinline__ float rocmfp4_u32_as_f32(uint32_t bits) {
#if defined(GGML_USE_HIP)
    return __uint_as_float(bits);
#else
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
#endif
}

// ROCmFP4 validates scale bytes before backend execution, so HIP/ROCm hot
// paths can decode finite unsigned E4M3 half-scales directly without the
// generic FP8 NaN handling used by other formats.
// EMBER FORK DIVERGENCE (engine/VENDOR.md): branch-free UE4M3 half-scale decode.
//
// Same closed form as rocmfpx_ue4m3_to_fp32_finite below, minus the non-finite
// clamp (scale bytes are validated before backend execution, per the note
// above). This variant is on the ROCMFP4 path, which the profile shows is the
// heavier of the two: mul_mat_vec_q<Q4_0_ROCMFP4_FAST> is 24.6% of decode --
// the single largest decode kernel -- and mul_mat_q<...,64> is 10.3% of
// prefill. Measured on gfx1151, removing the equivalent branch from the fp2
// helper cut its kernel by 11.6%.
//
//     value = m * 2^(exp-11),  m = (exp == 0) ? man<<1 : man|8
//
// VERIFIED bit-identical to the previous implementation for all 256 uint8_t
// inputs, including the x > 0x7e range this variant does not special-case --
// see test_rocmfp_scale.c.
static __device__ __forceinline__ float rocmfp4_ue4m3_to_fp32_half_finite(uint8_t x) {
    const uint32_t xi  = x;
    const uint32_t exp = (xi >> 3) & 0xFu;
    const uint32_t man = xi & 0x7u;

    const uint32_t m = (exp == 0u) ? (man << 1) : (man | 8u);

    return (float) m * rocmfp4_u32_as_f32((exp + 116u) << 23);
}

// EMBER FORK DIVERGENCE (engine/VENDOR.md): branch-free UE4M3 decode.
//
// This is the single hottest scalar helper in the ROCMFP path: it runs twice
// per block in load_tiles_rocmfp2_affine (mmq.cuh:597, 23.4% of prefill) and
// twice per block in vec_dot_rocmfp2_q8_1 (vecdotq.cuh:526-527, 18% of decode),
// plus the fp3/fp6/fp8 dequant paths.
//
// The previous form had two early returns. On gfx1151 those are not free: each
// compiles to a v_cmpx + s_and_saveexec_b32 / s_or_b32 exec-mask pair, and the
// measured gfx1151 assembly showed that 16-instruction branchy sequence
// repeated 8x in the mmq loader alone. Exec-mask writes also break VOPD
// dual-issue, because a VOPD pair must be two independent VALU ops with no
// intervening control flow (ISA 7.6).
//
// Closed form, no branches:
//     value = m * 2^(exp-11),  m = (exp == 0) ? man<<1 : man|8
// with m forced to 0 for the x > 0x7e non-finite range.
//   exp == 0: 2*man * 2^-11  == man/1024        (old denormal path)
//   exp >= 1: (8+man) * 2^(exp-11) == (1+man/8) * 2^(exp-8)  (old normal path)
// The 2^(exp-11) factor is built directly as (exp+116)<<23; exp is 4 bits, so
// the biased field stays in 116..131 and is always a valid normal float.
//
// Every remaining operation (v_and, v_lshlrev, v_add_nc, v_cndmask) is in the
// VOPD X/Y opcode tables (ISA 15.3.7, tables 91-92), so the pair can dual-issue
// where the branchy form could not.
//
// VERIFIED bit-identical to the previous implementation for all 256 uint8_t
// inputs -- see test_rocmfp_scale.c, which embeds the old form as reference.
static __device__ __forceinline__ float rocmfpx_ue4m3_to_fp32_finite(uint8_t x) {
    const uint32_t xi  = x;
    const uint32_t exp = (xi >> 3) & 0xFu;
    const uint32_t man = xi & 0x7u;

    uint32_t m = (exp == 0u) ? (man << 1) : (man | 8u);
    m = (xi > 0x7eu) ? 0u : m;

    return (float) m * rocmfp4_u32_as_f32((exp + 116u) << 23);
}

static __device__ __forceinline__ uint8_t rocmfpx_nearest_scale_ue4m3_cuda(float target_scale) {
    if (!(target_scale > 0.0f) || !isfinite(target_scale)) {
        return 0;
    }

    uint8_t lo = 1;
    uint8_t hi = 0x7e;
    while (lo < hi) {
        const uint8_t mid = lo + (hi - lo) / 2;
        if (rocmfpx_ue4m3_to_fp32_finite(mid) < target_scale) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo == 1) {
        return 1;
    }

    const float hi_scale = rocmfpx_ue4m3_to_fp32_finite(lo);
    const float lo_scale = rocmfpx_ue4m3_to_fp32_finite((uint8_t) (lo - 1));
    return (target_scale - lo_scale <= hi_scale - target_scale) ? (uint8_t) (lo - 1) : lo;
}

static __device__ __forceinline__ int8_t rocmfp4_decode_i8(uint8_t q) {
    q &= 0x0f;
    const int mag3 = q & 0x07;
    const int mag = mag3 <= 4 ? mag3 : 2*mag3 - 4;
    return (q & 0x08) ? -mag : mag;
}
