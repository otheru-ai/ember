// Exhaustive equivalence test for the UE4M3 scale decoders (branch-free
// closed form and the f16-converter form).
//
// rocmfpx_ue4m3_to_fp32_finite is the hottest scalar helper in the ROCMFP
// path -- it runs twice per block in both the MMQ prefill loader
// (mmq.cuh:597) and the MMVQ decode dot product (vecdotq.cuh:526-527). It was
// rewritten branch-free for gfx1151 (see the rationale block in
// engine/ggml/rocmfp4/rocmfp4_hip_scale.cuh).
//
// The domain is a single uint8_t, so equivalence is not a sampling question:
// all 256 inputs are checked bit-for-bit against the previous implementation,
// which is embedded here as the reference. Anything less would be inadequate,
// because DSpark speculative decode requires the target's numerics to be
// exactly reproducible -- a one-ULP change in a scale can desynchronise
// draft/target agreement (see the same warning at quantize.cu:251).
//
// This is a host test: the function is pure integer/float bit manipulation
// with no GPU dependency, so it runs in the GPU-free gauntlet.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (cond) {                                                        \
            g_pass++;                                                      \
        } else {                                                           \
            g_fail++;                                                      \
            printf("FAIL: %s\n", msg);                                     \
        }                                                                  \
    } while (0)

static float u32_as_f32(uint32_t b) {
    float f;
    memcpy(&f, &b, sizeof f);
    return f;
}

static uint32_t f32_as_u32(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof b);
    return b;
}

// Reference: the implementation that shipped before the rewrite, verbatim.
static float ue4m3_reference(uint8_t x) {
    if (x > 0x7e) {
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    if (exp == 0) {
        return (float) man * (1.0f / 1024.0f);
    }
    const uint32_t bits = ((uint32_t) exp + 119u) << 23 | ((uint32_t) man << 20);
    return u32_as_f32(bits);
}

// Branch-free form, mirroring rocmfp4_hip_scale.cuh exactly.
static float ue4m3_branchfree(uint8_t x) {
    const uint32_t xi  = x;
    const uint32_t exp = (xi >> 3) & 0xFu;
    const uint32_t man = xi & 0x7u;

    uint32_t m = (exp == 0u) ? (man << 1) : (man | 8u);
    m = (xi > 0x7eu) ? 0u : m;

    return (float) m * u32_as_f32((exp + 116u) << 23);
}

// Software IEEE binary16 -> binary32, exact, subnormals preserved. This is
// the reference for the f16-converter form: it stands in for V_CVT_F32_F16
// without trusting either the host compiler's _Float16 or the GPU's denormal
// mode, so a mismatch here is a mismatch in the algebra, not the toolchain.
static float f16_bits_to_f32_soft(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man        = h & 0x3FFu;
    if (exp == 0) {
        if (man == 0) {
            return u32_as_f32(sign);
        }
        // Subnormal: value = man * 2^-24. Normalise into a binary32.
        int e = -1;
        do {
            man <<= 1;
            e++;
        } while ((man & 0x400u) == 0);
        man &= 0x3FFu;
        return u32_as_f32(sign | ((uint32_t) (127 - 15 - e) << 23) | (man << 13));
    }
    if (exp == 0x1F) {
        return u32_as_f32(sign | 0x7F800000u | (man << 13));
    }
    return u32_as_f32(sign | ((exp + 127u - 15u) << 23) | (man << 13));
}

// f16-converter form, mirroring rocmfp4_ue4m3_bits_to_fp32_f16 with the
// _finite variant's input clamp (rocmfp4_hip_scale.cuh, ROCMFP4_SCALE_DECODE_F16).
static float ue4m3_f16form(uint8_t x) {
    const uint32_t xi = (x > 0x7eu) ? 0u : (uint32_t) x;
    const uint16_t h  = (uint16_t) ((xi << 7) & 0x3f80u);
    return f16_bits_to_f32_soft(h) * 128.0f;
}

// The half_finite variant has no clamp; above 0x7e it must agree with the
// closed form it replaces, which is what makes the two selectable at compile
// time without changing which bytes decode to what.
static float ue4m3_half_closed(uint8_t x) {
    const uint32_t xi  = x;
    const uint32_t exp = (xi >> 3) & 0xFu;
    const uint32_t man = xi & 0x7u;
    const uint32_t m   = (exp == 0u) ? (man << 1) : (man | 8u);
    return (float) m * u32_as_f32((exp + 116u) << 23);
}

static float ue4m3_half_f16form(uint8_t x) {
    const uint16_t h = (uint16_t) (((uint32_t) x << 7) & 0x3f80u);
    return f16_bits_to_f32_soft(h) * 128.0f;
}

#if defined(__FLT16_MANT_DIG__)
// Where the host compiler has _Float16, also run the exact expression the
// device header compiles, so the software decoder above is itself checked.
static float ue4m3_half_f16form_native(uint8_t x) {
    const uint16_t h = (uint16_t) (((uint32_t) x << 7) & 0x3f80u);
    _Float16 f;
    memcpy(&f, &h, sizeof f);
    return (float) f * 128.0f;
}
#endif

int main(void) {
    int mismatches = 0;
    // Sanity on the software f16 decoder itself: one normal, one subnormal,
    // the smallest subnormal, zero.
    CHECK(f16_bits_to_f32_soft(0x3C00) == 1.0f, "soft f16: 1.0");
    CHECK(f16_bits_to_f32_soft(0x0400) == 1.0f / 16384.0f, "soft f16: min normal 2^-14");
    CHECK(f16_bits_to_f32_soft(0x0001) == 1.0f / 16777216.0f, "soft f16: min subnormal 2^-24");
    CHECK(f16_bits_to_f32_soft(0x0000) == 0.0f, "soft f16: +0");

    {
        int mm_finite = 0, mm_half = 0, mm_native = 0;
        for (int i = 0; i < 256; ++i) {
            const uint8_t x = (uint8_t) i;
            if (f32_as_u32(ue4m3_f16form(x)) != f32_as_u32(ue4m3_reference(x))) {
                if (mm_finite < 8) {
                    printf("FAIL: f16 form x=0x%02x reference=0x%08x f16form=0x%08x\n",
                           i, f32_as_u32(ue4m3_reference(x)), f32_as_u32(ue4m3_f16form(x)));
                }
                mm_finite++;
            }
            if (f32_as_u32(ue4m3_half_f16form(x)) != f32_as_u32(ue4m3_half_closed(x))) {
                if (mm_half < 8) {
                    printf("FAIL: half f16 form x=0x%02x closed=0x%08x f16form=0x%08x\n",
                           i, f32_as_u32(ue4m3_half_closed(x)), f32_as_u32(ue4m3_half_f16form(x)));
                }
                mm_half++;
            }
#if defined(__FLT16_MANT_DIG__)
            if (f32_as_u32(ue4m3_half_f16form_native(x)) != f32_as_u32(ue4m3_half_f16form(x))) {
                mm_native++;
            }
#endif
        }
        CHECK(mm_finite == 0, "f16-converter UE4M3 decode must be bit-identical to the reference on all 256 inputs");
        CHECK(mm_half == 0, "f16-converter half_finite decode must be bit-identical to the closed form on all 256 inputs");
        CHECK(mm_native == 0, "host _Float16 conversion must agree with the software f16 decoder on all 256 inputs");
        CHECK(ue4m3_half_f16form(0x7f) == 240.0f, "half_finite f16 form: 0x7f decodes to 240 (no clamp)");
    }

    for (int i = 0; i < 256; ++i) {
        const uint8_t x = (uint8_t) i;
        const uint32_t a = f32_as_u32(ue4m3_reference(x));
        const uint32_t b = f32_as_u32(ue4m3_branchfree(x));
        if (a != b) {
            if (mismatches < 8) {
                printf("FAIL: x=0x%02x reference=0x%08x branchfree=0x%08x\n", i, a, b);
            }
            mismatches++;
        }
    }
    CHECK(mismatches == 0, "branch-free UE4M3 decode must be bit-identical on all 256 inputs");

    // Spot-check the two structural cases the closed form unifies, so a future
    // edit that breaks only one of them names itself.
    CHECK(ue4m3_branchfree(0x00) == 0.0f, "zero encodes to +0");
    CHECK(ue4m3_branchfree(0x03) == 3.0f / 1024.0f, "denormal path: man/1024");
    CHECK(ue4m3_branchfree(0x08) == 1.0f / 128.0f, "normal path: exp=1, man=0");
    CHECK(ue4m3_branchfree(0x7f) == 0.0f, "non-finite range clamps to 0");
    CHECK(ue4m3_branchfree(0xff) == 0.0f, "above the finite range clamps to 0");

    printf("%s: %d passed, %d failed\n", g_fail ? "FAIL" : "PASS", g_pass, g_fail);
    return g_fail != 0;
}
