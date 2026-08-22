// Exhaustive equivalence test for the branch-free UE4M3 scale decode.
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

int main(void) {
    int mismatches = 0;
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
