// Exhaustive check of the FP2 byte -> four-int8 unpack used by the ROCMFP2
// HIP kernels (vecdotq.cuh, rocmfpx_pack4_fp2_bits8_vec_cuda).
//
// The HIP path was reduced from spread + v_perm_b32 to the spread alone on
// the grounds that the FP2 code table is the identity. That claim is a
// property of two macros and one bit pattern, so it is checked here on all
// 256 input bytes against the switch-based decoder the CUDA path uses, and
// against a literal model of V_PERM_B32's byte select with the old operands.
// Host test, no GPU.

#include <stdint.h>
#include <stdio.h>

#include "../engine/ggml/rocmfpx/rocmfpx.h"

static int g_fail = 0;

static int decode_code(uint32_t code) {
    switch (code & 3u) {
        case 0: return ROCMFP2_KVALUE_0_I8;
        case 1: return ROCMFP2_KVALUE_1_I8;
        case 2: return ROCMFP2_KVALUE_2_I8;
        default: return ROCMFP2_KVALUE_3_I8;
    }
}

// Reference: one int8 per 2-bit field, little-endian into a 32-bit word.
static uint32_t unpack_reference(uint32_t bits8) {
    uint32_t w = 0;
    for (int i = 0; i < 4; ++i) {
        w |= ((uint32_t) (uint8_t) (int8_t) decode_code(bits8 >> (2 * i))) << (8 * i);
    }
    return w;
}

// The current HIP expression, verbatim.
static uint32_t unpack_hip(uint32_t bits8) {
    return ((bits8 >> 0) & 3u) |
           (((bits8 >> 2) & 3u) << 8) |
           (((bits8 >> 4) & 3u) << 16) |
           (((bits8 >> 6) & 3u) << 24);
}

// V_PERM_B32 D = perm(S0, S1, sel): for each result byte i, selector byte
// sel[i] in 0..7 picks byte sel[i] of the 64-bit {S0:S1} with S1 as the low
// word (RDNA3.5 ISA, V_PERM_B32). Selectors here are 0..3, so only S1 is read.
static uint32_t perm_b32(uint32_t s0, uint32_t s1, uint32_t sel) {
    const uint64_t src = ((uint64_t) s0 << 32) | s1;
    uint32_t d = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = (sel >> (8 * i)) & 0xffu;
        const uint32_t byte = (b < 8) ? (uint32_t) ((src >> (8 * b)) & 0xffu) : 0u;
        d |= byte << (8 * i);
    }
    return d;
}

static uint32_t unpack_old_hip(uint32_t bits8) {
    const uint32_t values =
        ((uint32_t) (uint8_t) (int8_t) ROCMFP2_KVALUE_0_I8) |
        ((uint32_t) (uint8_t) (int8_t) ROCMFP2_KVALUE_1_I8 << 8) |
        ((uint32_t) (uint8_t) (int8_t) ROCMFP2_KVALUE_2_I8 << 16) |
        ((uint32_t) (uint8_t) (int8_t) ROCMFP2_KVALUE_3_I8 << 24);
    return perm_b32(0, values, unpack_hip(bits8));
}

int main(void) {
    for (int i = 0; i < 256; ++i) {
        const uint32_t r = unpack_reference((uint32_t) i);
        const uint32_t h = unpack_hip((uint32_t) i);
        const uint32_t o = unpack_old_hip((uint32_t) i);
        if (r != h || r != o) {
            if (g_fail < 8) {
                printf("FAIL: x=0x%02x reference=0x%08x hip=0x%08x old_hip=0x%08x\n", i, r, h, o);
            }
            g_fail++;
        }
    }
    printf("%s: FP2 unpack %s on all 256 inputs\n", g_fail ? "FAIL" : "PASS",
           g_fail ? "diverges" : "bit-identical (reference, spread-only, spread+perm)");
    return g_fail != 0;
}
