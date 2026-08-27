#pragma once

/*
 * Exact signed-q8 decomposition for gfx1151 mixed-signedness I4 arithmetic.
 *
 * AMD's RDNA3.5 machine-readable ISA describes V_DOT8_I32_IU4 and
 * V_WMMA_I32_16X16X16_IU4 operands as signed or unsigned according to their
 * independent NEG modifiers.  Splitting a two's-complement q8 value into an
 * unsigned low nibble and a signed high nibble therefore preserves it exactly:
 *
 *     q = low_u4 + 16 * high_i4
 *
 * Source: AMD GPUOpen machine-readable ISA archive, amdgpu_isa_rdna3_5.xml
 * (archive-entry timestamp 2026-08-04, internal release date 2026-02-20),
 * downloaded from
 * https://gpuopen.com/download/machine-readable-isa/latest/
 *
 * This helper only exposes that integer identity.  It does not quantize an
 * activation, change the ROCMI4 storage format, or enable a GPU kernel path.
 */

#include <stdint.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define ROCMI4_EXACT_HD __host__ __device__
#else
#define ROCMI4_EXACT_HD
#endif

typedef struct {
    uint8_t low_u4;
    int8_t high_i4;
} rocmi4_q8_i4_parts;

static ROCMI4_EXACT_HD inline rocmi4_q8_i4_parts
rocmi4_q8_decompose_i4(int8_t value) {
    const uint8_t bits = (uint8_t) value;
    const uint8_t high = (uint8_t) (bits >> 4);
    const rocmi4_q8_i4_parts parts = {
        (uint8_t) (bits & 0x0fu),
        (int8_t) (high < 8u ? (int) high : (int) high - 16),
    };
    return parts;
}

static ROCMI4_EXACT_HD inline int
rocmi4_q8_recompose_i4(rocmi4_q8_i4_parts parts) {
    return (int) parts.low_u4 + 16 * (int) parts.high_i4;
}

/*
 * Pack eight q8 bytes held in two little-endian 32-bit words into one IU4
 * operand word.  The low result contains unsigned low nibbles; the high
 * result contains the two's-complement signed high nibbles.  Nibble order is
 * q0..q7 from least to most significant, matching the consecutive K values
 * loaded by the RDNA3.5 mirrored WMMA fragment.
 */
static ROCMI4_EXACT_HD inline uint32_t
rocmi4_compact_q8_nibbles4(uint32_t values) {
    values &= UINT32_C(0x0f0f0f0f);
    return (values & UINT32_C(0x0000000f)) |
           ((values & UINT32_C(0x00000f00)) >> 4) |
           ((values & UINT32_C(0x000f0000)) >> 8) |
           ((values & UINT32_C(0x0f000000)) >> 12);
}

/*
 * ROCMI4 stores K0..K15 in the low nibbles and K16..K31 in the high
 * nibbles of sixteen consecutive bytes.  Compact two consecutive groups of
 * four storage bytes without byte-interleaving them: one IU4 operand word
 * must remain K-contiguous within a gfx1151 A fragment.
 */
static ROCMI4_EXACT_HD inline uint32_t
rocmi4_pack_split_half_low_i4(uint32_t first4, uint32_t next4) {
    return rocmi4_compact_q8_nibbles4(first4) |
           (rocmi4_compact_q8_nibbles4(next4) << 16);
}

static ROCMI4_EXACT_HD inline uint32_t
rocmi4_pack_split_half_high_i4(uint32_t first4, uint32_t next4) {
    return rocmi4_compact_q8_nibbles4(first4 >> 4) |
           (rocmi4_compact_q8_nibbles4(next4 >> 4) << 16);
}

static ROCMI4_EXACT_HD inline uint32_t
rocmi4_pack_q8x8_low_u4(uint32_t first4, uint32_t next4) {
    return rocmi4_pack_split_half_low_i4(first4, next4);
}

static ROCMI4_EXACT_HD inline uint32_t
rocmi4_pack_q8x8_high_i4(uint32_t first4, uint32_t next4) {
    return rocmi4_pack_split_half_high_i4(first4, next4);
}

/* The runtime experiment is selected only by the exact, documented value. */
static ROCMI4_EXACT_HD inline int
rocmi4_w4a8_iu4_requested(const char * value) {
    return value != (const char *) 0 && value[0] == '1' && value[1] == '\0';
}

#undef ROCMI4_EXACT_HD
