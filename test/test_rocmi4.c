// GPU-free format contract for ROCmFPX Q4_0_ROCMI4 (upstream 16d05b8).
#include "rocmfpx.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_pass;
static int g_fail;
#define CHECK(c, m) do { if (c) ++g_pass; else { ++g_fail; printf("FAIL: %s\n", m); } } while (0)

int main(void) {
    CHECK(GGML_TYPE_Q4_0_ROCMI4 == 108, "canonical GGUF tensor type is 108");
    CHECK(GGML_FTYPE_MOSTLY_Q4_0_ROCMI4 == 118, "canonical GGUF file type is 118");
    CHECK(GGML_FTYPE_MOSTLY_Q2_0_ROCMFP2 == 119 &&
          GGML_FTYPE_MOSTLY_Q2_0_ROCMFP2_STRIX == 120,
          "legacy Ember Q2 recipe IDs no longer collide with canonical ROCMI4");
    CHECK(sizeof(block_rocmi4) == 17, "ROCMI4 block is exactly 17 bytes");
    CHECK(rocmfpx_row_size_i4(32) == 17, "32 values occupy one block");
    CHECK(rocmfpx_row_size_i4(64) == 34, "row size scales by whole blocks");

    float src[32];
    for (int i = 0; i < 16; ++i) {
        src[i] = (float) ((i % 15) - 7) * 0.5f;
        src[i + 16] = (float) (7 - (i % 15)) * 0.5f;
    }
    block_rocmi4 q;
    rocmfpx_quantize_row_i4_ref(src, &q, 32);
    CHECK(q.e == 0x38, "amax/7 selects canonical UE4M3 scale 0.5");
    bool packed = true;
    for (int i = 0; i < 16; ++i) {
        const uint8_t want = (uint8_t) ((((i % 15) - 7) & 15) | (((7 - (i % 15)) & 15) << 4));
        packed = packed && q.qs[i] == want;
    }
    CHECK(packed, "low/high nibbles preserve the split-half signed layout");

    float dst[32];
    rocmfpx_dequantize_row_i4(&q, dst, 32);
    CHECK(memcmp(src, dst, sizeof src) == 0, "all signed nibble codes round-trip exactly");

    q.e = 0x38;
    for (int i = 0; i < 16; ++i) q.qs[i] = (uint8_t) (i | ((15 - i) << 4));
    rocmfpx_dequantize_row_i4(&q, dst, 32);
    bool signed_codes = true;
    for (int i = 0; i < 16; ++i) {
        const int lo = i < 8 ? i : i - 16;
        const int hi_code = 15 - i;
        const int hi = hi_code < 8 ? hi_code : hi_code - 16;
        signed_codes = signed_codes && dst[i] == (float) lo*0.5f && dst[i + 16] == (float) hi*0.5f;
    }
    CHECK(signed_codes, "all sixteen two's-complement nibble codes decode correctly");
    CHECK(rocmfpx_validate_row_data_i4(&q, sizeof q), "finite scale validates");
    CHECK(!rocmfpx_validate_row_data_i4(&q, sizeof q - 1), "truncated block is rejected");
    q.e = 0x7f;
    CHECK(!rocmfpx_validate_row_data_i4(&q, sizeof q), "non-finite UE4M3 scale is rejected");

    float weights[32];
    for (int i = 0; i < 32; ++i) weights[i] = (i == 0) ? 1000.0f : 1.0f;
    block_rocmi4 weighted;
    CHECK(rocmfpx_quantize_i4(src, &weighted, 1, 32, weights) == sizeof weighted,
          "importance-weighted quantizer reports exact byte count");
    CHECK(rocmfpx_validate_row_data_i4(&weighted, sizeof weighted),
          "importance-weighted quantizer emits a valid block");

    printf("%s: %d passed, %d failed\n", g_fail ? "FAIL" : "PASS", g_pass, g_fail);
    return g_fail != 0;
}
