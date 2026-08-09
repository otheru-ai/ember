#include <cstdio>

#include "../engine/dflash/common/model_backend.h"
#include "../engine/dflash/common/prefill_attention_mode.h"
#include "../engine/dflash/deepseek4/deepseek4_attention_shape.h"

using dflash::common::PrefillAttentionMode;
using dflash::common::select_prefill_step_mode;
using dflash::common::deepseek4_attention_shape;
using dflash::common::deepseek4_decode_flash_eligible;
using dflash::common::deepseek4_decode_wmma_applicable;
using dflash::common::deepseek4_decode_flash_shmem_bytes;
using dflash::common::deepseek4_view_bound;
using dflash::common::GenerateResult;
using dflash::common::ModelBackend;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (condition) ++g_pass;                                            \
        else { ++g_fail; std::printf("  FAIL: %s\n", message); }           \
    } while (0)

int main() {
    std::printf("ember prefill policy tests\n");
    CHECK(select_prefill_step_mode(PrefillAttentionMode::Sparse, false,
                                   false, 0, 0) ==
              PrefillAttentionMode::Sparse,
          "target-only decode can retain sparse prefill");
    CHECK(select_prefill_step_mode(PrefillAttentionMode::Sparse, false,
                                   true, 4095, 4096) ==
              PrefillAttentionMode::Sparse,
          "DSpark prefix remains on configured batched prefill");
    CHECK(select_prefill_step_mode(PrefillAttentionMode::Sparse, false,
                                   true, 4096, 4096) ==
              PrefillAttentionMode::Exact,
          "DSpark capture window switches to exact prefill");
    CHECK(select_prefill_step_mode(PrefillAttentionMode::Dense, true,
                                   false, 0, 1024) ==
              PrefillAttentionMode::Exact,
          "explicit reference request forces exact prefill");

    // A one-token suffix after snapshot restore is a prefill, but every
    // multi-token/cached-input optimization is ineligible. It must reach the
    // dynamic graph's per-row raw-KV writer at row zero.
    const auto q1_dynamic = deepseek4_attention_shape(
        1, 128, false, true, true, true, false);
    CHECK(!q1_dynamic.causal_batch && !q1_dynamic.layer_major_batch &&
              !q1_dynamic.fused_causal &&
              q1_dynamic.first_raw_kv_write == 0,
          "q1 restored sparse prefill uses the dynamic row-zero writer");
    const auto q1_cached = deepseek4_attention_shape(
        1, 128, true, false, true, true, true);
    CHECK(!q1_cached.causal_batch && !q1_cached.layer_major_batch &&
              !q1_cached.fused_causal,
          "q1 never enters fused-causal attention even with cached inputs");
    const auto q2_explicit = deepseek4_attention_shape(
        2, 128, false, true, true, false, false);
    CHECK(q2_explicit.causal_batch && !q2_explicit.layer_major_batch &&
              !q2_explicit.fused_causal,
          "explicit q2 uses causal batching without layer-major attention");
    const auto q2_optimized = deepseek4_attention_shape(
        2, 128, false, true, true, true, false);
    CHECK(q2_optimized.causal_batch && q2_optimized.layer_major_batch,
          "optimized q2 makes layer-major eligibility explicit");
    const auto q2_cached = deepseek4_attention_shape(
        2, 128, true, false, true, true, true);
    CHECK(!q2_cached.causal_batch && !q2_cached.layer_major_batch &&
              q2_cached.fused_causal,
          "cached q2 with an attention mask uses fused-causal attention");
    CHECK(deepseek4_attention_shape(
              129, 128, false, true, true, true, false)
              .first_raw_kv_write == 1,
          "raw-KV persistence writes only the final SWA tail");
    // Decode flash attention is confined to the fused full-ring path, where the
    // row mask — not slot order — decides visibility.
    const std::size_t lds = 63u * 1024u;
    CHECK(deepseek4_decode_flash_eligible(1, 512, 128 + 512, 512, true, 32, lds),
          "fused full-ring decode qualifies for flash attention");
    CHECK(!deepseek4_decode_flash_eligible(1, 512, 96, 0, false, 32, lds),
          "unmasked q1 decode keeps the explicit reduction");
    CHECK(!deepseek4_decode_flash_eligible(2, 512, 128 + 512, 512, true, 32, lds),
          "multi-token steps stay on the prefill scheduler's decision");
    CHECK(!deepseek4_decode_flash_eligible(1, 256, 128 + 512, 512, true, 32, lds),
          "decode flash attention is confined to the D=512 kernel");
    CHECK(!deepseek4_decode_flash_eligible(1, 512, 0, 0, true, 32, lds),
          "an empty attention span never reaches the kernel");

    // Score staging is one float per KV row plus two per compressed block, so a
    // long compressed span falls back rather than overrunning LDS.
    CHECK(deepseek4_decode_flash_shmem_bytes(128 + 512, 512, 32) ==
              (128 + 512 + 2 * 16) * sizeof(float),
          "decode flash shared memory counts rows and compressed blocks");
    CHECK(deepseek4_decode_flash_shmem_bytes(128, 0, 32) == 128 * sizeof(float),
          "an uncompressed layer stages scores only");
    CHECK(deepseek4_decode_flash_eligible(1, 512, 15000, 14872, true, 32, lds),
          "a long-but-fitting compressed span still uses flash attention");
    CHECK(!deepseek4_decode_flash_eligible(1, 512, 16512, 16384, true, 32, lds),
          "a compressed span past the LDS budget falls back to explicit");

    // The WMMA decode kernel's shared memory is constant in n_attn, so when it
    // is applicable the span budget must not refuse it -- that would deny the
    // long contexts where it is worth the most.
    CHECK(deepseek4_decode_wmma_applicable(512, 64),
          "D=512 with 64 query heads reaches the WMMA decode kernel");
    CHECK(!deepseek4_decode_wmma_applicable(512, 32),
          "other head counts do not reach the WMMA decode kernel");
    CHECK(deepseek4_decode_flash_eligible(1, 512, 32896, 32768, true, 32, lds, 64),
          "WMMA-applicable shapes ignore the span-based LDS budget");
    CHECK(!deepseek4_decode_flash_eligible(1, 512, 32896, 32768, true, 32, lds, 32),
          "non-WMMA head counts still honour the span budget");
    CHECK(!deepseek4_decode_flash_eligible(2, 512, 1024, 896, true, 32, lds, 64),
          "multi-token steps stay ineligible regardless of head count");

    CHECK(deepseek4_view_bound(0, 0, 2048, 2048).valid,
          "q1 raw-KV row exactly fits its base allocation");
    CHECK(!deepseek4_view_bound(0, 1, 2048, 2048).valid,
          "q1 raw-KV row rejects a one-byte source offset overrun");
    CHECK(!deepseek4_view_bound(
              std::numeric_limits<std::size_t>::max(), 1, 1, 2048).valid,
          "raw-KV view guard rejects offset addition overflow");
    GenerateResult first;
    first.prefill_tokens = 1000;
    first.prefill_s = 4.0;
    first.prefill_mode = "hybrid";
    GenerateResult retry;
    retry.prefill_tokens = 1000;
    retry.prefill_s = 5.0;
    retry.prefill_mode = "sparse";
    GenerateResult merged =
        ModelBackend::merge_empty_spec_retry_result(first, retry);
    CHECK(merged.prefill_tokens == 2000 && merged.prefill_s == 9.0,
          "hidden AR retry accounts for both prefill attempts");
    CHECK(merged.prefill_mode == "mixed" &&
              merged.prefill_reason == "empty_spec_retry",
          "hidden AR retry reports its mixed execution policy");
    std::printf("──────────────────────────────\n");
    std::printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
