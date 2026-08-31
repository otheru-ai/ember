#include <limits>
#include <cstdio>

#include "../engine/dflash/common/model_backend.h"
#include "../engine/dflash/common/prefill_attention_mode.h"
#include "../engine/dflash/common/prefill_validation.h"
#include "../engine/dflash/deepseek4/deepseek4_attention_shape.h"

using dflash::common::PrefillAttentionMode;
using dflash::common::PrefillMarginDecision;
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

    const PrefillMarginDecision exact =
        dflash::common::validate_prefill_margin(
            {3, 4}, {3, 4}, {}, {});
    CHECK(exact.streams_exact && exact.accepted && !exact.margin_checked,
          "token-exact prefill passes without manufacturing a margin");

    // These three fixtures isolate the MARGIN clause. Their two- and
    // three-element vocabularies make total variation meaningless -- any
    // visible logit change is most of the distribution -- so they pass a
    // permissive tv_threshold and the TV clause is covered separately below.
    const PrefillMarginDecision exact_with_noise =
        dflash::common::validate_prefill_margin(
            {3, 4}, {3, 4},
            {{3.0f, 2.0f}, {5.0f, 1.0f}},
            {{3.1f, 2.0f}, {5.5f, 1.0f}},
            /*serving_temperature=*/0.6f, /*tv_threshold=*/1.0f);
    CHECK(exact_with_noise.streams_exact && exact_with_noise.accepted &&
              exact_with_noise.margin_checked &&
              exact_with_noise.numerics_index == 1 &&
              exact_with_noise.max_abs_logit_delta == 0.5f,
          "token-exact prefill reports its largest observed logit delta");

    const PrefillMarginDecision within_delta =
        dflash::common::validate_prefill_margin(
            {0}, {1}, {{1.0f, 0.9f, 0.0f}}, {{0.8f, 1.0f, 0.0f}},
            /*serving_temperature=*/0.6f, /*tv_threshold=*/1.0f);
    CHECK(!within_delta.streams_exact && within_delta.margin_checked &&
              within_delta.accepted && within_delta.mismatch_index == 0 &&
              within_delta.expected_token == 0 &&
              within_delta.actual_token == 1,
          "prefill disagreement passes when path delta exceeds q1 margin");

    const PrefillMarginDecision outside_delta =
        dflash::common::validate_prefill_margin(
            {0}, {1}, {{1.0f, 0.7f}}, {{0.8f, 0.9f}});
    CHECK(outside_delta.margin_checked && !outside_delta.accepted,
          "prefill disagreement fails when q1 margin exceeds path delta");

    const PrefillMarginDecision tied_boundary =
        dflash::common::validate_prefill_margin(
            {0}, {1}, {{1.0f, 0.5f}}, {{0.5f, 1.0f}});
    CHECK(tied_boundary.margin_checked && !tied_boundary.accepted,
          "prefill criterion uses the decided strict margin inequality");

    const PrefillMarginDecision eos_length_mismatch =
        dflash::common::validate_prefill_margin(
            {}, {1}, {{1.0f, 0.9f}}, {{0.8f, 1.0f}},
            /*serving_temperature=*/0.6f, /*tv_threshold=*/1.0f);
    CHECK(eos_length_mismatch.margin_checked &&
              eos_length_mismatch.accepted &&
              eos_length_mismatch.expected_token == -1,
          "immediate q1 stop still compares its seed-logit margin");

    // --- total-variation clause ---------------------------------------
    // Regression for the measured hole: a token-exact run whose distribution
    // is badly perturbed must NOT be accepted. Before this clause the observed
    // width-4 dense-MMQ control returned exact and accepted with TV 0.80.
    std::vector<float> broad_q1(512, 0.0f), broad_prod(512, 0.0f);
    for (size_t i = 0; i < broad_q1.size(); ++i) {
        broad_q1[i] = 0.001f * static_cast<float>(i % 7);
        broad_prod[i] = broad_q1[i];
    }
    broad_q1[3] = 9.0f;   broad_prod[3] = 9.0f;    // same argmax, both sides
    broad_q1[9] = 1.0f;   broad_prod[9] = 8.5f;    // large distributional move
    const PrefillMarginDecision exact_but_perturbed =
        dflash::common::validate_prefill_margin(
            {3}, {3}, {broad_q1}, {broad_prod});
    CHECK(exact_but_perturbed.streams_exact &&
              exact_but_perturbed.tv_checked &&
              !exact_but_perturbed.tv_within_bound &&
              !exact_but_perturbed.accepted,
          "token-exact prefill is rejected when the distribution diverges");

    // The converse: genuinely equivalent distributions stay accepted, so the
    // clause does not simply reject everything with logits attached.
    std::vector<float> near_q1(512, 0.0f), near_prod(512, 0.0f);
    for (size_t i = 0; i < near_q1.size(); ++i) {
        near_q1[i] = 0.001f * static_cast<float>(i % 7);
        near_prod[i] = near_q1[i] + 1e-6f;
    }
    near_q1[3] = 9.0f; near_prod[3] = 9.0f + 1e-6f;
    const PrefillMarginDecision exact_and_equivalent =
        dflash::common::validate_prefill_margin(
            {3}, {3}, {near_q1}, {near_prod});
    CHECK(exact_and_equivalent.streams_exact &&
              exact_and_equivalent.tv_checked &&
              exact_and_equivalent.tv_within_bound &&
              exact_and_equivalent.accepted,
          "token-exact prefill with an equivalent distribution still passes");

    // Non-finite logits must fail closed rather than yielding TV 0.
    std::vector<float> nan_row(4, 0.0f);
    nan_row[1] = std::numeric_limits<float>::quiet_NaN();
    const PrefillMarginDecision nonfinite =
        dflash::common::validate_prefill_margin(
            {0}, {0}, {nan_row}, {std::vector<float>(4, 0.0f)});
    CHECK(!nonfinite.tv_checked && !nonfinite.tv_within_bound &&
              !nonfinite.accepted,
          "non-finite logits fail the distributional check closed");

    // With no logits captured at all there is nothing to bound, so the
    // token-only verdict stands as it did before this clause existed.
    const PrefillMarginDecision exact_no_logits =
        dflash::common::validate_prefill_margin({3, 4}, {3, 4}, {}, {});
    CHECK(exact_no_logits.streams_exact && exact_no_logits.accepted &&
              !exact_no_logits.tv_checked,
          "token-exact without captured logits keeps the prior verdict");

    const PrefillMarginDecision missing_logits =
        dflash::common::validate_prefill_margin({0}, {1}, {}, {});
    CHECK(!missing_logits.margin_checked && !missing_logits.accepted,
          "prefill disagreement fails closed without captured logits");
    std::printf("──────────────────────────────\n");
    std::printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
