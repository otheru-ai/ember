// DeepSeek-V4-Flash DSpark speculative decode: DFlashTarget adapter + spec loop.
//
// The drafter (DSparkDrafter, deepseek4_dspark.{h,cpp}) proposes block_size
// candidates conditioned on captured target features. The default verifier is
// target-exact: it evaluates candidates through the ordinary q=1 target graph
// and stops before the first rejected token is written. Feature capture is a
// passive output of that graph. The DSpark Markov head (common/dspark_head.cpp)
// is target-agnostic and reused verbatim.
//
// Batched throughput mode (DFLASH_DS4_BATCH_VERIFY=1, with
// DFLASH_DS4_APPROX_VERIFY retained as a compatibility alias): ONE batched
// verify per step, verify width capped
// at DS4_SPEC_Q=4 tokens (seed + 3 candidates). With q <= ratio(4) the verify
// crosses at most one compression boundary and never aliases rolling-state
// rows, so partial-accept rollback needs no full KV snapshot:
//   - at-risk raw ring rows are saved before the verify; a partial block
//     restores them all before q=1 replay while a full block stays committed,
//   - comp rows are index-addressed (pos / ratio)        -> idempotent,
//   - n_comp / n_index_comp are pure functions of commit position,
//   - the other non-idempotent state is the ratio-4 compressor's complete
//     eight-row prev/current window, a few KB/layer, saved host-side before
//     the verify and restored before partial replay.  Saving only prev was
//     insufficient: a q-wide verify can wrap rejected future tokens into the
//     current half before the q=1 fallback reaches the same boundary.
// As in Dwarfstar, a fully accepted block keeps the batched target state. A
// partial accept rolls all the way back and replays the accepted prefix through
// the ordinary q=1 target graph. This makes the rejection path target-exact
// without paying for a second target pass on the high-acceptance fast path.
// The legacy full-snapshot rollback remains available with
// DFLASH_DS4_FULL_SNAP=1 when batched verification is enabled.

#include "deepseek4_dspark.h"
#include "deepseek4_dspark_scheduler.h"
#include "common/dspark_draft_compute_xdna.h"
#include "deepseek4_internal.h"
#include "internal.h"
#include "common/dspark_head.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace dflash::common {

// ── DFlashTarget adapter over the DS4 target ────────────────────────────
class DeepSeek4DFlashTarget : public DFlashTarget {
public:
    DeepSeek4DFlashTarget(const DeepSeek4Weights & w, DeepSeek4Cache & cache,
                          ggml_backend_t backend, int device, ggml_backend_t snap_backend,
                          std::vector<int> capture_ids, int mask_tok,
                          bool strict_verify)
        : w_(w), cache_(cache), backend_(backend), device_(device), snap_backend_(snap_backend),
          capture_ids_(std::move(capture_ids)), mask_tok_(mask_tok),
          strict_verify_(strict_verify) {}

    ~DeepSeek4DFlashTarget() override { clear_snapshot(); }

    bool verify_batch(const std::vector<int32_t> & tokens, int base_pos, int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr,
                      bool capture_ssm_intermediates = false) override {
        (void) capture_ssm_intermediates;
        const int n = (int) tokens.size();
        embed_buf_.resize((size_t) n * w_.n_embd);
        if (!w_.embedder.embed(tokens.data(), n, embed_buf_.data())) {
            std::fprintf(stderr, "[ds4-verify] embed FAILED n=%d tok0=%d tok1=%d vocab=%d\n",
                         n, n > 0 ? tokens[0] : -1, n > 1 ? tokens[1] : -1, w_.n_vocab);
            return false;
        }
        // Exact prefix verification: run the ordinary q=1 fused target graph
        // with passive feature outputs. Stop as soon as its argmax disagrees
        // with the next proposal. Because the rejected token is never fed, KV
        // and compressor state already end at the accepted prefix; no snapshot
        // or replay is needed.
        if (strict_verify_) {
            return verify_exact_prefix_embedded(tokens, base_pos, last_tok,
                                                all_argmax);
        }
        std::vector<int32_t> am;
        // q==1 occurs at every ratio-4 boundary. The cached decode path now
        // exposes the same capture/all-logits hooks and avoids rebuilding the
        // fragile one-token verifier graph at those boundary positions.
        if (!deepseek4_dspark_verify_forward(backend_, device_, w_, cache_, capture_ids_,
                                             embed_buf_.data(), tokens.data(), n, base_pos, am,
                                             keep_logits_ ? &verify_logits_ : nullptr,
                                             verify_features_, telemetry_,
                                             /*allow_graph_reuse=*/true)) {
            return false;
        }
        if (am.empty()) return false;
        last_tok = am.back();
        verify_n_ = n;
        if (all_argmax) *all_argmax = std::move(am);
        return true;
    }

    // Restore/replay seam used after a batched verifier rejects part of a
    // proposal. The caller must first restore target state to base_pos.
    bool verify_exact_prefix(const std::vector<int32_t> & tokens, int base_pos,
                             int & last_tok,
                             std::vector<int32_t> * all_argmax = nullptr) {
        const int n = (int) tokens.size();
        if (n <= 0) return false;
        embed_buf_.resize((size_t) n * w_.n_embd);
        if (!w_.embedder.embed(tokens.data(), n, embed_buf_.data())) return false;
        return verify_exact_prefix_embedded(tokens, base_pos, last_tok,
                                            all_argmax);
    }

    bool read_verify_logits(int n_tokens, std::vector<float> & out) override {
        if (!keep_logits_ || verify_logits_.empty()) return false;
        const size_t need = (size_t) n_tokens * w_.n_vocab;
        if (verify_logits_.size() < need) return false;
        out.assign(verify_logits_.begin(), verify_logits_.begin() + need);
        return true;
    }

    bool snapshot_kv() override { return deepseek4_snapshot_save(cache_, snap_backend_, snap_); }
    bool restore_kv() override { return deepseek4_snapshot_restore(snap_, cache_); }

    bool is_eos(int token) const override { return deepseek4_is_eos_tok(token, w_); }

    bool embed_tokens(const int32_t * tokens, int n, float * out) const override {
        return w_.embedder.embed(tokens, n, out);
    }

    bool project_hidden_to_tokens(const float * hidden, int n_tokens,
                                  std::vector<int32_t> & tokens_out) override {
        std::vector<float> logits;
        if (!project_hidden_to_logits(hidden, n_tokens, logits)) return false;
        tokens_out.resize(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            const float * row = logits.data() + (size_t) t * w_.n_vocab;
            int best = 0; float bv = row[0];
            for (int i = 1; i < w_.n_vocab; i++) if (row[i] > bv) { bv = row[i]; best = i; }
            tokens_out[t] = best;
        }
        return true;
    }

    // The drafter hidden is already out_norm'd (drafter tail); project with the
    // tied target lm_head only (mul_mat, no norm), matching the reference head.
    bool project_hidden_to_logits(const float * hidden, int n_tokens,
                                  std::vector<float> & logits_out) override {
        if (n_tokens <= 0) return false;
        ggml_init_params ip{};
        ip.mem_size = 32u * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return false;
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w_.n_embd, n_tokens);
        ggml_set_input(h);
        ggml_tensor * logits = ggml_mul_mat(ctx, w_.output, h);   // [n_vocab, n_tokens]
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return false;
        }
        ggml_backend_tensor_set(h, hidden, 0, sizeof(float) * (size_t) w_.n_embd * n_tokens);
        const ggml_status st = ggml_backend_graph_compute(backend_, gf);
        if (st != GGML_STATUS_SUCCESS) { ggml_gallocr_free(alloc); ggml_free(ctx); return false; }
        logits_out.resize((size_t) n_tokens * w_.n_vocab);
        ggml_backend_tensor_get(logits, logits_out.data(), 0, sizeof(float) * logits_out.size());
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return true;
    }

    ggml_tensor * lm_head_tensor() override { return w_.output; }
    int hidden_size() const override { return w_.n_embd; }
    int mask_token_id() const override { return mask_tok_; }
    const std::vector<int> & capture_layer_ids() const override { return capture_ids_; }

    void set_keep_logits(bool b) { keep_logits_ = b; }
    void set_telemetry(DeepSeek4StepTelemetry * t) { telemetry_ = t; }
    void set_strict_verify(bool b) { strict_verify_ = b; }
    const std::vector<float> & last_features() const { return verify_features_; }
    int last_verify_n() const { return verify_n_; }
    void clear_snapshot() { free_deepseek4_snapshot(snap_); }

private:
    bool verify_exact_prefix_embedded(
            const std::vector<int32_t> & tokens, int base_pos, int & last_tok,
            std::vector<int32_t> * all_argmax) {
        const int n = (int) tokens.size();
        if (n <= 0 || embed_buf_.size() < (size_t) n * w_.n_embd) return false;
        std::vector<int32_t> am_all;
        std::vector<float> feat_all;
        std::vector<float> logits_all;
        am_all.reserve(n);
        for (int t = 0; t < n; t++) {
            std::vector<int32_t> am1;
            std::vector<float> feat1;
            std::vector<float> logits1;
            if (!deepseek4_dspark_verify_forward(
                    backend_, device_, w_, cache_, capture_ids_,
                    embed_buf_.data() + (size_t) t * w_.n_embd,
                    tokens.data() + t, 1, base_pos + t, am1,
                    keep_logits_ ? &logits1 : nullptr, feat1, telemetry_,
                    /*allow_graph_reuse=*/true,
                    /*require_target_graph=*/true)) {
                return false;
            }
            if (am1.empty()) return false;
            am_all.push_back(am1[0]);
            feat_all.insert(feat_all.end(), feat1.begin(), feat1.end());
            if (keep_logits_) {
                logits_all.insert(logits_all.end(), logits1.begin(), logits1.end());
            }
            if (t + 1 < n && tokens[(size_t) t + 1] != am1[0]) break;
            // No token after EOS belongs to the sequence.  Besides avoiding
            // pointless work, this prevents an accepted speculative block from
            // advancing resident KV beyond the terminal token.
            if (deepseek4_is_eos_tok(tokens[(size_t)t], w_)) break;
        }
        verify_features_ = std::move(feat_all);
        if (keep_logits_) verify_logits_ = std::move(logits_all);
        last_tok = am_all.back();
        verify_n_ = (int) am_all.size();
        if (all_argmax) *all_argmax = std::move(am_all);
        return true;
    }

    const DeepSeek4Weights & w_;
    DeepSeek4Cache & cache_;
    ggml_backend_t backend_;
    int device_;
    ggml_backend_t snap_backend_;
    std::vector<int> capture_ids_;
    int mask_tok_;
    bool strict_verify_ = true;
    DeepSeek4Snapshot snap_{};
    DeepSeek4StepTelemetry * telemetry_ = nullptr;
    bool keep_logits_ = false;
    int verify_n_ = 0;
    std::vector<float> embed_buf_;
    std::vector<float> verify_logits_;
    std::vector<float> verify_features_;
};

namespace {

// Build the DraftWeights shim the target-agnostic dspark head expects (only the
// DSpark head fields + n_embd are read).
DraftWeights make_dspark_shim(const DSparkDrafter & d) {
    DraftWeights dw{};
    dw.n_embd = d.core.n_embd;
    dw.dspark.enabled = d.dspark_enabled;
    dw.dspark.markov_rank = d.markov_rank;
    dw.dspark.vocab_size = d.vocab_size;
    dw.dspark.confidence_dim = d.confidence_dim;
    dw.dspark.markov_w1 = d.markov_w1;
    dw.dspark.markov_w2 = d.markov_w2;
    dw.dspark.confidence_w = d.confidence_w;
    dw.dspark.confidence_b = d.confidence_b;
    return dw;
}

bool spec_env_flag(const char * name) {
    const char * v = std::getenv(name);
    return v && *v && *v != '0';
}

uint32_t spec_env_u32(const char * name, uint32_t fallback) {
    const char * value = std::getenv(name);
    if (!value || !*value) return fallback;
    char * end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > UINT_MAX) {
        std::fprintf(stderr,
                     "[ds4-spec] invalid %s='%s'; using %u\n",
                     name, value, fallback);
        return fallback;
    }
    return (uint32_t) parsed;
}

bool dspark_log_xdna_compare(
        const char * label,
        const std::vector<float> & actual,
        const std::vector<float> & reference) {
    if (actual.size() != reference.size() || actual.empty()) {
        std::fprintf(stderr,
                     "[ds4-spec] XDNA compare %s shape mismatch: "
                     "provider=%zu gpu=%zu\n",
                     label, actual.size(), reference.size());
        return false;
    }
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    double dot = 0.0;
    double actual_sq = 0.0;
    double reference_sq = 0.0;
    size_t nonfinite = 0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float a = actual[index];
        const float r = reference[index];
        if (!std::isfinite(a) || !std::isfinite(r)) {
            ++nonfinite;
            continue;
        }
        const float error = std::fabs(a - r);
        if (error > max_abs) max_abs = error;
        sum_abs += error;
        dot += static_cast<double>(a) * r;
        actual_sq += static_cast<double>(a) * a;
        reference_sq += static_cast<double>(r) * r;
    }
    const double cosine = actual_sq > 0.0 && reference_sq > 0.0
        ? dot / std::sqrt(actual_sq * reference_sq) : 0.0;
    const double mean_abs = sum_abs / static_cast<double>(actual.size());
    std::fprintf(stderr,
                 "[ds4-spec] XDNA compare %s: n=%zu nonfinite=%zu "
                 "max_abs=%.9g mean_abs=%.9g cosine=%.10f\n",
                 label, actual.size(), nonfinite, max_abs, mean_abs, cosine);

    // This is a hidden-boundary gate, not a bitwise gate: the provider changes
    // FP4 reduction order and uses compensated BF16 AIE weights.  These bounds
    // are tight enough to reject layout/shape failures while allowing the
    // expected backend rounding delta.  Token-for-token target verification
    // remains the final lossless-output gate.
    return nonfinite == 0 && max_abs <= 0.02f && cosine >= 0.99999;
}

DSparkSchedulerConfig spec_scheduler_config() {
    DSparkSchedulerConfig config;
    if (const char * value = std::getenv("DFLASH_DS4_SPEC_SCHEDULER")) {
        config.enabled = *value != '0';
    }
    config.window = spec_env_u32(
        "DFLASH_DS4_SPEC_SCHEDULER_WINDOW", config.window);
    config.skip_cycles = spec_env_u32(
        "DFLASH_DS4_SPEC_SCHEDULER_SKIP", config.skip_cycles);
    config.slow_skip_cycles = spec_env_u32(
        "DFLASH_DS4_SPEC_SCHEDULER_SLOW_SKIP", config.slow_skip_cycles);
    config.min_avg_accepted_milli = spec_env_u32(
        "DFLASH_DS4_SPEC_SCHEDULER_MIN_AVG_MILLI",
        config.min_avg_accepted_milli);
    config.max_extra_ms_per_accept_milli = spec_env_u32(
        "DFLASH_DS4_SPEC_SCHEDULER_MAX_MS_PER_ACCEPT_MILLI",
        config.max_extra_ms_per_accept_milli);
    config.max_extra_saved_ratio_milli = spec_env_u32(
        "DFLASH_DS4_SPEC_SCHEDULER_MAX_EXTRA_SAVED_RATIO_MILLI",
        config.max_extra_saved_ratio_milli);
    config.tail_min_tokens = spec_env_u32(
        "DFLASH_DS4_SPEC_SCHEDULER_TAIL_MIN_TOKENS",
        config.tail_min_tokens);
    config.gate_disable_accept_milli = spec_env_u32(
        "DFLASH_DS4_SPEC_GATE_DISABLE_MILLI",
        config.gate_disable_accept_milli);
    config.gate_enable_accept_milli = spec_env_u32(
        "DFLASH_DS4_SPEC_GATE_ENABLE_MILLI",
        config.gate_enable_accept_milli);
    config.gate_min_samples = spec_env_u32(
        "DFLASH_DS4_SPEC_GATE_MIN_SAMPLES", config.gate_min_samples);
    config.gate_reprobe_requests = spec_env_u32(
        "DFLASH_DS4_SPEC_GATE_REPROBE_REQUESTS",
        config.gate_reprobe_requests);
    config.gate_alpha_milli = spec_env_u32(
        "DFLASH_DS4_SPEC_GATE_ALPHA_MILLI", config.gate_alpha_milli);
    config.gate_probe_alpha_milli = spec_env_u32(
        "DFLASH_DS4_SPEC_GATE_PROBE_ALPHA_MILLI",
        config.gate_probe_alpha_milli);
    config.gate_recover_probes = spec_env_u32(
        "DFLASH_DS4_SPEC_GATE_RECOVER_PROBES", config.gate_recover_probes);
    return config;
}

// Per-position survival threshold for the confident-prefix rule.
//
// DEFAULT OFF. Set DFLASH_DS4_SPEC_CONFIDENCE_PREFIX_MILLI to a value in
// (0, 1000] to enable it (e.g. 900 = 0.90). At 0 the legacy cumulative-bucket
// policy below is used, which is exactly today's production behaviour.
//
// Why it ships disabled: measured on the real model 2026-07-29 at ds4's 0.90,
// matched A/B, same prompts as the pre-change sweep —
//     ptok    spec  accept      AR   delta   (old policy)
//   10914    8.34   1.000   17.40    -52%        (-8%)
//   15589   11.97   1.000   18.00    -34%        (+5%)
//   20264   11.40   1.000   16.54    -31%        (-6%)
//   24939   12.53   1.000   14.05    -11%       (-27%)
//   31189   11.55   1.000   12.79    -10%       (-45%)
// It fixed the far end and wrecked the near end. Cause: every observed
// confidence0 fell in 0.69-0.86, never reaching 0.90, so the confident prefix
// collapsed to zero and speculation stopped almost entirely (offered dropped to
// 0.06-0.12 candidates/step at accept 1.000). ds4's 0.90 is calibrated against
// its own DSpark checkpoint; ours is a REQUANTIZED artifact
// (DSpark-draft-Q4RMFP4-denseF16), and requantization plausibly shifted the
// confidence head's calibration. The threshold needs measuring against this
// drafter, not inheriting.
//
// Unexplained and worth profiling before re-enabling: with speculation almost
// fully suppressed both arms should converge, so -52% at 11k is larger than the
// discarded drafter work should cost. Get the [ds4-spec-t] stage breakdown first.
// Resolved once per request (not per process), so the threshold can be swept
// without restarting: /tmp/ds4_spec_conf_prefix overrides the env var, following
// the same live-tunable idiom as /tmp/ds4_spec_q and /tmp/ds4_awidth. Calibrating
// this against a given drafter is the whole point, and a restart-per-value sweep
// would reload ~98GiB of weights each time.
static float spec_confidence_prefix_threshold() {
    long milli = -1;
    if (std::FILE * f = std::fopen("/tmp/ds4_spec_conf_prefix", "r")) {
        long v = 0;
        if (std::fscanf(f, "%ld", &v) == 1) milli = v;
        std::fclose(f);
    }
    if (milli < 0) {
        static const long env_milli = [] {
            if (const char * e = std::getenv("DFLASH_DS4_SPEC_CONFIDENCE_PREFIX_MILLI")) {
                char * end = nullptr;
                errno = 0;
                const long n = std::strtol(e, &end, 10);
                if (errno == 0 && end != e && *end == '\0' && n >= 0 && n <= 1000) {
                    return n;
                }
                std::fprintf(stderr,
                             "[ds4-spec] invalid DFLASH_DS4_SPEC_CONFIDENCE_PREFIX_MILLI"
                             "='%s'; confident-prefix rule stays disabled\n", e);
            }
            return 0L;
        }();
        milli = env_milli;
    }
    if (milli <= 0 || milli > 1000) return 0.0f;   // 0 = legacy bucket policy
    return (float) milli / 1000.0f;
}

//
// This is ds4's policy, ported: dspark_confident_prefix_len (ds4.c:32205) walks
// the block and truncates the draft at the FIRST candidate whose predicted
// survival falls below the threshold, with e->dspark_confidence_threshold
// defaulting to 0.9f (ds4.c:55197). It is the paper's "estimated prefix survival
// probabilities" applied literally, one logit per block position.
//
// What this replaces, and why: the previous policy multiplied the first two or
// three confidences into a CUMULATIVE product and compared it against 0.40/0.30
// to pick a width bucket of 3 or 4, with a floor of 2. Two problems:
//   - a cumulative product against 0.30 is a far weaker test than each position
//     clearing 0.90, so the verify widened on low-confidence drafts;
//   - the floor of 2 meant one candidate was ALWAYS speculated whenever the
//     ratio-4 boundary allowed it, so "do not speculate this step" was
//     unreachable no matter how unconfident the drafter was.
// Measured consequence (2026-07-29 matched A/B, same prompt, /tmp/ds4_spec_q as
// the only variable): spec kept running at accept 0.60 for -27% and accept 0.71
// for -45% versus AR. A per-position 0.9 gate collapses the prefix toward zero
// in exactly that regime, which is also why ds4 needs no context ceiling at all.
//
// Artifacts without a compatible confidence head still fall back to the EWMA
// width policy below.
// Legacy cumulative-confidence bucket thresholds — the DEFAULT path, unchanged
// from what production runs today. Retained because the confident-prefix rule
// above measured worse on this drafter (see the table) and is opt-in until its
// threshold is calibrated here.
constexpr float kConfidenceQ3Threshold = 0.40f;
constexpr float kConfidenceQ4Threshold = 0.30f;

// Context-relaxation of the widening thresholds.
//
// The thresholds above were calibrated on short-context q=4 traces, where a
// wider verify costs meaningfully more. That economics inverts as the context
// grows: the indexer scoring pass and the KV read are done ONCE per verify
// regardless of q, and those are precisely the costs that scale with context
// (ratio-4 layers score ~pos/4 compressed rows to pick their top-512). So the
// acceptance probability needed to justify one more candidate FALLS as the
// context grows, and applying the short-context thresholds unchanged at 60k+
// declines to speculate in exactly the regime where each step is most
// expensive.
//
// Scale linearly from "no relaxation" at kSpecCtxRelaxStart to a floor factor
// at kSpecCtxRelaxFull. Set DFLASH_DS4_SPEC_CTX_RELAX=0 to restore the flat
// short-context policy for A/B.
constexpr int   kSpecCtxRelaxStart = 8192;
constexpr int   kSpecCtxRelaxFull  = 65536;
constexpr float kSpecCtxRelaxFloor = 0.25f;

static inline float spec_ctx_confidence_scale(int kv_pos, bool enabled) {
    if (!enabled || kv_pos <= kSpecCtxRelaxStart) return 1.0f;
    if (kv_pos >= kSpecCtxRelaxFull) return kSpecCtxRelaxFloor;
    const float t = (float) (kv_pos - kSpecCtxRelaxStart) /
                    (float) (kSpecCtxRelaxFull - kSpecCtxRelaxStart);
    return 1.0f - t * (1.0f - kSpecCtxRelaxFloor);
}

// ── Light rollback state ────────────────────────────────────────────────
// The ratio-4 compressor owns an eight-row [prev,current] window.  A q-wide
// verify can overwrite both halves, including wrapping rejected future rows
// into current before the ordinary q=1 fallback recompresses a boundary.
// Preserve all eight rows.  Ratio-128 states are pure position rings and the
// at-risk rows are overwritten position-for-position by replay, so skip them.
void save_ratio4_state(ggml_tensor * t, std::vector<uint8_t> & buf) {
    if (!t || t->ne[1] != 8) { buf.clear(); return; }
    const size_t bytes = ggml_nbytes(t);
    if (buf.size() != bytes) buf.resize(bytes);
    ggml_backend_tensor_get(t, buf.data(), 0, bytes);
}

void restore_ratio4_state(ggml_tensor * t,
                          const std::vector<uint8_t> & buf) {
    if (!t || buf.empty()) return;
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}

void spec_rollback_save_impl(const DeepSeek4Cache & cache,
                             DeepSeek4SpecRollback & rb,
                             int raw_pos, int raw_count) {
    rb.raw_pos = raw_pos;
    rb.raw_count = std::max(0, raw_count);
    rb.layers.resize(cache.layers.size());
    for (size_t il = 0; il < cache.layers.size(); ++il) {
        const DeepSeek4LayerCache & lc = cache.layers[il];
        DeepSeek4SpecRollback::Layer & s = rb.layers[il];
        save_ratio4_state(lc.attn_compressor.state_kv,       s.attn_kv);
        save_ratio4_state(lc.attn_compressor.state_score,    s.attn_sc);
        save_ratio4_state(lc.indexer_compressor.state_kv,    s.idx_kv);
        save_ratio4_state(lc.indexer_compressor.state_score, s.idx_sc);
        s.raw_row_bytes = 0;
        s.raw_rows.clear();
        if (lc.raw_kv && lc.raw_kv->ne[1] > 0 && rb.raw_count > 0) {
            s.raw_row_bytes = ggml_row_size(lc.raw_kv->type, lc.raw_kv->ne[0]);
            s.raw_rows.resize(s.raw_row_bytes * (size_t) rb.raw_count);
            for (int t = 0; t < rb.raw_count; ++t) {
                int row = (rb.raw_pos + t) % (int) lc.raw_kv->ne[1];
                if (row < 0) row += (int) lc.raw_kv->ne[1];
                ggml_backend_tensor_get(
                    lc.raw_kv,
                    s.raw_rows.data() + (size_t) t * s.raw_row_bytes,
                    (size_t) row * lc.raw_kv->nb[1], s.raw_row_bytes);
            }
        }
    }
    if (cache.hc_state) {
        const size_t bytes = ggml_nbytes(cache.hc_state);
        if (rb.hc_state.size() != bytes) rb.hc_state.resize(bytes);
        ggml_backend_tensor_get(cache.hc_state, rb.hc_state.data(), 0, bytes);
    }
}

// Truncate the cache to commit_pos. restore_prev is set when the verify
// crossed a ratio-4 boundary at-or-past commit_pos: q-wide evaluation may have
// modified either half of the rolling window with rejected tokens, so put the
// complete pre-verify state back. (A boundary strictly inside the committed
// range is a legitimate flush and must be kept.)
void spec_rollback_apply_impl(const DeepSeek4SpecRollback & rb,
                              const DeepSeek4Weights & w,
                              DeepSeek4Cache & cache,
                              int commit_pos,
                              bool restore_prev) {
    cache.cur_pos = commit_pos;
    for (size_t il = 0; il < cache.layers.size(); ++il) {
        DeepSeek4LayerCache & lc = cache.layers[il];
        const uint32_t ratio = il < w.compress_ratios.size() ? w.compress_ratios[il] : 0;
        if (ratio > 0) lc.n_comp = commit_pos / (int) ratio;
        if (ratio == 4) lc.n_index_comp = commit_pos / 4;
        if (restore_prev && il < rb.layers.size()) {
            const DeepSeek4SpecRollback::Layer & s = rb.layers[il];
            restore_ratio4_state(lc.attn_compressor.state_kv,       s.attn_kv);
            restore_ratio4_state(lc.attn_compressor.state_score,    s.attn_sc);
            restore_ratio4_state(lc.indexer_compressor.state_kv,    s.idx_kv);
            restore_ratio4_state(lc.indexer_compressor.state_score, s.idx_sc);
        }
        if (il < rb.layers.size() && lc.raw_kv && lc.raw_kv->ne[1] > 0) {
            const DeepSeek4SpecRollback::Layer & s = rb.layers[il];
            const int first_rejected = std::max(0, commit_pos - rb.raw_pos);
            for (int t = first_rejected;
                 t < rb.raw_count &&
                 s.raw_row_bytes > 0 &&
                 s.raw_rows.size() >= (size_t) (t + 1) * s.raw_row_bytes;
                 ++t) {
                int row = (rb.raw_pos + t) % (int) lc.raw_kv->ne[1];
                if (row < 0) row += (int) lc.raw_kv->ne[1];
                ggml_backend_tensor_set(
                    lc.raw_kv,
                    s.raw_rows.data() + (size_t) t * s.raw_row_bytes,
                    (size_t) row * lc.raw_kv->nb[1], s.raw_row_bytes);
            }
        }
    }
    if (restore_prev && cache.hc_state && !rb.hc_state.empty()) {
        ggml_backend_tensor_set(cache.hc_state, rb.hc_state.data(), 0, rb.hc_state.size());
    }
}

using SpecClock = std::chrono::steady_clock;

double spec_ms_since(SpecClock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(SpecClock::now() - t0).count() / 1000.0;
}

}  // namespace

DSparkSchedulerConfig dspark_scheduler_config_from_env() {
    return spec_scheduler_config();
}

// ── worker-scoped scheduler ─────────────────────────────────────────────────
// Constructed on first use per thread. Spec decode only ever runs on the single
// generation worker, so this needs no locking; the pointer form lets the AR path
// feed a baseline even before any spec decode has constructed the instance.
static thread_local DSparkProfitScheduler * g_worker_scheduler = nullptr;
static thread_local double g_pending_ar_target_eval_ms = 0.0;

DSparkProfitScheduler & dspark_worker_scheduler(const DSparkSchedulerConfig & config) {
    static thread_local DSparkProfitScheduler instance(config);
    g_worker_scheduler = &instance;
    if (g_pending_ar_target_eval_ms > 0.0) {
        instance.observe_target_eval(g_pending_ar_target_eval_ms);
        g_pending_ar_target_eval_ms = 0.0;
    }
    return instance;
}

DSparkProfitScheduler & dspark_worker_scheduler() {
    static thread_local const DSparkSchedulerConfig cfg =
        dspark_scheduler_config_from_env();
    return dspark_worker_scheduler(cfg);
}

void dspark_worker_note_target_eval(double elapsed_ms) {
    if (!(elapsed_ms > 0.0) || !std::isfinite(elapsed_ms)) return;
    if (g_worker_scheduler != nullptr) {
        g_worker_scheduler->observe_target_eval(elapsed_ms);
    } else if (g_pending_ar_target_eval_ms <= 0.0) {
        g_pending_ar_target_eval_ms = elapsed_ms;
    } else {
        g_pending_ar_target_eval_ms =
            0.75 * g_pending_ar_target_eval_ms + 0.25 * elapsed_ms;
    }
}

void deepseek4_spec_rollback_save(const DeepSeek4Cache & cache,
                                  DeepSeek4SpecRollback & rollback,
                                  int raw_pos,
                                  int raw_count) {
    spec_rollback_save_impl(cache, rollback, raw_pos, raw_count);
}

void deepseek4_spec_rollback_apply(const DeepSeek4SpecRollback & rollback,
                                   const DeepSeek4Weights & weights,
                                   DeepSeek4Cache & cache,
                                   int commit_pos,
                                   bool restore_prev) {
    spec_rollback_apply_impl(rollback, weights, cache, commit_pos, restore_prev);
}

struct DeepSeek4DSparkResidentProposal::Impl {
    int committed = 0;
    int32_t seed = -1;
    int q = 0;
    std::vector<float> noise_embed;
    std::unique_ptr<XdnaDSparkDraftJob> job;
    SpecClock::time_point submitted_at{};
};

DeepSeek4DSparkResidentProposal::DeepSeek4DSparkResidentProposal() = default;
DeepSeek4DSparkResidentProposal::~DeepSeek4DSparkResidentProposal() = default;
DeepSeek4DSparkResidentProposal::DeepSeek4DSparkResidentProposal(
    DeepSeek4DSparkResidentProposal &&) noexcept = default;
DeepSeek4DSparkResidentProposal &
DeepSeek4DSparkResidentProposal::operator=(
    DeepSeek4DSparkResidentProposal &&) noexcept = default;

bool DeepSeek4DSparkResidentProposal::pending() const {
    return impl_ && impl_->job;
}

void DeepSeek4DSparkResidentProposal::cancel() noexcept {
    if (impl_ && impl_->job) impl_->job->cancel();
    impl_.reset();
}

bool deepseek4_dspark_resident_prepare(
        ggml_backend_t backend,
        const DeepSeek4Weights & target_w,
        const DSparkDrafter & drafter,
        const std::vector<float> & feature_window,
        int committed,
        int32_t seed,
        int max_commit_tokens,
        XdnaDSparkDraftCompute & xdna_draft_compute,
        DeepSeek4DSparkResidentProposal & proposal,
        std::string * error) {
    if (error) error->clear();
    if (proposal.pending()) {
        if (error) *error = "resident DSpark proposal already pending";
        return false;
    }
    const int n_embd = target_w.n_embd;
    const int feat_row = drafter.n_target_layers * n_embd;
    if (committed < 0 || seed < 0 || n_embd <= 0 || feat_row <= 0 ||
        drafter.block_size <= 0 || max_commit_tokens < 2 ||
        feature_window.size() % (size_t)feat_row != 0 ||
        feature_window.size() / (size_t)feat_row > (size_t)target_w.n_swa ||
        !xdna_draft_compute.healthy()) {
        if (error) *error = "resident DSpark proposal inputs are invalid";
        return false;
    }

    auto impl = std::make_unique<DeepSeek4DSparkResidentProposal::Impl>();
    impl->committed = committed;
    impl->seed = seed;
    // Exact-prefix verification is deliberately capped at the validated
    // ratio-4 width. Unlike the batched verifier it never feeds a rejected
    // token, so no rollback state is required.
    impl->q = std::min({4, drafter.block_size, max_commit_tokens});
    impl->noise_embed.resize((size_t)n_embd * drafter.block_size);
    std::vector<int32_t> noise_ids((size_t)drafter.block_size,
                                   drafter.mask_token_id);
    noise_ids[0] = seed;
    if (!target_w.embedder.embed(noise_ids.data(), drafter.block_size,
                                 impl->noise_embed.data())) {
        if (error) *error = "resident DSpark seed embedding failed";
        return false;
    }

    const int ctx_len = (int)(feature_window.size() / (size_t)feat_row);
    const bool xdna_gpu_main =
        spec_env_flag("DFLASH_DSPARK_XDNA_GPU_MAIN");
    std::vector<float> main_context;
    std::vector<float> context_kv;
    if (xdna_gpu_main && ctx_len > 0 &&
        !deepseek4_dspark_project_main_context(
            backend, drafter, feature_window.data(), ctx_len,
            main_context, &context_kv)) {
        if (error) *error = "resident GPU main-context projection failed";
        return false;
    }
    XdnaDSparkDraftRequest request;
    request.committed = committed;
    request.ctx_len = ctx_len;
    request.noise_embed = impl->noise_embed.data();
    request.ctx_features = xdna_gpu_main || ctx_len == 0
        ? nullptr : feature_window.data();
    request.main_context = xdna_gpu_main && ctx_len > 0
        ? main_context.data() : nullptr;
    request.context_kv = xdna_gpu_main && ctx_len > 0
        ? context_kv.data() : nullptr;
    impl->submitted_at = SpecClock::now();
    impl->job = xdna_draft_compute.submit(request, error);
    if (!impl->job) return false;
    proposal.impl_ = std::move(impl);
    return true;
}

bool deepseek4_dspark_resident_finish(
        ggml_backend_t backend,
        int device,
        const DeepSeek4Weights & target_w,
        DeepSeek4Cache & target_cache,
        const DSparkDrafter & drafter,
        std::vector<float> & feature_window,
        DeepSeek4DSparkResidentProposal & proposal,
        std::vector<int32_t> & committed_tokens,
        int32_t & next_token,
        std::vector<float> & last_logits,
        int & offered_candidates,
        int & accepted_candidates,
        DeepSeek4DSparkResidentTiming & timing,
        std::string * error) {
    if (error) error->clear();
    committed_tokens.clear();
    last_logits.clear();
    next_token = -1;
    offered_candidates = 0;
    accepted_candidates = 0;
    timing = {};
    if (!proposal.impl_ || !proposal.impl_->job) {
        if (error) *error = "resident DSpark proposal is not pending";
        return false;
    }

    std::unique_ptr<DeepSeek4DSparkResidentProposal::Impl> impl =
        std::move(proposal.impl_);
    XdnaDSparkDraftOutput provider_output;
    const SpecClock::time_point provider_wait_t0 = SpecClock::now();
    if (!impl->job->wait(provider_output, error)) return false;
    const double provider_block_ms = spec_ms_since(provider_wait_t0);
    const double provider_age_ms = spec_ms_since(impl->submitted_at);
    const int n_embd = target_w.n_embd;
    const size_t hidden_need =
        (size_t)n_embd * (size_t)drafter.block_size;
    if (provider_output.hidden.size() != hidden_need) {
        if (error) *error = "resident DSpark provider returned wrong hidden shape";
        return false;
    }
    std::vector<float> gpu_reference_hidden;
    if (spec_env_flag("DFLASH_DSPARK_XDNA_COMPARE")) {
        const int feat_row = drafter.n_target_layers * n_embd;
        const int ctx_len = feat_row > 0
            ? (int)(feature_window.size() / (size_t)feat_row) : 0;
        std::vector<float> gpu_confidence_hidden;
        const bool gpu_ok = deepseek4_dspark_draft_forward(
            backend, drafter, impl->noise_embed.data(),
            ctx_len > 0 ? feature_window.data() : nullptr,
            ctx_len, impl->committed, gpu_reference_hidden,
            &gpu_confidence_hidden);
        const bool compare_ok = gpu_ok &&
            dspark_log_xdna_compare(
                "resident_normalized_hidden", provider_output.hidden,
                gpu_reference_hidden) &&
            dspark_log_xdna_compare(
                "resident_confidence_hidden",
                provider_output.confidence_hidden,
                gpu_confidence_hidden);
        if (!compare_ok &&
            spec_env_flag("DFLASH_DSPARK_XDNA_COMPARE_REQUIRED")) {
            if (error) *error = gpu_ok
                ? "resident XDNA hidden differential failed"
                : "resident GPU draft reference failed";
            return false;
        }
    }

    DeepSeek4DFlashTarget target(
        target_w, target_cache, backend, device, nullptr,
        drafter.capture_layer_ids, drafter.mask_token_id,
        /*strict_verify=*/false);
    target.set_keep_logits(true);
    DeepSeek4StepTelemetry verify_telemetry;
    const bool detailed_timing = spec_env_flag("DFLASH_DS4_TIMING");
    if (detailed_timing) target.set_telemetry(&verify_telemetry);
    DraftWeights draft_weights = make_dspark_shim(drafter);
    std::vector<float> padded_hidden(
        (size_t)n_embd * ((size_t)drafter.block_size + 1), 0.0f);
    std::vector<float> padded_confidence_hidden;
    std::vector<float> draft_confidence;
    std::memcpy(padded_hidden.data() + n_embd,
                provider_output.hidden.data(),
                hidden_need * sizeof(float));
    const bool confidence_available =
        provider_output.confidence_hidden.size() == hidden_need &&
        drafter.confidence_w && drafter.confidence_b &&
        (drafter.confidence_dim == n_embd ||
         drafter.confidence_dim == n_embd + drafter.markov_rank);
    if (confidence_available) {
        padded_confidence_hidden.assign(
            (size_t)n_embd * ((size_t)drafter.block_size + 1), 0.0f);
        std::memcpy(padded_confidence_hidden.data() + n_embd,
                    provider_output.confidence_hidden.data(),
                    hidden_need * sizeof(float));
    }
    std::vector<int32_t> draft_tokens;
    const SpecClock::time_point head_t0 = SpecClock::now();
    bool head_ok = dspark_markov_correct_greedy_chain_fused(
        draft_weights, backend, target.lm_head_tensor(), padded_hidden.data(),
        impl->q, impl->seed, draft_tokens,
        confidence_available ? &draft_confidence : nullptr,
        confidence_available ? padded_confidence_hidden.data() : nullptr);
    if (!head_ok) {
        head_ok = dspark_markov_correct_greedy_chain(
            draft_weights, backend, target, padded_hidden.data(), impl->q,
            impl->seed, 0.0f, draft_tokens);
    }
    if (!head_ok || (int)draft_tokens.size() < impl->q) {
        std::vector<int32_t> projected;
        if (!target.project_hidden_to_tokens(provider_output.hidden.data(),
                                             impl->q - 1, projected)) {
            if (error) *error = "resident DSpark tied head failed";
            return false;
        }
        draft_tokens.clear();
        draft_tokens.push_back(impl->seed);
        draft_tokens.insert(draft_tokens.end(), projected.begin(),
                            projected.end());
    }
    if ((int)draft_tokens.size() > impl->q)
        draft_tokens.resize((size_t)impl->q);
    if (!gpu_reference_hidden.empty()) {
        std::vector<float> gpu_padded_hidden(
            (size_t)n_embd * ((size_t)drafter.block_size + 1), 0.0f);
        std::memcpy(gpu_padded_hidden.data() + n_embd,
                    gpu_reference_hidden.data(), hidden_need * sizeof(float));
        std::vector<int32_t> gpu_draft_tokens;
        bool gpu_head_ok = dspark_markov_correct_greedy_chain_fused(
            draft_weights, backend, target.lm_head_tensor(),
            gpu_padded_hidden.data(), impl->q, impl->seed,
            gpu_draft_tokens, nullptr, nullptr);
        if (!gpu_head_ok) {
            gpu_head_ok = dspark_markov_correct_greedy_chain(
                draft_weights, backend, target, gpu_padded_hidden.data(),
                impl->q, impl->seed, 0.0f, gpu_draft_tokens);
        }
        std::fprintf(stderr,
                     "[xdna-dspark-head-compare] pos=%d seed=%d "
                     "provider0=%d gpu0=%d gpu_ok=%d\n",
                     impl->committed, impl->seed,
                     draft_tokens.size() > 1 ? draft_tokens[1] : -1,
                     gpu_draft_tokens.size() > 1 ? gpu_draft_tokens[1] : -1,
                     gpu_head_ok ? 1 : 0);
    }
    const double head_ms = spec_ms_since(head_t0);

    // The resident verifier has a different reduction topology from ordinary
    // AR and is therefore reserved for drafts that the support model itself
    // considers reliable. A declined first block leaves target KV untouched;
    // the resident coordinator falls through to its ordinary AR path and
    // disables speculation for the rest of this request. Zero preserves the
    // diagnostic always-admit behavior used by earlier prototypes.
    const uint32_t min_confidence_milli = spec_env_u32(
        "DFLASH_DS4_RESIDENT_MIN_CONFIDENCE_MILLI", 0);
    const bool confidence_declined = min_confidence_milli > 0 &&
        (draft_confidence.empty() ||
         draft_confidence[0] < (float)min_confidence_milli / 1000.0f);
    if (detailed_timing) {
        std::fprintf(stderr,
                     "[ds4-resident-confidence] pos=%d confidence0=%s%.3f "
                     "threshold=%.3f admit=%d\n",
                     impl->committed,
                     draft_confidence.empty() ? "n/a " : "",
                     draft_confidence.empty() ? 0.0f : draft_confidence[0],
                     (float)min_confidence_milli / 1000.0f,
                     confidence_declined ? 0 : 1);
    }
    if (confidence_declined) {
        timing.provider_age_s = provider_age_ms / 1000.0;
        timing.provider_block_s = provider_block_ms / 1000.0;
        timing.head_s = head_ms / 1000.0;
        return true;
    }

    // The resident commit boundary is the ordinary fused q=1 target graph.
    // A q-wide prepass was previously used as an admission filter, then rolled
    // back before exact replay. Cold-corpus testing showed that the prepass can
    // perturb later q=1 output even after every documented cache tensor is
    // restored. It also adds target work. Exact prefix verification already
    // stops at the first rejected proposal and leaves KV at the accepted
    // frontier, so commit that prefix directly.
    //
    // The approximate q-wide path remains an explicit diagnostic escape hatch;
    // it is never enabled by the packaged overlay.
    const bool approximate_commit =
        spec_env_flag("DFLASH_DS4_RESIDENT_APPROX_COMMIT");
    std::vector<int32_t> target_argmax;
    int verify_last = -1;
    const SpecClock::time_point verify_t0 = SpecClock::now();
    DeepSeek4SpecRollback rollback;
    if (approximate_commit) {
        deepseek4_spec_rollback_save(
            target_cache, rollback, impl->committed,
            (int)draft_tokens.size());
        if (!target.verify_batch(draft_tokens, impl->committed, verify_last,
                                 &target_argmax)) {
            deepseek4_spec_rollback_apply(
                rollback, target_w, target_cache, impl->committed, true);
            if (error) *error =
                "resident DSpark batched target verification failed";
            return false;
        }
    } else if (!target.verify_exact_prefix(
                   draft_tokens, impl->committed, verify_last,
                   &target_argmax)) {
        if (error) *error =
            "resident DSpark exact prefix verification failed";
        return false;
    }

    int accept = 1;
    for (int index = 0; index + 1 < (int)draft_tokens.size(); ++index) {
        if (index >= (int)target_argmax.size() ||
            draft_tokens[(size_t)index + 1] != target_argmax[(size_t)index])
            break;
        ++accept;
        if (deepseek4_is_eos_tok(draft_tokens[(size_t)index + 1], target_w))
            break;
    }
    if ((int)target_argmax.size() < accept) {
        if (approximate_commit) {
            deepseek4_spec_rollback_apply(
                rollback, target_w, target_cache, impl->committed, true);
        }
        if (error) *error = "resident DSpark verifier frontier mismatch";
        return false;
    }

    // A partial approximate block cannot be committed because q-wide logits
    // are non-authoritative. The exact verifier, in contrast, has evaluated
    // precisely `accept` q=1 rows and can commit that prefix, including the
    // already-pending seed token.
    if (approximate_commit && accept < (int)draft_tokens.size()) {
        deepseek4_spec_rollback_apply(
            rollback, target_w, target_cache, impl->committed, true);
        const double verify_ms = spec_ms_since(verify_t0);
        offered_candidates = (int)draft_tokens.size() - 1;
        accepted_candidates = 0;
        if (detailed_timing) {
            std::fprintf(stderr,
                         "[ds4-resident-spec] pos=%d q=%d accept=0 "
                         "partial=%d draft0=%d target0=%d provider_age=%.1fms "
                         "provider_block=%.1fms head=%.1fms verify=%.1fms "
                         "fallback=target-ar\n",
                         impl->committed, impl->q, accept,
                         draft_tokens.size() > 1 ? draft_tokens[1] : -1,
                         target_argmax.empty() ? -1 : target_argmax[0],
                         provider_age_ms, provider_block_ms, head_ms,
                         verify_ms);
        }
        timing.provider_age_s = provider_age_ms / 1000.0;
        timing.provider_block_s = provider_block_ms / 1000.0;
        timing.head_s = head_ms / 1000.0;
        timing.verify_s = verify_ms / 1000.0;
        return true;
    }
    const double verify_ms = spec_ms_since(verify_t0);

    if (target.last_verify_n() != accept) {
        if (error) *error = "resident DSpark verifier commit mismatch";
        return false;
    }

    std::vector<float> verify_logits;
    if (!target.read_verify_logits(accept, verify_logits) ||
        verify_logits.size() != (size_t)accept * target_w.n_vocab) {
        if (error) *error = "resident DSpark verifier logits missing";
        return false;
    }
    last_logits.assign(
        verify_logits.end() - target_w.n_vocab, verify_logits.end());
    next_token = target_argmax[(size_t)accept - 1];
    committed_tokens.assign(draft_tokens.begin(),
                            draft_tokens.begin() + accept);
    offered_candidates = (int)draft_tokens.size() - 1;
    accepted_candidates = accept - 1;

    const int feat_row = drafter.n_target_layers * n_embd;
    const std::vector<float> & features = target.last_features();
    if (feat_row <= 0 ||
        features.size() != (size_t)accept * (size_t)feat_row) {
        if (error) *error = "resident DSpark verifier features missing";
        return false;
    }
    for (int index = 0; index < accept; ++index) {
        const size_t rows = feature_window.size() / (size_t)feat_row;
        if (rows >= (size_t)target_w.n_swa) {
            std::memmove(feature_window.data(),
                         feature_window.data() + feat_row,
                         (feature_window.size() - (size_t)feat_row) *
                             sizeof(float));
            feature_window.resize(feature_window.size() - (size_t)feat_row);
        }
        const float * row = features.data() + (size_t)index * feat_row;
        feature_window.insert(feature_window.end(), row, row + feat_row);
    }

    if (spec_env_flag("DFLASH_DS4_TIMING")) {
        std::fprintf(stderr,
                     "[ds4-resident-spec] pos=%d q=%d accept=%d "
                     "provider_age=%.1fms provider_block=%.1fms "
                     "head=%.1fms verify=%.1fms mode=%s "
                     "graph(build/set/compute/read)=%.1f/%.1f/%.1f/%.1fms "
                     "fused(calls/rows)=%" PRIu64 "/%" PRIu64 "\n",
                     impl->committed, impl->q, accept,
                     provider_age_ms, provider_block_ms, head_ms, verify_ms,
                     approximate_commit ? "qwide-approx" : "q1-exact",
                     verify_telemetry.full_graph_build_us / 1000.0,
                     verify_telemetry.full_graph_set_us / 1000.0,
                     verify_telemetry.full_graph_compute_us / 1000.0,
                     verify_telemetry.full_graph_read_us / 1000.0,
                     verify_telemetry.fused_verify_calls,
                     verify_telemetry.fused_verify_rows);
    }
    timing.provider_age_s = provider_age_ms / 1000.0;
    timing.provider_block_s = provider_block_ms / 1000.0;
    timing.head_s = head_ms / 1000.0;
    timing.verify_s = verify_ms / 1000.0;
    return true;
}

// Batched target verify + capture: wraps the existing multi-token
// deepseek4_step_layer_range (dynamic attention + batched HC), which never
// touches the fused single-token 23 tok/s path, with the Ds4VerifyHooks that
// add per-layer mean-over-HC capture and full per-position logits.
bool deepseek4_dspark_verify_forward(ggml_backend_t backend,
                                     int device,
                                     const DeepSeek4Weights & w,
                                     DeepSeek4Cache & cache,
                                     const std::vector<int> & capture_layer_ids,
                                     const float * embed,
                                     const int32_t * token_ids,
                                     int n_tokens,
                                     int kv_start,
                                     std::vector<int32_t> & argmax_out,
                                     std::vector<float> * logits_out,
                                     std::vector<float> & capture_out,
                                     DeepSeek4StepTelemetry * telemetry,
                                     bool allow_graph_reuse,
                                     bool require_target_graph) {
    std::vector<float> hc_state;
    std::vector<float> all_logits;
    std::vector<float> last_logits;
    Ds4VerifyHooks hooks;
    hooks.capture_layer_ids = &capture_layer_ids;
    hooks.capture_out = &capture_out;
    hooks.all_logits_out = &all_logits;
    hooks.require_fused_q1 = require_target_graph;
    if (!deepseek4_step_layer_range(backend, device, w, cache, hc_state, embed, n_tokens, kv_start,
                                    0, w.n_layer, &last_logits, token_ids,
                                    telemetry, allow_graph_reuse,
                                    &hooks)) {
        std::fprintf(stderr, "[ds4-verify] step_layer_range returned false (n_tokens=%d kv_start=%d)\n",
                     n_tokens, kv_start);
        return false;
    }
    if ((int) all_logits.size() < w.n_vocab * n_tokens) {
        std::fprintf(stderr, "[ds4-verify] all_logits too small: got=%zu need=%d (cap=%zu)\n",
                     all_logits.size(), w.n_vocab * n_tokens, capture_out.size());
        return false;
    }
    const size_t capture_need = capture_layer_ids.size() *
        (size_t) w.n_embd * (size_t) n_tokens;
    if (capture_out.size() != capture_need) {
        std::fprintf(stderr,
                     "[ds4-verify] capture size mismatch: got=%zu need=%zu\n",
                     capture_out.size(), capture_need);
        return false;
    }
    argmax_out.resize(n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        const float * row = all_logits.data() + (size_t) t * w.n_vocab;
        int best = 0; float bv = row[0];
        for (int i = 1; i < w.n_vocab; i++) if (row[i] > bv) { bv = row[i]; best = i; }
        argmax_out[t] = best;
    }
    if (logits_out) *logits_out = std::move(all_logits);
    return true;
}

bool run_deepseek4_dspark_spec_decode(
        ggml_backend_t backend,
        int device,
        const DeepSeek4Weights & target_w,
        DeepSeek4Cache & target_cache,
        const DSparkDrafter & drafter,
        int committed,
        int last_tok,
        int n_gen,
        const float * prompt_feature_window,
        int win_len,
        std::vector<int32_t> & out_tokens,
        float * accept_rate_out,
        XdnaDSparkDraftCompute * xdna_draft_compute,
        const std::function<bool(int32_t)> & on_token) {
    const int n_embd = target_w.n_embd;
    const int n_tgt = drafter.n_target_layers;
    const int block = drafter.block_size;
    const int n_swa = target_w.n_swa;
    const int feat_row = n_tgt * n_embd;

    const bool debug = spec_env_flag("DFLASH_DS4_DSPARK_DEBUG");
    const bool timing = spec_env_flag("DFLASH_DS4_TIMING");
    const bool xdna_gpu_main =
        spec_env_flag("DFLASH_DSPARK_XDNA_GPU_MAIN");
    const bool xdna_compare =
        spec_env_flag("DFLASH_DSPARK_XDNA_COMPARE");
    const bool xdna_compare_required =
        spec_env_flag("DFLASH_DSPARK_XDNA_COMPARE_REQUIRED");
    const uint32_t xdna_compare_steps = spec_env_u32(
        "DFLASH_DSPARK_XDNA_COMPARE_STEPS", 1);
    // Exact target-authoritative verification is the production default.
    // The q-wide graph remains opt-in because its different reduction shape
    // can change near-tied target logits. DFLASH_DS4_APPROX_VERIFY is kept as
    // a compatibility alias for older deployments.
    const bool batch_verify_requested =
        spec_env_flag("DFLASH_DS4_BATCH_VERIFY") ||
        spec_env_flag("DFLASH_DS4_APPROX_VERIFY");
    const bool force_strict_verify =
        !batch_verify_requested || spec_env_flag("DFLASH_DS4_SEQ_VERIFY");
    // Qualify the q-wide path on the request's own exact-prefix behavior. This
    // keeps short/fragile answers on the q=1 graph while allowing repetitive,
    // high-acceptance generation to amortize target weights. Zero restores the
    // legacy immediate-batch behavior for diagnostics.
    const uint32_t batch_warmup_tokens = spec_env_u32(
        "DFLASH_DS4_BATCH_WARMUP_TOKENS", 48);
    DSparkBatchVerifyGate batch_gate(!force_strict_verify, batch_warmup_tokens);
    const bool full_snap = !force_strict_verify &&
        spec_env_flag("DFLASH_DS4_FULL_SNAP");
    const bool scheduler_log =
        timing || spec_env_flag("DFLASH_DS4_SPEC_SCHEDULER_LOG");
    // Worker-scoped, not per-call: ds4 keeps this on the session so the window,
    // the target-eval baseline and the lifetime accept counters survive across
    // decode calls. A fresh scheduler per request would restart the 4-cycle
    // window every time and start each request with no baseline at all.
    DSparkProfitScheduler & scheduler =
        dspark_worker_scheduler(spec_scheduler_config());
    // Laguna-style adaptive verify width: EWMA of accepted candidates, width =
    // ewma + 2 (avg_commit << block means the wide tail is usually wasted).
    // /tmp/ds4_awidth: 1 = on, 0 = off (default on).
    bool adaptive_width = true;
    if (std::FILE * f = std::fopen("/tmp/ds4_awidth", "r")) {
        int v = 1;
        if (std::fscanf(f, "%d", &v) == 1) adaptive_width = (v != 0);
        std::fclose(f);
    }
    const bool confidence_width_available = adaptive_width && !force_strict_verify &&
        drafter.confidence_w != nullptr && drafter.confidence_b != nullptr &&
        (drafter.confidence_dim == n_embd ||
         drafter.confidence_dim == n_embd + drafter.markov_rank);
    // Context relaxation defaults OFF, opt in with DFLASH_DS4_SPEC_CTX_RELAX=1.
    //
    // The throughput argument for it is sound (a wider verify reuses the same
    // indexer pass and KV read), but it optimises the wrong resource on this
    // box: a wider verify also holds MORE activations, and long context is
    // exactly where GTT is scarcest. Enabling it contributed to the 2026-07-28
    // OOM at 65k. Spec is now capped by DFLASH_DS4_SPEC_MAX_CTX anyway, so this
    // would only ever apply in the narrow band between the 8k relax floor and
    // that ceiling. Turn it on only alongside a measured headroom check.
    bool ctx_relax_enabled = false;
    if (const char * cr = std::getenv("DFLASH_DS4_SPEC_CTX_RELAX")) {
        if (*cr == '1') ctx_relax_enabled = true;
    }
    if (timing && confidence_width_available) {
        std::fprintf(stderr, "[ds4-spec] adaptive width policy=confidence ctx_relax=%d\n",
                     ctx_relax_enabled ? 1 : 0);
    }
    double ewma_accept = 1.5;

    // Fast path caps the verify at the compression ratio (4): one boundary max,
    // no rolling-state row aliasing -> snapshot-free rollback stays exact.
    // Full snapshots change rollback strategy but not the compressor-window
    // limit below. The legacy sequential measurement path is validated only
    // through q=4.
    int q_cap = full_snap ? block + 1 : 4;
    if (const char * qs = std::getenv("DFLASH_DS4_SPEC_Q")) {
        char * end = nullptr;
        errno = 0;
        const long parsed = std::strtol(qs, &end, 10);
        if (errno == 0 && end != qs && *end == '\0' &&
            parsed >= 2 && parsed <= block + 1) {
            const int v = static_cast<int>(parsed);
            q_cap = full_snap ? v : std::min(v, 4);
        } else {
            std::fprintf(stderr,
                         "[ds4-spec] invalid DFLASH_DS4_SPEC_Q='%s'; "
                         "using q=%d\n",
                         qs, q_cap);
        }
    }
    if (std::FILE * qf = std::fopen("/tmp/ds4_spec_q", "r")) {
        // Per-request override for perf experiments (no server restart needed).
        // q=1 disables drafting: pure AR pushed through the batched-verify path
        // (diagnoses batched-vs-sequential target divergence).
        int v = 0;
        if (std::fscanf(qf, "%d", &v) == 1 && v >= 1 && v <= block + 1) {
            q_cap = full_snap ? v : std::min(v, 4);
        }
        std::fclose(qf);
    }
    if (force_strict_verify && q_cap > 4) {
        std::fprintf(stderr,
                     "[ds4-spec] exact prefix verify supports q<=4; "
                     "capping requested q=%d to 4\n",
                     q_cap);
        q_cap = 4;
    }

    // Only the opt-in approximate verifier needs the legacy full snapshot.
    ggml_backend_t snap_backend = full_snap ? ggml_backend_cpu_init() : nullptr;
    if (full_snap && !snap_backend) {
        std::fprintf(stderr, "[ds4-spec] no CPU snapshot backend\n");
        return false;
    }

    DeepSeek4DFlashTarget target(target_w, target_cache, backend, device, snap_backend,
                                 drafter.capture_layer_ids, drafter.mask_token_id,
                                 /*strict_verify=*/batch_gate.strict_cycle());
    DraftWeights dw = make_dspark_shim(drafter);
    DeepSeek4SpecRollback rollback;
    DeepSeek4StepTelemetry tel{};
    if (timing) target.set_telemetry(&tel);

    // Host feature window ring [feat_row, n_swa] of absolute positions
    // [committed-N .. committed-1]. Seed from the prefill window.
    std::vector<float> feat_win((size_t) feat_row * n_swa, 0.0f);
    int win_have = win_len > n_swa ? n_swa : win_len;
    if (prompt_feature_window && win_have > 0) {
        // copy the last win_have columns of the prefill window
        const int src_off = (win_len - win_have);
        std::memcpy(feat_win.data(),
                    prompt_feature_window + (size_t) src_off * feat_row,
                    sizeof(float) * (size_t) feat_row * win_have);
    }
    int feat_count = win_have;   // number of valid feature columns ending at committed-1

    auto push_feature = [&](const float * col) {
        // Shift-append one feature column (keep last n_swa).
        if (feat_count >= n_swa) {
            std::memmove(feat_win.data(), feat_win.data() + feat_row,
                         sizeof(float) * (size_t) feat_row * (n_swa - 1));
            std::memcpy(feat_win.data() + (size_t) feat_row * (n_swa - 1), col,
                        sizeof(float) * feat_row);
        } else {
            std::memcpy(feat_win.data() + (size_t) feat_row * feat_count, col,
                        sizeof(float) * feat_row);
            feat_count++;
        }
    };

    int lt = last_tok;
    int pos = committed;      // absolute position of the seed (block slot 0)
    int n_generated = 0;
    long accept_sum = 0, offered_sum = 0, steps = 0;
    long agreement_offered = 0, agreement_matched = 0;
    bool scheduler_tail_handoff = false;
    bool ok = true;
    bool stop_requested = false;

    std::vector<float> noise_embed((size_t) n_embd * block);
    std::vector<float> xdna_main_context;
    std::vector<float> xdna_context_kv;
    std::vector<int32_t> noise_ids(block);
    std::vector<float> local_hidden, confidence_hidden;
    std::vector<float> padded_hidden((size_t) n_embd * (block + 1), 0.0f);
    std::vector<float> padded_confidence_hidden((size_t) n_embd * (block + 1), 0.0f);
    std::vector<int32_t> draft_tok, tgt_am;
    std::vector<float> draft_confidence;

    // Cumulative phase timings (ms).
    double tm_draft = 0, tm_head = 0, tm_save = 0, tm_verify = 0, tm_apply = 0, tm_feat = 0;
    const SpecClock::time_point run_t0 = SpecClock::now();
    uint32_t xdna_compared = 0;

    // Resolved once per request, as documented on spec_confidence_prefix_threshold()
    // — it opens /tmp/ds4_spec_conf_prefix, so leaving the call inside the decode
    // loop put an fopen/fscanf/fclose on every generated token. The sibling live
    // tunables (/tmp/ds4_awidth, /tmp/ds4_spec_q) are already hoisted this way.
    const float prefix_thr_base = spec_confidence_prefix_threshold();

    while (n_generated < n_gen) {
        const SpecClock::time_point cycle_t0 = SpecClock::now();
        const int ctx_len = feat_count < n_swa ? feat_count : n_swa;
        const int emit_left = n_gen - n_generated;
        const bool strict_cycle = batch_gate.strict_cycle();
        const bool use_confidence_width =
            batch_gate.active() && confidence_width_available;
        target.set_strict_verify(strict_cycle);

        // Decide the maximum useful width before invoking the support model.
        // Ratio-4 boundaries and rolling scheduler pauses allow only the seed
        // (q=1). Short tails leave this loop and use the normal AR seam.
        int q_step_cap = strict_cycle
                       ? std::min(q_cap, 4)
                       : std::min(q_cap, 4 - (pos & 3));
        if (adaptive_width && !use_confidence_width && !strict_cycle) {
            const int w_cap = (int) ewma_accept + 2;
            if (w_cap < q_step_cap) q_step_cap = w_cap;
        }
        // Never offer more tokens than this invocation may emit. This keeps KV
        // exactly at the spec→AR seam: accept <= q <= emit_left, so the final
        // verify cannot commit rejected phantom rows past the emitted stream.
        if (emit_left < q_step_cap) q_step_cap = emit_left;

        if (scheduler.tail_should_skip(emit_left)) {
            scheduler_tail_handoff = true;
            if (scheduler_log) {
                std::fprintf(stderr,
                             "[ds4-spec-sched] AR tail handoff pos=%d "
                             "remaining=%d\n",
                             pos, emit_left);
            }
            break;
        }
        const bool scheduled_skip =
            q_step_cap >= 2 && scheduler.take_scheduled_skip();
        if (scheduled_skip) q_step_cap = 1;

        if (scheduler_log && scheduled_skip) {
            std::fprintf(stderr,
                         "[ds4-spec-sched] target-only pos=%d remaining=%d "
                         "reason=profitability-pause\n",
                         pos, emit_left);
        }

        // Noise block = [seed] + [MASK]*(block-1).
        SpecClock::time_point t0 = SpecClock::now();
        if (q_step_cap >= 2) {
            noise_ids[0] = lt;
            for (int i = 1; i < block; i++) noise_ids[i] = drafter.mask_token_id;
            if (!target.embed_tokens(noise_ids.data(), block, noise_embed.data())) {
                ok = false;
                break;
            }

            // Drafter forward -> normalized states for token logits plus the
            // pre-output-norm states expected by the confidence head. The XDNA
            // seam is one whole support-model submission, following the
            // persistent/minimal-reconfiguration design in arXiv:2504.03083.
            // This monolithic loop collects immediately; resident batching can
            // retain the returned job while the GPU verifies another session.
            bool draft_ok = false;
            if (xdna_draft_compute && xdna_draft_compute->healthy()) {
                std::string provider_error;
                bool provider_ready = true;
                if (xdna_gpu_main && ctx_len > 0) {
                    provider_ready = deepseek4_dspark_project_main_context(
                        backend, drafter, feat_win.data(), ctx_len,
                        xdna_main_context, &xdna_context_kv);
                    if (!provider_ready) {
                        provider_error =
                            "GPU main-context projection failed";
                    }
                }

                if (provider_ready) {
                    XdnaDSparkDraftRequest request;
                    request.committed = pos;
                    request.ctx_len = ctx_len;
                    request.noise_embed = noise_embed.data();
                    request.ctx_features = xdna_gpu_main || ctx_len == 0
                        ? nullptr : feat_win.data();
                    request.main_context =
                        xdna_gpu_main && ctx_len > 0
                            ? xdna_main_context.data() : nullptr;
                    request.context_kv =
                        xdna_gpu_main && ctx_len > 0
                            ? xdna_context_kv.data() : nullptr;
                    auto job = xdna_draft_compute->submit(
                        request, &provider_error);
                    XdnaDSparkDraftOutput provider_output;
                    if (job && job->wait(provider_output, &provider_error)) {
                        local_hidden = std::move(provider_output.hidden);
                        confidence_hidden = std::move(
                            provider_output.confidence_hidden);
                        draft_ok = true;
                        if (xdna_compare &&
                            xdna_compared < xdna_compare_steps) {
                            std::vector<float> gpu_hidden;
                            std::vector<float> gpu_confidence_hidden;
                            const bool gpu_compare_ok =
                                deepseek4_dspark_draft_forward(
                                    backend, drafter, noise_embed.data(),
                                    ctx_len > 0 ? feat_win.data() : nullptr,
                                    ctx_len, pos, gpu_hidden,
                                    &gpu_confidence_hidden);
                            bool compare_ok = gpu_compare_ok;
                            if (gpu_compare_ok) {
                                compare_ok = dspark_log_xdna_compare(
                                    "normalized_hidden", local_hidden,
                                    gpu_hidden) &&
                                    dspark_log_xdna_compare(
                                        "confidence_hidden",
                                        confidence_hidden,
                                        gpu_confidence_hidden);
                            } else {
                                std::fprintf(stderr,
                                             "[ds4-spec] XDNA compare GPU "
                                             "reference failed\n");
                            }
                            ++xdna_compared;
                            if (!compare_ok && xdna_compare_required) {
                                std::fprintf(stderr,
                                             "[ds4-spec] XDNA hidden "
                                             "differential failed (required)\n");
                                ok = false;
                                break;
                            }
                        }
                    }
                }
                if (!draft_ok) {
                    std::fprintf(stderr,
                                 "[ds4-spec] XDNA drafter failed: %s%s\n",
                                 provider_error.c_str(),
                                 xdna_draft_compute->failure_is_fatal()
                                     ? " (required)" : "; falling back to GPU");
                    if (xdna_draft_compute->failure_is_fatal()) {
                        ok = false;
                        break;
                    }
                }
            }
            if (!draft_ok) {
                draft_ok = deepseek4_dspark_draft_forward(
                    backend, drafter, noise_embed.data(),
                    ctx_len > 0 ? feat_win.data() : nullptr,
                    ctx_len, pos, local_hidden,
                    use_confidence_width ? &confidence_hidden : nullptr);
            }
            if (!draft_ok) {
                std::fprintf(stderr, "[ds4-spec] drafter forward failed\n");
                ok = false;
                break;
            }
        }
        tm_draft += spec_ms_since(t0);

        if (debug && q_step_cap >= 2) {
            size_t lh_nan = 0; double lh_ss = 0;
            for (float v : local_hidden) { if (!std::isfinite(v)) lh_nan++; else lh_ss += (double) v * v; }
            std::fprintf(stderr, "[ds4-spec] hidden nnan=%zu/%zu rms=%.4f ctx_len=%d\n",
                         lh_nan, local_hidden.size(),
                         lh_nan < local_hidden.size() ? std::sqrt(lh_ss / (double) local_hidden.size()) : 0.0,
                         ctx_len);
        }

        // DSpark Markov chain over the first q_cap-1 candidates. Reference
        // predicts token i+1 from block slot i, so prepend a dummy row 0 and
        // let the (row-0-skipping) chain use slots 1..q-1.
        t0 = SpecClock::now();
        draft_tok.clear();
        draft_confidence.clear();
        // Set when the confident-prefix rule keeps zero candidates, so the
        // scheduler can apply ds4's no-draft pause (ds4.c:47655-47675).
        bool confidence_no_draft = false;
        bool ds_ok = false;
        if (q_step_cap >= 2) {
            std::memcpy(padded_hidden.data() + n_embd, local_hidden.data(),
                        sizeof(float) * (size_t) n_embd * block);
            if (use_confidence_width) {
                std::memcpy(padded_confidence_hidden.data() + n_embd,
                            confidence_hidden.data(),
                            sizeof(float) * (size_t) n_embd * block);
            }
            const bool fused_ok = dspark_markov_correct_greedy_chain_fused(
                            dw, backend, target.lm_head_tensor(), padded_hidden.data(),
                            q_step_cap, lt, draft_tok,
                            use_confidence_width ? &draft_confidence : nullptr,
                            use_confidence_width ? padded_confidence_hidden.data() : nullptr);
            ds_ok = fused_ok;
            if (!fused_ok) {
                // A failed fused head may have partially populated confidence.
                // The fallback Markov/projected tokens were not scored by that
                // head, so retaining those values would apply a confidence
                // decision to unrelated candidates.
                draft_confidence.clear();
                ds_ok = dspark_markov_correct_greedy_chain(dw, backend, target,
                            padded_hidden.data(), q_step_cap, lt, 0.0f, draft_tok);
            }
            if (!ds_ok || (int) draft_tok.size() < 2) {
                // Fallback: plain projection of the block hiddens.
                std::vector<int32_t> pj;
                if (!target.project_hidden_to_tokens(local_hidden.data(), q_step_cap - 1, pj)) {
                    ok = false;
                    break;
                }
                draft_tok.clear();
                draft_confidence.clear();
                draft_tok.push_back(lt);
                for (int i = 0; i < q_step_cap - 1; i++) draft_tok.push_back(pj[i]);
            }
        } else {
            draft_tok.push_back(lt);   // q=1: seed only, no speculation
        }
        // Confident-prefix rule (ds4 dspark_confident_prefix_len, ds4.c:32205).
        //
        // draft_confidence[i] is the sigmoid survival probability for candidate
        // i, i.e. draft_tok[i + 1] (draft_tok[0] is the seed). The confidences
        // are already sigmoid'd in the head graph (dspark_head.cpp:298), so they
        // are directly comparable to ds4's sigmoid_stable(confidence_logits[i]).
        //
        // Accept the longest leading run that individually clears the threshold
        // and truncate there. A run of zero is legal and means "no speculation
        // this step" — the seed alone, verified as a plain target eval. That is
        // the outcome ds4 reaches via proposal_len = 0, and it is what feeds the
        // scheduler's no-draft pause below.
        if (prefix_thr_base > 0.0f && use_confidence_width &&
            !draft_confidence.empty() && draft_tok.size() >= 2) {
            // OPT-IN confident-prefix rule (ds4 dspark_confident_prefix_len).
            // ctx_relax stays applied for A/B parity but defaults OFF (scale
            // 1.0); relaxing a per-position survival gate is not recommended,
            // since the measurements contradict its premise.
            const float thr = prefix_thr_base *
                              spec_ctx_confidence_scale(pos, ctx_relax_enabled);
            size_t confident = 0;
            while (confident < draft_confidence.size() &&
                   draft_confidence[confident] >= thr) {
                ++confident;
            }
            // Never keep more candidates than we have confidences for: an
            // unscored candidate has no survival estimate and must not ride
            // along on the strength of its predecessors.
            const size_t keep = confident + 1;   // seed + confident candidates
            if (draft_tok.size() > keep) draft_tok.resize(keep);
            if (confident == 0) confidence_no_draft = true;
        } else if (use_confidence_width && draft_confidence.size() >= 2 &&
                   draft_tok.size() >= 3) {
            // DEFAULT: legacy cumulative-product bucket policy (unchanged).
            const float ctx_scale = spec_ctx_confidence_scale(pos, ctx_relax_enabled);
            const float q3_thr = kConfidenceQ3Threshold * ctx_scale;
            const float q4_thr = kConfidenceQ4Threshold * ctx_scale;
            const float confidence_p2 = draft_confidence[0] * draft_confidence[1];
            int selected_q = confidence_p2 >= q3_thr ? 3 : 2;
            if (selected_q == 3 && draft_confidence.size() >= 3 && draft_tok.size() >= 4) {
                const float confidence_p3 = confidence_p2 * draft_confidence[2];
                if (confidence_p3 >= q4_thr) selected_q = 4;
            }
            if ((int) draft_tok.size() > selected_q) draft_tok.resize((size_t) selected_q);
        } else if (use_confidence_width && !strict_cycle) {
            // The fused head should always return confidence for a compatible
            // artifact. Preserve the old policy if a backend cannot do so.
            const int selected_q = (int) ewma_accept + 2;
            if ((int) draft_tok.size() > selected_q) draft_tok.resize((size_t) selected_q);
        }
        if ((int) draft_tok.size() > q_step_cap) draft_tok.resize(q_step_cap);
        const int q = (int) draft_tok.size();   // seed + candidates
        tm_head += spec_ms_since(t0);

        // No usable candidate this step: back off rather than re-drafting every
        // step (ds4.c:47655-47675). Reachable only because the confident-prefix
        // rule can keep zero — the previous bucketed policy floored at one.
        if (confidence_no_draft && q <= 1) {
            const bool have_conf = !draft_confidence.empty();
            const uint32_t skip = scheduler.note_no_draft(
                have_conf, have_conf ? draft_confidence[0] : 0.0f);
            if (scheduler_log && skip != 0) {
                std::fprintf(stderr,
                    "[ds4-spec-sched] no-draft pause skip=%u pos=%d "
                    "confidence0=%s%.3f lifetime_accepted=%llu long_accept=%d\n",
                    skip, pos, have_conf ? "" : "n/a ",
                    have_conf ? draft_confidence[0] : 0.0f,
                    (unsigned long long) scheduler.lifetime_accepted(),
                    (int) scheduler.long_accept_seen());
            }
        }

        if (debug) {
            std::fprintf(stderr, "[ds4-spec] dbg ds_ok=%d q=%d lt=%d draft=[%d %d %d %d]\n",
                         (int) ds_ok, q, lt,
                         q > 0 ? draft_tok[0] : -1, q > 1 ? draft_tok[1] : -1,
                         q > 2 ? draft_tok[2] : -1, q > 3 ? draft_tok[3] : -1);
        }

        // ── Rollback state save for the approximate verifier ──
        // Exact prefix verification never feeds a rejected token, so it needs
        // neither a cache copy nor the rolling-state rollback structure.
        t0 = SpecClock::now();
        if (full_snap && !strict_cycle) {
            if (!target.snapshot_kv()) {
                std::fprintf(stderr, "[ds4-spec] snapshot failed\n");
                ok = false;
                break;
            }
        } else if (!strict_cycle) {
            deepseek4_spec_rollback_save(target_cache, rollback, pos, q);
        }
        tm_save += spec_ms_since(t0);

        // First ratio-4 boundary position touched by this verify (p % 4 == 3).
        const int first_boundary = pos + (3 - (pos & 3));
        const bool boundary_crossed = first_boundary <= pos + q - 1;

        // ── Target verification + feature capture ──
        t0 = SpecClock::now();
        int verify_last = -1;
        if (!target.verify_batch(draft_tok, pos, verify_last, &tgt_am)) {
            if (full_snap && !strict_cycle) {
                if (!target.restore_kv()) {
                    std::fprintf(stderr, "[ds4-spec] restore after verify failure failed\n");
                }
            } else if (!strict_cycle) {
                deepseek4_spec_rollback_apply(
                    rollback, target_w, target_cache, pos, boundary_crossed);
            }
            std::fprintf(stderr, "[ds4-spec] verify failed\n");
            ok = false;
            break;
        }
        tm_verify += spec_ms_since(t0);

        // Accept the longest matching prefix. accept counts the seed (slot 0)
        // plus each candidate the target agrees with.
        int accept = 1;
        for (int i = 0; i < q - 1; i++) {
            if (i >= (int) tgt_am.size()) break;
            if (draft_tok[i + 1] == tgt_am[i]) accept++;
            else break;
        }
        if ((int) tgt_am.size() < accept) {
            std::fprintf(stderr,
                         "[ds4-spec] verifier returned no bonus logit "
                         "(q=%d accept=%d rows=%zu)\n",
                         q, accept, tgt_am.size());
            ok = false;
            break;
        }
        int matched = accept - 1;                             // accepted candidates
        int bonus = tgt_am[accept - 1];                       // target's token at the accept point
        int commit_pos = pos + accept;                        // seed + accepted candidates in KV
        agreement_offered += q - 1;
        agreement_matched += matched;
        if (spec_env_flag("DFLASH_DS4_AGREEMENT_LOG") && q > 1) {
            std::fprintf(stderr,
                         "[ds4-agreement] ctx=%d pos=%d offered=%d matched=%d "
                         "rate=%.3f draft0=%d target0=%d\n",
                         ctx_len, pos, q - 1, matched,
                         (double) matched / (double) (q - 1),
                         draft_tok[1], tgt_am.empty() ? -1 : tgt_am[0]);
        }

        if (timing && steps < 8 && q >= 2) {
            // Alignment probe: draft candidate i should match tgt_am[i-1]. A
            // consistent draft[i]==tgt_am[i] pattern instead = off-by-one.
            std::fprintf(stderr, "[ds4-spec-cmp] step=%ld pos=%d draft=[%d %d %d] tgt=[%d %d %d %d] acc=%d\n",
                         steps, pos,
                         q > 1 ? draft_tok[1] : -1, q > 2 ? draft_tok[2] : -1, q > 3 ? draft_tok[3] : -1,
                         !tgt_am.empty() ? tgt_am[0] : -1,
                         tgt_am.size() > 1 ? tgt_am[1] : -1,
                         tgt_am.size() > 2 ? tgt_am[2] : -1,
                         tgt_am.size() > 3 ? tgt_am[3] : -1,
                         accept);
        }

        // ── Rollback: truncate to the committed prefix ──
        // The bonus token is DEFERRED: it becomes the next step's seed, whose
        // KV is written then.
        t0 = SpecClock::now();
        bool exact_replay = false;
        if (!strict_cycle && accept < q) {
            // Dwarfstar-style partial accept: discard the entire batched
            // verifier mutation, then replay only the committed prefix through
            // the ordinary q=1 target graph. Besides restoring exact KV and
            // compressor state, this replaces the batched mismatch-row bonus
            // with the authoritative q=1 bonus.
            std::vector<int32_t> kv_toks;
            kv_toks.push_back(lt);
            for (int i = 1; i < accept; i++) kv_toks.push_back(draft_tok[i]);
            if (full_snap) {
                if (!target.restore_kv()) {
                    std::fprintf(stderr, "[ds4-spec] snapshot restore failed\n");
                    ok = false;
                    break;
                }
            } else {
                // A full rollback always restores the pre-verify compressor
                // halves and HC state; the q=1 replay will recreate any
                // legitimate boundary inside the accepted prefix.
                deepseek4_spec_rollback_apply(
                    rollback, target_w, target_cache, pos, true);
            }
            int replay_last = -1;
            std::vector<int32_t> replay_am;
            if (!target.verify_exact_prefix(
                    kv_toks, pos, replay_last, &replay_am) ||
                replay_am.empty()) {
                std::fprintf(stderr, "[ds4-spec] replay verify failed\n");
                ok = false;
                break;
            }
            // A q-wide accept can itself cross a near-tied q=1 decision. In
            // that case the exact replay stops at the earlier disagreement;
            // shrink the commit rather than failing the request or forcing a
            // token the ordinary target graph did not choose.
            if ((int) replay_am.size() < accept) {
                accept = (int) replay_am.size();
                matched = accept - 1;
                commit_pos = pos + accept;
            }
            bonus = replay_am.back();
            exact_replay = true;
        }
        // Exact-prefix mode and accept==q already end at the precise commit
        // position, so their target state is retained directly.
        tm_apply += spec_ms_since(t0);

        // Push the committed positions' features (slots 0..accept-1 = positions
        // pos..pos+accept-1) into the drafter's context window.
        t0 = SpecClock::now();
        const std::vector<float> & feats = target.last_features();
        const int fN = (strict_cycle || exact_replay)
            ? target.last_verify_n()
            : accept;
        for (int i = 0; i < fN; i++) push_feature(feats.data() + (size_t) i * feat_row);
        tm_feat += spec_ms_since(t0);

        // Output tokens this step = accepted candidates + bonus.
        bool hit_eos = false;
        for (int i = 1; i <= accept; i++) {
            const int t = (i < accept) ? draft_tok[i] : bonus;
            out_tokens.push_back(t);
            n_generated++;
            if (on_token && !on_token(t)) {
                stop_requested = true;
                break;
            }
            if (target.is_eos(t)) { hit_eos = true; break; }
            if (n_generated >= n_gen) break;
        }
        pos = commit_pos;              // seed + accepted candidates now in KV
        lt = bonus;                    // deferred bonus becomes next seed
        accept_sum += matched;
        offered_sum += q - 1;
        ewma_accept = 0.7 * ewma_accept + 0.3 * (double) matched;
        steps++;

        if (!force_strict_verify) {
            const bool was_active = batch_gate.active();
            batch_gate.note_cycle(q, accept);
            if (!was_active && batch_gate.active() && scheduler_log) {
                std::fprintf(stderr,
                    "[ds4-spec] batch verifier qualified after %u exact tokens\n",
                    batch_gate.progress());
            }
        }

        const double cycle_ms = spec_ms_since(cycle_t0);
        if (q == 1) {
            // Ratio-boundary, scheduler-pause and no-draft steps are seed-only,
            // but they still run through the verify graph — so this is a COLD
            // START baseline only, superseded as soon as the AR path reports a
            // genuine single-token target eval (dspark_worker_note_target_eval).
            scheduler.observe_target_eval_fallback(cycle_ms);
            if (confidence_no_draft) {
                const DSparkSchedulerDecision decision =
                    scheduler.note_spec_cycle(0, cycle_ms, true);
                if (scheduler_log && decision.paused) {
                    std::fprintf(stderr,
                        "[ds4-spec-sched] no-draft window pause=%u "
                        "avg_accept=%.2f reasons=%s%s\n",
                        decision.skip_cycles, decision.avg_accepted,
                        decision.low_accept ? "low-accept " : "",
                        decision.many_no_draft ? "many-no-draft" : "");
                }
            }
        } else {
            const DSparkSchedulerDecision decision =
                scheduler.note_spec_cycle((uint32_t) matched, cycle_ms);
            if (scheduler_log && decision.paused) {
                std::fprintf(stderr,
                    "[ds4-spec-sched] pause=%u avg_accept=%.2f "
                    "extra_per_accept=%.1fms saved=%.1fms extra=%.1fms "
                    "reasons=%s%s%s%s\n",
                    decision.skip_cycles, decision.avg_accepted,
                    decision.extra_per_accept_ms,
                    decision.saved_ms, decision.extra_ms,
                    decision.low_accept ? "low-accept " : "",
                    decision.slow_accept ? "slow-accept " : "",
                    decision.measured_unprofitable ? "unprofitable " : "",
                    decision.many_no_draft ? "many-no-draft" : "");
            }
        }
        if (timing && (steps <= 4 || (steps & 31) == 0)) {
            std::fprintf(stderr,
                "[ds4-spec-t] step=%ld q=%d acc=%d | draft=%.1f head=%.1f save=%.1f "
                "verify=%.1f apply=%.1f feat=%.1f ms (cum means)\n",
                steps, q, accept,
                tm_draft / steps, tm_head / steps, tm_save / steps,
                tm_verify / steps, tm_apply / steps, tm_feat / steps);
        }
        if (hit_eos || stop_requested) break;
    }

    const double total_ms = spec_ms_since(run_t0);
    if (accept_rate_out) {
        *accept_rate_out = offered_sum > 0
            ? (float) accept_sum / (float) offered_sum
            : 0.0f;
    }
    std::fprintf(stderr,
                 "[ds4-spec] gen=%d steps=%ld mean_accept=%.2f/%.2f "
                 "q_cap=%d verify=%s full_snap=%d\n",
                 n_generated, steps,
                 steps ? (double) accept_sum / steps : 0.0,
                 steps ? (double) offered_sum / steps : 0.0, q_cap,
                 force_strict_verify ? "exact-prefix" :
                 (batch_warmup_tokens ? "guarded-batch-replay" : "batch-replay"),
                 (int) full_snap);
    if (spec_env_flag("DFLASH_DS4_AGREEMENT_LOG")) {
        std::fprintf(stderr,
                     "[ds4-agreement] summary offered=%ld matched=%ld rate=%.3f\n",
                     agreement_offered, agreement_matched,
                     agreement_offered > 0
                         ? (double) agreement_matched / (double) agreement_offered
                         : 0.0);
    }
    if (scheduler_log && scheduler.enabled()) {
        std::fprintf(stderr,
                     "[ds4-spec-sched] target=%.1fms paused_steps=%llu "
                     "tail_handoff=%d\n",
                     scheduler.target_eval_ms(),
                     (unsigned long long) scheduler.skipped_cycles(),
                     scheduler_tail_handoff ? 1 : 0);
    }
    if (steps > 0) {
        std::fprintf(stderr,
            "[ds4-spec-t] TOTAL %.1f ms, %ld steps (%.1f ms/step), %d tok (%.1f tok/s) | "
            "means: draft=%.1f head=%.1f save=%.1f verify=%.1f apply=%.1f feat=%.1f ms\n",
            total_ms, steps, total_ms / steps, n_generated,
            total_ms > 0 ? n_generated * 1000.0 / total_ms : 0.0,
            tm_draft / steps, tm_head / steps, tm_save / steps,
            tm_verify / steps, tm_apply / steps, tm_feat / steps);
    }
    if (timing && steps > 0) {
        const double s = 1000.0 * steps;   // us -> ms per-step means
        std::fprintf(stderr,
            "[ds4-spec-t] verify tel/step: hc_pre_a=%.1f attn_b=%.1f attn_c=%.1f attn_r=%.1f "
            "hc_post_a=%.1f hc_pre_f=%.1f route(b/c/r/s)=%.1f/%.1f/%.1f/%.1f "
            "ffn(b/c/r)=%.1f/%.1f/%.1f eval=%.1f hot=%.1f cold=%.1f comb=%.1f part=%.1f "
            "full(b/s/c/r)=%.1f/%.1f/%.1f/%.1f "
            "fused_verify(calls/rows/avg_ms)=%llu/%llu/%.1f "
            "q2(calls/avg_ms)=%llu/%.1f q3=%llu/%.1f q4=%llu/%.1f "
            "ghits=%llu gbuilds=%llu ms\n",
            tel.hc_pre_attn_us / s, tel.attn_build_us / s, tel.attn_compute_us / s,
            tel.attn_read_us / s, tel.hc_post_attn_us / s, tel.hc_pre_ffn_us / s,
            tel.route_build_us / s, tel.route_compute_us / s, tel.route_read_us / s,
            tel.route_select_us / s,
            tel.ffn_build_us / s, tel.ffn_compute_us / s, tel.ffn_read_us / s,
            tel.ffn_eval_us / s, tel.ffn_hot_us / s, tel.ffn_cold_us / s,
            tel.ffn_combine_us / s, tel.ffn_partition_us / s,
            tel.full_graph_build_us / s, tel.full_graph_set_us / s,
            tel.full_graph_compute_us / s, tel.full_graph_read_us / s,
            (unsigned long long) tel.fused_verify_calls,
            (unsigned long long) tel.fused_verify_rows,
            tel.fused_verify_calls
                ? tel.fused_verify_compute_us /
                    (1000.0 * (double) tel.fused_verify_calls) : 0.0,
            (unsigned long long) tel.fused_verify_q_calls[2],
            tel.fused_verify_q_calls[2]
                ? tel.fused_verify_q_compute_us[2] /
                    (1000.0 * (double) tel.fused_verify_q_calls[2]) : 0.0,
            (unsigned long long) tel.fused_verify_q_calls[3],
            tel.fused_verify_q_calls[3]
                ? tel.fused_verify_q_compute_us[3] /
                    (1000.0 * (double) tel.fused_verify_q_calls[3]) : 0.0,
            (unsigned long long) tel.fused_verify_q_calls[4],
            tel.fused_verify_q_calls[4]
                ? tel.fused_verify_q_compute_us[4] /
                    (1000.0 * (double) tel.fused_verify_q_calls[4]) : 0.0,
            (unsigned long long) tel.ffn_hot_graph_hits,
            (unsigned long long) tel.ffn_hot_graph_builds);
    }
    // Snapshot buffers must be released while their backend is still alive.
    // clear_snapshot() is idempotent, so the target destructor remains a
    // safety net for future exits that are added above this point.
    target.clear_snapshot();
    if (snap_backend) ggml_backend_free(snap_backend);
    return ok;
}

}  // namespace dflash::common
