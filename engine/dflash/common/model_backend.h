// Model daemon backend interface.
//
// Stable C++ seam between Ember's C ABI bridge and the vendored DeepSeek4
// implementation. Only operations exercised by the in-process server belong
// here; the old multi-architecture stdin daemon is not part of Ember.
//
// Concrete backends own their GPU resources, weight/cache lifecycle, and
// generation strategy (autoregressive, speculative decode, etc.).

#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "token_mask.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "continuous_batch_executor.h"
#include "sampler.h"
#include "thinking_budget.h"

namespace dflash::common {

// Token callback for streaming generation. Called once per committed token.
// Return true to continue generation, false to abort.
using TokenCallback = std::function<bool(int32_t token)>;

// Inference observer callback for live status updates. Called by backends
// at each spec-decode step to report phase/detail. When empty, backends
// skip the call (zero overhead).
//   phase: "draft", "verify", "accept", "prefill_chunk"
//   detail: JSON string with step-specific data
using InferenceObserver = std::function<void(const char * phase,
                                             const std::vector<int32_t> & tokens)>;

// ─── I/O handle passed to backend methods that need protocol output ─────
struct DaemonIO {
    // Optional token callback. When set, emit() calls this for each token
    // (excluding the -1 sentinel). If it returns false, the `cancelled`
    // flag is set and the caller should abort generation.
    TokenCallback on_token;
    mutable bool cancelled = false;

    // Optional inference observer for /status page. When set, backends call
    // this at each spec-decode step with draft tokens and phase info.
    InferenceObserver observer;

    // Optional keepalive invoked periodically during long prefill. Streaming
    // clients receive no tokens until decode begins, so a multi-second prefill
    // (e.g. a cold 25k-token agent prompt at ~250 tok/s) sends zero bytes and
    // trips client idle timeouts. Backends call this at prefill chunk
    // boundaries (rate-limited by the callback itself). Returning false means
    // the client has disconnected — the backend sets `cancelled` and aborts.
    // Mirrors antirez/ds4's server_prefill_progress `: prefill` heartbeat.
    std::function<bool()> on_prefill_keepalive;

    // Publish a committed token. Negative sentinels from the removed stdin
    // daemon are ignored because Ember streams through callbacks only.
    void emit(int32_t value) const {
        if (on_token && value >= 0 && !on_token(value)) cancelled = true;
    }

    // Return an IO handle that also invokes `cb` for emitted tokens.
    DaemonIO with_token_callback(const TokenCallback &callback) const {
        DaemonIO output = *this;
        if (!callback) return output;
        TokenCallback existing = output.on_token;
        output.on_token = [existing, callback](int32_t token) -> bool {
            return (!existing || existing(token)) && callback(token);
        };
        return output;
    }
};

// ─── Generate request/result ────────────────────────────────────────────

// Thinking-budget force-close hook. Mirrors antirez/ds4 ds4_eval.c's
// hard_limit_reply_budget semantics: when the budget remaining (n_gen
// minus tokens committed so far) falls to hard_limit_remaining, the
// next sampled tokens get overridden with close_token_ids in order,
// giving the model the remaining budget to write a visible answer
// after the injected close-tag sequence.
//
// Single vs multi-token close:
//   Qwen3.6: </think> is one added_token (id 248069). close_token_ids
//            has size 1. One override + budget_close_injected=true.
//   DeepSeek/laguna: </think> tokenizes to 3 ordinary tokens
//            ([1718, 37947, 32] for DS-V3). close_token_ids has
//            size 3. Three consecutive overrides, then resume.
//
// This is "Level 2" of our thinking-budget migration: in-process
// mid-stream force-close, KV-continuous. Beats Level 1's phase-2
// reprompt because the model never sees a fresh prefill — its KV
// state continues naturally after the injected close.
//
// Current implementation: AR-decode only. When budget_hook is set,
// backends MAY route generation through their AR path (skipping spec
// decode) — the perf trade-off is acceptable since this only kicks in
// for thinking-enabled requests. Spec-decode integration is a follow-up.
struct BudgetHook {
    // Multi-token close sequence injected when `(n_gen - committed)`
    // drops to `hard_limit_remaining`. For Qwen3.x this is the
    // canonical "Considering the limited time..." summarize-and-stop
    // lead-in (tokenized at server startup); for non-qwen arches it's
    // a single close-tag token. Empty = hook disabled.
    std::vector<int32_t> close_token_ids;
    // Bare natural </think> sequence. The budget hook is armed only until this
    // sequence appears in generated output. Keeping it separate from the
    // server-authored close_token_ids prevents a second close from being
    // injected after the model has naturally begun its visible answer.
    std::vector<int32_t> natural_close_token_ids;
    int                  hard_limit_remaining = 0;
};

// Optional image rows already projected to the language-model width.  The
// projector remains a separate, lazily loaded mmproj artifact; generation owns
// only these request-scoped rows.  `prompt_offset` points at the first repeated
// image placeholder token, which remains in `prompt` for PLE hashing.
struct VisionEmbeddingRun {
    int prompt_offset = 0;
    int grid_t = 0;
    int grid_h = 0;
    int grid_w = 0;
    std::vector<float> embeddings; // [merged image tokens, model embedding]
};

struct EncodedVisionImage {
    int grid_t = 0;
    int grid_h = 0;
    int grid_w = 0;
    std::vector<float> embeddings;
};

struct GenerateRequest {
    std::vector<int32_t>       prompt;
    std::vector<VisionEmbeddingRun> vision;
    int                        n_gen       = 0;
    SamplerCfg                 sampler;
    bool                       do_sample   = false;
    // Optional inline-snap: snapshot at this position after prefill.
    int                        snap_pos    = -1;
    int                        snap_slot   = -1;
    // Optional token callback for streaming. When set, backends call this
    // for each committed token. If it returns false, generation aborts
    // immediately. This is the primary mechanism for client-disconnect
    // cancellation in the native HTTP server.
    TokenCallback              on_token;
    // Tool call hint tokens: pre-tokenized structural tokens that are
    // predictable with ~100% confidence (XML tags, function name, param names).
    // When non-null, the spec decode loop uses these as draft overrides,
    // bypassing draft model computation for covered positions.
    const std::vector<int32_t> * hint_tokens = nullptr;
    // Optional env-gated dflash stall recovery: when spec decode is about to
    // emit early EOS after an action preamble, inject a bare tool-call XML
    // prefix and continue in AR with KV state intact.
    const std::vector<int32_t> * stall_tool_prefix_tokens = nullptr;
    const std::vector<int32_t> * stall_action_suffix_tokens = nullptr;
    const std::vector<int32_t> * stall_skip_tokens = nullptr;
    // Optional thinking-budget hook — see BudgetHook docs above.
    BudgetHook                 budget_hook;
    // DSML tool-call region markers, tokenized by the server (which owns the
    // tokenizer). While decode is inside such a region the prompt-echo
    // watchdog is disarmed: re-emitting prompt content as a tool argument is
    // legitimate — writing back a file that was just read, or retrying a
    // rejected call with the same large payload after a validation error.
    // Empty = the watchdog behaves exactly as before.
    std::vector<int32_t>       tool_region_open_ids;
    std::vector<int32_t>       tool_region_close_ids;
    // Optional constrained decoding. When set, the sampler masks disallowed
    // tokens while the mask reports active(). Null = unconstrained, which is
    // the behaviour of every path that does not opt in.
    std::shared_ptr<TokenMask> token_mask;
    // Common retry knob. Upper layers set this after a speculative decode
    // path returns success but emits no tokens, so each backend can route the
    // retry through its existing AR path without copying retry policy.
    bool                       force_ar_decode = false;
    // Independent prefill policy. Exact prefill is reserved for explicit
    // reference/validation requests; forcing target-only decode must not turn
    // a long prompt into thousands of q=1 graph launches.
    bool                       force_exact_prefill = false;
    // B6 — structural-token greedy sampling. When set, the AR decode loop calls
    // this before sampling each token; returning true forces greedy argmax for
    // that token regardless of the sampler temperature (used to keep tool-call
    // DSML scaffolding well-formed at temperature > 0). Only consulted when
    // logit processing is active (temp/penalties); greedy runs already ignore it.
    std::function<bool()>      force_greedy_next;
};

// Stable, backend-independent generation failure categories. Backends should
// use these for recurrent failures so callers do not need to understand
// architecture-specific strings. `generate_error_code()` is the daemon/API
// wire representation and must remain backward-compatible once published.
enum class GenerateErrorCode {
    Incomplete,
    AdapterUnavailable,
    ContextOverflow,
    SamplingUnsupported,
    PrefillFailed,
    DecodeSeedMissing,
    DecodeFailed,
    InvalidSnapshotSlot,
    ModelParked,
    BackendSpecific,
};

constexpr std::string_view generate_error_code(GenerateErrorCode error) {
    switch (error) {
    case GenerateErrorCode::Incomplete:          return "incomplete";
    case GenerateErrorCode::AdapterUnavailable:  return "adapter_unavailable";
    case GenerateErrorCode::ContextOverflow:     return "context_overflow";
    case GenerateErrorCode::SamplingUnsupported: return "sampling_unsupported";
    case GenerateErrorCode::PrefillFailed:       return "prefill_failed";
    case GenerateErrorCode::DecodeSeedMissing:   return "decode_seed_missing";
    case GenerateErrorCode::DecodeFailed:        return "decode_failed";
    case GenerateErrorCode::InvalidSnapshotSlot: return "invalid_snapshot_slot";
    case GenerateErrorCode::ModelParked:         return "model_parked";
    case GenerateErrorCode::BackendSpecific:     return "backend_specific";
    }
    return "unknown_error";
}

struct GenerateError {
    GenerateErrorCode code = GenerateErrorCode::Incomplete;
    std::string detail;
};

struct GenerateResult {
    // Default to an incomplete failure so a backend must explicitly call
    // succeed() before returning a successful result.
    std::optional<GenerateError> error = GenerateError{};
    std::vector<int32_t>       tokens;
    int                        prefill_tokens = 0;
    double                     prefill_s   = 0.0;
    double                     decode_s    = 0.0;
    // Actual execution policy, surfaced to server telemetry so a slow request
    // can be attributed without inferring behavior from throughput.
    std::string                prefill_mode = "unknown";
    std::string                prefill_reason = "unknown";
    // True when the backend's Level 2 hook injected the </think> close
    // sequence during this generation (vs. the model self-closing). The
    // server uses this to attribute close_kind correctly: if the model
    // produced </think> naturally we report "natural"; if the hook fired
    // we report "hard". Without this flag, decoding the phase-1 token
    // stream and grepping for "</think>" cannot distinguish the two
    // (the injected close decodes identically).
    bool                       budget_forced_close = false;
    // True iff the AR decode loop's post-close watchdog detected an n-gram
    // repetition loop and broke out early. Caller surfaces this so clients
    // can mark the answer as unreliable rather than treating the
    // (truncated) content as a clean response.
    bool                       degenerate_decode_close = false;
    // Machine-readable watchdog cause. Empty for an ordinary completion;
    // otherwise repetition_detected, reasoning_cycle_detected, or
    // prompt_echo_detected. Kept separate from the OpenAI finish_reason so
    // compatibility clients may still receive the standard "length" value.
    std::string                termination_reason;
    // DFlash chain accept rate: accepted_draft_tokens / total_draft_positions.
    // 0.0 when spec decode did not run (AR fallback or no draft model).
    float                      accept_rate     = 0.0f;
    // True when spec decode actually ran (accept_rate==0 still needs a bandit update).
    bool                       spec_decode_ran = false;
    // Resident XDNA proposal-pipeline attribution. These are zero for ordinary
    // AR and monolithic DSpark. Provider age includes hidden overlap; provider
    // block is the portion paid on this request's critical path.
    int                        spec_cycles = 0;
    double                     spec_provider_age_s = 0.0;
    double                     spec_provider_block_s = 0.0;
    double                     spec_head_s = 0.0;
    double                     spec_verify_s = 0.0;
    // True when decode emitted only tokens that the API layer suppresses
    // (for example an immediate EOS/EOT). This is semantically equivalent
    // to zero output for clients and should take the same AR retry path as
    // an empty token vector.
    bool                       empty_visible_output = false;
    // True iff this generation actually persisted a reusable KV snapshot into
    // GenerateRequest::snap_slot. Callers that keep a logical prefix cache MUST
    // gate their token->slot commit on this: committing on generation success
    // alone poisons the cache (a later lookup reports a hit the backend can't
    // honor, forcing a full re-prefill with no repair snapshot reserved).
    bool                       snapshot_saved = false;

    bool ok() const {
        return !error.has_value();
    }

    std::string_view error_code() const {
        return error ? generate_error_code(error->code) : std::string_view{};
    }

    std::string_view error_detail() const {
        return error ? std::string_view(error->detail) : std::string_view{};
    }

    void succeed() {
        error.reset();
    }

    void fail(GenerateErrorCode code, std::string detail = {}) {
        error = GenerateError{code, std::move(detail)};
    }
};

// Optional resident-session interface used by the continuous-batch
// coordinator. Implementations keep model-coupled mutable state (KV, logits,
// sampler/RNG and decode frontier) isolated per session while sharing weights.
// Backends that do not implement this interface continue to use ModelBackend's
// monolithic generate() API.
struct ResidentBatchBackend : ContinuousBatchWorkBackend {
    struct SessionStatus {
        int prefilled_tokens = 0;
        bool decode_ready = false;
        bool terminal = false;
        bool cancelled = false;
        bool failed = false;
    };

    virtual bool resident_session_create(
        ContinuousBatchSessionId id,
        const GenerateRequest &request,
        const DaemonIO &io,
        int restore_slot,
        std::string *error) = 0;
    virtual bool resident_session_destroy(ContinuousBatchSessionId id) = 0;
    virtual bool resident_session_cancel(ContinuousBatchSessionId id) = 0;
    virtual bool resident_session_decode_ready(
        ContinuousBatchSessionId id) const = 0;
    virtual SessionStatus resident_session_status(
        ContinuousBatchSessionId id) const = 0;
    virtual GenerateResult resident_session_result(
        ContinuousBatchSessionId id) const = 0;
    virtual bool resident_session_snapshot(
        ContinuousBatchSessionId id, int slot) = 0;
};

// ─── Backend interface ──────────────────────────────────────────────────
struct ModelBackend {
    virtual ~ModelBackend() = default;

    // ── Generation ───────────────────────────────────────────────────
    // Run a full prefill + decode cycle. Backend owns the strategy
    // (autoregressive, speculative, batched, …).
    GenerateResult generate(const GenerateRequest & req, const DaemonIO & io) {
        GenerateResult result = generate_impl(req, io);
        if (!should_retry_empty_spec_decode(req, result)) return result;

        std::fprintf(stderr,
            "[backend] spec-decode produced zero tokens after %.3f s decode; "
            "retrying with AR decode\n",
            result.decode_s);
        GenerateRequest retry = req;
        retry.force_ar_decode = true;
        return merge_empty_spec_retry_result(result, generate_impl(retry, io));
    }

    virtual GenerateResult generate_impl(const GenerateRequest & req,
                                         const DaemonIO & io) = 0;

    // Decode, preprocess, and project one encoded still image. Architectures
    // without a vision tower fail closed. Qwen loads its separate BF16 mmproj
    // provider on the first call, so text-only startup/residency is unchanged.
    virtual bool encode_vision_image(const uint8_t *, size_t,
                                     EncodedVisionImage &,
                                     std::string & error) {
        error = "vision input is not supported by this backend";
        return false;
    }

    // ── Snapshots ────────────────────────────────────────────────────
    // With right-sized CPU-resident snapshots, each slot costs only
    // ~(cur_pos × 5 KB) of system RAM, so we can afford many slots.
    static constexpr int kMaxSlots = 64;

    // Release cached compute graphs / scratch arenas held for reuse, without
    // unloading the model. Called when the server has been idle: the caches are
    // thread_local on the generation worker, so they ratchet to the largest
    // context ever served and are otherwise only freed at model teardown.
    // Rebuilt lazily on the next request. Default: nothing cached, nothing to do.
    virtual void release_idle_graphs() {}

    virtual bool snapshot_save(int slot) = 0;
    virtual void snapshot_free(int slot) = 0;
    virtual bool snapshot_used(int slot) const = 0;
    virtual int  snapshot_cur_pos(int slot) const = 0;

    // RESTORE <slot> <prompt_path> <n_gen> — restore snapshot + generate.
    // Backend handles the diff-prefill and decode internally.
    GenerateResult restore_and_generate(int slot, const GenerateRequest & req,
                                        const DaemonIO & io) {
        GenerateResult result = restore_and_generate_impl(slot, req, io);
        if (!should_retry_empty_spec_decode(req, result)) return result;

        std::fprintf(stderr,
            "[backend] restored spec-decode slot=%d produced zero tokens after "
            "%.3f s decode; retrying with AR decode\n",
            slot, result.decode_s);
        GenerateRequest retry = req;
        retry.force_ar_decode = true;
        return merge_empty_spec_retry_result(result,
                                             restore_and_generate_impl(slot, retry, io));
    }

    virtual GenerateResult restore_and_generate_impl(int slot,
                                                     const GenerateRequest & req,
                                                     const DaemonIO & io) = 0;

    static bool should_retry_empty_spec_decode(const GenerateRequest & req,
                                               const GenerateResult & result) {
        return req.n_gen > 0
            && !req.force_ar_decode
            && result.ok()
            && result.spec_decode_ran
            && (result.tokens.empty() || result.empty_visible_output);
    }

    static GenerateResult merge_empty_spec_retry_result(
            const GenerateResult & first, GenerateResult retry) {
        retry.prefill_s += first.prefill_s;
        retry.prefill_tokens += first.prefill_tokens;
        retry.decode_s += first.decode_s;
        if (retry.prefill_mode != first.prefill_mode)
            retry.prefill_mode = "mixed";
        retry.prefill_reason = "empty_spec_retry";
        retry.accept_rate = first.accept_rate;
        retry.spec_decode_ran = first.spec_decode_ran || retry.spec_decode_ran;
        retry.budget_forced_close =
            first.budget_forced_close || retry.budget_forced_close;
        retry.degenerate_decode_close =
            first.degenerate_decode_close || retry.degenerate_decode_close;
        if (retry.termination_reason.empty())
            retry.termination_reason = first.termination_reason;
        // The AR retry re-runs prefill and re-saves the inline snapshot, but OR
        // the flags so a snapshot written on the first (spec) attempt is kept.
        retry.snapshot_saved = first.snapshot_saved || retry.snapshot_saved;
        return retry;
    }

    // ── Snapshot serialization (for ondisk prefix cache) ─────────────
    // Read-only reference to a snapshot's ggml tensors for serialization.
    struct SnapshotRef {
        ggml_context        * ctx     = nullptr;
        ggml_backend_buffer_t buf     = nullptr;
        int                   cur_pos = 0;
        int32_t               last_tok = -1;  // last prefill token (for decode seeding)
    };

    // Export a snapshot's tensor context + buffer for read-only access.
    // Ownership is NOT transferred — caller must only read tensor data.
    // Returns empty ref (ctx==nullptr) if slot is invalid or unused.
    virtual SnapshotRef snapshot_ref(int slot) const { (void)slot; return {}; }

    // Import a deserialized snapshot into the given slot. Backend takes
    // ownership of ctx and buf on success. On failure (returns false),
    // the caller is responsible for freeing ctx and buf.
    virtual bool snapshot_adopt(int slot, ggml_context * ctx,
                                ggml_backend_buffer_t buf, int cur_pos,
                                int32_t last_tok = -1) {
        (void)slot; (void)ctx; (void)buf; (void)cur_pos; (void)last_tok;
        return false;
    }

    // ── DFlash speculative decode support ────────────────────────────
    // Returns true if this backend can participate in DFlash spec decode
    // (i.e. it implements the DFlashTarget interface).
    virtual bool supports_dflash_spec_decode() const { return false; }

    // Return the DFlashTarget adapter for this backend. Only valid when
    // supports_dflash_spec_decode() returns true. Default returns nullptr.
    virtual class DFlashTarget * dflash_target() { return nullptr; }

    // Release oversized scratch buffers between requests to prevent VRAM
    // growth over time. Default is a no-op.
    virtual void release_scratch() {}

    // ── Routing data collection ──────────────────────────────────────
    // ── Cleanup ──────────────────────────────────────────────────────
    // Release all resources (weights, cache, snapshots, drafter).
    // Spark day-one bootstrap: when true, the server feeds local agent history
    // (Claude Code + Codex) through generate() before serving, then calls
    // spark_bootstrap_finalize to save the profile and rebuild placement so the
    // first session is already calibrated. Default: unsupported (live-traffic
    // calibration still applies).
    virtual bool spark_wants_bootstrap() const { return false; }
    virtual bool spark_bootstrap_finalize(const std::string & profile_path) {
        (void)profile_path; return false;
    }

    virtual void shutdown() = 0;
};

}  // namespace dflash::common
