// Backend ABI — the stable contract between Ember's C server and the model
// forward pass. Two implementations target it:
//   - backend_stub.c   : deterministic, GPU-free; drives the whole server
//                        pipeline in tests and CI.
//   - backend_dflash.cc: (milestone 4) extern-C shim over lucebox's
//                        libdflash_common.a — the tuned ROCMFP/graphs/DSpark
//                        forward pass + its byte-exact tokenizer.
//
// The server never sees ggml/HIP; it only calls these functions. Swapping the
// stub for the real bridge changes nothing above this line.
#ifndef EMBER_BACKEND_H
#define EMBER_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ember_backend ember_backend;

typedef enum {
    EMBER_DS4_PREFILL_SPARSE = 0,
    EMBER_DS4_PREFILL_EXACT  = 1,
    EMBER_DS4_PREFILL_DENSE  = 2,
} ember_ds4_prefill_mode;

typedef struct {
    const char *model_path;
    int         max_ctx;
    const char *model_name;   // advertised id for /v1/models
    int         expert_top_k; // 0 = model default
    const char *kv_cache_dir; // NULL = disk KV cache disabled
    long        kv_cache_mb;  // disk budget (0 = default)
    int         batch_sessions; // resident continuous-batch slots (1 = legacy)
    ember_ds4_prefill_mode ds4_prefill_mode; // sparse default; exact for quality reference
} ember_backend_config;

// Load a model. Returns NULL on failure and sets *err (caller frees) if non-NULL.
ember_backend *ember_backend_load(const ember_backend_config *cfg, char **err);
void           ember_backend_free(ember_backend *b);

// ── tokenizer (model-coupled; reused, not reimplemented) ──
// Encode `text` → token ids. Writes a malloc'd array to *ids_out (caller frees);
// returns the count (>=0), or -1 on error.
int  ember_backend_encode(ember_backend *b, const char *text, int32_t **ids_out);
// Detokenize one token to its raw byte string (borrowed, valid for backend life).
const char *ember_backend_token_text(ember_backend *b, int32_t id);

// ── generation ──
// on_token: called per generated token; return false to cancel (client gone).
// on_prefill: called periodically during prefill for the SSE keepalive; return
//   false to cancel. Either may be NULL.
typedef bool (*ember_token_cb)(int32_t token, void *ud);
typedef bool (*ember_keepalive_cb)(void *ud);

typedef struct {
    const int32_t     *prompt;
    int                n_prompt;
    int                max_tokens;
    bool               greedy;       // temperature==0
    // Decode policy and prefill policy are intentionally independent. An
    // immediate tool-result continuation uses target-only AR decode so a
    // completed action is not spuriously repeated, but can still use the
    // configured batched prefill path.
    bool               force_ar_decode;
    // Reference/validation escape hatch. This is never implied by target-only
    // decode; setting it explicitly selects tokenwise exact prompt prefill.
    bool               force_exact_prefill;
    float              temperature;
    float              top_p;
    int                top_k;        // 0 = disabled
    float              min_p;        // parsed for ds4 parity; backend may not enforce
    uint64_t           seed;         // sampling RNG seed (used when seed_set)
    bool               seed_set;
    // Penalties (lucebox SamplerCfg supports these; OpenAI clients send them).
    float              rep_pen;      // HF multiplicative repetition penalty (1.0 = off)
    int                rep_window;   // lookback for rep_pen (0 = backend default)
    float              freq_pen;     // OpenAI additive frequency penalty (0 = off)
    float              pres_pen;     // OpenAI additive presence penalty (0 = off)
    // DRY ("Don't Repeat Yourself"): penalise a token by the LENGTH of the
    // verbatim span it would extend, rather than by how often it has appeared.
    // rep_pen/freq_pen score a token by its own history, so damping a loop with
    // them also suppresses every legitimate reuse; DRY only bites once the model
    // is genuinely replaying a span. Names and defaults follow llama.cpp.
    // dry_multiplier 0 disables it, which is the default: this steers sampling,
    // so it ships off and is enabled explicitly (see CLAUDE.md on risky
    // parity features).
    // Sentinels differ per field because each one's legal client range differs:
    // dry_allowed_length 0 is a meaningful value, so it cannot double as unset.
    float              dry_multiplier;      // 0 = off
    float              dry_base;            // <=1 => backend default (1.75)
    int                dry_allowed_length;  // <0 => backend default (2)
    int                dry_window;          // 0 => backend default (1024),
                                            // <0 = whole history
    // Level-2 thinking force-close (mirrors lucebox BudgetHook / ds4's
    // hard_limit reply budget). When budget_close_ids is set, the backend
    // injects that token sequence once (max_tokens - committed) drops to
    // reply_budget, giving the model the reserved budget to emit a visible
    // answer after </think>. NOTE: enabling this routes generation through the
    // AR path (spec decode is skipped) — the same trade-off lucebox makes for
    // thinking requests. Leave budget_close_ids NULL to disable.
    const int32_t     *budget_close_ids;  // NULL = force-close disabled
    int                n_budget_close;     // length of budget_close_ids
    // Bare </think> token sequence used to disarm the hook after a natural
    // close. The server enables force-close only when this sequence is known.
    const int32_t     *budget_natural_close_ids;
    int                n_budget_natural_close;
    int                reply_budget;       // hard_limit_remaining (reserved reply)
    // KV reuse. restore_slot >= 0: restore that slot's KV (holding a prefix of
    // `prompt`) and prefill only the new suffix. snap_slot >= 0: snapshot the
    // KV at snap_pos into that slot for future reuse.
    int                restore_slot;  // -1 = fresh prefill
    int                snap_slot;     // -1 = no snapshot
    int                snap_pos;      // prefix length to snapshot
    ember_token_cb     on_token;
    ember_keepalive_cb on_prefill;
    void              *ud;
    // B6: structural-token greedy sampling. When set, the backend consults this
    // before sampling each token; returning true forces greedy argmax for that
    // token (used to keep tool-call DSML scaffolding well-formed at temp > 0).
    // Only consulted when the sampler processes logits; NULL disables. Called on
    // the same (worker) thread as on_token → no locking needed.
    bool             (*force_greedy)(void *ud);
    void              *fg_ud;
    // Constrained tool-call decoding. EBNF covering the tool_calls block, built
    // from this request's own tool schemas (ember_tool_grammar_build). The
    // backend compiles it once and masks disallowed tokens for the duration of
    // a DSML tool-call block only; prose and reasoning stay unconstrained.
    // NULL = unconstrained, and the stub backend ignores it entirely.
    const char        *tool_grammar;
} ember_gen_request;

typedef struct {
    int    n_generated;
    int    prefill_tokens;  // tokens actually evaluated after prefix restore
    double prefill_s;
    double decode_s;
    double accept_rate;   // DSpark; 0 if N/A
    bool   ok;
    bool   cancelled;
    char   finish_reason[16];  // "stop" | "length"
    // ── surfaced from the backend GenerateResult (were previously dropped) ──
    char   error_code[32];     // backend error code when !ok (else "")
    char   error_detail[192];  // backend error detail when !ok (else "")
    bool   budget_forced_close;    // Level-2 hook injected </think> (vs self-close)
    bool   degenerate_decode_close;// n-gram repetition loop → answer unreliable
    char   termination_reason[32]; // repetition/reasoning-cycle/prompt-echo, or ""
    bool   empty_visible_output;   // decode produced only suppressed tokens
    bool   spec_decode_ran;        // DSpark actually ran this generation
    char   prefill_mode[16];       // none | exact | dense | sparse | hybrid
    char   prefill_reason[32];     // configured | forced_exact | dspark_capture | ...
    // True iff THIS generation persisted a reusable KV snapshot into
    // req->snap_slot. The server commits its logical prefix entry only when set,
    // so a failed backend save can't poison the cache (#2).
    bool   snapshot_saved;
} ember_gen_result;

ember_gen_result ember_backend_generate(ember_backend *b,
                                        const ember_gen_request *req);

// Release the resident generation associated with the calling thread after the
// server has finished snapshot/tool-memory post-processing. No-op on the
// legacy single-session path.
void ember_backend_generation_release(ember_backend *b);
bool ember_backend_batch_enabled(const ember_backend *b);

typedef struct {
    bool     enabled;
    int      capacity;
    int      pending;
    int      resident;
    int      prefill_ready;
    int      decode_ready;
    int      in_flight;
    int      terminal;
    uint64_t admissions;
    uint64_t releases;
    uint64_t submissions;
    uint64_t decode_batches;
    uint64_t decode_rows;
    uint64_t prefill_tokens;
    uint64_t mixed_submissions;
    uint64_t coalesce_waits;
    uint64_t backend_failures;
    uint64_t backend_exceptions;
    int      max_decode_batch;
} ember_batch_stats;

bool ember_backend_batch_stats_get(const ember_backend *b,
                                   ember_batch_stats *stats);

// ── engine differential validation ──
// Runs a greedy autoregressive baseline, restores the exact prefill snapshot,
// then runs the normal speculative path from that same state.  When disk KV is
// enabled and the prompt is large enough to persist, it also round-trips the
// snapshot through disk and repeats the AR decode. With resident batching it
// admits two spec-eligible rows; DFLASH_DSPARK_XDNA_REQUIRED additionally
// requires both rows to report that speculation actually ran. Intended for an
// explicit startup validation command, not concurrent serving.
typedef struct {
    bool   ok;
    bool   snapshot_ok;
    bool   spec_checked;
    bool   spec_exact;
    bool   disk_checked;
    bool   disk_exact;
    bool   batch_checked;
    bool   batch_exact;
    bool   batch_spec_required;
    int    baseline_tokens;
    int    spec_tokens;
    int    disk_tokens;
    int    batch_rows;
    int    batch_tokens;
    int    batch_spec_rows;
    int    mismatch_index;      // -1 when all compared token streams match
    int32_t expected_token;
    int32_t actual_token;
    double spec_accept_rate;
    double batch_spec_accept_rate;
    char   detail[192];
} ember_validation_report;

bool ember_backend_validate(ember_backend *b, const int32_t *prompt,
                            int n_prompt, int n_gen,
                            ember_validation_report *report);

// ── hipBLAS strided-batched GEMM batchCount sweep ──
// Answers the one question the batched-decode design hangs on: can N decode
// rows share a single strided-batched GEMM and still produce BIT-IDENTICAL
// results to N separate calls?
//
// It has to be measured, not looked up. Dwarfstar found empirically on CUDA
// that "larger batchCount values let cuBLAS select a different reduction and
// change logits", and settled on 4 (ds4_cuda.cu:13205-13208). That constant is
// a property of cuBLAS on that hardware; nothing in the rocBLAS/hipBLAS docs or
// the ROCm 7.14 notes states an equivalent guarantee, so the gfx1151 number is
// unknown until run.
//
// The sweep also has to survive, not merely agree. This engine has already been
// bitten by batch-count-dependent ALGORITHM SELECTION: at N>1 the cuBLAS M=1
// heuristic picked a split-K kernel absent from the shipped library and
// poisoned the stream (moe_hybrid_ffn_eval.cpp:105-111). So a fault at some
// batchCount is a real possible outcome and is reported separately from a
// numeric divergence.
//
// GPU-only. The stub reports ok=false so the GPU-free gauntlet is unaffected.
typedef struct {
    bool ok;                // the sweep ran at all
    int  limit;             // highest batchCount attempted
    int  max_exact;         // largest count exact for every tested shape (>=1)
    int  first_divergent;   // smallest batchCount that differed, else 0
    int  first_fault;       // smallest batchCount that errored, else 0
    int  shapes_tested;
    double worst_rel;       // largest relative deviation observed
    char detail[256];
} ember_gemm_batch_report;

bool ember_backend_validate_gemm_batch(ember_backend *b, int limit,
                                       ember_gemm_batch_report *report);

// ── disk KV cache (cross-restart persistence; no-op if kv_cache_dir unset) ──
// These reuse the model-coupled snapshot serialization (ggml tensor I/O +
// layout fingerprint) from the underlying backend.

// Save reason, persisted in the disk cache (v3+) to steer eviction weighting
// only — never correctness. Numeric values match the engine DiskCacheReason.
// COLD (a cold system-prefix anchor) is eviction-protected; CONTINUED is a
// mid-generation waypoint a later superset store can demote; NORMAL is a
// routine turn-boundary / tool-call waypoint.
typedef enum {
    EMBER_KV_SAVE_NORMAL    = 0,
    EMBER_KV_SAVE_COLD      = 1,
    EMBER_KV_SAVE_CONTINUED = 2,
} ember_kv_save_reason;

bool ember_backend_disk_enabled(const ember_backend *b);
// Longest prefix of `prompt` present on disk, or 0.
int  ember_backend_disk_prefix(ember_backend *b, const int32_t *prompt, int n);
// Load the disk snapshot for prompt[0:len] into `slot`. Returns true on hit.
bool ember_backend_disk_lookup(ember_backend *b, const int32_t *prompt, int len, int slot);
// Persist slot's snapshot to disk keyed by prompt[0:cut]. `reason` is an
// ember_kv_save_reason (persisted; drives eviction weighting).
bool ember_backend_disk_save(ember_backend *b, int slot, const int32_t *prompt,
                             int cut, int reason);
// Model/file identity used to scope auxiliary disk state such as exact tool
// replay mappings. Returns false when disk caching is unavailable.
bool ember_backend_cache_identity(const ember_backend *b, uint8_t out[16]);

// ── introspection ──
// B3 Layer 2: snapshot the CURRENT KV (post-generation cur_pos) into `slot`,
// so the caller can continue from after a just-generated tool call instead of
// re-prefilling it next turn. Returns true iff a reusable snapshot was saved.
bool        ember_backend_snapshot_now(ember_backend *b, int slot);

// Tokens actually covered by slot's parked snapshot, or -1 if it holds none.
// This is NOT the same as the number of tokens the caller has emitted: decode
// writes a token's KV row at the START of the following step, so a snapshot
// taken after generation is one row behind the emitted stream (and speculative
// decode can leave it ahead). Cache keys must be built from this, never from
// the emitted count — a key longer than its KV describes a prefix the snapshot
// cannot honor.
int         ember_backend_snapshot_pos(const ember_backend *b, int slot);

// Release cached compute graphs and scratch arenas WITHOUT unloading the model.
// The DeepSeek4 caches are thread_local on the generation worker and ratchet to
// the largest context ever served (measured: GTT 12GB -> 28.6GB, never falling
// back while idle), so until now only a restart reclaimed them. MUST be called
// from the generation worker thread. Graphs rebuild lazily, so the first request
// after a reclaim pays that cost.
void        ember_backend_release_idle_graphs(ember_backend *b);

int         ember_backend_n_ctx(const ember_backend *b);
const char *ember_backend_model_name(const ember_backend *b);
int32_t     ember_backend_eos_id(const ember_backend *b);

#ifdef __cplusplus
}
#endif

#endif  // EMBER_BACKEND_H
