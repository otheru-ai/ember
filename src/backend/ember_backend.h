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

typedef struct {
    const char *model_path;
    int         max_ctx;
    const char *model_name;   // advertised id for /v1/models
    int         expert_top_k; // 0 = model default
    const char *kv_cache_dir; // NULL = disk KV cache disabled
    long        kv_cache_mb;  // disk budget (0 = default)
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
    // Level-2 thinking force-close (mirrors lucebox BudgetHook / ds4's
    // hard_limit reply budget). When budget_close_ids is set, the backend
    // injects that token sequence once (max_tokens - committed) drops to
    // reply_budget, giving the model the reserved budget to emit a visible
    // answer after </think>. NOTE: enabling this routes generation through the
    // AR path (spec decode is skipped) — the same trade-off lucebox makes for
    // thinking requests. Leave budget_close_ids NULL to disable.
    const int32_t     *budget_close_ids;  // NULL = force-close disabled
    int                n_budget_close;     // length of budget_close_ids
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
} ember_gen_request;

typedef struct {
    int    n_generated;
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
    bool   empty_visible_output;   // decode produced only suppressed tokens
    bool   spec_decode_ran;        // DSpark actually ran this generation
    // True iff THIS generation persisted a reusable KV snapshot into
    // req->snap_slot. The server commits its logical prefix entry only when set,
    // so a failed backend save can't poison the cache (#2).
    bool   snapshot_saved;
} ember_gen_result;

ember_gen_result ember_backend_generate(ember_backend *b,
                                        const ember_gen_request *req);

// ── disk KV cache (cross-restart persistence; no-op if kv_cache_dir unset) ──
// These reuse the model-coupled snapshot serialization (ggml tensor I/O +
// layout fingerprint) from the underlying backend.
bool ember_backend_disk_enabled(const ember_backend *b);
// Longest prefix of `prompt` present on disk, or 0.
int  ember_backend_disk_prefix(ember_backend *b, const int32_t *prompt, int n);
// Load the disk snapshot for prompt[0:len] into `slot`. Returns true on hit.
bool ember_backend_disk_lookup(ember_backend *b, const int32_t *prompt, int len, int slot);
// Persist slot's snapshot to disk keyed by prompt[0:cut].
bool ember_backend_disk_save(ember_backend *b, int slot, const int32_t *prompt, int cut);

// ── introspection ──
// B3 Layer 2: snapshot the CURRENT KV (post-generation cur_pos) into `slot`,
// so the caller can continue from after a just-generated tool call instead of
// re-prefilling it next turn. Returns true iff a reusable snapshot was saved.
bool        ember_backend_snapshot_now(ember_backend *b, int slot);

int         ember_backend_n_ctx(const ember_backend *b);
const char *ember_backend_model_name(const ember_backend *b);
int32_t     ember_backend_eos_id(const ember_backend *b);

#ifdef __cplusplus
}
#endif

#endif  // EMBER_BACKEND_H
