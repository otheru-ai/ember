// Deterministic, GPU-free backend implementing the ember_backend ABI. It does
// NOT run a model: encode is a reversible byte-identity tokenization (one token
// per byte, id = byte value + a reserved base) and generate replays a canned
// reply as tokens. This lets the entire server pipeline — template → encode →
// generate → detokenize → SSE — run and be tested without the HIP container.
// The real forward pass (backend_dflash.cc) replaces this behind the same ABI.
#include "ember_backend.h"

#include <limits.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TOK_BASE 256   // ids 0..255 map to raw bytes; >=256 reserved/specials
#define STUB_EOS 65535

struct ember_backend {
    char *model_name;
    int   n_ctx;
    int   batch_sessions;
    atomic_int active;
    atomic_int max_active;
    atomic_ullong admissions;
    // token_text scratch: one-byte strings, indexed by byte value
    char  byte_str[256][2];
};

ember_backend *ember_backend_load(const ember_backend_config *cfg, char **err) {
    if (err) *err = NULL;
    ember_backend *b = (ember_backend *)calloc(1, sizeof(ember_backend));
    if (!b) {
        if (err) *err = strdup("out of memory");
        return NULL;
    }
    b->model_name = strdup(cfg && cfg->model_name ? cfg->model_name
                                                  : "deepseek-v4-flash");
    if (!b->model_name) {
        free(b);
        if (err) *err = strdup("out of memory");
        return NULL;
    }
    b->n_ctx = cfg && cfg->max_ctx > 0 ? cfg->max_ctx : 65536;
    b->batch_sessions =
        cfg && cfg->batch_sessions > 1 ? cfg->batch_sessions : 1;
    for (int i = 0; i < 256; i++) { b->byte_str[i][0] = (char)i; b->byte_str[i][1] = 0; }
    return b;
}

void ember_backend_free(ember_backend *b) {
    if (!b) return;
    free(b->model_name);
    free(b);
}

int ember_backend_encode(ember_backend *b, const char *text, int32_t **ids_out) {
    (void)b;
    if (!ids_out) return -1;
    *ids_out = NULL;
    size_t n = text ? strlen(text) : 0;
    if (n > (size_t)INT_MAX ||
        n > SIZE_MAX / sizeof(int32_t)) return -1;
    int32_t *ids = (int32_t *)malloc((n ? n : 1) * sizeof(int32_t));
    if (!ids) return -1;
    for (size_t i = 0; i < n; i++) ids[i] = (unsigned char)text[i];
    *ids_out = ids;
    return (int)n;
}

const char *ember_backend_token_text(ember_backend *b, int32_t id) {
    if (id >= 0 && id < 256) return b->byte_str[id];
    return "";
}

bool ember_backend_vision_encode(ember_backend *b,
                                 const uint8_t *encoded, size_t encoded_size,
                                 ember_vision_image *out,
                                 char *error, size_t error_cap) {
    (void)b; (void)encoded; (void)encoded_size;
    if (out) memset(out, 0, sizeof(*out));
    if (error && error_cap)
        snprintf(error, error_cap, "%s", "vision input is not supported by the stub backend");
    return false;
}

void ember_backend_vision_image_free(ember_vision_image *image) {
    if (!image) return;
    free(image->embeddings);
    memset(image, 0, sizeof(*image));
}

static bool prompt_contains(const ember_gen_request *req, const char *needle) {
    size_t n = strlen(needle);
    if (!req || !req->prompt || n == 0 || req->n_prompt < (int)n)
        return false;
    for (int i = 0; i <= req->n_prompt - (int)n; ++i) {
        bool match = true;
        for (size_t j = 0; j < n; ++j) {
            if (req->prompt[i + (int)j] != (unsigned char)needle[j]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

static bool prompt_ends_with(const ember_gen_request *req, const char *suffix) {
    size_t n = suffix ? strlen(suffix) : 0;
    if (!req || !req->prompt || n == 0 || req->n_prompt < (int)n)
        return false;
    int start = req->n_prompt - (int)n;
    for (size_t i = 0; i < n; ++i)
        if (req->prompt[start + (int)i] != (unsigned char)suffix[i])
            return false;
    return true;
}

ember_gen_result ember_backend_generate(ember_backend *b,
                                        const ember_gen_request *req) {
    ember_gen_result r = {0};
    strcpy(r.finish_reason, "stop");
    strcpy(r.prefill_mode, "stub");
    strcpy(r.prefill_reason, "configured");
    if (!b || !req || !req->prompt || req->n_prompt < 0) {
        strcpy(r.error_code, "invalid_request");
        strcpy(r.error_detail, "invalid backend generation request");
        return r;
    }
    int active = atomic_fetch_add(&b->active, 1) + 1;
    r.prefill_tokens = req->n_prompt;
    atomic_fetch_add(&b->admissions, 1);
    int seen = atomic_load(&b->max_active);
    while (active > seen &&
           !atomic_compare_exchange_weak(&b->max_active, &seen, active)) {}

    // Simulate a short prefill with a couple of keepalive ticks.
    if (req->on_prefill) {
        for (int k = 0; k < 2; k++) {
            if (!req->on_prefill(req->ud)) {
                r.cancelled = true;
                strcpy(r.finish_reason, "stop");
                atomic_fetch_sub(&b->active, 1);
                return r;
            }
        }
    }

    // Tests may override the first and recovery turns to exercise protocol
    // paths without a GPU. These variables affect only the stub binary.
    const char *reply = getenv("EMBER_STUB_REPLY");
    if (prompt_ends_with(req, "</think>\n\n")) {
        const char *continued = getenv("EMBER_STUB_THINK_TOOL_REPLY");
        if (continued) reply = continued;
    }
    // Lets a test prove the auto-answer INSTRUCTION actually reached the
    // prompt, not merely that the tools were removed. Without this the primary
    // fix is unverifiable and the v1 failure repeats: the model never learns
    // why its tools vanished and improvises markup as text.
    if (prompt_contains(req, "[Automatic recovery]")) {
        const char *aa = getenv("EMBER_STUB_AUTOANSWER_REPLY");
        if (aa && aa[0]) reply = aa;
    }
    if (prompt_contains(req, "Tool error: invalid DSML tool call")) {
        const char *recovery_error = getenv("EMBER_STUB_RECOVERY_ERROR");
        if (recovery_error && recovery_error[0]) {
            snprintf(r.error_code, sizeof(r.error_code), "%s", recovery_error);
            snprintf(r.error_detail, sizeof(r.error_detail),
                     "stub recovery backend failure");
            atomic_fetch_sub(&b->active, 1);
            return r;
        }
        const char *recovery = getenv("EMBER_STUB_RECOVERY_REPLY");
        if (recovery) reply = recovery;
    }
    char fallback[256];
    if (getenv("EMBER_STUB_ECHO_SAMPLER")) {
        // The stub implements no DRY math, but it is the only GPU-free proof
        // that server-side sampler resolution actually reaches the ABI — the
        // request fields, the EMBER_DRY_MULTIPLIER default, and the per-field
        // "unset" sentinels. test_sampler covers the math below this seam and
        // never exercises main.c's resolution, so without these the plumbing
        // is untested.
        snprintf(fallback, sizeof(fallback),
                 "temp=%.3g top_p=%.3g top_k=%d min_p=%.3g rep_pen=%.3g "
                 "freq_pen=%.3g pres_pen=%.3g greedy=%d dry_mult=%.3g "
                 "dry_base=%.3g dry_allow=%d dry_win=%d",
                 req->temperature, req->top_p, req->top_k, req->min_p,
                 req->rep_pen, req->freq_pen, req->pres_pen,
                 req->greedy ? 1 : 0,
                 req->dry_multiplier, req->dry_base,
                 req->dry_allowed_length, req->dry_window);
        reply = fallback;
    }
    if (!reply) {
        snprintf(fallback, sizeof(fallback),
                 "Ember backend (stub): prompt was %d tokens; real DeepSeek "
                 "forward pass lands via the dflash bridge.", req->n_prompt);
        reply = fallback;
    }
    // Match the real bridge exactly: zero is a valid literal generation budget
    // (used by Anthropic cache-prewarm requests), not an omitted/unlimited cap.
    int budget = req->max_tokens >= 0 ? req->max_tokens : 1 << 20;
    long delay_us = 0;
    const char *delay_env = getenv("EMBER_STUB_TOKEN_DELAY_US");
    if (delay_env && delay_env[0]) {
        char *end = NULL;
        long parsed = strtol(delay_env, &end, 10);
        if (end != delay_env && *end == '\0' && parsed > 0 &&
            parsed <= 1000000)
            delay_us = parsed;
    }
    int n = 0;
    for (size_t i = 0; reply[i] && n < budget; i++) {
        if (delay_us > 0) {
            struct timespec delay = {
                .tv_sec = delay_us / 1000000,
                .tv_nsec = (delay_us % 1000000) * 1000,
            };
            while (nanosleep(&delay, &delay) != 0) {}
        }
        // The real backend records the sampled token before invoking the
        // callback. A callback cancellation therefore still counts the token
        // it just received; keep stub accounting and captured gen_ids aligned.
        n++;
        if (req->on_token && !req->on_token((unsigned char)reply[i], req->ud)) {
            r.cancelled = true;
            break;
        }
    }
    if (n >= budget) strcpy(r.finish_reason, "length");
    r.n_generated = n;
    r.ok = true;
    // GPU-free integration tests can exercise the server's typed watchdog
    // boundary without manufacturing a 512-token exact prompt echo.
    const char *termination = getenv("EMBER_STUB_TERMINATION_REASON");
    if (termination && termination[0]) {
        r.degenerate_decode_close = true;
        snprintf(r.termination_reason, sizeof(r.termination_reason), "%s",
                 termination);
    }
    r.prefill_s = 0.0;
    r.decode_s = 0.0;
    atomic_fetch_sub(&b->active, 1);
    return r;
}

bool ember_backend_validate_gemm_batch(ember_backend *b, int limit,
                                       ember_gemm_batch_report *report) {
    (void)b;
    (void)limit;
    // Deliberately not simulated. The sweep exists to measure a hipBLAS
    // property of real hardware; a stub answer would be a fabricated number,
    // and this repo has already paid for one of those.
    if (!report) return false;
    memset(report, 0, sizeof(*report));
    report->ok = false;
    snprintf(report->detail, sizeof(report->detail),
             "stub backend: GEMM batch sweep needs the real HIP backend");
    return false;
}

bool ember_backend_validate(ember_backend *b, const int32_t *prompt,
                            int n_prompt, int n_gen,
                            ember_validation_report *report) {
    (void)b;
    (void)prompt;
    (void)n_prompt;
    if (!report || n_gen < 1) return false;
    memset(report, 0, sizeof(*report));
    report->ok = true;
    report->snapshot_ok = true;
    report->prefill_checked = false;
    report->prefill_exact = true;
    report->spec_checked = false;
    report->spec_exact = true;
    report->disk_checked = false;
    report->disk_exact = true;
    report->batch_checked = b && b->batch_sessions > 1;
    report->batch_exact = true;
    report->baseline_tokens = n_gen;
    report->batch_rows = report->batch_checked ? 2 : 0;
    report->batch_tokens =
        report->batch_checked ? report->batch_rows * n_gen : 0;
    report->mismatch_index = -1;
    report->expected_token = -1;
    report->actual_token = -1;
    snprintf(report->detail, sizeof(report->detail),
             "stub backend: structural validation only");
    return true;
}

bool        ember_backend_snapshot_now(ember_backend *b, int slot) { (void)b; (void)slot; return false; }
int         ember_backend_snapshot_pos(const ember_backend *b, int slot) { (void)b; (void)slot; return -1; }
void        ember_backend_generation_release(ember_backend *b) { (void)b; }
bool        ember_backend_batch_enabled(const ember_backend *b) {
    return b && b->batch_sessions > 1;
}
bool ember_backend_batch_stats_get(const ember_backend *b,
                                   ember_batch_stats *stats) {
    if (!b || !stats) return false;
    memset(stats, 0, sizeof(*stats));
    stats->enabled = b->batch_sessions > 1;
    stats->capacity = b->batch_sessions;
    stats->resident = atomic_load(&b->active);
    stats->admissions = atomic_load(&b->admissions);
    stats->max_decode_batch = atomic_load(&b->max_active);
    return true;
}
void        ember_backend_release_idle_graphs(ember_backend *b) { (void)b; }
int         ember_backend_n_ctx(const ember_backend *b) { return b->n_ctx; }
const char *ember_backend_model_name(const ember_backend *b) { return b->model_name; }
int32_t     ember_backend_eos_id(const ember_backend *b) { (void)b; return STUB_EOS; }

// Disk KV cache — no-op in the stub (no real KV to persist).
bool ember_backend_disk_enabled(const ember_backend *b) { (void)b; return false; }
int  ember_backend_disk_prefix(ember_backend *b, const int32_t *p, int n) { (void)b;(void)p;(void)n; return 0; }
bool ember_backend_disk_lookup(ember_backend *b, const int32_t *p, int len, int slot) { (void)b;(void)p;(void)len;(void)slot; return false; }
bool ember_backend_disk_save(ember_backend *b, int slot, const int32_t *p, int cut, int reason) { (void)b;(void)slot;(void)p;(void)cut;(void)reason; return false; }
bool ember_backend_cache_identity(const ember_backend *b, uint8_t out[16]) {
    (void)b;
    (void)out;
    return false;
}
