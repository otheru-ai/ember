// Deterministic, GPU-free backend implementing the ember_backend ABI. It does
// NOT run a model: encode is a reversible byte-identity tokenization (one token
// per byte, id = byte value + a reserved base) and generate replays a canned
// reply as tokens. This lets the entire server pipeline — template → encode →
// generate → detokenize → SSE — run and be tested without the HIP container.
// The real forward pass (backend_dflash.cc) replaces this behind the same ABI.
#include "ember_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOK_BASE 256   // ids 0..255 map to raw bytes; >=256 reserved/specials
#define STUB_EOS 65535

struct ember_backend {
    char *model_name;
    int   n_ctx;
    // token_text scratch: one-byte strings, indexed by byte value
    char  byte_str[256][2];
};

ember_backend *ember_backend_load(const ember_backend_config *cfg, char **err) {
    (void)err;
    ember_backend *b = (ember_backend *)calloc(1, sizeof(ember_backend));
    b->model_name = strdup(cfg && cfg->model_name ? cfg->model_name
                                                  : "deepseek-v4-flash");
    b->n_ctx = cfg && cfg->max_ctx > 0 ? cfg->max_ctx : 65536;
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
    size_t n = text ? strlen(text) : 0;
    int32_t *ids = (int32_t *)malloc((n ? n : 1) * sizeof(int32_t));
    for (size_t i = 0; i < n; i++) ids[i] = (unsigned char)text[i];
    *ids_out = ids;
    return (int)n;
}

const char *ember_backend_token_text(ember_backend *b, int32_t id) {
    if (id >= 0 && id < 256) return b->byte_str[id];
    return "";
}

ember_gen_result ember_backend_generate(ember_backend *b,
                                        const ember_gen_request *req) {
    ember_gen_result r = {0};
    strcpy(r.finish_reason, "stop");

    // Simulate a short prefill with a couple of keepalive ticks.
    if (req->on_prefill) {
        for (int k = 0; k < 2; k++) {
            if (!req->on_prefill(req->ud)) { r.cancelled = true; strcpy(r.finish_reason, "stop"); return r; }
        }
    }

    // Canned reply, streamed as byte-tokens.
    char reply[256];
    snprintf(reply, sizeof(reply),
             "Ember backend (stub): prompt was %d tokens; real DeepSeek forward "
             "pass lands via the dflash bridge.", req->n_prompt);
    int budget = req->max_tokens > 0 ? req->max_tokens : 1 << 20;
    int n = 0;
    for (size_t i = 0; reply[i] && n < budget; i++, n++) {
        if (req->on_token && !req->on_token((unsigned char)reply[i], req->ud)) {
            r.cancelled = true;
            break;
        }
    }
    if (n >= budget) strcpy(r.finish_reason, "length");
    r.n_generated = n;
    r.ok = true;
    r.prefill_s = 0.0;
    r.decode_s = 0.0;
    (void)b;
    return r;
}

bool        ember_backend_snapshot_now(ember_backend *b, int slot) { (void)b; (void)slot; return false; }
int         ember_backend_n_ctx(const ember_backend *b) { return b->n_ctx; }
const char *ember_backend_model_name(const ember_backend *b) { return b->model_name; }
int32_t     ember_backend_eos_id(const ember_backend *b) { (void)b; return STUB_EOS; }

// Disk KV cache — no-op in the stub (no real KV to persist).
bool ember_backend_disk_enabled(const ember_backend *b) { (void)b; return false; }
int  ember_backend_disk_prefix(ember_backend *b, const int32_t *p, int n) { (void)b;(void)p;(void)n; return 0; }
bool ember_backend_disk_lookup(ember_backend *b, const int32_t *p, int len, int slot) { (void)b;(void)p;(void)len;(void)slot; return false; }
bool ember_backend_disk_save(ember_backend *b, int slot, const int32_t *p, int cut) { (void)b;(void)slot;(void)p;(void)cut; return false; }
