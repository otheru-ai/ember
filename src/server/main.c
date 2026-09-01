// ember-server: HTTP entry point. Drives the full pipeline through the backend
// ABI — render prompt (chat_template) → encode → ember_backend_generate →
// per-token detokenize → SSE stream. Only the backend forward pass is stubbed
// (backend_stub.c); swapping in backend_dflash.cc changes nothing here.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <fcntl.h>

#include "../backend/ember_backend.h"
#include "../common/buf.h"
#include "../common/json.h"
#include "../common/json_util.h"  // ember_json_escape (error-path JSON)
#include "../model/chat_template.h"
#include "../model/model_card.h"
#include "../model/kv_cache.h"
#include "../model/tool_memory.h"
#include "../model/dsml_decode.h"
#include "chat_api.h"
#include "http.h"
#include "sse.h"

#define PIPE "\xef\xbd\x9c"  // U+FF5C
#define EMBER_KV_DISK_SLOT (EMBER_KV_MAX_SLOTS - 1)  // staging slot for disk restores

// ── persistent generation worker ─────────────────────────────────────────
// All backend generation runs on ONE long-lived thread so lucebox's
// thread_local graph caches (the layer-major prefill cache, its shared gallocr,
// the DSpark draft cache) build once and stay warm. Previously every HTTP
// connection ran generation on its own short-lived detached thread (http.c
// conn_thread): each request got a fresh thread_local cache, rebuilt the
// ~918MB high-water prefill arena from scratch, and then ORPHANED it (leaked)
// when the connection thread exited — ~918MB of unreachable growth per full
// prefill. Connection threads now hand the job to this worker and block until
// it completes; the single worker serializes generation on its own (gen_lock
// is retained only as belt-and-braces, now always uncontended).
struct ember_server;  // defined below

typedef struct gen_job {
    struct ember_server      *srv;
    ember_chat_request       *req;   // borrowed; lives on the waiter's stack
                                     // (mutable: run_chat attaches B3 replay bytes)
    int                       fd;
    bool                      done;
    pthread_mutex_t           lock;
    pthread_cond_t            cond;
    struct gen_job           *next;
} gen_job;

// #3: cap the pending-job FIFO. With generation serialized, extra chat requests
// pile up as blocked connection threads; a bounded queue sheds load (503) once
// the backlog is this deep instead of growing unbounded.
#define EMBER_MAX_QUEUE_DEPTH 8

typedef struct {
    gen_job         *head, *tail;  // FIFO of pending jobs (guarded by lock)
    int              queued;       // #3: current FIFO depth (guarded by lock)
    pthread_mutex_t  lock;
    pthread_cond_t   cond;
    pthread_t        thread;
    bool             stop;         // set by gen_worker_stop to drain + exit
    bool             running;      // true between a successful start and stop
    ember_backend   *be;           // #5: freed on THIS thread at teardown
} gen_worker;

typedef struct ember_server {
    ember_backend    *be;
    ember_model_card  card;
    ember_kv_cache    kv;         // prefix KV cache (accessed only on the worker)
    ember_tool_memory tool_mem;   // B3: id -> exact sampled tool-call bytes (worker-only)
    pthread_mutex_t   gen_lock;   // single-slot: one generation at a time
    atomic_int        busy;       // reported by /status
    atomic_long       served;     // completed generations
    int32_t          *close_ids;  // Level-2 force-close token sequence (owned)
    int               n_close_ids;
    double            default_temp;  // temperature when the request omits it
    gen_worker        worker;     // persistent generation thread (see above)
} ember_server;

// Encode a chat marker to its single special-token id, or -1 if not one token.
static int32_t marker_id(ember_backend *be, const char *s) {
    int32_t *ids = NULL;
    int n = ember_backend_encode(be, s, &ids);
    int32_t r = (n == 1) ? ids[0] : -1;
    free(ids);
    return r;
}

static long now_unix(void) { return (long)time(NULL); }

// JSON-escape `s` into `b` (no surrounding quotes).
static void json_escape_str(ember_buf *b, const char *s) {
    for (const char *p = s; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == '"' || c == '\\') { ember_buf_putc(b, '\\'); ember_buf_putc(b, (char)c); }
        else if (c == '\n') ember_buf_puts(b, "\\n");
        else if (c >= 0x20) ember_buf_putc(b, (char)c);
        else ember_buf_printf(b, "\\u%04x", c);
    }
}

static void respond(int fd, int code, const char *ctype, const char *body) {
    ember_buf b = {0};
    ember_buf_printf(&b,
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Connection: close\r\n\r\n%s",
        code, code == 200 ? "OK" : "Error", ctype, strlen(body), body);
    ember_send_all(fd, b.ptr, b.len);
    ember_buf_free(&b);
}

// ── generation context threaded through the backend callbacks ──
typedef struct {
    ember_backend    *be;
    ember_sse_stream *st;        // NULL when collect_only
    ember_buf         acc;       // raw generated text (buffer-and-resplit source)
    ember_buf         scratch;   // per-call SSE output
    int               fd;
    bool              collect_only;  // accumulate without streaming (non-stream path)
    bool              disconnected;
    bool              has_tools;     // stop decode at the tool-calls end marker
    bool              started_thinking;  // B#1: prompt ended inside an open <think>
    time_t            last_ka;       // last prefill keepalive (throttle to ~4s)
    char            **stops;         // client stop strings (borrowed)
    int               n_stops;
    bool              hit_stop;      // a stop sequence was reached
    int32_t          *gen_ids;       // B3 L2: committed generated token ids
    int               n_gen_ids, gen_cap;
    bool              dsml_active;   // B6: track DSML decode state for greedy struct
    ember_dsml_tracker dsml;         // B6: structural-token greedy sampling state
} gen_ctx;

// B#1: stateful, thinking-aware tool-call stop helpers (mirrors ds4
// find_last_substr + thinking_state gating). A closer only halts generation
// when a matching opener precedes it in the VISIBLE region (after the last
// </think>); markers inside an unclosed <think> are not executable calls.
static const char *last_substr(const char *s, const char *needle) {
    const char *last = NULL, *p = s;
    while ((p = strstr(p, needle)) != NULL) { last = p; p++; }
    return last;
}
static bool inside_unclosed_think(const char *s, bool started_thinking) {
    const char *lo = last_substr(s, "<think>");
    const char *lc = last_substr(s, "</think>");
    if (lo && (!lc || lo > lc)) return true;   // a <think> opened after any close
    if (!lc && started_thinking) return true;  // prompt opened <think>, none closed yet
    return false;
}
static const char *visible_after_think(const char *s) {
    const char *lc = last_substr(s, "</think>");
    return lc ? lc + 8 : s;  // 8 == strlen("</think>")
}

static bool on_token(int32_t tok, void *ud) {
    gen_ctx *g = (gen_ctx *)ud;
    if (g->disconnected) return false;
    if (tok == ember_backend_eos_id(g->be)) return true;
    ember_buf_puts(&g->acc, ember_backend_token_text(g->be, tok));
    // B3 L2: remember committed token ids so a post-tool-call snapshot can be
    // keyed by the exact [prompt + generated] token sequence.
    if (g->n_gen_ids == g->gen_cap) {
        g->gen_cap = g->gen_cap ? g->gen_cap * 2 : 256;
        g->gen_ids = (int32_t *)realloc(g->gen_ids, (size_t)g->gen_cap * sizeof(int32_t));
    }
    if (g->gen_ids) g->gen_ids[g->n_gen_ids++] = tok;
    // B6: advance the DSML decode-state tracker over the accumulated text so the
    // NEXT token's sampling knows whether it lands on tool-call structure (greedy)
    // or payload (sampled). Runs on the worker thread; force_greedy reads it there.
    if (g->dsml_active && g->acc.ptr)
        ember_dsml_tracker_update(&g->dsml, g->acc.ptr, g->acc.len);
    // B#1/B#7: stateful, thinking-aware tool-call stop (ds4 observe_tool_markers
    // + thinking gating). Halt only at a closer that has a MATCHING opener
    // earlier in the VISIBLE output (after any </think>): an orphan </tool_calls>
    // must not truncate a normal answer, a mismatched closer must not terminate
    // the wrong family, and a marker inside unclosed <think> is not executable.
    // B#7: the native ds_engine format is one block per call and the model may
    // emit several — don't stop at the first native closer; run to EOS so every
    // block is generated (the parser handles multiple ds_engine blocks).
    if (g->has_tools && g->acc.ptr &&
        !inside_unclosed_think(g->acc.ptr, g->started_thinking)) {
        const char *vis   = visible_after_think(g->acc.ptr);
        const char *start = ember_find_tool_start(vis);
        const char *end   = start ? ember_find_tool_end(start) : NULL;
        bool native = start && strncmp(start, "<ds_engine_tool_use>", 20) == 0;
        if (start && end && !native) {
            g->acc.len = (size_t)(end - g->acc.ptr);
            g->acc.ptr[g->acc.len] = '\0';
            if (!g->collect_only) {
                g->scratch.len = 0; if (g->scratch.ptr) g->scratch.ptr[0] = '\0';
                ember_sse_update(g->st, g->acc.ptr, g->acc.len, false, &g->scratch);
                if (g->scratch.len) ember_send_all(g->fd, g->scratch.ptr, g->scratch.len);
            }
            return false;
        }
    }
    // Stop sequences (ds4): halt when a stop appears in the visible content
    // (after any </think>). Truncate there; the final flush emits up to the stop.
    if (g->n_stops > 0 && g->acc.ptr) {
        char *vis = strstr(g->acc.ptr, "</think>");
        vis = vis ? vis + 8 : g->acc.ptr;
        for (int si = 0; si < g->n_stops; si++) {
            char *h = strstr(vis, g->stops[si]);
            if (h) {
                g->acc.len = (size_t)(h - g->acc.ptr);
                g->acc.ptr[g->acc.len] = '\0';
                g->hit_stop = true;
                return false;
            }
        }
    }
    if (g->collect_only) return true;
    g->scratch.len = 0; if (g->scratch.ptr) g->scratch.ptr[0] = '\0';
    ember_sse_update(g->st, g->acc.ptr, g->acc.len, false, &g->scratch);
    if (g->scratch.len &&
        ember_send_all(g->fd, g->scratch.ptr, g->scratch.len) != 0) {
        g->disconnected = true;
        return false;
    }
    return true;
}

static bool on_prefill(void *ud) {
    gen_ctx *g = (gen_ctx *)ud;
    // Throttle keepalives to ~4s regardless of how often the backend calls us
    // (ds4 caps the `: prefill` cadence so a per-chunk backend can't spam).
    time_t now = time(NULL);
    if (g->last_ka != 0 && now - g->last_ka < 4) return true;
    g->last_ka = now;
    ember_buf ka = {0};
    ember_sse_keepalive(&ka);
    int rc = ember_send_all(g->fd, ka.ptr, ka.len);
    ember_buf_free(&ka);
    if (rc != 0) { g->disconnected = true; return false; }
    return true;
}

// B6: consulted by the backend before sampling each token — force greedy argmax
// when the generation frontier is tool-call DSML/JSON structure (see dsml_decode).
static bool gen_force_greedy(void *ud) {
    gen_ctx *g = (gen_ctx *)ud;
    return g->dsml_active && ember_dsml_force_greedy(&g->dsml);
}

// B3: mint a unique random tool-call id (128-bit, "call_" + 32 hex), like ds4
// random_tool_id. `out` must be >= 38 bytes.
static void mint_tool_id(char *out) {
    unsigned char r[16];
    int f = open("/dev/urandom", O_RDONLY);
    if (f < 0 || read(f, r, 16) != 16) {
        unsigned long s = (unsigned long)time(NULL) ^ ((unsigned long)getpid() << 20);
        for (int i = 0; i < 16; i++) { s = s * 6364136223846793005UL + 1442695040888963407UL; r[i] = (unsigned char)(s >> 33); }
    }
    if (f >= 0) close(f);
    static const char hx[] = "0123456789abcdef";
    memcpy(out, "call_", 5);
    for (int i = 0; i < 16; i++) { out[5 + 2*i] = hx[r[i] >> 4]; out[5 + 2*i + 1] = hx[r[i] & 15]; }
    out[37] = '\0';
}

// B3: before rendering, substitute assistant tool turns whose ids we stored, so
// the re-rendered history is TOKEN-identical to what was sampled (prerequisite
// for continuing from the post-tool-call KV). When we also hold the exact sampled
// token ids for the turn, emit a splice sentinel ("\x1f id \x1f") instead of the
// bytes: encode_with_splices() then splices those ids verbatim rather than
// re-tokenizing the DSML text (special DSML markers do not round-trip through
// detokenize->retokenize, which would break the token identity). If the tokens
// are absent (never happens once captured, but be safe), fall back to the raw
// bytes — a correct render, just re-tokenized (no continuation).
// Verify the mapped block names every function the client claims; else fall back
// to the canonical render. Runs on the worker (same thread as capture) → no lock.
static void attach_tool_memory(ember_server *srv, ember_chat_request *req) {
    for (int i = 0; i < req->n_messages; i++) {
        ember_chat_msg *m = &req->messages[i];
        if (strcmp(m->role, "assistant") != 0 || m->calls.len == 0 || m->raw_tool_text)
            continue;
        const char *raw = NULL;
        const char *splice_id = NULL;
        bool ok = true;
        for (int c = 0; c < m->calls.len; c++) {
            const char *id = m->calls.calls[c].id;
            const char *b = id ? ember_tool_memory_get(&srv->tool_mem, id) : NULL;
            if (!b) { ok = false; break; }
            if (!raw) { raw = b; splice_id = id; }
            else if (b != raw && strcmp(b, raw) != 0) { ok = false; break; }
        }
        for (int c = 0; ok && c < m->calls.len; c++) {
            const char *nm = m->calls.calls[c].name;
            char needle[256];
            if (!nm) { ok = false; break; }
            snprintf(needle, sizeof(needle), "invoke name=\"%s\"", nm);
            if (!strstr(raw, needle)) ok = false;  // mapped block must name this fn
        }
        if (!ok || !raw) continue;
        // All calls in one assistant turn share the same sampled turn (same bytes
        // AND the same token-id stream); splice_id addresses that stream.
        int n_tok = 0;
        if (splice_id && ember_tool_memory_get_tokens(&srv->tool_mem, splice_id, &n_tok)
                          && n_tok > 0) {
            char sent[64];
            snprintf(sent, sizeof(sent), "\x1f%s\x1f", splice_id);  // spliced at encode
            m->raw_tool_text = strdup(sent);
        } else {
            m->raw_tool_text = strdup(raw);                          // text fallback
        }
    }
}

// B3 L2: if this generation produced a complete tool call, snapshot the
// post-tool-call KV and commit it to the prefix cache keyed by the exact
// [prompt + generated] tokens. Next turn — which replays those same tokens via
// the tool-memory substitution — matches this snapshot and continues from here,
// prefilling only the new tool-result suffix instead of re-prefilling the call.
static void snapshot_post_toolcall(ember_server *srv, ember_backend *be,
                                   const int32_t *prompt_ids, int n_prompt,
                                   gen_ctx *g) {
    if (!g->has_tools || !g->acc.ptr || g->n_gen_ids <= 0) return;
    const char *st = ember_find_tool_start(g->acc.ptr);
    if (!st || !ember_find_tool_end(st)) return;      // no complete tool block
    int slot = ember_kv_reserve(&srv->kv);
    if (slot < 0) return;
    if (!ember_backend_snapshot_now(be, slot)) return; // backend parks post-gen KV
    int total = n_prompt + g->n_gen_ids;
    int32_t *seq = (int32_t *)malloc((size_t)total * sizeof(int32_t));
    if (!seq) return;
    memcpy(seq, prompt_ids, (size_t)n_prompt * sizeof(int32_t));
    memcpy(seq + n_prompt, g->gen_ids, (size_t)g->n_gen_ids * sizeof(int32_t));
    ember_kv_commit(&srv->kv, slot, seq, total);
    if (ember_backend_disk_enabled(be))
        ember_backend_disk_save(be, slot, seq, total);
    free(seq);
}

// B3: a minted id is exactly "call_" + 32 hex = 37 chars. Only content matching
// that shape between the sentinel markers is treated as a splice reference; a
// stray \x1f in text (never seen in model output) degrades to literal handling.
static bool looks_like_tool_id(const char *s, size_t n) {
    if (n != 37 || strncmp(s, "call_", 5) != 0) return false;
    for (size_t i = 5; i < n; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return false;
    return true;
}

static void ids_append(int32_t **ids, int *n, int *cap, const int32_t *src, int k) {
    if (k <= 0) return;
    if (*n + k > *cap) { *cap = (*n + k) * 2; *ids = (int32_t *)realloc(*ids, (size_t)*cap * sizeof(int32_t)); }
    if (*ids) { memcpy(*ids + *n, src, (size_t)k * sizeof(int32_t)); *n += k; }
}

// B3: encode the prompt, but where attach_tool_memory() inserted a splice
// sentinel (\x1f id \x1f), splice the EXACT sampled token ids for that id instead
// of re-tokenizing — DSML markers are special tokens that do not survive a
// detokenize->retokenize round trip, so re-tokenizing the replayed call would
// diverge from what was sampled and miss the post-tool-call snapshot. Text
// segments between sentinels are tokenized normally. Returns the combined length
// (>=0), or -1 on tokenizer error; *out_ids is malloc'd (caller frees).
static int encode_with_splices(ember_server *srv, ember_backend *be,
                               const char *prompt, int32_t **out_ids) {
    *out_ids = NULL;
    if (!strchr(prompt, '\x1f'))                     // fast path: nothing spliced
        return ember_backend_encode(be, prompt, out_ids);

    int32_t *ids = NULL; int n = 0, cap = 0;
    const char *p = prompt;
    while (*p) {
        const char *op = strchr(p, '\x1f');
        const char *text_end = op ? op : (p + strlen(p));
        if (text_end > p) {                          // text before the sentinel
            char *t = (char *)malloc((size_t)(text_end - p) + 1);
            if (!t) { free(ids); return -1; }
            memcpy(t, p, (size_t)(text_end - p)); t[text_end - p] = '\0';
            int32_t *tids = NULL;
            int tn = ember_backend_encode(be, t, &tids);
            free(t);
            if (tn < 0) { free(tids); free(ids); return -1; }
            ids_append(&ids, &n, &cap, tids, tn);
            free(tids);
        }
        if (!op) break;
        const char *cl = strchr(op + 1, '\x1f');
        if (cl && looks_like_tool_id(op + 1, (size_t)(cl - (op + 1)))) {
            char id[64];
            snprintf(id, sizeof(id), "%.*s", (int)(cl - (op + 1)), op + 1);
            int gn = 0;
            const int32_t *gids = ember_tool_memory_get_tokens(&srv->tool_mem, id, &gn);
            if (gids && gn > 0) {
                ids_append(&ids, &n, &cap, gids, gn);   // exact splice → snapshot hit
            } else {
                // Defensive: tokens evicted (can't happen between attach+encode on
                // the single worker thread). Re-tokenize the stored bytes so the
                // turn is still rendered correctly — just no continuation.
                const char *bytes = ember_tool_memory_get(&srv->tool_mem, id);
                if (bytes) {
                    int32_t *bids = NULL;
                    int bn = ember_backend_encode(be, bytes, &bids);
                    if (bn > 0) ids_append(&ids, &n, &cap, bids, bn);
                    free(bids);
                }
            }
            p = cl + 1;
        } else {
            p = op + 1;                              // lone \x1f (not a sentinel): drop
        }
    }
    *out_ids = ids;
    return n;
}

static void run_chat(ember_server *srv, ember_chat_request *req, int fd) {
    ember_backend *be = srv->be;
    // Single GPU slot: serialize generation. Health/models/status never lock,
    // so they stay responsive while a long generation holds the slot.
    pthread_mutex_lock(&srv->gen_lock);
    atomic_fetch_add(&srv->busy, 1);
    bool enable_thinking = req->thinking_enabled;
    attach_tool_memory(srv, req);  // B3: exact-DSML replay substitution
    char *prompt = ember_render_prompt(req, enable_thinking, req->think_mode, true);
    bool started_thinking = ember_prompt_ends_in_open_think(prompt);
    int32_t *ids = NULL;
    int n_prompt = encode_with_splices(srv, be, prompt, &ids);  // B3: exact-token splice
    free(prompt);

    // Defensive: a tokenizer error (documented -1) must not reach the backend
    // (would be UB: assign(NULL, NULL-1)).
    int n_ctx = ember_backend_n_ctx(be);
    if (n_prompt < 0) {
        respond(fd, 500, "application/json",
            "{\"error\":{\"message\":\"tokenization failed\",\"type\":\"server_error\"}}");
        free(ids);
        atomic_fetch_sub(&srv->busy, 1);
        pthread_mutex_unlock(&srv->gen_lock);
        return;
    }
    // Context-length guard (ds4 http_error_context_length_exceeded): reject a
    // prompt that already fills the window, with the OpenAI-shaped 400.
    if (n_prompt >= n_ctx) {
        ember_buf e = {0};
        ember_buf_printf(&e,
            "{\"error\":{\"message\":\"prompt is %d tokens, exceeds context %d\","
            "\"type\":\"invalid_request_error\",\"param\":\"messages\","
            "\"code\":\"context_length_exceeded\",\"n_prompt_tokens\":%d,"
            "\"n_ctx\":%d}}", n_prompt, n_ctx, n_prompt, n_ctx);
        respond(fd, 400, "application/json", e.ptr);
        ember_buf_free(&e);
        free(ids);
        atomic_fetch_sub(&srv->busy, 1);
        pthread_mutex_unlock(&srv->gen_lock);
        return;
    }

    // Monotonic response id (ds4 "chatcmpl-<seq>"): collision-free vs a
    // wall-clock id when requests share a second.
    static _Atomic unsigned long g_req_seq = 0;
    unsigned long seq = atomic_fetch_add(&g_req_seq, 1);
    char id[48];
    snprintf(id, sizeof(id), "chatcmpl-%lu", seq);
    long created = now_unix();

    // max_tokens: ds4 defaults to full context room when unset; a negative
    // value means 0 (generate nothing); always capped to remaining room.
    int room = n_ctx - n_prompt;
    int want = req->max_tokens_set ? (req->max_tokens < 0 ? 0 : req->max_tokens) : room;
    if (want > room) want = room;

    ember_gen_request greq = {0};
    greq.prompt = ids;
    greq.n_prompt = n_prompt;
    greq.max_tokens = want;
    // ds4 sampler defaults: top_p 1.0, top_k 0, min_p 0.05. Temperature defaults
    // to srv->default_temp when the request omits it (ds4 code default 1.0; the
    // production unit sets 0 so temp-omitting clients like Hermes run greedy and
    // hit the DSpark fast path).
    double eff_temp = req->temperature_set ? req->temperature : srv->default_temp;
    greq.greedy = (eff_temp == 0.0);
    greq.temperature = (float)eff_temp;
    greq.top_p = req->top_p_set ? (float)req->top_p : 1.0f;
    greq.top_k = req->top_k_set ? req->top_k : 0;
    greq.min_p = req->min_p_set ? (float)req->min_p : 0.05f;
    greq.seed = req->seed; greq.seed_set = req->seed_set;
    greq.rep_pen = req->rep_pen_set ? (float)req->rep_pen : 1.0f;
    greq.rep_window = req->rep_window_set ? req->rep_window : 0;
    greq.freq_pen = (float)req->freq_pen;
    greq.pres_pen = (float)req->pres_pen;
    // Level-2 thinking force-close: reserve hard_limit_reply_budget tokens so
    // the model always gets to emit a visible answer after </think>, instead of
    // thinking until it exhausts max_tokens (the bug that leaked raw reasoning).
    // Only armed when thinking is on, the card supplies a reply reserve, and the
    // combined budget leaves room to think.
    if (enable_thinking && srv->n_close_ids > 0 &&
        srv->card.hard_limit_reply_budget > 0 &&
        greq.max_tokens > srv->card.hard_limit_reply_budget) {
        greq.budget_close_ids = srv->close_ids;
        greq.n_budget_close   = srv->n_close_ids;
        greq.reply_budget     = srv->card.hard_limit_reply_budget;
    }

    // KV reuse: restore the longest cached prefix (backend prefills only the
    // suffix), and reserve a slot to snapshot this prompt for next turn. Under
    // gen_lock, so the cache is accessed single-threaded.
    int restore_slot = -1, restore_len = 0;
    ember_kv_lookup(&srv->kv, ids, n_prompt, &restore_slot, &restore_len);
    int snap_cut = ember_kv_snap_cut(&srv->kv, ids, n_prompt);
    // On an in-memory miss, fall back to the cross-restart disk cache.
    if (restore_slot < 0 && ember_backend_disk_enabled(be)) {
        int dl = ember_backend_disk_prefix(be, ids, n_prompt);
        if (dl > 0 && ember_backend_disk_lookup(be, ids, dl, EMBER_KV_DISK_SLOT)) {
            restore_slot = EMBER_KV_DISK_SLOT;
            restore_len = dl;
        }
    }
    int snap_slot = -1;
    if (snap_cut > restore_len && snap_cut > 0 && snap_cut <= n_prompt)
        snap_slot = ember_kv_reserve(&srv->kv);
    // Never snapshot into the slot we're restoring from (would alias the source).
    if (snap_slot == restore_slot) snap_slot = -1;
    greq.restore_slot = restore_slot;
    greq.snap_slot = snap_slot;
    greq.snap_pos = snap_slot >= 0 ? snap_cut : -1;

    int max_stop_len = 0;
    for (int si = 0; si < req->n_stop; si++) {
        int l = (int)strlen(req->stop[si]);
        if (l > max_stop_len) max_stop_len = l;
    }

    if (req->stream) {
        ember_buf hdr = {0};
        ember_sse_headers(&hdr, false);
        ember_send_all(fd, hdr.ptr, hdr.len);
        ember_buf_free(&hdr);

        ember_sse_stream st;
        ember_sse_init(&st, id, req->model, created, req->has_tools,
                       started_thinking, false);
        st.include_usage = req->stream_include_usage;
        st.cached_tokens = restore_len;
        st.stops = req->stop; st.n_stops = req->n_stop; st.max_stop_len = max_stop_len;
        // Initial role primer chunk (ds4), before any content delta.
        ember_buf rc = {0};
        ember_sse_role_chunk(&st, &rc);
        ember_send_all(fd, rc.ptr, rc.len);
        ember_buf_free(&rc);
        gen_ctx g = {0};
        g.be = be; g.st = &st; g.fd = fd; g.has_tools = req->has_tools;
        g.stops = req->stop; g.n_stops = req->n_stop;
        g.started_thinking = started_thinking;  // B#1: gate markers on thinking

        greq.on_token = on_token;
        greq.on_prefill = on_prefill;
        greq.ud = &g;
        // B6: only arm structural-greedy when sampling could actually corrupt the
        // DSML (temp > 0 with tools); greedy runs already argmax everything.
        g.dsml_active = req->has_tools && eff_temp > 0.0;
        if (g.dsml_active) {
            ember_dsml_tracker_init(&g.dsml);
            greq.force_greedy = gen_force_greedy;
            greq.fg_ud = &g;
        }
        ember_gen_result res = ember_backend_generate(be, &greq);
        // #2: commit the logical prefix ONLY when the backend actually saved the
        // snapshot. Gating on !res.cancelled instead poisons the cache — a failed
        // save still "succeeds" the generation, so the next lookup reports a hit
        // the backend can't honor (full re-prefill, no repair snapshot reserved).
        if (snap_slot >= 0 && res.ok && !g.disconnected && res.snapshot_saved) {
            ember_kv_commit(&srv->kv, snap_slot, ids, snap_cut);
            if (ember_backend_disk_enabled(be))
                ember_backend_disk_save(be, snap_slot, ids, snap_cut);
        }
        snapshot_post_toolcall(srv, be, ids, n_prompt, &g);  // B3 L2: park post-call KV

        g.scratch.len = 0; if (g.scratch.ptr) g.scratch.ptr[0] = '\0';
        ember_sse_update(&st, g.acc.ptr, g.acc.len, true, &g.scratch);
        bool had_tools = ember_sse_emit_tools(&st, g.acc.ptr, g.acc.len, &g.scratch);
        if (!res.ok && !g.disconnected) {
            // Backend failure mid-stream (ds4 sse_error_event).
            ember_sse_error(&st, res.error_code, res.error_detail, &g.scratch);
        } else {
            // degenerate n-gram loop → report "length" (truncated/unreliable), not "stop".
            const char *finish = g.disconnected ? "stop"
                               : (had_tools ? "tool_calls"
                               : (g.hit_stop ? "stop"
                               : (res.degenerate_decode_close ? "length" : res.finish_reason)));
            ember_sse_finish(&st, finish, n_prompt, res.n_generated, &g.scratch);
        }
        if (!g.disconnected) ember_send_all(fd, g.scratch.ptr, g.scratch.len);

        // B3: store the exact sampled bytes under each streamed tool-call id so a
        // later turn replays this call token-identically (post-call snapshot hit).
        for (int i = 0; i < st.n_tool_ids; i++)
            if (g.acc.ptr) ember_tool_memory_put(&srv->tool_mem, st.tool_ids[i],
                                                 g.acc.ptr, g.acc.len,
                                                 g.gen_ids, g.n_gen_ids);
        ember_sse_free(&st);
        ember_buf_free(&g.acc);
        free(g.gen_ids);  // B3 L2
        ember_buf_free(&g.scratch);
    } else {
        // Non-stream: collect all tokens, then split reasoning vs content and
        // build one chat.completion. (Tool-call structuring is a follow-on.)
        gen_ctx g = {0};
        g.be = be; g.fd = fd; g.collect_only = true; g.has_tools = req->has_tools;
        g.stops = req->stop; g.n_stops = req->n_stop;
        g.started_thinking = started_thinking;  // B#1: gate markers on thinking
        greq.on_token = on_token;
        greq.on_prefill = NULL;
        greq.ud = &g;
        g.dsml_active = req->has_tools && eff_temp > 0.0;  // B6 (see streaming path)
        if (g.dsml_active) {
            ember_dsml_tracker_init(&g.dsml);
            greq.force_greedy = gen_force_greedy;
            greq.fg_ud = &g;
        }
        ember_gen_result res = ember_backend_generate(be, &greq);
        // #2: commit only on a real backend snapshot save (see streaming path).
        if (snap_slot >= 0 && res.ok && res.snapshot_saved) {
            ember_kv_commit(&srv->kv, snap_slot, ids, snap_cut);
            if (ember_backend_disk_enabled(be))
                ember_backend_disk_save(be, snap_slot, ids, snap_cut);
        }
        snapshot_post_toolcall(srv, be, ids, n_prompt, &g);  // B3 L2: park post-call KV

        // Backend failure with no output → 500 with the surfaced error.
        if (!res.ok && res.n_generated == 0) {
            ember_buf e = {0};
            ember_buf_puts(&e, "{\"error\":{\"message\":");
            ember_json_escape(&e, res.error_detail[0] ? res.error_detail
                                                       : "backend generation failed");
            ember_buf_puts(&e, ",\"type\":\"server_error\",\"code\":");
            ember_json_escape(&e, res.error_code[0] ? res.error_code : "internal_error");
            ember_buf_puts(&e, "}}");
            respond(fd, 500, "application/json", e.ptr);
            ember_buf_free(&e);
            ember_buf_free(&g.acc);
            free(g.gen_ids);  // B3 L2
            ember_buf_free(&g.scratch);
            goto ns_done;
        }

        // Separate a leading <think>…</think> (or prompt-opened thinking) from
        // the visible content.
        const char *full = g.acc.ptr ? g.acc.ptr : "";
        const char *content = full;
        char *reasoning = NULL;
        if (started_thinking) {
            const char *close = strstr(full, "</think>");
            if (close) {
                reasoning = strndup(full, (size_t)(close - full));
                content = close + 8;
            } else {
                reasoning = strdup(full);
                content = "";
            }
        }

        // stop sequences (ds4): truncate visible content at the earliest stop
        // hit and report finish="stop".
        char *content_stop = NULL;
        bool hit_stop = false;
        for (int si = 0; si < req->n_stop; si++) {
            const char *h = strstr(content, req->stop[si]);
            if (!h) continue;
            size_t clen = (size_t)(h - content);
            if (!content_stop || clen < strlen(content_stop)) {
                free(content_stop);
                content_stop = strndup(content, clen);
            }
            hit_stop = true;
        }
        if (content_stop) content = content_stop;

        // Tool calls: parse the post-think text; if any, strip the DSML block
        // from content and set finish="tool_calls" (ds4 non-stream parity —
        // previously the raw DSML leaked into content with finish="stop").
        ember_tool_calls tc = {0};
        ember_parse_dsml_tool_calls(content, &tc);
        // B3: mint an id per call and remember the exact sampled turn bytes under
        // each (g.acc is the full generated assistant output, truncated at the
        // tool-block end), so a later replay renders this turn token-identically.
        for (int i = 0; i < tc.len; i++) {
            char tid[40]; mint_tool_id(tid);
            free(tc.calls[i].id);
            tc.calls[i].id = strdup(tid);
            if (g.acc.ptr) ember_tool_memory_put(&srv->tool_mem, tid, g.acc.ptr, g.acc.len,
                                                 g.gen_ids, g.n_gen_ids);
        }
        char *content_trimmed = NULL;
        const char *emit_content = content;
        const char *finish = hit_stop ? "stop"
                           : (res.degenerate_decode_close ? "length" : res.finish_reason);
        if (tc.len > 0) {
            const char *ts = ember_find_tool_start(content);
            size_t clen = ts ? (size_t)(ts - content) : strlen(content);
            while (clen > 0 && (content[clen-1]=='\n' || content[clen-1]==' ' ||
                                content[clen-1]=='\t' || content[clen-1]=='\r')) clen--;
            content_trimmed = strndup(content, clen);
            emit_content = content_trimmed;
            finish = "tool_calls";
        }

        ember_buf b = {0};
        ember_buf_printf(&b,
            "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":",
            id, created);
        ember_buf_putc(&b, '"'); json_escape_str(&b, req->model); ember_buf_putc(&b, '"');
        ember_buf_puts(&b, ",\"choices\":[{\"index\":0,\"message\":"
                          "{\"role\":\"assistant\",\"content\":");
        ember_buf_putc(&b, '"');
        json_escape_str(&b, emit_content);
        ember_buf_putc(&b, '"');
        if (reasoning) {
            ember_buf_puts(&b, ",\"reasoning_content\":\"");
            json_escape_str(&b, reasoning);
            ember_buf_putc(&b, '"');
        }
        if (tc.len > 0) {
            ember_buf_puts(&b, ",\"tool_calls\":[");
            for (int i = 0; i < tc.len; i++) {
                if (i) ember_buf_putc(&b, ',');
                ember_buf_printf(&b,
                    "{\"index\":%d,\"id\":\"%s\",\"type\":\"function\","      // B3: minted id
                    "\"function\":{\"name\":\"", i,
                    tc.calls[i].id ? tc.calls[i].id : id);
                json_escape_str(&b, tc.calls[i].name ? tc.calls[i].name : "");
                ember_buf_puts(&b, "\",\"arguments\":\"");
                json_escape_str(&b, tc.calls[i].arguments ? tc.calls[i].arguments : "{}");
                ember_buf_puts(&b, "\"}}");
            }
            ember_buf_putc(&b, ']');
        }
        double dtps = res.decode_s > 0 ? res.n_generated / res.decode_s : 0.0;
        double ptps = res.prefill_s > 0 ? (n_prompt - restore_len) / res.prefill_s : 0.0;
        ember_buf_printf(&b,
            "},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,"
            "\"completion_tokens\":%d,\"total_tokens\":%d,"
            "\"prompt_tokens_details\":{\"cached_tokens\":%d},\"timings\":{"
            "\"prefill_ms\":%.1f,\"prefill_tokens_per_sec\":%.1f,"
            "\"decode_ms\":%.1f,\"decode_tokens_per_sec\":%.2f},"
            "\"accept_rate\":%.3f,\"restored_prefix\":%d,"
            "\"backend\":{\"forced_close\":%s,\"degenerate\":%s,"
            "\"empty\":%s,\"spec_ran\":%s}}}",
            finish, n_prompt, res.n_generated,
            n_prompt + res.n_generated, restore_len,
            res.prefill_s * 1000.0, ptps,
            res.decode_s * 1000.0, dtps, res.accept_rate, restore_len,
            res.budget_forced_close ? "true" : "false",
            res.degenerate_decode_close ? "true" : "false",
            res.empty_visible_output ? "true" : "false",
            res.spec_decode_ran ? "true" : "false");
        respond(fd, 200, "application/json", b.ptr);
        ember_buf_free(&b);
        ember_tool_calls_free(&tc);
        free(content_trimmed);
        free(content_stop);
        free(reasoning);
        ember_buf_free(&g.acc);
        free(g.gen_ids);  // B3 L2
        ember_buf_free(&g.scratch);
    ns_done: ;
    }
    atomic_fetch_sub(&srv->busy, 1);
    atomic_fetch_add(&srv->served, 1);
    pthread_mutex_unlock(&srv->gen_lock);
    free(ids);
}

// Worker loop: pop one job, run it to completion on this thread, wake its
// waiter. Runs for the whole process lifetime, so every backend forward pass
// (and thus every thread_local graph cache it builds) stays on one thread.
// On stop it drains any already-queued jobs (so their waiters unblock) then
// exits, letting gen_worker_stop join before the backend is freed.
static void *gen_worker_main(void *arg) {
    gen_worker *w = (gen_worker *)arg;
    for (;;) {
        pthread_mutex_lock(&w->lock);
        while (!w->head && !w->stop) pthread_cond_wait(&w->cond, &w->lock);
        if (!w->head && w->stop) { pthread_mutex_unlock(&w->lock); break; }
        gen_job *job = w->head;
        w->head = job->next;
        if (!w->head) w->tail = NULL;
        w->queued--;  // #3: dequeued from the bounded FIFO
        pthread_mutex_unlock(&w->lock);

        run_chat(job->srv, job->req, job->fd);

        pthread_mutex_lock(&job->lock);
        job->done = true;
        pthread_cond_signal(&job->cond);
        pthread_mutex_unlock(&job->lock);
    }
    // #5: free the backend HERE, on the worker thread, as its last action. The
    // DeepSeek4 graph caches live in this thread's thread_local storage and are
    // unreachable from main; main must NOT free it again (would double-free).
    if (w->be) ember_backend_free(w->be);
    return NULL;
}

// Start the worker. Returns false (leaving nothing to clean up) if any of the
// mutex/cond/thread primitives fail — the caller MUST abort startup, because a
// missing worker would make every gen_worker_submit() block forever.
static bool gen_worker_start(gen_worker *w, ember_backend *be) {
    w->head = w->tail = NULL;
    w->queued = 0;
    w->be = be;  // #5: the worker owns backend teardown (runs on its own thread)
    w->stop = false;
    w->running = false;
    if (pthread_mutex_init(&w->lock, NULL) != 0) return false;
    if (pthread_cond_init(&w->cond, NULL) != 0) {
        pthread_mutex_destroy(&w->lock);
        return false;
    }
    if (pthread_create(&w->thread, NULL, gen_worker_main, w) != 0) {
        pthread_cond_destroy(&w->cond);
        pthread_mutex_destroy(&w->lock);
        return false;
    }
    w->running = true;
    return true;
}

// Signal the worker to drain and exit, join it, and destroy its primitives.
// Must run before the backend is freed so no forward pass is in flight. Safe to
// call on a worker that never started.
static void gen_worker_stop(gen_worker *w) {
    if (!w->running) return;
    pthread_mutex_lock(&w->lock);
    w->stop = true;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->lock);
    pthread_join(w->thread, NULL);
    pthread_cond_destroy(&w->cond);
    pthread_mutex_destroy(&w->lock);
    w->running = false;
}

// Enqueue a generation job and block until the worker finishes it. `job` lives
// on the calling connection thread's stack; that's safe precisely because this
// function does not return until job.done, so the worker never touches it after
// the frame is gone. Blocking here also preserves the old per-connection
// backpressure (a slow client still holds only its own thread).
// #3: returns false WITHOUT enqueuing when the FIFO is already at capacity, so
// the caller can respond 503 rather than let the backlog grow unbounded.
static bool gen_worker_submit(gen_worker *w, ember_server *srv,
                              ember_chat_request *req, int fd) {
    gen_job job = {0};
    job.srv = srv; job.req = req; job.fd = fd;
    pthread_mutex_init(&job.lock, NULL);
    pthread_cond_init(&job.cond, NULL);

    pthread_mutex_lock(&w->lock);
    if (w->queued >= EMBER_MAX_QUEUE_DEPTH) {  // #3: overloaded — shed this job
        pthread_mutex_unlock(&w->lock);
        pthread_mutex_destroy(&job.lock);
        pthread_cond_destroy(&job.cond);
        return false;
    }
    if (w->tail) w->tail->next = &job; else w->head = &job;
    w->tail = &job;
    w->queued++;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->lock);

    pthread_mutex_lock(&job.lock);
    while (!job.done) pthread_cond_wait(&job.cond, &job.lock);
    pthread_mutex_unlock(&job.lock);

    pthread_mutex_destroy(&job.lock);
    pthread_cond_destroy(&job.cond);
    return true;
}

static void handler(const ember_http_request *req, int fd, void *ud) {
    ember_server *srv = (ember_server *)ud;
    ember_backend *be = srv->be;
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/health") == 0) {
        respond(fd, 200, "text/plain", "ok\n");
        return;
    }
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/v1/models") == 0) {
        ember_buf b = {0};
        ember_buf_printf(&b,
            "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
            "\"owned_by\":\"otheru\",\"max_context_length\":%d}]}",
            ember_backend_model_name(be), ember_backend_n_ctx(be));
        respond(fd, 200, "application/json", b.ptr);
        ember_buf_free(&b);
        return;
    }
    if (strcmp(req->method, "POST") == 0 &&
        strncmp(req->path, "/v1/chat/completions", 20) == 0) {
        char *body = strndup(req->body, req->body_len);
        ember_json *root = ember_json_parse(body);
        ember_chat_request creq;
        if (root && ember_chat_request_parse(root, &creq)) {
            // Run generation on the persistent worker (keeps the backend's
            // thread_local graph caches warm); block until it completes.
            // #3: a full worker queue sheds load with a 503 instead of blocking.
            if (!gen_worker_submit(&srv->worker, srv, &creq, fd)) {
                respond(fd, 503, "application/json",
                    "{\"error\":{\"message\":\"server overloaded, retry later\","
                    "\"type\":\"server_error\",\"code\":\"overloaded\"}}");
            }
            ember_chat_request_free(&creq);
        } else {
            respond(fd, 400, "application/json",
                "{\"error\":{\"message\":\"invalid JSON request or missing messages\","
                "\"type\":\"invalid_request_error\"}}");
        }
        if (root) ember_json_free(root);
        free(body);
        return;
    }
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/status") == 0) {
        ember_buf b = {0};
        ember_buf_printf(&b,
            "{\"model\":\"%s\",\"ctx\":%d,\"busy\":%d,\"served\":%ld}",
            ember_backend_model_name(be), ember_backend_n_ctx(be),
            atomic_load(&srv->busy), atomic_load(&srv->served));
        respond(fd, 200, "application/json", b.ptr);
        ember_buf_free(&b);
        return;
    }
    respond(fd, 404, "application/json",
            "{\"error\":{\"message\":\"not found\"}}");
}

int main(int argc, char **argv) {
    int port = 8080;
    const char *model_path = NULL, *model_name = "deepseek-v4-flash";
    const char *card_path = NULL, *kv_dir = NULL;
    // Match lucebox production: the DSpark draft was calibrated against the
    // target running with expert-top-k=4, not the model default (6). Leaving
    // this at the model default diverges the target logits from the draft's
    // expectation — DSpark acceptance collapses (~0.95 -> ~0.68) and every
    // forward pass runs 50% more experts. Default to 4; allow override.
    int expert_top_k = 4;
    double default_temp = 1.0;  // ds4 code default; production unit passes 0 (greedy)
    // In-memory KV prefix-cache slots. Each committed slot holds a full-KV
    // snapshot (~448MB at ctx=65536), so 32 slots ≈ 14GiB — on a memory-tight
    // box that pushes the UMA model into swap and thrashes decode. Cap it.
    int prefix_slots = 32;
    int max_ctx = 65536;  // KV cache context; each snapshot is a full-KV buffer
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--model-name") == 0 && i + 1 < argc) model_name = argv[++i];
        else if (strcmp(argv[i], "--model-card") == 0 && i + 1 < argc) card_path = argv[++i];
        else if (strcmp(argv[i], "--kv-cache-dir") == 0 && i + 1 < argc) kv_dir = argv[++i];
        else if (strcmp(argv[i], "--ds4-expert-top-k") == 0 && i + 1 < argc) expert_top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--default-temperature") == 0 && i + 1 < argc) default_temp = atof(argv[++i]);
        else if (strcmp(argv[i], "--prefix-cache-slots") == 0 && i + 1 < argc) prefix_slots = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-ctx") == 0 && i + 1 < argc) max_ctx = atoi(argv[++i]);
    }
    if (prefix_slots < 1) prefix_slots = 1;
    if (prefix_slots > EMBER_KV_MAX_SLOTS - 1) prefix_slots = EMBER_KV_MAX_SLOTS - 1;
    ember_backend_config cfg = {0};
    cfg.model_path = model_path;
    cfg.model_name = model_name;
    cfg.max_ctx = max_ctx > 0 ? max_ctx : 65536;
    cfg.expert_top_k = expert_top_k;
    cfg.kv_cache_dir = kv_dir;
    cfg.kv_cache_mb = 0;
    char *err = NULL;
    ember_backend *be = ember_backend_load(&cfg, &err);
    if (!be) {
        fprintf(stderr, "[ember] backend load failed: %s\n", err ? err : "?");
        return 1;
    }
    ember_server srv = {0};
    srv.be = be;
    srv.default_temp = default_temp;
    pthread_mutex_init(&srv.gen_lock, NULL);
    ember_model_card_load(&srv.card, card_path);
    ember_kv_init(&srv.kv, prefix_slots,
                  marker_id(be, "<" PIPE "User" PIPE ">"),
                  marker_id(be, "<" PIPE "Assistant" PIPE ">"),
                  ember_backend_eos_id(be));
    ember_tool_memory_init(&srv.tool_mem, 512);  // B3: exact-DSML replay memory
    // Resolve the Level-2 force-close token sequence once at startup: the
    // card's terminator hint (a "wrap up now" directive + </think>) if present,
    // else a bare </think>. Encoded via the real tokenizer so the backend can
    // inject it verbatim at the reply-budget edge.
    const char *close_text =
        (srv.card.thinking_terminator_hint && srv.card.thinking_terminator_hint[0])
            ? srv.card.thinking_terminator_hint
            : "</think>";
    srv.n_close_ids = ember_backend_encode(be, close_text, &srv.close_ids);
    fprintf(stderr,
            "[ember] backend ready: %s (ctx=%d, reply_reserve=%d, close_seq=%d tok, "
            "default_temp=%.2f)\n",
            ember_backend_model_name(be), ember_backend_n_ctx(be),
            srv.card.hard_limit_reply_budget, srv.n_close_ids, srv.default_temp);
    // Start the persistent generation worker before accepting connections: all
    // backend forward passes run on it so the thread_local graph caches build
    // once and stay warm (see gen_worker above). If it can't start, abort —
    // serving anyway would wedge every request forever in gen_worker_submit().
    if (!gen_worker_start(&srv.worker, be)) {
        fprintf(stderr, "[ember] failed to start generation worker\n");
        ember_kv_free(&srv.kv);
        ember_tool_memory_free(&srv.tool_mem);
        ember_model_card_free(&srv.card);
        free(srv.close_ids);
        ember_backend_free(be);  // worker never ran → no worker TLS; free here
        return 1;
    }
    int rc = ember_http_serve(port, handler, &srv);
    // ember_http_serve currently loops forever; the shutdown path runs only if
    // that ever returns. #5: gen_worker_stop joins the worker, which frees the
    // backend as its LAST action on its own thread (so the DeepSeek4 graph
    // caches in its thread_local storage are reachable). main must NOT free the
    // backend again here — that would double-free.
    gen_worker_stop(&srv.worker);
    ember_kv_free(&srv.kv);
    ember_tool_memory_free(&srv.tool_mem);
    ember_model_card_free(&srv.card);
    free(srv.close_ids);
    return rc;
}
