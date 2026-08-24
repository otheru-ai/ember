// ember-server: HTTP entry point. Drives the full pipeline through the backend
// ABI — render prompt (chat_template) → encode → ember_backend_generate →
// per-token detokenize → SSE stream. Only the backend forward pass is stubbed
// (backend_stub.c); swapping in backend_dflash.cc changes nothing here.
#include <stdio.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
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
#include "../model/continuation.h"
#include "../model/dsml_decode.h"
#include "../model/tool_schema.h"
#include "../model/tool_grammar.h"
#include "chat_api.h"
#include "api_adapters.h"
#include "background_gate.h"
#include "compaction.h"
#include "http.h"
#include "sse.h"

#define PIPE "\xef\xbd\x9c"  // U+FF5C
static bool g_enable_cors;
// Off-by-default token forensic trace. Each committed output token is logged as
// an id plus exact decoded bytes, making model-selected whitespace distinguishable
// from detokenizer/SSE joining without changing the wire response.
static bool g_trace_tokens;

// ── persistent generation workers ────────────────────────────────────────
// Legacy backend generation runs on one long-lived thread so lucebox's
// thread_local graph caches (the layer-major prefill cache, its shared gallocr,
// the DSpark draft cache) build once and stay warm. Previously every HTTP
// connection ran generation on its own short-lived detached thread (http.c
// conn_thread): each request got a fresh thread_local cache, rebuilt the
// ~918MB high-water prefill arena from scratch, and then ORPHANED it (leaked)
// when the connection thread exited — ~918MB of unreachable growth per full
// prefill. Connection threads now hand the job to this worker and block until
// it completes. In continuous-batch mode a bounded pool dispatches concurrent
// sessions into the engine-owned resident coordinator.
struct ember_server;  // defined below

typedef struct gen_job {
    struct ember_server      *srv;
    ember_chat_request       *req;   // borrowed; lives on the waiter's stack
                                     // (mutable: run_chat attaches B3 replay bytes)
    int                       fd;
    bool                      done;
    double                    enqueued_at;
    pthread_mutex_t           lock;
    pthread_cond_t            cond;
    struct gen_job           *next;
} gen_job;

// #3: cap the pending-job FIFO. With generation serialized, extra chat requests
// pile up as blocked connection threads; a bounded queue sheds load (503) once
// the backlog is this deep instead of growing unbounded.
#define EMBER_MAX_QUEUE_DEPTH 8
#define EMBER_MAX_BATCH_SESSIONS 64

typedef struct {
    gen_job         *head, *tail;  // FIFO of pending jobs (guarded by lock)
    int              queued;       // #3: current FIFO depth (guarded by lock)
    int              active_jobs;  // dequeued requests still in run_chat
    pthread_mutex_t  lock;
    pthread_cond_t   cond;
    pthread_t        threads[EMBER_MAX_BATCH_SESSIONS];
    int              n_threads;
    bool             stop;         // set by gen_worker_stop to drain + exit
    bool             running;      // true between a successful start and stop
    ember_backend   *be;           // #5: freed on THIS thread at teardown
    ember_background_gate bg_gate; // foreground-priority maintenance scheduling
    // Idle graph reclaim. The DeepSeek4 compute-graph caches are thread_local
    // on THIS thread and only ever grew: they ratchet to the largest context
    // ever served and were previously freed only at model teardown, so a
    // restart was the sole way to reclaim (measured GTT 12GB -> 28.6GB, flat
    // while idle). Reclaim after a quiet period instead, on this thread.
    double           last_done_at;      // monotonic time the last job finished
    bool             graphs_dirty;      // a job ran since the last reclaim
    bool             reclaiming;
    double           idle_reclaim_secs; // 0 = disabled
} gen_worker;

typedef struct ember_server {
    ember_backend    *be;
    ember_model_card  card;
    ember_kv_cache    kv;         // prefix KV cache (guarded by state_lock)
    ember_tool_memory tool_mem;   // B3 exact tool-call bytes (guarded)
    ember_continuation_store continuations; // call-id set -> sampled KV frontier
    pthread_mutex_t   gen_lock;   // legacy monolithic generation serialization
    pthread_mutex_t   state_lock; // prefix/tool/continuation bookkeeping
    atomic_int        busy;       // reported by /status
    atomic_long       served;     // completed generations
    // Latest request-derived tool-loop alert, guarded by state_lock. This is
    // telemetry only: detection never reads it and has no cross-request state.
    int               tool_loop_report;
    // > 0 arms automatic loop recovery: see auto_answer_suppresses_tools().
    int               auto_answer_after_loop;
    long              auto_answer_count;
    long              tool_markup_leak_count;   // visible tool markup delivered
    long              last_tool_markup_leak_at;
    long              last_auto_answer_at;
    char              last_auto_answer_tool[128];
    int               last_tool_loop_rounds;
    bool              last_tool_loop_identical;  // strict signal, not call-only
    long              last_tool_loop_at;
    char              last_tool_loop_tool[128];
    // Progress lease (ember_chat_request_progress_lease): trailing tool rounds
    // that returned nothing new. Keys on the RESULT, so it sees the stalls both
    // tool-loop signals miss -- they key on the call. Telemetry only.
    int               no_progress_report;
    long              no_progress_count;
    long              last_no_progress_at;
    int               last_no_progress_rounds;
    char              last_no_progress_tool[128];
    // Latest non-progress turn, guarded by state_lock. Telemetry only, exactly
    // like the tool-loop fields above: nothing reads these to make a decision.
    long              nonprogress_count;
    long              last_nonprogress_at;
    int               last_nonprogress_tokens;
    bool              last_nonprogress_degenerate;
    // Every backend-flagged degenerate decode, whether or not it produced
    // output. Strictly wider than nonprogress_count above: the two overlap.
    long              degenerate_count;
    long              last_degenerate_at;
    int               last_degenerate_tokens;
    bool              last_degenerate_had_output;
    char              last_degenerate_reason[32];
    int32_t          *close_ids;  // Level-2 force-close token sequence (owned)
    int               n_close_ids;
    int32_t          *natural_close_ids; // bare </think> disarm sequence (owned)
    int               n_natural_close_ids;
    double            default_temp;  // card value unless CLI-overridden
    bool              auto_compact;  // --auto-compact: ds4-style context compaction
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

static double monotonic_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static struct timespec monotonic_deadline(double when) {
    if (when < 0.0) when = 0.0;
    struct timespec ts;
    ts.tv_sec = (time_t)when;
    ts.tv_nsec = (long)((when - (double)ts.tv_sec) * 1e9);
    if (ts.tv_nsec < 0) ts.tv_nsec = 0;
    if (ts.tv_nsec > 999999999L) ts.tv_nsec = 999999999L;
    return ts;
}

// JSON-escape `s` into `b` (no surrounding quotes).
static void json_escape_str(ember_buf *b, const char *s) {
    ember_json_escape_content(b, s);
}

static void append_tool_loop_json(ember_buf *b, int rounds,
                                  const char *tool, bool identical_results) {
    if (rounds <= 0) return;
    ember_buf_printf(b, "\"ember_tool_loop\":{\"rounds\":%d,\"tool\":",
                     rounds);
    ember_json_escape(b, tool ? tool : "");
    ember_buf_printf(b, ",\"identical_results\":%s}",
                     identical_results ? "true" : "false");
}

static bool respond(int fd, int code, const char *ctype, const char *body) {
    ember_buf b = {0};
    const char *reason = code == 200 ? "OK"
                       : code == 204 ? "No Content"
                       : code == 400 ? "Bad Request"
                       : code == 422 ? "Unprocessable Content"
                       : code == 404 ? "Not Found"
                       : code == 409 ? "Conflict"
                       : code == 500 ? "Internal Server Error"
                       : code == 503 ? "Service Unavailable" : "Error";
    size_t body_len = body ? strlen(body) : 0;
    ember_buf_printf(&b,
        "HTTP/1.1 %d %s\r\nContent-Length: %zu\r\n",
        code, reason, body_len);
    if (ctype && ctype[0])
        ember_buf_printf(&b, "Content-Type: %s\r\n", ctype);
    if (g_enable_cors)
        ember_buf_puts(&b,
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: *\r\n");
    ember_buf_puts(&b, "Connection: close\r\n\r\n");
    if (body_len) ember_buf_append(&b, body, body_len);
    bool sent = ember_send_all(fd, b.ptr, b.len) == 0;
    ember_buf_free(&b);
    return sent;
}

static void respond_api_error(int fd, ember_api_kind api, int status,
                              const char *message, const char *type,
                              const char *code) {
    ember_buf e = {0};
    if (api == EMBER_API_ANTHROPIC) {
        const char *anthropic_type = status == 503 ? "overloaded_error"
            : status == 400 ? "invalid_request_error"
            : status == 404 ? "not_found_error" : "api_error";
        ember_buf_puts(&e, "{\"type\":\"error\",\"error\":{\"type\":");
        ember_json_escape(&e, anthropic_type);
        ember_buf_puts(&e, ",\"message\":");
        ember_json_escape(&e, message ? message : "request failed");
        ember_buf_puts(&e, "}}");
    } else {
        ember_buf_puts(&e, "{\"error\":{\"message\":");
        ember_json_escape(&e, message ? message : "request failed");
        ember_buf_puts(&e, ",\"type\":");
        ember_json_escape(&e, type ? type : "server_error");
        if (code && code[0]) {
            ember_buf_puts(&e, ",\"code\":");
            ember_json_escape(&e, code);
        }
        ember_buf_puts(&e, "}}");
    }
    respond(fd, status, "application/json", e.ptr);
    ember_buf_free(&e);
}

// ── generation context threaded through the backend callbacks ──
typedef struct {
    ember_backend    *be;
    ember_sse_stream *st;        // NULL when collect_only
    ember_buf         acc;       // raw generated text (buffer-and-resplit source)
    ember_buf         scratch;   // per-call SSE output
    int               fd;
    bool              collect_only;  // accumulate without streaming (non-stream path)
    bool              keepalive_while_collecting; // hidden/buffered streaming attempt
    bool              disconnected;
    bool              has_tools;     // stop decode at the tool-calls end marker
    bool              started_thinking;  // B#1: prompt ended inside an open <think>
    time_t            last_ka;       // last prefill keepalive (throttle to ~4s)
    char            **stops;         // client stop strings (borrowed)
    int               n_stops;
    bool              hit_stop;      // a stop sequence was reached
    const char       *hit_stop_sequence; // borrowed exact matching request stop
    int32_t          *gen_ids;       // B3 L2: committed generated token ids
    int               n_gen_ids, gen_cap;
    const int32_t    *prompt_ids;    // borrowed during this backend call
    int               n_prompt_ids;
    bool              think_tool_recovery_suppressed; // exact prompt-like DSML
    bool              dsml_active;   // B6: track DSML decode state for greedy struct
    ember_dsml_tracker dsml;         // B6: structural-token greedy sampling state
} gen_ctx;

static bool send_generation_keepalive(gen_ctx *g) {
    time_t now = time(NULL);
    if (g->last_ka != 0 && now - g->last_ka < 4) return true;
    g->last_ka = now;
    ember_buf ka = {0};
    ember_sse_keepalive(&ka);
    int rc = ember_send_all(g->fd, ka.ptr, ka.len);
    ember_buf_free(&ka);
    if (rc == 0) return true;
    g->disconnected = true;
    return false;
}

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

// A complete DSML opener copied from the rendered tool preamble is not a model
// decision to execute a tool. The backend's full prompt-echo watchdog needs a
// conservative 512-token proof; at the earlier think-tool recovery boundary a
// 32-token exact suffix match is already enough to fail closed. This is only a
// suppression signal: generation keeps running so the authoritative watchdog
// can report prompt_echo_detected if the copy continues.
static bool generated_suffix_matches_prompt(const gen_ctx *g) {
    enum { MIN_PROMPT_LIKE_SUFFIX = 32 };
    if (!g || !g->prompt_ids || g->n_prompt_ids < MIN_PROMPT_LIKE_SUFFIX ||
        g->n_gen_ids < MIN_PROMPT_LIKE_SUFFIX)
        return false;
    const int span = MIN_PROMPT_LIKE_SUFFIX;
    const int32_t *suffix = g->gen_ids + g->n_gen_ids - span;
    for (int i = 0; i <= g->n_prompt_ids - span; ++i)
        if (memcmp(g->prompt_ids + i, suffix,
                   (size_t)span * sizeof(*suffix)) == 0)
            return true;
    return false;
}

static bool on_token(int32_t tok, void *ud) {
    gen_ctx *g = (gen_ctx *)ud;
    if (g->disconnected) return false;
    // Don't wait for a write to fail to learn the client left: tokens whose
    // deltas the encoder withholds produce no write at all, so a generation
    // could run to completion into a dead socket, holding the only slot.
    // Deliberately NOT gated on !collect_only. A non-streaming request writes
    // nothing until the very end, so a failed write can never detect its client
    // — poll() is the only signal it has. Gating this on the streaming path
    // (an earlier version of this fix did) left non-streaming callers able to
    // orphan a generation for the full duration of the request.
    if (ember_client_gone(g->fd)) {
        g->disconnected = true;
        return false;
    }
    // Tool-bearing native streams and hidden recovery attempts intentionally
    // buffer model bytes until validation. Keep their connections active while
    // the model is decoding even though there are no user-visible deltas yet.
    if (g->collect_only && g->keepalive_while_collecting &&
        !send_generation_keepalive(g))
        return false;
    if (tok == ember_backend_eos_id(g->be)) return true;
    const char *piece = ember_backend_token_text(g->be, tok);
    if (g_trace_tokens) {
        const size_t piece_len = strlen(piece);
        fprintf(stderr, "[ember-token] index=%d id=%d bytes=",
                g->n_gen_ids, tok);
        for (size_t i = 0; i < piece_len; ++i)
            fprintf(stderr, "%02x", (unsigned char)piece[i]);
        fprintf(stderr, "\n");
    }
    ember_buf_puts(&g->acc, piece);
    // B3 L2: remember committed token ids so a post-tool-call snapshot can be
    // keyed by the exact [prompt + generated] token sequence.
    if (g->n_gen_ids == g->gen_cap) {
        if (g->gen_cap > INT_MAX / 2)
            ember_buf_fatal("generated-token capture overflow");
        int next = g->gen_cap ? g->gen_cap * 2 : 256;
        int32_t *grown = (int32_t *)realloc(
            g->gen_ids, (size_t)next * sizeof(int32_t));
        if (!grown) ember_buf_fatal("out of memory capturing generated tokens");
        g->gen_ids = grown;
        g->gen_cap = next;
    }
    g->gen_ids[g->n_gen_ids++] = tok;
    // B6: advance the DSML decode-state tracker over the accumulated text so the
    // NEXT token's sampling knows whether it lands on tool-call structure (greedy)
    // or payload (sampled). Runs on the worker thread; force_greedy reads it there.
    if (g->dsml_active && g->acc.ptr)
        ember_dsml_tracker_update(&g->dsml, g->acc.ptr, g->acc.len);
    // ds4_server.c:11516-11555 recovers as soon as a complete tool stanza
    // opener appears inside reasoning. Stop this backend-owned eval loop at the
    // same boundary; run_chat then force-feeds only </think> and resumes the
    // same assistant turn. Waiting for EOS/max_tokens here lets the malformed
    // in-think stanza consume the whole turn before recovery can begin.
    if (g->has_tools && g->acc.ptr &&
        inside_unclosed_think(g->acc.ptr, g->started_thinking) &&
        ember_find_tool_start(g->acc.ptr) != NULL) {
        if (!g->think_tool_recovery_suppressed &&
            generated_suffix_matches_prompt(g)) {
            g->think_tool_recovery_suppressed = true;
            fprintf(stderr,
                    "[ember] suppressed in-think tool recovery: generated "
                    "DSML overlaps the rendered prompt exactly\n");
        }
        if (!g->think_tool_recovery_suppressed) return false;
    }
    // Stop sequences take precedence over tool markers when both become
    // complete in one decoded token. Otherwise a configured stop before a tool
    // block can be bypassed and the later tool executed.
    if (g->n_stops > 0 && g->acc.ptr &&
        !inside_unclosed_think(g->acc.ptr, g->started_thinking)) {
        const char *vis = visible_after_think(g->acc.ptr);
        char *earliest = NULL;
        const char *matched = NULL;
        for (int si = 0; si < g->n_stops; si++) {
            if (!g->stops[si] || !g->stops[si][0]) continue;
            char *h = strstr((char *)vis, g->stops[si]);
            if (h && (!earliest || h < earliest)) {
                earliest = h;
                matched = g->stops[si];
            }
        }
        if (earliest) {
            g->acc.len = (size_t)(earliest - g->acc.ptr);
            g->acc.ptr[g->acc.len] = '\0';
            g->hit_stop = true;
            g->hit_stop_sequence = matched;
            return false;
        }
    }
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
                if (g->scratch.len &&
                    ember_send_all(g->fd, g->scratch.ptr, g->scratch.len) != 0) {
                    g->disconnected = true;
                    return false;
                }
            }
            return false;
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
    // A cold 63k-token prefill is the longest silent window in the server and
    // exactly where clients time out. The keepalive write is a poor detector
    // here: the FIRST send after the peer's FIN still succeeds (it only draws
    // the RST that fails the SECOND), so a write-only check needs two 4s cycles
    // and misses the hangup entirely if the write is throttled away. poll()
    // sees the FIN on the next callback.
    if (ember_client_gone(g->fd)) {
        g->disconnected = true;
        return false;
    }
    // Non-streaming requests install this callback for the liveness check ONLY.
    // The response is a single JSON body, so emitting an SSE keepalive comment
    // into it would corrupt it — check and return before any write.
    if (g->collect_only && !g->keepalive_while_collecting) return true;
    // Throttle keepalives to ~4s regardless of how often the backend calls us
    // (ds4 caps the `: prefill` cadence so a per-chunk backend can't spam).
    return send_generation_keepalive(g);
}

// B6: consulted by the backend before sampling each token — force greedy argmax
// when the generation frontier is tool-call DSML/JSON structure (see dsml_decode).
static bool gen_force_greedy(void *ud) {
    gen_ctx *g = (gen_ctx *)ud;
    return g->dsml_active && ember_dsml_force_greedy(&g->dsml);
}

// Mint a process-independent 128-bit id. Persisted continuation keys must not
// collide when the server restarts and its in-process request counter resets.
static void mint_prefixed_id(const char *prefix, char *out, size_t cap) {
    unsigned char r[16];
    int f = open("/dev/urandom", O_RDONLY);
    if (f < 0 || read(f, r, 16) != 16) {
        static _Atomic unsigned long fallback_seq;
        unsigned long s = (unsigned long)time(NULL) ^
                          ((unsigned long)getpid() << 20) ^
                          atomic_fetch_add(&fallback_seq, 1);
        for (int i = 0; i < 16; i++) { s = s * 6364136223846793005UL + 1442695040888963407UL; r[i] = (unsigned char)(s >> 33); }
    }
    if (f >= 0) close(f);
    static const char hx[] = "0123456789abcdef";
    size_t p = strlen(prefix);
    if (cap < p + 33) {
        if (cap) out[0] = '\0';
        return;
    }
    memcpy(out, prefix, p);
    for (int i = 0; i < 16; i++) {
        size_t off = (size_t)i * 2;
        out[p + off] = hx[r[i] >> 4];
        out[p + off + 1] = hx[r[i] & 15];
    }
    out[p + 32] = '\0';
}

static void mint_tool_id(char *out) {
    mint_prefixed_id("call_", out, 40);
}

// A budget-forced turn contains a server-authored directive immediately before
// </think>. It is useful while decoding, but it is not model reasoning and must
// never become durable conversation history. attach_tool_memory() also strips
// the exact suffix from client histories produced by an older Ember deployment.
static size_t forced_close_reasoning_hint_len(const ember_server *srv) {
    const char *hint =
        srv && srv->card.thinking_terminator_hint
            ? srv->card.thinking_terminator_hint : NULL;
    if (!hint || !hint[0]) return 0;
    const char *close = strstr(hint, "</think>");
    return close ? (size_t)(close - hint) : strlen(hint);
}

static bool has_forced_close_hint(const ember_server *srv, const char *text) {
    size_t n = forced_close_reasoning_hint_len(srv);
    const char *hint =
        srv && srv->card.thinking_terminator_hint
            ? srv->card.thinking_terminator_hint : NULL;
    if (!text || !hint || n == 0) return false;
    size_t len = strlen(text);
    return len >= n && memcmp(text + len - n, hint, n) == 0;
}

static void strip_forced_close_hint(const ember_server *srv, char *reasoning,
                                    bool budget_forced_close) {
    if (!budget_forced_close || !reasoning) return;
    size_t n = forced_close_reasoning_hint_len(srv);
    size_t len = strlen(reasoning);
    if (n > 0 && len >= n && has_forced_close_hint(srv, reasoning))
        reasoning[len - n] = '\0';
}

// B3: before rendering, substitute assistant DSML blocks whose ids we stored.
// Reasoning/content remain canonical; only the sampled protocol block is exact,
// matching Dwarfstar's scope. When its byte boundaries aligned to whole sampled
// tokens, emit a splice sentinel ("\x1f id \x1f"): encode_with_splices() then
// inserts those ids rather than re-tokenizing special DSML markers. Otherwise
// fall back to the exact block bytes — correct rendering, with a possible cache
// miss only.
// Verify the mapped block names every function the client claims; else fall back
// to the canonical render. Runs on the worker (same thread as capture) → no lock.
static void attach_tool_memory(ember_server *srv, ember_chat_request *req) {
    pthread_mutex_lock(&srv->state_lock);
    for (int i = 0; i < req->n_messages; i++) {
        ember_chat_msg *m = &req->messages[i];
        if (strcmp(m->role, "assistant") != 0 || m->calls.len == 0 || m->raw_tool_text)
            continue;
        // Strip control text leaked by an older Ember deployment. Version-2
        // replay entries contain DSML only, so the canonical reasoning surface
        // remains authoritative in both thinking modes.
        if (m->reasoning &&
            has_forced_close_hint(srv, m->reasoning)) {
            size_t n = forced_close_reasoning_hint_len(srv);
            size_t len = strlen(m->reasoning);
            if (n <= len) m->reasoning[len - n] = '\0';
        }
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
        // The client controls the echoed id, name, and arguments. Only replay a
        // server-authored block when it parses to the same ordered calls.
        if (ok && raw && !ember_tool_calls_match_raw(raw, &m->calls)) ok = false;
        if (!ok || !raw) continue;
        // All calls in one assistant turn share the same sampled DSML block;
        // splice_id addresses that block's exact token stream.
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
    pthread_mutex_unlock(&srv->state_lock);
}

static int continuation_request_ids(const ember_chat_request *req,
                                    const char **ids, int cap) {
    if (!req || !req->continuation_only || req->n_messages <= 0) return 0;
    int n = 0;
    for (int i = 0; i < req->n_messages; ++i) {
        const ember_chat_msg *m = &req->messages[i];
        if ((strcmp(m->role, "tool") != 0 &&
             strcmp(m->role, "function") != 0) ||
            !m->tool_call_id || !m->tool_call_id[0])
            return 0;
        bool duplicate = false;
        for (int j = 0; j < n; ++j)
            if (strcmp(ids[j], m->tool_call_id) == 0) duplicate = true;
        if (!duplicate) {
            if (n >= cap) return 0;
            ids[n++] = m->tool_call_id;
        }
    }
    return n;
}

static char *continuation_ids_signature(const char *const *ids, int n) {
    ember_buf b = {0};
    for (int i = 0; i < n; ++i) {
        if (i) ember_buf_putc(&b, '\n');
        ember_buf_puts(&b, ids[i]);
    }
    char *s = ember_buf_take(&b);
    return s ? s : strdup("");
}

static bool generated_frontier_matches_text(ember_backend *be,
                                            const gen_ctx *g) {
    if (!be || !g || g->n_gen_ids < 0 || !g->acc.ptr) return false;
    ember_buf decoded = {0};
    for (int i = 0; i < g->n_gen_ids; ++i) {
        const char *piece = ember_backend_token_text(be, g->gen_ids[i]);
        if (piece) ember_buf_puts(&decoded, piece);
    }
    bool same = decoded.len == g->acc.len &&
        (decoded.len == 0 || !memcmp(decoded.ptr, g->acc.ptr, decoded.len));
    ember_buf_free(&decoded);
    return same;
}

static bool continuation_signature_matches(
        const char *signature, const char *const *ids, int n) {
    if (!signature) return false;
    int lines = signature[0] ? 1 : 0;
    for (const char *p = signature; *p; ++p)
        if (*p == '\n') lines++;
    if (lines != n) return false;
    for (int i = 0; i < n; ++i) {
        size_t len = strlen(ids[i]);
        bool found = false;
        for (const char *p = signature; p && *p;) {
            const char *end = strchr(p, '\n');
            size_t line_len = end ? (size_t)(end - p) : strlen(p);
            if (line_len == len && memcmp(p, ids[i], len) == 0) {
                found = true;
                break;
            }
            p = end ? end + 1 : NULL;
        }
        if (!found) return false;
    }
    return true;
}

// Resolve an ID-only tool-result request to the authoritative sampled frontier,
// then append a separately-tokenized suffix. This deliberately does not slice
// tokens from a full re-render: BPE may merge across the text boundary.
// Returns >=0 token count, -1 on allocation/tokenizer failure, or -2 when the
// continuation binding is missing/stale.
static int build_bound_continuation_prompt(ember_server *srv, ember_backend *be,
                                           const ember_chat_request *req,
                                           int32_t **out_ids,
                                           bool *started_thinking) {
    int call_cap = req && req->n_messages > 0 ? req->n_messages : 0;
    const char **call_ids = call_cap > 0
        ? calloc((size_t)call_cap, sizeof(*call_ids)) : NULL;
    if (!call_ids) return -2;
    int n_calls = continuation_request_ids(req, call_ids, call_cap);
    if (n_calls <= 0) { free(call_ids); return -2; }
    pthread_mutex_lock(&srv->state_lock);
    const ember_continuation_entry *e = NULL;
    if (req->continuation_key) {
        const char *key[] = {req->continuation_key};
        e = ember_continuation_get(&srv->continuations, req->api, key, 1);
        if (e && !continuation_signature_matches(
                     e->visible_text, call_ids, n_calls))
            e = NULL;
    } else {
        e = ember_continuation_get(&srv->continuations, req->api,
                                   call_ids, n_calls);
    }
    if (!e) {
        pthread_mutex_unlock(&srv->state_lock);
        free(call_ids);
        return -2;
    }
    // Omitting tools is valid: the authoritative prefix already contains the
    // schemas. Supplying a different schema block would silently continue under
    // the wrong contract, so reject it.
    if (req->tools_json &&
        (!e->tools_json || strcmp(req->tools_json, e->tools_json) != 0))
        {
            pthread_mutex_unlock(&srv->state_lock);
            free(call_ids);
            return -2;
        }

    char *suffix =
        ember_render_tool_continuation_suffix(req, req->thinking_enabled);
    if (!suffix) {
        pthread_mutex_unlock(&srv->state_lock);
        free(call_ids);
        return -1;
    }
    int32_t *suffix_ids = NULL;
    int n_suffix = ember_backend_encode(be, suffix, &suffix_ids);
    if (started_thinking)
        *started_thinking = ember_prompt_ends_in_open_think(suffix);
    free(suffix);
    if (n_suffix < 0 || e->n_frontier > INT_MAX - n_suffix) {
        free(suffix_ids);
        pthread_mutex_unlock(&srv->state_lock);
        free(call_ids);
        return -1;
    }
    int total = e->n_frontier + n_suffix;
    int32_t *ids =
        (int32_t *)malloc((size_t)(total > 0 ? total : 1) * sizeof(int32_t));
    if (!ids) {
        free(suffix_ids);
        pthread_mutex_unlock(&srv->state_lock);
        free(call_ids);
        return -1;
    }
    memcpy(ids, e->frontier_ids,
           (size_t)e->n_frontier * sizeof(int32_t));
    if (n_suffix > 0)
        memcpy(ids + e->n_frontier, suffix_ids,
               (size_t)n_suffix * sizeof(int32_t));
    free(suffix_ids);
    *out_ids = ids;
    pthread_mutex_unlock(&srv->state_lock);
    free(call_ids);
    return total;
}

static void remember_continuation(ember_server *srv,
                                  const ember_chat_request *req,
                                  const int32_t *prompt_ids, int n_prompt,
                                  const gen_ctx *g,
                                  const char *const *call_ids, int n_call_ids,
                                  const char *response_id) {
    if (!srv || !req || !g || !prompt_ids || n_prompt <= 0 ||
        !call_ids || n_call_ids <= 0 || g->n_gen_ids <= 0)
        return;
    if (!generated_frontier_matches_text(srv->be, g)) {
        fprintf(stderr,
                "[ember] continuation skipped: sampled token frontier does "
                "not exactly detokenize to stored assistant bytes\n");
        return;
    }
    if (n_prompt > INT_MAX - g->n_gen_ids) return;
    int total = n_prompt + g->n_gen_ids;
    int32_t *frontier =
        (int32_t *)malloc((size_t)total * sizeof(int32_t));
    if (!frontier) return;
    memcpy(frontier, prompt_ids, (size_t)n_prompt * sizeof(int32_t));
    memcpy(frontier + n_prompt, g->gen_ids,
           (size_t)g->n_gen_ids * sizeof(int32_t));
    pthread_mutex_lock(&srv->state_lock);
    (void)ember_continuation_put(&srv->continuations, req->api,
                                 call_ids, n_call_ids, frontier, total,
                                 NULL, req->tools_json);
    if (req->api == EMBER_API_RESPONSES &&
        response_id && response_id[0]) {
        const char *key[] = {response_id};
        char *signature =
            continuation_ids_signature(call_ids, n_call_ids);
        (void)ember_continuation_put(
            &srv->continuations, req->api, key, 1, frontier, total,
            signature, req->tools_json);
        free(signature);
    }
    pthread_mutex_unlock(&srv->state_lock);
    free(frontier);
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
    if (!generated_frontier_matches_text(be, g)) {
        fprintf(stderr,
                "[ember] post-tool snapshot skipped: sampled token frontier "
                "does not exactly detokenize to assistant bytes\n");
        return;
    }
    // Tool-like text before the final </think> is reasoning, including the
    // harmless dangling stanza that ds4-style think recovery leaves behind.
    const char *st = ember_find_tool_start(visible_after_think(g->acc.ptr));
    if (!st || !ember_find_tool_end(st)) {
        fprintf(stderr,
                "[ember] post-tool snapshot skipped: generated response has "
                "no complete DSML block\n");
        return;
    }
    ember_tool_calls parsed = {0};
    ember_parse_dsml_tool_calls(st, &parsed);
    bool executable = parsed.len > 0;
    ember_tool_calls_free(&parsed);
    if (!executable) {
        fprintf(stderr,
                "[ember] post-tool snapshot skipped: DSML block is not "
                "executable\n");
        return;
    }
    pthread_mutex_lock(&srv->state_lock);
    int slot = ember_kv_reserve(&srv->kv);
    pthread_mutex_unlock(&srv->state_lock);
    if (slot < 0) {
        fprintf(stderr,
                "[ember] post-tool snapshot skipped: no unpinned prefix "
                "slot (prompt=%d generated=%d)\n",
                n_prompt, g->n_gen_ids);
        return;
    }
    if (!ember_backend_snapshot_now(be, slot)) {
        fprintf(stderr,
                "[ember] post-tool snapshot failed: backend rejected slot=%d "
                "at prompt=%d generated=%d\n",
                slot, n_prompt, g->n_gen_ids);
        pthread_mutex_lock(&srv->state_lock);
        ember_kv_release(&srv->kv, slot);
        pthread_mutex_unlock(&srv->state_lock);
        return;
    }
    // Key by the KV the snapshot actually holds, NOT by what we emitted. Decode
    // writes a token's KV row at the start of the FOLLOWING step, so a snapshot
    // parked after generation is normally one row short of [prompt + generated].
    // Keying it with the emitted count made every tool-call turn log
    // "refusing mismatched checkpoint: snapshot_pos=N tokens=N+1" and disabled
    // the disk cache outright — each large-context turn then cold-prefilled ~63k
    // tokens, went silent long enough to trip the client's stream watchdog, and
    // left an orphaned generation holding the only slot. Restore already resumes
    // from the snapshot's own cur_pos, so a key trimmed to `covered` is exact and
    // costs at most the one re-fed token.
    int emitted = n_prompt + g->n_gen_ids;
    int total   = ember_backend_snapshot_pos(be, slot);
    if (total > emitted) total = emitted;  // speculative decode can park phantom rows
    if (total <= n_prompt) {
        fprintf(stderr,
                "[ember] post-tool snapshot failed: frontier did not advance "
                "(slot=%d snapshot=%d prompt=%d emitted=%d)\n",
                slot, total, n_prompt, emitted);
        pthread_mutex_lock(&srv->state_lock);
        ember_kv_release(&srv->kv, slot);
        pthread_mutex_unlock(&srv->state_lock);
        return;
    }
    int32_t *seq = (int32_t *)malloc((size_t)total * sizeof(int32_t));
    if (!seq) {
        pthread_mutex_lock(&srv->state_lock);
        ember_kv_release(&srv->kv, slot);
        pthread_mutex_unlock(&srv->state_lock);
        return;
    }
    memcpy(seq, prompt_ids, (size_t)n_prompt * sizeof(int32_t));
    memcpy(seq + n_prompt, g->gen_ids, (size_t)(total - n_prompt) * sizeof(int32_t));
    // A post-tool-call snapshot is a routine within-conversation waypoint.
    bool disk_saved = true;
    if (ember_backend_disk_enabled(be)) {
        disk_saved = ember_backend_disk_save(
            be, slot, seq, total, EMBER_KV_SAVE_NORMAL);
        if (!disk_saved) {
            fprintf(stderr,
                    "[ember] post-tool disk snapshot failed: slot=%d "
                    "frontier=%d\n",
                    slot, total);
        }
    }
    pthread_mutex_lock(&srv->state_lock);
    bool committed = ember_kv_commit(&srv->kv, slot, seq, total);
    if (!committed)
        ember_kv_release(&srv->kv, slot);
    pthread_mutex_unlock(&srv->state_lock);
    if (committed) {
        fprintf(stderr,
                "[ember] post-tool snapshot committed: slot=%d frontier=%d "
                "disk=%s\n",
                slot, total, disk_saved ? "saved" : "failed");
    } else {
        fprintf(stderr,
                "[ember] post-tool snapshot failed: logical commit rejected "
                "slot=%d frontier=%d\n",
                slot, total);
    }
    free(seq);
}

static void finish_prompt_snapshot(ember_server *srv, ember_backend *be,
                                   int slot, const int32_t *ids, int cut,
                                   bool saved, int reason) {
    if (slot < 0) return;
    // Keep the physical slot reserved until any durable copy has completed.
    // Only then publish the logical prefix, so another session cannot restore
    // or overwrite a half-finished checkpoint.
    if (saved && ember_backend_disk_enabled(be))
        (void)ember_backend_disk_save(be, slot, ids, cut, reason);
    pthread_mutex_lock(&srv->state_lock);
    if (!saved || !ember_kv_commit(&srv->kv, slot, ids, cut))
        ember_kv_release(&srv->kv, slot);
    pthread_mutex_unlock(&srv->state_lock);
}

static void remember_tool_block(ember_server *srv, ember_backend *be,
                                const char *id, const gen_ctx *g) {
    if (!srv || !be || !id || !g || !g->acc.ptr || g->n_gen_ids <= 0)
        return;
    const char *start = ember_find_tool_start(visible_after_think(g->acc.ptr));
    const char *end = start ? ember_find_tool_end(start) : NULL;
    if (!start || !end || end > g->acc.ptr + g->acc.len) return;
    // Native ds_engine output uses one sibling block per parallel call rather
    // than one wrapper around the complete call set. All ids from one assistant
    // turn must still resolve to the same exact sampled surface, otherwise a
    // later multi-call replay compares call 2+ against a copy of call 1 only.
    static const char native_open[] = "<ds_engine_tool_use>";
    if (!strncmp(start, native_open, sizeof(native_open) - 1)) {
        const char *cursor = end;
        for (;;) {
            while (cursor < g->acc.ptr + g->acc.len &&
                   isspace((unsigned char)*cursor))
                ++cursor;
            if (cursor >= g->acc.ptr + g->acc.len ||
                strncmp(cursor, native_open, sizeof(native_open) - 1) != 0)
                break;
            const char *next_end = ember_find_tool_end(cursor);
            if (!next_end || next_end > g->acc.ptr + g->acc.len) return;
            end = next_end;
            cursor = next_end;
        }
    }

    // Map the DSML byte span back to whole sampled tokens. If either boundary
    // lands inside a token, retain the exact bytes but let replay re-tokenize
    // them; using a partial token id would corrupt the rendered prompt.
    size_t start_off = (size_t)(start - g->acc.ptr);
    size_t end_off = (size_t)(end - g->acc.ptr);
    size_t off = 0;
    int first = -1, after = -1;
    for (int i = 0; i < g->n_gen_ids; ++i) {
        if (off == start_off && first < 0) first = i;
        const char *piece = ember_backend_token_text(be, g->gen_ids[i]);
        size_t piece_len = piece ? strlen(piece) : 0;
        if (piece_len > SIZE_MAX - off) break;
        off += piece_len;
        if (off == end_off) {
            after = i + 1;
            break;
        }
        if (off > end_off) break;
    }
    const int32_t *tokens = NULL;
    int n_tokens = 0;
    if (first >= 0 && after > first) {
        tokens = g->gen_ids + first;
        n_tokens = after - first;
    }
    pthread_mutex_lock(&srv->state_lock);
    (void)ember_tool_memory_put(&srv->tool_mem, id, start,
                                (size_t)(end - start),
                                tokens, n_tokens);
    pthread_mutex_unlock(&srv->state_lock);
}

static void persist_atomic_tool_frontier(
        ember_server *srv, ember_backend *be, const ember_chat_request *req,
        const int32_t *prompt_ids, int n_prompt, gen_ctx *g,
        ember_tool_calls *calls, const char *response_id) {
    if (!srv || !be || !req || !g || !calls || calls->len <= 0 ||
        g->disconnected)
        return;
    snapshot_post_toolcall(srv, be, prompt_ids, n_prompt, g);
    for (int i = 0; i < calls->len; ++i)
        remember_tool_block(srv, be, calls->calls[i].id, g);
    size_t n_calls = (size_t)(unsigned int)calls->len;
    const char **call_ids = calloc(n_calls, sizeof(*call_ids));
    if (!call_ids)
        ember_buf_fatal("out of memory storing continuation tool ids");
    for (int i = 0; i < calls->len; ++i)
        call_ids[i] = calls->calls[i].id;
    remember_continuation(srv, req, prompt_ids, n_prompt, g,
                          call_ids, calls->len, response_id);
    free(call_ids);
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

static bool ids_append(int32_t **ids, int *n, int *cap,
                       const int32_t *src, int k) {
    if (k <= 0) return true;
    if (!src || *n < 0 || k > INT_MAX - *n) return false;
    if (*n + k > *cap) {
        int need = *n + k;
        int next = need > INT_MAX / 2 ? need : need * 2;
        int32_t *grown = (int32_t *)realloc(
            *ids, (size_t)next * sizeof(int32_t));
        if (!grown) return false;
        *ids = grown;
        *cap = next;
    }
    memcpy(*ids + *n, src, (size_t)k * sizeof(int32_t));
    *n += k;
    return true;
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
    if (!srv || !be || !prompt || !out_ids) return -1;
    *out_ids = NULL;
    if (!strchr(prompt, '\x1f'))                     // fast path: nothing spliced
        return ember_backend_encode(be, prompt, out_ids);

    pthread_mutex_lock(&srv->state_lock);
    int32_t *ids = NULL; int n = 0, cap = 0;
    const char *p = prompt;
    while (*p) {
        const char *op = strchr(p, '\x1f');
        const char *text_end = op ? op : (p + strlen(p));
        if (text_end > p) {                          // text before the sentinel
            char *t = (char *)malloc((size_t)(text_end - p) + 1);
            if (!t) {
                free(ids);
                pthread_mutex_unlock(&srv->state_lock);
                return -1;
            }
            memcpy(t, p, (size_t)(text_end - p)); t[text_end - p] = '\0';
            int32_t *tids = NULL;
            int tn = ember_backend_encode(be, t, &tids);
            free(t);
            if (tn < 0) {
                free(tids);
                free(ids);
                pthread_mutex_unlock(&srv->state_lock);
                return -1;
            }
            if (!ids_append(&ids, &n, &cap, tids, tn)) {
                free(tids);
                free(ids);
                pthread_mutex_unlock(&srv->state_lock);
                return -1;
            }
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
                if (!ids_append(&ids, &n, &cap, gids, gn)) {
                    free(ids);
                    pthread_mutex_unlock(&srv->state_lock);
                    return -1;
                }
            } else {
                // Defensive: tokens evicted between requests. Re-tokenize the
                // stored bytes so the turn is still rendered correctly — just
                // without exact-token continuation.
                const char *bytes = ember_tool_memory_get(&srv->tool_mem, id);
                if (bytes) {
                    int32_t *bids = NULL;
                    int bn = ember_backend_encode(be, bytes, &bids);
                    if (bn > 0 &&
                        !ids_append(&ids, &n, &cap, bids, bn)) {
                        free(bids);
                        free(ids);
                        pthread_mutex_unlock(&srv->state_lock);
                        return -1;
                    }
                    free(bids);
                } else {
                    int32_t *literal_ids = NULL;
                    char *literal = strndup(
                        op, (size_t)(cl + 1 - op));
                    int literal_n = literal
                        ? ember_backend_encode(be, literal, &literal_ids)
                        : -1;
                    free(literal);
                    if (literal_n < 0 ||
                        !ids_append(
                            &ids, &n, &cap, literal_ids, literal_n)) {
                        free(literal_ids);
                        free(ids);
                        pthread_mutex_unlock(&srv->state_lock);
                        return -1;
                    }
                    free(literal_ids);
                }
            }
            p = cl + 1;
        } else {
            // A literal unit-separator is valid prompt text. Preserve it when
            // it does not form one of Ember's private splice sentinels.
            int32_t *literal_ids = NULL;
            int literal_n = ember_backend_encode(be, "\x1f", &literal_ids);
            if (literal_n < 0 ||
                !ids_append(&ids, &n, &cap, literal_ids, literal_n)) {
                free(literal_ids);
                free(ids);
                pthread_mutex_unlock(&srv->state_lock);
                return -1;
            }
            free(literal_ids);
            p = op + 1;
        }
    }
    *out_ids = ids;
    pthread_mutex_unlock(&srv->state_lock);
    return n;
}

// Parse generated tool output under an executable, not merely recoverable,
// policy. The low-level parser keeps Dwarfstar's missing-tail repair for
// compatibility and diagnostics; this gate prevents repaired, nested, or
// otherwise ambiguous calls from reaching a harness.
static _Thread_local char tool_validation_detail[512];

static bool tool_choice_allows_name(const ember_chat_request *req,
                                    const char *name) {
    if (!req || req->n_tool_choice_names == 0) return true;
    for (int i = 0; i < req->n_tool_choice_names; ++i)
        if (!strcmp(req->tool_choice_names[i], name ? name : "")) return true;
    return false;
}

static const char *validate_tool_call_schemas(
        const ember_chat_request *req, const ember_tool_calls *calls) {
    if (!calls || calls->len == 0) return NULL;
    if (!req || !req->has_tools || !req->tools_json)
        return "a tool call was generated when no tools were available";

    ember_json *tools = ember_json_parse(req->tools_json);
    if (!tools || tools->type != EMBER_JSON_ARRAY) {
        if (tools) ember_json_free(tools);
        // This is an invalid normalized request rather than a model mistake,
        // but it is still unsafe to execute a call against an unknown schema.
        return "the advertised tool schemas could not be validated";
    }
    if (ember_json_has_duplicate_keys(tools)) {
        ember_json_free(tools);
        return "the advertised tool schemas contained duplicate object keys";
    }

    const char *detail = NULL;
    for (int ci = 0; !detail && ci < calls->len; ++ci) {
        const ember_tool_call *call = &calls->calls[ci];
        if (!tool_choice_allows_name(req, call->name)) {
            detail = "the generated tool name was excluded by tool_choice";
            break;
        }
        const ember_json *matched_fn = NULL;
        for (int ti = 0; ti < ember_json_len(tools); ++ti) {
            const ember_json *tool = ember_json_at(tools, ti);
            const ember_json *fn = ember_json_get(tool, "function");
            const char *name =
                ember_json_str(ember_json_get(fn, "name"), NULL);
            if (name && call->name && strcmp(name, call->name) == 0) {
                matched_fn = fn;
                break;
            }
        }
        if (!matched_fn) {
            detail = "the generated tool name was not advertised";
            break;
        }

        ember_json *args = ember_json_parse(
            call->arguments ? call->arguments : "");
        if (!args || args->type != EMBER_JSON_OBJECT) {
            detail = "tool arguments were not a JSON object";
            if (args) ember_json_free(args);
            break;
        }
        if (ember_json_has_duplicate_keys(args)) {
            detail = "tool arguments contained duplicate object keys";
            ember_json_free(args);
            break;
        }
        const ember_json *params = ember_json_get(matched_fn, "parameters");
        if (!params) params = ember_json_get(matched_fn, "input_schema");
        const ember_json *strict_value = ember_json_get(matched_fn, "strict");
        if (strict_value && strict_value->type != EMBER_JSON_BOOL) {
            detail = "the advertised tool strict flag was not boolean";
            ember_json_free(args);
            break;
        }
        bool strict = ember_json_bool(strict_value, false);
        if (params) {
            tool_validation_detail[0] = '\0';
            if (!ember_tool_schema_validate(
                    args, params, params, strict, tool_validation_detail,
                    sizeof(tool_validation_detail))) {
                detail = tool_validation_detail[0]
                    ? tool_validation_detail
                    : "tool arguments did not satisfy the advertised schema";
            }
        }
        ember_json_free(args);
    }
    ember_json_free(tools);
    return detail;
}

static bool parse_executable_tool_calls(const ember_chat_request *req,
                                        const char *text,
                                        ember_tool_calls *out,
                                        const char **detail_out) {
    if (detail_out) *detail_out = NULL;
    ember_tool_parse_report report = {0};
    ember_parse_dsml_tool_calls_ex(text, out, &report);
    if (!report.found) {
        if (!req || !req->tool_choice_required)
            return true;  // ordinary assistant text, no required tool call
        if (detail_out)
            *detail_out = "tool_choice required at least one tool call";
        return false;
    }

    const char *detail = NULL;
    if (report.contaminated)
        detail = "nested DSML appeared inside a string tool argument";
    else if (report.invalid_json)
        detail = "a non-string tool argument was not valid JSON";
    else if (report.mixed_syntax)
        detail = "mixed DSML syntax families were generated";
    else if (report.malformed)
        detail = "nested or malformed tool-call tags were generated";
    else if (report.trailing)
        detail = "non-whitespace followed the tool-call wrapper";
    else if (report.repaired || !report.complete)
        detail = "the generated tool-call block was truncated";
    else if (out->len == 0)
        detail = "the generated tool-call block could not be parsed";
    else if (out->len != report.invocations)
        detail = "nested or mismatched tool invocations were generated";
    else if (req && !req->parallel_tool_calls && out->len > 1)
        detail = "parallel tool calls were disabled for this request";

    for (int i = 0; !detail && i < out->len; ++i) {
        ember_json *args =
            ember_json_parse(out->calls[i].arguments ?
                             out->calls[i].arguments : "");
        if (!args || args->type != EMBER_JSON_OBJECT)
            detail = "tool arguments were not a JSON object";
        if (args) ember_json_free(args);
    }
    if (!detail)
        detail = validate_tool_call_schemas(req, out);
    if (!detail) return true;

    ember_tool_calls_free(out);
    if (detail_out) *detail_out = detail;
    return false;
}

static bool tool_started_in_unclosed_think(const gen_ctx *g,
                                           bool started_thinking) {
    return g && g->has_tools && g->acc.ptr &&
           inside_unclosed_think(g->acc.ptr, started_thinking) &&
           ember_find_tool_start(g->acc.ptr) != NULL;
}

// Ember's backend owns the per-token eval loop, so on_token stops it at the
// opener but cannot inject a token directly the way ds4_server.c:10086-10149
// does. Preserve the same semantic boundary with one bounded continuation:
// append only "</think>\n\n" to the exact sampled frontier, then continue the
// SAME assistant turn. Unlike malformed-call recovery this opens no synthetic
// user/tool turn, resets no SSE state, and repeats no system prompt.
// Watchdog-stopped and prompt-overlapping output is excluded: echoed DSML
// documentation must never become executable merely because it appeared
// inside reasoning.
static bool continue_tool_started_in_think(
        ember_server *srv, ember_backend *be, const ember_chat_request *req,
        const ember_gen_request *base, const int32_t *prompt_ids, int n_prompt,
        bool started_thinking, gen_ctx *g, ember_gen_result *res) {
    if (!srv || !be || !req || !base || !prompt_ids || n_prompt < 0 ||
        !g || !res || !started_thinking || !res->ok ||
        res->degenerate_decode_close ||
        res->termination_reason[0] || g->disconnected ||
        g->think_tool_recovery_suppressed ||
        !tool_started_in_unclosed_think(g, started_thinking))
        return false;

    static const char inject[] = "</think>\n\n";
    int32_t *inject_ids = NULL;
    int n_inject = ember_backend_encode(be, inject, &inject_ids);
    if (n_inject <= 0) {
        free(inject_ids);
        return false;
    }
    // ds4 counts force-fed close tokens as completion tokens and against the
    // output limit: they are part of the assistant trajectory even though the
    // protocol adapter removes the marker from visible content.
    int remaining = base->max_tokens - res->n_generated - n_inject;
    if (remaining <= 0 ||
        n_prompt > INT_MAX - g->n_gen_ids ||
        n_prompt + g->n_gen_ids > INT_MAX - n_inject) {
        free(inject_ids);
        return false;
    }
    int combined_n = n_prompt + g->n_gen_ids + n_inject;
    int n_ctx = ember_backend_n_ctx(be);
    if (combined_n >= n_ctx) {
        free(inject_ids);
        return false;
    }
    if (remaining > n_ctx - combined_n) remaining = n_ctx - combined_n;
    if (remaining <= 0) {
        free(inject_ids);
        return false;
    }

    int32_t *combined =
        (int32_t *)malloc((size_t)combined_n * sizeof(*combined));
    if (!combined)
        ember_buf_fatal("out of memory building think-tool continuation");
    memcpy(combined, prompt_ids, (size_t)n_prompt * sizeof(*combined));
    memcpy(combined + n_prompt, g->gen_ids,
           (size_t)g->n_gen_ids * sizeof(*combined));
    memcpy(combined + n_prompt + g->n_gen_ids, inject_ids,
           (size_t)n_inject * sizeof(*combined));

    pthread_mutex_lock(&srv->state_lock);
    int recovery_slot = ember_kv_reserve(&srv->kv);
    pthread_mutex_unlock(&srv->state_lock);
    bool parked = recovery_slot >= 0 &&
                  ember_backend_snapshot_now(be, recovery_slot);
    if (recovery_slot >= 0 && !parked) {
        pthread_mutex_lock(&srv->state_lock);
        ember_kv_release(&srv->kv, recovery_slot);
        pthread_mutex_unlock(&srv->state_lock);
    }

    gen_ctx next = {0};
    next.be = be;
    next.st = g->st;
    next.fd = g->fd;
    next.collect_only = g->collect_only;
    next.keepalive_while_collecting = g->keepalive_while_collecting;
    next.last_ka = g->last_ka;
    next.has_tools = g->has_tools;
    next.stops = g->stops;
    next.n_stops = g->n_stops;
    next.started_thinking = started_thinking;
    next.prompt_ids = combined;
    next.n_prompt_ids = combined_n;
    ember_buf_append(&next.acc, g->acc.ptr, g->acc.len);
    ember_buf_puts(&next.acc, inject);
    next.gen_cap = g->n_gen_ids + n_inject;
    next.gen_ids = (int32_t *)malloc(
        (size_t)(next.gen_cap > 0 ? next.gen_cap : 1) * sizeof(int32_t));
    if (!next.gen_ids)
        ember_buf_fatal("out of memory capturing think-tool continuation");
    memcpy(next.gen_ids, g->gen_ids,
           (size_t)g->n_gen_ids * sizeof(int32_t));
    memcpy(next.gen_ids + g->n_gen_ids, inject_ids,
           (size_t)n_inject * sizeof(int32_t));
    next.n_gen_ids = next.gen_cap;
    free(inject_ids);

    next.dsml_active = base->temperature > 0.0f;
    if (next.dsml_active) {
        ember_dsml_tracker_init(&next.dsml);
        ember_dsml_tracker_update(&next.dsml, next.acc.ptr, next.acc.len);
    }

    // Flush any final reasoning bytes and let the injected close transition the
    // existing Chat SSE splitter to visible text. Native streams are already
    // collect-only and project the completed logical turn afterward.
    if (req->stream && !next.collect_only && next.st) {
        ember_sse_update(
            next.st, next.acc.ptr, next.acc.len, false, &next.scratch);
        if (next.scratch.len &&
            ember_send_all(next.fd, next.scratch.ptr, next.scratch.len) != 0)
            next.disconnected = true;
        next.scratch.len = 0;
        if (next.scratch.ptr) next.scratch.ptr[0] = '\0';
    }

    ember_gen_result first = *res;
    ember_gen_result second = {0};
    if (!next.disconnected) {
        ember_gen_request continuation = *base;
        continuation.prompt = combined;
        continuation.n_prompt = combined_n;
        continuation.max_tokens = remaining;
        continuation.restore_slot = parked ? recovery_slot : -1;
        continuation.snap_slot = -1;
        continuation.snap_pos = -1;
        continuation.on_token = on_token;
        continuation.on_prefill = on_prefill;
        continuation.ud = &next;
        continuation.force_greedy =
            next.dsml_active ? gen_force_greedy : NULL;
        continuation.fg_ud = next.dsml_active ? &next : NULL;
        // Thinking is now closed for this logical turn; a second budget hook
        // must not inject another close into the visible tool-call surface.
        continuation.budget_close_ids = NULL;
        continuation.n_budget_close = 0;
        continuation.reply_budget = 0;
        second = ember_backend_generate(be, &continuation);
    } else {
        second.ok = true;
        second.cancelled = true;
        snprintf(second.finish_reason, sizeof(second.finish_reason), "stop");
    }
    if (parked) {
        pthread_mutex_lock(&srv->state_lock);
        ember_kv_release(&srv->kv, recovery_slot);
        pthread_mutex_unlock(&srv->state_lock);
    }
    free(combined);

    second.n_generated += first.n_generated + n_inject;
    second.prefill_s += first.prefill_s;
    second.decode_s += first.decode_s;
    second.prefill_tokens += first.prefill_tokens;
    second.spec_cycles += first.spec_cycles;
    second.spec_provider_age_s += first.spec_provider_age_s;
    second.spec_provider_block_s += first.spec_provider_block_s;
    second.spec_head_s += first.spec_head_s;
    second.spec_verify_s += first.spec_verify_s;
    second.budget_forced_close =
        first.budget_forced_close || second.budget_forced_close;
    second.degenerate_decode_close =
        first.degenerate_decode_close || second.degenerate_decode_close;
    if (second.prefill_mode[0] &&
        strcmp(first.prefill_mode, second.prefill_mode) != 0)
        snprintf(second.prefill_mode, sizeof(second.prefill_mode), "mixed");
    snprintf(second.prefill_reason, sizeof(second.prefill_reason),
             "think_tool_recovery");

    ember_buf_free(&g->acc);
    free(g->gen_ids);
    ember_buf_free(&g->scratch);
    *g = next;
    *res = second;
    fprintf(stderr,
            "[ember] tool call started inside unclosed <think>; continued "
            "after an exact </think> injection (%d injected tokens)\n",
            n_inject);
    return true;
}

// Whether malformed-call recovery must be refused for this request.
//
// ds4_server.c:11671/11754 refuses it for every streaming request, but the
// reason it gives is that previously-emitted reasoning/content cannot be
// retracted — and that only bites once something renderable has gone out. A
// turn whose entire output was a single suppressed tool block has delivered
// nothing, so there is no turn boundary to destroy.
//
// Returning that turn empty is itself a failure mode: an agent that sees an
// empty response re-sends a byte-identical request, and a deterministic model
// reproduces the same malformed call. Deployment showed exactly that — four
// md5-identical requests in ninety seconds, each dropping a call missing a
// required property. Recovering in-stream breaks the cycle at the source.
//
// Set EMBER_STREAM_TOOL_RETRY=0 to restore ds4's blanket refusal.
static bool stream_tool_retry_enabled(void) {
    static _Thread_local int cached = -1;
    if (cached < 0) {
        const char *e = getenv("EMBER_STREAM_TOOL_RETRY");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

// A generation that finished cleanly, emitted no tool call and delivered no
// visible text is pure non-progress: the turn is spent and the agent advances
// nothing. The backend already flags the underlying conditions
// (degenerate_decode_close, empty_visible_output), but nothing surfaced them --
// no log line and no /status field -- so in deployment these turns were
// invisible unless someone parsed an individual response body.
//
// This can occur after context compaction when earlier tool results have been
// reduced to placeholders. Tool-loop signals cannot see the failure because
// there are no calls to compare.
//
// Diagnostic only, same contract as the tool-loop signals: it never changes
// finish_reason, never refuses a request, and keeps no cross-request state that
// detection reads back.
// The buffered counterpart of ember_sse_delivered_visible(): a turn has
// delivered something only if a NON-WHITESPACE byte reached the client. The two
// paths must agree, or the same turn is non-progress on one and fine on the
// other -- which is what a whitespace-only stub reply exposed.
static bool buffered_delivered_visible(const char *s) {
    for (; s && *s; ++s)
        if (*s != ' ' && *s != '\t' && *s != '\n' && *s != '\r') return true;
    return false;
}

static void note_nonprogress_turn(ember_server *srv,
                                  const ember_gen_result *res,
                                  bool had_tools, bool delivered_visible,
                                  const char *path) {
    if (!srv || !res || !res->ok || had_tools || delivered_visible) return;
    // A generation that never DECODED is not the model refusing to act -- it
    // is a request that asked for no tokens, stopped immediately, or was
    // abandoned. Requiring n_generated > 0 excludes those cases.
    if (res->n_generated <= 0) return;
    // ...and the model must have stopped of its OWN ACCORD. "length" means the
    // token cap cut it off mid-thought, which is a budget outcome, not a
    // refusal to act. Count only voluntary stops; token-budget exhaustion is a
    // different outcome and must not be classified as non-progress.
    if (strcmp(res->finish_reason, "stop") != 0) return;
    fprintf(stderr,
            "[ember] non-progress turn: %d completion tokens, no visible "
            "output, no tool call (%s%s%s, %s)\n",
            res->n_generated,
            res->degenerate_decode_close ? "degenerate" : "clean",
            res->empty_visible_output ? "+empty_visible" : "",
            res->termination_reason[0] ? "+watchdog" : "",
            path);
    pthread_mutex_lock(&srv->state_lock);
    srv->nonprogress_count++;
    srv->last_nonprogress_at = now_unix();
    srv->last_nonprogress_tokens = res->n_generated;
    srv->last_nonprogress_degenerate = res->degenerate_decode_close;
    pthread_mutex_unlock(&srv->state_lock);
}

// Wider than note_nonprogress_turn(): every degenerate decode the backend
// flags, including ones that DID produce output. That case was unsurfaced and
// it is the more damaging of the two.
//
// A degenerate compaction response can still contain visible output and thus
// evade the narrower non-progress signal. Recording all backend-flagged
// degeneracy makes that damaging case observable.
//
// The empty-output case is already named in the non-progress line, so only the
// with-output case logs here; the counter covers both.
//
// Diagnostic only: never changes finish_reason, never refuses a request.
static void note_degenerate_turn(ember_server *srv,
                                 const ember_gen_result *res,
                                 bool produced_output, const char *path) {
    if (!srv || !res || !res->ok || !res->degenerate_decode_close) return;
    if (produced_output)
        fprintf(stderr,
                "[ember] degenerate turn WITH output: %d completion tokens "
                "(%s, %s)\n",
                res->n_generated,
                res->termination_reason[0] ? res->termination_reason
                                           : "n-gram repetition",
                path);
    pthread_mutex_lock(&srv->state_lock);
    srv->degenerate_count++;
    srv->last_degenerate_at = now_unix();
    srv->last_degenerate_tokens = res->n_generated;
    srv->last_degenerate_had_output = produced_output;
    snprintf(srv->last_degenerate_reason, sizeof(srv->last_degenerate_reason),
             "%s", res->termination_reason);
    pthread_mutex_unlock(&srv->state_lock);
}

// The instruction that makes suppression comprehensible to the model.
//
// v1 of this feature removed the tools and said nothing. The model, mid-plan
// and conditioned by ~90 messages of tool calls, still intended to call one --
// so it improvised ASCII tool markup into VISIBLE output: <?DSML?tool_calls>,
// U+003F rather than U+FF5C. A call rendered as text may expose sensitive tool
// arguments, so removing tools must also explain the recovery to the model.
// Removing a capability does not discard the payload, it relocates it.
//
// Precedent for injecting a private instruction: compaction does exactly this
// (compaction.c builds an augmented view with a trailing user-role
// instruction). Role "user" matches that proven path for this template.
#define EMBER_AUTO_ANSWER_INSTRUCTION                                         \
    "[Automatic recovery] The tools are unavailable for this turn because "   \
    "the same call was repeated without making progress. Answer now, in "     \
    "plain prose, using only what is already in this conversation. Do NOT "   \
    "write tool-call markup as text -- it will not be executed and will be "  \
    "shown to the user verbatim. If something is genuinely unknown, say so "  \
    "and state what you would need to find it."

// Append it as a trailing message so the normal render path picks it up.
// strdup both fields: ember_chat_request_free() frees them.
static bool append_auto_answer_instruction(ember_chat_request *req) {
    if (!req || req->n_messages >= INT_MAX - 1) return false;
    ember_chat_msg *grown = (ember_chat_msg *)realloc(
        req->messages, (size_t)(req->n_messages + 1) * sizeof(*grown));
    if (!grown) return false;
    req->messages = grown;
    ember_chat_msg *m = &req->messages[req->n_messages];
    memset(m, 0, sizeof(*m));
    m->role = strdup("user");
    m->content = strdup(EMBER_AUTO_ANSWER_INSTRUCTION);
    if (!m->role || !m->content) { free(m->role); free(m->content); return false; }
    req->n_messages++;
    return true;
}

// Report tool markup that reached the client as text. Diagnostic only: the
// content has already been streamed by the time we can see all of it, so this
// makes the defect visible rather than preventing it. Blocking would need a
// decision about what to send instead, which is a separate change.
//
// Takes the VERDICT, not the text, because the two response paths know it in
// different ways: the stream records it at its emit site
// (ember_sse_delivered_tool_markup) and the buffered path holds the delivered
// string outright. Passing the caller's raw accumulator here is what made this
// counter fire on 66 of 98 deployment requests -- the accumulator contains the
// tool block that was correctly held back and parsed, never sent.
static void note_tool_markup_leak(ember_server *srv, bool leaked,
                                  const char *path) {
    if (!srv || !leaked) return;
    fprintf(stderr,
            "[ember] tool markup leaked into visible output (%s) -- the model "
            "wrote a call as text; it was NOT executed\n", path);
    pthread_mutex_lock(&srv->state_lock);
    srv->tool_markup_leak_count++;
    srv->last_tool_markup_leak_at = now_unix();
    pthread_mutex_unlock(&srv->state_lock);
}

static bool stream_retry_forbidden(const ember_chat_request *req,
                                   const gen_ctx *g) {
    if (!req->stream) return false;           // atomic: ds4's original path
    if (!stream_tool_retry_enabled()) return true;
    // No stream object means the bytes went somewhere this predicate cannot
    // reason about; refuse rather than guess.
    return !g->st || ember_sse_delivered_visible(g->st);
}


// Constrained tool-call decoding. Off by default: it changes what the model is
// able to emit, which is exactly the class of change this repo ships dark and
// enables in deployment explicitly (see CLAUDE.md). tool_schema.c still
// validates every call afterwards either way, so this is defence in depth
// A tool result is evidence that the previous action already happened, so the
// next decision is kept on the target's autoregressive path: speculative
// verification can change a near-tied token, and was once observed to re-emit
// an identical successful write_file call.
//
// That observation was never quantified. A 37,869-token differential against
// autoregressive (scripts/bench/tool_loop_differential.py, plus the engine's
// own --validate-prompt) found no divergence at all, while the rule withholds
// speculation from most turns of an agent workload. Setting this to 0 lifts it
// so the tool-result case can be measured directly, which is the one shape the
// differential cannot otherwise reach: eligible requests are forced to AR, so
// the comparison would be AR against AR.
//
// Default is unchanged. This exists to make the measurement possible, not to
// recommend turning the rule off.
static bool tool_result_forces_ar(void) {
    static _Thread_local int cached = -1;
    if (cached < 0) {
        const char *e = getenv("EMBER_TOOL_RESULT_AR");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

// rather than a replacement for the validator.
static bool tool_grammar_enabled(void) {
    static _Thread_local int cached = -1;
    if (cached < 0) {
        const char *e = getenv("EMBER_TOOL_GRAMMAR");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}

// Dwarfstar-style bounded malformed-tool recovery for ATOMIC requests. Keep the
// sampled invalid turn in model context, append a model-visible tool error, and
// decode one replacement turn. ds4_server.c:11671/11754 deliberately never does
// this inside an open stream: previously-emitted reasoning/content cannot be
// retracted, so grafting a hidden second assistant onto the same SSE response
// destroys the turn boundary. Watchdog/cancelled turns are also terminal and
// must not be reclassified by finding DSML documentation in their partial text.
static bool retry_malformed_tool_call(
        ember_server *srv, ember_backend *be, const ember_chat_request *req,
        ember_gen_request *base, int32_t **prompt_ids_io, int *n_prompt_io,
        bool *started_thinking_io, gen_ctx *g, ember_gen_result *res,
        int *hidden_tokens_out, bool *attempted_out) {
    if (attempted_out) *attempted_out = false;
    if (!srv || !be || !req || !base || !prompt_ids_io || !n_prompt_io ||
        !g || !res || !res->ok || stream_retry_forbidden(req, g) ||
        res->degenerate_decode_close || res->termination_reason[0] ||
        !req->has_tools ||
        !g->acc.ptr || g->n_gen_ids <= 0)
        return false;

    const char *scan = g->acc.ptr;
    if (*started_thinking_io) {
        const char *close = last_substr(scan, "</think>");
        // ds4_server.c:4792-4804: DSML in an unclosed reasoning block is not an
        // executable call. Its dedicated close-and-continue recovery owns that
        // case; malformed-call recovery must never reinterpret reasoning text.
        if (!close) return false;
        scan = close + 8;
    }
    if (!ember_find_tool_start(scan) && !req->tool_choice_required) return false;
    ember_tool_calls probe = {0};
    const char *parse_detail = NULL;
    bool malformed =
        !parse_executable_tool_calls(req, scan, &probe, &parse_detail);
    ember_tool_calls_free(&probe);
    if (!malformed) return false;

    bool close_thinking =
        inside_unclosed_think(g->acc.ptr, *started_thinking_io);
    char *suffix = ember_render_invalid_tool_recovery_suffix(
        req, close_thinking,
        parse_detail ? parse_detail :
                       "the generated block could not be parsed");
    if (!suffix) return false;
    int32_t *suffix_ids = NULL;
    int n_suffix = ember_backend_encode(be, suffix, &suffix_ids);
    free(suffix);
    if (n_suffix < 0 || *n_prompt_io > INT_MAX - g->n_gen_ids ||
        *n_prompt_io + g->n_gen_ids > INT_MAX - n_suffix) {
        free(suffix_ids);
        return false;
    }
    int new_n = *n_prompt_io + g->n_gen_ids + n_suffix;
    int n_ctx = ember_backend_n_ctx(be);
    // max_tokens bounds model output, not server-authored prompt tokens. The
    // suffix consumes context room but must neither starve the replacement nor
    // be reported/billed as completion tokens.
    int remaining = base->max_tokens - res->n_generated;
    if (new_n >= n_ctx || remaining <= 0) {
        free(suffix_ids);
        return false;
    }
    if (remaining > n_ctx - new_n) remaining = n_ctx - new_n;

    int32_t *combined =
        (int32_t *)malloc((size_t)new_n * sizeof(*combined));
    if (!combined) {
        free(suffix_ids);
        return false;
    }
    memcpy(combined, *prompt_ids_io,
           (size_t)*n_prompt_io * sizeof(*combined));
    memcpy(combined + *n_prompt_io, g->gen_ids,
           (size_t)g->n_gen_ids * sizeof(*combined));
    memcpy(combined + *n_prompt_io + g->n_gen_ids, suffix_ids,
           (size_t)n_suffix * sizeof(*combined));
    free(suffix_ids);

    // Park the current post-generation KV when the backend can do so. A failed
    // park only costs a full prefill; it never changes recovery correctness.
    pthread_mutex_lock(&srv->state_lock);
    int recovery_slot = ember_kv_reserve(&srv->kv);
    pthread_mutex_unlock(&srv->state_lock);
    bool parked = recovery_slot >= 0 &&
                  ember_backend_snapshot_now(be, recovery_slot);
    if (recovery_slot >= 0 && !parked) {
        pthread_mutex_lock(&srv->state_lock);
        ember_kv_release(&srv->kv, recovery_slot);
        pthread_mutex_unlock(&srv->state_lock);
    }

    gen_ctx next = {0};
    next.be = be;
    next.fd = g->fd;
    next.collect_only = true;
    next.keepalive_while_collecting = req->stream;
    next.has_tools = true;
    next.stops = req->stop;
    next.n_stops = req->n_stop;
    next.started_thinking = req->thinking_enabled;
    next.prompt_ids = combined;
    next.n_prompt_ids = new_n;
    next.dsml_active = base->temperature > 0.0f;
    if (next.dsml_active) ember_dsml_tracker_init(&next.dsml);

    ember_gen_request retry = *base;
    retry.prompt = combined;
    retry.n_prompt = new_n;
    retry.max_tokens = remaining;
    retry.restore_slot = parked ? recovery_slot : -1;
    retry.snap_slot = -1;
    retry.snap_pos = -1;
    retry.on_token = on_token;
    // Atomic recovery is still client-owned work. Poll liveness throughout its
    // potentially long prefill even though it emits no SSE keepalive.
    retry.on_prefill = on_prefill;
    retry.ud = &next;
    retry.force_greedy = next.dsml_active ? gen_force_greedy : NULL;
    retry.fg_ud = next.dsml_active ? &next : NULL;
    if (retry.reply_budget >= remaining) {
        retry.budget_close_ids = NULL;
        retry.n_budget_close = 0;
        retry.reply_budget = 0;
    }

    if (attempted_out) *attempted_out = true;
    ember_gen_result second = ember_backend_generate(be, &retry);
    // The replacement decode is still an ordinary assistant trajectory. Apply
    // the same ds4 live boundary if it starts its repaired call before closing
    // reasoning; otherwise on_token's early stop would leave a harmless but
    // unusable dangling opener and the atomic response could collapse to empty
    // content instead of completing the call.
    (void)continue_tool_started_in_think(
        srv, be, req, &retry, combined, new_n, req->thinking_enabled,
        &next, &second);
    if (parked) {
        pthread_mutex_lock(&srv->state_lock);
        ember_kv_release(&srv->kv, recovery_slot);
        pthread_mutex_unlock(&srv->state_lock);
    }
    if (!second.ok) {
        // Preserve the actual provider failure. Leaving the original successful
        // result in place misreports this as a 422 malformed model call.
        *res = second;
        ember_buf_free(&next.acc);
        free(next.gen_ids);
        ember_buf_free(&next.scratch);
        free(combined);
        return false;
    }

    int hidden = res->n_generated;
    // Preserve terminal safety provenance across the hidden retry. A clean
    // replacement must not make a force-closed/degenerate first turn eligible
    // for exact replay or hide that fact from the harness.
    second.budget_forced_close =
        res->budget_forced_close || second.budget_forced_close;
    second.degenerate_decode_close =
        res->degenerate_decode_close || second.degenerate_decode_close;
    if (!second.termination_reason[0] && res->termination_reason[0])
        snprintf(second.termination_reason,
                 sizeof(second.termination_reason), "%s",
                 res->termination_reason);
    // Timings describe all model work spent producing the response, including
    // the hidden failed attempt. Label mixed policies explicitly instead of
    // making operators infer them from an unexpectedly low aggregate rate.
    second.prefill_s += res->prefill_s;
    second.decode_s += res->decode_s;
    second.prefill_tokens += res->prefill_tokens;
    if (strcmp(res->prefill_mode, second.prefill_mode) != 0)
        snprintf(second.prefill_mode, sizeof(second.prefill_mode), "mixed");
    snprintf(second.prefill_reason, sizeof(second.prefill_reason),
             "malformed_retry");
    ember_buf_free(&g->acc);
    free(g->gen_ids);
    ember_buf_free(&g->scratch);
    *g = next;
    *res = second;
    free(*prompt_ids_io);
    *prompt_ids_io = combined;
    *n_prompt_io = new_n;
    *started_thinking_io = req->thinking_enabled;
    if (hidden_tokens_out) *hidden_tokens_out = hidden;
    const char *replacement_scan = next.acc.ptr ? next.acc.ptr : "";
    if (req->thinking_enabled) {
        const char *close = last_substr(replacement_scan, "</think>");
        replacement_scan = close ? close + 8 : "";
    }
    ember_tool_calls replacement_calls = {0};
    const char *replacement_detail = NULL;
    bool replacement_valid =
        !second.degenerate_decode_close &&
        parse_executable_tool_calls(
            req, replacement_scan, &replacement_calls, &replacement_detail);
    ember_tool_calls_free(&replacement_calls);
    if (replacement_valid) {
        fprintf(stderr,
                "[ember] malformed tool-call retry produced a valid "
                "replacement (%d hidden model tokens)\n", hidden);
    } else {
        fprintf(stderr,
                "[ember] malformed tool-call retry produced another invalid "
                "replacement (%d hidden model tokens; reason=%s; detail=%s)\n",
                hidden,
                second.termination_reason[0] ?
                    second.termination_reason : "none",
                replacement_detail ? replacement_detail : "none");
    }
    return true;
}

// Malformed tool calls on a streaming response degrade to assistant text and a
// normal stop, matching ds4 (ds4_server.c:5231-5241 "a malformed tool block is
// model output, not a server failure"). Ember previously ended such a turn at a
// typed error, which ds4 never does on any path: its chat stream suppresses and
// finishes "stop", and its Responses stream additionally flushes the suppressed
// bytes as text. The retry gate itself still matches ds4's `!stream` rule — it
// is only the fallback underneath that was missing.
//
// Set EMBER_STREAM_TOOL_ERROR=1 to restore the typed-error boundary without a
// rebuild.
static bool stream_tool_text_fallback(void) {
    static _Thread_local int cached = -1;
    if (cached < 0) {
        const char *e = getenv("EMBER_STREAM_TOOL_ERROR");
        cached = (e && e[0] == '1') ? 0 : 1;
    }
    return cached != 0;
}

// A progress-watchdog stop on a STREAMING response finishes normally instead of
// at a typed error, by exactly the reasoning above: degenerate output is model
// output, not a server failure, and a streaming client cannot retry.
//
// The error ember emits is explicit (type, code, retry_exhausted, and prose
// saying not to auto-continue) and the harness reads none of it: it branches
// only on whether a visible delta arrived first, so a reasoning-only stop is
// retried as a transport failure and a stop with visible content is "continued"
// as a length truncation. Neither is fixable from here — the harness is a
// separate codebase replaced on upgrade — but not handing it an error it will
// mishandle is. See test_tool_safety_server.py for both cases as assertions.
//
// The cost, stated honestly: the partial response reaches the user, including
// whatever the model was repeating. sse.c has already streamed those bytes when
// the watchdog fires, so they cannot be retracted. A reply that trails off beats
// a turn that dies — the same trade ds4 made at ds4_server.c:5231-5241.
//
// NON-STREAMING keeps the typed error: those clients can retry, which is why
// ds4 gates its equivalent on `!stream` and why the retry gate here does too.
//
// Set EMBER_STREAM_WATCHDOG_ERROR=1 to restore the typed-error boundary.
static bool stream_watchdog_text_fallback(void) {
    static _Thread_local int cached = -1;
    if (cached < 0) {
        const char *e = getenv("EMBER_STREAM_WATCHDOG_ERROR");
        cached = (e && e[0] == '1') ? 0 : 1;
    }
    return cached != 0;
}

// Forensics for the rejected block itself. Off by default: it prints model
// output to the journal, which normal operation must not do.
//
// The failure under investigation is a *complete* DSML block (the error frame
// reports partial_tool_call:false) whose invoke carries no parameters, produced
// with forced_close:false and degenerate:false -- so nothing on the server
// truncated it. Whether the model closed the invoke with no parameter at all,
// stopped part-way through writing one, or emitted malformed markup implies
// three different fixes, and only the raw bytes distinguish them.
static bool log_rejected_block(void) {
    static _Thread_local int cached = -1;
    if (cached < 0) {
        const char *e = getenv("EMBER_LOG_REJECTED_BLOCK");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}

static void dump_rejected_block(const char *raw, size_t raw_len,
                                size_t tool_start, const char *why) {
    if (!log_rejected_block() || !raw || tool_start >= raw_len) return;
    size_t n = raw_len - tool_start;
    if (n > 512) n = 512;   // bounded: enough for an invoke + its parameters
    fprintf(stderr, "[ember] rejected block (%s), %zu bytes from offset %zu:\n"
                    "%.*s\n[ember] end rejected block\n",
            why ? why : "invalid", raw_len - tool_start, tool_start,
            (int)n, raw + tool_start);
}

static void format_stream_tool_failure(char *dst, size_t cap,
                                       const char *detail,
                                       bool recovery_attempted) {
    snprintf(
        dst, cap,
        recovery_attempted
            ? "The model generated an unsafe or incomplete tool call after "
              "one model-visible recovery attempt; no tool call was emitted: %s"
            : "The model generated an unsafe or incomplete tool call; no "
              "automatic retry was attempted inside the streaming response "
              "and no tool call was emitted: %s",
        detail && detail[0] ? detail : "invalid generated tool call");
}

// Server-side DRY default, read once. EMBER_DRY_MULTIPLIER=0.8 is a reasonable
// starting point (llama.cpp's own suggested range is 0.5-1.0); 0 keeps it off.
// This exists because the loop DRY is meant to damp -- a model replaying the
// <think> content the template fed back to it -- happens on requests from a
// gateway that will never send a dry_* parameter of its own.
static bool parse_double_range(const char *s, const char *name,
                               double min, double max, double *out);

static float dry_default_multiplier(void) {
    // Batch sessions may enter request policy on different coordinator
    // threads. Per-thread lazy state avoids unsynchronized shared writes; the
    // process environment is fixed before any request thread starts.
    static _Thread_local float cached = -1.0f;
    if (cached >= 0.0f) return cached;
    const char *s = getenv("EMBER_DRY_MULTIPLIER");
    // parse_double_range, not atof: it validates and LOGS a bad value. atof
    // maps garbage to 0, and 0 is exactly "DRY off" -- so a mistyped flag would
    // be indistinguishable from deliberately disabling it, defeating the whole
    // point of the confirmation line below.
    double v = 0.0;
    if (s && s[0] &&
        !parse_double_range(s, "EMBER_DRY_MULTIPLIER", 0.0, 100.0, &v)) {
        v = 0.0;
    }
    cached = (float)v;
    if (cached > 0.0f)
        fprintf(stderr, "[ember] DRY sampler penalty enabled: multiplier=%.3g\n",
                (double)cached);
    return cached;
}

static bool generation_stalled(const ember_gen_result *res) {
    if (!res || !res->termination_reason[0]) return false;
    return !strcmp(res->termination_reason, "repetition_detected") ||
           !strcmp(res->termination_reason, "reasoning_cycle_detected") ||
           !strcmp(res->termination_reason, "prompt_echo_detected");
}

static void format_generation_stalled(char *dst, size_t cap,
                                      const ember_gen_result *res) {
    snprintf(dst, cap,
             "Generation stopped because Ember's progress watchdog detected "
             "%s. The partial response is unreliable and must not be "
             "auto-continued as an ordinary token-limit truncation.",
             res && res->termination_reason[0]
                 ? res->termination_reason : "a non-progressing decode");
}

static void respond_generation_stalled(
        int fd, const ember_chat_request *req, const ember_gen_result *res,
        int prompt_tokens, int completion_tokens) {
    char detail[384];
    format_generation_stalled(detail, sizeof(detail), res);
    const char *code = res && res->termination_reason[0]
        ? res->termination_reason : "generation_stalled";
    ember_buf e = {0};
    if (req && req->api == EMBER_API_ANTHROPIC) {
        ember_buf_puts(&e,
            "{\"type\":\"error\",\"error\":{\"type\":\"api_error\","
            "\"message\":");
        ember_json_escape(&e, detail);
        ember_buf_puts(&e, "}}");
    } else {
        ember_buf_puts(&e, "{\"error\":{\"message\":");
        ember_json_escape(&e, detail);
        ember_buf_puts(&e, ",\"type\":\"model_output_error\",\"code\":");
        ember_json_escape(&e, code);
        ember_buf_puts(&e, "},\"usage\":{");
        ember_buf_printf(&e,
            "\"prompt_tokens\":%d,\"completion_tokens\":%d,"
            "\"total_tokens\":%d,\"backend\":{\"degenerate\":true,"
            "\"termination_reason\":",
            prompt_tokens, completion_tokens,
            prompt_tokens + completion_tokens);
        ember_json_escape(&e, code);
        ember_buf_puts(&e, "}}}");
    }
    respond(fd, 422, "application/json", e.ptr);
    ember_buf_free(&e);
}

static void log_generation_performance(const ember_gen_result *res) {
    if (!res) return;
    const double prefill_tps = res->prefill_s > 0.0
        ? res->prefill_tokens / res->prefill_s : 0.0;
    const double decode_tps = res->decode_s > 0.0
        ? res->n_generated / res->decode_s : 0.0;
    fprintf(stderr,
            "[ember] generation prefill=%s reason=%s tokens=%d "
            "time=%.3fs rate=%.1f tok/s decode=%.3fs rate=%.2f tok/s "
            "spec=%s accept=%.3f\n",
            res->prefill_mode[0] ? res->prefill_mode : "unknown",
            res->prefill_reason[0] ? res->prefill_reason : "unknown",
            res->prefill_tokens, res->prefill_s, prefill_tps,
            res->decode_s, decode_tps,
            res->spec_decode_ran ? "yes" : "no", res->accept_rate);
}

static void run_chat(ember_server *srv, ember_chat_request *req, int fd) {
    ember_backend *be = srv->be;
    // NO ENGINE-SIDE TOOL-LOOP CEILING (deliberate ds4 parity). ds4 caps tool
    // rounds nowhere — ds4_agent.c:8448 is explicit ("deliberately no artificial
    // 'too many tool calls' ceiling here") and ds4_server.c does one generation
    // per request. Ember sits below an external harness, which owns loop/repeat
    // detection. The former ember_tool_loop_guard was removed for this reason;
    // do not reintroduce a repeat cap here. The request-derived report below is
    // additive observability only and never changes generation or finish reason.
    //
    // The legacy backend owns one mutable KV frontier and must stay serialized.
    // Resident batching isolates that frontier per session, so worker requests
    // may overlap while the engine coordinator serializes actual submissions.
    // CONCURRENCY (audited 2026-08-09). When batching is enabled `serialize` is
    // false and gen_lock is bypassed: gen_worker_start() creates
    // `batch_sessions` threads, all running gen_worker_main() -> run_chat().
    // Shared server state stays safe because every access is serialized by
    // state_lock:
    //   ember_kv_cache            all srv->kv accesses
    //   ember_continuation_store  all continuation accesses
    //   ember_tool_memory         writers and both borrowed-pointer readers
    // The DSML tracker is request-local in gen_ctx, not shared. Tool-memory
    // getters return interior pointers that eviction can free, so callers must
    // continue holding state_lock for the entire lifetime of those pointers.
    const bool serialize = !ember_backend_batch_enabled(be);
    if (serialize) pthread_mutex_lock(&srv->gen_lock);
    atomic_fetch_add(&srv->busy, 1);
    const int observed_tool_loop_rounds =
        ember_chat_request_tool_loop_rounds(req);
    const int observed_tool_loop_calls =
        ember_chat_request_tool_loop_calls(req);
    const char *observed_tool_loop_tool =
        ember_chat_request_tool_loop_tool(req);
    // Two signals, one report. The strict one (identical calls AND identical
    // results) is the stronger claim, so it wins when both fire; the weaker
    // call-signature one exists because regression testing showed
    // identical results are not a precondition for a loop -- see
    // ember_chat_request_tool_loop_calls(). `identical_results` tells a reader
    // which of the two produced the number.
    // ── automatic loop recovery (opt-in, off by default) ────────────────
    // Every other mechanism in this system is ADVISORY and the model ignores
    // all of them. Regression testing includes loops that survived an inline
    // warning, a user turn, context compaction, and the model's own promise to
    // change strategy.
    //
    // Suppressing the tools makes the repeat UNREPRESENTABLE rather than
    // discouraged: with has_tools unset the tools are never rendered into the
    // prompt and the DSML grammar is gated off, so the model cannot emit a
    // call it was never shown. This is the same mechanism tool_choice:"none"
    // already uses (chat_api.c:402-406).
    //
    // DELIBERATE EXCEPTION to the diagnostic-only contract the tool-loop
    // signals otherwise keep, hence off by default per the repo's rule for
    // risky parity features. It stays STATELESS: the evidence is the trailing
    // call run in the history the client just sent, never server-side memory,
    // so the same request always produces the same decision.
    //
    // Never overrides an explicit client demand for a tool call -- that would
    // break the caller's contract rather than the model's loop.
    const bool auto_answer =
        srv->auto_answer_after_loop > 0 &&
        observed_tool_loop_calls > srv->auto_answer_after_loop &&
        req->has_tools && !req->tool_choice_required;
    if (auto_answer) {
        fprintf(stderr,
                "[ember] auto-answer: %d identical \"%s\" calls > %d; "
                "suppressing tools for this turn\n",
                observed_tool_loop_calls,
                observed_tool_loop_tool ? observed_tool_loop_tool : "",
                srv->auto_answer_after_loop);
        req->has_tools = false;      // before any render; grammar gates on this
        if (!append_auto_answer_instruction(req))
            fprintf(stderr, "[ember] auto-answer: instruction not appended; "
                            "the model may improvise tool markup as text\n");
        pthread_mutex_lock(&srv->state_lock);
        srv->auto_answer_count++;
        srv->last_auto_answer_at = now_unix();
        snprintf(srv->last_auto_answer_tool,
                 sizeof(srv->last_auto_answer_tool), "%s",
                 observed_tool_loop_tool ? observed_tool_loop_tool : "");
        pthread_mutex_unlock(&srv->state_lock);
    }

    const bool strict_tool_loop = srv->tool_loop_report > 0 &&
        observed_tool_loop_rounds > srv->tool_loop_report;
    const bool loose_tool_loop = srv->tool_loop_report > 0 &&
        observed_tool_loop_calls > srv->tool_loop_report;
    const bool report_tool_loop = strict_tool_loop || loose_tool_loop;
    const int response_tool_loop_rounds = !report_tool_loop ? 0
        : strict_tool_loop ? observed_tool_loop_rounds : observed_tool_loop_calls;
    const bool response_tool_loop_identical = strict_tool_loop;
    if (report_tool_loop) {
        fprintf(stderr,
                "[ember] tool loop: %d identical %s for \"%s\"\n",
                response_tool_loop_rounds,
                strict_tool_loop ? "call+result rounds" : "call signatures",
                observed_tool_loop_tool ? observed_tool_loop_tool : "");
        pthread_mutex_lock(&srv->state_lock);
        // The REPORTED count, not the strict one: when only the weaker signal
        // fired, observed_tool_loop_rounds is 0 and /status would read "no loop"
        // for a request that just logged one.
        srv->last_tool_loop_rounds = response_tool_loop_rounds;
        srv->last_tool_loop_identical = response_tool_loop_identical;
        srv->last_tool_loop_at = now_unix();
        snprintf(srv->last_tool_loop_tool,
                 sizeof(srv->last_tool_loop_tool), "%s",
                 observed_tool_loop_tool ? observed_tool_loop_tool : "");
        pthread_mutex_unlock(&srv->state_lock);
    }

    // Third signal, a SIBLING of the two above rather than part of the same
    // report: both of those key on the call, and a common deployed stall
    // varies the call while an invariant wall returns the identical answer.
    // See ember_chat_request_progress_lease() for the measured cases.
    const char *stalled_tool = NULL;
    const int observed_no_progress =
        ember_chat_request_progress_lease(req, &stalled_tool);
    if (srv->no_progress_report > 0 &&
        observed_no_progress > srv->no_progress_report) {
        fprintf(stderr,
                "[ember] no progress: %d tool rounds returned nothing new "
                "(newest stale result from \"%s\")\n",
                observed_no_progress, stalled_tool ? stalled_tool : "");
        pthread_mutex_lock(&srv->state_lock);
        srv->no_progress_count++;
        srv->last_no_progress_at = now_unix();
        srv->last_no_progress_rounds = observed_no_progress;
        snprintf(srv->last_no_progress_tool,
                 sizeof(srv->last_no_progress_tool), "%s",
                 stalled_tool ? stalled_tool : "");
        pthread_mutex_unlock(&srv->state_lock);
    }
    bool enable_thinking = req->thinking_enabled;
    int32_t *ids = NULL;
    bool started_thinking = false;
    int n_prompt;
    if (req->continuation_only) {
        n_prompt = build_bound_continuation_prompt(
            srv, be, req, &ids, &started_thinking);
        if (n_prompt == -2) {
            respond_api_error(
                fd, req->api, 409,
                "continuation state is not available; retry by replaying the "
                "full message history",
                "invalid_request_error", "continuation_state_unavailable");
            atomic_fetch_sub(&srv->busy, 1);
            ember_backend_generation_release(be);
            if (serialize) pthread_mutex_unlock(&srv->gen_lock);
            return;
        }
    } else {
        attach_tool_memory(srv, req);  // B3: exact-DSML replay substitution
        if (req->raw_prompt) {
            n_prompt = ember_backend_encode(be, req->raw_prompt, &ids);
        } else {
            char *prompt =
                ember_render_prompt(req, enable_thinking,
                                    req->think_mode, true);
            started_thinking = ember_prompt_ends_in_open_think(prompt);
            n_prompt = encode_with_splices(srv, be, prompt, &ids);
            free(prompt);
        }
    }

    // Defensive: a tokenizer error (documented -1) must not reach the backend
    // (would be UB: assign(NULL, NULL-1)).
    int n_ctx = ember_backend_n_ctx(be);
    if (n_prompt < 0) {
        respond_api_error(fd, req->api, 500, "tokenization failed",
                          "server_error", "tokenization_failed");
        free(ids);
        atomic_fetch_sub(&srv->busy, 1);
        ember_backend_generation_release(be);
        if (serialize) pthread_mutex_unlock(&srv->gen_lock);
        return;
    }
    // Client-visible prompt size, captured BEFORE compaction so usage keeps
    // reporting what the client actually sent. Same convention the malformed
    // tool-call retry already follows when it grows the internal prompt.
    const int client_prompt_tokens_pre = n_prompt;

    // ds4-style context compaction (compaction.c). Deliberately ahead of the
    // context guard below: a history that would otherwise be rejected with a 400
    // gets rebuilt as system + summary + verbatim tail and served instead.
    //
    // On success req->messages has been rewritten, so the prompt must be
    // re-rendered and re-encoded through encode_with_splices — compaction does not
    // return tokens precisely so exact-DSML replay keeps its token identity.
    ember_compaction_report crep = {0};
    // Rendered lazily by the streaming branch only; freed at the common exit.
    // Zero-initialized here so the early returns below need no cleanup.
    ember_buf compaction_json = {0};
    if (srv->auto_compact && !req->raw_prompt && !req->continuation_only &&
        ember_compaction_needed(n_prompt, n_ctx)) {
        const char *why = n_prompt >= n_ctx ? "prompt exceeds context"
                                           : "soft limit before user turn";
        int compaction_slot = -1;
        if (ember_backend_disk_enabled(be)) {
            pthread_mutex_lock(&srv->state_lock);
            compaction_slot = ember_kv_reserve(&srv->kv);
            pthread_mutex_unlock(&srv->state_lock);
        }
        bool compacted = ember_compact_request(
            be, req, n_ctx, n_prompt, compaction_slot, why, &crep);
        if (compaction_slot >= 0) {
            pthread_mutex_lock(&srv->state_lock);
            ember_kv_release(&srv->kv, compaction_slot);
            pthread_mutex_unlock(&srv->state_lock);
        }
        if (compacted) {
            char *cp = ember_render_prompt(req, enable_thinking,
                                           req->think_mode, true);
            int32_t *cids = NULL;
            int cn = cp ? encode_with_splices(srv, be, cp, &cids) : -1;
            bool cthink = cp ? ember_prompt_ends_in_open_think(cp) : false;
            free(cp);
            if (cn > 0 && cn < n_ctx) {
                free(ids);
                ids = cids;
                n_prompt = cn;
                started_thinking = cthink;
                crep.compacted_tokens = cn;  // authoritative count
                fprintf(stderr,
                        "[ember] compacted %d -> %d tokens (summary=%d tail=%d "
                        "dropped=%d msgs) reason=%s\n",
                        crep.original_tokens, crep.compacted_tokens,
                        crep.summary_tokens, crep.tail_tokens,
                        crep.dropped_messages, crep.reason);
            } else {
                // The rebuild passed compaction's own check but failed the
                // splice-aware encode. Keep the original prompt: the guard below
                // will reject it honestly rather than serve a broken prompt.
                free(cids);
                crep.applied = false;
                snprintf(crep.error, sizeof(crep.error),
                         "compacted prompt failed splice encode (%d tokens)", cn);
                fprintf(stderr, "[ember] compaction abandoned: %s\n", crep.error);
            }
        } else {
            fprintf(stderr, "[ember] compaction skipped: %s\n", crep.error);
        }
    }

    // Context-length guard (ds4 http_error_context_length_exceeded): reject a
    // prompt that already fills the window, with the OpenAI-shaped 400.
    if (n_prompt >= n_ctx) {
        char message[160];
        snprintf(message, sizeof(message),
                 "prompt is %d tokens, exceeds context %d", n_prompt, n_ctx);
        ember_buf e = {0};
        ember_buf_printf(&e,
            "{\"error\":{\"message\":\"prompt is %d tokens, exceeds context %d\","
            "\"type\":\"invalid_request_error\",\"param\":\"messages\","
            "\"code\":\"context_length_exceeded\",\"n_prompt_tokens\":%d,"
            "\"n_ctx\":%d}}", n_prompt, n_ctx, n_prompt, n_ctx);
        if (req->api == EMBER_API_ANTHROPIC)
            respond_api_error(fd, req->api, 400, message,
                              "invalid_request_error",
                              "context_length_exceeded");
        else
            respond(fd, 400, "application/json", e.ptr);
        ember_buf_free(&e);
        free(ids);
        atomic_fetch_sub(&srv->busy, 1);
        ember_backend_generation_release(be);
        if (serialize) pthread_mutex_unlock(&srv->gen_lock);
        return;
    }
    // What the client sent, not what the model was conditioned on: when
    // compaction ran these differ, and the difference is reported in
    // usage.compaction rather than hidden in prompt_tokens.
    const int client_prompt_tokens = client_prompt_tokens_pre;

    char id[48];
    const char *id_prefix = req->api == EMBER_API_RESPONSES ? "resp_"
                          : req->api == EMBER_API_ANTHROPIC ? "msg_"
                          : req->api == EMBER_API_COMPLETIONS ? "cmpl-"
                          : "chatcmpl-";
    mint_prefixed_id(id_prefix, id, sizeof(id));
    long created = now_unix();

    // Lucebox parity: an omitted output limit uses the model-card default,
    // never the entire remaining context. The former ds4-compatible behavior
    // turned a small auxiliary request (an agent harness session compression) into a
    // ~130k-token permission slip when the client omitted max_tokens. Explicit
    // client limits remain supported for deliberate long generations; both
    // paths are still capped by the actual context room.
    int room = n_ctx - n_prompt;
    int want = req->max_tokens_set
        ? (req->max_tokens < 0 ? 0 : req->max_tokens)
        : srv->card.max_tokens;
    if (want < 0) want = 0;
    if (want > room) want = room;

    ember_gen_request greq = {0};
    // Constrained tool-call decoding, attached here rather than at any single
    // generate site: run_chat decodes through three of them (streaming,
    // buffered, tool-collect) and the malformed-call retry and think-tool
    // continuation are *base copies of this request, so they inherit it.
    // Freed at run_done.
    char *tool_grammar = NULL;
    if (tool_grammar_enabled() && req->has_tools && req->tools_json) {
        tool_grammar = ember_tool_grammar_build(req->tools_json,
                                                req->parallel_tool_calls);
        greq.tool_grammar = tool_grammar;   // NULL stays unconstrained
    }
    greq.prompt = ids;
    greq.n_prompt = n_prompt;
    greq.max_tokens = want;
    // An immediate tool result is evidence that the previous action already
    // happened. Keep only that next decision on the target's autoregressive
    // path. This does not force q=1 exact prefill, and older tool messages no
    // longer constrain unrelated later user turns.
    greq.force_ar_decode =
        tool_result_forces_ar() &&
        ember_chat_request_is_tool_result_continuation(req);
    greq.force_exact_prefill = false;
    // Lucebox parity (server/src/server/http_server.cpp): each omitted sampler
    // parameter resolves independently from the model card. Explicit request
    // values, including zero-valued additive penalties, always win. The CLI can
    // override only temperature; the other defaults remain model-specific.
    double eff_temp = req->temperature_set ? req->temperature : srv->default_temp;
    greq.greedy = (eff_temp == 0.0);
    greq.temperature = (float)eff_temp;
    greq.top_p = req->top_p_set ? (float)req->top_p : (float)srv->card.top_p;
    greq.top_k = req->top_k_set ? req->top_k : srv->card.top_k;
    greq.min_p = req->min_p_set ? (float)req->min_p : (float)srv->card.min_p;
    greq.seed = req->seed; greq.seed_set = req->seed_set;
    greq.rep_pen = req->rep_pen_set
        ? (float)req->rep_pen : (float)srv->card.repetition_penalty;
    greq.rep_window = req->rep_window_set ? req->rep_window : 0;
    greq.freq_pen = req->freq_pen_set ? (float)req->freq_pen : 0.0f;
    greq.pres_pen = req->pres_pen_set
        ? (float)req->pres_pen : (float)srv->card.presence_penalty;
    // DRY. Request wins; otherwise fall back to the server default, because the
    // clients that most need this (an agent gateway looping on its own replayed
    // reasoning) are exactly the ones that will never send the parameter. Ships
    // OFF -- it steers sampling, so it is enabled explicitly, per CLAUDE.md on
    // risky parity features.
    greq.dry_multiplier = req->dry_multiplier_set
        ? (float)req->dry_multiplier : dry_default_multiplier();
    greq.dry_base = req->dry_base_set ? (float)req->dry_base : 0.0f;
    greq.dry_allowed_length =
        req->dry_allowed_length_set ? req->dry_allowed_length : -1;
    greq.dry_window = req->dry_window_set ? req->dry_window : 0;
    // Level-2 thinking force-close: reserve hard_limit_reply_budget tokens so
    // the model always gets to emit a visible answer after </think>, instead of
    // thinking until it exhausts max_tokens (the bug that leaked raw reasoning).
    // Only armed when thinking is on, the card supplies a reply reserve, and the
    // combined budget leaves room to think.
    if (enable_thinking && srv->n_close_ids > 0 &&
        srv->n_natural_close_ids > 0 &&
        srv->card.hard_limit_reply_budget > 0 &&
        greq.max_tokens > srv->card.hard_limit_reply_budget) {
        // reasoning_effort tier -> think-token cap. The backend force-closes at
        // (max_tokens - reply_budget), so a think budget of T is expressed by
        // reserving (max_tokens - T). Reserve only ever GROWS beyond the card's
        // hard_limit_reply_budget, so the reply keeps at least its guaranteed
        // room and the empty-content guarantee is untouched.
        //
        // When the tier is looser than the room left by max_tokens the clamp in
        // ember_model_card_think_budget makes T == room, so reply_budget stays
        // exactly hard_limit_reply_budget: byte-identical to the pre-tier
        // behaviour. Tiers only bite when they are the binding constraint.
        int think_budget;
        if (req->reasoning_budget_tokens_set) {
            // Preserve the visible-reply guarantee even when the caller asks
            // for more reasoning than the combined output budget can hold.
            const int available =
                greq.max_tokens - srv->card.hard_limit_reply_budget;
            think_budget = req->reasoning_budget_tokens;
            if (think_budget > available) think_budget = available;
            if (think_budget < 0) think_budget = 0;
        } else {
            think_budget = ember_model_card_think_budget(
                &srv->card, req->reasoning_effort, greq.max_tokens);
        }
        int reserve = srv->card.hard_limit_reply_budget;
        if (greq.max_tokens - think_budget > reserve)
            reserve = greq.max_tokens - think_budget;
        greq.budget_close_ids = srv->close_ids;
        greq.n_budget_close   = srv->n_close_ids;
        greq.budget_natural_close_ids = srv->natural_close_ids;
        greq.n_budget_natural_close   = srv->n_natural_close_ids;
        greq.reply_budget     = reserve;
    }

    // KV reuse: restore the longest cached prefix (backend prefills only the
    // suffix), and reserve a slot to snapshot this prompt for next turn.
    // Serialized by state_lock, not gen_lock: batching bypasses gen_lock and
    // runs run_chat() concurrently on several generation workers.
    int restore_slot = -1, restore_len = 0;
    bool restore_pinned = false;
    pthread_mutex_lock(&srv->state_lock);
    ember_kv_lookup(&srv->kv, ids, n_prompt, &restore_slot, &restore_len);
    if (restore_slot >= 0) {
        restore_pinned = ember_kv_pin(&srv->kv, restore_slot);
        if (!restore_pinned) {
            restore_slot = -1;
            restore_len = 0;
        }
    }
    int snap_is_anchor = 0;
    int snap_cut = ember_kv_snap_cut(&srv->kv, ids, n_prompt, &snap_is_anchor);
    // A cold-prompt anchor (shared system-prefix, no completed turn yet) is
    // expensive to rebuild, so tag it COLD for eviction protection; a completed
    // turn boundary is a routine waypoint.
    const int snap_reason =
        snap_is_anchor ? EMBER_KV_SAVE_COLD : EMBER_KV_SAVE_NORMAL;
    pthread_mutex_unlock(&srv->state_lock);
    // Compare the resident hit with the cross-restart disk cache. A shorter
    // resident ancestor must not mask a newer post-tool snapshot on disk: at
    // exact-prefill rates, replaying that avoidable suffix can cost minutes.
    // Keep the resident snapshot pinned until the longer disk checkpoint has
    // loaded and committed successfully, so an I/O failure still has a safe
    // fallback.
    if (ember_backend_disk_enabled(be)) {
        int dl = ember_backend_disk_prefix(be, ids, n_prompt);
        int disk_slot = -1;
        if (dl > restore_len) {
            pthread_mutex_lock(&srv->state_lock);
            disk_slot = ember_kv_reserve(&srv->kv);
            pthread_mutex_unlock(&srv->state_lock);
        }
        bool loaded = disk_slot >= 0 &&
                      ember_backend_disk_lookup(be, ids, dl, disk_slot);
        pthread_mutex_lock(&srv->state_lock);
        if (loaded &&
            ember_kv_commit(&srv->kv, disk_slot, ids, dl) &&
            ember_kv_pin(&srv->kv, disk_slot)) {
            if (restore_pinned)
                ember_kv_unpin(&srv->kv, restore_slot);
            restore_slot = disk_slot;
            restore_len = dl;
            restore_pinned = true;
        } else if (disk_slot >= 0) {
            ember_kv_release(&srv->kv, disk_slot);
        }
        pthread_mutex_unlock(&srv->state_lock);
    }
    int snap_slot = -1;
    if (snap_cut > restore_len && snap_cut > 0 && snap_cut <= n_prompt) {
        pthread_mutex_lock(&srv->state_lock);
        snap_slot = ember_kv_reserve(&srv->kv);
        pthread_mutex_unlock(&srv->state_lock);
    }
    // Never snapshot into the slot we're restoring from (would alias the source).
    if (snap_slot == restore_slot) {
        pthread_mutex_lock(&srv->state_lock);
        ember_kv_release(&srv->kv, snap_slot);
        pthread_mutex_unlock(&srv->state_lock);
        snap_slot = -1;
    }
    greq.restore_slot = restore_slot;
    greq.snap_slot = snap_slot;
    greq.snap_pos = snap_slot >= 0 ? snap_cut : -1;

    // KV reuse is otherwise only visible as an absence: a turn that re-prefills
    // its whole context looks identical to one that never had a snapshot to
    // restore. Report the decision so a low coverage figure is attributable to
    // eviction (slot pressure) rather than inferred from prefill token counts.
    // One line per generation, alongside the existing generation summary.
    fprintf(stderr,
            "[ember] kv reuse: prompt=%d restored=%d (%.0f%%) slot=%d "
            "snap_cut=%d\n",
            n_prompt, restore_len,
            n_prompt > 0 ? 100.0 * restore_len / n_prompt : 0.0,
            restore_slot, snap_slot >= 0 ? snap_cut : -1);

    int max_stop_len = 0;
    for (int si = 0; si < req->n_stop; si++) {
        int l = (int)strlen(req->stop[si]);
        if (l > max_stop_len) max_stop_len = l;
    }

    if (req->stream && req->api == EMBER_API_CHAT) {
        ember_buf hdr = {0};
        ember_sse_headers(&hdr, g_enable_cors);
        bool header_ok = ember_send_all(fd, hdr.ptr, hdr.len) == 0;
        ember_buf_free(&hdr);
        if (!header_ok) {
            finish_prompt_snapshot(srv, be, snap_slot, ids, snap_cut,
                                   false, snap_reason);
            goto run_done;
        }

        ember_sse_stream st;
        ember_sse_init(&st, id, req->model, created, req->has_tools,
                       started_thinking, false);
        ember_sse_set_reasoning_filter(
            &st, srv->card.thinking_terminator_hint);
        st.include_usage = req->stream_include_usage;
        st.cached_tokens = restore_len;
        // Report compaction in the streaming usage chunk too. Note this rides on
        // stream_options.include_usage: a client that opts out of usage will not
        // see it, so the stderr line above remains the server's own record.
        if (crep.applied || crep.error[0]) {
            ember_compaction_append_json(&compaction_json, &crep);
            st.compaction_json = compaction_json.ptr;
        }
        st.stops = req->stop; st.n_stops = req->n_stop; st.max_stop_len = max_stop_len;
        // Initial role primer chunk (ds4), before any content delta.
        ember_buf rc = {0};
        ember_sse_role_chunk(&st, &rc);
        bool role_ok = ember_send_all(fd, rc.ptr, rc.len) == 0;
        ember_buf_free(&rc);
        if (!role_ok) {
            ember_sse_free(&st);
            finish_prompt_snapshot(srv, be, snap_slot, ids, snap_cut,
                                   false, snap_reason);
            goto run_done;
        }
        gen_ctx g = {0};
        g.be = be; g.st = &st; g.fd = fd; g.has_tools = req->has_tools;
        g.prompt_ids = ids; g.n_prompt_ids = n_prompt;
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
        finish_prompt_snapshot(
            srv, be, snap_slot, ids, snap_cut,
            res.ok && !g.disconnected && res.snapshot_saved, snap_reason);
        (void)continue_tool_started_in_think(
            srv, be, req, &greq, ids, n_prompt, started_thinking, &g, &res);
        bool unclosed_think_tool =
            tool_started_in_unclosed_think(&g, started_thinking);
        int hidden_recovery_tokens = 0;
        bool recovery_attempted = false;
        bool recovered = false;
        if (res.ok && !g.disconnected) {
            recovered = retry_malformed_tool_call(
                srv, be, req, &greq, &ids, &n_prompt,
                &started_thinking, &g, &res,
                &hidden_recovery_tokens, &recovery_attempted);
            if (recovered)
                ember_sse_reset_attempt(&st, started_thinking);
        }
        log_generation_performance(&res);
        const int completion_tokens =
            hidden_recovery_tokens + res.n_generated;
        g.scratch.len = 0; if (g.scratch.ptr) g.scratch.ptr[0] = '\0';
        const char *stream_scan = g.acc.ptr ? g.acc.ptr : "";
        if (started_thinking) {
            const char *close = last_substr(stream_scan, "</think>");
            // ds4_server.c:4792-4804: DSML before a completed reasoning close
            // is reasoning text, never an executable call.
            stream_scan = close ? close + 8 : "";
        }
        ember_tool_calls stream_probe = {0};
        const char *stream_tool_error = NULL;
        bool stream_tools_valid =
            parse_executable_tool_calls(
                req, stream_scan, &stream_probe, &stream_tool_error);
        bool had_tools = false;
        if (!generation_stalled(&res) && !unclosed_think_tool &&
            stream_tools_valid) {
            ember_sse_update(&st, g.acc.ptr, g.acc.len, true, &g.scratch);
            had_tools =
                ember_sse_emit_tools(&st, g.acc.ptr, g.acc.len, &g.scratch);
        }
        ember_tool_calls_free(&stream_probe);
        // A watchdog stop CLAIMS the turn regardless of how it is rendered.
        // Only the rendering is conditional: gating the branch itself would let
        // a stall fall through to the tool-call branches below and emit the very
        // typed error this exists to avoid — `unclosed_think_tool` is reachable
        // because a reasoning-phase stop is now exactly reasoning_cycle /
        // repetition (the echo rule is visible-only), and the tool branch is
        // reachable whenever EMBER_STREAM_TOOL_ERROR=1, which would report a
        // stall as invalid_tool_call: the misattribution that
        // test_tool_safety_server.py pins as "must never".
        const bool stalled_stream = generation_stalled(&res) && !g.disconnected;
        if (!res.ok && !g.disconnected) {
            // Backend failure mid-stream (ds4 sse_error_event).
            ember_sse_error(
                &st, res.error_code, res.error_detail, false, &g.scratch);
        } else if (stalled_stream && !stream_watchdog_text_fallback()) {
            char detail[384];
            format_generation_stalled(detail, sizeof(detail), &res);
            ember_sse_error(
                &st, res.termination_reason, detail, true, &g.scratch);
        } else if (!stalled_stream && unclosed_think_tool && !g.disconnected) {
            ember_sse_error(
                &st, "invalid_tool_call",
                "The model started a tool call inside an unclosed reasoning "
                "block and could not complete Ember's bounded </think> "
                "continuation; no tool call was emitted.",
                true, &g.scratch);
        } else if (!stalled_stream && !stream_tools_valid && !g.disconnected &&
                   !stream_tool_text_fallback()) {
            // Tool deltas were validation-gated, so the harness receives one
            // actionable terminal error and never observes a partial call.
            char detail[512];
            format_stream_tool_failure(
                detail, sizeof(detail), stream_tool_error,
                recovery_attempted);
            ember_sse_error(&st, "invalid_tool_call",
                            detail, recovery_attempted, &g.scratch);
        } else {
            // Malformed-call recovery (ds4_server.c:5231-5241): a malformed
            // tool block is model output, not a server failure. Drop the
            // suppressed block and stop normally rather than killing the turn —
            // a streaming agent cannot retry, so an error costs it the round.
            if (!stream_tools_valid && !g.disconnected) {
                dump_rejected_block(g.acc.ptr, g.acc.len, st.tool_start,
                                    stream_tool_error);
                ember_sse_discard_tool_block(&st, g.acc.len);
                fprintf(stderr,
                        "[ember] malformed tool call dropped; finishing turn "
                        "normally (%s)\n",
                        stream_tool_error ? stream_tool_error : "invalid call");
            }
            if (stalled_stream) {
                fprintf(stderr,
                        "[ember] progress watchdog stop (%s); finishing turn "
                        "normally on the streaming path\n",
                        res.termination_reason);
            }
            // Unknown/legacy degenerate stops without a typed watchdog reason
            // remain ordinary length truncations.
            //
            // A TYPED watchdog stop must NOT: "length" is precisely the signal
            // that tells a harness the reply was cut off and invites it to ask
            // for a continuation, which re-enters the same degenerate decode.
            // This can trigger repeated, useless continuations. Report "stop":
            // the turn is over, and the
            // reason is already in the log and in /status telemetry.
            const char *finish =
                g.disconnected || stalled_stream ? "stop"
                : had_tools ? "tool_calls"
                : g.hit_stop ? "stop"
                : res.degenerate_decode_close ? "length"
                : res.finish_reason;
            if (!stream_tools_valid && !g.disconnected)
                finish = ember_tool_parse_failure_finish(finish);
            // Disconnects are not the model's failure to progress.
            if (!g.disconnected) {
                const bool vis = ember_sse_delivered_visible(&st);
                note_tool_markup_leak(srv, ember_sse_delivered_tool_markup(&st),
                                      "stream");
                note_nonprogress_turn(srv, &res, had_tools, vis, "stream");
                note_degenerate_turn(srv, &res, had_tools || vis, "stream");
            }
            // client_prompt_tokens, not n_prompt: when compaction ran these
            // differ, and both response paths must agree on prompt_tokens
            // meaning "what the client sent". The compacted size is reported
            // separately in usage.compaction.
            st.prefill_tokens = res.prefill_tokens;
            st.prefill_s = res.prefill_s;
            st.decode_s = res.decode_s;
            st.accept_rate = res.accept_rate;
            st.spec_cycles = res.spec_cycles;
            st.spec_provider_age_s = res.spec_provider_age_s;
            st.spec_provider_block_s = res.spec_provider_block_s;
            st.spec_head_s = res.spec_head_s;
            st.spec_verify_s = res.spec_verify_s;
            st.prefill_mode = res.prefill_mode;
            st.prefill_reason = res.prefill_reason;
            st.termination_reason = res.termination_reason;
            st.reasoning_budget_exhausted = res.budget_forced_close;
            st.degenerate = res.degenerate_decode_close;
            st.tool_loop_rounds = response_tool_loop_rounds;
            st.tool_loop_identical = response_tool_loop_identical;
            st.tool_loop_tool = observed_tool_loop_tool;
            ember_sse_finish(&st, finish, client_prompt_tokens,
                             completion_tokens,
                             &g.scratch);
        }
        if (!g.disconnected && g.scratch.len &&
            ember_send_all(fd, g.scratch.ptr, g.scratch.len) != 0)
            g.disconnected = true;

        if (res.ok && !g.disconnected && !generation_stalled(&res) &&
            !unclosed_think_tool && stream_tools_valid &&
            !res.budget_forced_close && !res.degenerate_decode_close)
            snapshot_post_toolcall(srv, be, ids, n_prompt, &g);  // B3 L2

        // B3: store only the exact sampled DSML block under each streamed call
        // id. Reasoning/content are always re-rendered canonically.
        if (res.ok && !g.disconnected && !generation_stalled(&res) &&
            !unclosed_think_tool && stream_tools_valid &&
            !res.budget_forced_close && !res.degenerate_decode_close) {
            for (int i = 0; i < st.n_tool_ids; i++)
                remember_tool_block(srv, be, st.tool_ids[i], &g);
            if (had_tools)
                remember_continuation(
                    srv, req, ids, n_prompt, &g,
                    (const char *const *)st.tool_ids, st.n_tool_ids, NULL);
        }
        ember_sse_free(&st);
        ember_buf_free(&g.acc);
        free(g.gen_ids);  // B3 L2
        ember_buf_free(&g.scratch);
    } else if (req->stream) {
        // Responses, Anthropic Messages, and legacy Completions share Chat's
        // buffer-and-resplit parser, but a protocol sink emits their native
        // event taxonomy token-by-token instead of synthesizing it after the
        // model turn has completed.
        ember_buf hdr = {0};
        ember_sse_headers(&hdr, g_enable_cors);
        bool header_ok = ember_send_all(fd, hdr.ptr, hdr.len) == 0;
        ember_buf_free(&hdr);
        if (!header_ok) {
            finish_prompt_snapshot(srv, be, snap_slot, ids, snap_cut,
                                   false, snap_reason);
            goto run_done;
        }

        ember_sse_stream splitter;
        ember_sse_init(&splitter, id, req->model, created, req->has_tools,
                       started_thinking, false);
        ember_sse_set_reasoning_filter(
            &splitter, srv->card.thinking_terminator_hint);
        splitter.stops = req->stop;
        splitter.n_stops = req->n_stop;
        splitter.max_stop_len = max_stop_len;

        ember_protocol_stream protocol;
        ember_protocol_stream_init(
            &protocol, req, id, req->model, created, client_prompt_tokens);
        ember_protocol_stream_bind(&protocol, &splitter);
        ember_buf opening = {0};
        ember_protocol_stream_begin(&protocol, &opening);
        bool opening_ok = !opening.len ||
            ember_send_all(fd, opening.ptr, opening.len) == 0;
        ember_buf_free(&opening);
        if (!opening_ok) {
            ember_protocol_stream_free(&protocol);
            ember_sse_free(&splitter);
            finish_prompt_snapshot(srv, be, snap_slot, ids, snap_cut,
                                   false, snap_reason);
            goto run_done;
        }

        gen_ctx g = {0};
        g.be = be;
        g.st = &splitter;
        g.fd = fd;
        g.has_tools = req->has_tools;
        g.prompt_ids = ids;
        g.n_prompt_ids = n_prompt;
        g.stops = req->stop;
        g.n_stops = req->n_stop;
        g.started_thinking = started_thinking;
        // Native protocol terminal objects repeat the complete output. Buffer
        // tool-bearing attempts so a hidden replacement cannot make their live
        // deltas disagree with the terminal response.
        g.collect_only = req->has_tools;
        g.keepalive_while_collecting = req->has_tools;
        greq.on_token = on_token;
        greq.on_prefill = on_prefill;
        greq.ud = &g;
        g.dsml_active = req->has_tools && eff_temp > 0.0;
        if (g.dsml_active) {
            ember_dsml_tracker_init(&g.dsml);
            greq.force_greedy = gen_force_greedy;
            greq.fg_ud = &g;
        }

        ember_gen_result res = ember_backend_generate(be, &greq);
        finish_prompt_snapshot(
            srv, be, snap_slot, ids, snap_cut,
            res.ok && !g.disconnected && res.snapshot_saved, snap_reason);
        (void)continue_tool_started_in_think(
            srv, be, req, &greq, ids, n_prompt, started_thinking, &g, &res);
        bool unclosed_think_tool =
            tool_started_in_unclosed_think(&g, started_thinking);
        int hidden_recovery_tokens = 0;
        bool recovery_attempted = false;
        bool recovered = false;
        if (res.ok && !g.disconnected) {
            recovered = retry_malformed_tool_call(
                srv, be, req, &greq, &ids, &n_prompt,
                &started_thinking, &g, &res,
                &hidden_recovery_tokens, &recovery_attempted);
            if (recovered)
                ember_sse_reset_attempt(&splitter, started_thinking);
        }
        log_generation_performance(&res);
        const int completion_tokens =
            hidden_recovery_tokens + res.n_generated;
        g.scratch.len = 0;
        if (g.scratch.ptr) g.scratch.ptr[0] = '\0';

        // Build the terminal native response from the same final split used by
        // atomic requests. Validation happens before any buffered tool attempt
        // is projected onto the protocol stream.
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
        strip_forced_close_hint(srv, reasoning, res.budget_forced_close);
        ember_tool_calls tc = {0};
        const char *native_tool_error = NULL;
        bool native_tools_valid =
            parse_executable_tool_calls(
                req, content, &tc, &native_tool_error);
        bool had_tools = false;
        if (res.ok && !generation_stalled(&res) && !unclosed_think_tool &&
            native_tools_valid && !g.disconnected) {
            ember_sse_update(
                &splitter, g.acc.ptr, g.acc.len, true, &g.scratch);
            had_tools = ember_sse_emit_tools(
                &splitter, g.acc.ptr, g.acc.len, &g.scratch);
        }
        for (int i = 0; i < tc.len; ++i) {
            const char *stream_id =
                i < splitter.n_tool_ids ? splitter.tool_ids[i] : NULL;
            char fallback[40];
            if (!stream_id) {
                mint_tool_id(fallback);
                stream_id = fallback;
            }
            free(tc.calls[i].id);
            tc.calls[i].id = strdup(stream_id);
        }
        char *content_trimmed = NULL;
        const char *emit_content = content;
        // A rejected block is dropped from the terminal object too, so it agrees
        // with the deltas: the splitter suppressed everything from the tool
        // marker on (ember_sse_discard_tool_block), and untrimmed terminal
        // content would leak exactly the contaminated bytes validation refused.
        if (!native_tools_valid) ember_tool_calls_free(&tc);
        if (tc.len > 0 || !native_tools_valid) {
            const char *ts = ember_find_tool_start(content);
            size_t clen = ts ? (size_t)(ts - content) : strlen(content);
            while (clen > 0 &&
                   (content[clen - 1] == '\n' || content[clen - 1] == ' ' ||
                    content[clen - 1] == '\t' || content[clen - 1] == '\r'))
                --clen;
            content_trimmed = strndup(content, clen);
            emit_content = content_trimmed;
        }
        const char *finish =
            had_tools || tc.len > 0 ? "tool_calls"
            : g.hit_stop ? "stop"
            : res.degenerate_decode_close ? "length"
            : res.finish_reason;
        ember_protocol_result pr = {
            .id = id,
            .model = req->model,
            .created = created,
            .content = emit_content,
            .reasoning = reasoning,
            .calls = &tc,
            .finish_reason = finish,
            .stop_sequence = g.hit_stop_sequence,
            .termination_reason = res.termination_reason,
            .reasoning_budget_exhausted = res.budget_forced_close,
            .tool_loop_rounds = response_tool_loop_rounds,
            .tool_loop_identical = response_tool_loop_identical,
            .tool_loop_tool = observed_tool_loop_tool,
            .prompt_tokens = client_prompt_tokens,
            .completion_tokens = completion_tokens,
            .cached_tokens = restore_len,
        };
        if (!res.ok && !g.disconnected) {
            ember_protocol_stream_error(
                &protocol, res.error_code, res.error_detail, false,
                &g.scratch);
        } else if (generation_stalled(&res) && !g.disconnected) {
            char detail[384];
            format_generation_stalled(detail, sizeof(detail), &res);
            ember_protocol_stream_error(
                &protocol, res.termination_reason, detail, true,
                &g.scratch);
        } else if (unclosed_think_tool && !g.disconnected) {
            ember_protocol_stream_error(
                &protocol, "invalid_tool_call",
                "The model started a tool call inside an unclosed reasoning "
                "block and could not complete Ember's bounded </think> "
                "continuation; no tool call was emitted.",
                true, &g.scratch);
        } else if (!native_tools_valid && !g.disconnected &&
                   !stream_tool_text_fallback()) {
            char detail[512];
            format_stream_tool_failure(
                detail, sizeof(detail), native_tool_error,
                recovery_attempted);
            ember_protocol_stream_error(
                &protocol, "invalid_tool_call",
                detail,
                recovery_attempted,
                &g.scratch);
        } else if (!g.disconnected) {
            // Same ds4 recovery as the chat path.
            if (!native_tools_valid) {
                dump_rejected_block(g.acc.ptr, g.acc.len, splitter.tool_start,
                                    native_tool_error);
                ember_sse_discard_tool_block(&splitter, g.acc.len);
                pr.finish_reason =
                    ember_tool_parse_failure_finish(pr.finish_reason);
                fprintf(stderr,
                        "[ember] malformed tool call dropped; finishing turn "
                        "normally (%s)\n",
                        native_tool_error ? native_tool_error : "invalid call");
            }
            ember_protocol_stream_finish(&protocol, &pr, &g.scratch);
        }
        if (!g.disconnected && g.scratch.len &&
            ember_send_all(fd, g.scratch.ptr, g.scratch.len) != 0)
            g.disconnected = true;

        if (res.ok && !g.disconnected && !generation_stalled(&res) &&
            !unclosed_think_tool && native_tools_valid &&
            !res.budget_forced_close &&
            !res.degenerate_decode_close) {
            if (tc.len > 0)
                snapshot_post_toolcall(srv, be, ids, n_prompt, &g);
            for (int i = 0; i < splitter.n_tool_ids; ++i) {
                remember_tool_block(
                    srv, be, splitter.tool_ids[i], &g);
            }
            if (had_tools)
                remember_continuation(
                    srv, req, ids, n_prompt, &g,
                    (const char *const *)splitter.tool_ids,
                    splitter.n_tool_ids,
                    req->api == EMBER_API_RESPONSES ? id : NULL);
        }
        ember_tool_calls_free(&tc);
        free(content_trimmed);
        free(reasoning);
        ember_protocol_stream_free(&protocol);
        ember_sse_free(&splitter);
        ember_buf_free(&g.acc);
        free(g.gen_ids);
        ember_buf_free(&g.scratch);
    } else {
        // Non-stream: collect all tokens, then split reasoning vs content and
        // build one chat.completion. (Tool-call structuring is a follow-on.)
        gen_ctx g = {0};
        g.be = be; g.fd = fd; g.collect_only = true; g.has_tools = req->has_tools;
        g.prompt_ids = ids; g.n_prompt_ids = n_prompt;
        g.stops = req->stop; g.n_stops = req->n_stop;
        g.started_thinking = started_thinking;  // B#1: gate markers on thinking
        greq.on_token = on_token;
        // Installed for the disconnect check only — it returns before writing
        // when collect_only is set, so no SSE comment can leak into the JSON
        // body. Without it, a non-streaming request has NO liveness signal
        // during prefill, which is its longest phase; an abandoned 50k-token
        // prefill then ran to completion holding the only generation slot.
        greq.on_prefill = on_prefill;
        greq.ud = &g;
        g.dsml_active = req->has_tools && eff_temp > 0.0;  // B6 (see streaming path)
        if (g.dsml_active) {
            ember_dsml_tracker_init(&g.dsml);
            greq.force_greedy = gen_force_greedy;
            greq.fg_ud = &g;
        }
        ember_gen_result res = ember_backend_generate(be, &greq);
        // #2: commit only on a real backend snapshot save (see streaming path).
        finish_prompt_snapshot(
            srv, be, snap_slot, ids, snap_cut,
            res.ok && !g.disconnected && res.snapshot_saved, snap_reason);
        // Non-stream responses are atomic: never present partial output as a
        // successful completion or retain it in exact-tool replay memory.
        if (!res.ok) {
            respond_api_error(
                fd, req->api, 500,
                res.error_detail[0] ? res.error_detail
                                    : "backend generation failed",
                "server_error",
                res.error_code[0] ? res.error_code : "internal_error");
            ember_buf_free(&g.acc);
            free(g.gen_ids);  // B3 L2
            ember_buf_free(&g.scratch);
            goto ns_done;
        }

        (void)continue_tool_started_in_think(
            srv, be, req, &greq, ids, n_prompt, started_thinking, &g, &res);
        // The continuation is a second backend call. Preserve the atomic error
        // contract if that call itself fails.
        if (!res.ok) {
            respond_api_error(
                fd, req->api, 500,
                res.error_detail[0] ? res.error_detail
                                    : "backend generation failed",
                "server_error",
                res.error_code[0] ? res.error_code : "internal_error");
            ember_buf_free(&g.acc);
            free(g.gen_ids);
            ember_buf_free(&g.scratch);
            goto ns_done;
        }
        bool unclosed_think_tool =
            tool_started_in_unclosed_think(&g, started_thinking);

        int hidden_recovery_tokens = 0;
        bool recovery_attempted = false;
        (void)retry_malformed_tool_call(
            srv, be, req, &greq, &ids, &n_prompt, &started_thinking,
            &g, &res, &hidden_recovery_tokens, &recovery_attempted);
        if (!res.ok) {
            respond_api_error(
                fd, req->api, 500,
                res.error_detail[0] ? res.error_detail
                                    : "backend generation failed",
                "server_error",
                res.error_code[0] ? res.error_code : "internal_error");
            ember_buf_free(&g.acc);
            free(g.gen_ids);
            ember_buf_free(&g.scratch);
            goto ns_done;
        }
        // A replacement attempt is a fresh decode and can itself start its
        // tool stanza before </think>. retry_malformed_tool_call applies the
        // bounded close continuation; if it could not, preserve the typed
        // invalid-output boundary instead of returning an empty success.
        unclosed_think_tool =
            tool_started_in_unclosed_think(&g, started_thinking);
        log_generation_performance(&res);
        const int completion_tokens =
            hidden_recovery_tokens + res.n_generated;

        if (generation_stalled(&res)) {
            respond_generation_stalled(
                fd, req, &res, client_prompt_tokens, completion_tokens);
            ember_buf_free(&g.acc);
            free(g.gen_ids);
            ember_buf_free(&g.scratch);
            goto ns_done;
        }
        if (unclosed_think_tool) {
            if (req->api == EMBER_API_ANTHROPIC) {
                respond_api_error(
                    fd, req->api, 422,
                    "The model started a tool call inside an unclosed reasoning "
                    "block and could not complete Ember's bounded </think> "
                    "continuation; no tool call was emitted.",
                    "model_output_error", "invalid_tool_call");
                ember_buf_free(&g.acc);
                free(g.gen_ids);
                ember_buf_free(&g.scratch);
                goto ns_done;
            }
            ember_buf e = {0};
            ember_buf_puts(&e,
                "{\"error\":{\"message\":\"The model started a tool call "
                "inside an unclosed reasoning block and could not complete "
                "Ember's bounded </think> continuation; no tool call was "
                "emitted.\",\"type\":\"model_output_error\","
                "\"code\":\"invalid_tool_call\"},\"usage\":{");
            ember_buf_printf(&e,
                "\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                "\"total_tokens\":%d}}",
                client_prompt_tokens, completion_tokens,
                client_prompt_tokens + completion_tokens);
            respond(fd, 422, "application/json", e.ptr);
            ember_buf_free(&e);
            ember_buf_free(&g.acc);
            free(g.gen_ids);
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
        strip_forced_close_hint(srv, reasoning, res.budget_forced_close);

        // stop sequences (ds4): truncate visible content at the earliest stop
        // hit and report finish="stop".
        char *content_stop = NULL;
        bool hit_stop = g.hit_stop;
        const char *matched_stop = g.hit_stop_sequence;
        for (int si = 0; si < req->n_stop; si++) {
            const char *h = strstr(content, req->stop[si]);
            if (!h) continue;
            size_t clen = (size_t)(h - content);
            if (!content_stop || clen < strlen(content_stop)) {
                free(content_stop);
                content_stop = strndup(content, clen);
                matched_stop = req->stop[si];
            }
            hit_stop = true;
        }
        if (content_stop) content = content_stop;

        // Tool calls: parse the post-think text; if any, strip the DSML block
        // from content and set finish="tool_calls" (ds4 non-stream parity —
        // previously the raw DSML leaked into content with finish="stop").
        ember_tool_calls tc = {0};
        const char *tool_error = NULL;
        if (!parse_executable_tool_calls(req, content, &tc, &tool_error)) {
            // The bounded model-visible repair also failed. Never turn its
            // second malformed attempt into an executable call: return a typed
            // provider error so the harness can report/replan explicitly.
            if (req->api == EMBER_API_ANTHROPIC) {
                char detail[640];
                snprintf(detail, sizeof(detail),
                    recovery_attempted
                        ? "The model generated an unsafe or incomplete tool "
                          "call after one recovery attempt: %s"
                        : "The model generated an unsafe or incomplete tool "
                          "call and recovery could not be started: %s",
                    tool_error ? tool_error : "invalid tool output");
                respond_api_error(fd, req->api, 422, detail,
                                  "model_output_error", "invalid_tool_call");
                ember_tool_calls_free(&tc);
                free(content_stop);
                free(reasoning);
                ember_buf_free(&g.acc);
                free(g.gen_ids);
                ember_buf_free(&g.scratch);
                goto ns_done;
            }
            ember_buf e = {0};
            ember_buf_puts(&e,
                recovery_attempted
                    ? "{\"error\":{\"message\":\"The model generated an unsafe or "
                      "incomplete tool call after one recovery attempt: "
                    : "{\"error\":{\"message\":\"The model generated an unsafe or "
                      "incomplete tool call and recovery could not be started: ");
            json_escape_str(&e, tool_error ? tool_error :
                                           "invalid tool output");
            ember_buf_puts(&e,
                "\",\"type\":\"model_output_error\","
                "\"code\":\"invalid_tool_call\"},\"usage\":{");
            ember_buf_printf(&e,
                "\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                "\"total_tokens\":%d,\"backend\":{\"forced_close\":%s,"
                "\"degenerate\":%s}}}",
                client_prompt_tokens, completion_tokens,
                client_prompt_tokens + completion_tokens,
                res.budget_forced_close ? "true" : "false",
                res.degenerate_decode_close ? "true" : "false");
            respond(fd, 422, "application/json", e.ptr);
            ember_buf_free(&e);
            ember_tool_calls_free(&tc);
            free(content_stop);
            free(reasoning);
            ember_buf_free(&g.acc);
            free(g.gen_ids);
            ember_buf_free(&g.scratch);
            goto ns_done;
        }
        // B3: mint every id before building the response. Durable replay state
        // is committed only after that response is successfully delivered.
        for (int i = 0; i < tc.len; i++) {
            char tid[40]; mint_tool_id(tid);
            free(tc.calls[i].id);
            tc.calls[i].id = strdup(tid);
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
        // Same exclusion as the streaming path: a vanished client is not the
        // model failing to progress. The streaming site had this guard from
        // the start and the buffered one did not -- an asymmetry, not a
        // decision.
        if (!g.disconnected) {
            note_tool_markup_leak(srv, ember_text_has_tool_markup(emit_content),
                                  "buffered");
            const bool vis = buffered_delivered_visible(emit_content);
            note_nonprogress_turn(srv, &res, tc.len > 0, vis, "buffered");
            note_degenerate_turn(srv, &res, tc.len > 0 || vis, "buffered");
        }

        if (req->api != EMBER_API_CHAT) {
            ember_protocol_result pr = {
                .id = id,
                .model = req->model,
                .created = created,
                .content = emit_content,
                .reasoning = reasoning,
                .calls = &tc,
                .finish_reason = finish,
                .stop_sequence = matched_stop,
                .termination_reason = res.termination_reason,
                .reasoning_budget_exhausted = res.budget_forced_close,
                .tool_loop_rounds = response_tool_loop_rounds,
                .tool_loop_identical = response_tool_loop_identical,
                .tool_loop_tool = observed_tool_loop_tool,
                .prompt_tokens = client_prompt_tokens,
                .completion_tokens = completion_tokens,
                .cached_tokens = restore_len,
            };
            bool emitted = ember_protocol_emit(fd, req, &pr);
            if (emitted && tc.len > 0 && !res.budget_forced_close &&
                !res.degenerate_decode_close)
                persist_atomic_tool_frontier(
                    srv, be, req, ids, n_prompt, &g, &tc, id);
            ember_tool_calls_free(&tc);
            free(content_trimmed);
            free(content_stop);
            free(reasoning);
            ember_buf_free(&g.acc);
            free(g.gen_ids);
            ember_buf_free(&g.scratch);
            goto ns_done;
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
        double ptps = res.prefill_s > 0
            ? res.prefill_tokens / res.prefill_s : 0.0;
        ember_buf_printf(&b, "},\"finish_reason\":\"%s\"", finish);
        if (response_tool_loop_rounds > 0) {
            ember_buf_putc(&b, ',');
            append_tool_loop_json(&b, response_tool_loop_rounds,
                                  observed_tool_loop_tool,
                                  response_tool_loop_identical);
        }
        if (res.termination_reason[0]) {
            ember_buf_puts(&b, ",\"finish_details\":{\"type\":");
            ember_json_escape(&b, res.termination_reason);
            ember_buf_putc(&b, '}');
        }
        ember_buf_printf(&b,
            "}],\"usage\":{\"prompt_tokens\":%d,"
            "\"completion_tokens\":%d,\"total_tokens\":%d,"
            "\"prompt_tokens_details\":{\"cached_tokens\":%d},\"timings\":{"
            "\"prefill_ms\":%.1f,\"prefill_tokens\":%d,"
            "\"prefill_tokens_per_sec\":%.1f,"
            "\"decode_ms\":%.1f,\"decode_tokens_per_sec\":%.2f,"
            "\"spec_cycles\":%d,\"spec_provider_age_ms\":%.1f,"
            "\"spec_provider_block_ms\":%.1f,\"spec_head_ms\":%.1f,"
            "\"spec_verify_ms\":%.1f},"
            "\"accept_rate\":%.3f,\"restored_prefix\":%d,"
            "\"backend\":{\"forced_close\":%s,\"degenerate\":%s,"
            "\"empty\":%s,\"spec_ran\":%s,\"prefill_mode\":\"%s\","
            "\"prefill_reason\":\"%s\"",
            client_prompt_tokens, completion_tokens,
            client_prompt_tokens + completion_tokens, restore_len,
            res.prefill_s * 1000.0, res.prefill_tokens, ptps,
            res.decode_s * 1000.0, dtps, res.spec_cycles,
            res.spec_provider_age_s * 1000.0,
            res.spec_provider_block_s * 1000.0,
            res.spec_head_s * 1000.0, res.spec_verify_s * 1000.0,
            res.accept_rate, restore_len,
            res.budget_forced_close ? "true" : "false",
            res.degenerate_decode_close ? "true" : "false",
            res.empty_visible_output ? "true" : "false",
            res.spec_decode_ran ? "true" : "false",
            res.prefill_mode[0] ? res.prefill_mode : "unknown",
            res.prefill_reason[0] ? res.prefill_reason : "unknown");
        if (res.budget_forced_close)
            ember_buf_puts(&b,
                ",\"reasoning_stop_reason\":"
                "\"reasoning_budget_exhausted\"");
        if (res.termination_reason[0]) {
            ember_buf_puts(&b, ",\"termination_reason\":");
            ember_json_escape(&b, res.termination_reason);
        }
        ember_buf_putc(&b, '}');
        // Compaction is never silent: report it whenever it ran or was attempted.
        if (crep.applied || crep.error[0]) {
            ember_buf_putc(&b, ',');
            ember_compaction_append_json(&b, &crep);
        }
        ember_buf_puts(&b, "}}");
        bool emitted = respond(fd, 200, "application/json", b.ptr);
        if (emitted && tc.len > 0 && !res.budget_forced_close &&
            !res.degenerate_decode_close)
            persist_atomic_tool_frontier(
                srv, be, req, ids, n_prompt, &g, &tc, id);
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
run_done:
    free(tool_grammar);
    atomic_fetch_sub(&srv->busy, 1);
    atomic_fetch_add(&srv->served, 1);
    ember_backend_generation_release(be);
    if (restore_pinned) {
        pthread_mutex_lock(&srv->state_lock);
        ember_kv_unpin(&srv->kv, restore_slot);
        pthread_mutex_unlock(&srv->state_lock);
    }
    if (serialize) pthread_mutex_unlock(&srv->gen_lock);
    ember_buf_free(&compaction_json);
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
        gen_job *job = NULL;
        for (;;) {
            while (w->reclaiming && !w->stop)
                pthread_cond_wait(&w->cond, &w->lock);
            while (!w->head && !w->stop) {
                if (w->active_jobs > 0 || !w->graphs_dirty ||
                    w->idle_reclaim_secs <= 0.0) {
                    pthread_cond_wait(&w->cond, &w->lock);
                    continue;
                }
                // Quiet since the last job: wake at the reclaim deadline. A new
                // job signals the cond and cancels it, so active traffic never
                // pays the graph rebuild.
                struct timespec wake =
                    monotonic_deadline(w->last_done_at + w->idle_reclaim_secs);
                int rc = pthread_cond_timedwait(&w->cond, &w->lock, &wake);
                if (rc == ETIMEDOUT && !w->head && !w->stop &&
                    !w->reclaiming) {
                    // Release outside the lock: freeing GPU buffers can take
                    // milliseconds and submitters should not block on it. A job
                    // arriving meanwhile is picked up by the re-checked loop.
                    w->reclaiming = true;
                    pthread_mutex_unlock(&w->lock);
                    ember_backend_release_idle_graphs(w->be);
                    pthread_mutex_lock(&w->lock);
                    w->graphs_dirty = false;
                    w->reclaiming = false;
                    pthread_cond_broadcast(&w->cond);
                }
            }
            if (!w->head && w->stop) break;

            // Foreground work always jumps ahead of queued maintenance work.
            // Preserve FIFO order within each class.
            gen_job *prev = NULL, *pick_prev = NULL;
            for (gen_job *it = w->head; it; prev = it, it = it->next) {
                if (!it->req->background) {
                    job = it;
                    pick_prev = prev;
                    break;
                }
            }
            if (!job) {
                job = w->head;  // all queued jobs are background
                double now = monotonic_now();
                if (!w->stop &&
                    !ember_background_gate_ready(&w->bg_gate,
                                                 job->enqueued_at, now)) {
                    struct timespec wake = monotonic_deadline(
                        ember_background_gate_ready_at(&w->bg_gate,
                                                       job->enqueued_at));
                    job = NULL;
                    pthread_cond_timedwait(&w->cond, &w->lock, &wake);
                    continue;  // foreground arrival or eligibility deadline
                }
            }

            if (pick_prev) pick_prev->next = job->next;
            else w->head = job->next;
            if (w->tail == job) w->tail = pick_prev;
            job->next = NULL;
            break;
        }
        if (!job && w->stop) { pthread_mutex_unlock(&w->lock); break; }
        w->queued--;  // #3: dequeued from the bounded FIFO
        w->active_jobs++;
        pthread_mutex_unlock(&w->lock);

        run_chat(job->srv, job->req, job->fd);

        // Graphs are now populated for this request's shape; arm the reclaim.
        pthread_mutex_lock(&w->lock);
        w->active_jobs--;
        w->last_done_at = monotonic_now();
        w->graphs_dirty = true;
        pthread_cond_broadcast(&w->cond);
        pthread_mutex_unlock(&w->lock);

        pthread_mutex_lock(&job->lock);
        job->done = true;
        pthread_cond_signal(&job->cond);
        pthread_mutex_unlock(&job->lock);
    }
    // Legacy graph caches are thread-local to this sole worker, so preserve
    // same-thread teardown. In batch mode the engine coordinator owns teardown
    // and gen_worker_stop releases it after every dispatcher has joined.
    if (w->n_threads == 1 && w->be) ember_backend_free(w->be);
    return NULL;
}

// Start the worker. Returns false (leaving nothing to clean up) if any of the
// mutex/cond/thread primitives fail — the caller MUST abort startup, because a
// missing worker would make every gen_worker_submit() block forever.
static bool gen_worker_start(gen_worker *w, ember_backend *be,
                             double bg_idle_secs, double bg_max_wait_secs,
                             double idle_reclaim_secs, int n_threads) {
    w->head = w->tail = NULL;
    w->queued = 0;
    w->active_jobs = 0;
    w->be = be;  // #5: the worker owns backend teardown (runs on its own thread)
    w->n_threads = n_threads;
    w->stop = false;
    w->running = false;
    w->last_done_at = monotonic_now();
    w->graphs_dirty = false;
    w->reclaiming = false;
    w->idle_reclaim_secs = idle_reclaim_secs;
    ember_background_gate_init(&w->bg_gate, bg_idle_secs, bg_max_wait_secs);
    if (pthread_mutex_init(&w->lock, NULL) != 0) return false;
    pthread_condattr_t attr;
    if (pthread_condattr_init(&attr) != 0) {
        pthread_mutex_destroy(&w->lock);
        return false;
    }
    // Background eligibility uses CLOCK_MONOTONIC, so wall-clock corrections
    // cannot delay or prematurely release maintenance work.
    int attr_ok = pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
    int cond_ok = attr_ok == 0 ? pthread_cond_init(&w->cond, &attr) : attr_ok;
    pthread_condattr_destroy(&attr);
    if (cond_ok != 0) {
        pthread_mutex_destroy(&w->lock);
        return false;
    }
    int started = 0;
    for (; started < w->n_threads; ++started) {
        if (pthread_create(&w->threads[started], NULL,
                           gen_worker_main, w) == 0)
            continue;
        pthread_mutex_lock(&w->lock);
        w->stop = true;
        pthread_cond_broadcast(&w->cond);
        pthread_mutex_unlock(&w->lock);
        for (int j = 0; j < started; ++j)
            pthread_join(w->threads[j], NULL);
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
    pthread_cond_broadcast(&w->cond);
    pthread_mutex_unlock(&w->lock);
    for (int i = 0; i < w->n_threads; ++i)
        pthread_join(w->threads[i], NULL);
    if (w->n_threads > 1 && w->be) ember_backend_free(w->be);
    w->be = NULL;
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
    job.enqueued_at = monotonic_now();
    if (pthread_mutex_init(&job.lock, NULL) != 0) return false;
    if (pthread_cond_init(&job.cond, NULL) != 0) {
        pthread_mutex_destroy(&job.lock);
        return false;
    }

    pthread_mutex_lock(&w->lock);
    if (w->stop || w->queued >= EMBER_MAX_QUEUE_DEPTH) {
        pthread_mutex_unlock(&w->lock);
        pthread_mutex_destroy(&job.lock);
        pthread_cond_destroy(&job.cond);
        return false;
    }
    if (!req->background)
        ember_background_gate_note_foreground(&w->bg_gate, job.enqueued_at);
    // cppcheck-suppress autoVariables
    if (w->tail) w->tail->next = &job; else w->head = &job;
    // cppcheck-suppress autoVariables
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
    if (strcmp(req->method, "OPTIONS") == 0) {
        respond(fd, 204, NULL, "");
        return;
    }
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/health") == 0) {
        respond(fd, 200, "text/plain", "ok\n");
        return;
    }
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/v1/models") == 0) {
        ember_buf b = {0};
        ember_buf_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":");
        ember_json_escape(&b, ember_backend_model_name(be));
        ember_buf_printf(&b,
            ",\"object\":\"model\",\"owned_by\":\"otheru\","
            "\"max_context_length\":%d}]}",
            ember_backend_n_ctx(be));
        respond(fd, 200, "application/json", b.ptr);
        ember_buf_free(&b);
        return;
    }
    static const char model_prefix[] = "/v1/models/";
    if (strcmp(req->method, "GET") == 0 &&
        strncmp(req->path, model_prefix, sizeof(model_prefix) - 1) == 0) {
        const char *requested = req->path + sizeof(model_prefix) - 1;
        if (requested[0] &&
            strcmp(requested, ember_backend_model_name(be)) == 0) {
            ember_buf b = {0};
            ember_buf_puts(&b, "{\"id\":");
            ember_json_escape(&b, ember_backend_model_name(be));
            ember_buf_printf(&b,
                ",\"object\":\"model\",\"owned_by\":\"otheru\","
                "\"context_length\":%d,\"supported_parameters\":["
                "\"tools\",\"tool_choice\",\"max_tokens\",\"temperature\","
                "\"top_p\",\"top_k\",\"min_p\",\"stop\",\"seed\","
                "\"stream\",\"parallel_tool_calls\",\"reasoning_effort\","
                "\"reasoning_budget_tokens\"]}",
                ember_backend_n_ctx(be));
            respond(fd, 200, "application/json", b.ptr);
            ember_buf_free(&b);
        } else {
            respond(fd, 404, "application/json",
                "{\"error\":{\"message\":\"unknown model\"}}");
        }
        return;
    }
    bool is_chat = strcmp(req->path, "/v1/chat/completions") == 0;
    bool is_responses = strcmp(req->path, "/v1/responses") == 0;
    bool is_anthropic = strcmp(req->path, "/v1/messages") == 0;
    bool is_completion = strcmp(req->path, "/v1/completions") == 0;
    if (strcmp(req->method, "POST") == 0 &&
        (is_chat || is_responses || is_anthropic || is_completion)) {
        ember_json *root = ember_json_parse_n(req->body, req->body_len);
        ember_chat_request creq;
        char parse_err[192] = {0};
        bool parsed = root &&
            (is_responses
                ? ember_responses_request_parse(root, &creq,
                                                parse_err, sizeof(parse_err))
             : is_anthropic
                ? ember_anthropic_request_parse(root, &creq,
                                                parse_err, sizeof(parse_err))
             : is_completion
                ? ember_completion_request_parse(root, &creq,
                                                 parse_err, sizeof(parse_err))
                : ember_chat_request_parse(root, &creq));
        if (parsed) {
            creq.response_cors = g_enable_cors;
            // Run generation on the persistent worker (keeps the backend's
            // thread_local graph caches warm); block until it completes.
            // #3: a full worker queue sheds load with a 503 instead of blocking.
            if (!gen_worker_submit(&srv->worker, srv, &creq, fd)) {
                respond_api_error(fd, creq.api, 503,
                                  "server overloaded, retry later",
                                  "server_error", "overloaded");
            }
            ember_chat_request_free(&creq);
        } else {
            ember_api_kind api = is_anthropic ? EMBER_API_ANTHROPIC
                : is_responses ? EMBER_API_RESPONSES
                : is_completion ? EMBER_API_COMPLETIONS : EMBER_API_CHAT;
            respond_api_error(
                fd, api, 400,
                parse_err[0] ? parse_err
                    : "invalid JSON request or missing messages",
                "invalid_request_error", "invalid_request");
        }
        if (root) ember_json_free(root);
        return;
    }
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/status") == 0) {
        ember_buf b = {0};
        ember_batch_stats batch = {0};
        (void)ember_backend_batch_stats_get(be, &batch);
        int last_loop_rounds;
        bool last_loop_identical;
        long last_loop_at;
        long aa_count, aa_at, leak_count, leak_at;
        char aa_tool[sizeof(srv->last_auto_answer_tool)];
        long np_count, np_at;
        int np_tokens;
        bool np_degenerate;
        long lease_count, lease_at;
        int lease_rounds;
        char lease_tool[sizeof(srv->last_no_progress_tool)];
        long dg_count, dg_at;
        int dg_tokens;
        bool dg_output;
        char dg_reason[sizeof(srv->last_degenerate_reason)];
        char last_loop_tool[sizeof(srv->last_tool_loop_tool)];
        pthread_mutex_lock(&srv->state_lock);
        last_loop_rounds = srv->last_tool_loop_rounds;
        last_loop_identical = srv->last_tool_loop_identical;
        last_loop_at = srv->last_tool_loop_at;
        aa_count = srv->auto_answer_count;
        leak_count = srv->tool_markup_leak_count;
        leak_at = srv->last_tool_markup_leak_at;
        aa_at = srv->last_auto_answer_at;
        snprintf(aa_tool, sizeof(aa_tool), "%s", srv->last_auto_answer_tool);
        np_count = srv->nonprogress_count;
        np_at = srv->last_nonprogress_at;
        np_tokens = srv->last_nonprogress_tokens;
        np_degenerate = srv->last_nonprogress_degenerate;
        lease_count = srv->no_progress_count;
        lease_at = srv->last_no_progress_at;
        lease_rounds = srv->last_no_progress_rounds;
        snprintf(lease_tool, sizeof(lease_tool), "%s", srv->last_no_progress_tool);
        dg_count = srv->degenerate_count;
        dg_at = srv->last_degenerate_at;
        dg_tokens = srv->last_degenerate_tokens;
        dg_output = srv->last_degenerate_had_output;
        snprintf(dg_reason, sizeof(dg_reason), "%s", srv->last_degenerate_reason);
        snprintf(last_loop_tool, sizeof(last_loop_tool), "%s",
                 srv->last_tool_loop_tool);
        pthread_mutex_unlock(&srv->state_lock);
        ember_buf_puts(&b, "{\"model\":");
        ember_json_escape(&b, ember_backend_model_name(be));
        ember_buf_printf(&b,
            ",\"ctx\":%d,\"busy\":%d,\"served\":%ld,"
            "\"sampling_defaults\":{\"temperature\":%.9g,"
            "\"top_p\":%.9g,\"top_k\":%d,\"min_p\":%.9g,"
            "\"repetition_penalty\":%.9g,\"presence_penalty\":%.9g},"
            "\"tool_loop\":{\"report_after_repeats\":%d,\"last\":",
            ember_backend_n_ctx(be),
            atomic_load(&srv->busy), atomic_load(&srv->served),
            srv->default_temp, srv->card.top_p, srv->card.top_k,
            srv->card.min_p, srv->card.repetition_penalty,
            srv->card.presence_penalty, srv->tool_loop_report);
        if (last_loop_rounds > 0) {
            ember_buf_printf(&b,
                "{\"at\":%ld,\"rounds\":%d,\"tool\":",
                last_loop_at, last_loop_rounds);
            ember_json_escape(&b, last_loop_tool);
            ember_buf_printf(&b, ",\"identical_results\":%s}",
                             last_loop_identical ? "true" : "false");
        } else {
            ember_buf_puts(&b, "null");
        }
        ember_buf_putc(&b, '}');            // close tool_loop
        // Automatic loop recovery. armed_after mirrors the flag so an operator
        // can tell "never fired" from "not enabled".
        ember_buf_printf(&b,
            ",\"auto_answer\":{\"armed_after\":%d,\"count\":%ld,\"last\":",
            srv->auto_answer_after_loop, aa_count);
        if (aa_count > 0) {
            ember_buf_printf(&b, "{\"at\":%ld,\"tool\":", aa_at);
            ember_json_escape(&b, aa_tool);
            ember_buf_putc(&b, '}');
        } else {
            ember_buf_puts(&b, "null");
        }
        ember_buf_putc(&b, '}');            // close auto_answer
        ember_buf_printf(&b,
            ",\"tool_markup_leak\":{\"count\":%ld,\"last_at\":%ld}",
            leak_count, leak_count > 0 ? leak_at : 0L);
        // Turns that finished cleanly having emitted nothing at all: no visible
        // text and no tool call. Leading indicator of an agent losing the
        // ability to act -- see note_nonprogress_turn(). A SIBLING of
        // tool_loop, not a member of it.
        ember_buf_printf(&b, ",\"nonprogress\":{\"count\":%ld,\"last\":", np_count);
        if (np_count > 0)
            ember_buf_printf(&b,
                "{\"at\":%ld,\"completion_tokens\":%d,\"degenerate\":%s}",
                np_at, np_tokens, np_degenerate ? "true" : "false");
        else
            ember_buf_puts(&b, "null");
        ember_buf_putc(&b, '}');            // close nonprogress
        // Trailing tool rounds that returned nothing new. Orthogonal to
        // tool_loop, which keys on the call rather than the result: over the
        // regression corpus the two agree on 7 of 114 firing requests.
        // report_after mirrors the flag so "never fired" reads differently
        // from "not enabled".
        ember_buf_printf(&b,
            ",\"no_progress\":{\"report_after\":%d,\"count\":%ld,\"last\":",
            srv->no_progress_report, lease_count);
        if (lease_count > 0) {
            ember_buf_printf(&b, "{\"at\":%ld,\"rounds\":%d,\"tool\":",
                             lease_at, lease_rounds);
            ember_json_escape(&b, lease_tool);
            ember_buf_putc(&b, '}');
        } else {
            ember_buf_puts(&b, "null");
        }
        ember_buf_putc(&b, '}');            // close no_progress
        // Wider than nonprogress: every degenerate decode, output or not.
        ember_buf_printf(&b, ",\"degenerate\":{\"count\":%ld,\"last\":", dg_count);
        if (dg_count > 0) {
            ember_buf_printf(&b,
                "{\"at\":%ld,\"completion_tokens\":%d,\"produced_output\":%s,"
                "\"reason\":", dg_at, dg_tokens, dg_output ? "true" : "false");
            ember_json_escape(&b, dg_reason);
            ember_buf_putc(&b, '}');
        } else {
            ember_buf_puts(&b, "null");
        }
        ember_buf_putc(&b, '}');            // close degenerate
        ember_buf_printf(&b,
            ",\"continuous_batching\":{\"enabled\":%s,\"capacity\":%d,"
            "\"pending\":%d,\"resident\":%d,\"prefill_ready\":%d,"
            "\"decode_ready\":%d,\"in_flight\":%d,\"terminal\":%d,"
            "\"admissions\":%llu,\"releases\":%llu,"
            "\"submissions\":%llu,\"decode_batches\":%llu,"
            "\"decode_rows\":%llu,\"prefill_tokens\":%llu,"
            "\"mixed_submissions\":%llu,\"coalesce_waits\":%llu,"
            "\"backend_failures\":%llu,\"backend_exceptions\":%llu,"
            "\"max_decode_batch\":%d}}",
            batch.enabled ? "true" : "false", batch.capacity,
            batch.pending, batch.resident, batch.prefill_ready,
            batch.decode_ready, batch.in_flight, batch.terminal,
            (unsigned long long)batch.admissions,
            (unsigned long long)batch.releases,
            (unsigned long long)batch.submissions,
            (unsigned long long)batch.decode_batches,
            (unsigned long long)batch.decode_rows,
            (unsigned long long)batch.prefill_tokens,
            (unsigned long long)batch.mixed_submissions,
            (unsigned long long)batch.coalesce_waits,
            (unsigned long long)batch.backend_failures,
            (unsigned long long)batch.backend_exceptions,
            batch.max_decode_batch);
        respond(fd, 200, "application/json", b.ptr);
        ember_buf_free(&b);
        return;
    }
    respond(fd, 404, "application/json",
            "{\"error\":{\"message\":\"not found\"}}");
}

static void print_usage(FILE *out, const char *argv0) {
    fprintf(out,
        "Usage: %s -m MODEL [options]\n"
        "  --host IPV4                 listen address (default 127.0.0.1)\n"
        "  --port N                    listen port (default 8080)\n"
        "  --model-name ID             advertised model id\n"
        "  --model-card PATH           sampling/reasoning model card\n"
        "  --cors                      allow browser cross-origin requests\n"
        "  --auto-compact              ds4-style context compaction: at 85%% of\n"
        "                              context, rebuild history as system+summary+tail\n"
        "  --kv-cache-dir PATH         cross-restart KV cache directory\n"
        "  --kv-cache-mb MB            disk KV cache budget (0 = 131072 default)\n"
        "  --ds4-expert-top-k N        routed experts, 0=model default\n"
        "  --ds4-prefill MODE          exact, dense, or sparse (default sparse)\n"
        "                              dense aborts above ~50k context: only the\n"
        "                              sparse path bounds flash-attention shared\n"
        "                              memory by the indexer top-k\n"
        "  --default-temperature T     override model-card temperature\n"
        "  --tool-loop-report N        report after N identical call+result repeats (default 8)\n"
        "  --no-progress-report N      report after N tool rounds return nothing new (default 8)\n"
        "  --auto-answer-after-loop N  suppress tools for one turn once a request's\n"
        "                              history shows >N identical trailing calls\n"
        "                              (0=off; BEHAVIOURAL, not diagnostic)\n"
        "  --prefix-cache-slots N      resident prefix snapshots (default 8)\n"
        "  --batch-sessions N          resident concurrent sessions (default 1,\n"
        "                              max 64; >1 enables continuous batching)\n"
        "  --max-ctx N                 context length (default 131072)\n"
        "  --validate-prompt PATH      run AR/snapshot/DSpark/disk differential and exit\n"
        "  --validate-gemm-batch N     sweep HIP strided-batched GEMM counts 2..N\n"
        "                              against the one-row baseline, then exit\n"
        "                              (needs no model; safe alongside serving)\n"
        "  --validate-tokens N         tokens compared by validation (default 32)\n",
        argv0);
}

static const char *need_option_value(int *i, int argc, char **argv) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "[ember] missing value for %s\n", argv[*i]);
        return NULL;
    }
    return argv[++(*i)];
}

static bool parse_int_range(const char *s, const char *name,
                            long min, long max, int *out) {
    char *end = NULL;
    errno = 0;
    long v = s ? strtol(s, &end, 10) : 0;
    if (!s || !s[0] || errno || !end || *end || v < min || v > max) {
        fprintf(stderr, "[ember] invalid value for %s: %s\n",
                name, s ? s : "(missing)");
        return false;
    }
    *out = (int)v;
    return true;
}

static bool parse_double_range(const char *s, const char *name,
                               double min, double max, double *out) {
    char *end = NULL;
    errno = 0;
    double v = s ? strtod(s, &end) : 0.0;
    if (!s || !s[0] || errno || !end || *end || !isfinite(v) ||
        v < min || v > max) {
        fprintf(stderr, "[ember] invalid value for %s: %s\n",
                name, s ? s : "(missing)");
        return false;
    }
    *out = v;
    return true;
}

static bool env_seconds(const char *name, double fallback, double *out) {
    const char *s = getenv(name);
    if (!s || !s[0]) {
        *out = fallback;
        return true;
    }
    return parse_double_range(s, name, 0.0, 86400.0, out);
}

// Sweep HIP strided-batched GEMM batch counts against the one-row baseline.
// The result bounds any future cross-session batched decode: above max_exact,
// batching changes logits or faults and cannot replace the serial operation.
static int run_gemm_batch_sweep(ember_backend *be, int limit) {
    ember_gemm_batch_report r;
    if (!ember_backend_validate_gemm_batch(be, limit, &r)) {
        fprintf(stderr, "[ember-gemm-sweep] did not run: %s\n",
                r.detail[0] ? r.detail : "unknown");
        return 2;
    }
    printf("gemm batch sweep: shapes=%d limit=%d max_exact=%d",
           r.shapes_tested, r.limit, r.max_exact);
    if (r.first_divergent) printf(" first_divergent=%d", r.first_divergent);
    if (r.first_fault) printf(" first_fault=%d", r.first_fault);
    if (r.worst_rel > 0) printf(" worst_rel=%.3g", r.worst_rel);
    printf("\n  %s\n", r.detail);
    if (r.max_exact < 2) {
        printf("  VERDICT: no bit-exact batching available (max_exact=1)\n");
        return 1;
    }
    printf("  VERDICT: batched decode may use batchCount <= %d\n", r.max_exact);
    return 0;
}

static int run_backend_validation(ember_backend *be, const char *path,
                                  int n_gen) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ember-validate] cannot open %s: %s\n",
                path, strerror(errno));
        return 2;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 2;
    }
    long raw_len = ftell(f);
    if (raw_len <= 0 || raw_len > 64L * 1024L * 1024L ||
        fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr,
                "[ember-validate] prompt must be between 1 byte and 64 MiB\n");
        fclose(f);
        return 2;
    }
    char *raw = (char *)malloc((size_t)raw_len + 1);
    if (!raw || fread(raw, 1, (size_t)raw_len, f) != (size_t)raw_len) {
        fprintf(stderr, "[ember-validate] failed to read prompt\n");
        free(raw);
        fclose(f);
        return 2;
    }
    fclose(f);
    raw[raw_len] = '\0';

    int32_t *ids = NULL;
    int n_prompt = ember_backend_encode(be, raw, &ids);
    free(raw);
    if (n_prompt < 1 || !ids) {
        fprintf(stderr, "[ember-validate] prompt tokenization failed\n");
        free(ids);
        return 2;
    }
    if (n_prompt + n_gen > ember_backend_n_ctx(be)) {
        fprintf(stderr,
                "[ember-validate] prompt (%d) + validation tokens (%d) exceed ctx=%d\n",
                n_prompt, n_gen, ember_backend_n_ctx(be));
        free(ids);
        return 2;
    }

    ember_validation_report report;
    bool ran =
        ember_backend_validate(be, ids, n_prompt, n_gen, &report);
    free(ids);
    if (!ran) {
        fprintf(stderr, "[ember-validate] backend rejected validation request\n");
        return 2;
    }

    ember_buf out = {0};
    ember_buf_printf(
        &out,
        "{\"ok\":%s,\"prompt_tokens\":%d,\"requested_tokens\":%d,"
        "\"snapshot_ok\":%s,\"baseline_tokens\":%d,"
        "\"spec\":{\"checked\":%s,\"exact\":%s,\"tokens\":%d,"
        "\"accept_rate\":%.6f},"
        "\"disk\":{\"checked\":%s,\"exact\":%s,\"tokens\":%d},"
        "\"batch\":{\"checked\":%s,\"exact\":%s,\"rows\":%d,\"tokens\":%d,"
        "\"spec_required\":%s,\"spec_rows\":%d,"
        "\"spec_accept_rate\":%.6f},"
        "\"mismatch\":{\"index\":%d,\"expected\":%d,\"actual\":%d},"
        "\"detail\":",
        report.ok ? "true" : "false", n_prompt, n_gen,
        report.snapshot_ok ? "true" : "false", report.baseline_tokens,
        report.spec_checked ? "true" : "false",
        report.spec_exact ? "true" : "false", report.spec_tokens,
        report.spec_accept_rate,
        report.disk_checked ? "true" : "false",
        report.disk_exact ? "true" : "false", report.disk_tokens,
        report.batch_checked ? "true" : "false",
        report.batch_exact ? "true" : "false",
        report.batch_rows, report.batch_tokens,
        report.batch_spec_required ? "true" : "false",
        report.batch_spec_rows, report.batch_spec_accept_rate,
        report.mismatch_index, report.expected_token, report.actual_token);
    ember_json_escape(&out, report.detail);
    ember_buf_puts(&out, "}\n");
    fwrite(out.ptr, 1, out.len, stdout);
    ember_buf_free(&out);
    return report.ok ? 0 : 1;
}

static volatile sig_atomic_t g_shutdown_signal;

static void shutdown_signal_handler(int sig) {
    if (g_shutdown_signal) _exit(128 + sig);
    g_shutdown_signal = sig;
    ember_http_request_stop();
}

static bool install_shutdown_handlers(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = shutdown_signal_handler;
    sigemptyset(&sa.sa_mask);
    return sigaction(SIGINT, &sa, NULL) == 0 &&
           sigaction(SIGTERM, &sa, NULL) == 0;
}

int main(int argc, char **argv) {
    int port = 8080;
    const char *host = "127.0.0.1";
    const char *model_path = NULL, *model_name = "deepseek-v4-flash";
    const char *card_path = NULL, *kv_dir = NULL;
    long kv_cache_mb = 0;   // 0 = library default (131072 MB)
    // Seconds of quiet before cached compute graphs are released. Long enough
    // that an active agent (which pauses seconds-to-minutes between turns)
    // never pays the rebuild, short enough that an idle box gives the memory
    // back. EMBER_IDLE_RECLAIM_SECS=0 disables.
    double idle_reclaim_secs = 300.0;
    const char *validate_prompt = NULL;
    int validate_tokens = 32;
    int validate_gemm_batch = 0;
    // Preserve the measured lucebox deployment override: its older DSpark
    // pairing accepted substantially better with target top-k=4 (~0.95 vs
    // ~0.68) and avoided two expert evaluations per layer. The model and the
    // current drafter both advertise 6, so this is a performance calibration,
    // not an architectural match; revalidate it when changing either file.
    int expert_top_k = 4;
    ember_ds4_prefill_mode ds4_prefill_mode = EMBER_DS4_PREFILL_SPARSE;
    double default_temp = 0.6;
    bool default_temp_set = false;
    // Opt-in: compaction rewrites what a generation is conditioned on, so it is
    // never on by default. A bad summary costs the model its task state.
    bool auto_compact = false;
    // In-memory KV prefix-cache slots. Each committed slot holds a full-KV
    // snapshot, which tracks the KV cache itself: the engine reports 877.8 MB
    // at ctx=131072 (measured), so ~448MB at the old 65536 default. Eight slots
    // retain ~7GiB at the current default; callers may lower this on
    // memory-tight UMA systems with --prefix-cache-slots.
    int prefix_slots = 8;
    int batch_sessions = 1;
    // A value of 8 reports the ninth identical round. Lower thresholds produced
    // noisy reports for loops that resolved when the model changed strategy.
    // Reporting is additive and never caps tool rounds.
    int tool_loop_report = 8;
    // The progress lease has its own threshold: it keys on results rather than
    // calls, so its distribution differs and tuning one must not move the
    // other. Its default is also 8 so operators see the two advisory thresholds
    // at a consistent severity level.
    int no_progress_report = 8;
    int auto_answer_after_loop = 0;      // off: this one changes behaviour
    // The model advertises deepseek4.context_length = 1048576; 65536 was a
    // server-side default, not a model limit. Compressed MLA keeps the cache
    // tiny -- measured 877.8 MB at ctx=131072, because most layers compress 4:1
    // or 128:1 and the raw ring is a fixed 128 rows regardless of context. A
    // load at 131072 was verified on the 128 GB box: GTT 11.2 GiB, 24 GiB host
    // free. Raising further is mostly a prefix-slot budgeting question (each
    // snapshot scales with this value), not a KV or model one.
    int max_ctx = 131072;  // KV cache context; each snapshot is a full-KV buffer
    double bg_idle_secs = 5.0, bg_max_wait_secs = 60.0;
    bool options_ok = true;
    for (int i = 1; i < argc; i++) {
        const char *opt = argv[i], *v = NULL;
        if (strcmp(opt, "--help") == 0 || strcmp(opt, "-h") == 0) {
            print_usage(stdout, argv[0]);
            return 0;
        } else if (strcmp(opt, "--port") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v && parse_int_range(v, opt, 1, 65535, &port);
        } else if (strcmp(opt, "--host") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v && ember_http_host_valid(v);
            if (v && !options_ok)
                fprintf(stderr, "[ember] invalid IPv4 address for %s: %s\n",
                        opt, v);
            if (options_ok) host = v;
        } else if (strcmp(opt, "-m") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v != NULL;
            if (v) model_path = v;
        } else if (strcmp(opt, "--model-name") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v != NULL;
            if (v) model_name = v;
        } else if (strcmp(opt, "--model-card") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v != NULL;
            if (v) card_path = v;
        } else if (strcmp(opt, "--cors") == 0) {
            g_enable_cors = true;
        } else if (strcmp(opt, "--auto-compact") == 0) {
            auto_compact = true;
        } else if (strcmp(opt, "--kv-cache-dir") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v != NULL;
            if (v) kv_dir = v;
        } else if (strcmp(opt, "--kv-cache-mb") == 0) {
            // Disk budget for the cross-restart KV cache. The library default
            // is 131072 MB. Entries are scored by usefulness and age, so this
            // can be raised on hosts with additional storage without merely
            // hoarding stale snapshots. A useful retained entry can turn a
            // multi-minute cold prefill into a restore.
            v = need_option_value(&i, argc, argv);
            options_ok = v != NULL;
            if (v) {
                char *endp = NULL;
                errno = 0;
                long mb = strtol(v, &endp, 10);
                const uint64_t max_mb =
                    (uint64_t)SIZE_MAX / (1024u * 1024u);
                if (errno || endp == v || *endp != '\0' || mb < 0 ||
                    (uint64_t)mb > max_mb) {
                    fprintf(stderr, "ember: --kv-cache-mb expects a non-negative integer, got '%s'\n", v);
                    options_ok = false;
                } else {
                    kv_cache_mb = mb;
                }
            }
        } else if (strcmp(opt, "--ds4-expert-top-k") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v && parse_int_range(v, opt, 0, 256, &expert_top_k);
        } else if (strcmp(opt, "--ds4-prefill") == 0) {
            v = need_option_value(&i, argc, argv);
            if (!v) {
                options_ok = false;
            } else if (strcmp(v, "exact") == 0) {
                ds4_prefill_mode = EMBER_DS4_PREFILL_EXACT;
            } else if (strcmp(v, "dense") == 0) {
                ds4_prefill_mode = EMBER_DS4_PREFILL_DENSE;
            } else if (strcmp(v, "sparse") == 0) {
                ds4_prefill_mode = EMBER_DS4_PREFILL_SPARSE;
            } else {
                fprintf(stderr,
                        "[ember] --ds4-prefill expects exact, dense, or sparse, got '%s'\n",
                        v);
                options_ok = false;
            }
        } else if (strcmp(opt, "--default-temperature") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v && parse_double_range(v, opt, 0.0, 100.0, &default_temp);
            default_temp_set = options_ok;
        } else if (strcmp(opt, "--auto-answer-after-loop") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v &&
                parse_int_range(v, opt, 0, INT_MAX, &auto_answer_after_loop);
        } else if (strcmp(opt, "--tool-loop-report") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v &&
                parse_int_range(v, opt, 0, INT_MAX, &tool_loop_report);
        } else if (strcmp(opt, "--no-progress-report") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v &&
                parse_int_range(v, opt, 0, INT_MAX, &no_progress_report);
        } else if (strcmp(opt, "--prefix-cache-slots") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v && parse_int_range(v, opt, 1,
                                               EMBER_KV_MAX_SLOTS - 1,
                                               &prefix_slots);
        } else if (strcmp(opt, "--batch-sessions") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v &&
                parse_int_range(v, opt, 1, EMBER_MAX_BATCH_SESSIONS,
                                &batch_sessions);
        } else if (strcmp(opt, "--max-ctx") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v && parse_int_range(v, opt, 1, INT_MAX, &max_ctx);
        } else if (strcmp(opt, "--validate-gemm-batch") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v &&
                parse_int_range(v, opt, 2, 256, &validate_gemm_batch);
        } else if (strcmp(opt, "--validate-prompt") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v != NULL;
            if (v) validate_prompt = v;
        } else if (strcmp(opt, "--validate-tokens") == 0) {
            v = need_option_value(&i, argc, argv);
            options_ok = v &&
                parse_int_range(v, opt, 2, 4096, &validate_tokens);
        } else {
            fprintf(stderr, "[ember] unknown option: %s\n", opt);
            options_ok = false;
        }
        if (!options_ok) {
            print_usage(stderr, argv[0]);
            return 2;
        }
    }
    // The GEMM sweep measures a HIP library property, not a model property.
    // Running it before model load keeps the standalone form usable without
    // allocating the model weights.
    if (validate_gemm_batch > 0 && !validate_prompt)
        return run_gemm_batch_sweep(NULL, validate_gemm_batch);
    if (!model_path || !model_path[0]) {
        fprintf(stderr, "[ember] -m MODEL is required\n");
        print_usage(stderr, argv[0]);
        return 2;
    }
    if (strcmp(host, "127.0.0.1") != 0)
        fprintf(stderr,
                "[ember] WARNING: listening beyond loopback; Ember has no "
                "built-in authentication\n");
    if (!env_seconds("EMBER_BG_IDLE_SECS", 5.0, &bg_idle_secs) ||
        !env_seconds("EMBER_BG_MAX_WAIT_SECS", 60.0, &bg_max_wait_secs))
        return 2;
    const char *trace_tokens = getenv("EMBER_TRACE_TOKENS");
    g_trace_tokens = trace_tokens && trace_tokens[0] &&
                     strcmp(trace_tokens, "0") != 0;
    ember_backend_config cfg = {0};
    cfg.model_path = model_path;
    cfg.model_name = model_name;
    cfg.max_ctx = max_ctx > 0 ? max_ctx : 131072;
    cfg.expert_top_k = expert_top_k;
    cfg.kv_cache_dir = kv_dir;
    cfg.kv_cache_mb = kv_cache_mb;   // --kv-cache-mb, 0 = default
    cfg.batch_sessions = batch_sessions;
    cfg.ds4_prefill_mode = ds4_prefill_mode;
    char *err = NULL;
    ember_backend *be = ember_backend_load(&cfg, &err);
    if (!be) {
        fprintf(stderr, "[ember] backend load failed: %s\n", err ? err : "?");
        free(err);
        return 1;
    }
    free(err);
    if (validate_gemm_batch > 0) {
        const int rc = run_gemm_batch_sweep(be, validate_gemm_batch);
        if (rc > 1) {
            ember_backend_free(be);
            return rc;
        }
    }
    if (validate_prompt) {
        int rc =
            run_backend_validation(be, validate_prompt, validate_tokens);
        ember_backend_free(be);
        return rc;
    }
    ember_server srv = {0};
    srv.be = be;
    srv.auto_compact = auto_compact;
    srv.tool_loop_report = tool_loop_report;
    srv.no_progress_report = no_progress_report;
    srv.auto_answer_after_loop = auto_answer_after_loop;
    if (pthread_mutex_init(&srv.gen_lock, NULL) != 0) {
        fprintf(stderr, "[ember] failed to initialize generation lock\n");
        ember_backend_free(be);
        return 1;
    }
    if (pthread_mutex_init(&srv.state_lock, NULL) != 0) {
        fprintf(stderr, "[ember] failed to initialize shared-state lock\n");
        pthread_mutex_destroy(&srv.gen_lock);
        ember_backend_free(be);
        return 1;
    }
    ember_model_card_load(&srv.card, card_path);
    srv.default_temp = default_temp_set ? default_temp : srv.card.temperature;
    ember_kv_init(&srv.kv, prefix_slots,
                  marker_id(be, "<" PIPE "User" PIPE ">"),
                  marker_id(be, "<" PIPE "Assistant" PIPE ">"),
                  ember_backend_eos_id(be));
    ember_tool_memory_init(&srv.tool_mem, 512, 64u * 1024u * 1024u);
    ember_continuation_init(&srv.continuations, 64,
                            128u * 1024u * 1024u);
    if (kv_dir && kv_dir[0]) {
        uint8_t cache_identity[16];
        if (ember_backend_cache_identity(be, cache_identity)) {
            int loaded = ember_tool_memory_enable_persistence(
                &srv.tool_mem, kv_dir, cache_identity);
            if (loaded >= 0) {
                fprintf(stderr,
                        "[ember] exact tool replay persistence enabled"
                        " (%d mappings restored)\n",
                        loaded);
            } else {
                fprintf(stderr,
                        "[ember] warning: exact tool replay persistence"
                        " unavailable; continuing in RAM only\n");
            }
            int continuations = ember_continuation_enable_persistence(
                &srv.continuations, kv_dir, cache_identity);
            if (continuations >= 0) {
                fprintf(stderr,
                        "[ember] durable continuation bindings enabled"
                        " (%d frontiers restored)\n",
                        continuations);
            } else {
                fprintf(stderr,
                        "[ember] warning: durable continuation bindings"
                        " unavailable; continuing in RAM only\n");
            }
        }
    }
    // Resolve the Level-2 force-close token sequence once at startup: the
    // card's terminator hint (a "wrap up now" directive + </think>) if present,
    // else a bare </think>. Encoded via the real tokenizer so the backend can
    // inject it verbatim at the reply-budget edge.
    const char *close_text =
        (srv.card.thinking_terminator_hint && srv.card.thinking_terminator_hint[0])
            ? srv.card.thinking_terminator_hint
            : "</think>";
    srv.n_close_ids = ember_backend_encode(be, close_text, &srv.close_ids);
    srv.n_natural_close_ids =
        ember_backend_encode(be, "</think>", &srv.natural_close_ids);
    fprintf(stderr,
            "[ember] backend ready: %s (ctx=%d, reply_reserve=%d, "
            "close_seq=%d tok, natural_close_seq=%d tok, "
            "sampler={temp=%.3g,top_p=%.3g,top_k=%d,min_p=%.3g,"
            "rep_pen=%.3g,pres_pen=%.3g}, tool_loop_report=%d, "
            "no_progress_report=%d, "
            "batch_sessions=%d, bg_idle=%.1fs, "
            "bg_max_wait=%.1fs)\n",
            ember_backend_model_name(be), ember_backend_n_ctx(be),
            srv.card.hard_limit_reply_budget, srv.n_close_ids,
            srv.n_natural_close_ids, srv.default_temp, srv.card.top_p,
            srv.card.top_k, srv.card.min_p, srv.card.repetition_penalty,
            srv.card.presence_penalty, srv.tool_loop_report,
            srv.no_progress_report, batch_sessions,
            bg_idle_secs, bg_max_wait_secs);
    // Start the persistent generation worker before accepting connections: all
    // backend forward passes run on it so the thread_local graph caches build
    // once and stay warm (see gen_worker above). If it can't start, abort —
    // serving anyway would wedge every request forever in gen_worker_submit().
    if (!env_seconds("EMBER_IDLE_RECLAIM_SECS", idle_reclaim_secs,
                     &idle_reclaim_secs)) {
        ember_kv_free(&srv.kv);
        ember_tool_memory_free(&srv.tool_mem);
        ember_continuation_free(&srv.continuations);
        ember_model_card_free(&srv.card);
        free(srv.close_ids);
        free(srv.natural_close_ids);
        ember_backend_free(be);
        pthread_mutex_destroy(&srv.state_lock);
        pthread_mutex_destroy(&srv.gen_lock);
        return 2;
    }
    if (idle_reclaim_secs > 0.0)
        fprintf(stderr, "[ember] idle graph reclaim after %.0fs quiet\n", idle_reclaim_secs);
    if (!gen_worker_start(&srv.worker, be, bg_idle_secs, bg_max_wait_secs,
                          idle_reclaim_secs, batch_sessions)) {
        fprintf(stderr, "[ember] failed to start generation worker\n");
        ember_kv_free(&srv.kv);
        ember_tool_memory_free(&srv.tool_mem);
        ember_continuation_free(&srv.continuations);
        ember_model_card_free(&srv.card);
        free(srv.close_ids);
        free(srv.natural_close_ids);
        ember_backend_free(be);  // worker never ran → no worker TLS; free here
        pthread_mutex_destroy(&srv.state_lock);
        pthread_mutex_destroy(&srv.gen_lock);
        return 1;
    }
    if (!install_shutdown_handlers()) {
        fprintf(stderr, "[ember] failed to install shutdown signal handlers\n");
        gen_worker_stop(&srv.worker);
        ember_kv_free(&srv.kv);
        ember_tool_memory_free(&srv.tool_mem);
        ember_continuation_free(&srv.continuations);
        ember_model_card_free(&srv.card);
        free(srv.close_ids);
        free(srv.natural_close_ids);
        pthread_mutex_destroy(&srv.state_lock);
        pthread_mutex_destroy(&srv.gen_lock);
        return 1;
    }
    int rc = ember_http_serve(host, port, handler, &srv);
    // HTTP shutdown has stopped acceptance and drained connection threads.
    // gen_worker_stop now drains the queue and frees the backend as its LAST
    // action on its owning thread, where its thread_local graph caches live.
    gen_worker_stop(&srv.worker);
    ember_kv_free(&srv.kv);
    ember_tool_memory_free(&srv.tool_mem);
    ember_continuation_free(&srv.continuations);
    ember_model_card_free(&srv.card);
    free(srv.close_ids);
    free(srv.natural_close_ids);
    pthread_mutex_destroy(&srv.state_lock);
    pthread_mutex_destroy(&srv.gen_lock);
    return rc;
}
