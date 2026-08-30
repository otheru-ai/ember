#include "sse.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../common/json_util.h"
#include "../common/utf8.h"
#include "../model/tool_parser.h"
#include "../model/tool_parser_qwen4.h"

// B#5: mint a RANDOM, per-call unique id (mirrors ds4 random_tool_id) instead
// of hashing name+index. The old FNV hash gave e.g. the first `bash` call an
// identical id in every response; a client correlating tool results across
// requests would then collide. 128 bits from /dev/urandom, hex-encoded, with
// the `call_` prefix. All SSE emission runs on the single generation worker
// thread in legacy mode, while resident batching may use several workers. The
// fallback counter is atomic so even a /dev/urandom failure cannot collide.
// once per call (in the start delta), so it is stable across that call's deltas.
static bool sse_random_bytes(void *dst, size_t len) {
    unsigned char *p = (unsigned char *)dst;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) return false;
    while (len) {
        ssize_t n = read(fd, p, len);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { close(fd); return false; }
        p += (size_t)n; len -= (size_t)n;
    }
    close(fd);
    return true;
}
static void gen_call_id(char *out, size_t cap) {
    static _Atomic uint64_t fallback_ctr;
    unsigned char bytes[16];
    size_t pos = (size_t)snprintf(out, cap, "%s", "call_");
    if (pos >= cap) return;
    if (!sse_random_bytes(bytes, sizeof(bytes))) {
        uint64_t a = ((uint64_t)time(NULL) << 32) ^ (uint64_t)getpid();
        uint64_t b = atomic_fetch_add(&fallback_ctr, 1) + 1;
        b ^= (uint64_t)(uintptr_t)out;
        memcpy(bytes, &a, sizeof(a));
        memcpy(bytes + sizeof(a), &b, sizeof(b));
    }
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(bytes) && pos + 2 < cap; i++) {
        out[pos++] = hex[bytes[i] >> 4];
        out[pos++] = hex[bytes[i] & 15];
    }
    out[pos] = '\0';
}

// ─── tool-call markers ──────────────────────────────────────────────────
// DeepSeek-V4 emits native DSML with U+FF5C (｜, bytes EF BD 9C). Converted
// GGUFs damage it inconsistently, so we recognize all three spellings ds4/
// lucebox learned the hard way: full, short (leading "<｜" eaten as a special
// token prefix), and ascii (｜ flattened to '?').
#define PIPE "\xef\xbd\x9c"  // U+FF5C fullwidth vertical line (adjacent-string
                             // concatenation stops \x from eating 'D'/hex bytes)
// Whether a content slice carries anything the client would render. Trailing
// "\n\n" around a suppressed tool block is the common case and must not count.
static bool slice_has_visible(const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r')
            return true;
    }
    return false;
}

static const char *const TOOL_STARTS[] = {
    "<" PIPE "DSML" PIPE "tool_calls>",  // <｜DSML｜tool_calls>
    "<DSML" PIPE "tool_calls>",          // <DSML｜tool_calls>  (short spelling)
    "<?DSML?tool_calls>",                // ascii-degraded
    "<tool_calls>",                      // plain-XML (parser has a matching family)
    "<ds_engine_tool_use>",              // model's native format (see tool_parser.c)
    "<tool_call>",                       // Qwen qwen3_coder format
};
static const size_t N_TOOL_STARTS =
    sizeof(TOOL_STARTS) / sizeof(TOOL_STARTS[0]);

const char *ember_find_tool_start(const char *s) {
    const char *best = NULL;
    for (size_t i = 0; i < N_TOOL_STARTS; i++) {
        const char *hit = strstr(s, TOOL_STARTS[i]);
        if (hit && (!best || hit < best)) best = hit;
    }
    return best;
}

// Tool-calls END markers (closers of the openers above). Returns a pointer to
// the byte AFTER the earliest complete end marker, or NULL. ds4 halts decoding
// there so the model can't ramble after the call.
static const char *const TOOL_ENDS[] = {
    "</" PIPE "DSML" PIPE "tool_calls>",
    "</DSML" PIPE "tool_calls>",
    "</?DSML?tool_calls>",
    "</tool_calls>",
    "</ds_engine_tool_use>",
    "</tool_call>",
};
const char *ember_find_tool_end(const char *s) {
    if (!s) return NULL;
    const char *start = NULL;
    size_t family = 0;
    for (size_t i = 0; i < N_TOOL_STARTS; i++) {
        const char *hit = strstr(s, TOOL_STARTS[i]);
        if (hit && (!start || hit < start)) {
            start = hit;
            family = i;
        }
    }
    if (!start) return NULL;
    const char *end = strstr(start + strlen(TOOL_STARTS[family]), TOOL_ENDS[family]);
    return end ? end + strlen(TOOL_ENDS[family]) : NULL;
}

// ─── content holdback (ds4 text_stream_safe_limit) ──────────────────────
// Scanned per emitted slice. A marker split ACROSS two slices is not sought:
// ember_text_safe_limit() never cuts inside a partial marker, test_sse.c fuzzes
// every chunk size from 1 to prove it, and an overlap guarding that case was
// removed after mutation testing showed nothing could exercise it.
static bool slice_has_tool_markup(const char *s, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (s[i] != '<') continue;
        size_t p = i + 1;
        if (p < n && s[p] == '/') ++p;
        if (p >= n || isalnum((unsigned char)s[p])) continue;
        ++p;                                              // the delimiter byte
        while (p < n && !isalnum((unsigned char)s[p])) ++p;   // multi-byte U+FF5C
        if (n - p >= 4 && !memcmp(s + p, "DSML", 4)) return true;
    }
    return false;
}

bool ember_text_has_tool_markup(const char *s) {
    return s && (slice_has_tool_markup(s, strlen(s)) ||
                 strstr(s, "<tool_call>") != NULL);
}

bool ember_sse_delivered_tool_markup(const ember_sse_stream *st) {
    return st && st->delivered_tool_markup;
}

size_t ember_text_safe_limit(const char *raw, size_t start, size_t raw_len,
                             bool has_tools, bool final) {
    if (raw_len <= start) return raw_len;
    size_t limit = raw_len;
    // NOT gated on has_tools, deliberately. It used to be, on the assumption
    // that a request advertising no tools cannot produce tool markup. That is
    // false: a turn whose tools had been withdrawn mid-request emitted a complete
    // <?DSML?tool_calls> block -- an explicitly supported syntax family, not a
    // malformed imitation -- and because this holdback was skipped, its
    // sensitive tool arguments streamed to the user as ordinary content.
    //
    // The end-of-turn parse (parse_executable_tool_calls) has never been gated
    // on has_tools, so it DID recognise the call and reject it -- but by then
    // the bytes were sent, and ember_sse_discard_tool_block had no tool_start
    // to discard because this function never set one. Detection and parsing
    // must agree on when markup is possible: always.
    //
    // Cost when no tools are advertised: one strstr per update over the
    // trailing window, and up to 80 bytes of tail held until the final update.
    // (void) on the parameter keeps the ABI and callers unchanged.
    (void)has_tools;
    {
        const char *tool = ember_find_tool_start(raw + start);
        if (tool) {
            limit = (size_t)(tool - raw);
            // ds4 trim_tool_separator_ws: drop whitespace before the marker so
            // the "\n\n" separator never leaks into content (the client would
            // echo it and the next turn's render would double it).
            while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;
            return ember_utf8_stream_safe_len(raw, start, limit, true);
        }
        if (!final) {
            // Hold trailing whitespace (DSML separator is syntax, not text)…
            while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;
            // …and everything from the last '<' in the trailing window: it may
            // be the start of a marker split across tokens (every marker
            // begins with '<'). 80-byte scan matches ds4.
            const size_t kScan = 80;
            size_t scan = raw_len - start > kScan ? raw_len - kScan : start;
            for (size_t i = raw_len; i > scan; i--) {
                if (raw[i - 1] == '<') {
                    if (i - 1 < limit) limit = i - 1;
                    break;
                }
            }
        }
    }
    return ember_utf8_stream_safe_len(raw, start, limit, final);
}

// ─── delta emission (OpenAI chat.completion.chunk) ──────────────────────
static void sse_chat_delta_n(ember_sse_stream *st, const char *field,
                             const char *text, size_t len, ember_buf *out) {
    if (len == 0) return;
    if (st->sink.text_delta) {
        st->sink.text_delta(st->sink_ud,
                            strcmp(field, "reasoning_content") == 0,
                            text, len, out);
        return;
    }
    ember_buf_printf(out,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":",
        st->id, st->created);
    ember_json_escape(out, st->model);
    ember_buf_puts(out, ",\"choices\":[{\"index\":0,\"delta\":{");
    ember_json_escape(out, field);
    ember_buf_putc(out, ':');
    ember_json_escape_n(out, text, len);
    ember_buf_puts(out, "},\"finish_reason\":null}]}\n\n");
}

// ─── exported API ───────────────────────────────────────────────────────
static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (!p) ember_buf_fatal("out of memory initializing SSE stream");
    memcpy(p, s, n);
    return p;
}

void ember_sse_init(ember_sse_stream *st, const char *id, const char *model,
                    long created, bool has_tools, bool started_in_thinking,
                    bool enable_cors) {
    memset(st, 0, sizeof(*st));
    st->id = xstrdup(id);
    st->model = xstrdup(model);
    st->created = created;
    st->has_tools = has_tools;
    st->enable_cors = enable_cors;
    st->mode = started_in_thinking ? EMBER_SSE_THINKING : EMBER_SSE_TEXT;
    st->tool_idx = -1;  // no tool call started yet
}

void ember_sse_free(ember_sse_stream *st) {
    free(st->id);
    free(st->model);
    free(st->reasoning_filter);
    for (int i = 0; i < st->n_tool_ids; i++) free(st->tool_ids[i]);  // B3
    free(st->tool_ids);
    st->tool_ids = NULL;
    st->n_tool_ids = 0;
    st->tool_ids_cap = 0;
    st->id = st->model = NULL;
    st->reasoning_filter = NULL;
    st->reasoning_filter_len = 0;
}

void ember_sse_set_sink(ember_sse_stream *st, const ember_sse_sink *sink,
                        void *ud) {
    if (!st) return;
    if (sink) st->sink = *sink;
    else memset(&st->sink, 0, sizeof(st->sink));
    st->sink_ud = ud;
}

void ember_sse_set_qwen_tools(ember_sse_stream *st, const char *tools_json) {
    if (!st) return;
    st->qwen_tool_syntax = true;
    st->tools_json = tools_json;
}

void ember_sse_set_reasoning_filter(ember_sse_stream *st, const char *hint) {
    if (!st) return;
    free(st->reasoning_filter);
    st->reasoning_filter = NULL;
    st->reasoning_filter_len = 0;
    if (!hint || !hint[0]) return;
    const char *close = strstr(hint, "</think>");
    size_t n = close ? (size_t)(close - hint) : strlen(hint);
    if (n == 0) return;
    st->reasoning_filter = (char *)malloc(n + 1);
    if (!st->reasoning_filter)
        ember_buf_fatal("out of memory setting reasoning filter");
    memcpy(st->reasoning_filter, hint, n);
    st->reasoning_filter[n] = '\0';
    st->reasoning_filter_len = n;
}

void ember_sse_reset_attempt(ember_sse_stream *st,
                             bool started_in_thinking) {
    if (!st) return;
    for (int i = 0; i < st->n_tool_ids; ++i) {
        free(st->tool_ids[i]);
        st->tool_ids[i] = NULL;
    }
    st->n_tool_ids = 0;
    st->mode =
        started_in_thinking ? EMBER_SSE_THINKING : EMBER_SSE_TEXT;
    st->emit_pos = 0;
    st->checked_think_prefix = false;
    st->delivered_tool_markup = false;
    st->tool_start = 0;
    st->tool_idx = -1;
    st->tool_nparams = 0;
    st->tool_open = false;
    st->any_tool = false;
}

void ember_sse_headers(ember_buf *out, bool enable_cors) {
    ember_buf_puts(out, "HTTP/1.1 200 OK\r\n");
    if (enable_cors) ember_buf_puts(out, "Access-Control-Allow-Origin: *\r\n");
    ember_buf_puts(out,
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n");  // close-delimited framing (ds4 sse_headers)
}

void ember_sse_keepalive(ember_buf *out) {
    ember_buf_puts(out, ": prefill\n\n");
}

void ember_sse_role_chunk(ember_sse_stream *st, ember_buf *out) {
    ember_buf_printf(out,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":", st->id, st->created);
    ember_json_escape(out, st->model);
    ember_buf_puts(out,
        ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
        "\"finish_reason\":null}]}\n\n");
}

// ── incremental tool-call streaming (ds4 start-delta + args fragments) ──
static void tool_start_delta(ember_sse_stream *st, int idx, const char *name,
                             ember_buf *out) {
    char cid[40];
    gen_call_id(cid, sizeof(cid));  // B#5: random per-call id
    // B3: every emitted id must remain bound to the same exact sampled DSML
    // frontier. Silent truncation here makes call 17+ impossible to continue.
    if (st->n_tool_ids == st->tool_ids_cap) {
        if (st->tool_ids_cap > INT_MAX / 2)
            ember_buf_fatal("tool-call id count overflow");
        int next = st->tool_ids_cap ? st->tool_ids_cap * 2 : 4;
        char **grown = realloc(st->tool_ids, (size_t)next * sizeof(char *));
        if (!grown) ember_buf_fatal("out of memory storing tool-call ids");
        st->tool_ids = grown;
        st->tool_ids_cap = next;
    }
    st->tool_ids[st->n_tool_ids++] = xstrdup(cid);
    if (st->sink.tool_start) {
        st->sink.tool_start(st->sink_ud, idx, cid, name, out);
        return;
    }
    ember_buf_printf(out,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":", st->id, st->created);
    ember_json_escape(out, st->model);
    ember_buf_printf(out,
        ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{"
        "\"index\":%d,\"id\":\"%s\",\"type\":\"function\",\"function\":{\"name\":",
        idx, cid);
    ember_json_escape(out, name);
    ember_buf_puts(out, ",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n");
}
static void tool_args_delta(ember_sse_stream *st, int idx, const char *frag,
                            size_t len, ember_buf *out) {
    if (len == 0) return;
    if (st->sink.tool_args_delta) {
        st->sink.tool_args_delta(st->sink_ud, idx, frag, len, out);
        return;
    }
    ember_buf_printf(out,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":", st->id, st->created);
    ember_json_escape(out, st->model);
    ember_buf_printf(out,
        ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{"
        "\"index\":%d,\"function\":{\"arguments\":", idx);
    ember_json_escape_n(out, frag, len);  // arguments is a JSON *string*
    ember_buf_puts(out, "}}]},\"finish_reason\":null}]}\n\n");
}
// Emit any newly-available tool-call deltas from the buffered tool region.
// Idempotent across calls: state (tool_idx/tool_nparams/tool_open) tracks what
// has already been emitted so re-scanning the growing buffer only sends new
// headers + parameter fragments. On `final`, closes the open call's args.
static void emit_tool_stream(ember_sse_stream *st, const char *raw,
                             size_t raw_len, bool final, ember_buf *out) {
    const char *region = raw + st->tool_start;
    const ember_dsml_syntax *sx = ember_dsml_detect(region);
    if (!sx) {
        if (final && st->tool_open) { tool_args_delta(st, st->tool_idx, "}", 1, out); st->tool_open = false; }
        return;
    }
    const size_t iol = strlen(sx->invoke_open), icl = strlen(sx->invoke_close);
    const size_t pol = strlen(sx->param_open), pcl = strlen(sx->param_close);
    const char *cur = region;
    int idx = -1;
    while ((cur = strstr(cur, sx->invoke_open)) != NULL) {
        const char *tag_end = strchr(cur, '>');
        if (!tag_end) break;  // invoke opening tag not complete yet → wait
        idx++;
        char *name = ember_dsml_attr(cur + iol, tag_end + 1, "name");
        if (idx > st->tool_idx) {
            // B#4: never stream a tool-call header without a non-empty function
            // name — a header can't be retracted mid-stream, so a nameless call
            // must never emit a broken one. The invoke opening tag is already
            // complete here (tag_end found), so an absent/empty name is
            // malformed; buffer by giving up further tool-delta emission (ds4
            // openai_tool_start_invoke fails the stream) rather than emit it.
            if (!name || !name[0]) { free(name); break; }
            // close the previous call: "{}" if it had no params, else "}"
            if (st->tool_open) {
                if (st->tool_nparams == 0) tool_args_delta(st, st->tool_idx, "{}", 2, out);
                else                       tool_args_delta(st, st->tool_idx, "}", 1, out);
                st->tool_open = false;
            }
            tool_start_delta(st, idx, name, out);
            st->tool_idx = idx; st->tool_nparams = 0; st->tool_open = true; st->any_tool = true;
        }
        const char *inv_close = strstr(tag_end, sx->invoke_close);
        const char *inv_limit = inv_close ? inv_close : raw + raw_len;
        const char *p = tag_end + 1;
        int pcount = 0;
        while ((p = strstr(p, sx->param_open)) != NULL && p < inv_limit) {
            const char *ptag = strchr(p, '>');
            if (!ptag || ptag >= inv_limit) break;
            const char *pclose = strstr(ptag, sx->param_close);
            if (!pclose || pclose > inv_limit) break;  // parameter not complete yet
            if (idx == st->tool_idx && pcount >= st->tool_nparams) {
                char *key = ember_dsml_attr(p + pol, ptag + 1, "name");
                char *is_str = ember_dsml_attr(p + pol, ptag + 1, "string");
                if (key) {
                    ember_buf frag = {0};
                    ember_buf_putc(&frag, st->tool_nparams == 0 ? '{' : ',');
                    ember_dsml_append_arg(&frag, key, ptag + 1,
                                          (size_t)(pclose - (ptag + 1)), is_str);
                    tool_args_delta(st, idx, frag.ptr, frag.len, out);
                    ember_buf_free(&frag);
                    st->tool_nparams++;
                }
                free(key); free(is_str);
            }
            pcount++;
            p = pclose + pcl;
        }
        free(name);
        if (!inv_close) break;  // current invoke still streaming
        cur = inv_close + icl;
    }
    if (final && st->tool_open) {
        if (st->tool_nparams == 0) tool_args_delta(st, st->tool_idx, "{}", 2, out);
        else                       tool_args_delta(st, st->tool_idx, "}", 1, out);
        st->tool_open = false;
    }
    (void)iol;
}

// Longest suffix of raw[start:raw_len] that could still grow into either the
// configured force-close sequence (filter + "</think>") or a normal
// "</think>". Holding only a matching prefix keeps ordinary reasoning live;
// holding the whole configured filter once it appears lets us retract it when
// the close marker arrives.
static size_t reasoning_control_prefix_hold(
        const ember_sse_stream *st, const char *raw,
        size_t start, size_t raw_len) {
    static const char close[] = "</think>";
    const size_t close_len = sizeof(close) - 1;
    size_t avail = raw_len > start ? raw_len - start : 0;
    size_t best = 0;

    size_t close_max = avail < close_len - 1 ? avail : close_len - 1;
    for (size_t n = close_max; n > 0; --n) {
        if (memcmp(raw + raw_len - n, close, n) == 0) {
            best = n;
            break;
        }
    }

    if (st->reasoning_filter_len > SIZE_MAX - close_len)
        return best;
    size_t pattern_len = st->reasoning_filter_len + close_len;
    if (st->reasoning_filter_len == 0) return best;
    size_t max = avail < pattern_len - 1 ? avail : pattern_len - 1;
    for (size_t n = max; n > best; --n) {
        const char *tail = raw + raw_len - n;
        bool match = true;
        for (size_t i = 0; i < n; ++i) {
            char expected = i < st->reasoning_filter_len
                                ? st->reasoning_filter[i]
                                : close[i - st->reasoning_filter_len];
            if (tail[i] != expected) {
                match = false;
                break;
            }
        }
        if (match) return n;
    }
    return best;
}

void ember_sse_update(ember_sse_stream *st, const char *raw, size_t raw_len,
                      bool final, ember_buf *out) {
    if (!raw) return;

    if (st->mode == EMBER_SSE_THINKING) {
        // Strip a leading "<think>" opener once (some templates emit it as the
        // first visible bytes rather than opening it in the prompt).
        if (!st->checked_think_prefix) {
            static const char *open = "<think>";
            size_t open_len = 7;
            if (raw_len < open_len && strncmp(raw, open, raw_len) == 0 && !final)
                return;  // wait — may be a split "<think>"
            if (raw_len >= open_len && strncmp(raw, open, open_len) == 0)
                st->emit_pos = open_len;
            st->checked_think_prefix = true;
        }
        const char *close = strstr(raw + st->emit_pos, "</think>");
        size_t limit;
        if (close) {
            limit = (size_t)(close - raw);
            // The backend may inject a directive to force thinking closed at
            // its budget edge. It is control text, not model reasoning.
            if (st->reasoning_filter_len > 0 &&
                limit >= st->emit_pos + st->reasoning_filter_len &&
                memcmp(raw + limit - st->reasoning_filter_len,
                       st->reasoning_filter,
                       st->reasoning_filter_len) == 0)
                limit -= st->reasoning_filter_len;
        } else if (final) {
            limit = raw_len;
        } else {
            size_t hold = reasoning_control_prefix_hold(
                st, raw, st->emit_pos, raw_len);
            limit = raw_len - hold;
            limit = ember_utf8_stream_safe_len(raw, st->emit_pos, limit, false);
        }
        if (limit > st->emit_pos) {
            sse_chat_delta_n(st, "reasoning_content", raw + st->emit_pos,
                             limit - st->emit_pos, out);
            st->sent_reasoning = true;
            st->emit_pos = limit;
        }
        if (close) {
            st->emit_pos = (size_t)(close - raw) + 8;  // past "</think>"
            st->mode = EMBER_SSE_TEXT;
        } else {
            return;  // stay in THINKING (or SUPPRESS at final via fallthrough)
        }
    }

    if (st->mode == EMBER_SSE_TEXT) {
        const char *tool =
            st->has_tools ? ember_find_tool_start(raw + st->emit_pos) : NULL;
        size_t limit =
            ember_text_safe_limit(raw, st->emit_pos, raw_len, st->has_tools, final);
        // Stop-sequence holdback (ds4 stop_list_stream_safe_len): cut visible
        // content at the earliest complete stop; otherwise hold back the last
        // (max_stop_len-1) bytes so a stop forming across tokens never leaks.
        bool stop_complete = false;
        if (st->n_stops > 0 && limit > st->emit_pos) {
            size_t cut = limit;
            for (int si = 0; si < st->n_stops; si++) {
                size_t slen = strlen(st->stops[si]);
                if (!slen) continue;
                for (size_t i = st->emit_pos; i + slen <= limit; i++) {
                    if (memcmp(raw + i, st->stops[si], slen) == 0) {
                        if (i < cut) cut = i;
                        stop_complete = true;
                        break;
                    }
                }
            }
            if (!stop_complete && !final && st->max_stop_len > 1) {
                size_t hold = (size_t)(st->max_stop_len - 1);
                cut = limit > st->emit_pos + hold ? limit - hold : st->emit_pos;
            }
            limit = cut;
        }
        if (limit > st->emit_pos) {
            sse_chat_delta_n(st, "content", raw + st->emit_pos,
                             limit - st->emit_pos, out);
            st->sent_content = true;
            if (!st->sent_visible_content)
                st->sent_visible_content =
                    slice_has_visible(raw + st->emit_pos, limit - st->emit_pos);
            if (!st->delivered_tool_markup)
                st->delivered_tool_markup =
                    slice_has_tool_markup(raw + st->emit_pos,
                                          limit - st->emit_pos);
            st->emit_pos = limit;
        }
        if (stop_complete) {
            st->mode = EMBER_SSE_SUPPRESS;
        } else if (tool) {
            st->tool_start = (size_t)(tool - raw);
            st->emit_pos = st->tool_start;
            st->mode = EMBER_SSE_TOOL;
        } else if (final) {
            st->mode = EMBER_SSE_SUPPRESS;
        }
    }

    // EMBER_SSE_TOOL is intentionally silent here. A complete call can still
    // fail contamination, JSON, or request-schema validation, and bytes already
    // sent to a client cannot be retracted. The request worker performs those
    // checks and calls ember_sse_emit_tools() only for an executable block.
}

bool ember_sse_delivered_visible(const ember_sse_stream *st) {
    if (!st) return false;
    return st->sent_visible_content || st->any_tool;
}

const char *ember_tool_parse_failure_finish(const char *finish) {
    // ds4_server.c:5194-5201. Preserve a true length stop so callers can still
    // tell truncation from a completed turn; anything else is a normal stop.
    if (finish && strcmp(finish, "length") == 0) return "length";
    return "stop";
}

bool ember_sse_discard_tool_block(ember_sse_stream *st, size_t raw_len) {
    // ds4 openai_sse_stream_update:6449-6455. EMBER_SSE_TEXT rewound emit_pos to
    // the tool marker before going silent; when the block fails validation ds4's
    // chat stream drops to SUPPRESS and finishes "stop" without ever emitting
    // those bytes or an error frame.
    //
    // Ember deliberately follows the chat path rather than ds4's Responses path
    // (responses_sse_finish_live:7192-7215), which flushes the suppressed bytes
    // as text. A rejected block can be contaminated -- ASCII-degraded DSML
    // nested inside another call's argument is a real deployment shape, covered
    // by test_tool_safety_server -- and replaying it as assistant text would
    // hand a text-scanning consumer an executable-looking payload. Dropping it
    // keeps the validation guarantee while still not killing the turn.
    if (!st || st->mode != EMBER_SSE_TOOL) return false;
    st->mode = EMBER_SSE_SUPPRESS;
    st->emit_pos = raw_len;
    return true;
}

bool ember_sse_emit_tools(ember_sse_stream *st, const char *raw, size_t raw_len,
                          ember_buf *out) {
    if (st->mode != EMBER_SSE_TOOL) return false;
    // Validation-gated flush: emit the complete call and close its arguments.
    // This remains idempotent in case a caller probes the final result twice.
    if (!st->qwen_tool_syntax)
        emit_tool_stream(st, raw, raw_len, true, out);
    // Fallback for non-DSML formats (ds_engine_tool_use): the incremental walker
    // handles DSML only, so parse the buffered region and emit each call whole.
    if (!st->any_tool) {
        ember_tool_calls tc = {0};
        int n;
        if (st->qwen_tool_syntax) {
            ember_qwen_tool_parse_report report = {0};
            n = ember_parse_qwen_tool_calls(raw + st->tool_start,
                                            st->tools_json, &tc, &report);
        } else {
            n = ember_parse_dsml_tool_calls(raw + st->tool_start, &tc);
        }
        for (int i = 0; i < n; i++) {
            tool_start_delta(st, i, tc.calls[i].name, out);
            tool_args_delta(st, i, tc.calls[i].arguments,
                            strlen(tc.calls[i].arguments), out);
        }
        if (n > 0) st->any_tool = true;
        ember_tool_calls_free(&tc);
        if (n == 0) {
            // False-positive tool marker. TOOL_STARTS/TOOL_ENDS (:83, :106)
            // carry the Qwen markers unconditionally, so a DSML-profile stream
            // whose ordinary prose contains "<tool_call>" switches to
            // EMBER_SSE_TOOL and rewinds emit_pos to the marker -- while
            // main.c's parse_executable_tool_calls, which IS profile-aware,
            // sees report.found == false and correctly reports ordinary
            // assistant text (main.c:1314-1320). That verdict makes every
            // error and discard branch in the caller unreachable, so without
            // this flush every byte from the marker to the end of the turn is
            // dropped with no error, no log, and a normal "stop".
            //
            // This is NOT the ember_sse_discard_tool_block case. That drops a
            // block which parsed as a call and then failed validation, where
            // replaying contaminated markup as text is the documented risk.
            // Here nothing parsed as a call in any syntax family, and the
            // caller has already classified the same bytes as text.
            //
            // Delivering markup as visible content is an outcome this server
            // already accepts and counts rather than prevents -- see
            // note_tool_markup_leak (main.c:1698-1718). Tracking the slice
            // below is what lets that counter fire, which is why the flags are
            // updated here exactly as the EMBER_SSE_TEXT branch does.
            if (raw_len > st->tool_start) {
                const char *slice = raw + st->tool_start;
                const size_t slice_len = raw_len - st->tool_start;
                sse_chat_delta_n(st, "content", slice, slice_len, out);
                st->sent_content = true;
                if (!st->sent_visible_content)
                    st->sent_visible_content = slice_has_visible(slice, slice_len);
                // Unconditional, not slice_has_tool_markup(): st->tool_start
                // points AT a marker ember_find_tool_start just matched, so
                // markup is present by construction. It also cannot be
                // detected by that helper, which recognises DSML only --
                // ember_text_has_tool_markup (:142-145) adds "<tool_call>" but
                // the streaming helper does not. Extending the shared helper
                // instead would make the EMBER_SSE_TEXT branch count the marker
                // as a leak on requests carrying no tools at all, which is the
                // false-firing note_tool_markup_leak documents at
                // main.c:1704-1707.
                st->delivered_tool_markup = true;
            }
            // SUPPRESS makes this idempotent: a second probe returns at the
            // mode guard above instead of emitting the slice twice.
            st->emit_pos = raw_len;
            st->mode = EMBER_SSE_SUPPRESS;
        }
    }
    return st->any_tool;
}

static bool progress_error_code(const char *code) {
    return code &&
        (!strcmp(code, "repetition_detected") ||
         !strcmp(code, "reasoning_cycle_detected") ||
         !strcmp(code, "prompt_echo_detected"));
}

void ember_sse_error(ember_sse_stream *st, const char *code, const char *detail,
                     bool retry_exhausted, ember_buf *out) {
    (void)st;
    bool invalid_tool = code && strcmp(code, "invalid_tool_call") == 0;
    bool model_output = invalid_tool || progress_error_code(code);
    ember_buf_puts(out, "event: error\ndata: {\"error\":{\"message\":");
    ember_json_escape(out, detail && detail[0] ? detail : "backend generation failed");
    ember_buf_puts(out, ",\"type\":");
    ember_json_escape(out, model_output ? "model_output_error" : "server_error");
    ember_buf_puts(out, ",\"code\":");
    ember_json_escape(out, code && code[0] ? code : "internal_error");
    if (model_output) {
        ember_buf_printf(out, ",\"retry_exhausted\":%s",
                         retry_exhausted ? "true" : "false");
        ember_buf_puts(out, ",\"partial_tool_call\":false");
    }
    ember_buf_puts(out, "}}\n\n");
    ember_buf_puts(out, "data: [DONE]\n\n");
}

void ember_sse_finish(ember_sse_stream *st, const char *finish_reason,
                      int prompt_tokens, int completion_tokens, ember_buf *out) {
    ember_buf_printf(out,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":",
        st->id, st->created);
    ember_json_escape(out, st->model);
    ember_buf_printf(out,
        ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"",
        finish_reason);
    if (st->tool_loop_rounds > 0) {
        ember_buf_printf(out,
            ",\"ember_tool_loop\":{\"rounds\":%d,\"tool\":",
            st->tool_loop_rounds);
        ember_json_escape(out, st->tool_loop_tool ? st->tool_loop_tool : "");
        ember_buf_printf(out, ",\"identical_results\":%s}",
                         st->tool_loop_identical ? "true" : "false");
    }
    if (st->termination_reason && st->termination_reason[0]) {
        ember_buf_puts(out, ",\"finish_details\":{\"type\":");
        ember_json_escape(out, st->termination_reason);
        ember_buf_putc(out, '}');
    }
    ember_buf_puts(out, "}]}\n\n");
    // usage chunk — only when the client opted in via stream_options.include_usage
    if (!st->include_usage) { ember_buf_puts(out, "data: [DONE]\n\n"); return; }
    ember_buf_printf(out,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":",
        st->id, st->created);
    ember_json_escape(out, st->model);
    ember_buf_printf(out,
        ",\"choices\":[],\"usage\":{\"prompt_tokens\":%d,"
        "\"completion_tokens\":%d,\"total_tokens\":%d,"
        "\"prompt_tokens_details\":{\"cached_tokens\":%d},"
        "\"timings\":{\"prefill_ms\":%.1f,\"prefill_tokens\":%d,"
        "\"prefill_tokens_per_sec\":%.1f,\"decode_ms\":%.1f,"
        "\"decode_tokens_per_sec\":%.2f,\"spec_cycles\":%d,"
        "\"spec_provider_age_ms\":%.1f,"
        "\"spec_provider_block_ms\":%.1f,\"spec_head_ms\":%.1f,"
        "\"spec_verify_ms\":%.1f},\"accept_rate\":%.3f,"
        "\"backend\":{\"prefill_mode\":\"%s\","
        "\"prefill_reason\":\"%s\",\"forced_close\":%s,"
        "\"degenerate\":%s",
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
        st->cached_tokens, st->prefill_s * 1000.0, st->prefill_tokens,
        st->prefill_s > 0.0 ? st->prefill_tokens / st->prefill_s : 0.0,
        st->decode_s * 1000.0,
        st->decode_s > 0.0 ? completion_tokens / st->decode_s : 0.0,
        st->spec_cycles, st->spec_provider_age_s * 1000.0,
        st->spec_provider_block_s * 1000.0,
        st->spec_head_s * 1000.0, st->spec_verify_s * 1000.0,
        st->accept_rate,
        st->prefill_mode ? st->prefill_mode : "unknown",
        st->prefill_reason ? st->prefill_reason : "unknown",
        st->reasoning_budget_exhausted ? "true" : "false",
        st->degenerate ? "true" : "false");
    if (st->reasoning_budget_exhausted)
        ember_buf_puts(out,
            ",\"reasoning_stop_reason\":\"reasoning_budget_exhausted\"");
    if (st->termination_reason && st->termination_reason[0]) {
        ember_buf_puts(out, ",\"termination_reason\":");
        ember_json_escape(out, st->termination_reason);
    }
    ember_buf_putc(out, '}');
    if (st->compaction_json && st->compaction_json[0]) {
        ember_buf_putc(out, ',');
        ember_buf_puts(out, st->compaction_json);
    }
    ember_buf_puts(out, "}}\n\n");
    ember_buf_puts(out, "data: [DONE]\n\n");
}
