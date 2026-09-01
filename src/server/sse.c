#include "sse.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../common/json_util.h"
#include "../common/utf8.h"
#include "../model/tool_parser.h"

// B#5: mint a RANDOM, per-call unique id (mirrors ds4 random_tool_id) instead
// of hashing name+index. The old FNV hash gave e.g. the first `bash` call an
// identical id in every response; a client correlating tool results across
// requests would then collide. 128 bits from /dev/urandom, hex-encoded, with
// the `call_` prefix. All SSE emission runs on the single generation worker
// thread, so the fallback counter needs no synchronization. The id is emitted
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
    static uint64_t fallback_ctr;
    unsigned char bytes[16];
    size_t pos = (size_t)snprintf(out, cap, "%s", "call_");
    if (pos >= cap) return;
    if (!sse_random_bytes(bytes, sizeof(bytes))) {
        uint64_t a = ((uint64_t)time(NULL) << 32) ^ (uint64_t)getpid();
        uint64_t b = ++fallback_ctr ^ (uint64_t)(uintptr_t)out;
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
static const char *const TOOL_STARTS[] = {
    "<" PIPE "DSML" PIPE "tool_calls>",  // <｜DSML｜tool_calls>
    "<DSML" PIPE "tool_calls>",          // <DSML｜tool_calls>  (short spelling)
    "<?DSML?tool_calls>",                // ascii-degraded
    "<tool_calls>",                      // plain-XML (parser has a matching family)
    "<ds_engine_tool_use>",              // model's native format (see tool_parser.c)
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
};
const char *ember_find_tool_end(const char *s) {
    const char *best = NULL; size_t best_len = 0;
    for (size_t i = 0; i < sizeof(TOOL_ENDS)/sizeof(TOOL_ENDS[0]); i++) {
        const char *hit = strstr(s, TOOL_ENDS[i]);
        if (hit && (!best || hit < best)) { best = hit; best_len = strlen(TOOL_ENDS[i]); }
    }
    return best ? best + best_len : NULL;
}

// ─── content holdback (ds4 text_stream_safe_limit) ──────────────────────
size_t ember_text_safe_limit(const char *raw, size_t start, size_t raw_len,
                             bool has_tools, bool final) {
    if (raw_len <= start) return raw_len;
    size_t limit = raw_len;
    if (has_tools) {
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

// ─── public API ─────────────────────────────────────────────────────────
static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
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
    for (int i = 0; i < st->n_tool_ids; i++) free(st->tool_ids[i]);  // B3
    st->n_tool_ids = 0;
    st->id = st->model = NULL;
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
    // B3: record the id so run_chat can store this turn's exact sampled bytes
    // under it (exact-DSML replay). Bounded; extra parallel calls just won't
    // be individually replayable (falls back to canonical).
    if (st->n_tool_ids < 16) st->tool_ids[st->n_tool_ids++] = xstrdup(cid);
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
        } else if (final) {
            limit = raw_len;
        } else {
            // Hold a possible partial "</think>" (7 bytes) + incomplete UTF-8.
            size_t hold = 7;  // strlen("</think>") - 1
            limit = raw_len > st->emit_pos + hold ? raw_len - hold : st->emit_pos;
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
        if (st->n_stops > 0 && limit > st->emit_pos) {
            size_t cut = limit; bool complete = false;
            for (int si = 0; si < st->n_stops; si++) {
                size_t slen = strlen(st->stops[si]);
                if (!slen) continue;
                for (size_t i = st->emit_pos; i + slen <= limit; i++) {
                    if (memcmp(raw + i, st->stops[si], slen) == 0) {
                        if (i < cut) cut = i;
                        complete = true;
                        break;
                    }
                }
            }
            if (!complete && !final && st->max_stop_len > 1) {
                size_t hold = (size_t)(st->max_stop_len - 1);
                cut = limit > st->emit_pos + hold ? limit - hold : st->emit_pos;
            }
            limit = cut;
        }
        if (limit > st->emit_pos) {
            sse_chat_delta_n(st, "content", raw + st->emit_pos,
                             limit - st->emit_pos, out);
            st->sent_content = true;
            st->emit_pos = limit;
        }
        if (tool) {
            st->tool_start = (size_t)(tool - raw);
            st->emit_pos = st->tool_start;
            st->mode = EMBER_SSE_TOOL;  // switch to incremental tool_call deltas
        } else if (final) {
            st->mode = EMBER_SSE_SUPPRESS;
        }
    }

    // EMBER_SSE_TOOL: stream tool_call header + argument fragments incrementally
    // as invokes/parameters complete. DSML markup is never emitted as content.
    if (st->mode == EMBER_SSE_TOOL) {
        emit_tool_stream(st, raw, raw_len, final, out);
    }
}

bool ember_sse_emit_tools(ember_sse_stream *st, const char *raw, size_t raw_len,
                          ember_buf *out) {
    if (st->mode != EMBER_SSE_TOOL) return false;
    // Final flush: emit any remaining (now-complete) DSML deltas and close the
    // open call's args. Idempotent with the incremental emission during updates.
    emit_tool_stream(st, raw, raw_len, true, out);
    // Fallback for non-DSML formats (ds_engine_tool_use): the incremental walker
    // handles DSML only, so parse the buffered region and emit each call whole.
    if (!st->any_tool) {
        ember_tool_calls tc = {0};
        int n = ember_parse_dsml_tool_calls(raw + st->tool_start, &tc);
        for (int i = 0; i < n; i++) {
            char cid[40];
            gen_call_id(cid, sizeof(cid));  // B#5: random per-call id
            ember_buf_printf(out,
                "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                "\"created\":%ld,\"model\":", st->id, st->created);
            ember_json_escape(out, st->model);
            ember_buf_printf(out,
                ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{"
                "\"index\":%d,\"id\":\"%s\",\"type\":\"function\",\"function\":{\"name\":",
                i, cid);
            ember_json_escape(out, tc.calls[i].name);
            ember_buf_puts(out, ",\"arguments\":");
            ember_json_escape(out, tc.calls[i].arguments);
            ember_buf_puts(out, "}}]},\"finish_reason\":null}]}\n\n");
        }
        if (n > 0) st->any_tool = true;
        ember_tool_calls_free(&tc);
    }
    return st->any_tool;
}

void ember_sse_error(ember_sse_stream *st, const char *code, const char *detail,
                     ember_buf *out) {
    (void)st;
    ember_buf_puts(out, "event: error\ndata: {\"error\":{\"message\":");
    ember_json_escape(out, detail && detail[0] ? detail : "backend generation failed");
    ember_buf_puts(out, ",\"type\":\"server_error\",\"code\":");
    ember_json_escape(out, code && code[0] ? code : "internal_error");
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
        ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n",
        finish_reason);
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
        "\"prompt_tokens_details\":{\"cached_tokens\":%d}}}\n\n",
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
        st->cached_tokens);
    ember_buf_puts(out, "data: [DONE]\n\n");
}
