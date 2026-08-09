// Unit tests for the SSE streaming layer. These exercise the exact failure
// modes that broke lucebox's incremental emitter and had to be patched one at
// a time — here they must pass by construction. Drives the stream token-by-
// token (accumulating raw) the way the real server does.
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "../src/common/buf.h"
#include "../src/server/sse.h"
#include "fixtures_real_failures.h"

#define PIPE "\xef\xbd\x9c"  // U+FF5C

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (cond) { g_pass++; }                                             \
        else { g_fail++; printf("  FAIL: %s\n", msg); }                     \
    } while (0)

// Feed `full` to a fresh stream one UTF-8-agnostic byte-chunk at a time,
// collecting all emitted SSE bytes. `chunk` lets us force pathological splits.
static char *stream_in_chunks(const char *full, size_t chunk_sz, bool has_tools,
                              bool started_thinking) {
    ember_sse_stream st;
    ember_sse_init(&st, "chatcmpl_test", "deepseek-v4-flash", 1700000000,
                   has_tools, started_thinking, false);
    ember_buf out = {0};
    ember_buf acc = {0};  // accumulated raw
    size_t total = strlen(full);
    for (size_t i = 0; i < total; i += chunk_sz) {
        size_t n = chunk_sz < total - i ? chunk_sz : total - i;
        ember_buf_append(&acc, full + i, n);
        ember_sse_update(&st, acc.ptr, acc.len, false, &out);
    }
    ember_sse_update(&st, acc.ptr, acc.len, true, &out);  // final flush
    ember_sse_finish(&st, "stop", 10, 5, &out);
    ember_sse_free(&st);
    ember_buf_free(&acc);
    return ember_buf_take(&out);  // caller frees
}

// Same drive loop, but reports whether tool markup reached the client as
// CONTENT instead of returning the wire bytes.
static bool stream_markup_verdict(const char *full, size_t chunk_sz,
                                  bool has_tools) {
    ember_sse_stream st;
    ember_sse_init(&st, "cc", "m", 1700000000, has_tools, false, false);
    ember_buf out = {0}, acc = {0};
    size_t total = strlen(full);
    for (size_t i = 0; i < total; i += chunk_sz) {
        size_t n = chunk_sz < total - i ? chunk_sz : total - i;
        ember_buf_append(&acc, full + i, n);
        ember_sse_update(&st, acc.ptr, acc.len, false, &out);
    }
    ember_sse_update(&st, acc.ptr, acc.len, true, &out);
    const bool leaked = ember_sse_delivered_tool_markup(&st);
    ember_sse_free(&st);
    ember_buf_free(&acc);
    ember_buf_free(&out);
    return leaked;
}

// Extract the concatenated value of every delta.<field> from an SSE dump.
// Crude but sufficient: finds "field":"…" and unescapes \n \" \\.
static void collect_field(const char *sse, const char *field, ember_buf *dst) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\":\"", field);
    const char *p = sse;
    while ((p = strstr(p, needle))) {
        p += strlen(needle);
        while (*p && *p != '"') {
            if (p[0] == '\\' && p[1]) {
                char c = p[1];
                if (c == 'n') ember_buf_putc(dst, '\n');
                else if (c == 't') ember_buf_putc(dst, '\t');
                else if (c == 'r') ember_buf_putc(dst, '\r');
                else ember_buf_putc(dst, c);  // \" \\ etc.
                p += 2;
            } else {
                ember_buf_putc(dst, *p);
                p++;
            }
        }
    }
}

static void test_plain_content(void) {
    char *sse = stream_in_chunks("Hello, world!", 1, false, false);
    ember_buf c = {0};
    collect_field(sse, "content", &c);
    CHECK(strcmp(c.ptr ? c.ptr : "", "Hello, world!") == 0,
          "plain content reassembles byte-by-byte");
    CHECK(strstr(sse, "data: [DONE]") != NULL, "stream terminates with [DONE]");
    ember_buf_free(&c);
    free(sse);
}

static void test_split_emoji(void) {
    // 👋 = F0 9F 91 8B, 👍 = F0 9F 91 8D. chunk_sz=2 splits every emoji.
    char *sse = stream_in_chunks("Hey \xf0\x9f\x91\x8b\xf0\x9f\x91\x8d", 2,
                                 false, false);
    ember_buf c = {0};
    collect_field(sse, "content", &c);
    CHECK(strcmp(c.ptr, "Hey \xf0\x9f\x91\x8b\xf0\x9f\x91\x8d") == 0,
          "split emoji reassemble, not corrupted to U+FFFD");
    CHECK(strstr(sse, "\xef\xbf\xbd") == NULL, "no U+FFFD replacement chars");
    ember_buf_free(&c);
    free(sse);
}

static void test_reasoning_split(void) {
    // Prompt-opened thinking: bytes start as reasoning, close, then content.
    char *sse = stream_in_chunks("the answer is 4</think>The answer is 4.", 3,
                                 false, true);
    ember_buf r = {0}, c = {0};
    collect_field(sse, "reasoning_content", &r);
    collect_field(sse, "content", &c);
    CHECK(strcmp(r.ptr, "the answer is 4") == 0, "reasoning captured");
    CHECK(strcmp(c.ptr, "The answer is 4.") == 0, "content after </think>");
    CHECK(strstr(c.ptr, "</think>") == NULL, "close tag never leaks to content");
    ember_buf_free(&r); ember_buf_free(&c);
    free(sse);
}

static void test_split_think_close(void) {
    // Force "</think>" to split across chunks (chunk 3): must not leak a
    // fragment like "</thi" into reasoning nor "nk>" into content.
    char *sse = stream_in_chunks("reasoning here</think>answer", 3, false, true);
    ember_buf r = {0}, c = {0};
    collect_field(sse, "reasoning_content", &r);
    collect_field(sse, "content", &c);
    CHECK(strcmp(r.ptr, "reasoning here") == 0, "no </think> fragment in reasoning");
    CHECK(strcmp(c.ptr, "answer") == 0, "no </think> fragment in content");
    ember_buf_free(&r); ember_buf_free(&c);
    free(sse);
}

static void test_force_close_hint_filtered_from_reasoning(void) {
    const char *hint = "SERVER_FORCE_CLOSE\n</think>\n\n";
    const char *full =
        "natural reasoning SERVER_FORCE_CLOSE\n</think>\n\nanswer";
    ember_sse_stream st;
    ember_sse_init(&st, "cc", "m", 1700000000, false, true, false);
    ember_sse_set_reasoning_filter(&st, hint);
    ember_buf out = {0}, acc = {0};
    for (size_t i = 0; i < strlen(full); ++i) {
        ember_buf_append(&acc, full + i, 1);
        ember_sse_update(&st, acc.ptr, acc.len, false, &out);
        if (i == 7)
            CHECK(out.len > 0,
                  "force-close filter does not delay ordinary reasoning");
    }
    ember_sse_update(&st, acc.ptr, acc.len, true, &out);
    ember_buf reasoning = {0}, content = {0};
    collect_field(out.ptr ? out.ptr : "", "reasoning_content", &reasoning);
    collect_field(out.ptr ? out.ptr : "", "content", &content);
    CHECK(strcmp(reasoning.ptr ? reasoning.ptr : "", "natural reasoning ") == 0,
          "server force-close directive is not streamed as reasoning");
    CHECK(strcmp(content.ptr ? content.ptr : "", "\n\nanswer") == 0,
          "visible answer remains after filtered force-close");
    ember_buf_free(&reasoning);
    ember_buf_free(&content);
    ember_buf_free(&out);
    ember_buf_free(&acc);
    ember_sse_free(&st);
}

static void test_tool_marker_suppressed(void) {
    // A DSML tool marker mid-stream must not leak into visible content.
    const char *full =
        "Let me check.\n\n<" PIPE "DSML" PIPE "tool_calls>garbage";
    char *sse = stream_in_chunks(full, 2, /*has_tools=*/true, false);
    ember_buf c = {0};
    collect_field(sse, "content", &c);
    CHECK(strstr(c.ptr ? c.ptr : "", "DSML") == NULL,
          "DSML markup never streamed as content");
    CHECK(strstr(c.ptr ? c.ptr : "", "tool_calls") == NULL,
          "tool_calls markup suppressed");
    CHECK(strncmp(c.ptr, "Let me check.", 13) == 0, "preamble text still emitted");
    ember_buf_free(&c);
    free(sse);
}

static void test_short_dsml_spelling(void) {
    // Short spelling (leading "<｜" eaten) must also be recognized/suppressed.
    const char *full = "ok<DSML" PIPE "tool_calls>x";
    char *sse = stream_in_chunks(full, 1, true, false);
    ember_buf c = {0};
    collect_field(sse, "content", &c);
    CHECK(strcmp(c.ptr ? c.ptr : "", "ok") == 0,
          "short DSML spelling detected, only 'ok' as content");
    ember_buf_free(&c);
    free(sse);
}

static void test_utf8_safe_limit_direct(void) {
    // A trailing lone lead byte must be held back (not emitted mid-codepoint).
    const char *s = "abc\xf0";  // 'abc' + start of a 4-byte seq
    size_t lim = ember_text_safe_limit(s, 0, 4, false, false);
    CHECK(lim == 3, "trailing lead byte held back");
    lim = ember_text_safe_limit(s, 0, 4, false, true);  // final: release
    CHECK(lim == 4, "final releases the tail");
}

static void test_tool_calls_emitted(void);
static void test_tool_attempt_reset(void);
static void test_matching_tool_closer_required(void);
static void test_native_tool_id_is_registered(void);
static void test_stop_precedes_tool(void);
static void test_more_than_sixteen_tool_ids(void);
static void test_usage_reports_prefill_policy(void);
static void test_holdback_ignores_has_tools(void);
static void test_tool_loop_terminal_report(void);


// ── real degraded output through the chunk-size fuzzer ───────────────────────
// sse.c buffers and re-splits precisely because a marker or codepoint split
// across ANY number of chunks broke five different ways in the incremental
// design it replaced. The strongest input for that is not a hand-written
// marker but bytes deployment actually emitted: corrupted DSML lookalikes
// (U+003F where U+FF5C belongs, names like tool_cards / tool_alls), a
// fragmented shell string, and a genuine repetition loop.
//
// The contract under test is CONSERVATION: whatever these bytes are, the
// visible text that comes out must equal what went in, at every chunk size.
// A corrupted marker is not a tool call, so none of it may be suppressed as
// tool markup -- silently swallowing it is how raw markup, or a hole where
// text should be, reaches the user.
static void test_real_degraded_output_survives_every_chunk_size(void) {
    const struct { const char *name; const char *text; } specimens[] = {
        { "pseudo-marker A",   REAL_PSEUDO_MARKER_1297 },
        { "pseudo-marker B",    REAL_PSEUDO_MARKER_136  },
        { "fragmentation",    REAL_FRAGMENTATION_130  },
        { "repetition loop",   REAL_REPETITION_81      },
    };
    for (size_t i = 0; i < sizeof(specimens) / sizeof(specimens[0]); i++) {
        const char *want = specimens[i].text;
        bool all_ok = true;
        // 1,2,3,5,7,11 -- primes straddle multi-byte and marker boundaries so
        // no split is systematically avoided.
        const size_t sizes[] = {1, 2, 3, 5, 7, 11, 64};
        for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++) {
            char *sse = stream_in_chunks(want, sizes[k], true, false);
            ember_buf got = {0};
            collect_field(sse, "content", &got);
            if (strcmp(got.ptr ? got.ptr : "", want) != 0) all_ok = false;
            ember_buf_free(&got);
            free(sse);
        }
        CHECK(all_ok, specimens[i].name);
    }
}

// The tool-markup leak counter must describe what was SENT, not what was
// generated. Before the fix it scanned the caller's raw accumulator, which
// still holds the tool block that ember_text_safe_limit() correctly withheld
// and the parser then consumed -- so it fired on every ordinary tool call:
// Held-back parsed calls must not be counted as visible markup leaks.
static void test_delivered_tool_markup_verdict(void) {
    const char *real_call =
        "Let me check.\n"
        "<" PIPE "DSML" PIPE "tool_calls>"
        "<" PIPE "DSML" PIPE "invoke name=\"terminal\">"
        "<" PIPE "DSML" PIPE "parameter name=\"command\" string=\"true\">pwd"
        "</" PIPE "DSML" PIPE "parameter>"
        "</" PIPE "DSML" PIPE "invoke>"
        "</" PIPE "DSML" PIPE "tool_calls>";
    // A malformed imitation with NO tool_calls opener: nothing recognises it,
    // so it streams as text. This is the shape of a regression where repeated
    // markers and sensitive tool arguments reached the user.
    const char *orphan_markup =
        "Here is the call I would make:\n"
        "<?DSML?invoke name=\"skill_manage\">\n"
        "<?DSML?parameter name=\"action\" string=\"true\">create</?DSML?parameter>";
    const char *prose = "DSML is the markup this server parses. No markers here.";

    const size_t sizes[] = {1, 2, 3, 5, 7, 11, 64};
    bool call_clean = true, orphan_seen = true, prose_clean = true;
    for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++) {
        if (stream_markup_verdict(real_call, sizes[k], true))  call_clean = false;
        if (!stream_markup_verdict(orphan_markup, sizes[k], true)) orphan_seen = false;
        if (stream_markup_verdict(prose, sizes[k], true))      prose_clean = false;
    }
    CHECK(call_clean,
          "an ordinary tool call does not count as delivered markup (#24)");
    CHECK(orphan_seen,
          "markup with no recognised opener IS delivered, and is counted");
    CHECK(prose_clean, "prose that merely mentions DSML is not counted");

    // has_tools must not change the verdict -- that gate was the #15 bug.
    CHECK(stream_markup_verdict(orphan_markup, 5, false),
          "delivered markup is counted whether or not tools were advertised");

    // The shared predicate the buffered path uses must agree.
    CHECK(!ember_text_has_tool_markup(prose) &&
          ember_text_has_tool_markup(orphan_markup),
          "ember_text_has_tool_markup agrees with the streaming verdict");
}

int main(void) {
    test_real_degraded_output_survives_every_chunk_size();
    test_delivered_tool_markup_verdict();
    printf("ember sse tests\n");
    test_plain_content();
    test_split_emoji();
    test_reasoning_split();
    test_split_think_close();
    test_force_close_hint_filtered_from_reasoning();
    test_tool_marker_suppressed();
    test_short_dsml_spelling();
    test_utf8_safe_limit_direct();
    test_tool_calls_emitted();
    test_tool_attempt_reset();
    test_matching_tool_closer_required();
    test_native_tool_id_is_registered();
    test_stop_precedes_tool();
    test_more_than_sixteen_tool_ids();
    test_usage_reports_prefill_policy();
    test_holdback_ignores_has_tools();
    test_tool_loop_terminal_report();
    printf("──────────────────────────────\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

static void test_usage_reports_prefill_policy(void) {
    ember_sse_stream st;
    ember_buf out = {0};
    ember_sse_init(&st, "cc", "m", 1700000000, false, false, false);
    st.include_usage = true;
    st.cached_tokens = 100;
    st.prefill_tokens = 200;
    st.prefill_s = 2.0;
    st.decode_s = 1.0;
    st.accept_rate = 0.75;
    st.prefill_mode = "hybrid";
    st.prefill_reason = "dspark_capture";
    st.termination_reason = "prompt_echo_detected";
    st.reasoning_budget_exhausted = true;
    st.degenerate = true;
    ember_sse_finish(&st, "length", 300, 10, &out);
    CHECK(strstr(out.ptr, "\"prefill_tokens\":200") != NULL,
          "stream usage reports evaluated prefill tokens");
    CHECK(strstr(out.ptr, "\"prefill_tokens_per_sec\":100.0") != NULL,
          "stream usage reports exact prefill throughput denominator");
    CHECK(strstr(out.ptr, "\"prefill_mode\":\"hybrid\"") != NULL,
          "stream usage reports actual prefill mode");
    CHECK(strstr(out.ptr, "\"prefill_reason\":\"dspark_capture\"") != NULL,
          "stream usage reports why the mode was selected");
    CHECK(strstr(out.ptr,
          "\"finish_details\":{\"type\":\"prompt_echo_detected\"}") != NULL,
          "terminal chunk reports the typed watchdog cause");
    CHECK(strstr(out.ptr,
          "\"termination_reason\":\"prompt_echo_detected\"") != NULL,
          "usage backend repeats the machine-readable watchdog cause");
    CHECK(strstr(out.ptr,
          "\"reasoning_stop_reason\":\"reasoning_budget_exhausted\"") != NULL,
          "usage reports forced reasoning-budget transition");
    ember_buf_free(&out);
    ember_sse_free(&st);
}

// Both tool-loop signals report through the same terminal object, told apart
// only by identical_results. The weaker call-signature signal emits false and
// must not be silently rendered as the stronger claim.
static void check_tool_loop_terminal_report(bool identical, const char *expect,
                                            const char *msg) {
    const char *full = "reasoning</think>tool response";
    const size_t total = strlen(full);
    for (size_t chunk = 1; chunk <= total; ++chunk) {
        ember_sse_stream st;
        ember_buf out = {0}, acc = {0};
        ember_sse_init(&st, "cc", "m", 1700000000, true, true, false);
        st.tool_loop_rounds = 4;
        st.tool_loop_identical = identical;
        st.tool_loop_tool = "terminal";
        for (size_t i = 0; i < total; i += chunk) {
            size_t n = chunk < total - i ? chunk : total - i;
            ember_buf_append(&acc, full + i, n);
            ember_sse_update(&st, acc.ptr, acc.len, false, &out);
        }
        ember_sse_update(&st, acc.ptr, acc.len, true, &out);
        ember_sse_finish(&st, "tool_calls", 10, 5, &out);
        CHECK(strstr(out.ptr ? out.ptr : "", expect) != NULL, msg);
        CHECK(strstr(out.ptr ? out.ptr : "", "\"finish_reason\":\"tool_loop\"") == NULL,
              "loop report does not invent a finish-reason enum");
        ember_buf_free(&out);
        ember_buf_free(&acc);
        ember_sse_free(&st);
    }
}

// A complete tool-start marker must never stream as content, EVEN when the
// request advertised no tools. ember_text_safe_limit used to skip the search
// when has_tools was false, on the assumption that no tools means no markers.
// A turn whose tools had been withdrawn emitted a complete <?DSML?tool_calls>
// block -- a supported syntax family -- with sensitive arguments. The
// end-of-turn parse recognised and rejected the call, but the bytes were gone.
static void test_holdback_ignores_has_tools(void) {
    static const char *const MARKERS[] = {
        "<?DSML?tool_calls>",                      // the deployment spelling
        "<" PIPE "DSML" PIPE "tool_calls>",         // real U+FF5C
        "<tool_calls>",                            // plain-XML degradation
    };
    for (size_t m = 0; m < sizeof(MARKERS) / sizeof(MARKERS[0]); ++m) {
        char raw[512];
        snprintf(raw, sizeof raw, "here you go:\n\n%s\n<invoke name=\"x\">",
                 MARKERS[m]);
        const size_t raw_len = strlen(raw);
        const char *marker = strstr(raw, MARKERS[m]);
        const size_t marker_at = (size_t)(marker - raw);
        for (int has_tools = 0; has_tools <= 1; ++has_tools) {
            for (int final = 0; final <= 1; ++final) {
                size_t n = ember_text_safe_limit(raw, 0, raw_len,
                                                 has_tools != 0, final != 0);
                CHECK(n <= marker_at,
                      "tool marker withheld from content regardless of has_tools");
            }
        }
    }
}

static void test_tool_loop_terminal_report(void) {
    check_tool_loop_terminal_report(
        true,
        "\"finish_reason\":\"tool_calls\",\"ember_tool_loop\":"
        "{\"rounds\":4,\"tool\":\"terminal\",\"identical_results\":true}",
        "terminal loop report survives every byte chunk size");
    check_tool_loop_terminal_report(
        false,
        "\"finish_reason\":\"tool_calls\",\"ember_tool_loop\":"
        "{\"rounds\":4,\"tool\":\"terminal\",\"identical_results\":false}",
        "call-signature loop reports identical_results false at every chunk size");
}

// Reassemble the tool-call arguments from validation-gated streaming deltas:
// scan
// for every "arguments":"<body>" occurrence, unescape one JSON level, and
// concatenate. A conforming client does exactly this to rebuild the full args.
// Robust to how the emitter chunks fragments (an implementation detail).
static void collect_stream_args(const char *s, char *dst, size_t cap) {
    static const char key[] = "\"arguments\":\"";
    size_t di = 0;
    const char *p = s;
    while ((p = strstr(p, key)) != NULL) {
        p += sizeof(key) - 1;  // at string body
        while (*p) {
            if (*p == '\\' && (p[1] == '"' || p[1] == '\\')) {
                if (di < cap - 1) dst[di++] = p[1];
                p += 2;
                continue;
            }
            if (*p == '"') break;  // unescaped closing quote
            if (di < cap - 1) dst[di++] = *p;
            p++;
        }
    }
    dst[di] = '\0';
}

// Structured tool_calls emission is validation-gated: ordinary content may be
// live, but even a complete DSML block produces no tool delta until the caller
// explicitly commits it through ember_sse_emit_tools().
static void test_tool_calls_emitted(void) {
    ember_sse_stream st;
    ember_sse_init(&st, "cc", "m", 1700000000, /*has_tools=*/true, false, false);
    ember_buf out = {0}, acc = {0};
    const char *full =
        "I'll check.<" PIPE "DSML" PIPE "tool_calls>"
        "<" PIPE "DSML" PIPE "invoke name=\"get_weather\">"
        "<" PIPE "DSML" PIPE "parameter name=\"city\" string=\"true\">Tokyo"
        "</" PIPE "DSML" PIPE "parameter>"
        "</" PIPE "DSML" PIPE "invoke></" PIPE "DSML" PIPE "tool_calls>";
    ember_buf_puts(&acc, full);
    ember_sse_update(&st, acc.ptr, acc.len, false, &out);
    CHECK(strstr(out.ptr ? out.ptr : "", "\"name\":\"get_weather\"") == NULL,
          "complete unvalidated tool name remains buffered");
    CHECK(strstr(out.ptr ? out.ptr : "", "\"type\":\"function\"") == NULL,
          "complete unvalidated tool header remains buffered");
    ember_sse_update(&st, acc.ptr, acc.len, true, &out);
    CHECK(strstr(out.ptr ? out.ptr : "", "\"type\":\"function\"") == NULL,
          "final splitter flush still does not bypass validation");
    bool had = ember_sse_emit_tools(&st, acc.ptr, acc.len, &out);
    CHECK(had, "tool call present (finish_reason=tool_calls)");
    CHECK(strstr(out.ptr, "\"name\":\"get_weather\"") != NULL, "tool name in header delta");
    CHECK(strstr(out.ptr, "\"type\":\"function\"") != NULL, "function type");
    CHECK(strstr(out.ptr, "\"arguments\":\"\"") != NULL, "header opens with empty arguments");
    char args[256];
    collect_stream_args(out.ptr, args, sizeof(args));
    CHECK(strcmp(args, "{\"city\":\"Tokyo\"}") == 0, "reassembled args == {\"city\":\"Tokyo\"}");
    ember_buf_free(&out); ember_buf_free(&acc); ember_sse_free(&st);
}

static void test_tool_attempt_reset(void) {
    const char *bad =
        "first-attempt"
        "<" PIPE "DSML" PIPE "tool_calls>"
        "<" PIPE "DSML" PIPE "invoke name=\"write_file\">";
    const char *good =
        "<" PIPE "DSML" PIPE "tool_calls>"
        "<" PIPE "DSML" PIPE "invoke name=\"read_file\">"
        "<" PIPE "DSML" PIPE "parameter name=\"path\" string=\"true\">README.md"
        "</" PIPE "DSML" PIPE "parameter>"
        "</" PIPE "DSML" PIPE "invoke></" PIPE "DSML" PIPE "tool_calls>";
    ember_sse_stream st;
    ember_buf out = {0};
    ember_sse_init(&st, "cc", "m", 1700000000, true, false, false);
    ember_sse_update(&st, bad, strlen(bad), false, &out);
    CHECK(strstr(out.ptr ? out.ptr : "", "first-attempt") != NULL,
          "ordinary text from first attempt may remain live");
    CHECK(strstr(out.ptr ? out.ptr : "", "write_file") == NULL,
          "unvalidated first-attempt tool header is absent");

    ember_sse_reset_attempt(&st, false);
    ember_sse_update(&st, good, strlen(good), true, &out);
    CHECK(ember_sse_emit_tools(&st, good, strlen(good), &out),
          "replacement attempt can commit a validated tool");
    CHECK(strstr(out.ptr ? out.ptr : "", "\"name\":\"read_file\"") != NULL,
          "replacement tool is emitted");
    CHECK(strstr(out.ptr ? out.ptr : "", "write_file") == NULL,
          "discarded tool attempt never appears after reset");
    ember_buf_free(&out);
    ember_sse_free(&st);
}

static void test_matching_tool_closer_required(void) {
    CHECK(ember_find_tool_end("<tool_calls></?DSML?tool_calls>") == NULL,
          "mismatched closer family is rejected");
    CHECK(ember_find_tool_end("<tool_calls></tool_calls>") != NULL,
          "matching closer family is accepted");
}

static void test_native_tool_id_is_registered(void) {
    const char *raw =
        "<ds_engine_tool_use>"
        "<ds_engine_tool_use_name>run</ds_engine_tool_use_name>"
        "</ds_engine_tool_use>";
    ember_sse_stream st;
    ember_buf out = {0};
    ember_sse_init(&st, "cc", "m", 1700000000, true, false, false);
    ember_sse_update(&st, raw, strlen(raw), true, &out);
    bool had = ember_sse_emit_tools(&st, raw, strlen(raw), &out);
    CHECK(had, "native tool call emitted");
    CHECK(st.n_tool_ids == 1, "native streamed id registered for replay");
    CHECK(st.n_tool_ids == 1 && strstr(out.ptr, st.tool_ids[0]) != NULL,
          "registered native id matches emitted id");
    ember_buf_free(&out);
    ember_sse_free(&st);
}

static void test_stop_precedes_tool(void) {
    const char *raw =
        "visible STOP<tool_calls><invoke name=\"danger\"></invoke></tool_calls>";
    char *stops[] = {(char *)"STOP"};
    ember_sse_stream st;
    ember_buf out = {0};
    ember_sse_init(&st, "cc", "m", 1700000000, true, false, false);
    st.stops = stops;
    st.n_stops = 1;
    st.max_stop_len = 4;
    ember_sse_update(&st, raw, strlen(raw), true, &out);
    CHECK(st.mode == EMBER_SSE_SUPPRESS,
          "complete stop takes precedence over later tool marker");
    CHECK(!ember_sse_emit_tools(&st, raw, strlen(raw), &out),
          "tool after stop is not emitted");
    ember_buf content = {0};
    collect_field(out.ptr ? out.ptr : "", "content", &content);
    CHECK(strcmp(content.ptr ? content.ptr : "", "visible ") == 0,
          "content is cut at stop before tool");
    ember_buf_free(&content);
    ember_buf_free(&out);
    ember_sse_free(&st);
}

static void test_more_than_sixteen_tool_ids(void) {
    enum { N = 20 };
    ember_buf raw = {0}, out = {0};
    ember_buf_puts(&raw, "<tool_calls>");
    for (int i = 0; i < N; ++i)
        ember_buf_printf(&raw,
            "<invoke name=\"tool_%d\"></invoke>", i);
    ember_buf_puts(&raw, "</tool_calls>");
    ember_sse_stream st;
    ember_sse_init(&st, "cc", "m", 1700000000, true, false, false);
    ember_sse_update(&st, raw.ptr, raw.len, true, &out);
    CHECK(ember_sse_emit_tools(&st, raw.ptr, raw.len, &out),
          "many validated tools are emitted");
    CHECK(st.n_tool_ids == N,
          "every call id is retained beyond the former sixteen-call cap");
    CHECK(st.n_tool_ids == N && strstr(out.ptr, st.tool_ids[N - 1]) != NULL,
          "the final dynamic call id is the emitted id");
    ember_sse_free(&st);
    ember_buf_free(&raw);
    ember_buf_free(&out);
}
