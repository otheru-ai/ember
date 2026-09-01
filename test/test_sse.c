// Unit tests for the SSE streaming layer. These exercise the exact failure
// modes that broke lucebox's incremental emitter and had to be patched one at
// a time — here they must pass by construction. Drives the stream token-by-
// token (accumulating raw) the way the real server does.
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "../src/common/buf.h"
#include "../src/server/sse.h"

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

int main(void) {
    printf("ember sse tests\n");
    test_plain_content();
    test_split_emoji();
    test_reasoning_split();
    test_split_think_close();
    test_tool_marker_suppressed();
    test_short_dsml_spelling();
    test_utf8_safe_limit_direct();
    test_tool_calls_emitted();
    printf("──────────────────────────────\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

// Reassemble the tool-call arguments from *incremental* streaming deltas: scan
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

// appended: structured tool_calls emission — now *incremental* (header delta
// with empty arguments, then per-parameter argument fragments). See the arg
// deltas asserted below; ember_sse_emit_tools is the no-op-after-stream path.
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
    ember_sse_update(&st, acc.ptr, acc.len, true, &out);
    // Batch fallback must NOT re-emit once the stream already sent the call: it
    // still reports had=1 (so the caller sets finish_reason=tool_calls).
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
