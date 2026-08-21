#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/common/json.h"
#include "../src/common/json_util.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg)                                                    \
    do { if (cond) g_pass++; else { g_fail++; printf("  FAIL: %s\n", msg); } } while (0)

static void test_chat_request(void) {
    const char *body =
        "{\"model\":\"deepseek-v4-flash\",\"stream\":true,\"max_tokens\":4096,"
        "\"temperature\":0,\"reasoning_effort\":\"high\","
        "\"messages\":[{\"role\":\"system\",\"content\":\"You are helpful.\"},"
        "{\"role\":\"user\",\"content\":\"hi \\ud83d\\udc4b\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_time\"}}]}";
    ember_json *v = ember_json_parse(body);
    CHECK(v != NULL, "parse chat request");
    if (!v) return;
    CHECK(strcmp(ember_json_str(ember_json_get(v, "model"), "") ,
                 "deepseek-v4-flash") == 0, "model");
    CHECK(ember_json_bool(ember_json_get(v, "stream"), false) == true, "stream");
    CHECK(ember_json_num(ember_json_get(v, "max_tokens"), 0) == 4096, "max_tokens");
    CHECK(ember_json_num(ember_json_get(v, "temperature"), -1) == 0, "temperature 0");

    const ember_json *msgs = ember_json_get(v, "messages");
    CHECK(ember_json_len(msgs) == 2, "two messages");
    const ember_json *u = ember_json_at(msgs, 1);
    CHECK(strcmp(ember_json_str(ember_json_get(u, "role"), ""), "user") == 0, "user role");
    // 👋 surrogate pair → 👋 (F0 9F 91 8B)
    CHECK(strcmp(ember_json_str(ember_json_get(u, "content"), ""),
                 "hi \xf0\x9f\x91\x8b") == 0, "surrogate pair → emoji utf-8");

    const ember_json *tools = ember_json_get(v, "tools");
    CHECK(ember_json_len(tools) == 1, "one tool");
    const ember_json *fn = ember_json_get(ember_json_at(tools, 0), "function");
    CHECK(strcmp(ember_json_str(ember_json_get(fn, "name"), ""), "get_time") == 0,
          "nested tool function name");
    ember_json_free(v);
}

static void test_edge_cases(void) {
    ember_json *e = ember_json_parse("{}");
    CHECK(e && e->type == EMBER_JSON_OBJECT && ember_json_len(e) == 0, "empty object");
    ember_json_free(e);

    ember_json *a = ember_json_parse("[1, 2.5, -3, true, null, \"x\"]");
    CHECK(a && ember_json_len(a) == 6, "mixed array");
    CHECK(a && ember_json_num(ember_json_at(a, 1), 0) == 2.5, "float element");
    CHECK(a && ember_json_num(ember_json_at(a, 2), 0) == -3, "negative element");
    ember_json_free(a);

    CHECK(ember_json_parse("{bad}") == NULL, "syntax error → NULL");
    CHECK(ember_json_parse("{\"a\":}") == NULL, "missing value → NULL");
    CHECK(ember_json_parse("[1,2,") == NULL, "unterminated array → NULL");
    CHECK(ember_json_parse("{\"x\":\"line\nbreak\"}") == NULL,
          "unescaped control character rejected");
    CHECK(ember_json_parse("{\"x\":\"a\\u0000b\"}") == NULL,
          "escaped NUL rejected because DOM strings are NUL-terminated");
    static const char embedded_nul[] = "{\"x\":1}\0{\"hidden\":true}";
    CHECK(ember_json_parse_n(embedded_nul, sizeof(embedded_nul) - 1) == NULL,
          "length-delimited parser rejects hidden suffix after NUL");
    CHECK(ember_json_parse("1e400") == NULL,
          "non-finite number rejected");

    ember_json *hi = ember_json_parse("\"\\ud800x\"");
    CHECK(hi && strcmp(ember_json_str(hi, ""), "\xef\xbf\xbdx") == 0,
          "lone high surrogate becomes U+FFFD");
    ember_json_free(hi);
    ember_json *lo = ember_json_parse("\"\\udc00\"");
    CHECK(lo && strcmp(ember_json_str(lo, ""), "\xef\xbf\xbd") == 0,
          "lone low surrogate becomes U+FFFD");
    ember_json_free(lo);
    ember_json *pair = ember_json_parse("\"\\ud800\\u0041\"");
    CHECK(pair && strcmp(ember_json_str(pair, ""), "\xef\xbf\xbd" "A") == 0,
          "invalid pair does not consume following escape");
    ember_json_free(pair);
}

static void check_escaped(const char *input, size_t len, const char *expected,
                          const char *message) {
    ember_buf b = {0};
    ember_json_escape_n(&b, input, len);
    CHECK(strcmp(b.ptr ? b.ptr : "", expected) == 0, message);
    ember_json *parsed = ember_json_parse(b.ptr ? b.ptr : "");
    CHECK(parsed != NULL, "escaped UTF-8 is parseable JSON");
    ember_json_free(parsed);
    ember_buf_free(&b);
}

static void test_json_output_utf8(void) {
    check_escaped("\"\\\n\r\t\b\f\x01" "A", 9,
                  "\"\\\"\\\\\\n\\r\\t\\b\\f\\u0001A\"",
                  "ASCII controls and metacharacters are escaped");
    check_escaped("\xc2\xa2", 2, "\"\xc2\xa2\"",
                  "two-byte UTF-8 passes through");
    check_escaped("\xe0\xa0\x80\xed\x9f\xbf\xe1\x80\x80", 9,
                  "\"\xe0\xa0\x80\xed\x9f\xbf\xe1\x80\x80\"",
                  "three-byte UTF-8 boundary cases pass through");
    check_escaped("\xf0\x90\x80\x80\xf4\x8f\xbf\xbf\xf1\x80\x80\x80", 12,
                  "\"\xf0\x90\x80\x80\xf4\x8f\xbf\xbf\xf1\x80\x80\x80\"",
                  "four-byte UTF-8 boundary cases pass through");
    check_escaped("CJK \xe4\xb8\xad emoji \xf0\x9f\x91\x8b", 18,
                  "\"CJK \xe4\xb8\xad emoji \xf0\x9f\x91\x8b\"",
                  "valid UTF-8 passes through unchanged");
    check_escaped("cut \xf0\x9f", 6, "\"cut \xef\xbf\xbd\"",
                  "truncated UTF-8 becomes one replacement character");
    check_escaped("bad \xe2" "x", 6, "\"bad \xef\xbf\xbdx\"",
                  "invalid continuation does not consume next codepoint");
    check_escaped("\xed\xa0\x80", 3, "\"\xef\xbf\xbd\"",
                  "UTF-8 surrogate encoding is replaced");
    check_escaped("\xf4\x90\x80\x80", 4, "\"\xef\xbf\xbd\"",
                  "codepoints above U+10FFFF are replaced");
    check_escaped("\xc0\x80", 2, "\"\xef\xbf\xbd\"",
                  "overlong UTF-8 is replaced");
    check_escaped("\xc2" "A", 2, "\"\xef\xbf\xbd" "A\"",
                  "bad two-byte continuation preserves following ASCII");
    check_escaped("\xe2\x80" "A", 3, "\"\xef\xbf\xbd" "A\"",
                  "bad three-byte continuation is one replacement");
    check_escaped("\xf1\x80\x80" "A", 4, "\"\xef\xbf\xbd" "A\"",
                  "bad four-byte continuation is one replacement");
    check_escaped("\xf5\x80\x80\x80", 4, "\"\xef\xbf\xbd\"",
                  "out-of-range leading byte and continuations are replaced once");
    check_escaped("\x80\x80\x80", 3, "\"\xef\xbf\xbd\"",
                  "stray continuation run is replaced once");

    CHECK(ember_json_utf8_sequence_len(NULL, 0) == 0,
          "empty UTF-8 input has no sequence");
    CHECK(ember_json_invalid_utf8_span(NULL, 0) == 0,
          "empty invalid input has no span");

    ember_buf content = {0};
    ember_json_escape_content(&content, NULL);
    CHECK(content.len == 0, "NULL JSON content is empty");
    ember_json_escape(&content, NULL);
    CHECK(strcmp(content.ptr, "\"\"") == 0, "NULL JSON string becomes empty literal");
    ember_buf_free(&content);
}

int main(void) {
    printf("ember json tests\n");
    test_chat_request();
    test_edge_cases();
    test_json_output_utf8();
    printf("──────────────────────────────\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
