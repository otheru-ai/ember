#include "../src/model/tool_parser_qwen4.h"

#include <stdio.h>
#include <string.h>

static int g_pass;
static int g_fail;

#define CHECK(cond, msg) do { \
    if (cond) ++g_pass; \
    else { ++g_fail; fprintf(stderr, "FAIL: %s\n", msg); } \
} while (0)

static const char *TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"weather\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"city\":{\"type\":\"string\"},"
    "\"days\":{\"type\":\"integer\"},"
    "\"detail\":{\"type\":\"boolean\"},"
    "\"options\":{\"type\":\"object\"}}}}}]";

static void test_schema_coercion(void) {
    const char *text =
        "Reasoning first. <tool_call>\n<function=weather>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "<parameter=days>\n2\n</parameter>\n"
        "<parameter=detail>false</parameter>\n"
        "<parameter=options>{\"units\":\"c\"}</parameter>\n"
        "</function>\n</tool_call>";
    ember_tool_calls calls = {0};
    ember_qwen_tool_parse_report report = {0};
    int n = ember_parse_qwen_tool_calls(text, TOOLS, &calls, &report);
    CHECK(n == 1 && calls.len == 1, "one complete wrapper parsed");
    CHECK(report.found && report.complete && !report.malformed,
          "complete report populated");
    CHECK(calls.len == 1 && strcmp(calls.calls[0].name, "weather") == 0,
          "function name parsed");
    CHECK(calls.len == 1 && strcmp(calls.calls[0].arguments,
          "{\"city\":\"Paris\",\"days\":2,\"detail\":false,"
          "\"options\":{\"units\":\"c\"}}") == 0,
          "raw parameters coerced through advertised schema");
    ember_tool_calls_free(&calls);
}

static void test_parallel_wrappers(void) {
    const char *text =
        "<tool_call><function=weather><parameter=city>A</parameter>"
        "</function></tool_call>\n"
        "<tool_call><function=weather><parameter=city>B</parameter>"
        "</function></tool_call>";
    ember_tool_calls calls = {0};
    ember_qwen_tool_parse_report report = {0};
    CHECK(ember_parse_qwen_tool_calls(text, TOOLS, &calls, &report) == 2,
          "parallel calls are repeated wrappers");
    CHECK(report.wrappers == 2 && calls.len == 2,
          "parallel wrapper count retained");
    ember_tool_calls_free(&calls);
}

static void test_fail_closed_shapes(void) {
    const char *cases[] = {
        "<tool_call><function=weather></function>",
        "<tool_call><function=weather></function></tool_call> suffix",
        "<tool_call><function=weather><parameter=city>x"
            "<tool_call>evil</parameter></function></tool_call>",
        "<tool_call><function=weather><parameter=city>A</parameter>"
            "<parameter=city>B</parameter></function></tool_call>",
        "<tool_call><function=weather></function><function=other>"
            "</function></tool_call>",
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        ember_tool_calls calls = {0};
        ember_qwen_tool_parse_report report = {0};
        CHECK(ember_parse_qwen_tool_calls(cases[i], TOOLS, &calls, &report) == 0,
              "malformed Qwen wrapper is not executable");
        CHECK(calls.len == 0, "rejected wrapper leaves no calls");
        ember_tool_calls_free(&calls);
    }
}

static void test_unknown_parameter_stays_string(void) {
    const char *text =
        "<tool_call><function=weather><parameter=extra>{\"x\":1}"
        "</parameter></function></tool_call>";
    ember_tool_calls calls = {0};
    CHECK(ember_parse_qwen_tool_calls(text, TOOLS, &calls, NULL) == 1,
          "unknown parameter remains parseable for schema gate");
    CHECK(calls.len == 1 && strcmp(calls.calls[0].arguments,
          "{\"extra\":\"{\\\"x\\\":1}\"}") == 0,
          "unknown parameter is a string, not guessed JSON");
    ember_tool_calls_free(&calls);
}

int main(void) {
    test_schema_coercion();
    test_parallel_wrappers();
    test_fail_closed_shapes();
    test_unknown_parameter_stays_string();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
