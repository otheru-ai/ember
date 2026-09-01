#include <stdio.h>
#include <string.h>

#include "../src/model/tool_parser.h"

#define PIPE "\xef\xbd\x9c"  // U+FF5C

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg)                                                    \
    do { if (cond) g_pass++; else { g_fail++; printf("  FAIL: %s\n", msg); } } while (0)

static void test_single_string_arg(void) {
    const char *t =
        "<" PIPE "DSML" PIPE "tool_calls>\n"
        "<" PIPE "DSML" PIPE "invoke name=\"get_weather\">\n"
        "<" PIPE "DSML" PIPE "parameter name=\"city\" string=\"true\">Tokyo</" PIPE "DSML" PIPE "parameter>\n"
        "</" PIPE "DSML" PIPE "invoke>\n"
        "</" PIPE "DSML" PIPE "tool_calls>";
    ember_tool_calls tc = {0};
    int n = ember_parse_dsml_tool_calls(t, &tc);
    CHECK(n == 1, "one call parsed");
    CHECK(n == 1 && strcmp(tc.calls[0].name, "get_weather") == 0, "name");
    CHECK(n == 1 && strcmp(tc.calls[0].arguments, "{\"city\":\"Tokyo\"}") == 0,
          "string arg escaped");
    ember_tool_calls_free(&tc);
}

static void test_mixed_arg_types(void) {
    const char *t =
        "<" PIPE "DSML" PIPE "tool_calls>"
        "<" PIPE "DSML" PIPE "invoke name=\"search\">"
        "<" PIPE "DSML" PIPE "parameter name=\"q\" string=\"true\">cats</" PIPE "DSML" PIPE "parameter>"
        "<" PIPE "DSML" PIPE "parameter name=\"limit\" string=\"false\">5</" PIPE "DSML" PIPE "parameter>"
        "</" PIPE "DSML" PIPE "invoke>"
        "</" PIPE "DSML" PIPE "tool_calls>";
    ember_tool_calls tc = {0};
    ember_parse_dsml_tool_calls(t, &tc);
    CHECK(tc.len == 1 &&
          strcmp(tc.calls[0].arguments, "{\"q\":\"cats\",\"limit\":5}") == 0,
          "string=true escaped, string=false verbatim number");
    ember_tool_calls_free(&tc);
}

static void test_short_spelling(void) {
    const char *t =
        "<DSML" PIPE "tool_calls>"
        "<DSML" PIPE "invoke name=\"ping\">"
        "</DSML" PIPE "invoke>"
        "</DSML" PIPE "tool_calls>";
    ember_tool_calls tc = {0};
    ember_parse_dsml_tool_calls(t, &tc);
    CHECK(tc.len == 1 && strcmp(tc.calls[0].name, "ping") == 0,
          "short spelling, zero-arg call");
    CHECK(tc.len == 1 && strcmp(tc.calls[0].arguments, "{}") == 0, "empty args");
    ember_tool_calls_free(&tc);
}

static void test_ascii_spelling_multi(void) {
    const char *t =
        "<?DSML?tool_calls>"
        "<?DSML?invoke name=\"a\"><?DSML?parameter name=\"x\" string=\"true\">1</?DSML?parameter></?DSML?invoke>"
        "<?DSML?invoke name=\"b\"></?DSML?invoke>"
        "</?DSML?tool_calls>";
    ember_tool_calls tc = {0};
    ember_parse_dsml_tool_calls(t, &tc);
    CHECK(tc.len == 2, "ascii spelling, two invokes");
    CHECK(tc.len == 2 && strcmp(tc.calls[0].name, "a") == 0 &&
          strcmp(tc.calls[1].name, "b") == 0, "both names");
    ember_tool_calls_free(&tc);
}

static void test_no_tool_calls(void) {
    ember_tool_calls tc = {0};
    int n = ember_parse_dsml_tool_calls("just some prose, no tools here", &tc);
    CHECK(n == 0, "no false positives on prose");
    ember_tool_calls_free(&tc);
}

int main(void) {
    printf("ember tool_parser tests\n");
    test_single_string_arg();
    test_mixed_arg_types();
    test_short_spelling();
    test_ascii_spelling_multi();
    test_no_tool_calls();
    printf("──────────────────────────────\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
