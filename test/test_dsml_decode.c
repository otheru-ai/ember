// B6 — tests for the DSML decode-state tracker (structural-token greedy).
#include "../src/model/dsml_decode.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#define PIPE "\xef\xbd\x9c"  // U+FF5C

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

// Feed `s` to a fresh tracker and return the resulting decode state.
static ember_dsml_decode_state state_after(const char *s) {
    ember_dsml_tracker dt;
    ember_dsml_tracker_init(&dt);
    ember_dsml_tracker_update(&dt, s, strlen(s));
    return dt.decode;
}
static bool greedy_after(const char *s) {
    ember_dsml_tracker dt;
    ember_dsml_tracker_init(&dt);
    ember_dsml_tracker_update(&dt, s, strlen(s));
    return ember_dsml_force_greedy(&dt);
}

// Feed byte-by-byte (the real call pattern) and return the final greedy verdict.
static bool greedy_incremental(const char *s) {
    ember_dsml_tracker dt;
    ember_dsml_tracker_init(&dt);
    size_t n = strlen(s);
    for (size_t i = 1; i <= n; i++) ember_dsml_tracker_update(&dt, s, i);
    return ember_dsml_force_greedy(&dt);
}

int main(void) {
    printf("test_dsml_decode\n");

    // Plain prose before any tool call: outside → sample (not greedy).
    CHECK(state_after("Let me think about this.") == EMBER_DSML_OUTSIDE);
    CHECK(!greedy_after("Let me think about this."));

    // Inside the scaffolding (tags) → structural → greedy.
    const char *scaffold =
        "<" PIPE "DSML" PIPE "tool_calls>\n<" PIPE "DSML" PIPE "invoke name=\"get_time\">";
    CHECK(state_after(scaffold) == EMBER_DSML_STRUCTURAL);
    CHECK(greedy_after(scaffold));

    // A string="true" parameter body → payload → sample (NOT greedy).
    const char *string_param =
        "<" PIPE "DSML" PIPE "tool_calls>\n<" PIPE "DSML" PIPE "invoke name=\"run\">"
        "<" PIPE "DSML" PIPE "parameter name=\"cmd\" string=\"true\">ls -la /var";
    CHECK(state_after(string_param) == EMBER_DSML_STRING_BODY);
    CHECK(!greedy_after(string_param));

    // A JSON (non-string) parameter: structure greedy, string value sampled.
    const char *json_struct =
        "<" PIPE "DSML" PIPE "tool_calls>\n<" PIPE "DSML" PIPE "invoke name=\"f\">"
        "<" PIPE "DSML" PIPE "parameter name=\"arguments\">{\"path\": ";
    CHECK(state_after(json_struct) == EMBER_DSML_JSON_STRUCTURAL);
    CHECK(greedy_after(json_struct));

    const char *json_string =
        "<" PIPE "DSML" PIPE "tool_calls>\n<" PIPE "DSML" PIPE "invoke name=\"f\">"
        "<" PIPE "DSML" PIPE "parameter name=\"arguments\">{\"path\": \"/etc/host";
    CHECK(state_after(json_string) == EMBER_DSML_JSON_STRING);
    CHECK(!greedy_after(json_string));

    // Closing the value returns to JSON structure → greedy again.
    const char *json_after_string =
        "<" PIPE "DSML" PIPE "tool_calls>\n<" PIPE "DSML" PIPE "invoke name=\"f\">"
        "<" PIPE "DSML" PIPE "parameter name=\"arguments\">{\"path\": \"/etc/hosts\", ";
    CHECK(state_after(json_after_string) == EMBER_DSML_JSON_STRUCTURAL);

    // After the whole tool_calls block closes → outside again.
    const char *closed =
        "<" PIPE "DSML" PIPE "tool_calls>\n<" PIPE "DSML" PIPE "invoke name=\"get_time\">"
        "</" PIPE "DSML" PIPE "invoke>\n</" PIPE "DSML" PIPE "tool_calls>";
    CHECK(state_after(closed) == EMBER_DSML_OUTSIDE);
    CHECK(!greedy_after(closed));

    // Byte-by-byte feeding must reach the same verdicts (the real call pattern).
    CHECK(greedy_incremental(scaffold));
    CHECK(!greedy_incremental(string_param));
    CHECK(greedy_incremental(json_struct));
    CHECK(!greedy_incremental(json_string));

    // Reasoning that merely mentions the tag name in prose stays outside until a
    // real opener appears.
    CHECK(!greedy_after("I will call the get_time tool_calls now."));

    if (failures == 0) printf("  all passed\n");
    return failures ? 1 : 0;
}
