#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/model/tool_parser.h"
#include "fixtures_real_failures.h"

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

static void test_attributes_are_bounded_and_exact(void) {
    const char *tag = " xname=\"wrong\" name=\"right\" string=\"false\">";
    const char *limit = tag + strlen(tag);
    char *name = ember_dsml_attr(tag, limit, "name");
    CHECK(name && strcmp(name, "right") == 0,
          "attribute lookup does not match a suffix of another name");
    free(name);

    const char *t =
        "<tool_calls><invoke xname=\"wrong\" name=\"run\">"
        "<parameter name=\"x\" string=\"false\">not-json</parameter>"
        "</invoke></tool_calls>";
    ember_tool_calls tc = {0};
    ember_parse_dsml_tool_calls(t, &tc);
    CHECK(tc.len == 1 && strcmp(tc.calls[0].name, "run") == 0,
          "invoke uses the exact name attribute");
    CHECK(tc.len == 1 && strcmp(tc.calls[0].arguments, "{\"x\":null}") == 0,
          "invalid raw parameter cannot corrupt arguments JSON");
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

static void test_replay_requires_matching_arguments(void) {
    const char *raw =
        "<" PIPE "DSML" PIPE "tool_calls>"
        "<" PIPE "DSML" PIPE "invoke name=\"search\">"
        "<" PIPE "DSML" PIPE "parameter name=\"q\" string=\"true\">old"
        "</" PIPE "DSML" PIPE "parameter>"
        "</" PIPE "DSML" PIPE "invoke>"
        "</" PIPE "DSML" PIPE "tool_calls>";
    ember_tool_calls expected = {0};
    expected.calls = calloc(1, sizeof(*expected.calls));
    expected.len = expected.cap = 1;
    expected.calls[0].name = strdup("search");
    expected.calls[0].arguments = strdup("{ \"q\" : \"old\" }");
    CHECK(ember_tool_calls_match_raw(raw, &expected),
          "replay accepts JSON-equivalent arguments");
    free(expected.calls[0].arguments);
    expected.calls[0].arguments = strdup("{\"q\":\"changed\"}");
    CHECK(!ember_tool_calls_match_raw(raw, &expected),
          "replay rejects changed client arguments");
    free(expected.calls[0].name);
    expected.calls[0].name = strdup("old");
    free(expected.calls[0].arguments);
    expected.calls[0].arguments = strdup("{\"q\":\"old\"}");
    CHECK(!ember_tool_calls_match_raw(raw, &expected),
          "replay does not match a name appearing only in argument text");
    ember_tool_calls_free(&expected);
}

static void test_parse_report_rejects_nested_dsml_in_string(void) {
    // Production failure shape: while inside write_file.content the model copied
    // a fresh assistant/tool turn. The nested terminal parameter closer was
    // previously mistaken for the outer content closer, yielding truncated but
    // executable Python.
    const char *t =
        "<?DSML?tool_calls>"
        "<?DSML?invoke name=\"write_file\">"
        "<?DSML?parameter name=\"path\" string=\"true\">/tmp/x.py"
        "</?DSML?parameter>"
        "<?DSML?parameter name=\"content\" string=\"true\">"
        "\"\"\"usage\\n"
        "Considering the limited time by the user.\\n</think>\\n"
        "<?DSML?tool_calls>"
        "<?DSML?invoke name=\"terminal\">"
        "<?DSML?parameter name=\"command\" string=\"true\">python3 --version"
        "</?DSML?parameter>"
        "</?DSML?invoke>"
        "</?DSML?tool_calls>"
        "</?DSML?parameter>"
        "</?DSML?invoke>"
        "</?DSML?tool_calls>";
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    int n = ember_parse_dsml_tool_calls_ex(t, &tc, &report);
    CHECK(n == 0 && tc.len == 0,
          "nested DSML in a string argument is never executable");
    CHECK(report.found && report.contaminated,
          "nested DSML contamination is reported");
    ember_tool_calls_free(&tc);
}

static void test_parse_report_marks_repaired_tail_incomplete(void) {
    const char *t =
        "<" PIPE "DSML" PIPE "tool_calls>"
        "<" PIPE "DSML" PIPE "invoke name=\"write_file\">"
        "<" PIPE "DSML" PIPE "parameter name=\"path\" string=\"true\">/tmp/x.py";
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    int n = ember_parse_dsml_tool_calls_ex(t, &tc, &report);
    CHECK(n == 1, "compatibility parser still recovers a truncated tail");
    CHECK(report.found && !report.complete && report.repaired,
          "executable gate can distinguish repaired output");
    ember_tool_calls_free(&tc);
}

static void test_executable_report_rejects_invalid_raw_json(void) {
    const char *t =
        "<tool_calls><invoke name=\"run\">"
        "<parameter name=\"x\" string=\"false\">not-json</parameter>"
        "</invoke></tool_calls>";
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    CHECK(ember_parse_dsml_tool_calls_ex(t, &tc, &report) == 0 &&
          report.invalid_json,
          "executable parser rejects invalid string=false JSON");
    ember_tool_calls_free(&tc);
}

static void test_wrapper_is_authoritative(void) {
    const char *trailing =
        "<tool_calls></tool_calls>"
        "<invoke name=\"danger\"></invoke>";
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    CHECK(ember_parse_dsml_tool_calls_ex(trailing, &tc, &report) == 0,
          "invoke after wrapper is not parsed");
    CHECK(report.found && report.trailing && report.invocations == 0,
          "non-whitespace after wrapper is reported");
    ember_tool_calls_free(&tc);

    const char *mixed =
        "<tool_calls><invoke name=\"safe\"></invoke>"
        "<?DSML?invoke name=\"danger\"></?DSML?invoke></tool_calls>";
    memset(&report, 0, sizeof(report));
    CHECK(ember_parse_dsml_tool_calls_ex(mixed, &tc, &report) == 0 &&
          report.mixed_syntax,
          "mixed DSML families are rejected as one executable block");
    ember_tool_calls_free(&tc);

    const char *mixed_native =
        "<tool_calls><invoke name=\"safe\"></invoke>"
        "<ds_engine_tool_use>"
        "<ds_engine_tool_use_name>danger</ds_engine_tool_use_name>"
        "</ds_engine_tool_use></tool_calls>";
    memset(&report, 0, sizeof(report));
    CHECK(ember_parse_dsml_tool_calls_ex(mixed_native, &tc, &report) == 0 &&
          report.mixed_syntax,
          "native and DSML tool syntax cannot share one wrapper");
    ember_tool_calls_free(&tc);
}

static void test_malformed_nested_tags_are_not_executable(void) {
    const char *nested =
        "<tool_calls><invoke name=\"safe\">"
        "<invoke name=\"danger\"></invoke></invoke></tool_calls>";
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    CHECK(ember_parse_dsml_tool_calls_ex(nested, &tc, &report) == 0 &&
          report.malformed,
          "same-family nested invokes are rejected");
    ember_tool_calls_free(&tc);

    const char *bad_attr =
        "<tool_calls><invoke name=\"safe\">"
        "<parameter string=\"maybe\">x</parameter>"
        "</invoke></tool_calls>";
    memset(&report, 0, sizeof(report));
    CHECK(ember_parse_dsml_tool_calls_ex(bad_attr, &tc, &report) == 0 &&
          report.malformed,
          "missing names and invalid string attributes are rejected");
    ember_tool_calls_free(&tc);

    const char *native_trailing =
        "<ds_engine_tool_use>"
        "<ds_engine_tool_use_name>safe</ds_engine_tool_use_name>"
        "</ds_engine_tool_use>not-protocol";
    memset(&report, 0, sizeof(report));
    CHECK(ember_parse_dsml_tool_calls_ex(native_trailing, &tc, &report) == 0 &&
          report.trailing,
          "native tool blocks reject trailing non-protocol output");
    ember_tool_calls_free(&tc);

    const char *native_parallel =
        "reasoning before tools\n"
        "<ds_engine_tool_use>"
        "<ds_engine_tool_use_name>first</ds_engine_tool_use_name>"
        "</ds_engine_tool_use>\n"
        "<ds_engine_tool_use>"
        "<ds_engine_tool_use_name>second</ds_engine_tool_use_name>"
        "</ds_engine_tool_use>\n";
    memset(&report, 0, sizeof(report));
    CHECK(ember_parse_dsml_tool_calls_ex(native_parallel, &tc, &report) == 2 &&
          report.complete && !report.trailing && report.invocations == 2,
          "adjacent native blocks remain valid parallel calls");
    ember_tool_calls_free(&tc);
}


// ── real degraded output, captured in production ─────────────────────────────
// Everything below is a sanitized reduction of captured failure shapes, not an
// invented parser grammar. Real corruption is messier than a minimal unit case: these
// four carry at least five distinct mutations between them --
//
//   <?DSML?tool_caddy>   marker name bleeding a nearby word ("Caddy config")
//   <?DSML?tool_cards>   ditto ("index card")
//   <?DSML?tool_alls>    a dropped character
//   name= "skill_manage" a spurious space after the '='
//   </?DSML?_manage>     a mangled closing tag
//
// and, crucially, U+003F question marks where the DSML delimiter U+FF5C should
// be. None of it is a tool call, and the contract is that the parser says so:
// ember must classify these as ordinary text, because treating a corrupted
// marker as a call is how a malformed block reaches a tool executor.
static void test_real_degraded_output_is_not_a_tool_call(void) {
    const struct { const char *name; const char *text; } specimens[] = {
        { "seq=1297 pseudo-marker (pre-DRY)",      REAL_PSEUDO_MARKER_1297 },
        { "seq=136 pseudo-marker (post-breaker)",  REAL_PSEUDO_MARKER_136  },
        { "seq=130 fragmentation + pseudo-marker", REAL_FRAGMENTATION_130  },
        { "seq=81 bech32 repetition loop",         REAL_REPETITION_81      },
    };
    for (size_t i = 0; i < sizeof(specimens) / sizeof(specimens[0]); i++) {
        ember_tool_calls tc = {0};
        int n = ember_parse_dsml_tool_calls(specimens[i].text, &tc);
        CHECK(n == 0 && tc.len == 0, specimens[i].name);
        ember_tool_calls_free(&tc);
    }
}

// The same specimens must not be mistaken for a REPLAY of a known-good call
// either -- match_raw is what decides whether stored bytes may be spliced back
// verbatim, so a false positive there would resurrect corrupted markup.
static void test_real_degraded_output_never_matches_a_replay(void) {
    const char *good =
        "<" PIPE "DSML" PIPE "tool_calls>\n"
        "<" PIPE "DSML" PIPE "invoke name=\"skill_manage\">\n"
        "</" PIPE "DSML" PIPE "invoke>\n"
        "</" PIPE "DSML" PIPE "tool_calls>";
    ember_tool_calls expected = {0};
    int n = ember_parse_dsml_tool_calls(good, &expected);
    CHECK(n == 1, "control: the well-formed call still parses");
    CHECK(!ember_tool_calls_match_raw(REAL_PSEUDO_MARKER_136, &expected),
          "seq=136 corrupted markup must not match a stored replay");
    CHECK(!ember_tool_calls_match_raw(REAL_FRAGMENTATION_130, &expected),
          "seq=130 corrupted markup must not match a stored replay");
    ember_tool_calls_free(&expected);
}

int main(void) {
    test_real_degraded_output_is_not_a_tool_call();
    test_real_degraded_output_never_matches_a_replay();
    printf("ember tool_parser tests\n");
    test_single_string_arg();
    test_mixed_arg_types();
    test_attributes_are_bounded_and_exact();
    test_short_spelling();
    test_ascii_spelling_multi();
    test_no_tool_calls();
    test_replay_requires_matching_arguments();
    test_parse_report_rejects_nested_dsml_in_string();
    test_parse_report_marks_repaired_tail_incomplete();
    test_executable_report_rejects_invalid_raw_json();
    test_wrapper_is_authoritative();
    test_malformed_nested_tags_are_not_executable();
    printf("──────────────────────────────\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
