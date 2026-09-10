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

static void test_parse_report_keeps_nested_dsml_in_string_whole(void) {
    // Production failure shape: while inside write_file.content the model copied
    // a fresh assistant/tool turn. Taking the FIRST inner parameter closer as
    // the outer one yielded truncated but executable Python, so this was
    // rejected outright (report.contaminated) and the turn died with a 422.
    //
    // Rejecting was the safe half of the answer but not the right one: the
    // model's intent is plain, and the value is recoverable exactly when the
    // nesting balances. Matching closers by DEPTH keeps the value whole, so the
    // call is faithful rather than truncated -- the hazard the old rule guarded
    // against cannot occur. Unbalanced nesting is still refused; see below.
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
    const char *flag = getenv("EMBER_DSML_NESTED_VALUES");
    if (flag && flag[0] == '1') {
        CHECK(n == 1 && tc.len == 1, "a balanced nested value is recoverable");
        CHECK(!report.contaminated && !report.malformed && !report.trailing,
              "balanced nesting is not a protocol error");
        if (tc.len == 1) {
            CHECK(strcmp(tc.calls[0].name, "write_file") == 0,
                  "the outer call is the one executed");
            // The whole point: the value must carry the ENTIRE nested block. If
            // the inner closer had ended it, "python3 --version" would be absent
            // and the written file silently truncated.
            CHECK(strstr(tc.calls[0].arguments, "python3 --version") != NULL,
                  "the nested block survives inside the value, untruncated");
            CHECK(strstr(tc.calls[0].arguments, "/tmp/x.py") != NULL,
                  "the earlier parameter is still parsed");
        }
    } else {
        // Dark by default: the pre-existing contract is unchanged, so a release
        // that does not enable the feature behaves exactly as before.
        CHECK(n == 0 && tc.len == 0,
              "dark: nested DSML in a string argument is never executable");
        CHECK(report.found && report.contaminated,
              "dark: nested DSML contamination is reported");
    }
    ember_tool_calls_free(&tc);
}

static void test_parse_report_rejects_unbalanced_nested_dsml(void) {
    // The safety half. Here the nested block opens a parameter it never closes,
    // so no faithful outer value exists at any depth. This must stay
    // unexecutable: emitting it would truncate the argument, which is the exact
    // hazard the contamination guard was written for.
    const char *t =
        "<?DSML?tool_calls>"
        "<?DSML?invoke name=\"write_file\">"
        "<?DSML?parameter name=\"content\" string=\"true\">"
        "<?DSML?parameter name=\"inner\" string=\"true\">never closed"
        "</?DSML?invoke>"
        "</?DSML?tool_calls>";
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    int n = ember_parse_dsml_tool_calls_ex(t, &tc, &report);
    CHECK(n == 0 && tc.len == 0,
          "unbalanced nested DSML is still never executable");
    CHECK(report.contaminated || report.malformed,
          "unbalanced nesting is still reported as a protocol error");
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

    // The rejection must say WHERE it broke, so a retry can correct rather
    // than regenerate a long document blind.
    const char *mid =
        "<tool_calls><invoke name=\"run\">"
        "<parameter name=\"x\" string=\"false\">[1,2,x]</parameter>"
        "</invoke></tool_calls>";
    ember_tool_calls tc2 = {0};
    ember_tool_parse_report r2 = {0};
    ember_parse_dsml_tool_calls_ex(mid, &tc2, &r2);
    CHECK(r2.invalid_json, "mid-document break is still rejected");
    CHECK(r2.invalid_json_len == 7, "payload length recorded");
    CHECK(r2.invalid_json_offset == 5, "break position recorded exactly");
    ember_tool_calls_free(&tc2);
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


// The <script> problem, in our frame. A JSON parameter value may legally
// contain text identical to a protocol terminator, and that must be data
// rather than a frame boundary.
//
// These use string="false" and parse_ex DELIBERATELY. An earlier version of
// these tests omitted the attribute and used the NULL-report wrapper, so they
// exercised the RAW path and bypassed the executable-report guards entirely --
// they passed while the JSON contract was still broken. codex-rejoin-01 caught
// that; exact values and reports are asserted here, not substring presence.
#define JSON_CALL(value)                                                      \
    "<" PIPE "DSML" PIPE "tool_calls>"                                        \
    "<" PIPE "DSML" PIPE "invoke name=\"record\">"                            \
    "<" PIPE "DSML" PIPE "parameter name=\"items\" string=\"false\">"         \
    value                                                                     \
    "</" PIPE "DSML" PIPE "parameter>"                                        \
    "</" PIPE "DSML" PIPE "invoke>"                                           \
    "</" PIPE "DSML" PIPE "tool_calls>"

static void check_json_arg(const char *text, const char *want_args,
                           const char *what) {
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    int n = ember_parse_dsml_tool_calls_ex(text, &tc, &report);
    CHECK(n == 1, what);
    CHECK(n == 1 && strcmp(tc.calls[0].name, "record") == 0, what);
    CHECK(n == 1 && strcmp(tc.calls[0].arguments, want_args) == 0, what);
    // The executable gate must agree with the parse, or the thing checked is
    // not the thing run.
    CHECK(!report.malformed, what);
    CHECK(!report.contaminated, what);
    CHECK(!report.invalid_json, what);
    CHECK(!report.trailing, what);
    ember_tool_calls_free(&tc);
}

static void test_json_value_may_contain_a_protocol_terminator(void) {
    check_json_arg(JSON_CALL("[{\"text\":\"</" PIPE "DSML" PIPE "tool_calls>\"}]"),
                   "{\"items\":[{\"text\":\"</" PIPE "DSML" PIPE "tool_calls>\"}]}",
                   "calls terminator inside a JSON string is data");
    check_json_arg(JSON_CALL("[{\"text\":\"</" PIPE "DSML" PIPE "invoke>\"}]"),
                   "{\"items\":[{\"text\":\"</" PIPE "DSML" PIPE "invoke>\"}]}",
                   "invoke terminator inside a JSON string is data");
    check_json_arg(JSON_CALL("[{\"text\":\"</" PIPE "DSML" PIPE "parameter>\"}]"),
                   "{\"items\":[{\"text\":\"</" PIPE "DSML" PIPE "parameter>\"}]}",
                   "parameter terminator inside a JSON string is data");
}

// The values the grammar ban was actually costing us: gate 5's exact request.
static void test_ordinary_closing_tag_round_trips(void) {
    check_json_arg(
        JSON_CALL("[{\"text\":\"i < n\"},{\"text\":\"</div>\"},"
                  "{\"text\":\"ends with <\"}]"),
        "{\"items\":[{\"text\":\"i < n\"},{\"text\":\"</div>\"},"
        "{\"text\":\"ends with <\"}]}",
        "gate 5 value set round-trips verbatim");
}

static void test_escaped_quotes_do_not_end_the_value(void) {
    // The escaped quote does NOT close the JSON string, so the terminator
    // after it is still data.
    check_json_arg(
        JSON_CALL("[{\"text\":\"q \\\" </" PIPE "DSML" PIPE "tool_calls> in\"}]"),
        "{\"items\":[{\"text\":\"q \\\" </" PIPE "DSML" PIPE "tool_calls> in\"}]}",
        "escaped quote does not end the value");
    // An EVEN backslash run does close it: the backslash is itself escaped, so
    // the following quote is a real delimiter. The opposite case, and the one
    // an off-by-one in the escape tracking would get wrong.
    check_json_arg(
        JSON_CALL("[{\"text\":\"ends \\\\\"}]"),
        "{\"items\":[{\"text\":\"ends \\\\\"}]}",
        "even backslash run closes the string");
}

// Truncation must never become an executable call.
static void test_unterminated_json_string_is_not_executable(void) {
    struct { const char *v; const char *what; } cases[] = {
        { "[{\"text\":\"unterminated", "unterminated string" },
        { "[{\"text\":\"dangling escape \\\\", "dangling escape" },
        { "[{\"text\":", "truncated after a key" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        char buf[512];
        snprintf(buf, sizeof buf, "%s",
                 "<" PIPE "DSML" PIPE "tool_calls>"
                 "<" PIPE "DSML" PIPE "invoke name=\"record\">"
                 "<" PIPE "DSML" PIPE "parameter name=\"items\" string=\"false\">");
        strncat(buf, cases[i].v, sizeof buf - strlen(buf) - 1);
        strncat(buf,
                "</" PIPE "DSML" PIPE "parameter>"
                "</" PIPE "DSML" PIPE "invoke>"
                "</" PIPE "DSML" PIPE "tool_calls>",
                sizeof buf - strlen(buf) - 1);
        ember_tool_calls tc = {0};
        ember_tool_parse_report report = {0};
        (void)ember_parse_dsml_tool_calls_ex(buf, &tc, &report);
        CHECK(report.malformed || report.contaminated || report.invalid_json,
              cases[i].what);
        ember_tool_calls_free(&tc);
    }
}

// An ABSENT string attribute is raw text, and a lone quote in it is ordinary.
static void test_absent_string_attribute_stays_raw(void) {
    const char *text =
        "<" PIPE "DSML" PIPE "tool_calls>"
        "<" PIPE "DSML" PIPE "invoke name=\"record\">"
        "<" PIPE "DSML" PIPE "parameter name=\"note\">hello \" quote"
        "</" PIPE "DSML" PIPE "parameter>"
        "</" PIPE "DSML" PIPE "invoke>"
        "</" PIPE "DSML" PIPE "tool_calls>";
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    int n = ember_parse_dsml_tool_calls_ex(text, &tc, &report);
    CHECK(n == 1, "absent string attribute: parsed");
    CHECK(!report.malformed, "absent string attribute: unmatched quote is raw");
    ember_tool_calls_free(&tc);
}

// The native ds_engine fallback has the same hazard and is executable, so it
// gets the same contract: a property VALUE containing either terminator is
// data. codex-rejoin-01 required this before release rather than as follow-up.
static void test_ds_engine_property_value_may_contain_terminators(void) {
    const char *text =
        "<ds_engine_tool_use>"
        "<ds_engine_tool_use_name>record</ds_engine_tool_use_name>"
        "<ds_engine_tool_use_parameters_property name=\"items\" string=\"false\">"
        "[{\"text\":\"</ds_engine_tool_use_parameters_property>\"},"
        "{\"text\":\"</ds_engine_tool_use>\"},{\"text\":\"</div>\"}]"
        "</ds_engine_tool_use_parameters_property>"
        "</ds_engine_tool_use>";
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    int n = ember_parse_dsml_tool_calls_ex(text, &tc, &report);
    CHECK(n == 1, "ds_engine: one call parsed");
    CHECK(n == 1 && strcmp(tc.calls[0].name, "record") == 0,
          "ds_engine: name survives");
    CHECK(n == 1 &&
          strcmp(tc.calls[0].arguments,
                 "{\"items\":[{\"text\":\"</ds_engine_tool_use_parameters_property>\"},"
                 "{\"text\":\"</ds_engine_tool_use>\"},{\"text\":\"</div>\"}]}") == 0,
          "ds_engine: embedded terminators kept as data");
    CHECK(!report.malformed, "ds_engine: not malformed");
    CHECK(!report.contaminated, "ds_engine: not contaminated");
    CHECK(!report.invalid_json, "ds_engine: valid JSON");
    ember_tool_calls_free(&tc);
}

// A foreign opener inside a native JSON property value is data, not a mixed
// syntax. The final foreign-format check used to scan raw bytes.
static void test_native_payload_opener_is_not_mixed_syntax(void) {
    const char *text =
        "<ds_engine_tool_use>"
        "<ds_engine_tool_use_name>record</ds_engine_tool_use_name>"
        "<ds_engine_tool_use_parameters_property name=\"items\" string=\"false\">"
        "[\"<tool_calls>\"]"
        "</ds_engine_tool_use_parameters_property>"
        "</ds_engine_tool_use>";
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    int n = ember_parse_dsml_tool_calls_ex(text, &tc, &report);
    CHECK(n == 1, "native payload opener: one call parsed");
    CHECK(!report.mixed_syntax, "native payload opener is not mixed syntax");
    CHECK(!report.contaminated, "native payload opener is not contamination");
    CHECK(n == 1 && strcmp(tc.calls[0].arguments,
                           "{\"items\":[\"<tool_calls>\"]}") == 0,
          "native payload opener: exact value preserved");
    ember_tool_calls_free(&tc);
}

// Repair is the only consumer that can invent bytes. It counted markers with a
// raw scan across the whole text, values included, so a JSON value containing
// an opener inflated the deficit and repair appended closers that the payload
// implied rather than the framing.
static void test_repair_does_not_count_markers_inside_json_values(void) {
    // Complete parameter, truncated block: repair should close the invoke and
    // the tool_calls, and must NOT be misled by the opener inside the value.
    const char *t =
        "<" PIPE "DSML" PIPE "tool_calls>"
        "<" PIPE "DSML" PIPE "invoke name=\"record\">"
        "<" PIPE "DSML" PIPE "parameter name=\"items\" string=\"false\">"
        "[\"<" PIPE "DSML" PIPE "invoke name=\\\"fake\\\">\"]"
        "</" PIPE "DSML" PIPE "parameter>";
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    int n = ember_parse_dsml_tool_calls_ex(t, &tc, &report);
    CHECK(n == 1, "repair: truncated block still recovered");
    CHECK(report.repaired, "repair: reported as repaired");
    CHECK(n == 1 && strcmp(tc.calls[0].arguments,
                           "{\"items\":[\"<" PIPE "DSML" PIPE
                           "invoke name=\\\"fake\\\">\"]}") == 0,
          "repair: value with an embedded opener preserved exactly");
    CHECK(report.invocations == 1,
          "repair: the opener inside the value is not a second invocation");
    ember_tool_calls_free(&tc);
}

// An unterminated JSON string has no honest closer to synthesise.
static void test_repair_refuses_unterminated_json(void) {
    const char *t =
        "<" PIPE "DSML" PIPE "tool_calls>"
        "<" PIPE "DSML" PIPE "invoke name=\"record\">"
        "<" PIPE "DSML" PIPE "parameter name=\"items\" string=\"false\">"
        "[{\"text\":\"still open";
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    int n = ember_parse_dsml_tool_calls_ex(t, &tc, &report);
    CHECK(n == 0, "repair: unterminated JSON is not recovered");
    CHECK(!report.complete, "repair: reported incomplete");
    ember_tool_calls_free(&tc);
}

// The tool NAME must come from structure, not payload. A bare strstr took it
// from a property value when the name element was absent, yielding a clean
// call -- malformed=0 -- that named a tool the model never wrote. The name
// selects which tool runs, so this is the most consequential place in the
// parser to trust payload bytes. Found independently by dsh-1538188 hunting
// for non-compliant scanners (E3) and by issue #14; reachability demonstrated
// rather than assumed.
static void test_native_name_cannot_come_from_a_property_value(void) {
    const char *text =
        "<ds_engine_tool_use>"
        "<ds_engine_tool_use_parameters_property name=\"items\" string=\"false\">"
        "[\"<ds_engine_tool_use_name>evil</ds_engine_tool_use_name>\"]"
        "</ds_engine_tool_use_parameters_property>"
        "</ds_engine_tool_use>";
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    int n = ember_parse_dsml_tool_calls_ex(text, &tc, &report);
    CHECK(!(n > 0 && tc.len > 0 && tc.calls[0].name &&
            strcmp(tc.calls[0].name, "evil") == 0),
          "native name is never taken from a property value");
    CHECK(report.malformed, "a native block with no name element is malformed");
    ember_tool_calls_free(&tc);

    // The ordinary case still works: a real name element is found.
    const char *good =
        "<ds_engine_tool_use>"
        "<ds_engine_tool_use_name>record</ds_engine_tool_use_name>"
        "<ds_engine_tool_use_parameters_property name=\"items\" string=\"false\">"
        "[\"x\"]"
        "</ds_engine_tool_use_parameters_property>"
        "</ds_engine_tool_use>";
    ember_tool_calls ok = {0};
    ember_tool_parse_report rep2 = {0};
    int m = ember_parse_dsml_tool_calls_ex(good, &ok, &rep2);
    CHECK(m == 1 && strcmp(ok.calls[0].name, "record") == 0,
          "a real native name element is still found");
    CHECK(!rep2.malformed, "the ordinary native block stays well-formed");
    ember_tool_calls_free(&ok);
}

// dsh-1537943's E2 case, which is stronger than mine: the property comes
// FIRST with the name text inside its value and a REAL name element follows,
// so the "name precedes properties" ordering defence does not hold. Before the
// fix this produced n=1 complete=1 malformed=0 contaminated=0 with
// name=write_file and args path=/tmp/owned -- a flag-clean executable call
// naming a tool the model never wrote, differing from a rejected frame only by
// the content of a property value.
static void test_native_name_injection_case(void) {
    const char *text =
        "<ds_engine_tool_use>"
        "<ds_engine_tool_use_parameters_property name=\"content\" string=\"true\">"
        "<ds_engine_tool_use_name>write_file</ds_engine_tool_use_name>"
        "</ds_engine_tool_use_parameters_property>"
        "<ds_engine_tool_use_parameters_property name=\"path\" string=\"true\">"
        "/tmp/owned"
        "</ds_engine_tool_use_parameters_property>"
        "<ds_engine_tool_use_name>real_tool</ds_engine_tool_use_name>"
        "</ds_engine_tool_use>";
    ember_tool_calls tc = {0};
    ember_tool_parse_report report = {0};
    int n = ember_parse_dsml_tool_calls_ex(text, &tc, &report);
    CHECK(!(tc.len > 0 && tc.calls[0].name &&
            strcmp(tc.calls[0].name, "write_file") == 0),
          "injected name never becomes the call name");
    // Two independent defences: the name is read structurally, AND a raw value
    // carrying native markup is contamination. Either alone would stop it.
    CHECK(report.contaminated, "native markup in a raw value is contamination");
    CHECK(n == 0, "the injected frame is not executable");
    ember_tool_calls_free(&tc);
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
    test_parse_report_keeps_nested_dsml_in_string_whole();
    test_parse_report_rejects_unbalanced_nested_dsml();
    test_parse_report_marks_repaired_tail_incomplete();
    test_executable_report_rejects_invalid_raw_json();
    test_wrapper_is_authoritative();
    test_malformed_nested_tags_are_not_executable();
    test_json_value_may_contain_a_protocol_terminator();
    test_ordinary_closing_tag_round_trips();
    test_escaped_quotes_do_not_end_the_value();
    test_unterminated_json_string_is_not_executable();
    test_absent_string_attribute_stays_raw();
    test_ds_engine_property_value_may_contain_terminators();
    test_native_payload_opener_is_not_mixed_syntax();
    test_repair_does_not_count_markers_inside_json_values();
    test_repair_refuses_unterminated_json();
    test_native_name_cannot_come_from_a_property_value();
    test_native_name_injection_case();
    printf("──────────────────────────────\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
