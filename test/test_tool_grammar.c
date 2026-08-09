// The grammar's job is to make production's failure shapes unrepresentable.
// These checks assert on the emitted EBNF text: that a tool with a required
// property cannot reach its closing tag without a <parameter>, and that the
// empty invoke survives ONLY for a tool that genuinely declares no properties.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/model/tool_grammar.h"

static int pass, fail;
#define CHECK(x) do { if (x) pass++; else { fail++; fprintf(stderr, \
    "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); } } while (0)

#define PIPE       "\xef\xbd\x9c"
#define DSML_OPEN  "<" PIPE "DSML" PIPE
#define DSML_CLOSE "</" PIPE "DSML" PIPE

static bool has(const char *hay, const char *needle) {
    return hay && strstr(hay, needle) != NULL;
}

// The exact production failure: terminal invoked with no <parameter> at all.
static void test_required_property_is_mandatory(void) {
    const char *tools =
        "[{\"type\":\"function\",\"function\":{\"name\":\"terminal\","
        "\"parameters\":{\"type\":\"object\",\"required\":[\"command\"],"
        "\"properties\":{\"command\":{\"type\":\"string\"}}}}}]";
    char *g = ember_tool_grammar_build(tools, true);
    CHECK(g != NULL);
    // The tool's production must reference its required-parameter rule, so the
    // closing invoke tag is unreachable until a <parameter> has been emitted.
    CHECK(has(g, "t0_terminal ::= "));
    CHECK(has(g, "t0_terminal_r0"));
    // ...and that rule must be mandatory, never optional.
    CHECK(!has(g, "t0_terminal_r0?"));
    // The empty body form must NOT appear for a tool that requires a property.
    CHECK(!has(g, "t0_terminal ::= \"" DSML_OPEN "invoke name=\\\"terminal\\\">\" ws \"\" ws"));
    CHECK(has(g, "string=\\\"true\\\">\" strval "));
    free(g);
}

// browser_back declares no properties: the one case where <invoke></invoke> is
// legitimate, and the only case the grammar may allow it.
static void test_parameterless_tool_allows_empty_body(void) {
    const char *tools =
        "[{\"type\":\"function\",\"function\":{\"name\":\"browser_back\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";
    char *g = ember_tool_grammar_build(tools, true);
    CHECK(g != NULL);
    CHECK(has(g, "ws \"\" ws"));
    free(g);
}

static void test_optional_properties_are_skippable(void) {
    const char *tools =
        "[{\"type\":\"function\",\"function\":{\"name\":\"grep\","
        "\"parameters\":{\"type\":\"object\",\"required\":[\"pattern\"],"
        "\"properties\":{\"pattern\":{\"type\":\"string\"},"
        "\"limit\":{\"type\":\"integer\"}}}}}]";
    char *g = ember_tool_grammar_build(tools, true);
    CHECK(g != NULL);
    CHECK(has(g, "t0_grep_r0"));      // pattern, required
    CHECK(has(g, "t0_grep_o1?"));     // limit, optional and skippable
    // Required must be ordered before optional in the body.
    const char *body = g ? strstr(g, "t0_grep ::= ") : NULL;
    const char *r = body ? strstr(body, "t0_grep_r0") : NULL;
    const char *o = body ? strstr(body, "t0_grep_o1") : NULL;
    CHECK(r && o && r < o);
    // A non-string property takes the JSON form.
    CHECK(has(g, "string=\\\"false\\\">\" jsonval "));
    free(g);
}

// A property with no declared type must stay unconstrained between the two
// forms; guessing one would reject valid output.
static void test_untyped_property_allows_both_forms(void) {
    const char *tools =
        "[{\"type\":\"function\",\"function\":{\"name\":\"anytool\","
        "\"parameters\":{\"type\":\"object\",\"required\":[\"v\"],"
        "\"properties\":{\"v\":{}}}}}]";
    char *g = ember_tool_grammar_build(tools, true);
    CHECK(g != NULL);
    CHECK(has(g, "string=\\\"true\\\">\" strval "));
    CHECK(has(g, "string=\\\"false\\\">\" jsonval "));
    CHECK(has(g, " | "));
    free(g);
}

static void test_parallel_calls_gate(void) {
    const char *tools =
        "[{\"type\":\"function\",\"function\":{\"name\":\"a\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";
    char *par = ember_tool_grammar_build(tools, true);
    char *one = ember_tool_grammar_build(tools, false);
    CHECK(has(par, "invokes ::= invoke (ws invoke)*"));
    CHECK(has(one, "invokes ::= invoke\n"));
    CHECK(!has(one, "(ws invoke)*"));
    free(par);
    free(one);
}

// Sanitizing tool names into rule identifiers must not collide two distinct
// tools onto one rule.
static void test_sanitized_names_stay_unique(void) {
    const char *tools =
        "[{\"function\":{\"name\":\"a.b\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{}}}},"
        "{\"function\":{\"name\":\"a-b\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{}}}}]";
    char *g = ember_tool_grammar_build(tools, true);
    CHECK(g != NULL);
    CHECK(has(g, "t0_a_b ::= "));
    CHECK(has(g, "t1_a_b ::= "));
    CHECK(has(g, "invoke ::= t0_a_b | t1_a_b"));
    free(g);
}

// No usable tool means decode unconstrained, not an error.
static void test_no_tools_returns_null(void) {
    CHECK(ember_tool_grammar_build(NULL, true) == NULL);
    CHECK(ember_tool_grammar_build("", true) == NULL);
    CHECK(ember_tool_grammar_build("[]", true) == NULL);
    CHECK(ember_tool_grammar_build("not json", true) == NULL);
    CHECK(ember_tool_grammar_build("{\"not\":\"an array\"}", true) == NULL);
    // An entry with no name is skipped rather than emitting a broken rule.
    CHECK(ember_tool_grammar_build("[{\"function\":{}}]", true) == NULL);
}

// invoke must not nest: the block rule admits only whole invokes in sequence.
static void test_invoke_does_not_nest(void) {
    const char *tools =
        "[{\"function\":{\"name\":\"t\",\"parameters\":{\"type\":\"object\","
        "\"required\":[\"x\"],\"properties\":{\"x\":{\"type\":\"string\"}}}}}]";
    char *g = ember_tool_grammar_build(tools, true);
    CHECK(g != NULL);
    // The only reference to `invoke` is from the block rule; a tool rule never
    // recurses into one.
    const char *tool_rule = g ? strstr(g, "t0_t ::= ") : NULL;
    CHECK(tool_rule && !has(tool_rule, "invokes"));
    CHECK(has(g, "root ::= \"" DSML_OPEN "tool_calls>\""));
    CHECK(has(g, "\"" DSML_CLOSE "invoke>\""));
    free(g);
}

// The production shape: `patch` requires only `mode`, but requires path +
// old_string + new_string once mode="replace". The model emitted {mode,
// new_string} 17 times before a guardrail stopped it.
static const char *PATCH_TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"patch\",\"parameters\":{"
    "\"type\":\"object\",\"required\":[\"mode\"],\"properties\":{"
    "\"mode\":{\"type\":\"string\",\"enum\":[\"replace\",\"patch\"]},"
    "\"path\":{\"type\":\"string\"},"
    "\"old_string\":{\"type\":\"string\"},"
    "\"new_string\":{\"type\":\"string\"},"
    "\"replace_all\":{\"type\":\"boolean\"},"
    "\"patch\":{\"type\":\"string\"}},"
    "\"allOf\":["
    "{\"if\":{\"properties\":{\"mode\":{\"const\":\"replace\"}}},"
    " \"then\":{\"required\":[\"path\",\"old_string\",\"new_string\"]}},"
    "{\"if\":{\"properties\":{\"mode\":{\"const\":\"patch\"}}},"
    " \"then\":{\"required\":[\"patch\"]}}]}}}]";

static void test_discriminated_branches(void) {
    char *g = ember_tool_grammar_build(PATCH_TOOLS, true);
    CHECK(g != NULL);
    // One alternative per branch, not one rule for the tool.
    CHECK(has(g, "t0_patch_b0 ::= "));
    CHECK(has(g, "t1_patch_b1 ::= "));
    CHECK(has(g, "invoke ::= t0_patch_b0 | t1_patch_b1"));
    // The discriminator is pinned to its literal in each branch, so choosing a
    // branch is what commits the model to that branch's obligations.
    CHECK(has(g, "name=\\\"mode\\\" string=\\\"true\\\">replace"));
    CHECK(has(g, "name=\\\"mode\\\" string=\\\"true\\\">patch"));
    // Branch 0 makes path/old_string/new_string mandatory (no '?').
    const char *b0 = g ? strstr(g, "t0_patch_b0 ::= ") : NULL;
    CHECK(b0 && has(b0, "t0_patch_b0_r1"));
    CHECK(!has(g, "t0_patch_b0_r1?"));
    // replace_all stays optional in the replace branch.
    CHECK(has(g, "t0_patch_b0_o"));
    free(g);
}

// Without conditionals the tool keeps its previous unconditional shape.
static void test_no_conditionals_unchanged(void) {
    const char *tools =
        "[{\"function\":{\"name\":\"t\",\"parameters\":{\"type\":\"object\","
        "\"required\":[\"x\"],\"properties\":{\"x\":{\"type\":\"string\"}}}}}]";
    char *g = ember_tool_grammar_build(tools, true);
    CHECK(g != NULL);
    CHECK(has(g, "t0_t ::= "));
    CHECK(!has(g, "_b0"));
    free(g);
}

// A shape the generator does not understand must leave the tool
// unconditional rather than guessing -- over-constraining rejects valid calls.
static void test_unrecognised_conditional_is_ignored(void) {
    const char *tools =
        "[{\"function\":{\"name\":\"t\",\"parameters\":{\"type\":\"object\","
        "\"required\":[\"mode\"],\"properties\":{"
        "\"mode\":{\"type\":\"string\"},\"a\":{\"type\":\"string\"}},"
        // if with two pinned properties: not the discriminated form
        "\"allOf\":[{\"if\":{\"properties\":{\"mode\":{\"const\":\"x\"},"
        "\"a\":{\"const\":\"y\"}}},\"then\":{\"required\":[\"a\"]}}]}}}]";
    char *g = ember_tool_grammar_build(tools, true);
    CHECK(g != NULL);
    CHECK(has(g, "t0_t ::= "));
    CHECK(!has(g, "_b0"));
    free(g);
}

// Branches keyed off different properties are not a clean partition -- a call
// could satisfy two at once -- so the tool stays unconditional.
static void test_mixed_discriminators_ignored(void) {
    const char *tools =
        "[{\"function\":{\"name\":\"t\",\"parameters\":{\"type\":\"object\","
        "\"required\":[\"m\"],\"properties\":{"
        "\"m\":{\"type\":\"string\"},\"n\":{\"type\":\"string\"},"
        "\"a\":{\"type\":\"string\"}},"
        "\"allOf\":[{\"if\":{\"properties\":{\"m\":{\"const\":\"x\"}}},"
        "  \"then\":{\"required\":[\"a\"]}},"
        " {\"if\":{\"properties\":{\"n\":{\"const\":\"y\"}}},"
        "  \"then\":{\"required\":[\"a\"]}}]}}}]";
    char *g = ember_tool_grammar_build(tools, true);
    CHECK(g != NULL);
    CHECK(!has(g, "_b0"));
    free(g);
}

// A discriminator value with NO conditional is still a legal call. Emitting
// only the branch values would make it unrepresentable -- skill_manage's
// "delete", cronjob's and process's "list" are all real examples.
static void test_enum_values_without_branches_survive(void) {
    const char *tools =
        "[{\"function\":{\"name\":\"skill_manage\",\"parameters\":{"
        "\"type\":\"object\",\"required\":[\"action\",\"name\"],"
        "\"properties\":{"
        "\"action\":{\"type\":\"string\","
        "  \"enum\":[\"create\",\"patch\",\"delete\"]},"
        "\"name\":{\"type\":\"string\"},"
        "\"content\":{\"type\":\"string\"},"
        "\"old_string\":{\"type\":\"string\"},"
        "\"new_string\":{\"type\":\"string\"}},"
        "\"allOf\":["
        "{\"if\":{\"properties\":{\"action\":{\"const\":\"create\"}}},"
        " \"then\":{\"required\":[\"content\"]}},"
        "{\"if\":{\"properties\":{\"action\":{\"const\":\"patch\"}}},"
        " \"then\":{\"required\":[\"old_string\",\"new_string\"]}}]}}}]";
    char *g = ember_tool_grammar_build(tools, true);
    CHECK(g != NULL);
    // Three enum values -> three alternatives, including the un-branched one.
    CHECK(has(g, "t0_skill_manage_b0 ::= "));
    CHECK(has(g, "t1_skill_manage_b1 ::= "));
    CHECK(has(g, "t2_skill_manage_b2 ::= "));
    CHECK(has(g, "string=\\\"true\\\">create"));
    CHECK(has(g, "string=\\\"true\\\">patch"));
    CHECK(has(g, "string=\\\"true\\\">delete"));   // <- would be lost before
    free(g);
}

// With conditionals but no enum on the discriminator we cannot enumerate the
// alternatives, so the tool keeps its unconditional shape.
static void test_conditionals_without_enum_stay_unconditional(void) {
    const char *tools =
        "[{\"function\":{\"name\":\"t\",\"parameters\":{"
        "\"type\":\"object\",\"required\":[\"mode\"],"
        "\"properties\":{\"mode\":{\"type\":\"string\"},"
        "\"a\":{\"type\":\"string\"}},"
        "\"allOf\":[{\"if\":{\"properties\":{\"mode\":{\"const\":\"x\"}}},"
        " \"then\":{\"required\":[\"a\"]}}]}}}]";
    char *g = ember_tool_grammar_build(tools, true);
    CHECK(g != NULL);
    CHECK(has(g, "t0_t ::= "));
    CHECK(!has(g, "_b0"));
    free(g);
}

int main(void) {
    test_enum_values_without_branches_survive();
    test_conditionals_without_enum_stay_unconditional();
    test_discriminated_branches();
    test_no_conditionals_unchanged();
    test_unrecognised_conditional_is_ignored();
    test_mixed_discriminators_ignored();
    test_required_property_is_mandatory();
    test_parameterless_tool_allows_empty_body();
    test_optional_properties_are_skippable();
    test_untyped_property_allows_both_forms();
    test_parallel_calls_gate();
    test_sanitized_names_stay_unique();
    test_no_tools_returns_null();
    test_invoke_does_not_nest();
    printf("test_tool_grammar: %d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
