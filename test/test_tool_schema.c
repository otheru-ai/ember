#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/common/json.h"
#include "../src/model/tool_schema.h"

static int pass, fail;
#define CHECK(x) do { if (x) pass++; else { fail++; fprintf(stderr, \
    "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); } } while (0)

static bool validates(const char *instance_text, const char *schema_text,
                      bool strict, char *err, size_t err_cap) {
    ember_json *instance = ember_json_parse(instance_text);
    ember_json *schema = ember_json_parse(schema_text);
    bool ok = instance && schema && ember_tool_schema_validate(
        instance, schema, schema, strict, err, err_cap);
    ember_json_free(instance);
    ember_json_free(schema);
    return ok;
}

static void test_large_number_equality(void) {
    char err[256];
    CHECK(validates("9007199254740992", "{\"const\":9007199254740992}",
                    true, err, sizeof(err)));
    CHECK(!validates("9007199254740993", "{\"const\":9007199254740992}",
                     true, err, sizeof(err)));
    CHECK(validates("1.00e2", "{\"enum\":[100]}",
                    true, err, sizeof(err)));
    CHECK(!validates("9007199254740993.5", "{\"type\":\"integer\"}",
                     true, err, sizeof(err)));
}

static void test_recursive_constraints(void) {
    const char *schema =
        "{\"type\":\"object\",\"additionalProperties\":false,"
        "\"required\":[\"path\",\"options\"],\"properties\":{"
        "\"path\":{\"type\":\"string\",\"minLength\":2,"
          "\"pattern\":\"^/\"},"
        "\"options\":{\"type\":\"object\",\"required\":[\"mode\"],"
          "\"properties\":{\"mode\":{\"enum\":[\"safe\",\"fast\"]},"
          "\"retries\":{\"type\":\"integer\",\"minimum\":0,"
          "\"maximum\":3}},\"additionalProperties\":false},"
        "\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},"
          "\"uniqueItems\":true}}}";
    char err[256];
    CHECK(validates("{\"path\":\"/tmp\",\"options\":{\"mode\":\"safe\","
                    "\"retries\":2},\"tags\":[\"a\",\"b\"]}",
                    schema, true, err, sizeof(err)));
    CHECK(!validates("{\"path\":7,\"options\":{\"mode\":\"safe\"}}",
                     schema, true, err, sizeof(err)) && strstr(err, "$.path"));
    CHECK(!validates("{\"path\":\"/x\",\"options\":{\"mode\":\"other\"}}",
                     schema, true, err, sizeof(err)) && strstr(err, "enum"));
    CHECK(!validates("{\"path\":\"/x\",\"options\":{\"mode\":\"safe\","
                     "\"extra\":1}}", schema, true, err, sizeof(err)) &&
          strstr(err, "additional property"));
    CHECK(!validates("{\"path\":\"/x\",\"options\":{\"mode\":\"safe\"},"
                     "\"tags\":[\"a\",\"a\"]}", schema, true,
                     err, sizeof(err)) && strstr(err, "not unique"));
}

static void test_refs_and_combinators(void) {
    const char *schema =
        "{\"$defs\":{\"id\":{\"type\":\"integer\",\"minimum\":1}},"
        "\"type\":\"object\",\"properties\":{"
        "\"id\":{\"$ref\":\"#/$defs/id\"},"
        "\"value\":{\"oneOf\":[{\"type\":\"string\"},"
          "{\"type\":\"number\"}]}}}";
    char err[256];
    CHECK(validates("{\"id\":2,\"value\":\"x\"}", schema, true,
                    err, sizeof(err)));
    CHECK(!validates("{\"id\":0,\"value\":true}", schema, true,
                     err, sizeof(err)));
    CHECK(validates("3", "{\"allOf\":[{\"type\":\"number\"},"
                    "{\"minimum\":1}],\"not\":{\"const\":4}}",
                    true, err, sizeof(err)));
    CHECK(!validates("4", "{\"not\":{\"const\":4}}", true,
                     err, sizeof(err)));
}

static void test_strict_and_duplicates(void) {
    char err[256];
    CHECK(!validates("{}", "{\"type\":\"object\",\"mystery\":true}",
                     true, err, sizeof(err)) && strstr(err, "unsupported"));
    CHECK(validates("{}", "{\"type\":\"object\",\"mystery\":true}",
                    false, err, sizeof(err)));
    CHECK(!validates("{\"x\":1,\"x\":2}", "{\"type\":\"object\"}",
                     true, err, sizeof(err)) && strstr(err, "duplicate"));
    CHECK(validates("6", "{\"type\":\"integer\",\"multipleOf\":3}",
                    true, err, sizeof(err)));
    CHECK(!validates("7", "{\"type\":\"integer\",\"multipleOf\":3}",
                     true, err, sizeof(err)));
    CHECK(!validates("{}", "{\"type\":\"object\",\"properties\":{"
                     "\"optional\":{\"type\":\"string\",\"mystery\":1}}}",
                     true, err, sizeof(err)) && strstr(err, "unsupported"));
    CHECK(!validates("\"x@example.com\"", "{\"type\":\"string\","
                     "\"format\":\"email\"}", true, err, sizeof(err)) &&
          strstr(err, "unsupported"));
    CHECK(!validates("{}", "{\"not\":null}", false,
                     err, sizeof(err)) && strstr(err, "object or boolean"));
    CHECK(!validates("[]", "{\"type\":\"array\",\"uniqueItems\":1}",
                     false, err, sizeof(err)) && strstr(err, "boolean"));
}

// ── regression coverage for the rest of the validation vocabulary ────────
//
// The validator is the only thing standing between a generated tool call and a
// harness that will execute it, so every keyword it claims to support needs a
// test that proves it both accepts a conforming instance and REJECTS a
// non-conforming one. A keyword that silently accepts everything is worse than
// an unsupported one: strict mode would reject the unsupported keyword, but a
// broken implementation of a known keyword weakens the contract invisibly.

static void test_contains_cardinality(void) {
    char err[256];
    const char *ints = "{\"type\":\"array\",\"contains\":{\"type\":\"integer\"}}";
    CHECK(validates("[1,\"a\"]", ints, true, err, sizeof(err)));
    CHECK(!validates("[\"a\",\"b\"]", ints, true, err, sizeof(err)) &&
          strstr(err, "contains cardinality"));

    // minContains 0 makes `contains` an assertion about the upper bound only.
    CHECK(validates("[\"a\"]",
                    "{\"type\":\"array\",\"contains\":{\"type\":\"integer\"},"
                    "\"minContains\":0}", true, err, sizeof(err)));
    const char *two =
        "{\"type\":\"array\",\"contains\":{\"const\":\"x\"},\"minContains\":2}";
    CHECK(validates("[\"x\",\"x\",\"y\"]", two, true, err, sizeof(err)));
    CHECK(!validates("[\"x\",\"y\"]", two, true, err, sizeof(err)) &&
          strstr(err, "contains cardinality"));

    const char *at_most_one =
        "{\"type\":\"array\",\"contains\":{\"type\":\"integer\"},"
        "\"maxContains\":1}";
    CHECK(validates("[1,\"a\"]", at_most_one, true, err, sizeof(err)));
    CHECK(!validates("[1,2]", at_most_one, true, err, sizeof(err)) &&
          strstr(err, "contains cardinality"));

    // Preflight rejects a non-integer bound before any instance is examined.
    CHECK(!validates("[]", "{\"type\":\"array\",\"contains\":true,"
                     "\"minContains\":1.5}", true, err, sizeof(err)) &&
          strstr(err, "minContains"));
}

static void test_structural_equality(void) {
    char err[256];
    // const/enum equality must recurse through arrays and objects, not just
    // compare scalars. A shallow compare would accept any array of the right
    // length as equal to the const.
    CHECK(validates("[1,[2,3]]", "{\"const\":[1,[2,3]]}", true,
                    err, sizeof(err)));
    CHECK(!validates("[1,[2,4]]", "{\"const\":[1,[2,3]]}", true,
                     err, sizeof(err)) && strstr(err, "const"));
    CHECK(!validates("[1]", "{\"const\":[1,[2,3]]}", true, err, sizeof(err)));

    // Object equality is key-order insensitive but length sensitive.
    const char *obj = "{\"const\":{\"a\":1,\"b\":{\"c\":2}}}";
    CHECK(validates("{\"b\":{\"c\":2},\"a\":1}", obj, true, err, sizeof(err)));
    CHECK(!validates("{\"a\":1}", obj, true, err, sizeof(err)));
    CHECK(!validates("{\"a\":1,\"b\":{\"c\":3}}", obj, true, err, sizeof(err)));

    // enum over non-scalar values goes through the same comparison.
    const char *enumeration = "{\"enum\":[[1,2],{\"k\":\"v\"}]}";
    CHECK(validates("[1,2]", enumeration, true, err, sizeof(err)));
    CHECK(validates("{\"k\":\"v\"}", enumeration, true, err, sizeof(err)));
    CHECK(!validates("[2,1]", enumeration, true, err, sizeof(err)) &&
          strstr(err, "enum"));

    // uniqueItems is the third consumer of structural equality.
    const char *uniq = "{\"type\":\"array\",\"uniqueItems\":true}";
    CHECK(!validates("[[1,2],[1,2]]", uniq, true, err, sizeof(err)) &&
          strstr(err, "not unique"));
    CHECK(validates("[[1,2],[1,3]]", uniq, true, err, sizeof(err)));
    CHECK(!validates("[{\"a\":1},{\"a\":1}]", uniq, true, err, sizeof(err)) &&
          strstr(err, "not unique"));
    CHECK(validates("[{\"a\":1},{\"a\":2}]", uniq, true, err, sizeof(err)));
}

static void test_type_arrays(void) {
    char err[256];
    const char *nullable = "{\"type\":[\"string\",\"null\"]}";
    CHECK(validates("\"x\"", nullable, true, err, sizeof(err)));
    CHECK(validates("null", nullable, true, err, sizeof(err)));
    CHECK(!validates("1", nullable, true, err, sizeof(err)) &&
          strstr(err, "schema type"));

    // Preflight rejects malformed type arrays rather than treating an unknown
    // name as "matches nothing", which would silently reject every instance.
    CHECK(!validates("\"x\"", "{\"type\":[\"string\",\"bogus\"]}", true,
                     err, sizeof(err)) && strstr(err, "type array is invalid"));
    CHECK(!validates("\"x\"", "{\"type\":[\"string\",\"string\"]}", true,
                     err, sizeof(err)) && strstr(err, "duplicates"));
    CHECK(!validates("\"x\"", "{\"type\":[1]}", true, err, sizeof(err)) &&
          strstr(err, "type array is invalid"));
    CHECK(!validates("\"x\"", "{\"type\":[]}", true, err, sizeof(err)) &&
          strstr(err, "non-empty array"));
}

static void test_prefix_items(void) {
    char err[256];
    const char *tuple =
        "{\"type\":\"array\",\"prefixItems\":[{\"type\":\"string\"},"
        "{\"type\":\"integer\"}],\"items\":{\"type\":\"boolean\"}}";
    CHECK(validates("[\"a\",1,true,false]", tuple, true, err, sizeof(err)));
    CHECK(!validates("[\"a\",\"b\"]", tuple, true, err, sizeof(err)) &&
          strstr(err, "$[1]"));
    // items applies only past the prefix, so a bad tail element is reported at
    // its own index rather than against the tuple schema.
    CHECK(!validates("[\"a\",1,\"c\"]", tuple, true, err, sizeof(err)) &&
          strstr(err, "$[2]"));
    // A short array checks only the prefix entries it actually has.
    CHECK(validates("[\"a\"]", tuple, true, err, sizeof(err)));
    // prefixItems may be empty (unlike allOf/anyOf/oneOf).
    CHECK(validates("[1]", "{\"type\":\"array\",\"prefixItems\":[]}", true,
                    err, sizeof(err)));
    CHECK(!validates("[]", "{\"type\":\"array\",\"prefixItems\":{}}", true,
                     err, sizeof(err)) && strstr(err, "schema array"));
}

static void test_pattern_and_property_names(void) {
    char err[256];
    const char *patterned =
        "{\"type\":\"object\",\"patternProperties\":{"
        "\"^s_\":{\"type\":\"string\"},\"^n_\":{\"type\":\"number\"}},"
        "\"additionalProperties\":false}";
    CHECK(validates("{\"s_a\":\"x\",\"n_b\":2}", patterned, true,
                    err, sizeof(err)));
    CHECK(!validates("{\"s_a\":1}", patterned, true, err, sizeof(err)) &&
          strstr(err, "$.s_a"));
    // A pattern match is what makes a property "covered"; an unmatched key
    // still falls through to additionalProperties.
    CHECK(!validates("{\"other\":1}", patterned, true, err, sizeof(err)) &&
          strstr(err, "additional property"));

    // An uncompilable pattern key must fail closed, not match nothing.
    CHECK(!validates("{}", "{\"type\":\"object\",\"patternProperties\":{"
                     "\"[\":{\"type\":\"string\"}}}", true, err, sizeof(err)) &&
          strstr(err, "patternProperties key is invalid"));

    const char *names =
        "{\"type\":\"object\",\"propertyNames\":{\"pattern\":\"^[a-z]+$\"}}";
    CHECK(validates("{\"abc\":1,\"def\":2}", names, true, err, sizeof(err)));
    CHECK(!validates("{\"A1\":1}", names, true, err, sizeof(err)) &&
          strstr(err, "pattern"));

    CHECK(validates("\"12345\"", "{\"pattern\":\"^\\\\d+$\"}",
                    true, err, sizeof(err)));
    CHECK(!validates("\"dd\"", "{\"pattern\":\"^\\\\d+$\"}",
                     true, err, sizeof(err)));
    CHECK(!validates("\"x\"", "{\"pattern\":\"(?=x)x\"}",
                     true, err, sizeof(err)) && strstr(err, "invalid"));
    CHECK(validates("\"abc_123\"", "{\"pattern\":\"^\\\\w+$\"}",
                    true, err, sizeof(err)));
    CHECK(validates("\"a b\"", "{\"pattern\":\"^a\\\\sb$\"}",
                    true, err, sizeof(err)));
    CHECK(validates("\"a.b\"", "{\"pattern\":\"^a\\\\.b$\"}",
                    true, err, sizeof(err)));
    CHECK(!validates("\"line\\nbreak\"", "{\"pattern\":\"^.*$\"}",
                     true, err, sizeof(err)) && strstr(err, "pattern"));
    CHECK(!validates("\"1\"", "{\"pattern\":\"[[:digit:]]\"}",
                     true, err, sizeof(err)) && strstr(err, "invalid"));

    // additionalProperties as a schema (rather than false) constrains the
    // uncovered properties instead of banning them.
    const char *typed_extras =
        "{\"type\":\"object\",\"properties\":{\"known\":{\"type\":\"string\"}},"
        "\"additionalProperties\":{\"type\":\"integer\"}}";
    CHECK(validates("{\"known\":\"x\",\"extra\":1}", typed_extras, true,
                    err, sizeof(err)));
    CHECK(!validates("{\"known\":\"x\",\"extra\":\"y\"}", typed_extras, true,
                     err, sizeof(err)) && strstr(err, "$.extra"));
}

static void test_dependent_required(void) {
    char err[256];
    const char *card =
        "{\"type\":\"object\",\"dependentRequired\":{\"card\":[\"cvv\",\"exp\"]}}";
    CHECK(validates("{\"card\":\"x\",\"cvv\":\"1\",\"exp\":\"2\"}", card, true,
                    err, sizeof(err)));
    // No trigger property -> the dependency is inert.
    CHECK(validates("{\"unrelated\":1}", card, true, err, sizeof(err)));
    CHECK(!validates("{\"card\":\"x\",\"cvv\":\"1\"}", card, true,
                     err, sizeof(err)) && strstr(err, "dependent property exp"));

    // Preflight rejects malformed dependency declarations.
    CHECK(!validates("{}", "{\"dependentRequired\":[]}", true,
                     err, sizeof(err)) && strstr(err, "must be an object"));
    CHECK(!validates("{}", "{\"dependentRequired\":{\"a\":\"b\"}}", true,
                     err, sizeof(err)) && strstr(err, "entry is invalid"));
    CHECK(!validates("{}", "{\"dependentRequired\":{\"a\":[1]}}", true,
                     err, sizeof(err)) && strstr(err, "entry is invalid"));
    CHECK(!validates("{}", "{\"dependentRequired\":{\"a\":[\"x\",\"x\"]}}",
                     true, err, sizeof(err)) && strstr(err, "duplicates"));
}

static void test_conditional_branches(void) {
    char err[256];
    const char *conditional =
        "{\"type\":\"object\","
        "\"if\":{\"type\":\"object\",\"required\":[\"kind\"],"
          "\"properties\":{\"kind\":{\"const\":\"a\"}}},"
        "\"then\":{\"required\":[\"a_field\"]},"
        "\"else\":{\"required\":[\"b_field\"]}}";
    CHECK(validates("{\"kind\":\"a\",\"a_field\":1}", conditional, true,
                    err, sizeof(err)));
    CHECK(!validates("{\"kind\":\"a\"}", conditional, true, err, sizeof(err)) &&
          strstr(err, "a_field"));
    CHECK(validates("{\"kind\":\"b\",\"b_field\":1}", conditional, true,
                    err, sizeof(err)));
    CHECK(!validates("{\"kind\":\"b\"}", conditional, true, err, sizeof(err)) &&
          strstr(err, "b_field"));
    // `if` failing must select `else`, not fail the whole schema.
    CHECK(!validates("{}", conditional, true, err, sizeof(err)) &&
          strstr(err, "b_field"));
}

static void test_ref_resolution_and_recursion_limits(void) {
    char err[256];
    // JSON-Pointer escapes: ~1 is '/', ~0 is '~'.
    CHECK(validates("2", "{\"$defs\":{\"a/b\":{\"type\":\"integer\"}},"
                    "\"$ref\":\"#/$defs/a~1b\"}", true, err, sizeof(err)));
    CHECK(validates("2", "{\"$defs\":{\"a~b\":{\"type\":\"integer\"}},"
                    "\"$ref\":\"#/$defs/a~0b\"}", true, err, sizeof(err)));
    CHECK(!validates("\"x\"", "{\"$defs\":{\"a/b\":{\"type\":\"integer\"}},"
                     "\"$ref\":\"#/$defs/a~1b\"}", true, err, sizeof(err)));
    CHECK(!validates("1", "{\"$ref\":\"#/$defs/missing\"}", true,
                     err, sizeof(err)) && strstr(err, "unresolved"));
    CHECK(!validates("1", "{\"$ref\":\"http://example.com/s\"}", true,
                     err, sizeof(err)) && strstr(err, "unresolved"));

    // THE security case for this module: a self-referential $ref must
    // terminate on the depth limit instead of recursing until the stack dies.
    // Preflight only resolves a $ref, so the cycle is caught in validate_node.
    CHECK(!validates("1", "{\"$defs\":{\"node\":{\"$ref\":\"#/$defs/node\"}},"
                     "\"$ref\":\"#/$defs/node\"}", true, err, sizeof(err)) &&
          strstr(err, "recursion exceeds limit"));
    CHECK(!validates("1", "{\"$ref\":\"#\"}", true, err, sizeof(err)) &&
          strstr(err, "recursion exceeds limit"));
}

// Deeply nested schemas must hit the declared depth ceiling rather than
// exhausting the C stack. Built programmatically because the literal would be
// unreadable, and sized past EMBER_SCHEMA_MAX_DEPTH (128).
static void test_depth_limit(void) {
    enum { DEPTH = 200 };
    // Per level: 24 bytes of opening schema ({"type":"array","items":) plus a
    // closing brace; the instance adds one bracket per side.
    const size_t schema_cap = (size_t)DEPTH * 32 + 64;
    const size_t instance_cap = (size_t)DEPTH * 2 + 64;
    char *schema = malloc(schema_cap);
    char *instance = malloc(instance_cap);
    if (!schema || !instance) { free(schema); free(instance); CHECK(false); return; }
    size_t sn = 0, in = 0;
    for (int i = 0; i < DEPTH; ++i) {
        sn += (size_t)snprintf(schema + sn, schema_cap - sn,
                               "{\"type\":\"array\",\"items\":");
        in += (size_t)snprintf(instance + in, instance_cap - in, "[");
    }
    sn += (size_t)snprintf(schema + sn, schema_cap - sn, "{\"type\":\"integer\"}");
    in += (size_t)snprintf(instance + in, instance_cap - in, "1");
    for (int i = 0; i < DEPTH; ++i) {
        sn += (size_t)snprintf(schema + sn, schema_cap - sn, "}");
        in += (size_t)snprintf(instance + in, instance_cap - in, "]");
    }
    CHECK(sn < schema_cap && in < instance_cap);
    // The property that matters is that this terminates and fails closed. The
    // message is deliberately NOT asserted here: the accumulated "$.items.items
    // .items…" path fills any reasonable error buffer before the reason can be
    // written, so a deep failure currently reports path only (see
    // Asserting the truncated text would
    // pin that defect in place.
    char err[256];
    CHECK(!validates(instance, schema, true, err, sizeof(err)));
    free(schema);
    free(instance);
}

int main(void) {
    test_large_number_equality();
    test_recursive_constraints();
    test_refs_and_combinators();
    test_strict_and_duplicates();
    test_contains_cardinality();
    test_structural_equality();
    test_type_arrays();
    test_prefix_items();
    test_pattern_and_property_names();
    test_dependent_required();
    test_conditional_branches();
    test_ref_resolution_and_recursion_limits();
    test_depth_limit();
    printf("tool schema tests: %d passed, %d failed\n", pass, fail);
    return fail != 0;
}
