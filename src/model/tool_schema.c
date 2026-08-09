#include "tool_schema.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <regex.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/buf.h"

#define EMBER_SCHEMA_MAX_DEPTH 128

typedef struct {
    const ember_json *root;
    bool strict;
    char *err;
    size_t err_cap;
} schema_ctx;

static bool schema_fail(schema_ctx *ctx, const char *fmt, ...) {
    if (ctx && ctx->err && ctx->err_cap && !ctx->err[0]) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(ctx->err, ctx->err_cap, fmt, ap);
        va_end(ap);
    }
    return false;
}

typedef struct {
    char *digits;
    long long exponent;
    bool negative;
    bool zero;
} normalized_number;

// Normalize an RFC-8259 decimal as digits * 10^exponent. The JSON DOM keeps
// the original number token specifically so equality need not collapse at the
// double-precision boundary (for example 9007199254740992 vs 9007199254740993).
static bool normalize_number(const ember_json *v, normalized_number *out) {
    memset(out, 0, sizeof(*out));
    if (!v || v->type != EMBER_JSON_NUMBER || !v->num_raw) return false;
    const char *p = v->num_raw;
    if (*p == '-') { out->negative = true; ++p; }

    ember_buf digits = {0};
    size_t fractional = 0;
    bool after_dot = false;
    while (*p && *p != 'e' && *p != 'E') {
        if (*p == '.') { after_dot = true; ++p; continue; }
        ember_buf_putc(&digits, *p++);
        if (after_dot) ++fractional;
    }
    long long explicit_exp = 0;
    if (*p) {
        char *end = NULL;
        errno = 0;
        explicit_exp = strtoll(p + 1, &end, 10);
        if (errno == ERANGE || !end || *end) {
            ember_buf_free(&digits);
            return false;
        }
    }

    size_t first = 0;
    while (first < digits.len && digits.ptr[first] == '0') ++first;
    if (first == digits.len) {
        ember_buf_free(&digits);
        out->digits = strdup("0");
        if (!out->digits) ember_buf_fatal("out of memory normalizing JSON number");
        out->zero = true;
        out->negative = false;
        return true;
    }
    size_t last = digits.len;
    while (last > first && digits.ptr[last - 1] == '0') --last;
    size_t trailing = digits.len - last;
    if (fractional > (size_t)LLONG_MAX || trailing > (size_t)LLONG_MAX ||
        explicit_exp < LLONG_MIN + (long long)fractional ||
        explicit_exp - (long long)fractional >
            LLONG_MAX - (long long)trailing) {
        ember_buf_free(&digits);
        return false;
    }
    out->exponent = explicit_exp - (long long)fractional +
                    (long long)trailing;
    out->digits = strndup(digits.ptr + first, last - first);
    ember_buf_free(&digits);
    if (!out->digits) ember_buf_fatal("out of memory normalizing JSON number");
    return true;
}

static bool json_number_equal(const ember_json *a, const ember_json *b) {
    normalized_number na = {0}, nb = {0};
    bool oka = normalize_number(a, &na);
    bool okb = normalize_number(b, &nb);
    if (!oka || !okb) {
        free(na.digits);
        free(nb.digits);
        return false;
    }
    bool equal = (na.zero && nb.zero) ||
        (na.negative == nb.negative && na.exponent == nb.exponent &&
         strcmp(na.digits, nb.digits) == 0);
    free(na.digits);
    free(nb.digits);
    return equal;
}

static bool json_number_is_integer(const ember_json *v) {
    normalized_number n;
    if (!normalize_number(v, &n)) return false;
    bool result = n.zero || n.exponent >= 0;
    free(n.digits);
    return result;
}

static bool json_equal(const ember_json *a, const ember_json *b) {
    if (!a || !b || a->type != b->type) return false;
    switch (a->type) {
        case EMBER_JSON_NULL: return true;
        case EMBER_JSON_BOOL: return a->u.b == b->u.b;
        case EMBER_JSON_NUMBER: return json_number_equal(a, b);
        case EMBER_JSON_STRING: return strcmp(a->u.str, b->u.str) == 0;
        case EMBER_JSON_ARRAY:
            if (ember_json_len(a) != ember_json_len(b)) return false;
            for (int i = 0; i < ember_json_len(a); ++i)
                if (!json_equal(ember_json_at(a, i), ember_json_at(b, i)))
                    return false;
            return true;
        case EMBER_JSON_OBJECT:
            if (ember_json_len(a) != ember_json_len(b) ||
                ember_json_has_duplicate_keys(a) ||
                ember_json_has_duplicate_keys(b)) return false;
            for (int i = 0; i < ember_json_len(a); ++i) {
                const char *key = ember_json_key(a, i);
                if (!json_equal(ember_json_at(a, i), ember_json_get(b, key)))
                    return false;
            }
            return true;
    }
    return false;
}

static const char *type_name(ember_json_type type) {
    switch (type) {
        case EMBER_JSON_NULL: return "null";
        case EMBER_JSON_BOOL: return "boolean";
        case EMBER_JSON_NUMBER: return "number";
        case EMBER_JSON_STRING: return "string";
        case EMBER_JSON_ARRAY: return "array";
        case EMBER_JSON_OBJECT: return "object";
    }
    return "unknown";
}

static bool is_integer(double d) {
    return isfinite(d) && floor(d) == d;
}

static bool type_matches(const ember_json *v, const char *wanted) {
    if (!wanted) return false;
    if (!strcmp(wanted, "null")) return v->type == EMBER_JSON_NULL;
    if (!strcmp(wanted, "boolean")) return v->type == EMBER_JSON_BOOL;
    if (!strcmp(wanted, "number")) return v->type == EMBER_JSON_NUMBER;
    if (!strcmp(wanted, "integer"))
        return v->type == EMBER_JSON_NUMBER && json_number_is_integer(v);
    if (!strcmp(wanted, "string")) return v->type == EMBER_JSON_STRING;
    if (!strcmp(wanted, "array")) return v->type == EMBER_JSON_ARRAY;
    if (!strcmp(wanted, "object")) return v->type == EMBER_JSON_OBJECT;
    return false;
}

static size_t utf8_codepoints(const char *s) {
    size_t n = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p)
        if ((*p & 0xc0u) != 0x80u) ++n;
    return n;
}

// JSON Schema patterns use ECMA-262 syntax. libc only supplies POSIX ERE, so
// passing patterns straight to regcomp() silently changes their meaning (most
// dangerously, `\d` becomes a literal `d`). Translate the common portable
// subset and reject constructs whose boolean semantics cannot be preserved.
static char *ecma_pattern_to_posix(const char *pattern, bool *valid,
                                   bool *has_wildcard) {
    ember_buf out = {0};
    bool in_class = false;
    bool previous_quantifier = false;
    *valid = false;
    *has_wildcard = false;
    for (const unsigned char *p = (const unsigned char *)pattern; *p; ++p) {
        unsigned char c = *p;
        if (c >= 0x80) goto unsupported;
        if (c == '\\') {
            unsigned char e = *++p;
            if (!e || (e >= '1' && e <= '9') || strchr("bBckpPuUvVxDWS", e))
                goto unsupported;
            if (e == 'd') ember_buf_puts(&out, in_class ? "0-9" : "[0-9]");
            else if (e == 'w')
                ember_buf_puts(&out, in_class ? "A-Za-z0-9_" : "[A-Za-z0-9_]");
            else if (e == 's') {
                if (in_class) goto unsupported;
                ember_buf_puts(&out, "[[:space:]]");
            } else if (strchr(".^$*+?()[]{}|\\/-", e)) {
                ember_buf_putc(&out, '\\');
                ember_buf_putc(&out, (char)e);
            } else goto unsupported;
            previous_quantifier = false;
            continue;
        }
        if (!in_class && c == '(' && p[1] == '?') goto unsupported;
        if (in_class && c == '[' &&
            (p[1] == ':' || p[1] == '.' || p[1] == '=')) goto unsupported;
        if (!in_class && c == '.') *has_wildcard = true;
        if (!in_class && c == '?' && previous_quantifier) goto unsupported;
        if (c == '[') in_class = true;
        else if (c == ']' && in_class) in_class = false;
        ember_buf_putc(&out, (char)c);
        previous_quantifier = !in_class &&
            (c == '*' || c == '+' || c == '?' || c == '}');
    }
    if (in_class) goto unsupported;
    *valid = true;
    return ember_buf_take(&out);

unsupported:
    ember_buf_free(&out);
    return NULL;
}

static bool regex_matches(const char *pattern, const char *value,
                          bool *valid_pattern) {
    bool wildcard = false;
    char *translated = ecma_pattern_to_posix(pattern, valid_pattern, &wildcard);
    if (!translated) return false;
    regex_t re;
    int rc = regcomp(&re, translated, REG_EXTENDED | REG_NOSUB);
    free(translated);
    if (rc != 0) {
        *valid_pattern = false;
        return false;
    }
    *valid_pattern = true;
    // POSIX '.' includes newlines without REG_NEWLINE, unlike ECMA-262. Fail
    // closed for that divergent input rather than accidentally broadening it.
    bool divergent_wildcard = wildcard &&
        (strchr(value, '\n') || strchr(value, '\r'));
    rc = divergent_wildcard ? REG_NOMATCH : regexec(&re, value, 0, NULL, 0);
    regfree(&re);
    return rc == 0;
}

static bool keyword_known(const char *key) {
    static const char *known[] = {
        "$schema", "$id", "$anchor", "$defs", "definitions", "$ref",
        "$comment", "title", "description", "default", "deprecated",
        "readOnly", "writeOnly", "examples", "type", "enum", "const",
        "multipleOf", "maximum", "exclusiveMaximum", "minimum",
        "exclusiveMinimum", "maxLength", "minLength", "pattern", "maxItems",
        "minItems", "uniqueItems", "maxContains", "minContains", "items",
        "prefixItems", "contains", "maxProperties", "minProperties",
        "required", "dependentRequired", "properties", "patternProperties",
        "additionalProperties", "propertyNames", "allOf", "anyOf", "oneOf",
        "not", "if", "then", "else"
    };
    for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); ++i)
        if (!strcmp(key, known[i])) return true;
    return false;
}

static const ember_json *resolve_ref(const ember_json *root, const char *ref);
static bool schema_preflight(schema_ctx *ctx, const ember_json *schema,
                             const char *path, int depth);

static void schema_child_key(char out[320], const char *path,
                             const char *key) {
    snprintf(out, 320, "%.*s.%.*s", 150, path ? path : "",
             150, key ? key : "");
}

static void schema_child_index(char out[320], const char *path, int index) {
    snprintf(out, 320, "%.*s[%d]", 300, path ? path : "", index);
}

static bool schema_map_preflight(schema_ctx *ctx, const ember_json *map,
                                 const char *path, int depth) {
    if (!map || map->type != EMBER_JSON_OBJECT)
        return schema_fail(ctx, "%s must be an object of schemas", path);
    for (int i = 0; i < ember_json_len(map); ++i) {
        char child[320];
        schema_child_key(child, path, ember_json_key(map, i));
        if (!schema_preflight(ctx, ember_json_at(map, i), child, depth + 1))
            return false;
    }
    return true;
}

static bool schema_list_preflight(schema_ctx *ctx, const ember_json *list,
                                  const char *path, int depth,
                                  bool require_nonempty) {
    if (!list || list->type != EMBER_JSON_ARRAY ||
        (require_nonempty && ember_json_len(list) == 0))
        return schema_fail(ctx, "%s must be a non-empty schema array", path);
    for (int i = 0; i < ember_json_len(list); ++i) {
        char child[320]; schema_child_index(child, path, i);
        if (!schema_preflight(ctx, ember_json_at(list, i), child, depth + 1))
            return false;
    }
    return true;
}

static bool valid_schema_type_name(const char *name) {
    static const char *names[] = {
        "null", "boolean", "number", "integer", "string", "array", "object"
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i)
        if (name && !strcmp(name, names[i])) return true;
    return false;
}

// Validate the schema document independently of the particular instance. This
// prevents an invalid/unsupported constraint in an optional property, a losing
// anyOf branch, or `not` from being silently skipped and weakening strict mode.
static bool schema_preflight(schema_ctx *ctx, const ember_json *schema,
                             const char *path, int depth) {
    if (!schema)
        return schema_fail(ctx, "%s schema is missing", path);
    if (depth > EMBER_SCHEMA_MAX_DEPTH)
        return schema_fail(ctx, "%s schema nesting exceeds limit", path);
    if (schema->type == EMBER_JSON_BOOL) return true;
    if (schema->type != EMBER_JSON_OBJECT)
        return schema_fail(ctx, "%s schema must be an object or boolean", path);
    if (ember_json_has_duplicate_keys(schema))
        return schema_fail(ctx, "%s schema contains duplicate keys", path);
    if (ctx->strict) {
        for (int i = 0; i < ember_json_len(schema); ++i) {
            const char *key = ember_json_key(schema, i);
            if (!keyword_known(key))
                return schema_fail(ctx,
                    "%s schema uses unsupported strict keyword %s", path, key);
        }
    }

    const ember_json *type = ember_json_get(schema, "type");
    if (type) {
        if (type->type == EMBER_JSON_STRING) {
            if (!valid_schema_type_name(type->u.str))
                return schema_fail(ctx, "%s schema type is invalid", path);
        } else if (type->type == EMBER_JSON_ARRAY && ember_json_len(type) > 0) {
            for (int i = 0; i < ember_json_len(type); ++i) {
                const char *name = ember_json_str(ember_json_at(type, i), NULL);
                if (!valid_schema_type_name(name))
                    return schema_fail(ctx, "%s schema type array is invalid", path);
                for (int j = 0; j < i; ++j)
                    if (!strcmp(name, ember_json_str(ember_json_at(type, j), "")))
                        return schema_fail(ctx, "%s schema type array has duplicates", path);
            }
        } else {
            return schema_fail(ctx, "%s schema type must be a string or non-empty array", path);
        }
    }

    const ember_json *enumeration = ember_json_get(schema, "enum");
    if (enumeration && (enumeration->type != EMBER_JSON_ARRAY ||
                        ember_json_len(enumeration) == 0))
        return schema_fail(ctx, "%s schema enum must be a non-empty array", path);

    static const char *numeric[] = {
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"
    };
    for (size_t i = 0; i < sizeof(numeric) / sizeof(numeric[0]); ++i) {
        const ember_json *v = ember_json_get(schema, numeric[i]);
        if (v && (v->type != EMBER_JSON_NUMBER || !isfinite(v->u.num)))
            return schema_fail(ctx, "%s schema %s must be numeric", path, numeric[i]);
    }
    const ember_json *multiple = ember_json_get(schema, "multipleOf");
    if (multiple && (multiple->type != EMBER_JSON_NUMBER ||
                     !isfinite(multiple->u.num) || multiple->u.num <= 0.0))
        return schema_fail(ctx, "%s schema multipleOf must be positive", path);

    static const char *counts[] = {
        "minLength", "maxLength", "minItems", "maxItems", "minContains",
        "maxContains", "minProperties", "maxProperties"
    };
    for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {
        const ember_json *v = ember_json_get(schema, counts[i]);
        if (v && (v->type != EMBER_JSON_NUMBER || !is_integer(v->u.num) ||
                  v->u.num < 0.0 || v->u.num > 2147483647.0))
            return schema_fail(ctx, "%s schema %s is invalid", path, counts[i]);
    }
    const ember_json *unique = ember_json_get(schema, "uniqueItems");
    if (unique && unique->type != EMBER_JSON_BOOL)
        return schema_fail(ctx, "%s schema uniqueItems must be boolean", path);
    const ember_json *pattern = ember_json_get(schema, "pattern");
    if (pattern) {
        bool valid = false;
        if (pattern->type != EMBER_JSON_STRING ||
            (regex_matches(pattern->u.str, "", &valid), !valid))
            return schema_fail(ctx, "%s schema pattern is invalid", path);
    }

    static const char *schema_maps[] = {
        "$defs", "definitions", "properties", "patternProperties"
    };
    for (size_t i = 0; i < sizeof(schema_maps) / sizeof(schema_maps[0]); ++i) {
        const ember_json *map = ember_json_get(schema, schema_maps[i]);
        if (!map) continue;
        char child[320]; snprintf(child, sizeof(child), "%s.%s", path, schema_maps[i]);
        if (!schema_map_preflight(ctx, map, child, depth + 1)) return false;
        if (!strcmp(schema_maps[i], "patternProperties")) {
            for (int j = 0; j < ember_json_len(map); ++j) {
                bool valid = false;
                (void)regex_matches(ember_json_key(map, j), "", &valid);
                if (!valid)
                    return schema_fail(ctx,
                        "%s schema patternProperties key is invalid", path);
            }
        }
    }
    static const char *schema_lists[] = {
        "allOf", "anyOf", "oneOf", "prefixItems"
    };
    for (size_t i = 0; i < sizeof(schema_lists) / sizeof(schema_lists[0]); ++i) {
        const ember_json *list = ember_json_get(schema, schema_lists[i]);
        if (!list) continue;
        char child[320]; snprintf(child, sizeof(child), "%s.%s", path, schema_lists[i]);
        if (!schema_list_preflight(ctx, list, child, depth + 1,
                                   strcmp(schema_lists[i], "prefixItems") != 0))
            return false;
    }
    static const char *single_schemas[] = {
        "items", "contains", "additionalProperties", "propertyNames",
        "not", "if", "then", "else"
    };
    for (size_t i = 0; i < sizeof(single_schemas) / sizeof(single_schemas[0]); ++i) {
        const ember_json *child_schema = ember_json_get(schema, single_schemas[i]);
        if (!child_schema) continue;
        char child[320]; snprintf(child, sizeof(child), "%s.%s", path, single_schemas[i]);
        if (!schema_preflight(ctx, child_schema, child, depth + 1)) return false;
    }

    const ember_json *required = ember_json_get(schema, "required");
    if (required) {
        if (required->type != EMBER_JSON_ARRAY)
            return schema_fail(ctx, "%s schema required must be an array", path);
        for (int i = 0; i < ember_json_len(required); ++i) {
            const char *name = ember_json_str(ember_json_at(required, i), NULL);
            if (!name)
                return schema_fail(ctx, "%s schema required is invalid", path);
            for (int j = 0; j < i; ++j)
                if (!strcmp(name, ember_json_str(ember_json_at(required, j), "")))
                    return schema_fail(ctx, "%s schema required has duplicates", path);
        }
    }
    const ember_json *dependent = ember_json_get(schema, "dependentRequired");
    if (dependent) {
        if (dependent->type != EMBER_JSON_OBJECT)
            return schema_fail(ctx, "%s schema dependentRequired must be an object", path);
        for (int i = 0; i < ember_json_len(dependent); ++i) {
            const ember_json *names = ember_json_at(dependent, i);
            if (!names || names->type != EMBER_JSON_ARRAY)
                return schema_fail(ctx, "%s dependentRequired entry is invalid", path);
            for (int j = 0; j < ember_json_len(names); ++j) {
                const char *name = ember_json_str(ember_json_at(names, j), NULL);
                if (!name)
                    return schema_fail(ctx, "%s dependentRequired entry is invalid", path);
                for (int k = 0; k < j; ++k)
                    if (!strcmp(name, ember_json_str(ember_json_at(names, k), "")))
                        return schema_fail(ctx, "%s dependentRequired has duplicates", path);
            }
        }
    }
    const ember_json *ref = ember_json_get(schema, "$ref");
    if (ref) {
        const char *name = ember_json_str(ref, NULL);
        if (!name || !resolve_ref(ctx->root, name))
            return schema_fail(ctx, "%s schema has an unresolved local $ref", path);
    }
    return true;
}

static const ember_json *resolve_ref(const ember_json *root, const char *ref) {
    if (!root || !ref) return NULL;
    if (!strcmp(ref, "#")) return root;
    if (strncmp(ref, "#/", 2) != 0) return NULL;
    const ember_json *cur = root;
    const char *p = ref + 2;
    while (cur && *p) {
        const char *slash = strchr(p, '/');
        size_t raw_n = slash ? (size_t)(slash - p) : strlen(p);
        char *key = (char *)malloc(raw_n + 1);
        if (!key) return NULL;
        size_t n = 0;
        for (size_t i = 0; i < raw_n; ++i) {
            if (p[i] == '~' && i + 1 < raw_n && p[i + 1] == '0') {
                key[n++] = '~'; ++i;
            } else if (p[i] == '~' && i + 1 < raw_n && p[i + 1] == '1') {
                key[n++] = '/'; ++i;
            } else {
                key[n++] = p[i];
            }
        }
        key[n] = '\0';
        cur = ember_json_get(cur, key);
        free(key);
        if (!slash) break;
        p = slash + 1;
    }
    return cur;
}

static bool validate_node(schema_ctx *ctx, const ember_json *instance,
                          const ember_json *schema, const char *path,
                          int depth);

static bool test_branch(const schema_ctx *ctx, const ember_json *instance,
                        const ember_json *schema, const char *path, int depth) {
    char ignored[1] = {0};
    schema_ctx probe = *ctx;
    probe.err = ignored;
    probe.err_cap = sizeof(ignored);
    return validate_node(&probe, instance, schema, path, depth);
}

static bool validate_schema_array(schema_ctx *ctx, const ember_json *instance,
                                  const ember_json *array, const char *path,
                                  int depth, const char *kind) {
    if (!array || array->type != EMBER_JSON_ARRAY)
        return schema_fail(ctx, "%s schema keyword %s must be an array", path, kind);
    if (!strcmp(kind, "allOf")) {
        for (int i = 0; i < ember_json_len(array); ++i)
            if (!validate_node(ctx, instance, ember_json_at(array, i), path,
                               depth + 1)) return false;
        return true;
    }
    int matches = 0;
    for (int i = 0; i < ember_json_len(array); ++i)
        if (test_branch(ctx, instance, ember_json_at(array, i), path, depth + 1))
            ++matches;
    if (!strcmp(kind, "anyOf") && matches == 0)
        return schema_fail(ctx, "%s did not match anyOf", path);
    if (!strcmp(kind, "oneOf") && matches != 1)
        return schema_fail(ctx, "%s matched %d oneOf branches", path, matches);
    return true;
}

static bool validate_type(schema_ctx *ctx, const ember_json *instance,
                          const ember_json *type, const char *path) {
    if (!type) return true;
    bool match = false;
    if (type->type == EMBER_JSON_STRING) {
        match = type_matches(instance, type->u.str);
    } else if (type->type == EMBER_JSON_ARRAY) {
        for (int i = 0; i < ember_json_len(type); ++i) {
            const char *name = ember_json_str(ember_json_at(type, i), NULL);
            if (!name) return schema_fail(ctx, "%s schema type array is invalid", path);
            if (type_matches(instance, name)) match = true;
        }
    } else {
        return schema_fail(ctx, "%s schema type must be a string or array", path);
    }
    if (!match)
        return schema_fail(ctx, "%s is %s and does not satisfy schema type",
                           path, type_name(instance->type));
    return true;
}

static bool validate_number(schema_ctx *ctx, const ember_json *v,
                            const ember_json *s, const char *path) {
    if (v->type != EMBER_JSON_NUMBER) return true;
    struct bound { const char *key; bool lower; bool exclusive; } bounds[] = {
        {"minimum", true, false}, {"exclusiveMinimum", true, true},
        {"maximum", false, false}, {"exclusiveMaximum", false, true},
    };
    for (size_t i = 0; i < sizeof(bounds) / sizeof(bounds[0]); ++i) {
        const ember_json *b = ember_json_get(s, bounds[i].key);
        if (!b) continue;
        if (b->type != EMBER_JSON_NUMBER)
            return schema_fail(ctx, "%s schema %s must be numeric", path,
                               bounds[i].key);
        bool ok = bounds[i].lower
            ? (bounds[i].exclusive ? v->u.num > b->u.num : v->u.num >= b->u.num)
            : (bounds[i].exclusive ? v->u.num < b->u.num : v->u.num <= b->u.num);
        if (!ok) return schema_fail(ctx, "%s violates %s", path, bounds[i].key);
    }
    const ember_json *multiple = ember_json_get(s, "multipleOf");
    if (multiple) {
        if (multiple->type != EMBER_JSON_NUMBER || multiple->u.num <= 0.0)
            return schema_fail(ctx, "%s schema multipleOf must be positive", path);
        double q = v->u.num / multiple->u.num;
        double nearest = round(q);
        if (fabs(q - nearest) > 1e-9 * fmax(1.0, fabs(q)))
            return schema_fail(ctx, "%s violates multipleOf", path);
    }
    return true;
}

static bool validate_string(schema_ctx *ctx, const ember_json *v,
                            const ember_json *s, const char *path) {
    if (v->type != EMBER_JSON_STRING) return true;
    size_t length = utf8_codepoints(v->u.str);
    const ember_json *min = ember_json_get(s, "minLength");
    const ember_json *max = ember_json_get(s, "maxLength");
    if (min && (min->type != EMBER_JSON_NUMBER || !is_integer(min->u.num) ||
                min->u.num < 0.0))
        return schema_fail(ctx, "%s schema minLength is invalid", path);
    if (max && (max->type != EMBER_JSON_NUMBER || !is_integer(max->u.num) ||
                max->u.num < 0.0))
        return schema_fail(ctx, "%s schema maxLength is invalid", path);
    if (min && (double)length < min->u.num)
        return schema_fail(ctx, "%s is shorter than minLength", path);
    if (max && (double)length > max->u.num)
        return schema_fail(ctx, "%s is longer than maxLength", path);
    const ember_json *pattern = ember_json_get(s, "pattern");
    if (pattern) {
        if (pattern->type != EMBER_JSON_STRING)
            return schema_fail(ctx, "%s schema pattern must be a string", path);
        bool valid = false;
        bool match = regex_matches(pattern->u.str, v->u.str, &valid);
        if (!valid) return schema_fail(ctx, "%s schema pattern is invalid", path);
        if (!match) return schema_fail(ctx, "%s does not match pattern", path);
    }
    return true;
}

static bool validate_array(schema_ctx *ctx, const ember_json *v,
                           const ember_json *s, const char *path, int depth) {
    if (v->type != EMBER_JSON_ARRAY) return true;
    int n = ember_json_len(v);
    const ember_json *min = ember_json_get(s, "minItems");
    const ember_json *max = ember_json_get(s, "maxItems");
    if (min && (min->type != EMBER_JSON_NUMBER || !is_integer(min->u.num) ||
                min->u.num < 0.0))
        return schema_fail(ctx, "%s schema minItems is invalid", path);
    if (max && (max->type != EMBER_JSON_NUMBER || !is_integer(max->u.num) ||
                max->u.num < 0.0))
        return schema_fail(ctx, "%s schema maxItems is invalid", path);
    if (min && (double)n < min->u.num)
        return schema_fail(ctx, "%s has fewer than minItems", path);
    if (max && (double)n > max->u.num)
        return schema_fail(ctx, "%s has more than maxItems", path);
    if (ember_json_bool(ember_json_get(s, "uniqueItems"), false)) {
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (json_equal(ember_json_at(v, i), ember_json_at(v, j)))
                    return schema_fail(ctx, "%s array items are not unique", path);
    }
    const ember_json *prefix = ember_json_get(s, "prefixItems");
    int prefixed = 0;
    if (prefix) {
        if (prefix->type != EMBER_JSON_ARRAY)
            return schema_fail(ctx, "%s schema prefixItems must be an array", path);
        prefixed = ember_json_len(prefix) < n ? ember_json_len(prefix) : n;
        for (int i = 0; i < prefixed; ++i) {
            char child[320]; snprintf(child, sizeof(child), "%s[%d]", path, i);
            if (!validate_node(ctx, ember_json_at(v, i), ember_json_at(prefix, i),
                               child, depth + 1)) return false;
        }
    }
    const ember_json *items = ember_json_get(s, "items");
    if (items) {
        for (int i = prefixed; i < n; ++i) {
            char child[320]; snprintf(child, sizeof(child), "%s[%d]", path, i);
            if (!validate_node(ctx, ember_json_at(v, i), items, child,
                               depth + 1)) return false;
        }
    }
    const ember_json *contains = ember_json_get(s, "contains");
    if (contains) {
        int matches = 0;
        for (int i = 0; i < n; ++i)
            if (test_branch(ctx, ember_json_at(v, i), contains, path, depth + 1))
                ++matches;
        int min_contains = 1, max_contains = -1;
        const ember_json *mn = ember_json_get(s, "minContains");
        const ember_json *mx = ember_json_get(s, "maxContains");
        if (mn && (mn->type != EMBER_JSON_NUMBER ||
                   !is_integer(mn->u.num) || mn->u.num < 0 ||
                   mn->u.num > 2147483647.0))
            return schema_fail(ctx, "%s schema minContains is invalid", path);
        if (mx && (mx->type != EMBER_JSON_NUMBER ||
                   !is_integer(mx->u.num) || mx->u.num < 0 ||
                   mx->u.num > 2147483647.0))
            return schema_fail(ctx, "%s schema maxContains is invalid", path);
        if (mn) min_contains = (int)mn->u.num;
        if (mx) max_contains = (int)mx->u.num;
        if (matches < min_contains || (max_contains >= 0 && matches > max_contains))
            return schema_fail(ctx, "%s violates contains cardinality", path);
    }
    return true;
}

static bool object_property_matches(const ember_json *patterns,
                                    const char *key, bool *schema_error) {
    if (!patterns) return false;
    if (patterns->type != EMBER_JSON_OBJECT) {
        *schema_error = true;
        return false;
    }
    bool matched = false;
    for (int i = 0; i < ember_json_len(patterns); ++i) {
        bool valid = false;
        if (regex_matches(ember_json_key(patterns, i), key, &valid)) matched = true;
        if (!valid) *schema_error = true;
    }
    return matched;
}

static bool validate_object(schema_ctx *ctx, const ember_json *v,
                            const ember_json *s, const char *path, int depth) {
    if (v->type != EMBER_JSON_OBJECT) return true;
    if (ember_json_has_duplicate_keys(v))
        return schema_fail(ctx, "%s contains duplicate object keys", path);
    int n = ember_json_len(v);
    const ember_json *min = ember_json_get(s, "minProperties");
    const ember_json *max = ember_json_get(s, "maxProperties");
    if (min && (min->type != EMBER_JSON_NUMBER ||
                !is_integer(min->u.num) || min->u.num < 0))
        return schema_fail(ctx, "%s schema minProperties is invalid", path);
    if (max && (max->type != EMBER_JSON_NUMBER ||
                !is_integer(max->u.num) || max->u.num < 0))
        return schema_fail(ctx, "%s schema maxProperties is invalid", path);
    if (min && (double)n < min->u.num)
        return schema_fail(ctx, "%s has fewer than minProperties", path);
    if (max && (double)n > max->u.num)
        return schema_fail(ctx, "%s has more than maxProperties", path);

    const ember_json *required = ember_json_get(s, "required");
    if (required) {
        if (required->type != EMBER_JSON_ARRAY)
            return schema_fail(ctx, "%s schema required must be an array", path);
        for (int i = 0; i < ember_json_len(required); ++i) {
            const char *name = ember_json_str(ember_json_at(required, i), NULL);
            if (!name) return schema_fail(ctx, "%s schema required is invalid", path);
            if (!ember_json_get(v, name))
                return schema_fail(ctx, "%s is missing required property %s", path, name);
        }
    }

    const ember_json *properties = ember_json_get(s, "properties");
    if (properties && properties->type != EMBER_JSON_OBJECT)
        return schema_fail(ctx, "%s schema properties must be an object", path);
    const ember_json *patterns = ember_json_get(s, "patternProperties");
    const ember_json *additional = ember_json_get(s, "additionalProperties");
    const ember_json *property_names = ember_json_get(s, "propertyNames");
    for (int i = 0; i < n; ++i) {
        const char *key = ember_json_key(v, i);
        const ember_json *value = ember_json_at(v, i);
        char child[320]; snprintf(child, sizeof(child), "%s.%s", path, key);
        if (property_names) {
            ember_json key_value = {.type = EMBER_JSON_STRING};
            key_value.u.str = (char *)key;
            if (!validate_node(ctx, &key_value, property_names, child, depth + 1))
                return false;
        }
        bool covered = false;
        const ember_json *property_schema = properties
            ? ember_json_get(properties, key) : NULL;
        if (property_schema) {
            covered = true;
            if (!validate_node(ctx, value, property_schema, child, depth + 1))
                return false;
        }
        bool pattern_error = false;
        bool pattern_match = object_property_matches(patterns, key, &pattern_error);
        if (pattern_error)
            return schema_fail(ctx, "%s schema patternProperties is invalid", path);
        if (pattern_match) {
            covered = true;
            for (int pi = 0; pi < ember_json_len(patterns); ++pi) {
                bool valid = false;
                if (regex_matches(ember_json_key(patterns, pi), key, &valid) &&
                    !validate_node(ctx, value, ember_json_at(patterns, pi), child,
                                   depth + 1)) return false;
            }
        }
        if (!covered && additional) {
            if (additional->type == EMBER_JSON_BOOL && !additional->u.b)
                return schema_fail(ctx, "%s contains additional property %s", path, key);
            if (additional->type != EMBER_JSON_BOOL &&
                !validate_node(ctx, value, additional, child, depth + 1))
                return false;
        }
    }

    const ember_json *dependent = ember_json_get(s, "dependentRequired");
    if (dependent) {
        if (dependent->type != EMBER_JSON_OBJECT)
            return schema_fail(ctx, "%s schema dependentRequired must be an object", path);
        for (int i = 0; i < ember_json_len(dependent); ++i) {
            const char *trigger = ember_json_key(dependent, i);
            if (!ember_json_get(v, trigger)) continue;
            const ember_json *deps = ember_json_at(dependent, i);
            if (!deps || deps->type != EMBER_JSON_ARRAY)
                return schema_fail(ctx, "%s dependentRequired entry is invalid", path);
            for (int j = 0; j < ember_json_len(deps); ++j) {
                const char *name = ember_json_str(ember_json_at(deps, j), NULL);
                if (!name) return schema_fail(ctx, "%s dependentRequired entry is invalid", path);
                if (!ember_json_get(v, name))
                    return schema_fail(ctx, "%s requires dependent property %s", path, name);
            }
        }
    }
    return true;
}

static bool validate_node(schema_ctx *ctx, const ember_json *instance,
                          const ember_json *schema, const char *path,
                          int depth) {
    if (!instance || !schema)
        return schema_fail(ctx, "%s has a missing schema or instance", path);
    if (depth > EMBER_SCHEMA_MAX_DEPTH)
        return schema_fail(ctx, "%s schema recursion exceeds limit", path);
    if (schema->type == EMBER_JSON_BOOL)
        return schema->u.b ? true : schema_fail(ctx, "%s is rejected by false schema", path);
    if (schema->type != EMBER_JSON_OBJECT)
        return schema_fail(ctx, "%s schema must be an object or boolean", path);
    if (ember_json_has_duplicate_keys(schema))
        return schema_fail(ctx, "%s schema contains duplicate keys", path);
    if (ctx->strict) {
        for (int i = 0; i < ember_json_len(schema); ++i) {
            const char *key = ember_json_key(schema, i);
            if (!keyword_known(key))
                return schema_fail(ctx, "%s schema uses unsupported strict keyword %s",
                                   path, key);
        }
    }

    const ember_json *ref = ember_json_get(schema, "$ref");
    if (ref) {
        const char *name = ember_json_str(ref, NULL);
        const ember_json *target = resolve_ref(ctx->root, name);
        if (!name || !target)
            return schema_fail(ctx, "%s schema has an unresolved local $ref", path);
        if (!validate_node(ctx, instance, target, path, depth + 1)) return false;
    }
    const ember_json *all = ember_json_get(schema, "allOf");
    const ember_json *any = ember_json_get(schema, "anyOf");
    const ember_json *one = ember_json_get(schema, "oneOf");
    if (all && !validate_schema_array(ctx, instance, all, path, depth, "allOf"))
        return false;
    if (any && !validate_schema_array(ctx, instance, any, path, depth, "anyOf"))
        return false;
    if (one && !validate_schema_array(ctx, instance, one, path, depth, "oneOf"))
        return false;
    const ember_json *not_schema = ember_json_get(schema, "not");
    if (not_schema && test_branch(ctx, instance, not_schema, path, depth + 1))
        return schema_fail(ctx, "%s matches forbidden schema", path);
    const ember_json *if_schema = ember_json_get(schema, "if");
    if (if_schema) {
        bool condition = test_branch(ctx, instance, if_schema, path, depth + 1);
        const ember_json *branch = ember_json_get(schema, condition ? "then" : "else");
        if (branch && !validate_node(ctx, instance, branch, path, depth + 1))
            return false;
    }

    if (!validate_type(ctx, instance, ember_json_get(schema, "type"), path))
        return false;
    const ember_json *constant = ember_json_get(schema, "const");
    if (constant && !json_equal(instance, constant))
        return schema_fail(ctx, "%s does not equal const", path);
    const ember_json *values = ember_json_get(schema, "enum");
    if (values) {
        if (values->type != EMBER_JSON_ARRAY)
            return schema_fail(ctx, "%s schema enum must be an array", path);
        bool match = false;
        for (int i = 0; i < ember_json_len(values); ++i)
            if (json_equal(instance, ember_json_at(values, i))) match = true;
        if (!match) return schema_fail(ctx, "%s is not an allowed enum value", path);
    }
    return validate_number(ctx, instance, schema, path) &&
           validate_string(ctx, instance, schema, path) &&
           validate_array(ctx, instance, schema, path, depth) &&
           validate_object(ctx, instance, schema, path, depth);
}

bool ember_tool_schema_validate(const ember_json *instance,
                                const ember_json *schema,
                                const ember_json *root_schema,
                                bool strict,
                                char *err, size_t err_cap) {
    if (err && err_cap) err[0] = '\0';
    schema_ctx ctx = {
        .root = root_schema ? root_schema : schema,
        .strict = strict,
        .err = err,
        .err_cap = err_cap,
    };
    if (!schema_preflight(&ctx, ctx.root, "$", 0)) return false;
    return validate_node(&ctx, instance, schema, "$", 0);
}
