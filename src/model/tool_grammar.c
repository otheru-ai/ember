#include "tool_grammar.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "../common/buf.h"
#include "../common/json.h"

#define PIPE       "\xef\xbd\x9c"        // U+FF5C ｜ (matches chat_template.c)
#define DSML_OPEN  "<" PIPE "DSML" PIPE
#define DSML_CLOSE "</" PIPE "DSML" PIPE

// A tool may declare more required properties once a discriminator takes a
// particular value:
//
//   "required": ["mode"],
//   "allOf": [{"if":   {"properties": {"mode": {"const": "replace"}}, ...},
//              "then": {"required": ["path", "old_string", "new_string"]}}]
//
// One of these describes a single branch. See the header for why a grammar is
// the right place to enforce this.
typedef struct {
    const char       *disc_name;      // "mode"
    const char       *disc_value;     // "replace"
    const ember_json *extra_required; // then.required
} schema_branch;

#define EMBER_TOOL_GRAMMAR_MAX_BRANCHES 8

// An EBNF string literal. Only `\` and `"` need escaping; the DSML markers are
// ordinary UTF-8 bytes to the grammar parser.
static void put_lit(ember_buf *b, const char *s) {
    ember_buf_putc(b, '"');
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') ember_buf_putc(b, '\\');
        ember_buf_putc(b, *s);
    }
    ember_buf_putc(b, '"');
}

// Rule identifier for a tool. The ordinal prefix keeps names unique even when
// sanitizing maps two distinct tool names onto the same identifier.
static char *rule_name(const char *tool, int ordinal) {
    ember_buf b = {0};
    ember_buf_printf(&b, "t%d_", ordinal);
    for (const char *p = tool; *p; p++) {
        unsigned char c = (unsigned char)*p;
        ember_buf_putc(&b, (isalnum(c) || c == '_') ? (char)c : '_');
    }
    return ember_buf_take(&b);
}

// One <parameter> alternative. `is_str` selects the string="true" form (raw
// text) or string="false" (verbatim JSON), matching append_dsml_args().
static void put_param_alt(ember_buf *out, const char *pname, bool is_str) {
    ember_buf head = {0};
    ember_buf_puts(&head, DSML_OPEN "parameter name=\"");
    ember_buf_puts(&head, pname);
    ember_buf_puts(&head, is_str ? "\" string=\"true\">" : "\" string=\"false\">");
    put_lit(out, head.ptr ? head.ptr : "");
    ember_buf_free(&head);
    ember_buf_puts(out, is_str ? " strval " : " jsonval ");
    put_lit(out, DSML_CLOSE "parameter>");
}

// A property's declared JSON type, or NULL when absent or a union. NULL means
// "do not constrain the string/JSON form".
static const char *decl_type(const ember_json *prop) {
    const ember_json *t = ember_json_get(prop, "type");
    return (t && t->type == EMBER_JSON_STRING) ? ember_json_str(t, NULL) : NULL;
}

static void put_param_rule(ember_buf *out, const char *rule,
                           const char *pname, const ember_json *prop) {
    const char *ty = decl_type(prop);
    ember_buf_puts(out, rule);
    ember_buf_puts(out, " ::= ");
    if (!ty) {
        put_param_alt(out, pname, true);
        ember_buf_puts(out, " | ");
        put_param_alt(out, pname, false);
    } else {
        put_param_alt(out, pname, strcmp(ty, "string") == 0);
    }
    ember_buf_putc(out, '\n');
}

// The discriminator is pinned to the branch's value: emitting it is what
// commits the model to this branch's required set, so it cannot be left open.
static void put_param_rule_const(ember_buf *out, const char *rule,
                                 const char *pname, const char *value) {
    ember_buf head = {0};
    ember_buf_puts(&head, DSML_OPEN "parameter name=\"");
    ember_buf_puts(&head, pname);
    ember_buf_puts(&head, "\" string=\"true\">");
    ember_buf_puts(&head, value);
    ember_buf_puts(&head, DSML_CLOSE "parameter>");
    ember_buf_puts(out, rule);
    ember_buf_puts(out, " ::= ");
    put_lit(out, head.ptr ? head.ptr : "");
    ember_buf_putc(out, '\n');
    ember_buf_free(&head);
}

static bool name_in_array(const ember_json *arr, const char *name) {
    if (!arr || arr->type != EMBER_JSON_ARRAY) return false;
    for (int i = 0; i < ember_json_len(arr); i++) {
        const char *s = ember_json_str(ember_json_at(arr, i), NULL);
        if (s && strcmp(s, name) == 0) return true;
    }
    return false;
}

// Collect if/then branches from `allOf`. Only the discriminated form is
// recognised -- an `if` whose properties pin exactly one property to a const.
// Anything else is ignored, leaving the tool with its unconditional shape,
// because a grammar that guessed at a constraint it did not understand would
// reject valid calls.
static int collect_branches(const ember_json *params, schema_branch *out,
                            int cap) {
    const ember_json *all = params ? ember_json_get(params, "allOf") : NULL;
    if (!all || all->type != EMBER_JSON_ARRAY) return 0;
    int n = 0;
    for (int i = 0; i < ember_json_len(all) && n < cap; i++) {
        const ember_json *e = ember_json_at(all, i);
        const ember_json *iff = ember_json_get(e, "if");
        const ember_json *then = ember_json_get(e, "then");
        if (!iff || !then) continue;
        const ember_json *ip = ember_json_get(iff, "properties");
        if (!ip || ip->type != EMBER_JSON_OBJECT || ember_json_len(ip) != 1)
            continue;
        const char *dname = ember_json_key(ip, 0);
        const ember_json *cv = ember_json_get(ember_json_at(ip, 0), "const");
        const char *dval = ember_json_str(cv, NULL);
        const ember_json *req = ember_json_get(then, "required");
        if (!dname || !dval || !req || req->type != EMBER_JSON_ARRAY) continue;
        out[n].disc_name = dname;
        out[n].disc_value = dval;
        out[n].extra_required = req;
        n++;
    }
    // Every branch must key off the same property, or the alternation would not
    // be a clean partition and a call could satisfy two branches at once.
    for (int i = 1; i < n; i++) {
        if (strcmp(out[i].disc_name, out[0].disc_name) != 0) return 0;
    }
    return n;
}

// Emit one production for a tool (or for one branch of it) plus the parameter
// rules it references, and append its name to the `invoke` alternation.
static void emit_tool_rule(ember_buf *rules, ember_buf *alts, const char *rid,
                           const char *name, const ember_json *props,
                           const ember_json *required,
                           const schema_branch *br, bool first_alt) {
    ember_buf body = {0};
    int nparts = 0;

    // Pass 0: the discriminator, pinned. Pass 1: required (base + branch).
    // Pass 2: optionals, each skippable. See the ORDERING note in the header.
    for (int pass = 0; pass < 3; pass++) {
        for (int p = 0; props && p < ember_json_len(props); p++) {
            const char *pname = ember_json_key(props, p);
            if (!pname) continue;

            const bool is_disc = br && strcmp(pname, br->disc_name) == 0;
            const bool is_req = name_in_array(required, pname) ||
                (br && name_in_array(br->extra_required, pname));

            if (pass == 0 && !is_disc) continue;
            if (pass == 1 && (is_disc || !is_req)) continue;
            if (pass == 2 && (is_disc || is_req)) continue;

            ember_buf prule = {0};
            ember_buf_printf(&prule, "%s_%c%d", rid,
                             pass == 0 ? 'd' : (pass == 1 ? 'r' : 'o'), nparts);
            if (pass == 0)
                put_param_rule_const(rules, prule.ptr, pname, br->disc_value);
            else
                put_param_rule(rules, prule.ptr, pname, ember_json_at(props, p));

            if (nparts) ember_buf_puts(&body, " ws ");
            ember_buf_puts(&body, prule.ptr);
            if (pass == 2) ember_buf_putc(&body, '?');
            ember_buf_free(&prule);
            nparts++;
        }
    }

    ember_buf head = {0};
    ember_buf_puts(&head, DSML_OPEN "invoke name=\"");
    ember_buf_puts(&head, name);
    ember_buf_puts(&head, "\">");

    ember_buf_puts(rules, rid);
    ember_buf_puts(rules, " ::= ");
    put_lit(rules, head.ptr ? head.ptr : "");
    ember_buf_puts(rules, " ws ");
    // A tool with no properties at all still needs a body. The empty invoke is
    // legal ONLY here -- this is the one case production gets right.
    ember_buf_puts(rules, nparts ? body.ptr : "\"\"");
    ember_buf_puts(rules, " ws ");
    put_lit(rules, DSML_CLOSE "invoke>");
    ember_buf_putc(rules, '\n');

    if (!first_alt) ember_buf_puts(alts, " | ");
    ember_buf_puts(alts, rid);

    ember_buf_free(&head);
    ember_buf_free(&body);
}

char *ember_tool_grammar_build(const char *tools_json, bool allow_parallel) {
    if (!tools_json || !tools_json[0]) return NULL;
    ember_json *root = ember_json_parse(tools_json);
    if (!root || root->type != EMBER_JSON_ARRAY) {
        if (root) ember_json_free(root);
        return NULL;
    }

    ember_buf rules = {0};     // per-tool and per-parameter productions
    ember_buf alts = {0};      // right-hand side of `invoke`
    int emitted = 0;

    for (int i = 0; i < ember_json_len(root); i++) {
        const ember_json *entry = ember_json_at(root, i);
        if (!entry) continue;
        // Accept both {"type":"function","function":{...}} and a bare schema.
        const ember_json *fn = ember_json_get(entry, "function");
        if (!fn) fn = entry;

        const char *name = ember_json_str(ember_json_get(fn, "name"), NULL);
        if (!name || !name[0]) continue;

        const ember_json *params = ember_json_get(fn, "parameters");
        const ember_json *props = params ? ember_json_get(params, "properties") : NULL;
        const ember_json *required = params ? ember_json_get(params, "required") : NULL;
        if (props && props->type != EMBER_JSON_OBJECT) props = NULL;

        schema_branch branches[EMBER_TOOL_GRAMMAR_MAX_BRANCHES];
        const int nbranch =
            collect_branches(params, branches, EMBER_TOOL_GRAMMAR_MAX_BRANCHES);

        if (nbranch == 0) {
            char *rid = rule_name(name, emitted);
            if (!rid) continue;
            emit_tool_rule(&rules, &alts, rid, name, props, required, NULL,
                           emitted == 0);
            free(rid);
            emitted++;
            continue;
        }

        // The discriminator's own enum is the complete set of legal values. We
        // must emit an alternative for EVERY one of them, not just the ones
        // carrying a conditional: a value with no `if` branch is still a valid
        // call, and omitting it would make that action UNREPRESENTABLE.
        //
        // Real schemas are full of these -- skill_manage's "delete",
        // cronjob's/process's "list" -- so branch-only emission would silently
        // forbid calls the tool supports. Over-constraining is the worse
        // failure: it cannot be recovered from, whereas under-constraining
        // merely leaves the validator to catch it.
        const ember_json *disc_prop =
            props ? ember_json_get(props, branches[0].disc_name) : NULL;
        const ember_json *disc_enum =
            disc_prop ? ember_json_get(disc_prop, "enum") : NULL;
        if (!disc_enum || disc_enum->type != EMBER_JSON_ARRAY ||
            ember_json_len(disc_enum) == 0) {
            // Without an enum we cannot enumerate the alternatives safely, so
            // fall back to the unconditional shape rather than guess.
            char *rid = rule_name(name, emitted);
            if (!rid) continue;
            emit_tool_rule(&rules, &alts, rid, name, props, required, NULL,
                           emitted == 0);
            free(rid);
            emitted++;
            continue;
        }

        for (int v = 0; v < ember_json_len(disc_enum); v++) {
            const char *val = ember_json_str(ember_json_at(disc_enum, v), NULL);
            if (!val) continue;
            schema_branch b = { branches[0].disc_name, val, NULL };
            for (int k = 0; k < nbranch; k++) {
                if (strcmp(branches[k].disc_value, val) == 0) {
                    b.extra_required = branches[k].extra_required;
                    break;
                }
            }
            ember_buf idb = {0};
            ember_buf_printf(&idb, "%s_b%d", name, v);
            char *rid = rule_name(idb.ptr, emitted);
            ember_buf_free(&idb);
            if (!rid) continue;
            emit_tool_rule(&rules, &alts, rid, name, props, required, &b,
                           emitted == 0);
            free(rid);
            emitted++;
        }
    }

    ember_json_free(root);

    if (!emitted) {
        ember_buf_free(&rules);
        ember_buf_free(&alts);
        return NULL;
    }

    ember_buf out = {0};
    ember_buf_puts(&out, "root ::= ");
    put_lit(&out, DSML_OPEN "tool_calls>");
    ember_buf_puts(&out, " ws invokes ws ");
    put_lit(&out, DSML_CLOSE "tool_calls>");
    ember_buf_putc(&out, '\n');
    ember_buf_puts(&out, allow_parallel ? "invokes ::= invoke (ws invoke)*\n"
                                        : "invokes ::= invoke\n");
    ember_buf_puts(&out, "invoke ::= ");
    ember_buf_puts(&out, alts.ptr);
    ember_buf_putc(&out, '\n');
    ember_buf_puts(&out, "ws ::= [ \\t\\n\\r]*\n");
    // A value may contain '<' -- it just may not open a DSML close tag.
    //
    // This started as [^<]*, which was wrong and shipped: it made EVERY value
    // containing '<' unemittable, so the model could not write `i < n`,
    // `#include <stdio.h>`, or any comparison. Production hit it immediately --
    // the model tried to patch C code, was forced to close the parameter at the
    // '<', got truncated content, and retried in a loop. It even diagnosed the
    // cause itself: "The patch keeps truncating at `<`".
    //
    // chat_template.c's preamble only requires escaping a literal
    // </|DSML|parameter> (as &lt;/...), so bare '<' is legal. Forbidding '<'
    // followed by '/' keeps the closing tag unambiguously findable while
    // allowing everything real code contains. A trailing '<' is allowed
    // explicitly, since a value may legitimately end with one.
    ember_buf_puts(&out, "strval ::= ([^<] | \"<\" [^/])* (\"<\")?\n");
    ember_buf_puts(&out, "jsonval ::= ([^<] | \"<\" [^/])* (\"<\")?\n");
    ember_buf_puts(&out, rules.ptr);

    ember_buf_free(&rules);
    ember_buf_free(&alts);
    return ember_buf_take(&out);
}
