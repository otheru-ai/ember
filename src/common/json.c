#include "json.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "buf.h"

// #4(c): cap recursion so deeply nested input can't exhaust the C stack.
#define EMBER_JSON_MAX_DEPTH 256

typedef struct {
    const char *p;
    const char *end;
    bool        ok;
    int         depth;  // #4(c): current container-nesting depth
} jp;

static void skip_ws(jp *j) {
    while (j->p < j->end) {
        char c = *j->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') j->p++;
        else break;
    }
}

static ember_json *jnew(ember_json_type t) {
    ember_json *v = (ember_json *)calloc(1, sizeof(ember_json));
    if (!v) ember_buf_fatal("out of memory parsing JSON");
    v->type = t;
    return v;
}

static ember_json *parse_value(jp *j);
static ember_json *parse_value_inner(jp *j);

// Parse a JSON string literal (assumes *p == '"'). Returns malloc'd unescaped C
// string, or NULL on error.
static char *parse_string_raw(jp *j) {
    if (j->p >= j->end || *j->p != '"') { j->ok = false; return NULL; }
    j->p++;
    ember_buf b = {0};
    while (j->p < j->end && *j->p != '"') {
        char c = *j->p++;
        if (c == '\\') {
            if (j->p >= j->end) { j->ok = false; ember_buf_free(&b); return NULL; }
            char e = *j->p++;
            switch (e) {
                case '"':  ember_buf_putc(&b, '"');  break;
                case '\\': ember_buf_putc(&b, '\\'); break;
                case '/':  ember_buf_putc(&b, '/');  break;
                case 'n':  ember_buf_putc(&b, '\n'); break;
                case 't':  ember_buf_putc(&b, '\t'); break;
                case 'r':  ember_buf_putc(&b, '\r'); break;
                case 'b':  ember_buf_putc(&b, '\b'); break;
                case 'f':  ember_buf_putc(&b, '\f'); break;
                case 'u': {
                    if (j->end - j->p < 4) { j->ok = false; ember_buf_free(&b); return NULL; }
                    unsigned cp = 0;
                    for (int k = 0; k < 4; k++) {
                        char h = *j->p++;
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                        else { j->ok = false; ember_buf_free(&b); return NULL; }
                    }
                    // Surrogate pair. As in Dwarfstar, replace an unmatched
                    // surrogate with U+FFFD but do not consume a following
                    // non-low escape: it remains a separate code point.
                    if (cp >= 0xD800 && cp <= 0xDBFF && j->end - j->p >= 6 &&
                        j->p[0] == '\\' && j->p[1] == 'u') {
                        const char *low_start = j->p;
                        j->p += 2;
                        unsigned lo = 0;
                        bool lo_ok = true;
                        for (int k = 0; k < 4; k++) {
                            char h = *j->p++;
                            lo <<= 4;
                            if (h >= '0' && h <= '9') lo |= (unsigned)(h - '0');
                            else if (h >= 'a' && h <= 'f') lo |= (unsigned)(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') lo |= (unsigned)(h - 'A' + 10);
                            else lo_ok = false;
                        }
                        // Only combine a valid low surrogate; otherwise emit the
                        // replacement char (avoids codepoint underflow on garbage).
                        if (lo_ok && lo >= 0xDC00 && lo <= 0xDFFF)
                            cp = 0x10000 + ((cp - 0xD800) << 10) +
                                 (lo - 0xDC00);
                        else {
                            j->p = low_start;
                            cp = 0xFFFDu;
                        }
                    }
                    if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFDu;
                    // The DOM exposes strings as NUL-terminated `char *` and
                    // cannot preserve U+0000 without a parser/validator
                    // differential. Fail closed for executable request data.
                    if (cp == 0) {
                        j->ok = false;
                        ember_buf_free(&b);
                        return NULL;
                    }
                    // encode UTF-8
                    if (cp < 0x80) ember_buf_putc(&b, (char)cp);
                    else if (cp < 0x800) {
                        ember_buf_putc(&b, (char)(0xC0 | (cp >> 6)));
                        ember_buf_putc(&b, (char)(0x80 | (cp & 0x3F)));
                    } else if (cp < 0x10000) {
                        ember_buf_putc(&b, (char)(0xE0 | (cp >> 12)));
                        ember_buf_putc(&b, (char)(0x80 | ((cp >> 6) & 0x3F)));
                        ember_buf_putc(&b, (char)(0x80 | (cp & 0x3F)));
                    } else {
                        ember_buf_putc(&b, (char)(0xF0 | (cp >> 18)));
                        ember_buf_putc(&b, (char)(0x80 | ((cp >> 12) & 0x3F)));
                        ember_buf_putc(&b, (char)(0x80 | ((cp >> 6) & 0x3F)));
                        ember_buf_putc(&b, (char)(0x80 | (cp & 0x3F)));
                    }
                    break;
                }
                default: j->ok = false; ember_buf_free(&b); return NULL;
            }
        } else {
            // RFC 8259 strings may not contain unescaped control characters.
            if ((unsigned char)c < 0x20) {
                j->ok = false;
                ember_buf_free(&b);
                return NULL;
            }
            ember_buf_putc(&b, c);
        }
    }
    if (j->p >= j->end) { j->ok = false; ember_buf_free(&b); return NULL; }
    j->p++;  // closing quote
    char *s = ember_buf_take(&b);
    return s ? s : strdup("");
}

static void arr_push(ember_json *v, const char *key, ember_json *item) {
    int n = v->u.arr.count;
    ember_json **items = (ember_json **)realloc(
        v->u.arr.items, (size_t)(n + 1) * sizeof(ember_json *));
    if (!items) ember_buf_fatal("out of memory growing JSON array");
    v->u.arr.items = items;
    v->u.arr.items[n] = item;
    if (v->type == EMBER_JSON_OBJECT) {
        char **keys = (char **)realloc(
            v->u.arr.keys, (size_t)(n + 1) * sizeof(char *));
        if (!keys) ember_buf_fatal("out of memory growing JSON object");
        v->u.arr.keys = keys;
        v->u.arr.keys[n] = (char *)key;  // owns
    }
    v->u.arr.count = n + 1;
}

// #4(c): depth-guarded wrapper. Every value parse costs one stack level; refuse
// once nesting passes EMBER_JSON_MAX_DEPTH so pathological input can't recurse
// the stack to exhaustion. Balanced increment/decrement keeps sibling counts
// accurate.
static ember_json *parse_value(jp *j) {
    if (++j->depth > EMBER_JSON_MAX_DEPTH) { j->ok = false; --j->depth; return NULL; }
    ember_json *v = parse_value_inner(j);
    --j->depth;
    return v;
}

static ember_json *parse_value_inner(jp *j) {
    skip_ws(j);
    if (j->p >= j->end) { j->ok = false; return NULL; }
    char c = *j->p;
    if (c == '"') {
        char *s = parse_string_raw(j);
        if (!j->ok) return NULL;
        ember_json *v = jnew(EMBER_JSON_STRING);
        v->u.str = s;
        return v;
    }
    if (c == '{') {
        j->p++;
        ember_json *v = jnew(EMBER_JSON_OBJECT);
        skip_ws(j);
        if (j->p < j->end && *j->p == '}') { j->p++; return v; }
        while (1) {
            skip_ws(j);
            char *key = parse_string_raw(j);
            if (!j->ok) { ember_json_free(v); return NULL; }
            skip_ws(j);
            if (j->p >= j->end || *j->p != ':') { j->ok = false; free(key); ember_json_free(v); return NULL; }
            j->p++;
            ember_json *item = parse_value(j);
            if (!j->ok) { free(key); ember_json_free(v); return NULL; }
            arr_push(v, key, item);
            skip_ws(j);
            if (j->p < j->end && *j->p == ',') { j->p++; continue; }
            if (j->p < j->end && *j->p == '}') { j->p++; break; }
            j->ok = false; ember_json_free(v); return NULL;
        }
        return v;
    }
    if (c == '[') {
        j->p++;
        ember_json *v = jnew(EMBER_JSON_ARRAY);
        skip_ws(j);
        if (j->p < j->end && *j->p == ']') { j->p++; return v; }
        while (1) {
            ember_json *item = parse_value(j);
            if (!j->ok) { ember_json_free(v); return NULL; }
            arr_push(v, NULL, item);
            skip_ws(j);
            if (j->p < j->end && *j->p == ',') { j->p++; continue; }
            if (j->p < j->end && *j->p == ']') { j->p++; break; }
            j->ok = false; ember_json_free(v); return NULL;
        }
        return v;
    }
    if (c == 't' && j->end - j->p >= 4 && strncmp(j->p, "true", 4) == 0) {
        j->p += 4; ember_json *v = jnew(EMBER_JSON_BOOL); v->u.b = true; return v;
    }
    if (c == 'f' && j->end - j->p >= 5 && strncmp(j->p, "false", 5) == 0) {
        j->p += 5; ember_json *v = jnew(EMBER_JSON_BOOL); v->u.b = false; return v;
    }
    if (c == 'n' && j->end - j->p >= 4 && strncmp(j->p, "null", 4) == 0) {
        j->p += 4; return jnew(EMBER_JSON_NULL);
    }
    // number
    {
        // #4(a): validate the RFC 8259 number grammar explicitly instead of
        // trusting strtod, which over-accepts leading '+', leading zeros, hex,
        // "inf"/"nan" and locale oddities. Walk the exact grammar
        // ( -? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE][+-]?[0-9]+)? ) to fix the
        // token span, then hand only that span to strtod for the double value.
        const char *ns = j->p;
        const char *p = j->p;
        if (p < j->end && *p == '-') p++;                     // optional minus
        if (p >= j->end || *p < '0' || *p > '9') { j->ok = false; return NULL; }
        if (*p == '0') {                                      // int: 0 | [1-9][0-9]*
            p++;                                              // no leading zeros
        } else {
            while (p < j->end && *p >= '0' && *p <= '9') p++;
        }
        if (p < j->end && *p == '.') {                        // fraction
            p++;
            if (p >= j->end || *p < '0' || *p > '9') { j->ok = false; return NULL; }
            while (p < j->end && *p >= '0' && *p <= '9') p++;
        }
        if (p < j->end && (*p == 'e' || *p == 'E')) {         // exponent
            p++;
            if (p < j->end && (*p == '+' || *p == '-')) p++;
            if (p >= j->end || *p < '0' || *p > '9') { j->ok = false; return NULL; }
            while (p < j->end && *p >= '0' && *p <= '9') p++;
        }
        // Span [ns,p) is now valid JSON number syntax. Run strtod on the exact
        // NUL-terminated token (not the raw stream) so it can't re-read past the
        // span — e.g. "0x1f" validated only "0". Reject values this DOM cannot
        // represent as a finite double; allowing infinity into sampler/model
        // configuration later causes undefined casts and non-finite math.
        j->p = p;
        ember_json *v = jnew(EMBER_JSON_NUMBER);
        v->num_raw = strndup(ns, (size_t)(p - ns));  // preserve exact token
        if (!v->num_raw) ember_buf_fatal("out of memory parsing JSON number");
        errno = 0;
        v->u.num = strtod(v->num_raw, NULL);
        if (!isfinite(v->u.num) || errno == ERANGE) {
            ember_json_free(v);
            j->ok = false;
            return NULL;
        }
        return v;
    }
}

ember_json *ember_json_parse_n(const char *text, size_t len) {
    if (!text) return NULL;
    jp j = {.p = text, .end = text + len, .ok = true, .depth = 0};
    ember_json *v = parse_value(&j);
    if (!j.ok) { if (v) ember_json_free(v); return NULL; }
    skip_ws(&j);
    // #4(b): the top-level value must consume the whole input; reject trailing
    // non-whitespace (e.g. "{} garbage", "1 2") that lenient parsers accept.
    if (j.p != j.end) { if (v) ember_json_free(v); return NULL; }
    return v;
}

ember_json *ember_json_parse(const char *text) {
    return text ? ember_json_parse_n(text, strlen(text)) : NULL;
}

void ember_json_free(ember_json *v) {
    if (!v) return;
    if (v->type == EMBER_JSON_STRING) {
        free(v->u.str);
    } else if (v->type == EMBER_JSON_ARRAY || v->type == EMBER_JSON_OBJECT) {
        for (int i = 0; i < v->u.arr.count; i++) {
            if (v->u.arr.keys) free(v->u.arr.keys[i]);
            ember_json_free(v->u.arr.items[i]);
        }
        free(v->u.arr.items);
        free(v->u.arr.keys);
    }
    free(v->num_raw);  // NULL for non-number nodes
    free(v);
}

const ember_json *ember_json_get(const ember_json *obj, const char *key) {
    if (!obj || obj->type != EMBER_JSON_OBJECT) return NULL;
    for (int i = 0; i < obj->u.arr.count; i++) {
        if (strcmp(obj->u.arr.keys[i], key) == 0) return obj->u.arr.items[i];
    }
    return NULL;
}

const char *ember_json_str(const ember_json *v, const char *dflt) {
    return (v && v->type == EMBER_JSON_STRING) ? v->u.str : dflt;
}
double ember_json_num(const ember_json *v, double dflt) {
    return (v && v->type == EMBER_JSON_NUMBER) ? v->u.num : dflt;
}
bool ember_json_bool(const ember_json *v, bool dflt) {
    return (v && v->type == EMBER_JSON_BOOL) ? v->u.b : dflt;
}
int ember_json_len(const ember_json *v) {
    return (v && (v->type == EMBER_JSON_ARRAY || v->type == EMBER_JSON_OBJECT))
               ? v->u.arr.count : 0;
}
const ember_json *ember_json_at(const ember_json *v, int i) {
    if (!v || (v->type != EMBER_JSON_ARRAY && v->type != EMBER_JSON_OBJECT))
        return NULL;
    if (i < 0 || i >= v->u.arr.count) return NULL;
    return v->u.arr.items[i];
}

const char *ember_json_key(const ember_json *obj, int i) {
    if (!obj || obj->type != EMBER_JSON_OBJECT || !obj->u.arr.keys) return NULL;
    if (i < 0 || i >= obj->u.arr.count) return NULL;
    return obj->u.arr.keys[i];
}

bool ember_json_has_duplicate_keys(const ember_json *v) {
    if (!v) return false;
    if (v->type == EMBER_JSON_OBJECT) {
        for (int i = 0; i < v->u.arr.count; ++i) {
            for (int j = i + 1; j < v->u.arr.count; ++j)
                if (strcmp(v->u.arr.keys[i], v->u.arr.keys[j]) == 0)
                    return true;
            if (ember_json_has_duplicate_keys(v->u.arr.items[i])) return true;
        }
    } else if (v->type == EMBER_JSON_ARRAY) {
        for (int i = 0; i < v->u.arr.count; ++i)
            if (ember_json_has_duplicate_keys(v->u.arr.items[i])) return true;
    }
    return false;
}

#include "json_util.h"

static void dump_into(const ember_json *v, ember_buf *b) {
    if (!v) { ember_buf_puts(b, "null"); return; }
    switch (v->type) {
        case EMBER_JSON_NULL:   ember_buf_puts(b, "null"); break;
        case EMBER_JSON_BOOL:   ember_buf_puts(b, v->u.b ? "true" : "false"); break;
        case EMBER_JSON_NUMBER: {
            if (v->num_raw) { ember_buf_puts(b, v->num_raw); break; }  // byte-exact
            double d = v->u.num;
            if (d == (double)(long long)d) ember_buf_printf(b, "%lld", (long long)d);
            else ember_buf_printf(b, "%g", d);
            break;
        }
        case EMBER_JSON_STRING: ember_json_escape(b, v->u.str); break;
        case EMBER_JSON_ARRAY:
            ember_buf_putc(b, '[');
            for (int i = 0; i < v->u.arr.count; i++) {
                if (i) ember_buf_putc(b, ',');
                dump_into(v->u.arr.items[i], b);
            }
            ember_buf_putc(b, ']');
            break;
        case EMBER_JSON_OBJECT:
            ember_buf_putc(b, '{');
            for (int i = 0; i < v->u.arr.count; i++) {
                if (i) ember_buf_putc(b, ',');
                ember_json_escape(b, v->u.arr.keys[i]);
                ember_buf_putc(b, ':');
                dump_into(v->u.arr.items[i], b);
            }
            ember_buf_putc(b, '}');
            break;
    }
}

char *ember_json_dump(const ember_json *v) {
    ember_buf b = {0};
    dump_into(v, &b);
    char *s = ember_buf_take(&b);
    return s ? s : strdup("null");
}
