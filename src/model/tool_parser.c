#include "tool_parser.h"

#include <stdlib.h>
#include <string.h>

#include "../common/buf.h"
#include "../common/json_util.h"

#define PIPE "\xef\xbd\x9c"  // U+FF5C

// One delimiter family per spelling. The parser picks the family whose
// tool_calls opener appears first, then parses consistently within it.
// (ember_dsml_syntax is declared in tool_parser.h for the streaming emitter.)
static const ember_dsml_syntax SYNTAX[] = {
    {"<" PIPE "DSML" PIPE "tool_calls>",  "</" PIPE "DSML" PIPE "tool_calls>",
     "<" PIPE "DSML" PIPE "invoke",       "</" PIPE "DSML" PIPE "invoke>",
     "<" PIPE "DSML" PIPE "parameter",    "</" PIPE "DSML" PIPE "parameter>"},
    {"<DSML" PIPE "tool_calls>",  "</DSML" PIPE "tool_calls>",
     "<DSML" PIPE "invoke",       "</DSML" PIPE "invoke>",
     "<DSML" PIPE "parameter",    "</DSML" PIPE "parameter>"},
    {"<?DSML?tool_calls>",  "</?DSML?tool_calls>",
     "<?DSML?invoke",       "</?DSML?invoke>",
     "<?DSML?parameter",    "</?DSML?parameter>"},
    // Plain-XML degradation (ds4 style 1): the model drops the DSML pipes.
    {"<tool_calls>",  "</tool_calls>",
     "<invoke",       "</invoke>",
     "<parameter",    "</parameter>"},
};
static const int N_SYNTAX = sizeof(SYNTAX) / sizeof(SYNTAX[0]);

static char *xstrndup(const char *s, size_t n) {
    char *p = (char *)malloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

// ds4 dsml_unescape_text: reverse the entity escapes the prompt tells the model
// to use (&lt; &gt; &amp; &quot; &apos;). Returns a newly-allocated string.
static char *dsml_unescape_n(const char *s, size_t n) {
    ember_buf b = {0};
    for (size_t i = 0; i < n;) {
        if (s[i] == '&') {
            if (i + 4 <= n && !memcmp(s + i, "&lt;", 4))   { ember_buf_putc(&b, '<'); i += 4; continue; }
            if (i + 4 <= n && !memcmp(s + i, "&gt;", 4))   { ember_buf_putc(&b, '>'); i += 4; continue; }
            if (i + 5 <= n && !memcmp(s + i, "&amp;", 5))  { ember_buf_putc(&b, '&'); i += 5; continue; }
            if (i + 6 <= n && !memcmp(s + i, "&quot;", 6)) { ember_buf_putc(&b, '"'); i += 6; continue; }
            if (i + 6 <= n && !memcmp(s + i, "&apos;", 6)) { ember_buf_putc(&b, '\''); i += 6; continue; }
        }
        ember_buf_putc(&b, s[i]); i++;
    }
    char *r = ember_buf_take(&b);
    return r ? r : strdup("");
}

// Extract attribute value: key="..." starting the search at `tag`. Returns a
// newly-allocated value or NULL. `*end` (optional) receives the byte after the
// closing quote.
char *ember_dsml_attr(const char *tag, const char *tag_limit, const char *key) {
    char needle[32];
    snprintf(needle, sizeof(needle), "%s=\"", key);
    const char *b = strstr(tag, needle);
    if (!b || b >= tag_limit) return NULL;
    b += strlen(needle);
    const char *e = memchr(b, '"', (size_t)(tag_limit - b));
    if (!e) return NULL;
    return dsml_unescape_n(b, (size_t)(e - b));  // ds4 unescapes attribute values
}
#define attr ember_dsml_attr  // internal callers keep the short name

static void push_call(ember_tool_calls *out, char *name, char *args) {
    if (out->len == out->cap) {
        out->cap = out->cap ? out->cap * 2 : 4;
        out->calls = (ember_tool_call *)realloc(
            out->calls, (size_t)out->cap * sizeof(ember_tool_call));
    }
    out->calls[out->len].name = name;
    out->calls[out->len].arguments = args;
    out->calls[out->len].id = NULL;  // B3: model output has no id; minted at emit
    out->len++;
}

void ember_tool_calls_free(ember_tool_calls *tc) {
    for (int i = 0; i < tc->len; i++) {
        free(tc->calls[i].name);
        free(tc->calls[i].arguments);
        free(tc->calls[i].id);  // B3
    }
    free(tc->calls);
    tc->calls = NULL;
    tc->len = tc->cap = 0;
}

const ember_dsml_syntax *ember_dsml_detect(const char *s) {
    if (!s) return NULL;
    const ember_dsml_syntax *sx = NULL;
    const char *first = NULL;
    for (int i = 0; i < N_SYNTAX; i++) {
        const char *h = strstr(s, SYNTAX[i].calls_open);
        if (h && (!first || h < first)) { first = h; sx = &SYNTAX[i]; }
    }
    return sx;
}

const ember_dsml_syntax *ember_dsml_syntaxes(int *n) {
    if (n) *n = N_SYNTAX;
    return SYNTAX;
}

void ember_dsml_append_arg(ember_buf *b, const char *key, const char *val,
                           size_t val_len, const char *is_str) {
    ember_json_escape(b, key);
    ember_buf_putc(b, ':');
    if (is_str && strcmp(is_str, "false") == 0) {
        const char *v = val; size_t vl = val_len;
        while (vl && (*v==' '||*v=='\n'||*v=='\t'||*v=='\r')) { v++; vl--; }
        while (vl && (v[vl-1]==' '||v[vl-1]=='\n'||v[vl-1]=='\t'||v[vl-1]=='\r')) vl--;
        if (vl == 0) ember_buf_puts(b, "null");
        else ember_buf_append(b, v, vl);
    } else {
        char *u = dsml_unescape_n(val, val_len);
        ember_json_escape(b, u);
        free(u);
    }
}

// The DeepSeek-V4 model also emits a NATIVE tool format when the DSML preamble
// doesn't fully steer it (seen leaking to clients). Structurally distinct from
// DSML: one <ds_engine_tool_use> block per call, name as a child element, params
// under a wrapper as <..._property name= string=> elements.
#define DSE_OPEN   "<ds_engine_tool_use>"
#define DSE_CLOSE  "</ds_engine_tool_use>"
#define DSE_NAME_O "<ds_engine_tool_use_name>"
#define DSE_NAME_C "</ds_engine_tool_use_name>"
#define DSE_PROP_O "<ds_engine_tool_use_parameters_property"
#define DSE_PROP_C "</ds_engine_tool_use_parameters_property>"

static char *trim_dup(const char *s, size_t n) {
    while (n && (*s==' '||*s=='\n'||*s=='\t'||*s=='\r')) { s++; n--; }
    while (n && (s[n-1]==' '||s[n-1]=='\n'||s[n-1]=='\t'||s[n-1]=='\r')) n--;
    return xstrndup(s, n);
}

static int parse_ds_engine(const char *text, ember_tool_calls *out) {
    const char *cur = text;
    while ((cur = strstr(cur, DSE_OPEN)) != NULL) {
        const char *tu_close = strstr(cur, DSE_CLOSE);
        const char *tu_limit = tu_close ? tu_close : text + strlen(text);
        // name (child element)
        char *name = NULL;
        const char *no = strstr(cur, DSE_NAME_O);
        if (no && no < tu_limit) {
            no += strlen(DSE_NAME_O);
            const char *nc = strstr(no, DSE_NAME_C);
            if (nc && nc <= tu_limit) name = trim_dup(no, (size_t)(nc - no));
        }
        // parameters (property elements with name= string= attrs, value as content)
        ember_buf args = {0};
        ember_buf_putc(&args, '{');
        int nparam = 0;
        const char *p = cur;
        while ((p = strstr(p, DSE_PROP_O)) != NULL && p < tu_limit) {
            const char *ptag = strchr(p, '>');
            if (!ptag || ptag >= tu_limit) break;
            const char *pclose = strstr(ptag, DSE_PROP_C);
            if (!pclose || pclose > tu_limit) break;
            char *key = ember_dsml_attr(p + strlen(DSE_PROP_O), ptag + 1, "name");
            char *is_str = ember_dsml_attr(p + strlen(DSE_PROP_O), ptag + 1, "string");
            if (key) {
                if (nparam++) ember_buf_putc(&args, ',');
                ember_dsml_append_arg(&args, key, ptag + 1,
                                      (size_t)(pclose - (ptag + 1)), is_str);
            }
            free(key); free(is_str);
            p = pclose + strlen(DSE_PROP_C);
        }
        ember_buf_putc(&args, '}');
        if (name && name[0]) push_call(out, name, ember_buf_take(&args));
        else { free(name); ember_buf_free(&args); }
        if (!tu_close) break;  // truncated final block
        cur = tu_close + strlen(DSE_CLOSE);
    }
    return out->len;
}

int ember_parse_dsml_tool_calls(const char *text, ember_tool_calls *out) {
    if (!text) return 0;

    // Dispatch to whichever tool format appears earliest: DSML or the model's
    // native ds_engine_tool_use.
    const ember_dsml_syntax *sx = ember_dsml_detect(text);
    const char *dsml_at = sx ? strstr(text, sx->calls_open) : NULL;
    const char *dse_at = strstr(text, DSE_OPEN);
    if (dse_at && (!dsml_at || dse_at < dsml_at)) return parse_ds_engine(text, out);
    if (!sx) return 0;
    const char *first = dsml_at;

    // #B4 / ds4 try_repair_dsml: if the block opener is present but its closer is
    // missing (generation truncated mid-call), append the missing closers so the
    // already-typed invokes still parse. The previous rule only fired when at
    // least one COMPLETE parameter existed, so it could recover neither a
    // zero-parameter call nor a call whose final parameter closer was lost.
    // Instead count the six tag types over the block (from the first opener,
    // like ds4) and append the missing closers in reverse nesting order:
    // parameters, then invokes, then tool_calls. Require at least one invoke
    // opener so a bare tool_calls opener never fabricates a call, and refuse when
    // any closer already outnumbers its opener (not a truncation pattern — the
    // unsigned differences below would otherwise wrap). The repaired text is
    // validated by the normal parse below: an invoke without a name yields no
    // call, so a repair that recovers nothing emits nothing.
    char *repaired = NULL;
    if (!strstr(first, sx->calls_close) && strstr(first, sx->invoke_open)) {
        const size_t co_l = strlen(sx->calls_open),  cc_l = strlen(sx->calls_close);
        const size_t io_l = strlen(sx->invoke_open), ic_l = strlen(sx->invoke_close);
        const size_t po_l = strlen(sx->param_open),  pc_l = strlen(sx->param_close);
        size_t tos = 0, toe = 0, ios = 0, ioe = 0, pos = 0, poe = 0;
        for (const char *p = first; *p;) {
            if      (!strncmp(p, sx->calls_close,  cc_l)) { toe++; p += cc_l; }
            else if (!strncmp(p, sx->calls_open,   co_l)) { tos++; p += co_l; }
            else if (!strncmp(p, sx->invoke_close, ic_l)) { ioe++; p += ic_l; }
            else if (!strncmp(p, sx->invoke_open,  io_l)) { ios++; p += io_l; }
            else if (!strncmp(p, sx->param_close,  pc_l)) { poe++; p += pc_l; }
            else if (!strncmp(p, sx->param_open,   po_l)) { pos++; p += po_l; }
            else p++;
        }
        if (toe <= tos && ioe <= ios && poe <= pos) {
            ember_buf r = {0};
            ember_buf_puts(&r, text);
            for (size_t i = 0; i < pos - poe; i++) ember_buf_puts(&r, sx->param_close);
            for (size_t i = 0; i < ios - ioe; i++) ember_buf_puts(&r, sx->invoke_close);
            for (size_t i = 0; i < tos - toe; i++) ember_buf_puts(&r, sx->calls_close);
            repaired = ember_buf_take(&r);
            text = repaired ? repaired : text;
            first = strstr(text, sx->calls_open);
        }
    }

    const size_t inv_open_len = strlen(sx->invoke_open);
    const size_t param_open_len = strlen(sx->param_open);
    const size_t param_close_len = strlen(sx->param_close);

    const char *cur = first;
    while ((cur = strstr(cur, sx->invoke_open)) != NULL) {
        const char *tag_end = strchr(cur, '>');
        if (!tag_end) break;
        const char *inv_close = strstr(tag_end, sx->invoke_close);
        if (!inv_close) break;

        char *name = attr(cur + inv_open_len, tag_end + 1, "name");

        // Build the arguments JSON object from the invoke's parameters.
        ember_buf args = {0};
        ember_buf_putc(&args, '{');
        int nparam = 0;
        const char *p = tag_end + 1;
        while ((p = strstr(p, sx->param_open)) != NULL && p < inv_close) {
            const char *ptag_end = strchr(p, '>');
            if (!ptag_end || ptag_end >= inv_close) break;
            const char *pclose = strstr(ptag_end, sx->param_close);
            if (!pclose || pclose > inv_close) break;

            char *key = attr(p + param_open_len, ptag_end + 1, "name");
            char *is_str = attr(p + param_open_len, ptag_end + 1, "string");
            const char *val = ptag_end + 1;
            size_t val_len = (size_t)(pclose - val);

            if (key) {
                if (nparam++) ember_buf_putc(&args, ',');
                ember_dsml_append_arg(&args, key, val, val_len, is_str);
            }
            free(key);
            free(is_str);
            p = pclose + param_close_len;
        }
        ember_buf_putc(&args, '}');

        // #B4: validate before emitting — an invoke with a missing or empty name
        // is not a usable call (matches parse_ds_engine's name && name[0] guard),
        // so a repaired-but-nameless block recovers nothing rather than a bogus
        // call.
        if (name && name[0]) {
            push_call(out, name, ember_buf_take(&args));
        } else {
            free(name);
            ember_buf_free(&args);
        }
        cur = inv_close + strlen(sx->invoke_close);
    }
    free(repaired);
    return out->len;
}
