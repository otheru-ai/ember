#include "tool_parser.h"

#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "../common/buf.h"
#include "../common/json.h"
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

static bool contains_n(const char *s, size_t n, const char *needle) {
    size_t m = needle ? strlen(needle) : 0;
    if (!s || m == 0 || n < m) return false;
    for (size_t i = 0; i <= n - m; ++i)
        if (memcmp(s + i, needle, m) == 0) return true;
    return false;
}

// DSML string values are raw, unquoted payload. A nested protocol opener inside
// one is not data unless its leading '<' was escaped; otherwise the first nested
// closing parameter tag terminates the outer value. Reject the whole generated
// turn rather than silently returning a truncated argument.
//
// DELIBERATE DIVERGENCE FROM ds4: ds4_server.c dsml_parse_nested_params_object
// (~4736) *structures* looser nested-XML params into a JSON object. Ember
// instead rejects nested markup (report->contaminated) and routes to the
// single-shot malformed-tool-call retry. Ember's stance is strictly safer (it
// never emits a silently-truncated argument), at the cost of not salvaging a
// nested call the way ds4 does. Revisit only if production telemetry shows
// DeepSeek-V4 drifting into unescaped nested params — then port the ds4
// structurer behind ember's off-by-default posture for risky parity features.
// Nested-value capture. OFF by default: it changes which generated blocks are
// executable, exactly the class of change this repo ships dark and enables in
// deployment explicitly (see CLAUDE.md, and the divergence note below).
//
// Off  -- a nested opener inside a string value is contamination; the whole
//         turn is refused and routed to malformed-call recovery. Safe, but it
//         also refuses balanced values the model plainly meant, and when
//         recovery cannot start the turn dies with a 422.
// On   -- closers are matched by DEPTH, so a balanced nested block is captured
//         whole. Unbalanced nesting is still refused: a truncated argument is
//         never emitted either way, which is the property that matters.
static bool nested_values_enabled(void) {
    static _Thread_local int cached = -1;
    if (cached < 0) {
        const char *e = getenv("EMBER_DSML_NESTED_VALUES");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}

// The first close tag after a value is not necessarily ITS close. A model can
// legitimately put a whole nested block inside a string argument -- writing a
// file whose content contains DSML, or quoting an earlier turn back. Taking the
// first close truncates that value, and a silently-truncated argument is
// precisely the hazard the contamination check exists to prevent. So scan with
// depth: the value keeps its nested block whole and stays faithful to what the
// model actually emitted.
//
// Returns the matching close, or NULL when the nesting never balances -- that
// IS genuinely truncated output, and it stays contaminated and unexecutable.
//
// open_tag is matched as a prefix (the real openers carry attributes and end at
// a later '>'), which is what makes a nested opener count toward depth.
static const char *matching_close(const char *from, const char *open_tag,
                                  const char *close_tag) {
    if (!from || !open_tag || !close_tag) return NULL;
    if (!nested_values_enabled()) return strstr(from, close_tag);
    const size_t o_l = strlen(open_tag), c_l = strlen(close_tag);
    if (!o_l || !c_l) return NULL;
    int depth = 0;
    for (const char *p = from; *p;) {
        if (!strncmp(p, close_tag, c_l)) {
            if (depth == 0) return p;
            --depth;
            p += c_l;
        } else if (!strncmp(p, open_tag, o_l)) {
            if (depth == INT_MAX) return NULL;
            ++depth;
            p += o_l;
        } else {
            ++p;
        }
    }
    return NULL;
}

// A nested invoke opener is only a structural error when it sits in the
// invoke's own body. Inside a parameter VALUE it is data -- the model quoting a
// turn, or writing a file that contains one -- so walk the body and jump over
// each parameter's value using its matching close.
static bool has_structural_tag(const char *from, const char *limit,
                               const ember_dsml_syntax *sx, const char *tag) {
    if (!from || !limit || !tag || from >= limit) return false;
    if (!nested_values_enabled()) {
        const char *hit = strstr(from, tag);
        return hit && hit < limit;
    }
    const size_t t_l = strlen(tag);
    const size_t po_l = strlen(sx->param_open);
    for (const char *p = from; p < limit;) {
        if (!strncmp(p, sx->param_open, po_l)) {
            const char *ptag = strchr(p, '>');
            if (!ptag || ptag >= limit) return false;
            const char *pc =
                matching_close(ptag, sx->param_open, sx->param_close);
            if (!pc || pc >= limit) return false;
            p = pc;                      // skip the value wholesale
            continue;
        }
        if (!strncmp(p, tag, t_l)) return true;
        ++p;
    }
    return false;
}

static bool contains_nested_tool_markup(const char *s, size_t n) {
    for (int i = 0; i < N_SYNTAX; ++i) {
        if (contains_n(s, n, SYNTAX[i].calls_open) ||
            contains_n(s, n, SYNTAX[i].invoke_open) ||
            contains_n(s, n, SYNTAX[i].param_open))
            return true;
    }
    return contains_n(s, n, "<ds_engine_tool_use>");
}

static char *xstrndup(const char *s, size_t n) {
    if (n == SIZE_MAX) ember_buf_fatal("tool parser string size overflow");
    char *p = (char *)malloc(n + 1);
    if (!p) ember_buf_fatal("out of memory parsing tool call");
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
    if (!tag || !tag_limit || !key || tag > tag_limit) return NULL;
    const size_t key_len = strlen(key);
    const char *b = NULL;
    for (const char *p = tag;
         p <= tag_limit && (size_t)(tag_limit - p) >= key_len + 2;
         ++p) {
        // Attribute names must begin at the tag start or after whitespace.
        // A raw strstr("name=\"") incorrectly treats xname="..." as name.
        if (p != tag && !isspace((unsigned char)p[-1])) continue;
        if (memcmp(p, key, key_len) == 0 &&
            p[key_len] == '=' && p[key_len + 1] == '"') {
            b = p + key_len + 2;
            break;
        }
    }
    if (!b) return NULL;
    const char *e = memchr(b, '"', (size_t)(tag_limit - b));
    if (!e) return NULL;
    return dsml_unescape_n(b, (size_t)(e - b));  // ds4 unescapes attribute values
}
#define attr ember_dsml_attr  // internal callers keep the short name

static void push_call(ember_tool_calls *out, char *name, char *args) {
    if (out->len == out->cap) {
        if (out->cap > INT_MAX / 2)
            ember_buf_fatal("too many tool calls");
        int next = out->cap ? out->cap * 2 : 4;
        ember_tool_call *grown = (ember_tool_call *)realloc(
            out->calls, (size_t)next * sizeof(ember_tool_call));
        if (!grown) ember_buf_fatal("out of memory growing tool calls");
        out->calls = grown;
        out->cap = next;
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

bool ember_dsml_append_arg(ember_buf *b, const char *key, const char *val,
                           size_t val_len, const char *is_str) {
    bool valid = true;
    ember_json_escape(b, key);
    ember_buf_putc(b, ':');
    if (is_str && strcmp(is_str, "false") == 0) {
        const char *v = val; size_t vl = val_len;
        while (vl && (*v==' '||*v=='\n'||*v=='\t'||*v=='\r')) { v++; vl--; }
        while (vl && (v[vl-1]==' '||v[vl-1]=='\n'||v[vl-1]=='\t'||v[vl-1]=='\r')) vl--;
        if (vl == 0) {
            ember_buf_puts(b, "null");
            valid = false;
        } else {
            // The model controls this text. Never splice it verbatim into the
            // arguments object: malformed JSON would corrupt the entire tool
            // call streamed to the client. Parse and compact valid values;
            // Keep the output syntactically valid for diagnostics, but report
            // malformed raw JSON to the executable gate. Dwarfstar's silent
            // null coercion is unsafe when the advertised schema expects a
            // number/object/string and the client will execute this result.
            char *raw = xstrndup(v, vl);
            ember_json *parsed = ember_json_parse(raw);
            free(raw);
            if (parsed) {
                char *compact = ember_json_dump(parsed);
                ember_buf_puts(b, compact);
                free(compact);
                ember_json_free(parsed);
            } else {
                ember_buf_puts(b, "null");
                valid = false;
            }
        }
    } else {
        char *u = dsml_unescape_n(val, val_len);
        ember_json_escape(b, u);
        free(u);
    }
    return valid;
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

static int parse_ds_engine(const char *text, ember_tool_calls *out,
                           ember_tool_parse_report *report) {
    const char *cur = text;
    while ((cur = strstr(cur, DSE_OPEN)) != NULL) {
        if (report) report->found = true;
        if (report) report->invocations++;
        const char *tu_close = strstr(cur, DSE_CLOSE);
        if (!tu_close && report) report->complete = false;
        const char *tu_limit = tu_close ? tu_close : text + strlen(text);
        const char *nested = strstr(cur + strlen(DSE_OPEN), DSE_OPEN);
        if (nested && nested < tu_limit && report) report->malformed = true;
        // name (child element)
        char *name = NULL;
        const char *no = strstr(cur, DSE_NAME_O);
        if (no && no < tu_limit) {
            no += strlen(DSE_NAME_O);
            const char *nc = strstr(no, DSE_NAME_C);
            if (nc && nc <= tu_limit) name = trim_dup(no, (size_t)(nc - no));
        }
        if ((!name || !name[0]) && report) report->malformed = true;
        // parameters (property elements with name= string= attrs, value as content)
        ember_buf args = {0};
        ember_buf_putc(&args, '{');
        int nparam = 0;
        const char *p = cur;
        while ((p = strstr(p, DSE_PROP_O)) != NULL && p < tu_limit) {
            const char *ptag = strchr(p, '>');
            if (!ptag || ptag >= tu_limit) {
                if (report) report->malformed = true;
                break;
            }
            const char *pclose = strstr(ptag, DSE_PROP_C);
            if (!pclose || pclose > tu_limit) {
                if (report) report->malformed = true;
                break;
            }
            char *key = ember_dsml_attr(p + strlen(DSE_PROP_O), ptag + 1, "name");
            char *is_str = ember_dsml_attr(p + strlen(DSE_PROP_O), ptag + 1, "string");
            if (!key || !key[0] ||
                (is_str && strcmp(is_str, "true") && strcmp(is_str, "false"))) {
                if (report) report->malformed = true;
            } else {
                if (contains_nested_tool_markup(
                        ptag + 1, (size_t)(pclose - (ptag + 1))) &&
                    report)
                    report->contaminated = true;
                if (nparam++) ember_buf_putc(&args, ',');
                if (!ember_dsml_append_arg(
                        &args, key, ptag + 1,
                        (size_t)(pclose - (ptag + 1)), is_str) && report)
                    report->invalid_json = true;
            }
            free(key); free(is_str);
            p = pclose + strlen(DSE_PROP_C);
        }
        ember_buf_putc(&args, '}');
        if (name && name[0]) push_call(out, name, ember_buf_take(&args));
        else { free(name); ember_buf_free(&args); }
        if (!tu_close) break;  // truncated final block
        cur = tu_close + strlen(DSE_CLOSE);
        const char *next = cur;
        while (*next && isspace((unsigned char)*next)) ++next;
        if (*next && strncmp(next, DSE_OPEN, strlen(DSE_OPEN)) != 0 && report)
            report->trailing = true;
    }
    if (report) {
        const ember_dsml_syntax *foreign = ember_dsml_detect(text);
        if (foreign && strstr(text, foreign->calls_open))
            report->mixed_syntax = true;
    }
    if (report && (report->contaminated || report->invalid_json ||
                   report->trailing || report->mixed_syntax ||
                   report->malformed)) {
        ember_tool_calls_free(out);
        return 0;
    }
    return out->len;
}

int ember_parse_dsml_tool_calls_ex(const char *text, ember_tool_calls *out,
                                   ember_tool_parse_report *report) {
    if (report) {
        memset(report, 0, sizeof(*report));
        report->complete = true;
    }
    if (!text) return 0;

    // Dispatch to whichever tool format appears earliest: DSML or the model's
    // native ds_engine_tool_use.
    const ember_dsml_syntax *sx = ember_dsml_detect(text);
    const char *dsml_at = sx ? strstr(text, sx->calls_open) : NULL;
    const char *dse_at = strstr(text, DSE_OPEN);
    if (dse_at && (!dsml_at || dse_at < dsml_at))
        return parse_ds_engine(text, out, report);
    if (!sx) return 0;
    const char *first = dsml_at;
    const char *original_close = matching_close(
        first + strlen(sx->calls_open), sx->calls_open, sx->calls_close);
    if (report) {
        report->found = true;
        report->complete = original_close != NULL;
        if (original_close) {
            const char *tail = original_close + strlen(sx->calls_close);
            while (*tail && isspace((unsigned char)*tail)) ++tail;
            report->trailing = *tail != '\0';
            // All of these ask "did the model mix or nest FORMATS", which is
            // a question about structure. Inside a parameter value the same
            // bytes are data, so every scan skips values.
            for (int i = 0; i < N_SYNTAX; ++i) {
                if (&SYNTAX[i] == sx) continue;
                if (has_structural_tag(first, original_close, sx,
                                       SYNTAX[i].calls_open) ||
                    has_structural_tag(first, original_close, sx,
                                       SYNTAX[i].invoke_open)) {
                    report->mixed_syntax = true;
                    break;
                }
            }
            if (has_structural_tag(first + strlen(sx->calls_open),
                                   original_close, sx, sx->calls_open))
                report->malformed = true;
            if (has_structural_tag(first, original_close, sx, DSE_OPEN))
                report->mixed_syntax = true;
        }
    }

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
        if (report) report->complete = false;
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
            if (repaired && report) report->repaired = true;
            text = repaired ? repaired : text;
            first = strstr(text, sx->calls_open);
        }
    }

    const size_t inv_open_len = strlen(sx->invoke_open);
    const size_t param_open_len = strlen(sx->param_open);
    const size_t param_close_len = strlen(sx->param_close);

    const char *block_close = matching_close(first + strlen(sx->calls_open),
                                             sx->calls_open, sx->calls_close);
    const char *cur = first;
    while ((cur = strstr(cur, sx->invoke_open)) != NULL &&
           (!block_close || cur < block_close)) {
        const char *tag_end = strchr(cur, '>');
        if (!tag_end || (block_close && tag_end >= block_close)) {
            if (report) report->malformed = true;
            break;
        }
        const char *inv_close =
            matching_close(tag_end, sx->invoke_open, sx->invoke_close);
        if (!inv_close || (block_close && inv_close >= block_close)) {
            if (report) report->malformed = true;
            break;
        }
        if (report) report->invocations++;

        if (has_structural_tag(tag_end + 1, inv_close, sx,
                              sx->invoke_open) && report)
            report->malformed = true;

        char *name = attr(cur + inv_open_len, tag_end + 1, "name");
        if ((!name || !name[0]) && report) report->malformed = true;

        // Build the arguments JSON object from the invoke's parameters.
        ember_buf args = {0};
        ember_buf_putc(&args, '{');
        int nparam = 0;
        const char *p = tag_end + 1;
        while ((p = strstr(p, sx->param_open)) != NULL && p < inv_close) {
            const char *ptag_end = strchr(p, '>');
            if (!ptag_end || ptag_end >= inv_close) {
                if (report) report->malformed = true;
                break;
            }
            const char *pclose =
                matching_close(ptag_end, sx->param_open, sx->param_close);
            if (!pclose || pclose > inv_close) {
                // Nesting never balanced, so there is no faithful value to
                // emit. Report contamination specifically when the unbalanced
                // remainder carries protocol markup: that is the shape the
                // guard exists for, and it reads differently in the 422 than
                // an ordinary truncation.
                if (report) {
                    const char *end = pclose ? pclose : inv_close;
                    if (end > ptag_end + 1 &&
                        contains_nested_tool_markup(
                            ptag_end + 1, (size_t)(end - (ptag_end + 1))))
                        report->contaminated = true;
                    else
                        report->malformed = true;
                }
                break;
            }

            char *key = attr(p + param_open_len, ptag_end + 1, "name");
            char *is_str = attr(p + param_open_len, ptag_end + 1, "string");
            const char *val = ptag_end + 1;
            size_t val_len = (size_t)(pclose - val);

            if (!key || !key[0] ||
                (is_str && strcmp(is_str, "true") && strcmp(is_str, "false"))) {
                if (report) report->malformed = true;
            } else {
                // When dark, any nested markup is contamination. When enabled,
                // matching_close already proved this value balanced, so the
                // block was captured whole rather than truncated at the first
                // inner closer -- not an error.
                if (!nested_values_enabled() &&
                    contains_nested_tool_markup(val, val_len) && report)
                    report->contaminated = true;
                if (nparam++) ember_buf_putc(&args, ',');
                if (!ember_dsml_append_arg(
                        &args, key, val, val_len, is_str) && report)
                    report->invalid_json = true;
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
    if (report && (report->contaminated || report->invalid_json ||
                   report->trailing || report->mixed_syntax ||
                   report->malformed)) {
        ember_tool_calls_free(out);
        return 0;
    }
    return out->len;
}

int ember_parse_dsml_tool_calls(const char *text, ember_tool_calls *out) {
    return ember_parse_dsml_tool_calls_ex(text, out, NULL);
}

static bool json_args_equal(const char *a, const char *b) {
    ember_json *ja = ember_json_parse(a ? a : "{}");
    ember_json *jb = ember_json_parse(b ? b : "{}");
    if (!ja || !jb) {
        if (ja) ember_json_free(ja);
        if (jb) ember_json_free(jb);
        return false;
    }
    char *da = ember_json_dump(ja);
    char *db = ember_json_dump(jb);
    bool equal = da && db && strcmp(da, db) == 0;
    free(da);
    free(db);
    ember_json_free(ja);
    ember_json_free(jb);
    return equal;
}

bool ember_tool_calls_match_raw(const char *raw, const ember_tool_calls *expected) {
    if (!raw || !expected) return false;
    ember_tool_calls parsed = {0};
    ember_parse_dsml_tool_calls(raw, &parsed);
    bool equal = parsed.len == expected->len;
    for (int i = 0; equal && i < parsed.len; i++) {
        const char *pn = parsed.calls[i].name;
        const char *en = expected->calls[i].name;
        equal = pn && en && strcmp(pn, en) == 0 &&
                json_args_equal(parsed.calls[i].arguments,
                                expected->calls[i].arguments);
    }
    ember_tool_calls_free(&parsed);
    return equal;
}
