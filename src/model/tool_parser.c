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

#define DSE_OPEN   "<ds_engine_tool_use>"
#define DSE_CLOSE  "</ds_engine_tool_use>"
#define DSE_NAME_O "<ds_engine_tool_use_name>"
#define DSE_NAME_C "</ds_engine_tool_use_name>"
#define DSE_PROP_O "<ds_engine_tool_use_parameters_property"
#define DSE_PROP_C "</ds_engine_tool_use_parameters_property>"


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
// A protocol terminator inside a JSON string literal is DATA, not a frame
// boundary. This is the <script> problem: the answer is to parse the frame
// rather than scan it, so the payload never has to be restricted.
//
// Deliberately NOT gated on EMBER_DSML_NESTED_VALUES. Depth tracking is a
// nesting policy; string awareness is correctness, and a marker inside a
// quoted value is never a terminator under any policy.
//
// Returns NULL when the value ends inside a string or on a dangling escape.
// That makes an incomplete value NON-RECOVERABLE rather than executable, which
// is the property codex-rejoin-01 asked for: repair must never turn a truncated
// payload into a successful call.
static const char *json_value_close(const char *from, const char *open_tag,
                                    const char *close_tag) {
    if (!from || !open_tag || !close_tag) return NULL;
    const size_t o_l = strlen(open_tag), c_l = strlen(close_tag);
    if (!o_l || !c_l) return NULL;
    const bool nested = nested_values_enabled();
    int depth = 0;
    bool in_string = false, escaped = false;
    for (const char *p = from; *p;) {
        if (in_string) {
            if (escaped)         escaped = false;
            else if (*p == '\\') escaped = true;
            else if (*p == '"')  in_string = false;
            ++p;
            continue;
        }
        if (*p == '"') { in_string = true; ++p; continue; }
        if (!strncmp(p, close_tag, c_l)) {
            if (depth == 0) return p;
            --depth;
            p += c_l;
            continue;
        }
        if (nested && !strncmp(p, open_tag, o_l)) {
            if (depth == INT_MAX) return NULL;
            ++depth;
            p += o_l;
            continue;
        }
        ++p;
    }
    return NULL;
}

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

// Scan for a FRAME terminator -- a tool_calls or invoke close -- treating every
// parameter block as opaque. A parameter VALUE may legally contain text
// identical to a terminator, so the frame scan must step over the value rather
// than through it. Without this the outer scan cuts at an embedded marker and
// the parameter-level fix never gets a chance to run, which is why a
// parser-only change is not sufficient: sse.c truncates on the same boundary
// before the parser ever sees the text.
//
// A parameter whose value cannot be closed makes the whole frame unresolvable
// and returns NULL, so a truncated payload stays non-executable.
static const char *frame_close(const char *from, const char *open_tag,
                               const char *close_tag,
                               const ember_dsml_syntax *sx) {
    if (!from || !open_tag || !close_tag) return NULL;
    if (!sx) return matching_close(from, open_tag, close_tag);
    const size_t o_l = strlen(open_tag), c_l = strlen(close_tag);
    const size_t po_l = strlen(sx->param_open), pc_l = strlen(sx->param_close);
    if (!o_l || !c_l || !po_l || !pc_l) return NULL;
    const bool nested = nested_values_enabled();
    int depth = 0;
    for (const char *p = from; *p;) {
        if (!strncmp(p, sx->param_open, po_l)) {
            const char *ptag_end = strchr(p, '>');
            if (!ptag_end) return NULL;
            const char *pc = ember_dsml_value_close(
                ptag_end, sx->param_open, sx->param_close,
                ember_dsml_param_is_json(p, po_l, ptag_end + 1));
            if (!pc) return NULL;
            p = pc + pc_l;
            continue;
        }
        if (!strncmp(p, close_tag, c_l)) {
            if (depth == 0) return p;
            --depth;
            p += c_l;
            continue;
        }
        if (nested && !strncmp(p, open_tag, o_l)) {
            if (depth == INT_MAX) return NULL;
            ++depth;
            p += o_l;
            continue;
        }
        ++p;
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
    // The value-skipping walk below is now used regardless of
    // EMBER_DSML_NESTED_VALUES. The old shortcut scanned raw bytes, so a
    // calls_open inside a JSON string read as structural contamination and the
    // call was rejected -- the same class of defect as the boundary scan, and
    // it did not depend on the nesting policy.
    const size_t t_l = strlen(tag);
    const size_t po_l = strlen(sx->param_open);
    for (const char *p = from; p < limit;) {
        if (!strncmp(p, sx->param_open, po_l)) {
            const char *ptag = strchr(p, '>');
            if (!ptag || ptag >= limit) return false;
            const char *pc = ember_dsml_value_close(
                ptag, sx->param_open, sx->param_close,
                ember_dsml_param_is_json(p, po_l, ptag + 1));
            if (!pc || pc >= limit) return false;
            p = pc;                      // skip the value wholesale
            continue;
        }
        if (!strncmp(p, tag, t_l)) return true;
        ++p;
    }
    return false;
}

static bool starts_with_n(const char *s, size_t n, size_t i,
                          const char *needle) {
    const size_t l = strlen(needle);
    return i + l <= n && !memcmp(s + i, needle, l);
}

static bool markup_at(const char *s, size_t n, size_t i) {
    for (int k = 0; k < N_SYNTAX; ++k) {
        if (starts_with_n(s, n, i, SYNTAX[k].calls_open) ||
            starts_with_n(s, n, i, SYNTAX[k].invoke_open) ||
            starts_with_n(s, n, i, SYNTAX[k].param_open))
            return true;
    }
    // The native format's CHILD markers too. Only DSE_OPEN was listed, so a
    // property value carrying a name or property opener raised no contamination
    // at all -- which is how an injected name element sat in a value with
    // contaminated=0. For the DSML families all three openers are covered; the
    // native family was covered by one. Raised by dsh-1537943 as the second
    // half of the E2 mechanism.
    return starts_with_n(s, n, i, DSE_OPEN) ||
           starts_with_n(s, n, i, DSE_NAME_O) ||
           starts_with_n(s, n, i, DSE_PROP_O);
}

// Protocol markup inside a JSON STRING is data, exactly as it is for the
// boundary scan. Flagging it as contamination rejected valid calls in every
// dialect -- the native opener included -- which is the same defect one layer
// up. `json_value` false keeps the old whole-region policy for raw text.
static bool contains_nested_tool_markup_ctx(const char *s, size_t n,
                                            bool json_value) {
    if (!s) return false;
    bool in_string = false, escaped = false;
    for (size_t i = 0; i < n; ++i) {
        if (json_value) {
            if (in_string) {
                if (escaped)          escaped = false;
                else if (s[i] == '\\') escaped = true;
                else if (s[i] == '"') in_string = false;
                continue;
            }
            if (s[i] == '"') { in_string = true; continue; }
        }
        if (markup_at(s, n, i)) return true;
    }
    return false;
}

static bool contains_nested_tool_markup(const char *s, size_t n) {
    return contains_nested_tool_markup_ctx(s, n, false);
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
const char *ember_dsml_matching_close(const char *from, const char *open_tag,
                                      const char *close_tag) {
    return matching_close(from, open_tag, close_tag);
}

bool ember_dsml_param_is_json(const char *tag, size_t open_len,
                              const char *tag_limit) {
    char *is_str = ember_dsml_attr(tag + open_len, tag_limit, "string");
    const bool json_value = is_str && !strcmp(is_str, "false");
    free(is_str);
    return json_value;
}

const char *ember_dsml_value_close(const char *from, const char *open_tag,
                                   const char *close_tag, bool json_value) {
    return json_value ? json_value_close(from, open_tag, close_tag)
                      : matching_close(from, open_tag, close_tag);
}

static const char *dse_frame_close(const char *from, const char *close_tag);

const char *ember_dsml_frame_close(const char *from, const char *open_tag,
                                   const char *close_tag,
                                   const ember_dsml_syntax *sx) {
    // ember_dsml_detect returns NULL for the native ds_engine format, so
    // without this the upstream stop path fell back to the raw matcher and
    // stopped INSIDE a JSON value -- the native parse was correct while the
    // stream was cut short. Found by codex-rejoin-01.
    if (!sx && open_tag && !strcmp(open_tag, DSE_OPEN))
        return dse_frame_close(from, close_tag);
    return frame_close(from, open_tag, close_tag, sx);
}

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

// Where the last string="false" payload stopped parsing. append_arg has three
// callers and a published signature, so a thread-local carries this diagnostic
// rather than churning the API for it. Written on failure; read by the caller
// that sets report->invalid_json for the same call, on the same thread.
static _Thread_local size_t tp_last_json_err_off;
static _Thread_local size_t tp_last_json_err_len;

void ember_tool_parser_last_json_error(size_t off, size_t len) {
    tp_last_json_err_off = off;
    tp_last_json_err_len = len;
}

static void tp_record_json_error(ember_tool_parse_report *report) {
    if (!report || report->invalid_json) return;   // keep the first failure
    report->invalid_json_offset = tp_last_json_err_off;
    report->invalid_json_len = tp_last_json_err_len;
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
            size_t err_off = 0;
            ember_json *parsed = ember_json_parse_at(raw, vl, &err_off);
            free(raw);
            if (parsed) {
                char *compact = ember_json_dump(parsed);
                ember_buf_puts(b, compact);
                free(compact);
                ember_json_free(parsed);
            } else {
                ember_buf_puts(b, "null");
                valid = false;
                ember_tool_parser_last_json_error(err_off, vl);
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

static char *trim_dup(const char *s, size_t n) {
    while (n && (*s==' '||*s=='\n'||*s=='\t'||*s=='\r')) { s++; n--; }
    while (n && (s[n-1]==' '||s[n-1]=='\n'||s[n-1]=='\t'||s[n-1]=='\r')) n--;
    return xstrndup(s, n);
}

// ds_engine's native format has the same framing hazard and used plain strstr
// for tool_use and property termination. A property VALUE may legally contain
// text identical to either terminator, so the frame scan steps over each
// property the way frame_close does for DSML.
static const char *dse_frame_close(const char *from, const char *close_tag) {
    if (!from || !close_tag) return NULL;
    const size_t c_l = strlen(close_tag);
    const size_t po_l = strlen(DSE_PROP_O), pc_l = strlen(DSE_PROP_C);
    for (const char *p = from; *p;) {
        if (!strncmp(p, DSE_PROP_O, po_l)) {
            const char *ptag = strchr(p, '>');
            if (!ptag) return NULL;
            const char *pc = ember_dsml_value_close(
                ptag, DSE_PROP_O, DSE_PROP_C,
                ember_dsml_param_is_json(p, po_l, ptag + 1));
            if (!pc) return NULL;
            p = pc + pc_l;
            continue;
        }
        if (!strncmp(p, close_tag, c_l)) return p;
        ++p;
    }
    return NULL;
}

static int parse_ds_engine(const char *text, ember_tool_calls *out,
                           ember_tool_parse_report *report) {
    const char *cur = text;
    while ((cur = strstr(cur, DSE_OPEN)) != NULL) {
        if (report) report->found = true;
        if (report) report->invocations++;
        const char *tu_close = dse_frame_close(cur + strlen(DSE_OPEN), DSE_CLOSE);
        if (!tu_close && report) report->complete = false;
        const char *tu_limit = tu_close ? tu_close : text + strlen(text);
        // Nested detection steps over property values too: an opener inside a
        // property value is data, not a nested invocation.
        const char *nested = dse_frame_close(cur + strlen(DSE_OPEN), DSE_OPEN);
        if (nested && nested < tu_limit && report) report->malformed = true;
        // name (child element). Found STRUCTURALLY: a bare strstr here took
        // the tool name from a property VALUE when the name element was
        // absent, producing a clean call -- malformed=0 -- whose name came
        // from payload content. The name selects which tool runs, so that is
        // the most consequential place in this parser to trust payload bytes.
        // Reachability demonstrated with a native block carrying no name
        // element and a property value containing one.
        //
        // dse_frame_close already steps over property values; the previous
        // comment below claimed every native scan did so, which was one scan
        // too strong. Found by dsh-1538188 (E3) and dsh-1537943 (E2) reaching
        // the same line from opposite directions.
        char *name = NULL;
        const char *no = dse_frame_close(cur + strlen(DSE_OPEN), DSE_NAME_O);
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
            char *key = ember_dsml_attr(p + strlen(DSE_PROP_O), ptag + 1, "name");
            char *is_str = ember_dsml_attr(p + strlen(DSE_PROP_O), ptag + 1, "string");
            const char *pclose = ember_dsml_value_close(
                ptag, DSE_PROP_O, DSE_PROP_C,
                is_str && !strcmp(is_str, "false"));
            if (!pclose || pclose > tu_limit) {
                if (report) report->malformed = true;
                free(key); free(is_str);
                break;
            }
            if (!key || !key[0] ||
                (is_str && strcmp(is_str, "true") && strcmp(is_str, "false"))) {
                if (report) report->malformed = true;
            } else {
                if (contains_nested_tool_markup_ctx(
                        ptag + 1, (size_t)(pclose - (ptag + 1)),
                        is_str && !strcmp(is_str, "false")) &&
                    report)
                    report->contaminated = true;
                if (nparam++) ember_buf_putc(&args, ',');
                if (!ember_dsml_append_arg(
                        &args, key, ptag + 1,
                        (size_t)(pclose - (ptag + 1)), is_str) && report)
                    { tp_record_json_error(report); report->invalid_json = true; }
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
        // Scan the STRUCTURE, not the payload: a foreign opener inside a JSON
        // property value is data, and flagging it made ["<tool_calls>"] report
        // mixed_syntax in the native format. Property values are stepped over,
        // as they now are in the tool_use, nested, property and NAME scans.
        //
        // The earlier wording claimed every native scan already did this while
        // the name scan did not, and dsh-1538188 used exactly that gap as the
        // lead that found it: a comment asserting a property the code lacks is
        // the same defect one level up. Enumerating the scans is deliberate --
        // "every other scan" cannot be checked by a reader, a list can.
        const size_t po_l = strlen(DSE_PROP_O), pc_l = strlen(DSE_PROP_C);
        for (const char *p = text; *p && !report->mixed_syntax;) {
            if (!strncmp(p, DSE_PROP_O, po_l)) {
                const char *ptag = strchr(p, '>');
                if (!ptag) break;
                const char *pc = ember_dsml_value_close(
                    ptag, DSE_PROP_O, DSE_PROP_C,
                    ember_dsml_param_is_json(p, po_l, ptag + 1));
                if (!pc) break;
                p = pc + pc_l;
                continue;
            }
            for (int i = 0; i < N_SYNTAX; ++i) {
                if (!strncmp(p, SYNTAX[i].calls_open,
                             strlen(SYNTAX[i].calls_open))) {
                    report->mixed_syntax = true;
                    break;
                }
            }
            ++p;
        }
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
    // Same context-aware contract as the stop finder and the extractor. With
    // the old scanner an exact calls_close inside a valid JSON string set
    // report.trailing and rejected the call, even though ember_find_tool_end
    // correctly reached the real end -- the checked and the executed boundary
    // disagreed.
    const char *original_close = frame_close(
        first + strlen(sx->calls_open), sx->calls_open, sx->calls_close, sx);
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
    // Repair is the ONLY consumer that can invent bytes, so a raw scan is least
    // acceptable here. The counting walked the whole text with strncmp,
    // markers inside parameter VALUES included, and then appended one closer
    // per apparent deficit -- synthesising framing out of payload content.
    // Completeness is judged with the shared frame scan for the same reason.
    if (!frame_close(first + strlen(sx->calls_open), sx->calls_open,
                     sx->calls_close, sx) &&
        strstr(first, sx->invoke_open)) {
        if (report) report->complete = false;
        const size_t co_l = strlen(sx->calls_open),  cc_l = strlen(sx->calls_close);
        const size_t io_l = strlen(sx->invoke_open), ic_l = strlen(sx->invoke_close);
        const size_t po_l = strlen(sx->param_open),  pc_l = strlen(sx->param_close);
        size_t tos = 0, toe = 0, ios = 0, ioe = 0, pos = 0, poe = 0;
        bool unterminated_value = false;
        for (const char *p = first; *p;) {
            if (!strncmp(p, sx->param_open, po_l)) {
                const char *ptag = strchr(p, '>');
                pos++;
                if (!ptag) break;          // opener truncated mid-tag
                const bool json_v =
                    ember_dsml_param_is_json(p, po_l, ptag + 1);
                const char *pc = ember_dsml_value_close(
                    ptag, sx->param_open, sx->param_close, json_v);
                if (pc) {
                    poe++;
                    p = pc + pc_l;
                    continue;
                }
                if (json_v) {
                    // A JSON value that ends inside a string or on a dangling
                    // escape has no honest closer to synthesise: appending one
                    // would turn a truncated payload into a complete-looking
                    // call, which is what repair must never do.
                    unterminated_value = true;
                    break;
                }
                // A RAW value with no closer is the historical truncated-tail
                // case, which repair is meant to recover and which the
                // executable gate then rejects via report.repaired. Keep the
                // old deficit accounting and keep scanning past the tag so
                // later markers are still counted.
                p = ptag + 1;
                continue;
            }
            if      (!strncmp(p, sx->calls_close,  cc_l)) { toe++; p += cc_l; }
            else if (!strncmp(p, sx->calls_open,   co_l)) { tos++; p += co_l; }
            else if (!strncmp(p, sx->invoke_close, ic_l)) { ioe++; p += ic_l; }
            else if (!strncmp(p, sx->invoke_open,  io_l)) { ios++; p += io_l; }
            else if (!strncmp(p, sx->param_close,  pc_l)) { poe++; p += pc_l; }
            else p++;
        }
        if (!unterminated_value && toe <= tos && ioe <= ios && poe <= pos) {
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

    const char *block_close = frame_close(first + strlen(sx->calls_open),
                                          sx->calls_open, sx->calls_close, sx);
    const char *cur = first;
    while ((cur = strstr(cur, sx->invoke_open)) != NULL &&
           (!block_close || cur < block_close)) {
        const char *tag_end = strchr(cur, '>');
        if (!tag_end || (block_close && tag_end >= block_close)) {
            if (report) report->malformed = true;
            break;
        }
        const char *inv_close =
            frame_close(tag_end, sx->invoke_open, sx->invoke_close, sx);
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
            // Read string= FIRST: it decides how the value must be scanned.
            // A JSON value needs quote and escape awareness so a protocol
            // terminator inside a string literal is treated as data; raw
            // string=true text keeps plain scanning, because there quotes are
            // ordinary characters that may be unmatched and a literal
            // terminator is encoded with the existing &lt; entity convention.
            char *is_str_early = attr(p + param_open_len, ptag_end + 1, "string");
            // JSON scanning ONLY for an explicit string="false". An ABSENT
            // attribute is raw text, which is what append_arg and the
            // documented contract already assume -- treating it as JSON
            // rejected a formerly valid value like: hello " quote, because the
            // lone quote read as the start of a JSON string.
            const bool json_value =
                is_str_early && !strcmp(is_str_early, "false");
            const char *pclose = ember_dsml_value_close(
                ptag_end, sx->param_open, sx->param_close, json_value);
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
                free(is_str_early);
                break;
            }

            char *key = attr(p + param_open_len, ptag_end + 1, "name");
            char *is_str = is_str_early;
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
                    contains_nested_tool_markup_ctx(
                        val, val_len, is_str && !strcmp(is_str, "false")) &&
                    report)
                    report->contaminated = true;
                if (nparam++) ember_buf_putc(&args, ',');
                if (!ember_dsml_append_arg(
                        &args, key, val, val_len, is_str) && report)
                    { tp_record_json_error(report); report->invalid_json = true; }
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
    // Reparse WITH the report. The NULL-report wrapper repairs a truncated tail
    // and returns a complete-looking call, so an INCOMPLETE prefix of a stream
    // compared equal to the finished call and replay attached its exact tokens.
    // codex-rejoin-01's all-byte sweep found 155 such prefixes across five
    // families; the repair conversion widened the set from 96, so this became
    // worse before it got better. Replay shares the parser but not the
    // executable gate, so it has to apply the equivalent checks itself.
    ember_tool_parse_report report = {0};
    ember_parse_dsml_tool_calls_ex(raw, &parsed, &report);
    bool equal = report.found && report.complete && !report.repaired &&
                 !report.malformed && !report.contaminated &&
                 !report.invalid_json && !report.trailing &&
                 !report.mixed_syntax &&
                 parsed.len == report.invocations &&
                 parsed.len == expected->len;
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
