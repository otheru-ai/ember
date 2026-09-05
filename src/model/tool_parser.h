// DeepSeek DSML tool-call parser: extract structured {name, arguments-json}
// calls from a generated tool-call block. Handles the three spellings a
// converted GGUF produces (full U+FF5C, short leading-pipe-eaten, ascii '?').
//
// Block shape:
//   <｜DSML｜tool_calls>
//   <｜DSML｜invoke name="get_weather">
//   <｜DSML｜parameter name="city" string="true">Tokyo</｜DSML｜parameter>
//   <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
//   </｜DSML｜invoke>
//   </｜DSML｜tool_calls>
//
// string="true"  → value is a raw string (JSON-escaped as-is)
// string="false" → value is verbatim JSON (number/bool/array/object)
#ifndef EMBER_TOOL_PARSER_H
#define EMBER_TOOL_PARSER_H

#include <stddef.h>
#include <stdbool.h>
#include "../common/buf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char *name;        // function name (owned)
    char *arguments;   // JSON object string (owned)
    char *id;          // B3: OpenAI tool-call id (owned; NULL for freshly parsed
                       // model output — minted at emit time, or echoed from
                       // request history for exact-DSML replay)
} ember_tool_call;

typedef struct {
    ember_tool_call *calls;
    int              len;
    int              cap;
} ember_tool_calls;

// Parse diagnostics used by the server's executable-tool gate. The legacy
// parser can repair a missing tail for compatibility, but repaired or
// protocol-contaminated calls must not be executed without a model-visible
// retry.
typedef struct {
    bool found;         // a recognized tool-call opener was present
    bool complete;      // the original text contained its closing block
    bool repaired;      // missing closing tags were synthesized
    bool contaminated;  // a string payload contained nested protocol markup
    bool invalid_json;  // string=false payload was not one complete JSON value
    bool trailing;      // non-whitespace followed a complete DSML wrapper
    bool mixed_syntax;  // another DSML spelling appeared inside the wrapper
    bool malformed;     // nested/mismatched tags or invalid attributes
    int  invocations;   // invocation openers inside the authoritative wrapper
} ember_tool_parse_report;

void ember_tool_calls_free(ember_tool_calls *tc);

// Parse every <｜DSML｜invoke> in `text` into `out`. Returns the number of calls
// found (>=0). Any recognized spelling is accepted; mixed spellings within one
// text are handled. `out` must be zero-initialized.
int ember_parse_dsml_tool_calls(const char *text, ember_tool_calls *out);

// Extended form with structural diagnostics. Calls parsed from contaminated
// string payloads are discarded as a unit (returns 0).
int ember_parse_dsml_tool_calls_ex(const char *text, ember_tool_calls *out,
                                   ember_tool_parse_report *report);

// Parse `raw` and verify that it describes exactly the expected calls, including
// function names and JSON-equivalent arguments in the same order.
bool ember_tool_calls_match_raw(const char *raw, const ember_tool_calls *expected);

// ── exposed for the incremental streaming emitter (sse.c) ──
// One DSML spelling family (full / short / ascii / plain-XML).
typedef struct {
    const char *calls_open, *calls_close;
    const char *invoke_open, *invoke_close;
    const char *param_open,  *param_close;
} ember_dsml_syntax;

// The family whose tool_calls opener appears earliest in `s`, or NULL if none.
const ember_dsml_syntax *ember_dsml_detect(const char *s);

// The full table of recognized spelling families (*n set to the count). Exposed
// for the DSML decode tracker (dsml_decode.c), which scans from a position.
const ember_dsml_syntax *ember_dsml_syntaxes(int *n);

// Extract attribute `key` (key="value") from within [tag, tag_limit) → a newly
// allocated DSML-unescaped value, or NULL. Caller frees.
char *ember_dsml_attr(const char *tag, const char *tag_limit, const char *key);

// Matching close for a DSML tag pair, honouring EMBER_DSML_NESTED_VALUES.
// Dark (default): identical to strstr, i.e. the first close.
// Enabled: skips balanced nested open/close pairs so a nested block inside a
// string value does not terminate the outer tag. EVERY site that walks DSML
// must use this -- the generation stop (main.c), the SSE emitter (sse.c) and
// the parser each had their own first-match walker, and fixing only the parser
// left validated arguments differing from emitted ones (.coord 1067).
const char *ember_dsml_matching_close(const char *from, const char *open_tag,
                                      const char *close_tag);

// Append one parameter as a JSON member `"key":value` into `b` (no comma/brace).
// string="true"/unset → DSML-unescaped then JSON-string; string="false" → raw
// JSON (null if empty). `is_str` is the raw `string="..."` attribute or NULL.
bool ember_dsml_append_arg(ember_buf *b, const char *key,
                           const char *val, size_t val_len, const char *is_str);

#ifdef __cplusplus
}
#endif

#endif  // EMBER_TOOL_PARSER_H
