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

void ember_tool_calls_free(ember_tool_calls *tc);

// Parse every <｜DSML｜invoke> in `text` into `out`. Returns the number of calls
// found (>=0). Any recognized spelling is accepted; mixed spellings within one
// text are handled. `out` must be zero-initialized.
int ember_parse_dsml_tool_calls(const char *text, ember_tool_calls *out);

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

// Append one parameter as a JSON member `"key":value` into `b` (no comma/brace).
// string="true"/unset → DSML-unescaped then JSON-string; string="false" → raw
// JSON (null if empty). `is_str` is the raw `string="..."` attribute or NULL.
void ember_dsml_append_arg(ember_buf *b, const char *key,
                           const char *val, size_t val_len, const char *is_str);

#ifdef __cplusplus
}
#endif

#endif  // EMBER_TOOL_PARSER_H
