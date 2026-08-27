// Qwen3-Coder XML tool-call parser used by Qwen3.8-Flash-Next.
//
// The wire shape is not XML in the general sense: names are carried in tag
// openers (`<function=name>`, `<parameter=name>`) and parameter bodies are raw
// text.  Type recovery therefore comes from the request's advertised JSON
// Schema, matching the qwen3_coder parser used by the official serving recipes.
#ifndef EMBER_TOOL_PARSER_QWEN4_H
#define EMBER_TOOL_PARSER_QWEN4_H

#include <stdbool.h>

#include "tool_parser.h"

typedef struct {
    bool found;
    bool complete;
    bool malformed;
    bool contaminated;
    bool trailing;
    int wrappers;
} ember_qwen_tool_parse_report;

// Parse complete Qwen tool-call wrappers from model output. `tools_json` is the
// exact OpenAI tools array used to render the prompt and supplies parameter
// types. Unknown parameters remain strings and are rejected later by Ember's
// ordinary executable schema boundary. `out` must be zero-initialized.
int ember_parse_qwen_tool_calls(const char *text, const char *tools_json,
                                ember_tool_calls *out,
                                ember_qwen_tool_parse_report *report);

#endif  // EMBER_TOOL_PARSER_QWEN4_H
