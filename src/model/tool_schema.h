// Fail-closed JSON Schema validation for generated tool arguments.
//
// The model is not a trust boundary. Prompt instructions and structural-greedy
// DSML decoding improve its output, but only this post-generation check decides
// whether arguments may be exposed as executable. The validator implements the
// validation vocabulary commonly used by OpenAI/Anthropic function schemas,
// including local references and recursive object/array constraints. In strict
// mode an unknown schema keyword is rejected instead of silently weakening the
// contract.
#ifndef EMBER_TOOL_SCHEMA_H
#define EMBER_TOOL_SCHEMA_H

#include <stdbool.h>
#include <stddef.h>

#include "../common/json.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validate `instance` against `schema`. `schema` may be an object or boolean.
// `root_schema` supplies the document used by local $ref pointers and normally
// equals `schema`. On failure `err` receives a concise path-qualified reason.
bool ember_tool_schema_validate(const ember_json *instance,
                                const ember_json *schema,
                                const ember_json *root_schema,
                                bool strict,
                                char *err, size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif
