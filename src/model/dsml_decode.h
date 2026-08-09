// B6 — DSML decode-state tracker (structural-token greedy sampling).
//
// Ports ds4's dsml_decode_tracker. While the model is generating a tool call,
// the DSML scaffolding (tags, tag attributes, and JSON structure between the
// arg values) must be well-formed for the tool-call parser to recover it. At
// temperature > 0, sampling noise can corrupt those structural tokens. This
// tracker classifies, byte by byte, whether the CURRENT generation frontier is
// protocol structure (decode greedily) or arbitrary payload — a string body or
// a JSON string value (sample at the request's temperature). The backend
// consults ember_dsml_force_greedy() before sampling each token.
//
// It is a forgiving recognizer, not a validator: malformed DSML still falls
// through to the normal tool-call parser. It runs on the generation worker
// thread only (same thread as the token callback) → no locking.
#ifndef EMBER_DSML_DECODE_H
#define EMBER_DSML_DECODE_H

#include <stddef.h>
#include <stdbool.h>

#include "tool_parser.h"  // ember_dsml_syntax

typedef enum {
    EMBER_DSML_OUTSIDE,          // not inside a tool call
    EMBER_DSML_STRUCTURAL,       // DSML tags / scaffolding → greedy
    EMBER_DSML_STRING_BODY,      // string="true" parameter body → payload
    EMBER_DSML_JSON_STRUCTURAL,  // JSON structure (braces, commas) → greedy
    EMBER_DSML_JSON_STRING,      // inside a JSON string value → payload
} ember_dsml_decode_state;

typedef enum {
    EMBER_DSML_TRK_SEARCH,
    EMBER_DSML_TRK_STRUCTURAL,
    EMBER_DSML_TRK_STRING_BODY,
    EMBER_DSML_TRK_JSON_PARAM,
    EMBER_DSML_TRK_DONE,
} ember_dsml_track_mode;

typedef struct {
    ember_dsml_track_mode      mode;
    ember_dsml_decode_state    decode;
    const ember_dsml_syntax   *syn;
    size_t                     pos;
    bool                       json_in_string;
    bool                       json_escaped;
} ember_dsml_tracker;

void ember_dsml_tracker_init(ember_dsml_tracker *dt);

// Re-classify the decode state given the full accumulated generated text so far
// (`raw[0:raw_len]`). Incremental: only scans forward from the last position.
void ember_dsml_tracker_update(ember_dsml_tracker *dt, const char *raw, size_t raw_len);

// True when the next token is protocol structure inside a tool call and should
// be decoded greedily (argmax) regardless of temperature.
bool ember_dsml_force_greedy(const ember_dsml_tracker *dt);

#endif  // EMBER_DSML_DECODE_H
