// Extract a normalized chat-completion request from a parsed JSON body.
// Handles content as a plain string OR as an array of {type:"text",text:...}
// parts (multimodal shape — text parts flattened, non-text ignored for now).
#ifndef EMBER_CHAT_API_H
#define EMBER_CHAT_API_H

#include <stdbool.h>
#include <stdint.h>
#include "../common/json.h"
#include "../model/continuation.h"
#include "../model/tool_parser.h"   // ember_tool_calls (assistant history replay)

// Reasoning-effort → thinking mode, mirroring ds4's ds4_think_mode.
// NONE = thinking off; HIGH = default reasoning; MAX = maximum-effort prefix.
typedef enum { EMBER_THINK_NONE, EMBER_THINK_HIGH, EMBER_THINK_MAX } ember_think_mode;

typedef enum {
    EMBER_TOOL_CHOICE_AUTO,
    EMBER_TOOL_CHOICE_NONE,
    EMBER_TOOL_CHOICE_REQUIRED,
    EMBER_TOOL_CHOICE_NAMED,
    EMBER_TOOL_CHOICE_ALLOWED,
} ember_tool_choice_kind;

typedef struct {
    char            *role;      // "system" | "developer" | "user" | "assistant" | "tool"
    char            *content;   // flattened text (owned; "" if none)
    char            *name;      // tool name for role="tool" (owned; NULL otherwise)
    char            *reasoning; // assistant reasoning_content from history (owned; NULL)
    char            *tool_call_id; // B3: role="tool" result's tool_call_id (owned; NULL)
    char            *raw_tool_text; // B3: exact sampled DSML block for replay
                       // (owned; set by the tool-memory attach pass when this
                       // assistant turn's call ids resolve; reasoning/content
                       // remain canonical and only this replaces rendered calls)
    ember_tool_calls calls;     // assistant tool_calls parsed from history (may be empty)
} ember_chat_msg;

typedef struct {
    ember_api_kind  api;
    char           *model;
    char           *raw_prompt;   // legacy /v1/completions prompt, otherwise NULL
    ember_chat_msg *messages;
    int             n_messages;
    char           *tools_json;   // re-serialized tools array, or NULL
    bool            has_tools;
    bool            stream;
    bool            stream_include_usage;  // stream_options.include_usage
    // Responses reasoning summaries are opt-in (`reasoning.summary`). The
    // model may still think when this is false; only the wire representation
    // suppresses the summary item/events.
    bool            reasoning_summary_emit;
    bool            background;  // ember_background: defer until foreground idle
    int             max_tokens;   // value; see max_tokens_set
    bool            max_tokens_set;
    // Sampler surface (ds4 parity). *_set distinguishes client-supplied from default.
    double          temperature;  bool temperature_set;
    double          top_p;        bool top_p_set;
    int             top_k;        bool top_k_set;
    double          min_p;        bool min_p_set;
    uint64_t        seed;         bool seed_set;
    // Penalties. *_set preserves explicit zero overrides when model-card
    // defaults are nonzero.
    double          rep_pen;      bool rep_pen_set;
    int             rep_window;   bool rep_window_set;
    double          freq_pen;     bool freq_pen_set;
    double          pres_pen;     bool pres_pen_set;
    // DRY sequence-repetition penalty (llama.cpp-compatible names). Only
    // dry_multiplier arms it; the others refine an already active penalty.
    double          dry_multiplier;     bool dry_multiplier_set;
    double          dry_base;           bool dry_base_set;
    int             dry_allowed_length; bool dry_allowed_length_set;
    int             dry_window;         bool dry_window_set;
    char          **stop;         int  n_stop;   // stop strings (owned)
    // Thinking / reasoning-effort (resolved per ds4 semantics).
    char           *reasoning_effort;  // raw string (owned or NULL; kept for logs)
    // Explicit phase-1 cap. Accepts Ember's reasoning_budget_tokens and the
    // vLLM-compatible thinking_token_budget alias; omitted uses the model-card
    // effort tier. Zero requests an immediate transition to visible output.
    int             reasoning_budget_tokens;
    bool            reasoning_budget_tokens_set;
    ember_think_mode think_mode;       // resolved mode (valid when thinking_enabled)
    bool            thinking_enabled;  // default true (ds4); "none"/explicit-off → false
    // Tool-choice constraints are repeated in the model prompt and enforced at
    // the executable boundary. This is intentionally independent of whether a
    // backend offers grammar-constrained decoding.
    ember_tool_choice_kind tool_choice;
    bool            tool_choice_none;
    bool            tool_choice_required;
    char          **tool_choice_names;
    int             n_tool_choice_names;
    bool            parallel_tool_calls;  // default true
    // True when the request contains only tool results. Such a request has no
    // stateless prompt and must resolve through a protocol-bound continuation.
    bool            continuation_only;
    // Responses previous_response_id, when used for a tool-output continuation.
    char           *continuation_key;
    // Internal transport setting, populated by the HTTP handler after parsing.
    bool            response_cors;
} ember_chat_request;

// Parse from a JSON object. Returns false if `root` is not a valid request
// (missing messages array). On success, free with ember_chat_request_free.
bool ember_chat_request_parse(const ember_json *root, ember_chat_request *out);
// True only when this request is the immediate continuation of tool output.
// Historical tool messages earlier in a normal conversation do not constrain
// the decode strategy for every later user turn.
bool ember_chat_request_is_tool_result_continuation(
    const ember_chat_request *r);

// Count the complete trailing assistant-tool/result rounds whose ordered tool
// call sets (name + normalized arguments) and corresponding result bytes are
// identical. Call ids are deliberately ignored: clients mint a fresh id on
// every round. Arguments are normalized through ember_json_parse() followed by
// ember_json_dump(), which removes insignificant serialization whitespace but
// does NOT sort object keys. A lone round or a non-repeating tail returns 0.
//
// This is a pure full-history diagnostic, never a tool-round ceiling. It has no
// server-side session state, does not reject or stop a request, and deliberately
// does not cover continuation_only requests (which have no assistant history to
// inspect) or multi-round cycles such as A/B/A/B.
int ember_chat_request_tool_loop_rounds(const ember_chat_request *r);

// Count the trailing tool-CALLING assistant turns whose ordered call sets match
// the newest one by name + normalized arguments. Deliberately weaker than
// ember_chat_request_tool_loop_rounds(): result bytes are ignored entirely, and
// the scan looks THROUGH intervening tool results, assistant prose and user
// turns rather than stopping at the first one.
//
// Both relaxations come from a production loop the strict signal could not see
// (2026-08-08): the model re-emitted a byte-identical web_search call on eleven
// consecutive turns while the tool returned 112, 923, 26426 and 30033 bytes in
// turn, and a user message sat in the middle of the run. Requiring identical
// results, and refusing to look past a user turn, each independently reduced
// that to "no loop". Result-size changes are expected when an external agent
// compacts or truncates tool output between otherwise identical calls.
//
// Same contract as the strict signal: a pure diagnostic with no server-side
// state, which never rejects a request, never alters tool_calls/tool_use, and
// imposes no tool-round ceiling. A lone call or a non-repeating tail returns 0.
// It does not cover continuation_only requests or A/B/A/B cycles.
int ember_chat_request_tool_loop_calls(const ember_chat_request *r);

// First tool name in the newest complete trailing round, borrowed from r. This
// is intended only to label a nonzero tool-loop report.
const char *ember_chat_request_tool_loop_tool(const ember_chat_request *r);
void ember_chat_request_free(ember_chat_request *r);

#endif  // EMBER_CHAT_API_H
