// Extract a normalized chat-completion request from a parsed JSON body.
// Handles content as a plain string OR as an array of {type:"text",text:...}
// parts (multimodal shape — text parts flattened, non-text ignored for now).
#ifndef EMBER_CHAT_API_H
#define EMBER_CHAT_API_H

#include <stdbool.h>
#include <stdint.h>
#include "../common/json.h"
#include "../model/tool_parser.h"   // ember_tool_calls (assistant history replay)

// Reasoning-effort → thinking mode, mirroring ds4's ds4_think_mode.
// NONE = thinking off; HIGH = default reasoning; MAX = maximum-effort prefix.
typedef enum { EMBER_THINK_NONE, EMBER_THINK_HIGH, EMBER_THINK_MAX } ember_think_mode;

typedef struct {
    char            *role;      // "system" | "developer" | "user" | "assistant" | "tool"
    char            *content;   // flattened text (owned; "" if none)
    char            *name;      // tool name for role="tool" (owned; NULL otherwise)
    char            *reasoning; // assistant reasoning_content from history (owned; NULL)
    char            *tool_call_id; // B3: role="tool" result's tool_call_id (owned; NULL)
    char            *raw_tool_text; // B3: exact sampled assistant-turn bytes for
                       // replay (owned; set by the tool-memory attach pass in
                       // run_chat when this assistant turn's call ids resolve;
                       // when set, chat_template emits it verbatim instead of the
                       // canonical reasoning+content+tool_calls render)
    ember_tool_calls calls;     // assistant tool_calls parsed from history (may be empty)
} ember_chat_msg;

typedef struct {
    char           *model;
    ember_chat_msg *messages;
    int             n_messages;
    char           *tools_json;   // re-serialized tools array, or NULL
    bool            has_tools;
    bool            stream;
    bool            stream_include_usage;  // stream_options.include_usage
    int             max_tokens;   // value; see max_tokens_set
    bool            max_tokens_set;
    // Sampler surface (ds4 parity). *_set distinguishes client-supplied from default.
    double          temperature;  bool temperature_set;
    double          top_p;        bool top_p_set;
    int             top_k;        bool top_k_set;
    double          min_p;        bool min_p_set;
    uint64_t        seed;         bool seed_set;
    // Penalties (default off: rep_pen 1.0, rep_window 0, freq/pres 0).
    double          rep_pen;      bool rep_pen_set;
    int             rep_window;   bool rep_window_set;
    double          freq_pen;     double pres_pen;
    char          **stop;         int  n_stop;   // stop strings (owned)
    // Thinking / reasoning-effort (resolved per ds4 semantics).
    char           *reasoning_effort;  // raw string (owned or NULL; kept for logs)
    ember_think_mode think_mode;       // resolved mode (valid when thinking_enabled)
    bool            thinking_enabled;  // default true (ds4); "none"/explicit-off → false
    // Tool choice: only "none" is materially handled (disables tools), per ds4.
    bool            tool_choice_none;
} ember_chat_request;

// Parse from a JSON object. Returns false if `root` is not a valid request
// (missing messages array). On success, free with ember_chat_request_free.
bool ember_chat_request_parse(const ember_json *root, ember_chat_request *out);
void ember_chat_request_free(ember_chat_request *r);

#endif  // EMBER_CHAT_API_H
