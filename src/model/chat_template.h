// DeepSeek-V4 DSML chat-template rendering: messages[] → prompt string.
//
// Renders the native DSML the model was trained on:
//   <｜begin▁of▁sentence｜>[tool preamble][system]<｜User｜>…<｜Assistant｜>…
//   <｜end▁of▁sentence｜> … <｜Assistant｜><think>|</think>
//
// The tool preamble is the piece whose absence made lucebox dump raw tool JSON
// and emit calls as prose; here it renders the DSML invoke schema the parser
// (tool_parser.c) reads back. Tool RESULTS render as <tool_result> under a
// <｜User｜> turn.
#ifndef EMBER_CHAT_TEMPLATE_H
#define EMBER_CHAT_TEMPLATE_H

#include <stdbool.h>
#include "../server/chat_api.h"

// Render `req` to a newly-allocated prompt string (caller frees).
//   enable_thinking      → close the prompt with <think> (model starts thinking)
//                          vs </think> (answer directly)
//   think_mode           → EMBER_THINK_MAX injects the max-effort prefix after BOS
//   add_generation_prompt→ append the trailing <｜Assistant｜> opener (gated on a
//                          pending assistant turn, per ds4)
char *ember_render_prompt(const ember_chat_request *req, bool enable_thinking,
                          ember_think_mode think_mode, bool add_generation_prompt);

// Render only the text appended after an exact sampled tool-call frontier:
// EOS + grouped tool results + the next assistant opener. Used for ID-bound
// continuations; callers tokenize this independently to avoid BPE boundary
// assumptions.
char *ember_render_tool_continuation_suffix(const ember_chat_request *req,
                                            bool enable_thinking);

// Render a model-visible recovery turn after malformed generated DSML. The
// invalid assistant turn remains in model context but is not returned to the
// client; this suffix asks the model to emit one corrected action.
char *ember_render_invalid_tool_recovery_suffix(
    const ember_chat_request *req, bool close_thinking, const char *detail);

// True if `prompt` ends inside an open <think> (ignoring trailing whitespace).
// Drives the streaming layer's started_in_thinking.
bool ember_prompt_ends_in_open_think(const char *prompt);

#endif  // EMBER_CHAT_TEMPLATE_H
