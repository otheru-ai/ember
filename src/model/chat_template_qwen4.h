// Qwen3.8-Flash-Next chat-template rendering.
//
// Qwen4Exp uses ChatML plus Qwen XML-style tool calls.  It is deliberately
// separate from chat_template.h: DeepSeek's DSML markers are sampled special
// tokens and changing that renderer would invalidate exact-token replay and KV
// prefix keys for the release-default model.
#ifndef EMBER_CHAT_TEMPLATE_QWEN4_H
#define EMBER_CHAT_TEMPLATE_QWEN4_H

#include <stdbool.h>

#include "../server/chat_api.h"

// Render the official Qwen/Qwen3.8-Flash-Next template pinned at model
// revision f5d08274bafd880402bd16f5e3e6c514136ec06c.  Returns newly allocated
// text, or NULL when the normalized history cannot be represented by the
// official template contract.
char *ember_qwen4_render_prompt(const ember_chat_request *req,
                                bool enable_thinking,
                                bool add_generation_prompt);

// Render the suffix after an exact sampled Qwen assistant tool-call frontier.
// Only normalized tool-result messages are accepted.
char *ember_qwen4_render_tool_continuation_suffix(
    const ember_chat_request *req, bool enable_thinking);

#endif  // EMBER_CHAT_TEMPLATE_QWEN4_H
