#include "chat_template_qwen4.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "../common/buf.h"
#include "../common/json.h"

#define IM_START "<|im_start|>"
#define IM_END   "<|im_end|>"

static void append_trimmed(ember_buf *b, const char *text) {
    const unsigned char *start = (const unsigned char *)(text ? text : "");
    const unsigned char *end = start + strlen((const char *)start);
    while (start < end && isspace(*start)) ++start;
    while (end > start && isspace(end[-1])) --end;
    ember_buf_append(b, (const char *)start, (size_t)(end - start));
}

static bool is_role(const ember_chat_msg *msg, const char *role) {
    return msg && msg->role && strcmp(msg->role, role) == 0;
}

static bool append_message_content(ember_buf *out,
                                   const ember_chat_msg *msg,
                                   bool allow_images) {
    if (!msg || msg->n_parts <= 0) {
        append_trimmed(out, msg ? msg->content : "");
        return true;
    }
    ember_buf rendered = {0};
    for (int i = 0; i < msg->n_parts; ++i) {
        const ember_content_part *part = &msg->parts[i];
        if (part->kind == EMBER_CONTENT_TEXT) {
            ember_buf_puts(&rendered, part->text ? part->text : "");
        } else if (part->kind == EMBER_CONTENT_IMAGE_URL && allow_images) {
            // Pinned Qwen3.8 tokenizer_config.json render_content macro at
            // f5d08274bafd880402bd16f5e3e6c514136ec06c. The processor replaces
            // image_pad with the exact number of vision embeddings later.
            ember_buf_puts(&rendered,
                "<|vision_start|><|image_pad|><|vision_end|>");
        } else {
            ember_buf_free(&rendered);
            return false;
        }
    }
    append_trimmed(out, rendered.ptr ? rendered.ptr : "");
    ember_buf_free(&rendered);
    return true;
}

static const char *reasoning_instruction(const ember_chat_request *req,
                                         bool enable_thinking) {
    if (!enable_thinking) return "";
    const char *effort = req->reasoning_effort;
    if (!effort || !effort[0] || strcmp(effort, "xhigh") == 0) {
        return "Reasoning effort is set to xhigh. Please think carefully "
               "through the task, validate key assumptions, consider "
               "plausible alternatives, and prioritize correctness, "
               "consistency, and clarity in the final answer.";
    }
    if (strcmp(effort, "medium") == 0) return "";
    if (strcmp(effort, "low") == 0) {
        return "Reasoning effort is set to low. Keep your thinking brief and "
               "focused, moving directly to the conclusion without "
               "unnecessary elaboration.";
    }
    return NULL;
}

static bool append_tool_schemas(ember_buf *b, const char *tools_json) {
    ember_json *tools = ember_json_parse(tools_json ? tools_json : "");
    if (!tools || tools->type != EMBER_JSON_ARRAY) {
        ember_json_free(tools);
        return false;
    }
    for (int i = 0; i < ember_json_len(tools); ++i) {
        const ember_json *tool = ember_json_at(tools, i);
        if (!tool || tool->type != EMBER_JSON_OBJECT) {
            ember_json_free(tools);
            return false;
        }
        char *dump = ember_json_dump(tool);
        if (!dump) {
            ember_json_free(tools);
            return false;
        }
        ember_buf_putc(b, '\n');
        ember_buf_puts(b, dump);
        free(dump);
    }
    ember_json_free(tools);
    return true;
}

static bool append_qwen_tool_calls(ember_buf *b,
                                   const ember_tool_calls *calls,
                                   bool content_present) {
    for (int i = 0; calls && i < calls->len; ++i) {
        const ember_tool_call *call = &calls->calls[i];
        ember_json *args = ember_json_parse(call->arguments ? call->arguments : "");
        if (!call->name || !call->name[0] || !args ||
            args->type != EMBER_JSON_OBJECT) {
            ember_json_free(args);
            return false;
        }
        if (i > 0 || content_present) ember_buf_puts(b, "\n\n");
        ember_buf_puts(b, "<tool_call>\n<function=");
        ember_buf_puts(b, call->name);
        ember_buf_puts(b, ">\n");
        for (int j = 0; j < ember_json_len(args); ++j) {
            const char *key = ember_json_key(args, j);
            const ember_json *value = ember_json_at(args, j);
            if (!key || !key[0] || !value) {
                ember_json_free(args);
                return false;
            }
            ember_buf_puts(b, "<parameter=");
            ember_buf_puts(b, key);
            ember_buf_puts(b, ">\n");
            if (value->type == EMBER_JSON_STRING) {
                ember_buf_puts(b, ember_json_str(value, ""));
            } else {
                char *dump = ember_json_dump(value);
                if (!dump) {
                    ember_json_free(args);
                    return false;
                }
                ember_buf_puts(b, dump);
                free(dump);
            }
            ember_buf_puts(b, "\n</parameter>\n");
        }
        ember_buf_puts(b, "</function>\n</tool_call>");
        ember_json_free(args);
        content_present = true;
    }
    return true;
}

static bool append_system(ember_buf *b, const ember_chat_request *req,
                          const char *instruction) {
    const bool first_system = req->n_messages > 0 &&
                              is_role(&req->messages[0], "system");
    ember_buf sys = {0};
    if (instruction[0]) ember_buf_puts(&sys, instruction);

    if (req->has_tools) {
        if (sys.len) ember_buf_puts(&sys, "\n\n");
        ember_buf_puts(&sys,
            "# Tools\n\nYou have access to the following functions:\n\n<tools>");
        if (!append_tool_schemas(&sys, req->tools_json)) {
            ember_buf_free(&sys);
            return false;
        }
        ember_buf_puts(&sys,
            "\n</tools>\n\nIf you choose to call a function ONLY reply in "
            "the following format with NO suffix:\n\n<tool_call>\n"
            "<function=example_function_name>\n<parameter=example_parameter_1>\n"
            "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\nthat can span\n"
            "multiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
            "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
            "format: an inner <function=...></function> block must be nested "
            "within <tool_call></tool_call> XML tags\n- Required parameters "
            "MUST be specified\n- You may provide optional reasoning for your "
            "function call in natural language BEFORE the function call, but "
            "NOT after\n- If there is no function call available, answer the "
            "question like normal with your current knowledge and do not tell "
            "the user about function calls\n</IMPORTANT>");
    }

    if (first_system) {
        ember_buf content = {0};
        if (!append_message_content(&content, &req->messages[0], false)) {
            ember_buf_free(&content);
            ember_buf_free(&sys);
            return false;
        }
        if (content.len) {
            if (sys.len) ember_buf_puts(&sys, "\n\n");
            ember_buf_append(&sys, content.ptr, content.len);
        }
        ember_buf_free(&content);
    }

    if (sys.len) {
        ember_buf_puts(b, IM_START "system\n");
        ember_buf_append(b, sys.ptr, sys.len);
        ember_buf_puts(b, IM_END "\n");
    }
    ember_buf_free(&sys);
    return true;
}

char *ember_qwen4_render_prompt(const ember_chat_request *req,
                                bool enable_thinking,
                                bool add_generation_prompt) {
    if (!req || req->n_messages <= 0 || !req->messages) return NULL;
    const char *instruction = reasoning_instruction(req, enable_thinking);
    if (!instruction) return NULL;

    for (int i = 0; i < req->n_messages; ++i) {
        if (is_role(&req->messages[i], "system") && i != 0) return NULL;
        if (is_role(&req->messages[i], "developer")) return NULL;
    }

    ember_buf b = {0};
    if (!append_system(&b, req, instruction)) {
        ember_buf_free(&b);
        return NULL;
    }

    for (int i = 0; i < req->n_messages; ++i) {
        const ember_chat_msg *msg = &req->messages[i];
        if (is_role(msg, "system")) continue;
        if (is_role(msg, "user")) {
            ember_buf_puts(&b, IM_START "user\n");
            if (!append_message_content(&b, msg, true)) {
                ember_buf_free(&b);
                return NULL;
            }
            ember_buf_puts(&b, IM_END "\n");
        } else if (is_role(msg, "assistant")) {
            ember_buf visible = {0};
            if (!append_message_content(&visible, msg, true)) {
                ember_buf_free(&visible);
                ember_buf_free(&b);
                return NULL;
            }
            ember_buf reasoning = {0};
            append_trimmed(&reasoning, msg->reasoning);
            ember_buf_puts(&b, IM_START "assistant\n<think>\n");
            ember_buf_append(&b, reasoning.ptr ? reasoning.ptr : "", reasoning.len);
            ember_buf_puts(&b, "\n</think>\n\n");
            ember_buf_append(&b, visible.ptr ? visible.ptr : "", visible.len);
            if (!append_qwen_tool_calls(&b, &msg->calls, visible.len > 0)) {
                ember_buf_free(&visible);
                ember_buf_free(&reasoning);
                ember_buf_free(&b);
                return NULL;
            }
            ember_buf_free(&visible);
            ember_buf_free(&reasoning);
            ember_buf_puts(&b, IM_END "\n");
        } else if (is_role(msg, "tool") || is_role(msg, "function")) {
            const bool opens = i == 0 ||
                !(is_role(&req->messages[i - 1], "tool") ||
                  is_role(&req->messages[i - 1], "function"));
            const bool closes = i + 1 == req->n_messages ||
                !(is_role(&req->messages[i + 1], "tool") ||
                  is_role(&req->messages[i + 1], "function"));
            if (opens) ember_buf_puts(&b, IM_START "user");
            ember_buf_puts(&b, "\n<tool_response>\n");
            append_trimmed(&b, msg->content);
            ember_buf_puts(&b, "\n</tool_response>");
            if (closes) ember_buf_puts(&b, IM_END "\n");
        } else {
            ember_buf_free(&b);
            return NULL;
        }
    }

    if (add_generation_prompt) {
        ember_buf_puts(&b, IM_START "assistant\n<think>\n");
        if (!enable_thinking) ember_buf_puts(&b, "\n</think>\n\n");
    }

    char *rendered = ember_buf_take(&b);
    return rendered ? rendered : strdup("");
}

char *ember_qwen4_render_tool_continuation_suffix(
        const ember_chat_request *req, bool enable_thinking) {
    if (!req || req->n_messages <= 0 || !req->messages) return NULL;
    ember_buf b = {0};
    ember_buf_puts(&b, IM_END "\n" IM_START "user");
    for (int i = 0; i < req->n_messages; ++i) {
        const ember_chat_msg *msg = &req->messages[i];
        if (!is_role(msg, "tool") && !is_role(msg, "function")) {
            ember_buf_free(&b);
            return NULL;
        }
        ember_buf_puts(&b, "\n<tool_response>\n");
        append_trimmed(&b, msg->content);
        ember_buf_puts(&b, "\n</tool_response>");
    }
    ember_buf_puts(&b, IM_END "\n" IM_START "assistant\n<think>\n");
    if (!enable_thinking) ember_buf_puts(&b, "\n</think>\n\n");
    char *rendered = ember_buf_take(&b);
    return rendered ? rendered : strdup("");
}
