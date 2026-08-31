#include "api_adapters.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../common/buf.h"
#include "../common/json_util.h"
#include "http.h"

static void set_err(char *err, size_t cap, const char *msg) {
    if (err && cap) snprintf(err, cap, "%s", msg);
}

static void append_json_field(ember_buf *b, const char *name,
                              const ember_json *v, bool *comma) {
    if (!v) return;
    char *dump = ember_json_dump(v);
    if (!dump) return;
    if (*comma) ember_buf_putc(b, ',');
    ember_json_escape(b, name);
    ember_buf_putc(b, ':');
    ember_buf_puts(b, dump);
    *comma = true;
    free(dump);
}

static char *content_text(const ember_json *v) {
    if (!v) return strdup("");
    if (v->type == EMBER_JSON_STRING)
        return strdup(ember_json_str(v, ""));
    ember_buf b = {0};
    if (v->type == EMBER_JSON_ARRAY) {
        for (int i = 0; i < ember_json_len(v); ++i) {
            const ember_json *part = ember_json_at(v, i);
            if (!part || part->type != EMBER_JSON_OBJECT) continue;
            const char *type =
                ember_json_str(ember_json_get(part, "type"), "");
            if (strcmp(type, "text") && strcmp(type, "input_text") &&
                strcmp(type, "output_text") && strcmp(type, "summary_text"))
                continue;
            const char *text =
                ember_json_str(ember_json_get(part, "text"), "");
            ember_buf_puts(&b, text);
        }
    } else {
        char *dump = ember_json_dump(v);
        ember_buf_puts(&b, dump ? dump : "");
        free(dump);
    }
    char *out = ember_buf_take(&b);
    return out ? out : strdup("");
}

static bool responses_message_content_is_supported(const ember_json *v,
                                                    char *err,
                                                    size_t err_cap) {
    if (!v || v->type == EMBER_JSON_STRING) return true;
    if (v->type != EMBER_JSON_ARRAY) {
        set_err(err, err_cap,
                "Responses message content must be a string or block array");
        return false;
    }
    for (int i = 0; i < ember_json_len(v); ++i) {
        const ember_json *part = ember_json_at(v, i);
        const char *type = ember_json_str(
            ember_json_get(part, "type"), NULL);
        if (!part || part->type != EMBER_JSON_OBJECT || !type) {
            set_err(err, err_cap,
                    "Responses content blocks require a string type");
            return false;
        }
        if (!strcmp(type, "text") || !strcmp(type, "input_text") ||
            !strcmp(type, "output_text") || !strcmp(type, "summary_text"))
            continue;
        if (!strcmp(type, "input_image")) {
            const ember_json *image_url = ember_json_get(part, "image_url");
            if (image_url && (image_url->type == EMBER_JSON_STRING ||
                              image_url->type == EMBER_JSON_OBJECT))
                continue;
            set_err(err, err_cap,
                    "Responses input_image requires image_url");
            return false;
        }
        set_err(err, err_cap, "unsupported Responses message content block");
        return false;
    }
    return true;
}

static void message_begin(ember_buf *messages, bool *comma,
                          const char *role, const char *content) {
    if (*comma) ember_buf_putc(messages, ',');
    ember_buf_puts(messages, "{\"role\":");
    ember_json_escape(messages, role);
    ember_buf_puts(messages, ",\"content\":");
    ember_json_escape(messages, content ? content : "");
    *comma = true;
}

// Preserve Responses content order while translating its block names to the
// normalized Chat surface consumed by chat_api.c. Image URLs remain values only
// until that parser decodes and bounds the data URL into request-owned bytes.
static void responses_message_begin(ember_buf *messages, bool *comma,
                                    const char *role,
                                    const ember_json *content) {
    if (!content || content->type != EMBER_JSON_ARRAY) {
        message_begin(messages, comma, role,
                      ember_json_str(content, ""));
        return;
    }
    if (*comma) ember_buf_putc(messages, ',');
    ember_buf_puts(messages, "{\"role\":");
    ember_json_escape(messages, role);
    ember_buf_puts(messages, ",\"content\":[");
    bool part_comma = false;
    for (int i = 0; i < ember_json_len(content); ++i) {
        const ember_json *part = ember_json_at(content, i);
        const char *type = ember_json_str(ember_json_get(part, "type"), "");
        if (part_comma) ember_buf_putc(messages, ',');
        if (!strcmp(type, "input_image")) {
            char *url = ember_json_dump(ember_json_get(part, "image_url"));
            ember_buf_puts(messages,
                           "{\"type\":\"image_url\",\"image_url\":");
            ember_buf_puts(messages, url ? url : "null");
            ember_buf_putc(messages, '}');
            free(url);
        } else {
            ember_buf_puts(messages, "{\"type\":\"text\",\"text\":");
            ember_json_escape(messages,
                ember_json_str(ember_json_get(part, "text"), ""));
            ember_buf_putc(messages, '}');
        }
        part_comma = true;
    }
    ember_buf_putc(messages, ']');
    *comma = true;
}

static void message_end(ember_buf *messages) {
    ember_buf_putc(messages, '}');
}

static void append_common_fields(ember_buf *body, const ember_json *root,
                                 bool *comma, bool responses) {
    static const char *fields[] = {
        "model", "stream", "temperature", "top_p", "top_k", "seed",
        "min_p", "stop",
        "frequency_penalty", "presence_penalty", "repetition_penalty",
        "rep_window", "stream_options", "reasoning_budget_tokens",
        "thinking_token_budget", "thinking_budget_tokens",
        "parallel_tool_calls"
    };
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i)
        append_json_field(body, fields[i], ember_json_get(root, fields[i]), comma);
    const ember_json *max =
        ember_json_get(root, responses ? "max_output_tokens" : "max_tokens");
    if (max) append_json_field(body, "max_tokens", max, comma);
    const ember_json *background = ember_json_get(root, "background");
    if (background)
        append_json_field(body, "ember_background", background, comma);
}

// Responses tool lists may contain namespace groups discovered at runtime by
// tool_search. Ember's chat renderer consumes flat OpenAI function tools, so
// flatten a namespace by qualifying each child name exactly as Dwarfstar does.
static bool append_flat_response_tool(ember_buf *out, const ember_json *tool,
                                      const char *prefix, bool *comma,
                                      char *err, size_t err_cap) {
    if (!tool || tool->type != EMBER_JSON_OBJECT) {
        set_err(err, err_cap, "Responses tool entries must be objects");
        return false;
    }
    const char *type =
        ember_json_str(ember_json_get(tool, "type"), "function");
    if (!strcmp(type, "namespace")) {
        const char *name =
            ember_json_str(ember_json_get(tool, "name"), NULL);
        const ember_json *children = ember_json_get(tool, "tools");
        if (!name || !name[0] || !children ||
            children->type != EMBER_JSON_ARRAY) {
            set_err(err, err_cap,
                    "Responses namespace tools require name and tools array");
            return false;
        }
        size_t pn = prefix ? strlen(prefix) : 0;
        size_t nn = strlen(name);
        if (pn > SIZE_MAX - nn - 1) {
            set_err(err, err_cap, "qualified tool name is too long");
            return false;
        }
        char *qualified = malloc(pn + nn + 1);
        if (!qualified) {
            set_err(err, err_cap, "out of memory");
            return false;
        }
        if (pn) memcpy(qualified, prefix, pn);
        memcpy(qualified + pn, name, nn + 1);
        for (int i = 0; i < ember_json_len(children); ++i) {
            if (!append_flat_response_tool(
                    out, ember_json_at(children, i), qualified, comma,
                    err, err_cap)) {
                free(qualified);
                return false;
            }
        }
        free(qualified);
        return true;
    }
    if (strcmp(type, "function") && strcmp(type, "custom")) {
        if (err && err_cap)
            snprintf(err, err_cap,
                     "Responses tool type %s is not supported", type);
        return false;
    }
    const char *name =
        ember_json_str(ember_json_get(tool, "name"), NULL);
    if (!name || !name[0]) {
        set_err(err, err_cap, "Responses function tool requires a name");
        return false;
    }
    size_t pn = prefix ? strlen(prefix) : 0;
    size_t nn = strlen(name);
    if (pn > SIZE_MAX - nn - 1) {
        set_err(err, err_cap, "qualified tool name is too long");
        return false;
    }
    char *qualified = malloc(pn + nn + 1);
    if (!qualified) {
        set_err(err, err_cap, "out of memory");
        return false;
    }
    if (pn) memcpy(qualified, prefix, pn);
    memcpy(qualified + pn, name, nn + 1);

    if (*comma) ember_buf_putc(out, ',');
    ember_buf_puts(out, "{\"type\":\"function\",\"function\":{\"name\":");
    ember_json_escape(out, qualified);
    bool fc = true;
    append_json_field(out, "description",
                      ember_json_get(tool, "description"), &fc);
    const ember_json *params = ember_json_get(tool, "parameters");
    if (!params) params = ember_json_get(tool, "input_schema");
    if (params && params->type != EMBER_JSON_OBJECT) {
        free(qualified);
        set_err(err, err_cap,
                "Responses function tool parameters must be an object");
        return false;
    }
    append_json_field(out, "parameters", params, &fc);
    append_json_field(out, "strict", ember_json_get(tool, "strict"), &fc);
    ember_buf_puts(out, "}}");
    *comma = true;
    free(qualified);
    return true;
}

static bool is_response_call(const ember_json *item) {
    const char *t = ember_json_str(ember_json_get(item, "type"), "");
    return !strcmp(t, "function_call") ||
           !strcmp(t, "custom_tool_call") ||
           !strcmp(t, "hosted_tool_call") ||
           !strcmp(t, "local_shell_call") ||
           !strcmp(t, "web_search_call") ||
           !strcmp(t, "tool_search_call") ||
           !strcmp(t, "image_generation_call");
}

static bool is_response_output(const ember_json *item) {
    const char *t = ember_json_str(ember_json_get(item, "type"), "");
    return !strcmp(t, "function_call_output") ||
           !strcmp(t, "custom_tool_call_output") ||
           !strcmp(t, "hosted_tool_call_output") ||
           !strcmp(t, "local_shell_call_output") ||
           !strcmp(t, "web_search_call_output") ||
           !strcmp(t, "tool_search_output") ||
           !strcmp(t, "tool_search_call_output") ||
           !strcmp(t, "image_generation_call_output");
}

static bool append_response_call(ember_buf *m, const ember_json *item,
                                 int idx, char *err, size_t err_cap) {
    if (idx) ember_buf_putc(m, ',');
    const char *id = ember_json_str(ember_json_get(item, "call_id"), NULL);
    if (!id) id = ember_json_str(ember_json_get(item, "id"), "");
    const char *type = ember_json_str(ember_json_get(item, "type"), "");
    const char *wire_name = ember_json_str(ember_json_get(item, "name"), "");
    const char *ns =
        ember_json_str(ember_json_get(item, "namespace"), "");
    const char *name = wire_name;
    if (!strcmp(type, "tool_search_call"))
        name = "tool_search";
    else if (!strcmp(type, "local_shell_call"))
        name = "local_shell";
    else if (!strcmp(type, "web_search_call") ||
             !strcmp(type, "image_generation_call"))
        name = type;
    size_t nsn = ns ? strlen(ns) : 0;
    size_t namen = name ? strlen(name) : 0;
    if (!id[0] || !namen) {
        set_err(err, err_cap,
                "Responses tool calls require call_id and name");
        return false;
    }
    char *qualified = NULL;
    if (nsn && namen) {
        if (nsn > SIZE_MAX - namen - 1) {
            set_err(err, err_cap, "qualified call name is too long");
            return false;
        }
        qualified = malloc(nsn + namen + 1);
        if (!qualified) {
            set_err(err, err_cap, "out of memory");
            return false;
        }
        memcpy(qualified, ns, nsn);
        memcpy(qualified + nsn, name, namen + 1);
        name = qualified;
    }
    const ember_json *av = ember_json_get(item, "action");
    if (!av) av = ember_json_get(item, "arguments");
    if (!av) av = ember_json_get(item, "input");
    char *args = NULL;
    if (av && av->type == EMBER_JSON_STRING)
        args = strdup(ember_json_str(av, "{}"));
    else
        args = av ? ember_json_dump(av) : strdup("{}");
    if (!args) {
        free(qualified);
        set_err(err, err_cap, "out of memory");
        return false;
    }
    ember_buf_puts(m, "{\"id\":");
    ember_json_escape(m, id);
    ember_buf_puts(m, ",\"type\":\"function\",\"function\":{\"name\":");
    ember_json_escape(m, name);
    ember_buf_puts(m, ",\"arguments\":");
    ember_json_escape(m, args ? args : "{}");
    ember_buf_puts(m, "}}");
    free(qualified);
    free(args);
    return true;
}

static char *reasoning_summary(const ember_json *item) {
    const ember_json *summary = ember_json_get(item, "summary");
    return content_text(summary);
}

static void flush_pending_reasoning(ember_buf *messages, bool *comma,
                                    char **pending_reasoning) {
    if (!pending_reasoning || !*pending_reasoning ||
        !(*pending_reasoning)[0]) return;
    message_begin(messages, comma, "assistant", "");
    ember_buf_puts(messages, ",\"reasoning_content\":");
    ember_json_escape(messages, *pending_reasoning);
    message_end(messages);
    free(*pending_reasoning);
    *pending_reasoning = NULL;
}

static bool response_item_status_complete(const ember_json *item,
                                          char *err, size_t err_cap) {
    const ember_json *status = ember_json_get(item, "status");
    if (!status) return true;
    if (status->type != EMBER_JSON_STRING ||
        strcmp(ember_json_str(status, ""), "completed")) {
        set_err(err, err_cap,
                "Responses replay items must have completed status");
        return false;
    }
    return true;
}

static bool build_responses_messages(const ember_json *input,
                                     ember_buf *messages,
                                     ember_buf *discovered_tools,
                                     bool *discovered_comma,
                                     char *err, size_t err_cap) {
    bool comma = false;
    if (!input) return false;
    if (input->type == EMBER_JSON_STRING) {
        message_begin(messages, &comma, "user", ember_json_str(input, ""));
        message_end(messages);
        return true;
    }
    if (input->type != EMBER_JSON_ARRAY) return false;
    char *pending_reasoning = NULL;
    for (int i = 0; i < ember_json_len(input); ++i) {
        const ember_json *item = ember_json_at(input, i);
        if (!item || item->type != EMBER_JSON_OBJECT) {
            set_err(err, err_cap,
                    "Responses input array entries must be objects");
            free(pending_reasoning);
            return false;
        }
        if (!response_item_status_complete(item, err, err_cap)) {
            free(pending_reasoning);
            return false;
        }
        const char *type = ember_json_str(ember_json_get(item, "type"), "");
        if (!strcmp(type, "reasoning")) {
            char *part = reasoning_summary(item);
            const char *content =
                ember_json_str(ember_json_get(item, "content"), "");
            ember_buf combined = {0};
            if (pending_reasoning && pending_reasoning[0])
                ember_buf_puts(&combined, pending_reasoning);
            if (part && part[0]) {
                if (combined.len) ember_buf_putc(&combined, '\n');
                ember_buf_puts(&combined, part);
            }
            if (content && content[0]) {
                if (combined.len) ember_buf_putc(&combined, '\n');
                ember_buf_puts(&combined, content);
            }
            free(part);
            free(pending_reasoning);
            pending_reasoning = ember_buf_take(&combined);
            continue;
        }
        if (is_response_call(item)) {
            message_begin(messages, &comma, "assistant", "");
            if (pending_reasoning && pending_reasoning[0]) {
                ember_buf_puts(messages, ",\"reasoning_content\":");
                ember_json_escape(messages, pending_reasoning);
            }
            free(pending_reasoning);
            pending_reasoning = NULL;
            ember_buf_puts(messages, ",\"tool_calls\":[");
            int calls = 0;
            while (i < ember_json_len(input)) {
                const ember_json *call = ember_json_at(input, i);
                if (!call || !is_response_call(call)) break;
                if (!append_response_call(
                        messages, call, calls++, err, err_cap)) {
                    free(pending_reasoning);
                    return false;
                }
                i++;
            }
            i--;
            ember_buf_puts(messages, "]}");
            continue;
        }
        if (is_response_output(item)) {
            flush_pending_reasoning(messages, &comma, &pending_reasoning);
            const char *id =
                ember_json_str(ember_json_get(item, "call_id"), "");
            if (!id[0])
                id = ember_json_str(ember_json_get(item, "id"), "");
            const ember_json *body = ember_json_get(item, "output");
            if (!body) body = ember_json_get(item, "result");
            if (!body && !strcmp(type, "tool_search_output"))
                body = ember_json_get(item, "tools");
            char *text = NULL;
            if (!strcmp(type, "tool_search_output") && body)
                text = ember_json_dump(body);
            else
                text = content_text(body);
            if (!text) text = strdup("");
            message_begin(messages, &comma, "tool", text);
            ember_buf_puts(messages, ",\"tool_call_id\":");
            ember_json_escape(messages, id);
            message_end(messages);
            free(text);
            if (!strcmp(type, "tool_search_output")) {
                const ember_json *tools = ember_json_get(item, "tools");
                if (!tools || tools->type != EMBER_JSON_ARRAY) {
                    set_err(err, err_cap,
                            "tool_search_output tools must be an array");
                    free(pending_reasoning);
                    return false;
                }
                for (int ti = 0; ti < ember_json_len(tools); ++ti) {
                    if (!append_flat_response_tool(
                            discovered_tools, ember_json_at(tools, ti), "",
                            discovered_comma, err, err_cap)) {
                        free(pending_reasoning);
                        return false;
                    }
                }
            }
            continue;
        }
        if (!strcmp(type, "message") || !type[0]) {
            const char *role =
                ember_json_str(ember_json_get(item, "role"), "user");
            if (strcmp(role, "assistant"))
                flush_pending_reasoning(
                    messages, &comma, &pending_reasoning);
            const ember_json *message_content = ember_json_get(item, "content");
            if (!responses_message_content_is_supported(
                    message_content, err, err_cap)) {
                free(pending_reasoning);
                return false;
            }
            responses_message_begin(messages, &comma, role, message_content);
            if (!strcmp(role, "assistant") &&
                pending_reasoning && pending_reasoning[0]) {
                ember_buf_puts(messages, ",\"reasoning_content\":");
                ember_json_escape(messages, pending_reasoning);
            }
            message_end(messages);
            free(pending_reasoning);
            pending_reasoning = NULL;
            continue;
        }
        if (!strcmp(type, "compaction") ||
            !strcmp(type, "context_compaction"))
            continue;
        if (err && err_cap)
            snprintf(err, err_cap,
                     "unsupported Responses input item type: %s",
                     type[0] ? type : "(missing)");
        free(pending_reasoning);
        return false;
    }
    flush_pending_reasoning(messages, &comma, &pending_reasoning);
    return true;
}

static bool append_responses_tools(ember_buf *body, const ember_json *tools,
                                   const ember_buf *discovered,
                                   bool *comma, char *err, size_t err_cap) {
    if (tools && tools->type != EMBER_JSON_ARRAY) {
        set_err(err, err_cap, "Responses tools must be an array");
        return false;
    }
    if ((!tools || ember_json_len(tools) == 0) &&
        (!discovered || discovered->len == 0))
        return true;
    if (*comma) ember_buf_putc(body, ',');
    ember_buf_puts(body, "\"tools\":[");
    bool tc = false;
    for (int i = 0; i < ember_json_len(tools); ++i) {
        if (!append_flat_response_tool(
                body, ember_json_at(tools, i), "", &tc, err, err_cap))
            return false;
    }
    if (discovered && discovered->len) {
        if (tc) ember_buf_putc(body, ',');
        ember_buf_append(body, discovered->ptr, discovered->len);
    }
    ember_buf_putc(body, ']');
    *comma = true;
    return true;
}

bool ember_responses_request_parse(const ember_json *root,
                                   ember_chat_request *out,
                                   char *err, size_t err_cap) {
    if (!root || root->type != EMBER_JSON_OBJECT ||
        ember_json_has_duplicate_keys(root)) {
        set_err(err, err_cap, "invalid Responses request");
        return false;
    }
    const ember_json *previous = ember_json_get(root, "previous_response_id");
    const ember_json *conversation = ember_json_get(root, "conversation");
    if (conversation && conversation->type != EMBER_JSON_NULL) {
        set_err(err, err_cap,
                "conversation is not supported; replay full input");
        return false;
    }
    if (previous && previous->type != EMBER_JSON_NULL &&
        previous->type != EMBER_JSON_STRING) {
        set_err(err, err_cap, "previous_response_id must be a string");
        return false;
    }
    const ember_json *instructions = ember_json_get(root, "instructions");
    if (instructions && instructions->type != EMBER_JSON_NULL &&
        instructions->type != EMBER_JSON_STRING) {
        set_err(err, err_cap, "Responses instructions must be a string");
        return false;
    }
    const ember_json *tool_choice = ember_json_get(root, "tool_choice");
    if (tool_choice && tool_choice->type != EMBER_JSON_NULL) {
        if (tool_choice->type != EMBER_JSON_STRING &&
            tool_choice->type != EMBER_JSON_OBJECT) {
            set_err(err, err_cap, "invalid Responses tool_choice");
            return false;
        }
    }
    const ember_json *reasoning = ember_json_get(root, "reasoning");
    if (reasoning && reasoning->type != EMBER_JSON_NULL &&
        reasoning->type != EMBER_JSON_OBJECT) {
        set_err(err, err_cap, "Responses reasoning must be an object");
        return false;
    }
    if (reasoning && reasoning->type == EMBER_JSON_OBJECT) {
        const ember_json *effort = ember_json_get(reasoning, "effort");
        const ember_json *summary = ember_json_get(reasoning, "summary");
        if ((effort && effort->type != EMBER_JSON_STRING) ||
            (summary && summary->type != EMBER_JSON_STRING)) {
            set_err(err, err_cap, "Responses reasoning fields must be strings");
            return false;
        }
    }
    ember_buf messages = {0};
    bool mc = false;
    if (instructions && instructions->type == EMBER_JSON_STRING &&
        ember_json_str(instructions, "")[0]) {
        message_begin(&messages, &mc, "system",
                      ember_json_str(instructions, ""));
        message_end(&messages);
    }
    ember_buf input_messages = {0};
    ember_buf discovered_tools = {0};
    bool discovered_comma = false;
    if (!build_responses_messages(
            ember_json_get(root, "input"), &input_messages,
            &discovered_tools, &discovered_comma, err, err_cap)) {
        ember_buf_free(&messages);
        ember_buf_free(&input_messages);
        ember_buf_free(&discovered_tools);
        if (!err || !err[0])
            set_err(err, err_cap,
                    "Responses input must be a string or item array");
        return false;
    }
    if (input_messages.len) {
        if (mc) ember_buf_putc(&messages, ',');
        ember_buf_append(&messages, input_messages.ptr, input_messages.len);
    }
    ember_buf_free(&input_messages);
    ember_buf body = {0};
    ember_buf_putc(&body, '{');
    bool comma = false;
    append_common_fields(&body, root, &comma, true);
    append_json_field(&body, "tool_choice", tool_choice, &comma);
    if (reasoning && reasoning->type == EMBER_JSON_OBJECT) {
        const ember_json *effort = ember_json_get(reasoning, "effort");
        append_json_field(&body, "reasoning_effort", effort, &comma);
    }
    if (!append_responses_tools(
            &body, ember_json_get(root, "tools"), &discovered_tools,
            &comma, err, err_cap)) {
        ember_buf_free(&messages);
        ember_buf_free(&discovered_tools);
        ember_buf_free(&body);
        return false;
    }
    ember_buf_free(&discovered_tools);
    if (comma) ember_buf_putc(&body, ',');
    ember_buf_puts(&body, "\"messages\":[");
    ember_buf_append(&body, messages.ptr ? messages.ptr : "", messages.len);
    ember_buf_puts(&body, "]}");
    ember_buf_free(&messages);
    ember_json *normalized = ember_json_parse(body.ptr);
    ember_buf_free(&body);
    bool ok = normalized && ember_chat_request_parse(normalized, out);
    if (normalized) ember_json_free(normalized);
    if (!ok) {
        set_err(err, err_cap, "invalid normalized Responses input");
        return false;
    }
    out->api = EMBER_API_RESPONSES;
    if (reasoning && reasoning->type == EMBER_JSON_OBJECT) {
        const ember_json *summary = ember_json_get(reasoning, "summary");
        if (summary && summary->type == EMBER_JSON_STRING) {
            const char *mode = ember_json_str(summary, "");
            out->reasoning_summary_emit =
                !strcmp(mode, "auto") ||
                !strcmp(mode, "concise") ||
                !strcmp(mode, "detailed");
        }
    }
    if (previous && previous->type == EMBER_JSON_STRING) {
        if (!out->continuation_only) {
            ember_chat_request_free(out);
            set_err(err, err_cap,
                    "previous_response_id currently requires tool output input");
            return false;
        }
        out->continuation_key =
            strdup(ember_json_str(previous, ""));
        if (!out->continuation_key || !out->continuation_key[0]) {
            ember_chat_request_free(out);
            set_err(err, err_cap,
                    "previous_response_id must not be empty");
            return false;
        }
    }
    return true;
}

bool ember_completion_request_parse(const ember_json *root,
                                    ember_chat_request *out,
                                    char *err, size_t err_cap) {
    if (!root || root->type != EMBER_JSON_OBJECT ||
        ember_json_has_duplicate_keys(root)) {
        set_err(err, err_cap, "invalid Completions request");
        return false;
    }
    const ember_json *prompt = ember_json_get(root, "prompt");
    const char *text = NULL;
    if (prompt && prompt->type == EMBER_JSON_STRING) {
        text = ember_json_str(prompt, "");
    } else if (prompt && prompt->type == EMBER_JSON_ARRAY &&
               ember_json_len(prompt) == 1) {
        const ember_json *first = ember_json_at(prompt, 0);
        if (first && first->type == EMBER_JSON_STRING)
            text = ember_json_str(first, "");
    }
    if (!text) {
        set_err(err, err_cap,
                "prompt must be a string or a single-element string array");
        return false;
    }

    ember_buf body = {0};
    ember_buf_putc(&body, '{');
    bool comma = false;
    append_common_fields(&body, root, &comma, false);
    if (comma) ember_buf_putc(&body, ',');
    ember_buf_puts(&body,
        "\"messages\":[{\"role\":\"user\",\"content\":\"\"}]}");
    ember_json *normalized = ember_json_parse(body.ptr);
    ember_buf_free(&body);
    bool ok = normalized && ember_chat_request_parse(normalized, out);
    if (normalized) ember_json_free(normalized);
    if (!ok) {
        set_err(err, err_cap, "invalid normalized Completions input");
        return false;
    }
    out->raw_prompt = strdup(text);
    if (!out->raw_prompt) {
        ember_chat_request_free(out);
        set_err(err, err_cap, "out of memory");
        return false;
    }
    out->api = EMBER_API_COMPLETIONS;
    out->thinking_enabled = false;
    out->think_mode = EMBER_THINK_NONE;
    return true;
}

static void append_anthropic_tools(ember_buf *body, const ember_json *tools,
                                   bool *comma) {
    if (!tools || tools->type != EMBER_JSON_ARRAY ||
        ember_json_len(tools) == 0) return;
    if (*comma) ember_buf_putc(body, ',');
    ember_buf_puts(body, "\"tools\":[");
    for (int i = 0; i < ember_json_len(tools); ++i) {
        if (i) ember_buf_putc(body, ',');
        const ember_json *t = ember_json_at(tools, i);
        ember_buf_puts(body, "{\"type\":\"function\",\"function\":{");
        bool fc = false;
        append_json_field(body, "name", ember_json_get(t, "name"), &fc);
        append_json_field(body, "description",
                          ember_json_get(t, "description"), &fc);
        append_json_field(body, "parameters",
                          ember_json_get(t, "input_schema"), &fc);
        append_json_field(body, "strict", ember_json_get(t, "strict"), &fc);
        ember_buf_puts(body, "}}");
    }
    ember_buf_putc(body, ']');
    *comma = true;
}

static void append_anthropic_message(ember_buf *messages, bool *comma,
                                     const ember_json *m) {
    const char *role = ember_json_str(ember_json_get(m, "role"), "user");
    const ember_json *content = ember_json_get(m, "content");
    if (!content || content->type == EMBER_JSON_STRING) {
        message_begin(messages, comma, role,
                      content ? ember_json_str(content, "") : "");
        message_end(messages);
        return;
    }
    if (content->type != EMBER_JSON_ARRAY) return;
    if (!strcmp(role, "assistant")) {
        ember_buf text = {0}, reasoning = {0};
        int n_calls = 0;
        for (int i = 0; i < ember_json_len(content); ++i) {
            const ember_json *p = ember_json_at(content, i);
            const char *t = ember_json_str(ember_json_get(p, "type"), "");
            if (!strcmp(t, "text"))
                ember_buf_puts(&text,
                    ember_json_str(ember_json_get(p, "text"), ""));
            else if (!strcmp(t, "thinking"))
                ember_buf_puts(&reasoning,
                    ember_json_str(ember_json_get(p, "thinking"), ""));
            else if (!strcmp(t, "tool_use")) n_calls++;
        }
        message_begin(messages, comma, "assistant", text.ptr ? text.ptr : "");
        if (reasoning.len) {
            ember_buf_puts(messages, ",\"reasoning_content\":");
            ember_json_escape_n(messages, reasoning.ptr, reasoning.len);
        }
        if (n_calls) {
            ember_buf_puts(messages, ",\"tool_calls\":[");
            int ci = 0;
            for (int i = 0; i < ember_json_len(content); ++i) {
                const ember_json *p = ember_json_at(content, i);
                if (strcmp(ember_json_str(ember_json_get(p, "type"), ""),
                           "tool_use")) continue;
                if (ci++) ember_buf_putc(messages, ',');
                const char *id =
                    ember_json_str(ember_json_get(p, "id"), "");
                const char *name =
                    ember_json_str(ember_json_get(p, "name"), "");
                char *args = ember_json_dump(ember_json_get(p, "input"));
                ember_buf_puts(messages, "{\"id\":");
                ember_json_escape(messages, id);
                ember_buf_puts(messages,
                    ",\"type\":\"function\",\"function\":{\"name\":");
                ember_json_escape(messages, name);
                ember_buf_puts(messages, ",\"arguments\":");
                ember_json_escape(messages, args ? args : "{}");
                ember_buf_puts(messages, "}}");
                free(args);
            }
            ember_buf_putc(messages, ']');
        }
        message_end(messages);
        ember_buf_free(&text);
        ember_buf_free(&reasoning);
        return;
    }

    ember_buf parts = {0};
    bool parts_comma = false;
    for (int i = 0; i < ember_json_len(content); ++i) {
        const ember_json *p = ember_json_at(content, i);
        const char *t = ember_json_str(ember_json_get(p, "type"), "");
        if (!strcmp(t, "text")) {
            if (parts_comma) ember_buf_putc(&parts, ',');
            ember_buf_puts(&parts, "{\"type\":\"text\",\"text\":");
            ember_json_escape(&parts,
                ember_json_str(ember_json_get(p, "text"), ""));
            ember_buf_putc(&parts, '}');
            parts_comma = true;
        } else if (!strcmp(t, "image")) {
            const ember_json *source = ember_json_get(p, "source");
            const char *media_type = ember_json_str(
                ember_json_get(source, "media_type"), "");
            const char *data = ember_json_str(
                ember_json_get(source, "data"), "");
            if (parts_comma) ember_buf_putc(&parts, ',');
            ember_buf_puts(&parts,
                "{\"type\":\"image_url\",\"image_url\":\"data:");
            ember_buf_puts(&parts, media_type);
            ember_buf_puts(&parts, ";base64,");
            ember_buf_puts(&parts, data);
            ember_buf_puts(&parts, "\"}");
            parts_comma = true;
        } else if (!strcmp(t, "tool_result")) {
            if (parts_comma) {
                if (*comma) ember_buf_putc(messages, ',');
                ember_buf_puts(messages,
                    "{\"role\":\"user\",\"content\":[");
                ember_buf_append(messages, parts.ptr, parts.len);
                ember_buf_puts(messages, "]}");
                *comma = true;
                parts.len = 0;
                if (parts.ptr) parts.ptr[0] = '\0';
                parts_comma = false;
            }
            char *result = content_text(ember_json_get(p, "content"));
            message_begin(messages, comma, "tool", result);
            ember_buf_puts(messages, ",\"tool_call_id\":");
            ember_json_escape(messages,
                ember_json_str(ember_json_get(p, "tool_use_id"), ""));
            message_end(messages);
            free(result);
        }
    }
    if (parts_comma) {
        if (*comma) ember_buf_putc(messages, ',');
        ember_buf_puts(messages, "{\"role\":\"user\",\"content\":[");
        ember_buf_append(messages, parts.ptr, parts.len);
        ember_buf_puts(messages, "]}");
        *comma = true;
    }
    ember_buf_free(&parts);
}

static bool validate_anthropic_text_blocks(const ember_json *value,
                                           char *err, size_t err_cap) {
    if (!value || value->type == EMBER_JSON_STRING) return true;
    if (value->type != EMBER_JSON_ARRAY) {
        set_err(err, err_cap,
                "Anthropic text content must be a string or block array");
        return false;
    }
    for (int i = 0; i < ember_json_len(value); ++i) {
        const ember_json *part = ember_json_at(value, i);
        const ember_json *type = ember_json_get(part, "type");
        const ember_json *text = ember_json_get(part, "text");
        if (!part || part->type != EMBER_JSON_OBJECT ||
            !type || type->type != EMBER_JSON_STRING ||
            strcmp(ember_json_str(type, ""), "text") ||
            !text || text->type != EMBER_JSON_STRING) {
            set_err(err, err_cap,
                    "Anthropic text blocks require type=text and string text");
            return false;
        }
    }
    return true;
}

static bool validate_anthropic_payload(const ember_json *root,
                                       char *err, size_t err_cap) {
    const ember_json *system = ember_json_get(root, "system");
    if (system && !validate_anthropic_text_blocks(system, err, err_cap))
        return false;

    const ember_json *tools = ember_json_get(root, "tools");
    if (tools) {
        if (tools->type != EMBER_JSON_ARRAY) {
            set_err(err, err_cap, "Anthropic tools must be an array");
            return false;
        }
        for (int i = 0; i < ember_json_len(tools); ++i) {
            const ember_json *tool = ember_json_at(tools, i);
            const ember_json *name = ember_json_get(tool, "name");
            const ember_json *schema = ember_json_get(tool, "input_schema");
            if (!tool || tool->type != EMBER_JSON_OBJECT ||
                !name || name->type != EMBER_JSON_STRING ||
                !ember_json_str(name, "")[0] ||
                !schema || schema->type != EMBER_JSON_OBJECT) {
                set_err(err, err_cap,
                        "Anthropic tools require name and input_schema object");
                return false;
            }
        }
    }

    const ember_json *messages = ember_json_get(root, "messages");
    for (int mi = 0; mi < ember_json_len(messages); ++mi) {
        const ember_json *message = ember_json_at(messages, mi);
        const ember_json *role_value = ember_json_get(message, "role");
        const ember_json *content = ember_json_get(message, "content");
        const char *role = ember_json_str(role_value, "");
        if (!message || message->type != EMBER_JSON_OBJECT ||
            !role_value || role_value->type != EMBER_JSON_STRING ||
            (strcmp(role, "user") && strcmp(role, "assistant")) ||
            !content) {
            set_err(err, err_cap,
                    "Anthropic messages require user/assistant role and content");
            return false;
        }
        if (content->type == EMBER_JSON_STRING) continue;
        if (content->type != EMBER_JSON_ARRAY) {
            set_err(err, err_cap,
                    "Anthropic message content must be a string or array");
            return false;
        }
        for (int bi = 0; bi < ember_json_len(content); ++bi) {
            const ember_json *block = ember_json_at(content, bi);
            const ember_json *type_value = ember_json_get(block, "type");
            const char *type = ember_json_str(type_value, "");
            if (!block || block->type != EMBER_JSON_OBJECT ||
                !type_value || type_value->type != EMBER_JSON_STRING) {
                set_err(err, err_cap,
                        "Anthropic content blocks require a string type");
                return false;
            }
            if (!strcmp(type, "text")) {
                const ember_json *text = ember_json_get(block, "text");
                if (!text || text->type != EMBER_JSON_STRING) {
                    set_err(err, err_cap,
                            "Anthropic text blocks require string text");
                    return false;
                }
                continue;
            }
            if (!strcmp(role, "assistant") && !strcmp(type, "thinking")) {
                const ember_json *thinking = ember_json_get(block, "thinking");
                if (!thinking || thinking->type != EMBER_JSON_STRING) {
                    set_err(err, err_cap,
                            "Anthropic thinking blocks require string thinking");
                    return false;
                }
                continue;
            }
            if (!strcmp(role, "assistant") && !strcmp(type, "tool_use")) {
                const ember_json *id = ember_json_get(block, "id");
                const ember_json *name = ember_json_get(block, "name");
                const ember_json *input = ember_json_get(block, "input");
                if (!id || id->type != EMBER_JSON_STRING ||
                    !ember_json_str(id, "")[0] ||
                    !name || name->type != EMBER_JSON_STRING ||
                    !ember_json_str(name, "")[0] ||
                    !input || input->type != EMBER_JSON_OBJECT) {
                    set_err(err, err_cap,
                            "Anthropic tool_use requires id, name, and object input");
                    return false;
                }
                continue;
            }
            if (!strcmp(role, "user") && !strcmp(type, "image")) {
                const ember_json *source = ember_json_get(block, "source");
                const char *source_type = ember_json_str(
                    ember_json_get(source, "type"), "");
                const char *media_type = ember_json_str(
                    ember_json_get(source, "media_type"), "");
                const ember_json *data = ember_json_get(source, "data");
                const bool media_ok = !strcmp(media_type, "image/png") ||
                    !strcmp(media_type, "image/jpeg") ||
                    !strcmp(media_type, "image/webp") ||
                    !strcmp(media_type, "image/gif");
                if (!source || source->type != EMBER_JSON_OBJECT ||
                    strcmp(source_type, "base64") || !media_ok ||
                    !data || data->type != EMBER_JSON_STRING ||
                    !ember_json_str(data, "")[0]) {
                    set_err(err, err_cap,
                            "Anthropic image blocks require a supported base64 source");
                    return false;
                }
                continue;
            }
            if (!strcmp(role, "user") && !strcmp(type, "tool_result")) {
                const ember_json *id = ember_json_get(block, "tool_use_id");
                if (!id || id->type != EMBER_JSON_STRING ||
                    !ember_json_str(id, "")[0] ||
                    !validate_anthropic_text_blocks(
                        ember_json_get(block, "content"), err, err_cap)) {
                    if (!err || !err[0])
                        set_err(err, err_cap,
                                "Anthropic tool_result requires tool_use_id");
                    return false;
                }
                continue;
            }
            set_err(err, err_cap,
                    "unsupported Anthropic content block for message role");
            return false;
        }
    }
    return true;
}

bool ember_anthropic_request_parse(const ember_json *root,
                                   ember_chat_request *out,
                                   char *err, size_t err_cap) {
    if (!root || root->type != EMBER_JSON_OBJECT ||
        ember_json_has_duplicate_keys(root)) {
        set_err(err, err_cap, "invalid Anthropic request");
        return false;
    }
    const ember_json *msgs = ember_json_get(root, "messages");
    if (!msgs || msgs->type != EMBER_JSON_ARRAY) {
        set_err(err, err_cap, "Anthropic messages must be an array");
        return false;
    }
    if (!validate_anthropic_payload(root, err, err_cap)) return false;
    const ember_json *thinking = ember_json_get(root, "thinking");
    if (thinking && thinking->type != EMBER_JSON_OBJECT) {
        set_err(err, err_cap, "Anthropic thinking must be an object");
        return false;
    }
    if (thinking) {
        const ember_json *type_value = ember_json_get(thinking, "type");
        const char *type = ember_json_str(type_value, NULL);
        const ember_json *budget = ember_json_get(thinking, "budget_tokens");
        if (!type || (strcmp(type, "enabled") && strcmp(type, "disabled")) ||
            (budget && budget->type != EMBER_JSON_NUMBER)) {
            set_err(err, err_cap, "invalid Anthropic thinking configuration");
            return false;
        }
    }
    const ember_json *anthropic_choice = ember_json_get(root, "tool_choice");
    const char *normalized_choice = NULL;
    const char *forced_name = NULL;
    bool disable_parallel = false;
    if (anthropic_choice && anthropic_choice->type != EMBER_JSON_NULL) {
        if (anthropic_choice->type == EMBER_JSON_STRING) {
            normalized_choice = ember_json_str(anthropic_choice, "");
        } else if (anthropic_choice->type == EMBER_JSON_OBJECT) {
            normalized_choice = ember_json_str(
                ember_json_get(anthropic_choice, "type"), "");
            forced_name = ember_json_str(
                ember_json_get(anthropic_choice, "name"), NULL);
            const ember_json *disable = ember_json_get(
                anthropic_choice, "disable_parallel_tool_use");
            if (disable) {
                if (disable->type != EMBER_JSON_BOOL) {
                    set_err(err, err_cap,
                            "disable_parallel_tool_use must be boolean");
                    return false;
                }
                disable_parallel = disable->u.b;
            }
        } else {
            set_err(err, err_cap, "invalid Anthropic tool_choice");
            return false;
        }
        if (strcmp(normalized_choice, "auto") &&
            strcmp(normalized_choice, "none") &&
            strcmp(normalized_choice, "any") &&
            strcmp(normalized_choice, "tool")) {
            set_err(err, err_cap, "invalid Anthropic tool_choice type");
            return false;
        }
        if (!strcmp(normalized_choice, "tool") &&
            (!forced_name || !forced_name[0])) {
            set_err(err, err_cap,
                    "Anthropic tool_choice type=tool requires name");
            return false;
        }
    }
    ember_buf normalized_msgs = {0};
    bool mc = false;
    const ember_json *system = ember_json_get(root, "system");
    if (system) {
        char *text = content_text(system);
        message_begin(&normalized_msgs, &mc, "system", text);
        message_end(&normalized_msgs);
        free(text);
    }
    for (int i = 0; i < ember_json_len(msgs); ++i)
        append_anthropic_message(&normalized_msgs, &mc, ember_json_at(msgs, i));

    ember_buf body = {0};
    ember_buf_putc(&body, '{');
    bool comma = false;
    append_common_fields(&body, root, &comma, false);
    if (normalized_choice) {
        if (comma) ember_buf_putc(&body, ',');
        ember_json_escape(&body, "tool_choice");
        ember_buf_putc(&body, ':');
        if (!strcmp(normalized_choice, "any")) {
            ember_json_escape(&body, "required");
        } else if (!strcmp(normalized_choice, "tool")) {
            ember_buf_puts(&body,
                "{\"type\":\"function\",\"function\":{\"name\":");
            ember_json_escape(&body, forced_name);
            ember_buf_puts(&body, "}}");
        } else {
            ember_json_escape(&body, normalized_choice);
        }
        comma = true;
    }
    if (disable_parallel) {
        if (comma) ember_buf_putc(&body, ',');
        ember_buf_puts(&body, "\"parallel_tool_calls\":false");
        comma = true;
    }
    append_json_field(&body, "stop", ember_json_get(root, "stop_sequences"), &comma);
    if (thinking && thinking->type == EMBER_JSON_OBJECT) {
        const char *type =
            ember_json_str(ember_json_get(thinking, "type"), "");
        if (comma) ember_buf_putc(&body, ',');
        ember_buf_puts(&body, "\"enable_thinking\":");
        ember_buf_puts(&body, !strcmp(type, "disabled") ? "false" : "true");
        comma = true;
        const ember_json *budget = ember_json_get(thinking, "budget_tokens");
        if (budget && strcmp(type, "disabled"))
            append_json_field(
                &body, "reasoning_budget_tokens", budget, &comma);
    }
    append_anthropic_tools(&body, ember_json_get(root, "tools"), &comma);
    if (comma) ember_buf_putc(&body, ',');
    ember_buf_puts(&body, "\"messages\":[");
    ember_buf_append(&body, normalized_msgs.ptr ? normalized_msgs.ptr : "",
                     normalized_msgs.len);
    ember_buf_puts(&body, "]}");
    ember_buf_free(&normalized_msgs);
    ember_json *normalized = ember_json_parse(body.ptr);
    ember_buf_free(&body);
    bool ok = normalized && ember_chat_request_parse(normalized, out);
    if (normalized) ember_json_free(normalized);
    if (!ok) {
        set_err(err, err_cap, "invalid normalized Anthropic input");
        return false;
    }
    out->api = EMBER_API_ANTHROPIC;
    return true;
}

static void append_usage(ember_buf *b, const ember_protocol_result *r,
                         bool responses) {
    if (responses)
        ember_buf_printf(b,
            "\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d,"
            "\"total_tokens\":%d,\"input_tokens_details\":{\"cached_tokens\":%d,"
            "\"cache_write_tokens\":0},\"output_tokens_details\":"
            "{\"reasoning_tokens\":0}}",
            r->prompt_tokens, r->completion_tokens,
            r->prompt_tokens + r->completion_tokens, r->cached_tokens);
    else
        ember_buf_printf(b,
            "\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}",
            r->prompt_tokens, r->completion_tokens);
}

static void append_tool_loop(ember_buf *b,
                             const ember_protocol_result *r) {
    if (!r || r->tool_loop_rounds <= 0) return;
    ember_buf_printf(b, "\"ember_tool_loop\":{\"rounds\":%d,\"tool\":",
                     r->tool_loop_rounds);
    ember_json_escape(b, r->tool_loop_tool ? r->tool_loop_tool : "");
    ember_buf_printf(b, ",\"identical_results\":%s}",
                     r->tool_loop_identical ? "true" : "false");
}

static const char *responses_id_suffix(const char *response_id) {
    if (!response_id) return "unknown";
    const char *underscore = strchr(response_id, '_');
    return underscore && underscore[1] ? underscore + 1 : response_id;
}

static void responses_item_id(char *out, size_t cap, const char *prefix,
                              const char *response_id, int index) {
    const char *suffix = responses_id_suffix(response_id);
    if (index >= 0)
        snprintf(out, cap, "%s%s_%d", prefix, suffix, index);
    else
        snprintf(out, cap, "%s%s", prefix, suffix);
}

static void append_responses_output(ember_buf *b,
                                    const ember_protocol_result *r,
                                    bool emit_reasoning,
                                    const char *item_status) {
    ember_buf_puts(b, "\"output\":[");
    bool comma = false;
    if (emit_reasoning && r->reasoning && r->reasoning[0]) {
        char item_id[72];
        responses_item_id(item_id, sizeof(item_id), "rs_", r->id, -1);
        ember_buf_puts(b, "{\"id\":");
        ember_json_escape(b, item_id);
        ember_buf_puts(b, ",\"type\":\"reasoning\",\"status\":");
        ember_json_escape(b, item_status);
        ember_buf_puts(b, ",\"summary\":["
            "{\"type\":\"summary_text\",\"text\":");
        ember_json_escape(b, r->reasoning);
        ember_buf_puts(b, "}]");
        ember_buf_putc(b, '}');
        comma = true;
    }
    if (r->content && r->content[0]) {
        char item_id[72];
        responses_item_id(item_id, sizeof(item_id), "msg_", r->id, -1);
        if (comma) ember_buf_putc(b, ',');
        ember_buf_puts(b, "{\"id\":");
        ember_json_escape(b, item_id);
        ember_buf_puts(b, ",\"type\":\"message\",\"status\":");
        ember_json_escape(b, item_status);
        ember_buf_puts(b,
            ",\"role\":\"assistant\","
            "\"content\":[{\"type\":\"output_text\","
            "\"text\":");
        ember_json_escape(b, r->content);
        ember_buf_puts(b, ",\"annotations\":[]}]}");
        comma = true;
    }
    for (int i = 0; r->calls && i < r->calls->len; ++i) {
        ember_tool_call *tc = &r->calls->calls[i];
        char item_id[72];
        responses_item_id(item_id, sizeof(item_id), "fc_", r->id, i);
        if (comma) ember_buf_putc(b, ',');
        ember_buf_puts(b, "{\"id\":");
        ember_json_escape(b, item_id);
        ember_buf_puts(b, ",\"type\":\"function_call\",\"status\":");
        ember_json_escape(b, item_status);
        ember_buf_puts(b, ",\"call_id\":");
        ember_json_escape(b, tc->id ? tc->id : "");
        ember_buf_puts(b, ",\"name\":");
        ember_json_escape(b, tc->name ? tc->name : "");
        ember_buf_puts(b, ",\"arguments\":");
        ember_json_escape(b, tc->arguments ? tc->arguments : "{}");
        ember_buf_putc(b, '}');
        comma = true;
    }
    ember_buf_putc(b, ']');
}

static void build_responses_json(ember_buf *b,
                                 const ember_protocol_result *r,
                                 bool emit_reasoning) {
    const char *status =
        r->finish_reason && !strcmp(r->finish_reason, "length")
            ? "incomplete" : "completed";
    ember_buf_puts(b, "{\"id\":");
    ember_json_escape(b, r->id);
    ember_buf_printf(b,
        ",\"object\":\"response\",\"created_at\":%ld,"
        "\"status\":\"%s\",\"model\":", r->created, status);
    ember_json_escape(b, r->model);
    if (r->tool_loop_rounds > 0) {
        ember_buf_putc(b, ',');
        append_tool_loop(b, r);
    }
    ember_buf_putc(b, ',');
    append_responses_output(
        b, r, emit_reasoning,
        !strcmp(status, "incomplete") ? "incomplete" : "completed");
    ember_buf_puts(b, ",\"error\":null,\"incomplete_details\":");
    if (!strcmp(status, "incomplete")) {
        ember_buf_puts(b, "{\"reason\":");
        ember_json_escape(
            b, r->termination_reason && r->termination_reason[0]
                ? r->termination_reason : "max_output_tokens");
        ember_buf_puts(b, "},");
    } else
        ember_buf_puts(b, "null,");
    append_usage(b, r, true);
    ember_buf_putc(b, '}');
}

static void event(ember_buf *b, const char *type, const char *json) {
    ember_buf_puts(b, "event: ");
    ember_buf_puts(b, type);
    ember_buf_puts(b, "\ndata: ");
    ember_buf_puts(b, json);
    ember_buf_puts(b, "\n\n");
}

static void responses_event(ember_buf *b, const char *type, const char *json,
                            int *sequence) {
    (void)type;
    size_t n = strlen(json);
    ember_buf numbered = {0};
    if (n > 0 && json[n - 1] == '}') {
        ember_buf_append(&numbered, json, n - 1);
        ember_buf_printf(&numbered, ",\"sequence_number\":%d}", (*sequence)++);
        ember_buf_puts(b, "data: ");
        ember_buf_puts(b, numbered.ptr);
        ember_buf_puts(b, "\n\n");
    } else {
        ember_buf_puts(b, "data: ");
        ember_buf_puts(b, json);
        ember_buf_puts(b, "\n\n");
    }
    ember_buf_free(&numbered);
}

static bool send_http(int fd, const char *ctype, const char *body, size_t n,
                      bool cors) {
    ember_buf out = {0};
    ember_buf_printf(&out,
        "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %zu\r\n",
        ctype, n);
    if (cors)
        ember_buf_puts(&out,
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: *\r\n");
    ember_buf_puts(&out, "Connection: close\r\n\r\n");
    ember_buf_append(&out, body, n);
    bool ok = ember_send_all(fd, out.ptr, out.len) == 0;
    ember_buf_free(&out);
    return ok;
}

// Buffered Responses reply. ember_protocol_emit routes stream=true to
// emit_buffered_stream_live BEFORE reaching any of the three emit_* functions,
// so req->stream is always false here. Do not add an SSE path back into this
// function: the previous one was unreachable for exactly that reason and gcov
// recorded it as never executed.
static bool emit_responses(int fd, const ember_chat_request *req,
                           const ember_protocol_result *r) {
    ember_buf full = {0};
    build_responses_json(&full, r, req->reasoning_summary_emit);
    bool ok = send_http(fd, "application/json", full.ptr, full.len,
                        req->response_cors);
    ember_buf_free(&full);
    return ok;
}

static const char *anthropic_stop(const ember_protocol_result *r) {
    if (r->calls && r->calls->len > 0) return "tool_use";
    if (r->finish_reason && !strcmp(r->finish_reason, "length"))
        return "max_tokens";
    if (r->stop_sequence) return "stop_sequence";
    return "end_turn";
}

static void append_anthropic_content(ember_buf *b,
                                     const ember_protocol_result *r) {
    ember_buf_puts(b, "\"content\":[");
    bool comma = false;
    // DeepSeek reasoning has no Anthropic-verifiable signature. Do not label it
    // as an authenticated `thinking` block or synthesize an empty signature.
    if (r->content && r->content[0]) {
        if (comma) ember_buf_putc(b, ',');
        ember_buf_puts(b, "{\"type\":\"text\",\"text\":");
        ember_json_escape(b, r->content);
        ember_buf_putc(b, '}');
        comma = true;
    }
    for (int i = 0; r->calls && i < r->calls->len; ++i) {
        ember_tool_call *tc = &r->calls->calls[i];
        if (comma) ember_buf_putc(b, ',');
        ember_buf_puts(b, "{\"type\":\"tool_use\",\"id\":");
        ember_json_escape(b, tc->id ? tc->id : "");
        ember_buf_puts(b, ",\"name\":");
        ember_json_escape(b, tc->name ? tc->name : "");
        ember_buf_puts(b, ",\"input\":");
        const char *args = tc->arguments ? tc->arguments : "{}";
        ember_json *parsed = ember_json_parse(args);
        if (parsed) {
            char *dump = ember_json_dump(parsed);
            ember_buf_puts(b, dump ? dump : "{}");
            free(dump);
            ember_json_free(parsed);
        } else {
            ember_buf_puts(b, "{}");
        }
        ember_buf_putc(b, '}');
        comma = true;
    }
    ember_buf_putc(b, ']');
}

static void build_anthropic_json(ember_buf *b,
                                 const ember_protocol_result *r) {
    ember_buf_puts(b, "{\"id\":");
    ember_json_escape(b, r->id);
    ember_buf_puts(b, ",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
    ember_json_escape(b, r->model);
    ember_buf_putc(b, ',');
    append_anthropic_content(b, r);
    ember_buf_puts(b, ",\"stop_reason\":");
    ember_json_escape(b, anthropic_stop(r));
    ember_buf_puts(b, ",\"stop_sequence\":");
    if (r->stop_sequence) ember_json_escape(b, r->stop_sequence);
    else ember_buf_puts(b, "null");
    if (r->tool_loop_rounds > 0) {
        ember_buf_putc(b, ',');
        append_tool_loop(b, r);
    }
    ember_buf_putc(b, ',');
    append_usage(b, r, false);
    ember_buf_putc(b, '}');
}

static void protocol_active_reset(ember_protocol_stream *st) {
    st->active = EMBER_PROTOCOL_STREAM_NONE;
    st->active_tool_index = -1;
    free(st->active_id);
    free(st->active_call_id);
    free(st->active_name);
    st->active_id = NULL;
    st->active_call_id = NULL;
    st->active_name = NULL;
    st->active_value.len = 0;
    if (st->active_value.ptr) st->active_value.ptr[0] = '\0';
}

static void protocol_responses_event(ember_protocol_stream *st,
                                     ember_buf *out, const char *type,
                                     ember_buf *json) {
    responses_event(out, type, json->ptr, &st->sequence);
    ember_buf_free(json);
}

static void protocol_close_active_status(ember_protocol_stream *st,
                                         ember_buf *out,
                                         const char *item_status) {
    if (!st || st->active == EMBER_PROTOCOL_STREAM_NONE) return;
    ember_buf j = {0};
    if (st->req->api == EMBER_API_RESPONSES) {
        if (st->active == EMBER_PROTOCOL_STREAM_REASONING) {
            ember_buf_printf(&j,
                "{\"type\":\"response.reasoning_summary_text.done\","
                "\"item_id\":");
            ember_json_escape(&j, st->reasoning_id);
            ember_buf_printf(&j,
                ",\"output_index\":%d,\"summary_index\":0,\"text\":",
                st->output_index);
            ember_json_escape_n(&j, st->active_value.ptr ? st->active_value.ptr : "",
                                st->active_value.len);
            ember_buf_putc(&j, '}');
            protocol_responses_event(
                st, out, "response.reasoning_summary_text.done", &j);
            ember_buf_printf(&j,
                "{\"type\":\"response.reasoning_summary_part.done\","
                "\"item_id\":");
            ember_json_escape(&j, st->reasoning_id);
            ember_buf_printf(&j,
                ",\"output_index\":%d,\"summary_index\":0,"
                "\"part\":{\"type\":\"summary_text\",\"text\":",
                st->output_index);
            ember_json_escape_n(&j, st->active_value.ptr ? st->active_value.ptr : "",
                                st->active_value.len);
            ember_buf_puts(&j, "}}");
            protocol_responses_event(
                st, out, "response.reasoning_summary_part.done", &j);
            ember_buf_printf(&j,
                "{\"type\":\"response.output_item.done\",\"output_index\":%d,"
                "\"item\":{\"id\":", st->output_index);
            ember_json_escape(&j, st->reasoning_id);
            ember_buf_puts(&j, ",\"type\":\"reasoning\",\"status\":");
            ember_json_escape(&j, item_status);
            ember_buf_puts(&j,
                ",\"summary\":[{\"type\":\"summary_text\","
                "\"text\":");
            ember_json_escape_n(&j, st->active_value.ptr ? st->active_value.ptr : "",
                                st->active_value.len);
            ember_buf_puts(&j, "}]}}");
            protocol_responses_event(st, out, "response.output_item.done", &j);
        } else if (st->active == EMBER_PROTOCOL_STREAM_TEXT) {
            ember_buf_puts(&j,
                "{\"type\":\"response.output_text.done\",\"item_id\":");
            ember_json_escape(&j, st->message_id);
            ember_buf_printf(&j,
                ",\"output_index\":%d,\"content_index\":0,\"text\":",
                st->output_index);
            ember_json_escape_n(&j, st->active_value.ptr ? st->active_value.ptr : "",
                                st->active_value.len);
            ember_buf_putc(&j, '}');
            protocol_responses_event(st, out, "response.output_text.done", &j);
            ember_buf_puts(&j,
                "{\"type\":\"response.content_part.done\",\"item_id\":");
            ember_json_escape(&j, st->message_id);
            ember_buf_printf(&j,
                ",\"output_index\":%d,\"content_index\":0,\"part\":"
                "{\"type\":\"output_text\",\"text\":", st->output_index);
            ember_json_escape_n(&j, st->active_value.ptr ? st->active_value.ptr : "",
                                st->active_value.len);
            ember_buf_puts(&j, ",\"annotations\":[]}}");
            protocol_responses_event(st, out, "response.content_part.done", &j);
            ember_buf_printf(&j,
                "{\"type\":\"response.output_item.done\",\"output_index\":%d,"
                "\"item\":{\"id\":", st->output_index);
            ember_json_escape(&j, st->message_id);
            ember_buf_puts(&j, ",\"type\":\"message\",\"status\":");
            ember_json_escape(&j, item_status);
            ember_buf_puts(&j,
                ",\"role\":\"assistant\",\"content\":["
                "{\"type\":\"output_text\",\"text\":");
            ember_json_escape_n(&j, st->active_value.ptr ? st->active_value.ptr : "",
                                st->active_value.len);
            ember_buf_puts(&j, ",\"annotations\":[]}]}}");
            protocol_responses_event(st, out, "response.output_item.done", &j);
        } else if (st->active == EMBER_PROTOCOL_STREAM_TOOL) {
            ember_buf_puts(&j,
                "{\"type\":\"response.function_call_arguments.done\","
                "\"item_id\":");
            ember_json_escape(&j, st->active_id ? st->active_id : "");
            ember_buf_printf(&j, ",\"output_index\":%d,\"name\":",
                             st->output_index);
            ember_json_escape(&j, st->active_name ? st->active_name : "");
            ember_buf_puts(&j, ",\"arguments\":");
            ember_json_escape_n(&j, st->active_value.ptr ? st->active_value.ptr : "",
                                st->active_value.len);
            ember_buf_putc(&j, '}');
            protocol_responses_event(
                st, out, "response.function_call_arguments.done", &j);
            ember_buf_printf(&j,
                "{\"type\":\"response.output_item.done\",\"output_index\":%d,"
                "\"item\":{\"id\":", st->output_index);
            ember_json_escape(&j, st->active_id ? st->active_id : "");
            ember_buf_puts(&j,
                ",\"type\":\"function_call\",\"status\":");
            ember_json_escape(&j, item_status);
            ember_buf_puts(&j,
                ","
                "\"call_id\":");
            ember_json_escape(&j,
                              st->active_call_id ? st->active_call_id : "");
            ember_buf_puts(&j, ",\"name\":");
            ember_json_escape(&j, st->active_name ? st->active_name : "");
            ember_buf_puts(&j, ",\"arguments\":");
            ember_json_escape_n(&j, st->active_value.ptr ? st->active_value.ptr : "",
                                st->active_value.len);
            ember_buf_puts(&j, "}}");
            protocol_responses_event(st, out, "response.output_item.done", &j);
        }
        st->output_index++;
    } else if (st->req->api == EMBER_API_ANTHROPIC) {
        ember_buf_printf(&j,
            "{\"type\":\"content_block_stop\",\"index\":%d}", st->block_index);
        event(out, "content_block_stop", j.ptr);
        ember_buf_free(&j);
        st->block_index++;
    }
    protocol_active_reset(st);
}

static void protocol_close_active(ember_protocol_stream *st, ember_buf *out) {
    protocol_close_active_status(st, out, "completed");
}

static void protocol_start_reasoning(ember_protocol_stream *st,
                                     ember_buf *out) {
    if (st->req->api == EMBER_API_RESPONSES) {
        if (!st->req->reasoning_summary_emit) return;
        ember_buf j = {0};
        ember_buf_printf(&j,
            "{\"type\":\"response.output_item.added\",\"output_index\":%d,"
            "\"item\":{\"id\":", st->output_index);
        ember_json_escape(&j, st->reasoning_id);
        ember_buf_puts(&j,
            ",\"type\":\"reasoning\",\"status\":\"in_progress\","
            "\"summary\":[]}}");
        protocol_responses_event(st, out, "response.output_item.added", &j);
        ember_buf_puts(&j,
            "{\"type\":\"response.reasoning_summary_part.added\","
            "\"item_id\":");
        ember_json_escape(&j, st->reasoning_id);
        ember_buf_printf(&j,
            ",\"output_index\":%d,\"summary_index\":0,"
            "\"part\":{\"type\":\"summary_text\",\"text\":\"\"}}",
            st->output_index);
        protocol_responses_event(
            st, out, "response.reasoning_summary_part.added", &j);
    } else {
        return;
    }
    st->active = EMBER_PROTOCOL_STREAM_REASONING;
}

static void protocol_start_text(ember_protocol_stream *st, ember_buf *out) {
    ember_buf j = {0};
    if (st->req->api == EMBER_API_RESPONSES) {
        ember_buf_printf(&j,
            "{\"type\":\"response.output_item.added\",\"output_index\":%d,"
            "\"item\":{\"id\":",
            st->output_index);
        ember_json_escape(&j, st->message_id);
        ember_buf_puts(&j,
            ",\"type\":\"message\",\"status\":\"in_progress\","
            "\"role\":\"assistant\",\"content\":[]}}");
        protocol_responses_event(st, out, "response.output_item.added", &j);
        ember_buf_puts(&j,
            "{\"type\":\"response.content_part.added\",\"item_id\":");
        ember_json_escape(&j, st->message_id);
        ember_buf_printf(&j,
            ",\"output_index\":%d,\"content_index\":0,\"part\":"
            "{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}",
            st->output_index);
        protocol_responses_event(st, out, "response.content_part.added", &j);
    } else if (st->req->api == EMBER_API_ANTHROPIC) {
        ember_buf_printf(&j,
            "{\"type\":\"content_block_start\",\"index\":%d,"
            "\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
            st->block_index);
        event(out, "content_block_start", j.ptr);
        ember_buf_free(&j);
    }
    st->active = EMBER_PROTOCOL_STREAM_TEXT;
}

static void protocol_text_delta(void *ud, bool reasoning,
                                const char *text, size_t len, ember_buf *out) {
    ember_protocol_stream *st = (ember_protocol_stream *)ud;
    if (!st || len == 0) return;
    if (reasoning && (st->req->api == EMBER_API_COMPLETIONS ||
                      st->req->api == EMBER_API_ANTHROPIC)) return;
    if (reasoning && st->req->api == EMBER_API_RESPONSES &&
        !st->req->reasoning_summary_emit)
        return;
    ember_protocol_stream_kind want =
        reasoning ? EMBER_PROTOCOL_STREAM_REASONING
                  : EMBER_PROTOCOL_STREAM_TEXT;
    if (st->active != want) {
        protocol_close_active(st, out);
        if (reasoning) protocol_start_reasoning(st, out);
        else protocol_start_text(st, out);
    }
    if (st->req->api == EMBER_API_COMPLETIONS) {
        ember_buf j = {0};
        ember_buf_puts(&j, "{\"id\":");
        ember_json_escape(&j, st->id);
        ember_buf_printf(&j,
            ",\"object\":\"text_completion\",\"created\":%ld,\"model\":",
            st->created);
        ember_json_escape(&j, st->model);
        ember_buf_puts(&j, ",\"choices\":[{\"index\":0,\"text\":");
        ember_json_escape_n(&j, text, len);
        ember_buf_puts(&j,
            ",\"logprobs\":null,\"finish_reason\":null}]}\n\n");
        ember_buf_puts(out, "data: ");
        ember_buf_append(out, j.ptr, j.len);
        ember_buf_free(&j);
        return;
    }
    ember_buf_append(&st->active_value, text, len);
    ember_buf j = {0};
    if (st->req->api == EMBER_API_RESPONSES) {
        if (reasoning) {
            ember_buf_printf(&j,
                "{\"type\":\"response.reasoning_summary_text.delta\","
                "\"item_id\":");
            ember_json_escape(&j, st->reasoning_id);
            ember_buf_printf(&j,
                ",\"output_index\":%d,\"summary_index\":0,\"delta\":",
                st->output_index);
            ember_json_escape_n(&j, text, len);
            ember_buf_putc(&j, '}');
            protocol_responses_event(
                st, out, "response.reasoning_summary_text.delta", &j);
        } else {
            ember_buf_puts(&j,
                "{\"type\":\"response.output_text.delta\",\"item_id\":");
            ember_json_escape(&j, st->message_id);
            ember_buf_printf(&j,
                ",\"output_index\":%d,\"content_index\":0,\"delta\":",
                st->output_index);
            ember_json_escape_n(&j, text, len);
            ember_buf_putc(&j, '}');
            protocol_responses_event(st, out, "response.output_text.delta", &j);
        }
    } else if (st->req->api == EMBER_API_ANTHROPIC && !reasoning) {
        ember_buf_printf(&j,
            "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":",
            st->block_index);
        ember_json_escape(&j, "text_delta");
        ember_buf_puts(&j, ",\"text\":");
        ember_json_escape_n(&j, text, len);
        ember_buf_puts(&j, "}}");
        event(out, "content_block_delta", j.ptr);
        ember_buf_free(&j);
    }
}

static void protocol_tool_start(void *ud, int index, const char *id,
                                const char *name, ember_buf *out) {
    ember_protocol_stream *st = (ember_protocol_stream *)ud;
    if (!st) return;
    protocol_close_active(st, out);
    st->active = EMBER_PROTOCOL_STREAM_TOOL;
    st->active_tool_index = index;
    char item_id[72];
    responses_item_id(
        item_id, sizeof(item_id), "fc_", st->id, index);
    st->active_id = strdup(item_id);
    st->active_call_id = strdup(id ? id : "");
    st->active_name = strdup(name ? name : "");
    ember_buf j = {0};
    if (st->req->api == EMBER_API_RESPONSES) {
        ember_buf_printf(&j,
            "{\"type\":\"response.output_item.added\",\"output_index\":%d,"
            "\"item\":{\"id\":", st->output_index);
        ember_json_escape(&j, st->active_id ? st->active_id : "");
        ember_buf_puts(&j,
            ",\"type\":\"function_call\",\"status\":\"in_progress\","
            "\"call_id\":");
        ember_json_escape(&j,
                          st->active_call_id ? st->active_call_id : "");
        ember_buf_puts(&j, ",\"name\":");
        ember_json_escape(&j, st->active_name ? st->active_name : "");
        ember_buf_puts(&j, ",\"arguments\":\"\"}}");
        protocol_responses_event(st, out, "response.output_item.added", &j);
    } else if (st->req->api == EMBER_API_ANTHROPIC) {
        ember_buf_printf(&j,
            "{\"type\":\"content_block_start\",\"index\":%d,"
            "\"content_block\":{\"type\":\"tool_use\",\"id\":", st->block_index);
        ember_json_escape(&j,
                          st->active_call_id ? st->active_call_id : "");
        ember_buf_puts(&j, ",\"name\":");
        ember_json_escape(&j, st->active_name ? st->active_name : "");
        ember_buf_puts(&j, ",\"input\":{}}}");
        event(out, "content_block_start", j.ptr);
        ember_buf_free(&j);
    }
}

static void protocol_tool_args_delta(void *ud, int index,
                                     const char *text, size_t len,
                                     ember_buf *out) {
    ember_protocol_stream *st = (ember_protocol_stream *)ud;
    if (!st || len == 0 || st->active != EMBER_PROTOCOL_STREAM_TOOL ||
        st->active_tool_index != index)
        return;
    ember_buf_append(&st->active_value, text, len);
    ember_buf j = {0};
    if (st->req->api == EMBER_API_RESPONSES) {
        ember_buf_puts(&j,
            "{\"type\":\"response.function_call_arguments.delta\","
            "\"item_id\":");
        ember_json_escape(&j, st->active_id ? st->active_id : "");
        ember_buf_printf(&j, ",\"output_index\":%d,\"delta\":", st->output_index);
        ember_json_escape_n(&j, text, len);
        ember_buf_putc(&j, '}');
        protocol_responses_event(
            st, out, "response.function_call_arguments.delta", &j);
    } else if (st->req->api == EMBER_API_ANTHROPIC) {
        ember_buf_printf(&j,
            "{\"type\":\"content_block_delta\",\"index\":%d,"
            "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":",
            st->block_index);
        ember_json_escape_n(&j, text, len);
        ember_buf_puts(&j, "}}");
        event(out, "content_block_delta", j.ptr);
        ember_buf_free(&j);
    }
}

void ember_protocol_stream_init(ember_protocol_stream *st,
                                const ember_chat_request *req,
                                const char *id, const char *model,
                                long created, int prompt_tokens) {
    memset(st, 0, sizeof(*st));
    st->req = req;
    st->id = id;
    st->model = model;
    st->created = created;
    st->prompt_tokens = prompt_tokens;
    st->active_tool_index = -1;
    responses_item_id(
        st->reasoning_id, sizeof(st->reasoning_id), "rs_", id, -1);
    responses_item_id(
        st->message_id, sizeof(st->message_id), "msg_", id, -1);
}

void ember_protocol_stream_free(ember_protocol_stream *st) {
    if (!st) return;
    free(st->active_id);
    free(st->active_call_id);
    free(st->active_name);
    ember_buf_free(&st->active_value);
    memset(st, 0, sizeof(*st));
}

void ember_protocol_stream_bind(ember_protocol_stream *st,
                                ember_sse_stream *splitter) {
    const ember_sse_sink sink = {
        .text_delta = protocol_text_delta,
        .tool_start = protocol_tool_start,
        .tool_args_delta = protocol_tool_args_delta,
    };
    ember_sse_set_sink(splitter, &sink, st);
}

void ember_protocol_stream_begin(ember_protocol_stream *st, ember_buf *out) {
    if (st->req->api == EMBER_API_RESPONSES) {
        ember_buf j = {0};
        ember_buf_puts(&j,
            "{\"type\":\"response.created\",\"response\":{\"id\":");
        ember_json_escape(&j, st->id);
        ember_buf_printf(&j,
            ",\"object\":\"response\",\"created_at\":%ld,"
            "\"status\":\"in_progress\",\"model\":", st->created);
        ember_json_escape(&j, st->model);
        ember_buf_puts(&j, ",\"output\":[]}}");
        protocol_responses_event(st, out, "response.created", &j);
    } else if (st->req->api == EMBER_API_ANTHROPIC) {
        ember_buf j = {0};
        ember_buf_puts(&j,
            "{\"type\":\"message_start\",\"message\":{\"id\":");
        ember_json_escape(&j, st->id);
        ember_buf_puts(&j,
            ",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
        ember_json_escape(&j, st->model);
        ember_buf_printf(&j,
            ",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,"
            "\"usage\":{\"input_tokens\":%d,\"output_tokens\":0}}}",
            st->prompt_tokens);
        event(out, "message_start", j.ptr);
        ember_buf_free(&j);
    }
}

void ember_protocol_stream_finish(ember_protocol_stream *st,
                                  const ember_protocol_result *r,
                                  ember_buf *out) {
    protocol_close_active_status(
        st, out,
        r->finish_reason && !strcmp(r->finish_reason, "length")
            ? "incomplete" : "completed");
    if (st->req->api == EMBER_API_RESPONSES) {
        ember_buf full = {0}, j = {0};
        build_responses_json(&full, r, st->req->reasoning_summary_emit);
        const char *terminal =
            r->finish_reason && !strcmp(r->finish_reason, "length")
                ? "response.incomplete" : "response.completed";
        ember_buf_puts(&j, "{\"type\":");
        ember_json_escape(&j, terminal);
        ember_buf_puts(&j, ",\"response\":");
        ember_buf_append(&j, full.ptr, full.len);
        ember_buf_putc(&j, '}');
        protocol_responses_event(st, out, terminal, &j);
        ember_buf_free(&full);
    } else if (st->req->api == EMBER_API_ANTHROPIC) {
        ember_buf j = {0};
        ember_buf_puts(&j,
            "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":");
        ember_json_escape(&j, anthropic_stop(r));
        ember_buf_puts(&j, ",\"stop_sequence\":");
        if (r->stop_sequence) ember_json_escape(&j, r->stop_sequence);
        else ember_buf_puts(&j, "null");
        if (r->tool_loop_rounds > 0) {
            ember_buf_putc(&j, ',');
            append_tool_loop(&j, r);
        }
        ember_buf_printf(&j,
            "},\"usage\":{\"output_tokens\":%d}}",
            r->completion_tokens);
        event(out, "message_delta", j.ptr);
        ember_buf_free(&j);
        event(out, "message_stop", "{\"type\":\"message_stop\"}");
    } else if (st->req->api == EMBER_API_COMPLETIONS) {
        ember_buf j = {0};
        ember_buf_puts(&j, "data: {\"id\":");
        ember_json_escape(&j, st->id);
        ember_buf_printf(&j,
            ",\"object\":\"text_completion\",\"created\":%ld,\"model\":",
            st->created);
        ember_json_escape(&j, st->model);
        ember_buf_puts(&j,
            ",\"choices\":[{\"index\":0,\"text\":\"\",\"logprobs\":null,"
            "\"finish_reason\":");
        ember_json_escape(&j, r->finish_reason ? r->finish_reason : "stop");
        ember_buf_puts(&j, "}]}\n\ndata: [DONE]\n\n");
        ember_buf_append(out, j.ptr, j.len);
        ember_buf_free(&j);
    }
}

void ember_protocol_stream_error(ember_protocol_stream *st,
                                 const char *code, const char *detail,
                                 bool retry_exhausted, ember_buf *out) {
    protocol_close_active_status(st, out, "incomplete");
    const char *message =
        detail && detail[0] ? detail : "backend generation failed";
    bool invalid_tool = code && strcmp(code, "invalid_tool_call") == 0;
    bool progress_stop = code &&
        (!strcmp(code, "repetition_detected") ||
         !strcmp(code, "reasoning_cycle_detected") ||
         !strcmp(code, "prompt_echo_detected"));
    bool model_output = invalid_tool || progress_stop;
    if (st->req->api == EMBER_API_ANTHROPIC) {
        ember_buf j = {0};
        ember_buf_puts(&j,
            "{\"type\":\"error\",\"error\":{\"type\":\"api_error\","
            "\"message\":");
        ember_json_escape(&j, message);
        ember_buf_puts(&j, "}}");
        event(out, "error", j.ptr);
        ember_buf_free(&j);
    } else if (st->req->api == EMBER_API_RESPONSES) {
        ember_buf j = {0};
        ember_buf_puts(&j, "{\"type\":\"error\",\"code\":");
        ember_json_escape(&j, code && code[0] ? code : "internal_error");
        ember_buf_puts(&j, ",\"message\":");
        ember_json_escape(&j, message);
        ember_buf_puts(&j, ",\"param\":null");
        if (model_output) {
            ember_buf_printf(&j, ",\"retry_exhausted\":%s",
                             retry_exhausted ? "true" : "false");
            ember_buf_puts(&j, ",\"partial_tool_call\":false");
        }
        ember_buf_putc(&j, '}');
        protocol_responses_event(st, out, "error", &j);
        ember_buf_free(&j);
    } else {
        ember_buf_puts(out, "event: error\ndata: {\"error\":{\"message\":");
        ember_json_escape(out, message);
        ember_buf_puts(out, ",\"type\":");
        ember_json_escape(
            out, model_output ? "model_output_error" : "server_error");
        ember_buf_puts(out, ",\"code\":");
        ember_json_escape(out, code && code[0] ? code : "internal_error");
        if (model_output) {
            ember_buf_printf(out, ",\"retry_exhausted\":%s",
                             retry_exhausted ? "true" : "false");
            ember_buf_puts(out, ",\"partial_tool_call\":false");
        }
        ember_buf_puts(out, "}}\n\n");
    }
}

static bool send_stream_piece(int fd, ember_buf *piece) {
    if (!piece->len) return true;
    bool ok = ember_send_all(fd, piece->ptr, piece->len) == 0;
    piece->len = 0;
    if (piece->ptr) piece->ptr[0] = '\0';
    return ok;
}

// Compatibility path for callers that already have a completed turn (unit
// tests and any future replay endpoint). It uses the same live state machine
// and writes each lifecycle stage separately, so it cannot recreate the old
// one-large-buffer backpressure failure.
static bool emit_buffered_stream_live(int fd, const ember_chat_request *req,
                                      const ember_protocol_result *r) {
    ember_buf piece = {0};
    ember_sse_headers(&piece, req->response_cors);
    if (!send_stream_piece(fd, &piece)) goto fail;

    ember_protocol_stream st;
    ember_protocol_stream_init(
        &st, req, r->id, r->model, r->created, r->prompt_tokens);
    ember_protocol_stream_begin(&st, &piece);
    if (!send_stream_piece(fd, &piece)) goto fail_stream;

    if (r->reasoning && r->reasoning[0]) {
        protocol_text_delta(
            &st, true, r->reasoning, strlen(r->reasoning), &piece);
        if (!send_stream_piece(fd, &piece)) goto fail_stream;
    }
    if (r->content && r->content[0]) {
        protocol_text_delta(
            &st, false, r->content, strlen(r->content), &piece);
        if (!send_stream_piece(fd, &piece)) goto fail_stream;
    }
    for (int i = 0; r->calls && i < r->calls->len; ++i) {
        const ember_tool_call *tc = &r->calls->calls[i];
        protocol_tool_start(
            &st, i, tc->id ? tc->id : "", tc->name ? tc->name : "", &piece);
        if (!send_stream_piece(fd, &piece)) goto fail_stream;
        const char *args = tc->arguments ? tc->arguments : "{}";
        protocol_tool_args_delta(&st, i, args, strlen(args), &piece);
        if (!send_stream_piece(fd, &piece)) goto fail_stream;
    }
    ember_protocol_stream_finish(&st, r, &piece);
    if (!send_stream_piece(fd, &piece)) goto fail_stream;
    ember_protocol_stream_free(&st);
    ember_buf_free(&piece);
    return true;

fail_stream:
    ember_protocol_stream_free(&st);
fail:
    ember_buf_free(&piece);
    return false;
}

// Buffered Anthropic reply. Streaming is emit_buffered_stream_live's job — see
// the note on emit_responses.
static bool emit_anthropic(int fd, const ember_chat_request *req,
                           const ember_protocol_result *r) {
    ember_buf full = {0};
    build_anthropic_json(&full, r);
    bool ok = send_http(fd, "application/json", full.ptr, full.len,
                        req->response_cors);
    ember_buf_free(&full);
    return ok;
}

static void build_completion_json(ember_buf *b,
                                  const ember_protocol_result *r) {
    ember_buf_puts(b, "{\"id\":");
    ember_json_escape(b, r->id);
    ember_buf_printf(b,
        ",\"object\":\"text_completion\",\"created\":%ld,\"model\":",
        r->created);
    ember_json_escape(b, r->model);
    ember_buf_puts(b, ",\"choices\":[{\"index\":0,\"text\":");
    ember_json_escape(b, r->content ? r->content : "");
    ember_buf_puts(b, ",\"logprobs\":null,\"finish_reason\":");
    ember_json_escape(b, r->finish_reason ? r->finish_reason : "stop");
    ember_buf_puts(b, "}],");
    append_usage(b, r, true);
    ember_buf_putc(b, '}');
}

// Buffered legacy Completions reply. Streaming is emit_buffered_stream_live's
// job — see the note on emit_responses.
static bool emit_completion(int fd, const ember_chat_request *req,
                            const ember_protocol_result *r) {
    ember_buf full = {0};
    build_completion_json(&full, r);
    bool ok = send_http(fd, "application/json", full.ptr, full.len,
                        req->response_cors);
    ember_buf_free(&full);
    return ok;
}

bool ember_protocol_emit(int fd, const ember_chat_request *req,
                         const ember_protocol_result *result) {
    if (!req || !result) return false;
    if (req->stream)
        return emit_buffered_stream_live(fd, req, result);
    if (req->api == EMBER_API_RESPONSES)
        return emit_responses(fd, req, result);
    if (req->api == EMBER_API_ANTHROPIC)
        return emit_anthropic(fd, req, result);
    if (req->api == EMBER_API_COMPLETIONS)
        return emit_completion(fd, req, result);
    return false;
}
