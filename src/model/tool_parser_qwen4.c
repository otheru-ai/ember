#include "tool_parser_qwen4.h"

#include <ctype.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../common/buf.h"
#include "../common/json.h"
#include "../common/json_util.h"

#define CALL_OPEN   "<tool_call>"
#define CALL_CLOSE  "</tool_call>"
#define FUNC_OPEN   "<function="
#define FUNC_CLOSE  "</function>"
#define PARAM_OPEN  "<parameter="
#define PARAM_CLOSE "</parameter>"

static const char *skip_ws(const char *at, const char *end) {
    while (at < end && isspace((unsigned char)*at)) ++at;
    return at;
}

static bool valid_tag_name(const char *start, const char *end) {
    if (start >= end) return false;
    for (const char *at = start; at < end; ++at) {
        const unsigned char ch = (unsigned char)*at;
        if (isspace(ch) || ch == '<' || ch == '>' || ch == '=') return false;
    }
    return true;
}

static char *dup_range(const char *start, const char *end) {
    const size_t length = (size_t)(end - start);
    char *copy = (char *)malloc(length + 1);
    if (!copy) ember_buf_fatal("out of memory parsing Qwen tool call");
    memcpy(copy, start, length);
    copy[length] = '\0';
    return copy;
}

static const ember_json *schema_for_call(const ember_json *tools,
                                         const char *call_name) {
    for (int i = 0; tools && i < ember_json_len(tools); ++i) {
        const ember_json *tool = ember_json_at(tools, i);
        const ember_json *function = ember_json_get(tool, "function");
        if (!function) function = tool;
        const char *name = ember_json_str(ember_json_get(function, "name"), NULL);
        if (name && strcmp(name, call_name) == 0)
            return ember_json_get(function, "parameters");
    }
    return NULL;
}

static const ember_json *schema_for_param(const ember_json *schema,
                                          const char *param_name) {
    const ember_json *properties = ember_json_get(schema, "properties");
    return ember_json_get(properties, param_name);
}

static bool value_matches_type(const ember_json *value, const char *type) {
    if (!value || !type) return false;
    if (strcmp(type, "string") == 0) return value->type == EMBER_JSON_STRING;
    if (strcmp(type, "integer") == 0 || strcmp(type, "int") == 0)
        return value->type == EMBER_JSON_NUMBER && value->num_raw &&
               !strchr(value->num_raw, '.') && !strchr(value->num_raw, 'e') &&
               !strchr(value->num_raw, 'E');
    if (strcmp(type, "number") == 0) return value->type == EMBER_JSON_NUMBER;
    if (strcmp(type, "boolean") == 0 || strcmp(type, "bool") == 0)
        return value->type == EMBER_JSON_BOOL;
    if (strcmp(type, "object") == 0) return value->type == EMBER_JSON_OBJECT;
    if (strcmp(type, "array") == 0) return value->type == EMBER_JSON_ARRAY;
    if (strcmp(type, "null") == 0) return value->type == EMBER_JSON_NULL;
    return false;
}

static const char *schema_type(const ember_json *schema) {
    const ember_json *type = ember_json_get(schema, "type");
    if (type && type->type == EMBER_JSON_STRING) return type->u.str;
    return NULL;
}

static void append_coerced_value(ember_buf *args, const char *raw,
                                 const ember_json *schema) {
    const char *type = schema_type(schema);
    if (type && strcmp(type, "string") != 0) {
        ember_json *candidate = ember_json_parse(raw);
        if (value_matches_type(candidate, type)) {
            char *dump = ember_json_dump(candidate);
            ember_buf_puts(args, dump ? dump : "null");
            free(dump);
            ember_json_free(candidate);
            return;
        }
        ember_json_free(candidate);
    }
    ember_json_escape(args, raw);
}

static bool push_call(ember_tool_calls *out, char *name, char *arguments) {
    if (out->len == INT_MAX) return false;
    if (out->len == out->cap) {
        if (out->cap > INT_MAX / 2) return false;
        int next = out->cap > 0 ? out->cap * 2 : 2;
        if ((size_t)next > SIZE_MAX / sizeof(*out->calls))
            return false;
        ember_tool_call *grown = (ember_tool_call *)realloc(
            out->calls, (size_t)next * sizeof(*out->calls));
        if (!grown) ember_buf_fatal("out of memory parsing Qwen tool calls");
        out->calls = grown;
        out->cap = next;
    }
    out->calls[out->len++] = (ember_tool_call){
        .name = name,
        .arguments = arguments,
        .id = NULL,
    };
    return true;
}

static bool range_contains(const char *start, const char *end,
                           const char *needle) {
    const char *found = strstr(start, needle);
    return found && found < end;
}

static bool parse_wrapper(const char *body, const char *body_end,
                          const ember_json *tools, ember_tool_calls *out,
                          ember_qwen_tool_parse_report *report) {
    const char *at = skip_ws(body, body_end);
    if ((size_t)(body_end - at) < strlen(FUNC_OPEN) ||
        strncmp(at, FUNC_OPEN, strlen(FUNC_OPEN)) != 0) return false;
    at += strlen(FUNC_OPEN);
    const char *name_end = memchr(at, '>', (size_t)(body_end - at));
    if (!name_end || !valid_tag_name(at, name_end)) return false;
    char *name = dup_range(at, name_end);
    const ember_json *call_schema = schema_for_call(tools, name);
    at = skip_ws(name_end + 1, body_end);

    ember_buf args = {0};
    ember_buf_putc(&args, '{');
    int n_params = 0;
    char **keys = NULL;
    int n_keys = 0;
    while ((size_t)(body_end - at) >= strlen(PARAM_OPEN) &&
           strncmp(at, PARAM_OPEN, strlen(PARAM_OPEN)) == 0) {
        at += strlen(PARAM_OPEN);
        const char *key_end = memchr(at, '>', (size_t)(body_end - at));
        if (!key_end || !valid_tag_name(at, key_end)) goto malformed;
        char *key = dup_range(at, key_end);
        const char *value = key_end + 1;
        const char *param_close = strstr(value, PARAM_CLOSE);
        if (!param_close || param_close > body_end) {
            free(key);
            goto malformed;
        }
        const char *value_end = param_close;
        if (range_contains(value, value_end, CALL_OPEN) ||
            range_contains(value, value_end, FUNC_OPEN) ||
            range_contains(value, value_end, PARAM_OPEN)) {
            report->contaminated = true;
            free(key);
            goto malformed;
        }
        if (value < value_end && *value == '\n') ++value;
        if (value < value_end && value_end[-1] == '\n') --value_end;

        for (int i = 0; i < n_keys; ++i) {
            if (strcmp(keys[i], key) == 0) {
                free(key);
                goto malformed;
            }
        }
        if (n_keys == INT_MAX ||
            (size_t)(n_keys + 1) > SIZE_MAX / sizeof(*keys)) {
            free(key);
            goto malformed;
        }
        char **grown = (char **)realloc(
            keys, (size_t)(n_keys + 1) * sizeof(*keys));
        if (!grown) ember_buf_fatal("out of memory parsing Qwen parameters");
        keys = grown;
        keys[n_keys++] = key;
        if (n_params++) ember_buf_putc(&args, ',');
        ember_json_escape(&args, key);
        ember_buf_putc(&args, ':');
        char *raw = dup_range(value, value_end);
        append_coerced_value(&args, raw, schema_for_param(call_schema, key));
        free(raw);
        at = skip_ws(param_close + strlen(PARAM_CLOSE), body_end);
    }
    if ((size_t)(body_end - at) < strlen(FUNC_CLOSE) ||
        strncmp(at, FUNC_CLOSE, strlen(FUNC_CLOSE)) != 0) goto malformed;
    at = skip_ws(at + strlen(FUNC_CLOSE), body_end);
    if (at != body_end) goto malformed;
    ember_buf_putc(&args, '}');

    char *arguments = ember_buf_take(&args);
    for (int i = 0; i < n_keys; ++i) free(keys[i]);
    free(keys);
    if (!push_call(out, name, arguments)) {
        free(name);
        free(arguments);
        return false;
    }
    return true;

malformed:
    free(name);
    for (int i = 0; i < n_keys; ++i) free(keys[i]);
    free(keys);
    ember_buf_free(&args);
    return false;
}

int ember_parse_qwen_tool_calls(const char *text, const char *tools_json,
                                ember_tool_calls *out,
                                ember_qwen_tool_parse_report *report) {
    if (!out) return 0;
    ember_qwen_tool_parse_report local = {0};
    if (!report) report = &local;
    memset(report, 0, sizeof(*report));
    if (!text) return 0;
    const char *text_end = text + strlen(text);

    ember_json *tools = ember_json_parse(tools_json ? tools_json : "[]");
    if (!tools || tools->type != EMBER_JSON_ARRAY) {
        ember_json_free(tools);
        report->malformed = true;
        return 0;
    }

    const char *at = strstr(text, CALL_OPEN);
    if (!at) {
        ember_json_free(tools);
        return 0;
    }
    report->found = true;
    for (;;) {
        const char *body = at + strlen(CALL_OPEN);
        const char *close = strstr(body, CALL_CLOSE);
        if (!close) {
            report->malformed = true;
            break;
        }
        if (!parse_wrapper(body, close, tools, out, report)) {
            report->malformed = true;
            break;
        }
        ++report->wrappers;
        at = skip_ws(close + strlen(CALL_CLOSE), text_end);
        if (!*at) {
            report->complete = true;
            break;
        }
        if (strncmp(at, CALL_OPEN, strlen(CALL_OPEN)) != 0) {
            report->trailing = true;
            break;
        }
    }
    ember_json_free(tools);
    if (!report->complete || report->malformed || report->contaminated ||
        report->trailing) {
        ember_tool_calls_free(out);
        return 0;
    }
    return out->len;
}
