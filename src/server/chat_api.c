#include "chat_api.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../common/buf.h"

// #4(d): a syntactically valid JSON number can still overflow to +/-inf via a
// huge exponent (e.g. 1e400), and casting a non-finite or out-of-range double
// straight to int/uint64_t is undefined behavior. These guards reject non-
// finite input (returning false so the caller leaves the field unset) and clamp
// finite values into the target integer range before the cast.
static bool json_num_to_int(double d, int *out) {
    if (!isfinite(d)) return false;
    if (d <= (double)INT_MIN) { *out = INT_MIN; return true; }
    if (d >= (double)INT_MAX) { *out = INT_MAX; return true; }
    *out = (int)d;
    return true;
}
static bool json_num_to_u64(double d, uint64_t *out) {
    if (!isfinite(d)) return false;
    if (d <= 0.0) { *out = 0; return true; }
    if (d >= 18446744073709551616.0) { *out = UINT64_MAX; return true; }  // 2^64
    *out = (uint64_t)d;
    return true;
}

static char *dup_or(const char *s, const char *dflt) {
    const char *v = s ? s : dflt;
    return v ? strdup(v) : NULL;
}

// content is either a string or an array of parts; flatten text parts.
static char *flatten_content(const ember_json *content) {
    if (!content) return strdup("");
    if (content->type == EMBER_JSON_STRING) return strdup(ember_json_str(content, ""));
    if (content->type == EMBER_JSON_ARRAY) {
        ember_buf b = {0};
        for (int i = 0; i < ember_json_len(content); i++) {
            const ember_json *part = ember_json_at(content, i);
            const char *type = ember_json_str(ember_json_get(part, "type"), "");
            if (strcmp(type, "text") == 0) {
                ember_buf_puts(&b, ember_json_str(ember_json_get(part, "text"), ""));
            }
        }
        char *s = ember_buf_take(&b);
        return s ? s : strdup("");
    }
    return strdup("");
}

// ds4 parse_reasoning_effort_name: none→NONE(off), minimal..xhigh→HIGH, max→MAX.
// Returns true if recognized. Unknown strings resolve to HIGH (lenient).
static bool effort_to_mode(const char *s, ember_think_mode *mode, bool *enabled) {
    if (!s) return false;
    if (strcmp(s, "none") == 0)     { *mode = EMBER_THINK_NONE; *enabled = false; return true; }
    if (strcmp(s, "max") == 0)      { *mode = EMBER_THINK_MAX;  *enabled = true;  return true; }
    if (strcmp(s, "minimal") == 0 || strcmp(s, "low") == 0 ||
        strcmp(s, "medium") == 0 || strcmp(s, "high") == 0 || strcmp(s, "xhigh") == 0) {
        *mode = EMBER_THINK_HIGH; *enabled = true; return true;
    }
    *mode = EMBER_THINK_HIGH; *enabled = true; return false;  // unknown → lenient HIGH
}

static void stop_push(ember_chat_request *r, const char *s) {
    if (!s || !s[0]) return;
    r->stop = (char **)realloc(r->stop, (size_t)(r->n_stop + 1) * sizeof(char *));
    r->stop[r->n_stop++] = strdup(s);
}

// Parse an OpenAI assistant tool_calls array into msg->calls.
static void parse_history_tool_calls(const ember_json *arr, ember_chat_msg *msg) {
    if (!arr || arr->type != EMBER_JSON_ARRAY) return;
    int n = ember_json_len(arr);
    for (int i = 0; i < n; i++) {
        const ember_json *tc = ember_json_at(arr, i);
        const ember_json *fn = ember_json_get(tc, "function");
        const char *name = ember_json_str(ember_json_get(fn, "name"), NULL);
        if (!name) continue;
        // OpenAI spec: function.arguments is a JSON *string* holding a JSON
        // object. Be tolerant (ds4 parse_function_call): accept a raw object/
        // array too, else the args are lost.
        const ember_json *a = ember_json_get(fn, "arguments");
        char *args_owned;
        if (a && (a->type == EMBER_JSON_OBJECT || a->type == EMBER_JSON_ARRAY))
            args_owned = ember_json_dump(a);
        else {
            const char *s = ember_json_str(a, "{}");
            args_owned = strdup(s && s[0] ? s : "{}");
        }
        if (msg->calls.len >= msg->calls.cap) {
            msg->calls.cap = msg->calls.cap ? msg->calls.cap * 2 : 4;
            msg->calls.calls = (ember_tool_call *)realloc(
                msg->calls.calls, (size_t)msg->calls.cap * sizeof(ember_tool_call));
        }
        msg->calls.calls[msg->calls.len].name = strdup(name);
        msg->calls.calls[msg->calls.len].arguments = args_owned ? args_owned : strdup("{}");
        // B3: keep the tool-call id so exact-DSML replay can look up the sampled
        // bytes for this call (and associate parallel results).
        const char *tcid = ember_json_str(ember_json_get(tc, "id"), NULL);
        msg->calls.calls[msg->calls.len].id = tcid ? strdup(tcid) : NULL;
        msg->calls.len++;
    }
}

bool ember_chat_request_parse(const ember_json *root, ember_chat_request *out) {
    memset(out, 0, sizeof(*out));
    if (!root || root->type != EMBER_JSON_OBJECT) return false;

    const ember_json *msgs = ember_json_get(root, "messages");
    if (!msgs || msgs->type != EMBER_JSON_ARRAY) return false;

    out->model = dup_or(ember_json_str(ember_json_get(root, "model"), NULL),
                        "deepseek-v4-flash");
    out->stream = ember_json_bool(ember_json_get(root, "stream"), false);

    // stream_options.include_usage
    const ember_json *so = ember_json_get(root, "stream_options");
    if (so && so->type == EMBER_JSON_OBJECT)
        out->stream_include_usage =
            ember_json_bool(ember_json_get(so, "include_usage"), false);

    const ember_json *mt = ember_json_get(root, "max_tokens");
    if (!mt) mt = ember_json_get(root, "max_completion_tokens");
    if (mt && mt->type == EMBER_JSON_NUMBER) {
        int v;
        if (json_num_to_int(ember_json_num(mt, 0), &v)) {  // #4(d)
            out->max_tokens = v;
            out->max_tokens_set = true;
        }
    }

    // Sampler surface.
    const ember_json *temp = ember_json_get(root, "temperature");
    if (temp && temp->type == EMBER_JSON_NUMBER) { out->temperature = temp->u.num; out->temperature_set = true; }
    const ember_json *tp = ember_json_get(root, "top_p");
    if (tp && tp->type == EMBER_JSON_NUMBER) { out->top_p = tp->u.num; out->top_p_set = true; }
    const ember_json *tk = ember_json_get(root, "top_k");
    if (tk && tk->type == EMBER_JSON_NUMBER) {
        int v;
        if (json_num_to_int(tk->u.num, &v)) { out->top_k = v; out->top_k_set = true; }  // #4(d)
    }
    const ember_json *mp = ember_json_get(root, "min_p");
    if (mp && mp->type == EMBER_JSON_NUMBER) { out->min_p = mp->u.num; out->min_p_set = true; }
    const ember_json *seed = ember_json_get(root, "seed");
    if (seed && seed->type == EMBER_JSON_NUMBER) {
        uint64_t v;
        if (json_num_to_u64(seed->u.num, &v)) { out->seed = v; out->seed_set = true; }  // #4(d)
    }
    // Penalties. repetition_penalty (HF/vLLM) with rep_pen alias; OpenAI additive
    // frequency_penalty / presence_penalty.
    const ember_json *rp = ember_json_get(root, "repetition_penalty");
    if (!rp) rp = ember_json_get(root, "rep_pen");
    if (rp && rp->type == EMBER_JSON_NUMBER) { out->rep_pen = rp->u.num; out->rep_pen_set = true; }
    const ember_json *rw = ember_json_get(root, "rep_window");
    if (rw && rw->type == EMBER_JSON_NUMBER) {
        int v;
        if (json_num_to_int(rw->u.num, &v)) { out->rep_window = v; out->rep_window_set = true; }  // #4(d)
    }
    out->freq_pen = ember_json_num(ember_json_get(root, "frequency_penalty"), 0.0);
    out->pres_pen = ember_json_num(ember_json_get(root, "presence_penalty"), 0.0);

    // stop: string or array of strings.
    const ember_json *stop = ember_json_get(root, "stop");
    if (stop) {
        if (stop->type == EMBER_JSON_STRING) stop_push(out, ember_json_str(stop, NULL));
        else if (stop->type == EMBER_JSON_ARRAY)
            for (int i = 0; i < ember_json_len(stop); i++)
                stop_push(out, ember_json_str(ember_json_at(stop, i), NULL));
    }

    // Thinking / reasoning-effort. ds4 default: thinking ON (HIGH).
    out->thinking_enabled = true;
    out->think_mode = EMBER_THINK_HIGH;
    out->reasoning_effort =
        dup_or(ember_json_str(ember_json_get(root, "reasoning_effort"), NULL), NULL);
    if (out->reasoning_effort)
        effort_to_mode(out->reasoning_effort, &out->think_mode, &out->thinking_enabled);
    // Explicit boolean overrides (enable_thinking / thinking).
    const ember_json *et = ember_json_get(root, "enable_thinking");
    if (!et) et = ember_json_get(root, "thinking");
    if (et && et->type == EMBER_JSON_BOOL) {
        out->thinking_enabled = et->u.b;
        if (!out->thinking_enabled) out->think_mode = EMBER_THINK_NONE;
    }

    // tool_choice: honor "none" (disables tools). Object/"required" left as auto.
    const ember_json *tc = ember_json_get(root, "tool_choice");
    if (tc && tc->type == EMBER_JSON_STRING &&
        strcmp(ember_json_str(tc, ""), "none") == 0)
        out->tool_choice_none = true;

    const ember_json *tools = ember_json_get(root, "tools");
    if (tools && tools->type == EMBER_JSON_ARRAY && ember_json_len(tools) > 0 &&
        !out->tool_choice_none) {
        out->has_tools = true;
        out->tools_json = ember_json_dump(tools);
    }

    int n = ember_json_len(msgs);
    out->messages = (ember_chat_msg *)calloc((size_t)(n > 0 ? n : 1),
                                             sizeof(ember_chat_msg));
    out->n_messages = n;
    for (int i = 0; i < n; i++) {
        const ember_json *m = ember_json_at(msgs, i);
        out->messages[i].role = dup_or(ember_json_str(ember_json_get(m, "role"), ""), "");
        out->messages[i].content = flatten_content(ember_json_get(m, "content"));
        out->messages[i].name =
            dup_or(ember_json_str(ember_json_get(m, "name"), NULL), NULL);
        out->messages[i].reasoning =
            dup_or(ember_json_str(ember_json_get(m, "reasoning_content"), NULL), NULL);
        out->messages[i].tool_call_id =  // B3: associates a tool result with its call
            dup_or(ember_json_str(ember_json_get(m, "tool_call_id"), NULL), NULL);
        parse_history_tool_calls(ember_json_get(m, "tool_calls"), &out->messages[i]);
    }
    return true;
}

void ember_chat_request_free(ember_chat_request *r) {
    free(r->model);
    free(r->tools_json);
    free(r->reasoning_effort);
    for (int i = 0; i < r->n_stop; i++) free(r->stop[i]);
    free(r->stop);
    for (int i = 0; i < r->n_messages; i++) {
        free(r->messages[i].role);
        free(r->messages[i].content);
        free(r->messages[i].name);
        free(r->messages[i].reasoning);
        free(r->messages[i].tool_call_id);  // B3
        free(r->messages[i].raw_tool_text); // B3
        ember_tool_calls_free(&r->messages[i].calls);
    }
    free(r->messages);
    memset(r, 0, sizeof(*r));
}
