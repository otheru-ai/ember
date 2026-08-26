#include "chat_api.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../common/buf.h"

// #4(d): a syntactically valid JSON number can still overflow to +/-inf via a
// huge exponent (e.g. 1e400), and casting a non-finite or out-of-range double
// straight to int/uint64_t is undefined behavior. These guards reject non-
// finite, fractional, and out-of-range input rather than silently truncating or
// saturating client values.
static bool json_num_to_int(double d, int *out) {
    if (!isfinite(d) || floor(d) != d || d < (double)INT_MIN ||
        d > (double)INT_MAX) return false;
    *out = (int)d;
    return true;
}
static bool json_num_to_u64(const ember_json *v, uint64_t *out) {
    if (!v || v->type != EMBER_JSON_NUMBER || !v->num_raw ||
        !v->num_raw[0] || v->num_raw[0] == '-') return false;
    for (const char *p = v->num_raw; *p; ++p)
        if (*p < '0' || *p > '9') return false;
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(v->num_raw, &end, 10);
    if (errno == ERANGE || !end || *end) return false;
    *out = (uint64_t)parsed;
    return true;
}

static char *dup_or(const char *s, const char *dflt) {
    const char *v = s ? s : dflt;
    if (!v) return NULL;
    char *copy = strdup(v);
    if (!copy) ember_buf_fatal("out of memory parsing chat request");
    return copy;
}

// content is either a string or an array of parts; flatten text parts.
static char *flatten_content(const ember_json *content, bool *ok) {
    *ok = true;
    if (!content) return dup_or("", NULL);
    if (content->type == EMBER_JSON_NULL) return dup_or("", NULL);
    if (content->type == EMBER_JSON_STRING)
        return dup_or(ember_json_str(content, ""), NULL);
    if (content->type == EMBER_JSON_ARRAY) {
        ember_buf b = {0};
        for (int i = 0; i < ember_json_len(content); i++) {
            const ember_json *part = ember_json_at(content, i);
            if (!part || part->type != EMBER_JSON_OBJECT) goto invalid;
            const ember_json *type_node = ember_json_get(part, "type");
            const char *type = ember_json_str(type_node, NULL);
            if (!type) goto invalid;
            if (strcmp(type, "text") == 0) {
                const ember_json *text = ember_json_get(part, "text");
                if (!text || text->type != EMBER_JSON_STRING) goto invalid;
                ember_buf_puts(&b, text->u.str);
            } // Non-text multimodal parts remain intentionally ignored.
        }
        char *s = ember_buf_take(&b);
        return s ? s : dup_or("", NULL);
invalid:
        ember_buf_free(&b);
        *ok = false;
        return NULL;
    }
    *ok = false;
    return NULL;
}

// ds4 parse_reasoning_effort_name: none→NONE(off), minimal..xhigh→HIGH, max→MAX.
// Returns true if recognized. Unknown strings resolve to HIGH (lenient).
static bool effort_to_mode(const char *s, ember_think_mode *mode, bool *enabled) {
    if (!s) return false;
    if (strcmp(s, "none") == 0)     { *mode = EMBER_THINK_NONE; *enabled = false; return true; }
    if (strcmp(s, "max") == 0)      { *mode = EMBER_THINK_MAX;  *enabled = true;  return true; }
    // "x-high" is the spelling used by the model card and by OpenAI; "xhigh"
    // was accepted here historically. Take both, or a request would resolve a
    // think mode here and a different effort tier in model_card.c tier_for().
    if (strcmp(s, "minimal") == 0 || strcmp(s, "low") == 0 ||
        strcmp(s, "medium") == 0 || strcmp(s, "high") == 0 ||
        strcmp(s, "xhigh") == 0 || strcmp(s, "x-high") == 0) {
        *mode = EMBER_THINK_HIGH; *enabled = true; return true;
    }
    *mode = EMBER_THINK_HIGH; *enabled = true; return false;  // unknown → lenient HIGH
}

static void stop_push(ember_chat_request *r, const char *s) {
    if (!s || !s[0]) return;
    if (r->n_stop == INT_MAX ||
        (size_t)(r->n_stop + 1) > SIZE_MAX / sizeof(char *))
        ember_buf_fatal("too many stop strings");
    char **grown = (char **)realloc(
        r->stop, (size_t)(r->n_stop + 1) * sizeof(char *));
    if (!grown) ember_buf_fatal("out of memory parsing stop strings");
    r->stop = grown;
    r->stop[r->n_stop++] = dup_or(s, NULL);
}

static void tool_choice_name_push(ember_chat_request *r, const char *name) {
    if (!name || !name[0]) return;
    for (int i = 0; i < r->n_tool_choice_names; ++i)
        if (!strcmp(r->tool_choice_names[i], name)) return;
    if (r->n_tool_choice_names == INT_MAX ||
        (size_t)(r->n_tool_choice_names + 1) > SIZE_MAX / sizeof(char *))
        ember_buf_fatal("too many tool-choice names");
    char **grown = realloc(
        r->tool_choice_names,
        (size_t)(r->n_tool_choice_names + 1) * sizeof(char *));
    if (!grown) ember_buf_fatal("out of memory parsing tool choice");
    r->tool_choice_names = grown;
    r->tool_choice_names[r->n_tool_choice_names++] = dup_or(name, NULL);
}

static const char *tool_choice_object_name(const ember_json *choice) {
    const ember_json *fn = ember_json_get(choice, "function");
    const char *name = ember_json_str(ember_json_get(fn, "name"), NULL);
    if (!name) name = ember_json_str(ember_json_get(choice, "name"), NULL);
    return name;
}

static bool parse_tool_choice(const ember_json *choice,
                              ember_chat_request *out) {
    out->tool_choice = EMBER_TOOL_CHOICE_AUTO;
    if (!choice || choice->type == EMBER_JSON_NULL) return true;
    if (choice->type == EMBER_JSON_STRING) {
        const char *mode = choice->u.str;
        if (!strcmp(mode, "auto")) return true;
        if (!strcmp(mode, "none")) {
            out->tool_choice = EMBER_TOOL_CHOICE_NONE;
            out->tool_choice_none = true;
            return true;
        }
        if (!strcmp(mode, "required") || !strcmp(mode, "any")) {
            out->tool_choice = EMBER_TOOL_CHOICE_REQUIRED;
            out->tool_choice_required = true;
            return true;
        }
        return false;
    }
    if (choice->type != EMBER_JSON_OBJECT) return false;
    const char *type = ember_json_str(ember_json_get(choice, "type"), "");
    if (!strcmp(type, "function") || !strcmp(type, "tool")) {
        const char *name = tool_choice_object_name(choice);
        if (!name || !name[0]) return false;
        out->tool_choice = EMBER_TOOL_CHOICE_NAMED;
        out->tool_choice_required = true;
        tool_choice_name_push(out, name);
        return true;
    }
    if (!strcmp(type, "allowed_tools")) {
        const ember_json *tools = ember_json_get(choice, "tools");
        const char *mode = ember_json_str(
            ember_json_get(choice, "mode"), "auto");
        if (!tools || tools->type != EMBER_JSON_ARRAY ||
            (strcmp(mode, "auto") && strcmp(mode, "required"))) return false;
        for (int i = 0; i < ember_json_len(tools); ++i) {
            const ember_json *tool = ember_json_at(tools, i);
            const char *name = tool_choice_object_name(tool);
            if (!name || !name[0]) return false;
            tool_choice_name_push(out, name);
        }
        if (out->n_tool_choice_names == 0) return false;
        out->tool_choice = EMBER_TOOL_CHOICE_ALLOWED;
        out->tool_choice_required = !strcmp(mode, "required");
        return true;
    }
    return false;
}

static bool advertised_tool_name(const ember_json *tools, const char *wanted) {
    for (int i = 0; tools && i < ember_json_len(tools); ++i) {
        const ember_json *tool = ember_json_at(tools, i);
        const ember_json *fn = ember_json_get(tool, "function");
        const char *name = ember_json_str(
            ember_json_get(fn ? fn : tool, "name"), NULL);
        if (name && !strcmp(name, wanted)) return true;
    }
    return false;
}

// Parse an OpenAI assistant tool_calls array into msg->calls.
static bool parse_history_tool_calls(const ember_json *arr, ember_chat_msg *msg) {
    if (!arr) return true;
    if (arr->type != EMBER_JSON_ARRAY) return false;
    int n = ember_json_len(arr);
    for (int i = 0; i < n; i++) {
        const ember_json *tc = ember_json_at(arr, i);
        const ember_json *fn = ember_json_get(tc, "function");
        const char *name = ember_json_str(ember_json_get(fn, "name"), NULL);
        if (!tc || tc->type != EMBER_JSON_OBJECT || !fn ||
            fn->type != EMBER_JSON_OBJECT || !name || !name[0]) return false;
        // OpenAI spec: function.arguments is a JSON *string* holding a JSON
        // object. Be tolerant (ds4 parse_function_call): accept a raw object/
        // array too, else the args are lost.
        const ember_json *a = ember_json_get(fn, "arguments");
        char *args_owned;
        if (a && (a->type == EMBER_JSON_OBJECT || a->type == EMBER_JSON_ARRAY))
            args_owned = ember_json_dump(a);
        else {
            if (a && a->type != EMBER_JSON_STRING) return false;
            const char *s = ember_json_str(a, "{}");
            args_owned = dup_or(s && s[0] ? s : "{}", NULL);
        }
        if (msg->calls.len >= msg->calls.cap) {
            if (msg->calls.cap > INT_MAX / 2)
                ember_buf_fatal("too many historical tool calls");
            msg->calls.cap = msg->calls.cap ? msg->calls.cap * 2 : 4;
            ember_tool_call *grown = (ember_tool_call *)realloc(
                msg->calls.calls, (size_t)msg->calls.cap * sizeof(ember_tool_call));
            if (!grown) ember_buf_fatal("out of memory parsing historical tool calls");
            msg->calls.calls = grown;
        }
        msg->calls.calls[msg->calls.len].name = dup_or(name, NULL);
        msg->calls.calls[msg->calls.len].arguments =
            args_owned ? args_owned : dup_or("{}", NULL);
        // B3: keep the tool-call id so exact-DSML replay can look up the sampled
        // bytes for this call (and associate parallel results).
        const char *tcid = ember_json_str(ember_json_get(tc, "id"), NULL);
        msg->calls.calls[msg->calls.len].id = dup_or(tcid, NULL);
        msg->calls.len++;
    }
    return true;
}

static bool valid_sampler_float(double v) {
    return isfinite(v) && v >= -(double)FLT_MAX && v <= (double)FLT_MAX;
}

// GCC 13's path analyzer reports request-owned strings as leaked at the next
// local declaration after assignment. On success ownership is deliberately
// returned through out; every invalid path calls ember_chat_request_free(out).
// ASan/LSan and the parser tests cover both lifetimes. Keep the suppression on
// this ownership-producing function only so unrelated analyzer findings fail.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wanalyzer-malloc-leak"
#endif
bool ember_chat_request_parse(const ember_json *root, ember_chat_request *out) {
    memset(out, 0, sizeof(*out));
    if (!root || root->type != EMBER_JSON_OBJECT ||
        ember_json_has_duplicate_keys(root)) return false;
    out->api = EMBER_API_CHAT;
    out->parallel_tool_calls = true;

    const ember_json *msgs = ember_json_get(root, "messages");
    if (!msgs || msgs->type != EMBER_JSON_ARRAY) return false;

    const ember_json *model = ember_json_get(root, "model");
    if (model && model->type != EMBER_JSON_STRING) goto invalid;
    out->model = dup_or(ember_json_str(model, NULL), "deepseek-v4-flash");
    const ember_json *stream = ember_json_get(root, "stream");
    const ember_json *background = ember_json_get(root, "ember_background");
    if ((stream && stream->type != EMBER_JSON_BOOL) ||
        (background && background->type != EMBER_JSON_BOOL)) goto invalid;
    out->stream = ember_json_bool(stream, false);
    out->background = ember_json_bool(background, false);

    // stream_options.include_usage
    const ember_json *so = ember_json_get(root, "stream_options");
    if (so) {
        if (so->type != EMBER_JSON_OBJECT) goto invalid;
        const ember_json *include = ember_json_get(so, "include_usage");
        if (include && include->type != EMBER_JSON_BOOL) goto invalid;
        out->stream_include_usage = ember_json_bool(include, false);
    }

    const ember_json *mt = ember_json_get(root, "max_tokens");
    if (!mt) mt = ember_json_get(root, "max_completion_tokens");
    if (mt) {
        int v;
        if (mt->type == EMBER_JSON_NUMBER &&
            json_num_to_int(ember_json_num(mt, 0), &v) && v >= 0) {
            out->max_tokens = v;
            out->max_tokens_set = true;
        } else goto invalid;
    }

    // Sampler surface.
    const ember_json *temp = ember_json_get(root, "temperature");
    if (temp) {
        if (temp->type != EMBER_JSON_NUMBER ||
            !valid_sampler_float(temp->u.num) || temp->u.num < 0.0) goto invalid;
        out->temperature = temp->u.num;
        out->temperature_set = true;
    }
    const ember_json *tp = ember_json_get(root, "top_p");
    if (tp) {
        if (tp->type != EMBER_JSON_NUMBER || !valid_sampler_float(tp->u.num) || tp->u.num < 0.0 ||
            tp->u.num > 1.0) goto invalid;
        out->top_p = tp->u.num;
        out->top_p_set = true;
    }
    const ember_json *tk = ember_json_get(root, "top_k");
    if (tk) {
        int v;
        if (tk->type == EMBER_JSON_NUMBER &&
            json_num_to_int(tk->u.num, &v) && v >= 0) {
            out->top_k = v;
            out->top_k_set = true;
        } else goto invalid;
    }
    const ember_json *mp = ember_json_get(root, "min_p");
    if (mp) {
        if (mp->type != EMBER_JSON_NUMBER || !valid_sampler_float(mp->u.num) || mp->u.num < 0.0 ||
            mp->u.num > 1.0) goto invalid;
        out->min_p = mp->u.num;
        out->min_p_set = true;
    }
    const ember_json *seed = ember_json_get(root, "seed");
    if (seed) {
        uint64_t v;
        if (json_num_to_u64(seed, &v)) {
            out->seed = v;
            out->seed_set = true;
        } else goto invalid;
    }
    // Penalties. repetition_penalty (HF/vLLM) with rep_pen alias; OpenAI additive
    // frequency_penalty / presence_penalty.
    const ember_json *rp = ember_json_get(root, "repetition_penalty");
    if (!rp) rp = ember_json_get(root, "rep_pen");
    if (rp) {
        if (rp->type != EMBER_JSON_NUMBER ||
            !valid_sampler_float(rp->u.num) || rp->u.num <= 0.0) goto invalid;
        out->rep_pen = rp->u.num;
        out->rep_pen_set = true;
    }
    const ember_json *rw = ember_json_get(root, "rep_window");
    if (rw) {
        int v;
        if (rw->type == EMBER_JSON_NUMBER &&
            json_num_to_int(rw->u.num, &v) && v >= 0) {
            out->rep_window = v;
            out->rep_window_set = true;
        } else goto invalid;
    }
    const ember_json *fp = ember_json_get(root, "frequency_penalty");
    if (fp) {
        if (fp->type != EMBER_JSON_NUMBER || !valid_sampler_float(fp->u.num) ||
            fp->u.num < -2.0 || fp->u.num > 2.0)
            goto invalid;
        out->freq_pen = fp->u.num;
        out->freq_pen_set = true;
    }
    const ember_json *pp = ember_json_get(root, "presence_penalty");
    if (pp) {
        if (pp->type != EMBER_JSON_NUMBER || !valid_sampler_float(pp->u.num) ||
            pp->u.num < -2.0 || pp->u.num > 2.0)
            goto invalid;
        out->pres_pen = pp->u.num;
        out->pres_pen_set = true;
    }

    // DRY. Names match llama.cpp so existing clients and tuning guides carry
    // over unchanged. Only dry_multiplier arms it; the rest refine an already
    // active penalty, so they are accepted but inert on their own.
    const ember_json *dm = ember_json_get(root, "dry_multiplier");
    if (dm) {
        if (dm->type != EMBER_JSON_NUMBER || !valid_sampler_float(dm->u.num) ||
            dm->u.num < 0.0 || dm->u.num > 100.0)
            goto invalid;
        out->dry_multiplier = dm->u.num;
        out->dry_multiplier_set = true;
    }
    const ember_json *db = ember_json_get(root, "dry_base");
    if (db) {
        // <=1 makes the penalty non-increasing in match length, i.e. it stops
        // being DRY at all. Reject rather than silently ignore.
        if (db->type != EMBER_JSON_NUMBER || !valid_sampler_float(db->u.num) ||
            db->u.num <= 1.0 || db->u.num > 8.0)
            goto invalid;
        out->dry_base = db->u.num;
        out->dry_base_set = true;
    }
    const ember_json *dal = ember_json_get(root, "dry_allowed_length");
    if (dal) {
        int v;
        if (dal->type != EMBER_JSON_NUMBER ||
            !json_num_to_int(dal->u.num, &v) || v < 0 || v > 4096)
            goto invalid;
        out->dry_allowed_length = v;
        out->dry_allowed_length_set = true;
    }
    const ember_json *dw = ember_json_get(root, "dry_penalty_last_n");
    if (!dw) dw = ember_json_get(root, "dry_window");
    if (dw) {
        // llama.cpp uses -1 for "whole context"; the sampler reads <=0 the same
        // way, so both spellings land on the same behaviour.
        int v;
        if (dw->type != EMBER_JSON_NUMBER ||
            !json_num_to_int(dw->u.num, &v) || v < -1 || v > 1048576)
            goto invalid;
        out->dry_window = v;
        out->dry_window_set = true;
    }

    // stop: string or array of strings.
    const ember_json *stop = ember_json_get(root, "stop");
    if (stop) {
        if (stop->type == EMBER_JSON_STRING) stop_push(out, ember_json_str(stop, NULL));
        else if (stop->type == EMBER_JSON_ARRAY) {
            for (int i = 0; i < ember_json_len(stop); i++) {
                const ember_json *item = ember_json_at(stop, i);
                if (!item || item->type != EMBER_JSON_STRING) goto invalid;
                stop_push(out, item->u.str);
            }
        } else if (stop->type != EMBER_JSON_NULL) goto invalid;
    }

    // Thinking / reasoning-effort. ds4 default: thinking ON (HIGH).
    out->thinking_enabled = true;
    out->think_mode = EMBER_THINK_HIGH;
    const ember_json *effort = ember_json_get(root, "reasoning_effort");
    if (effort && effort->type != EMBER_JSON_STRING) goto invalid;
    out->reasoning_effort = dup_or(ember_json_str(effort, NULL), NULL);
    if (out->reasoning_effort)
        effort_to_mode(out->reasoning_effort, &out->think_mode, &out->thinking_enabled);
    const ember_json *reasoning_budget =
        ember_json_get(root, "reasoning_budget_tokens");
    if (!reasoning_budget)
        reasoning_budget = ember_json_get(root, "thinking_token_budget");
    if (!reasoning_budget)
        reasoning_budget = ember_json_get(root, "thinking_budget_tokens");
    if (reasoning_budget) {
        int v;
        if (reasoning_budget->type != EMBER_JSON_NUMBER ||
            !json_num_to_int(ember_json_num(reasoning_budget, 0), &v) ||
            v < 0) {
            goto invalid;
        }
        out->reasoning_budget_tokens = v;
        out->reasoning_budget_tokens_set = true;
    }
    // Explicit overrides. Keep the historical boolean spellings and accept
    // DeepSeek's current OpenAI-compatible object shape, emitted by Reasonix
    // v1.31.3 and deepseek-harness
    // packages/llm/llm-deepseek/src/serialize.ts:343-367
    // (`thinking.type=enabled|disabled`). `enable_thinking` retains precedence
    // when a client supplies both fields.
    const ember_json *enable_thinking =
        ember_json_get(root, "enable_thinking");
    const ember_json *thinking = ember_json_get(root, "thinking");
    const ember_json *et = enable_thinking ? enable_thinking : thinking;
    if (et) {
        if (et->type == EMBER_JSON_BOOL) {
            out->thinking_enabled = et->u.b;
        } else if (!enable_thinking && et->type == EMBER_JSON_OBJECT) {
            const char *type = ember_json_str(
                ember_json_get(et, "type"), NULL);
            if (!type || (strcmp(type, "enabled") && strcmp(type, "disabled")))
                goto invalid;
            out->thinking_enabled = strcmp(type, "disabled") != 0;
        } else {
            goto invalid;
        }
        if (!out->thinking_enabled) out->think_mode = EMBER_THINK_NONE;
    }

    const ember_json *tc = ember_json_get(root, "tool_choice");
    if (!parse_tool_choice(tc, out)) goto invalid;
    const ember_json *parallel = ember_json_get(root, "parallel_tool_calls");
    if (parallel) {
        if (parallel->type != EMBER_JSON_BOOL) goto invalid;
        out->parallel_tool_calls = parallel->u.b;
    }

    const ember_json *tools = ember_json_get(root, "tools");
    if (tools && tools->type != EMBER_JSON_ARRAY) goto invalid;
    for (int i = 0; tools && i < ember_json_len(tools); ++i) {
        const ember_json *tool = ember_json_at(tools, i);
        const ember_json *fn = ember_json_get(tool, "function");
        if (!tool || tool->type != EMBER_JSON_OBJECT || !fn ||
            fn->type != EMBER_JSON_OBJECT) goto invalid;
        const ember_json *tool_name = ember_json_get(fn, "name");
        const ember_json *params = ember_json_get(fn, "parameters");
        if (!tool_name || tool_name->type != EMBER_JSON_STRING ||
            !tool_name->u.str[0] ||
            (params && params->type != EMBER_JSON_OBJECT &&
             params->type != EMBER_JSON_BOOL)) goto invalid;
    }
    if (tools && tools->type == EMBER_JSON_ARRAY && ember_json_len(tools) > 0 &&
        !out->tool_choice_none) {
        out->has_tools = true;
        out->tools_json = ember_json_dump(tools);
    }
    if (out->tool_choice_required && !out->has_tools) goto invalid;
    for (int i = 0; i < out->n_tool_choice_names; ++i)
        if (!advertised_tool_name(tools, out->tool_choice_names[i])) goto invalid;

    int n = ember_json_len(msgs);
    out->messages = (ember_chat_msg *)calloc((size_t)(n > 0 ? n : 1),
                                             sizeof(ember_chat_msg));
    if (!out->messages) ember_buf_fatal("out of memory parsing messages");
    out->n_messages = n;
    for (int i = 0; i < n; i++) {
        const ember_json *m = ember_json_at(msgs, i);
        if (!m || m->type != EMBER_JSON_OBJECT) goto invalid;
        const ember_json *role = ember_json_get(m, "role");
        const ember_json *name = ember_json_get(m, "name");
        const ember_json *reasoning = ember_json_get(m, "reasoning_content");
        const ember_json *call_id = ember_json_get(m, "tool_call_id");
        if (!role || role->type != EMBER_JSON_STRING || !role->u.str[0] ||
            (name && name->type != EMBER_JSON_STRING) ||
            (reasoning && reasoning->type != EMBER_JSON_STRING) ||
            (call_id && call_id->type != EMBER_JSON_STRING)) goto invalid;
        out->messages[i].role = dup_or(role->u.str, "");
        bool content_ok = false;
        out->messages[i].content =
            flatten_content(ember_json_get(m, "content"), &content_ok);
        if (!content_ok) goto invalid;
        out->messages[i].name =
            dup_or(ember_json_str(name, NULL), NULL);
        out->messages[i].reasoning =
            dup_or(ember_json_str(reasoning, NULL), NULL);
        out->messages[i].tool_call_id =  // B3: associates a tool result with its call
            dup_or(ember_json_str(call_id, NULL), NULL);
        if (!parse_history_tool_calls(
                ember_json_get(m, "tool_calls"), &out->messages[i])) goto invalid;
    }
    out->continuation_only = n > 0;
    for (int i = 0; i < n; ++i) {
        const char *role = out->messages[i].role;
        if ((strcmp(role, "tool") != 0 && strcmp(role, "function") != 0) ||
            !out->messages[i].tool_call_id ||
            !out->messages[i].tool_call_id[0]) {
            out->continuation_only = false;
            break;
        }
    }
    return true;

invalid:
    ember_chat_request_free(out);
    return false;
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

bool ember_chat_request_is_tool_result_continuation(
        const ember_chat_request *r) {
    if (!r) return false;
    if (r->continuation_only) return true;
    if (r->n_messages <= 0) return false;
    const char *role = r->messages[r->n_messages - 1].role;
    return role &&
        (strcmp(role, "tool") == 0 || strcmp(role, "function") == 0);
}

static bool tool_result_role(const ember_chat_msg *m) {
    return m && m->role &&
        (!strcmp(m->role, "tool") || !strcmp(m->role, "function"));
}

typedef struct {
    int assistant_idx;
    int results_start;
    int results_end;
} tool_history_round;

// Find the newest complete round ending at or before `end`. Tool results must
// be contiguous and immediately follow an assistant tool-call message; this is
// the normal OpenAI full-history shape and avoids matching across user turns.
static bool trailing_tool_round(const ember_chat_request *r, int end,
                                tool_history_round *out) {
    if (!r || r->continuation_only || end < 1 ||
        !tool_result_role(&r->messages[end]))
        return false;
    int start = end;
    while (start > 0 && tool_result_role(&r->messages[start - 1])) --start;
    int ai = start - 1;
    if (ai < 0 || !r->messages[ai].role ||
        strcmp(r->messages[ai].role, "assistant") ||
        r->messages[ai].calls.len <= 0)
        return false;
    out->assistant_idx = ai;
    out->results_start = start;
    out->results_end = end;
    return true;
}

static int matching_result_index(const ember_chat_request *r,
                                 const tool_history_round *round,
                                 const ember_tool_call *call, int call_index,
                                 bool *used) {
    // Prefer the explicit id binding within a round. The id is used only to
    // associate a result with its call; it is never compared across rounds.
    if (call->id && call->id[0]) {
        for (int i = round->results_start; i <= round->results_end; ++i) {
            const ember_chat_msg *m = &r->messages[i];
            if (!used[i - round->results_start] && m->tool_call_id &&
                !strcmp(m->tool_call_id, call->id))
                return i;
        }
        return -1;
    }

    // Legacy function-role histories may not carry ids. Preserve ordered-set
    // semantics by pairing the nth call with the nth result in that case.
    int i = round->results_start + call_index;
    if (i > round->results_end || used[i - round->results_start]) return -1;
    return i;
}

static char *normalized_arguments(const char *arguments) {
    ember_json *j = ember_json_parse(arguments ? arguments : "{}");
    if (!j) return NULL;
    char *dump = ember_json_dump(j);
    ember_json_free(j);
    return dump;
}

// Name + normalized arguments only. Shared by both tool-loop signals so the
// notion of "the same call" cannot drift between them.
static bool call_args_equal(const ember_tool_call *at, const ember_tool_call *bt) {
    if (strcmp(at->name ? at->name : "", bt->name ? bt->name : "")) return false;
    char *aa = normalized_arguments(at->arguments);
    char *ba = normalized_arguments(bt->arguments);
    const bool equal = aa && ba && !strcmp(aa, ba);
    free(aa);
    free(ba);
    return equal;
}

static bool tool_rounds_equal(const ember_chat_request *r,
                              const tool_history_round *a,
                              const tool_history_round *b) {
    const ember_tool_calls *ac = &r->messages[a->assistant_idx].calls;
    const ember_tool_calls *bc = &r->messages[b->assistant_idx].calls;
    if (ac->len != bc->len || ac->len <= 0) return false;
    int an = a->results_end - a->results_start + 1;
    int bn = b->results_end - b->results_start + 1;
    if (an != ac->len || bn != bc->len) return false;
    bool *a_used = calloc((size_t)an, sizeof(*a_used));
    bool *b_used = calloc((size_t)bn, sizeof(*b_used));
    if (!a_used || !b_used)
        ember_buf_fatal("out of memory comparing tool-loop history");
    bool equal = true;
    for (int i = 0; i < ac->len && equal; ++i) {
        const ember_tool_call *at = &ac->calls[i];
        const ember_tool_call *bt = &bc->calls[i];
        if (!call_args_equal(at, bt)) {
            equal = false;
            break;
        }
        int ar = matching_result_index(r, a, at, i, a_used);
        int br = matching_result_index(r, b, bt, i, b_used);
        if (ar < 0 || br < 0) {
            equal = false;
            break;
        }
        a_used[ar - a->results_start] = true;
        b_used[br - b->results_start] = true;
        const char *av = r->messages[ar].content ? r->messages[ar].content : "";
        const char *bv = r->messages[br].content ? r->messages[br].content : "";
        if (strcmp(av, bv)) equal = false;
    }
    free(a_used);
    free(b_used);
    return equal;
}

int ember_chat_request_tool_loop_rounds(const ember_chat_request *r) {
    if (!r || r->continuation_only || r->n_messages < 4) return 0;
    tool_history_round newest;
    if (!trailing_tool_round(r, r->n_messages - 1, &newest)) return 0;
    int rounds = 1;
    tool_history_round cursor = newest;
    for (;;) {
        tool_history_round previous;
        if (!trailing_tool_round(r, cursor.assistant_idx - 1, &previous) ||
            !tool_rounds_equal(r, &newest, &previous))
            break;
        ++rounds;
        cursor = previous;
    }
    return rounds > 1 ? rounds : 0;
}

// Newest assistant message at or before `end` that carries tool calls, or -1.
static int assistant_calls_at_or_before(const ember_chat_request *r, int end) {
    for (int i = end; i >= 0; --i) {
        const ember_chat_msg *m = &r->messages[i];
        if (m->role && !strcmp(m->role, "assistant") && m->calls.len > 0) return i;
    }
    return -1;
}

int ember_chat_request_tool_loop_calls(const ember_chat_request *r) {
    if (!r || r->continuation_only || r->n_messages < 2) return 0;
    const int newest = assistant_calls_at_or_before(r, r->n_messages - 1);
    if (newest < 0) return 0;
    const ember_tool_calls *nc = &r->messages[newest].calls;
    int count = 1;
    int cursor = newest;
    for (;;) {
        const int prev = assistant_calls_at_or_before(r, cursor - 1);
        if (prev < 0) break;
        const ember_tool_calls *pc = &r->messages[prev].calls;
        if (pc->len != nc->len) break;
        bool same = true;
        for (int i = 0; i < nc->len && same; ++i)
            same = call_args_equal(&nc->calls[i], &pc->calls[i]);
        if (!same) break;
        ++count;
        cursor = prev;
    }
    return count > 1 ? count : 0;
}

const char *ember_chat_request_tool_loop_tool(const ember_chat_request *r) {
    if (!r) return NULL;
    tool_history_round newest;
    if (trailing_tool_round(r, r->n_messages - 1, &newest)) {
        const ember_tool_calls *calls = &r->messages[newest.assistant_idx].calls;
        if (calls->len > 0) return calls->calls[0].name;
    }
    // Fall back to the newest tool-calling assistant turn, so the call-signature
    // signal can still be labelled when the history ends on a user turn or on
    // assistant prose — both of which leave no COMPLETE trailing round. Purely
    // additive: this returns a name only where NULL was returned before.
    const int i = assistant_calls_at_or_before(r, r->n_messages - 1);
    return i >= 0 ? r->messages[i].calls.calls[0].name : NULL;
}

static bool is_tool_result(const ember_chat_msg *m) {
    return m->role && !strcmp(m->role, "tool");
}

// The tool that produced result message `idx`: the explicit `name` field when
// the client sends one, else the call whose id it answers. Never NULL, so the
// effect key stays well-defined for legacy histories that carry neither.
static const char *result_tool_name(const ember_chat_request *r, int idx) {
    const ember_chat_msg *m = &r->messages[idx];
    if (m->name && m->name[0]) return m->name;
    if (m->tool_call_id && m->tool_call_id[0]) {
        for (int i = idx - 1; i >= 0; --i) {
            const ember_tool_calls *c = &r->messages[i].calls;
            for (int k = 0; k < c->len; ++k)
                if (c->calls[k].id && !strcmp(c->calls[k].id, m->tool_call_id))
                    return c->calls[k].name ? c->calls[k].name : "";
        }
    }
    return "";
}

// Trailing harness annotations: text a client appends to a tool RESULT after
// the tool's own payload. Excluded from the effect key, because a client that
// appends a repetition COUNTER to the very result it is warning about destroys
// the byte identity every downstream consumer needs -- including this one.
//
// Regression analysis showed that treating these annotations as part of the
// result hides stalled rounds even though the underlying payload is unchanged.
//
// DELIBERATELY NARROW, in the shape of tool_parser.c's SYNTAX[] table: only
// annotations that describe the repetition itself belong here. Some clients
// also append "[OUT-OF-BAND USER MESSAGE ...]", which is new input rather than
// metadata about a repeat -- trimming that would erase a real difference, so
// it is excluded on purpose. The real fix is upstream: a harness should report
// a repeat beside the result, not inside it.
static const char *const RESULT_ANNOTATIONS[] = {
    "[Tool loop warning:",      // common client annotation; carries "count=N"
};

// Length of `s` up to the last trailing annotation, or its full length. The
// annotation must start its own block (blank line before it) so the marker
// appearing inside a tool's own output cannot truncate the payload.
static size_t result_payload_len(const char *s) {
    if (!s) return 0;
    size_t len = strlen(s);
    for (size_t k = 0; k < sizeof RESULT_ANNOTATIONS / sizeof *RESULT_ANNOTATIONS; ++k) {
        const char *hit = NULL;
        for (const char *p = s; (p = strstr(p, RESULT_ANNOTATIONS[k])) != NULL; ++p)
            hit = p;                                  // last wins
        if (!hit) continue;
        size_t i = (size_t)(hit - s), newlines = 0;
        while (i > 0 && isspace((unsigned char)s[i - 1])) {
            if (s[i - 1] == '\n') ++newlines;
            --i;
        }
        if (newlines >= 2 && i < len) len = i;
    }
    return len;
}

// True when (tool, payload) already appeared as a tool result before `before`.
// `plen` is the per-request payload length of every message, computed once:
// recomputing it per comparison would make this quadratic in result BYTES.
static bool effect_seen_before(const ember_chat_request *r, const size_t *plen,
                               int before, const char *name, int probe) {
    for (int i = 0; i < before; ++i) {
        if (!is_tool_result(&r->messages[i])) continue;
        if (plen[i] != plen[probe]) continue;
        if (memcmp(r->messages[i].content ? r->messages[i].content : "",
                   r->messages[probe].content ? r->messages[probe].content : "",
                   plen[probe]))
            continue;
        if (!strcmp(result_tool_name(r, i), name)) return true;
    }
    return false;
}

static bool has_visible_text(const char *s) {
    for (; s && *s; ++s)
        if (!isspace((unsigned char)*s)) return true;
    return false;
}

int ember_chat_request_progress_lease(const ember_chat_request *r,
                                      const char **stalled_tool) {
    if (stalled_tool) *stalled_tool = NULL;
    if (!r || r->continuation_only || r->n_messages < 2) return 0;
    int lease = 0;
    const char *stalled = NULL;
    size_t *plen = calloc((size_t)r->n_messages, sizeof(*plen));
    if (!plen) ember_buf_fatal("out of memory measuring tool-result payloads");
    for (int i = 0; i < r->n_messages; ++i)
        if (is_tool_result(&r->messages[i]))
            plen[i] = result_payload_len(r->messages[i].content);
    for (int i = 0; i < r->n_messages; ++i) {
        const ember_chat_msg *m = &r->messages[i];
        if (!m->role || strcmp(m->role, "assistant")) continue;
        if (m->calls.len <= 0) {
            // A turn that spoke to the client is progress by definition.
            if (has_visible_text(m->content)) {
                lease = 0;
                stalled = NULL;
            }
            continue;
        }
        // Prose ALONGSIDE a call deliberately does not renew: a stalling model
        // narrates every round ("let me try that again"), and counting that as
        // a completion would neuter the signal. The corpus could not separate
        // the two rules -- both produce an identical distribution -- so this
        // picks the one that cannot be talked out of firing.
        bool novel = false, evaluable = false;
        const char *stale_name = NULL;
        for (int j = i + 1; j < r->n_messages; ++j) {
            const ember_chat_msg *t = &r->messages[j];
            if (t->role && !strcmp(t->role, "assistant")) break;
            if (!is_tool_result(t)) continue;
            evaluable = true;
            const char *name = result_tool_name(r, j);
            if (effect_seen_before(r, plen, j, name, j))
                stale_name = name;
            else
                novel = true;
        }
        // An open round -- calls emitted, results not back yet -- is the turn
        // being served right now. It has produced no effect to judge.
        if (!evaluable) continue;
        if (novel) {
            lease = 0;
            stalled = NULL;
        } else {
            ++lease;
            stalled = stale_name;
        }
    }
    free(plen);
    if (stalled_tool && lease > 0) *stalled_tool = stalled;
    return lease;
}

void ember_chat_request_free(ember_chat_request *r) {
    free(r->model);
    free(r->raw_prompt);
    free(r->tools_json);
    free(r->reasoning_effort);
    free(r->continuation_key);
    for (int i = 0; i < r->n_tool_choice_names; ++i)
        free(r->tool_choice_names[i]);
    free(r->tool_choice_names);
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
