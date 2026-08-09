#include "chat_api.h"

#include <float.h>
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
    if (!v) return NULL;
    char *copy = strdup(v);
    if (!copy) ember_buf_fatal("out of memory parsing chat request");
    return copy;
}

// content is either a string or an array of parts; flatten text parts.
static char *flatten_content(const ember_json *content) {
    if (!content) return dup_or("", NULL);
    if (content->type == EMBER_JSON_STRING)
        return dup_or(ember_json_str(content, ""), NULL);
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
        return s ? s : dup_or("", NULL);
    }
    return dup_or("", NULL);
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
}

static bool valid_sampler_float(double v) {
    return isfinite(v) && v >= -(double)FLT_MAX && v <= (double)FLT_MAX;
}

bool ember_chat_request_parse(const ember_json *root, ember_chat_request *out) {
    memset(out, 0, sizeof(*out));
    if (!root || root->type != EMBER_JSON_OBJECT) return false;
    out->api = EMBER_API_CHAT;
    out->parallel_tool_calls = true;

    const ember_json *msgs = ember_json_get(root, "messages");
    if (!msgs || msgs->type != EMBER_JSON_ARRAY) return false;

    out->model = dup_or(ember_json_str(ember_json_get(root, "model"), NULL),
                        "deepseek-v4-flash");
    out->stream = ember_json_bool(ember_json_get(root, "stream"), false);
    out->background =
        ember_json_bool(ember_json_get(root, "ember_background"), false);

    // stream_options.include_usage
    const ember_json *so = ember_json_get(root, "stream_options");
    if (so && so->type == EMBER_JSON_OBJECT)
        out->stream_include_usage =
            ember_json_bool(ember_json_get(so, "include_usage"), false);

    const ember_json *mt = ember_json_get(root, "max_tokens");
    if (!mt) mt = ember_json_get(root, "max_completion_tokens");
    if (mt && mt->type == EMBER_JSON_NUMBER) {
        int v;
        if (json_num_to_int(ember_json_num(mt, 0), &v) && v >= 0) {  // #4(d)
            out->max_tokens = v;
            out->max_tokens_set = true;
        } else goto invalid;
    }

    // Sampler surface.
    const ember_json *temp = ember_json_get(root, "temperature");
    if (temp && temp->type == EMBER_JSON_NUMBER) {
        if (!valid_sampler_float(temp->u.num) || temp->u.num < 0.0) goto invalid;
        out->temperature = temp->u.num;
        out->temperature_set = true;
    }
    const ember_json *tp = ember_json_get(root, "top_p");
    if (tp && tp->type == EMBER_JSON_NUMBER) {
        if (!valid_sampler_float(tp->u.num) || tp->u.num < 0.0 ||
            tp->u.num > 1.0) goto invalid;
        out->top_p = tp->u.num;
        out->top_p_set = true;
    }
    const ember_json *tk = ember_json_get(root, "top_k");
    if (tk && tk->type == EMBER_JSON_NUMBER) {
        int v;
        if (json_num_to_int(tk->u.num, &v) && v >= 0) {
            out->top_k = v;
            out->top_k_set = true;
        } else goto invalid;
    }
    const ember_json *mp = ember_json_get(root, "min_p");
    if (mp && mp->type == EMBER_JSON_NUMBER) {
        if (!valid_sampler_float(mp->u.num) || mp->u.num < 0.0 ||
            mp->u.num > 1.0) goto invalid;
        out->min_p = mp->u.num;
        out->min_p_set = true;
    }
    const ember_json *seed = ember_json_get(root, "seed");
    if (seed && seed->type == EMBER_JSON_NUMBER) {
        uint64_t v;
        if (json_num_to_u64(seed->u.num, &v)) { out->seed = v; out->seed_set = true; }  // #4(d)
    }
    // Penalties. repetition_penalty (HF/vLLM) with rep_pen alias; OpenAI additive
    // frequency_penalty / presence_penalty.
    const ember_json *rp = ember_json_get(root, "repetition_penalty");
    if (!rp) rp = ember_json_get(root, "rep_pen");
    if (rp && rp->type == EMBER_JSON_NUMBER) {
        if (!valid_sampler_float(rp->u.num) || rp->u.num <= 0.0) goto invalid;
        out->rep_pen = rp->u.num;
        out->rep_pen_set = true;
    }
    const ember_json *rw = ember_json_get(root, "rep_window");
    if (rw && rw->type == EMBER_JSON_NUMBER) {
        int v;
        if (json_num_to_int(rw->u.num, &v) && v >= 0) {
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
        if (dal->type != EMBER_JSON_NUMBER || !valid_sampler_float(dal->u.num) ||
            dal->u.num < 0.0 || dal->u.num > 4096.0)
            goto invalid;
        out->dry_allowed_length = (int)dal->u.num;
        out->dry_allowed_length_set = true;
    }
    const ember_json *dw = ember_json_get(root, "dry_penalty_last_n");
    if (!dw) dw = ember_json_get(root, "dry_window");
    if (dw) {
        // llama.cpp uses -1 for "whole context"; the sampler reads <=0 the same
        // way, so both spellings land on the same behaviour.
        if (dw->type != EMBER_JSON_NUMBER || !valid_sampler_float(dw->u.num) ||
            dw->u.num < -1.0 || dw->u.num > 1048576.0)
            goto invalid;
        out->dry_window = (int)dw->u.num;
        out->dry_window_set = true;
    }

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
    // Explicit boolean overrides (enable_thinking / thinking).
    const ember_json *et = ember_json_get(root, "enable_thinking");
    if (!et) et = ember_json_get(root, "thinking");
    if (et && et->type == EMBER_JSON_BOOL) {
        out->thinking_enabled = et->u.b;
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
