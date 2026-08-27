#include "model_card.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/json.h"

static char *dupz(const char *s) { return s ? strdup(s) : NULL; }

static int card_int(const ember_json *v, int dflt, int min_value) {
    if (!v || v->type != EMBER_JSON_NUMBER || !isfinite(v->u.num) ||
        trunc(v->u.num) != v->u.num || v->u.num < (double)min_value ||
        v->u.num > (double)INT_MAX)
        return dflt;
    return (int)v->u.num;
}

static double card_float(const ember_json *v, double dflt,
                         double min_value, double max_value) {
    if (!v || v->type != EMBER_JSON_NUMBER || !isfinite(v->u.num) ||
        v->u.num < min_value || v->u.num > max_value)
        return dflt;
    return v->u.num;
}

static void defaults(ember_model_card *c) {
    c->max_tokens = 16384;
    c->hard_limit_reply_budget = 1024;  // DeepSeek-V4-Flash is terse
    c->thinking_terminator_hint = dupz(
        "Considering the limited time by the user, I have to give the solution "
        "based on the thinking directly now.\n</think>\n\n");
    c->tiers.low = 4096;   c->tiers.medium = 8192;  c->tiers.high = 16384;
    c->tiers.xhigh = 24576; c->tiers.max = 32768;
    // DeepSeek-V4-Flash source-card defaults. These are also the safe fallback
    // when the sidecar is absent or malformed, matching this server's single-
    // architecture scope.
    c->temperature = 0.6;
    c->top_p = 0.95;
    c->top_k = 40;
    c->min_p = 0.0;
    c->presence_penalty = 0.0;
    c->repetition_penalty = 1.0;
    c->context_extension.available = false;
}

bool ember_model_card_load(ember_model_card *card, const char *path) {
    memset(card, 0, sizeof(*card));
    defaults(card);
    if (!path) return false;

    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0 || n > (1 << 20)) { fclose(f); return false; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return false; }
    size_t rd = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (rd != (size_t)n) { free(buf); return false; }
    buf[rd] = '\0';

    ember_json *v = ember_json_parse(buf);
    free(buf);
    if (!v) return false;

    card->max_tokens = card_int(
        ember_json_get(v, "max_tokens"), card->max_tokens, 1);
    card->hard_limit_reply_budget = card_int(
        ember_json_get(v, "hard_limit_reply_budget"),
        card->hard_limit_reply_budget, 0);

    const ember_json *hint = ember_json_get(v, "thinking_terminator_hint");
    if (hint && hint->type == EMBER_JSON_STRING) {
        free(card->thinking_terminator_hint);
        card->thinking_terminator_hint = dupz(ember_json_str(hint, ""));
    }

    const ember_json *t = ember_json_get(v, "reasoning_effort_tiers");
    if (t && t->type == EMBER_JSON_OBJECT) {
        card->tiers.low = card_int(
            ember_json_get(t, "low"), card->tiers.low, 0);
        card->tiers.medium = card_int(
            ember_json_get(t, "medium"), card->tiers.medium, 0);
        card->tiers.high = card_int(
            ember_json_get(t, "high"), card->tiers.high, 0);
        card->tiers.xhigh = card_int(
            ember_json_get(t, "x-high"), card->tiers.xhigh, 0);
        card->tiers.max = card_int(
            ember_json_get(t, "max"), card->tiers.max, 0);
    }
    const ember_json *s = ember_json_get(v, "sampling");
    if (s && s->type == EMBER_JSON_OBJECT) {
        // Ported from lucebox server/src/server/model_card.cpp's complete
        // SamplingDefaults surface. Every parsed field is consumed by
        // run_chat; do not add card-only/dead sampling fields here.
        card->temperature = card_float(
            ember_json_get(s, "temperature"), card->temperature, 0.0, 2.0);
        card->top_p = card_float(
            ember_json_get(s, "top_p"), card->top_p, 0.0, 1.0);
        card->top_k = card_int(
            ember_json_get(s, "top_k"), card->top_k, 0);
        card->min_p = card_float(
            ember_json_get(s, "min_p"), card->min_p, 0.0, 1.0);
        card->presence_penalty = card_float(
            ember_json_get(s, "presence_penalty"),
            card->presence_penalty, -2.0, 2.0);
        double repetition_penalty = card_float(
            ember_json_get(s, "repetition_penalty"),
            card->repetition_penalty, 0.0, (double)FLT_MAX);
        if (repetition_penalty > 0.0)
            card->repetition_penalty = repetition_penalty;
    }
    const ember_json *extension = ember_json_get(v, "context_extension");
    if (extension && extension->type == EMBER_JSON_OBJECT) {
        const ember_json *type = ember_json_get(extension, "type");
        const int native_context = card_int(
            ember_json_get(extension, "native_context"), 0, 1);
        const int max_context = card_int(
            ember_json_get(extension, "max_context"), 0, 1);
        const double factor = card_float(
            ember_json_get(extension, "factor"), 0.0, 1.0, (double)FLT_MAX);
        // Cards are advisory rather than authority. Accept only a complete,
        // internally consistent static-YaRN tuple; the model-specific backend
        // performs the stricter architecture/provenance check at load time.
        if (type && type->type == EMBER_JSON_STRING &&
            strcmp(ember_json_str(type, ""), "static_yarn") == 0 &&
            native_context > 0 && max_context > native_context &&
            factor > 1.0) {
            card->context_extension.available = true;
            card->context_extension.native_context = native_context;
            card->context_extension.max_context = max_context;
            card->context_extension.factor = factor;
        }
    }
    ember_json_free(v);
    card->loaded = true;
    return true;
}

void ember_model_card_free(ember_model_card *card) {
    free(card->thinking_terminator_hint);
    card->thinking_terminator_hint = NULL;
}

static int tier_for(const ember_model_card *c, const char *effort) {
    if (!effort) return c->tiers.high;
    if (strcmp(effort, "low") == 0)    return c->tiers.low;
    if (strcmp(effort, "medium") == 0) return c->tiers.medium;
    if (strcmp(effort, "high") == 0)   return c->tiers.high;
    // Both spellings: the card JSON and OpenAI use "x-high", chat_api.c's
    // effort_to_mode historically matched "xhigh". Accept either here so the
    // tier and the think mode can never disagree about the same request.
    if (strcmp(effort, "x-high") == 0 || strcmp(effort, "xhigh") == 0)
        return c->tiers.xhigh;
    if (strcmp(effort, "max") == 0)    return c->tiers.max;
    return c->tiers.high;
}

int ember_model_card_think_budget(const ember_model_card *card,
                                  const char *effort, int client_max_tokens) {
    int effective_max = client_max_tokens > 0 ? client_max_tokens : card->max_tokens;
    int room = effective_max - card->hard_limit_reply_budget;  // reserve for reply
    if (room <= 0) return 0;                                    // no room to think
    int tier = tier_for(card, effort);
    return tier < room ? tier : room;
}
