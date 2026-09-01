#include "model_card.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/json.h"

static char *dupz(const char *s) { return s ? strdup(s) : NULL; }

static void defaults(ember_model_card *c) {
    c->max_tokens = 16384;
    c->complex_problem_max_tokens = 32768;
    c->hard_limit_reply_budget = 1024;  // DeepSeek-V4-Flash is terse
    c->thinking_terminator_hint = dupz(
        "Considering the limited time by the user, I have to give the solution "
        "based on the thinking directly now.\n</think>\n\n");
    c->tiers.low = 4096;   c->tiers.medium = 8192;  c->tiers.high = 16384;
    c->tiers.xhigh = 24576; c->tiers.max = 32768;
    c->temperature = 0.6; c->top_p = 0.95;
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
    size_t rd = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[rd] = '\0';

    ember_json *v = ember_json_parse(buf);
    free(buf);
    if (!v) return false;

    card->max_tokens = (int)ember_json_num(ember_json_get(v, "max_tokens"), card->max_tokens);
    card->complex_problem_max_tokens = (int)ember_json_num(
        ember_json_get(v, "complex_problem_max_tokens"), card->complex_problem_max_tokens);
    card->hard_limit_reply_budget = (int)ember_json_num(
        ember_json_get(v, "hard_limit_reply_budget"), card->hard_limit_reply_budget);

    const ember_json *hint = ember_json_get(v, "thinking_terminator_hint");
    if (hint && hint->type == EMBER_JSON_STRING) {
        free(card->thinking_terminator_hint);
        card->thinking_terminator_hint = dupz(ember_json_str(hint, ""));
    }

    const ember_json *t = ember_json_get(v, "reasoning_effort_tiers");
    if (t && t->type == EMBER_JSON_OBJECT) {
        card->tiers.low   = (int)ember_json_num(ember_json_get(t, "low"),    card->tiers.low);
        card->tiers.medium= (int)ember_json_num(ember_json_get(t, "medium"), card->tiers.medium);
        card->tiers.high  = (int)ember_json_num(ember_json_get(t, "high"),   card->tiers.high);
        card->tiers.xhigh = (int)ember_json_num(ember_json_get(t, "x-high"), card->tiers.xhigh);
        card->tiers.max   = (int)ember_json_num(ember_json_get(t, "max"),    card->tiers.max);
    }
    const ember_json *s = ember_json_get(v, "sampling");
    if (s && s->type == EMBER_JSON_OBJECT) {
        card->temperature = ember_json_num(ember_json_get(s, "temperature"), card->temperature);
        card->top_p = ember_json_num(ember_json_get(s, "top_p"), card->top_p);
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
    if (strcmp(effort, "x-high") == 0) return c->tiers.xhigh;
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
