// Model card sidecar: per-model token-budget + sampling defaults, and the
// thinking-budget math that guarantees a visible answer after force-close.
//
// This is the logic whose ABSENCE made lucebox leak raw reasoning to the user:
// with no terminator hint and a mis-sized reply reserve, the model would exhaust
// its budget thinking and emit empty content. The card sizes the think budget
// as (effective_max_tokens - hard_limit_reply_budget), clamped by the requested
// effort tier, and supplies the terminator hint the backend injects at the
// budget edge.
#ifndef EMBER_MODEL_CARD_H
#define EMBER_MODEL_CARD_H

#include <stdbool.h>

typedef struct {
    int    max_tokens;                 // standard combined cap
    int    complex_problem_max_tokens; // drives x-high/max tiers
    int    hard_limit_reply_budget;    // tokens reserved post-</think>
    char  *thinking_terminator_hint;   // injected at the budget edge (owned)
    struct { int low, medium, high, xhigh, max; } tiers;
    double temperature, top_p;
    bool   loaded;
} ember_model_card;

// Load from a JSON sidecar. On missing file/parse error, fills conservative
// DeepSeek-V4-Flash defaults and returns false (still usable).
bool ember_model_card_load(ember_model_card *card, const char *path);
void ember_model_card_free(ember_model_card *card);

// Think-token budget for a request. `effort` is "low".."max" (NULL → "high").
// `client_max_tokens` is the request's max_tokens (0 → use the card's).
// Returns >= 0; 0 disables thinking (budget too small to leave a reply).
int ember_model_card_think_budget(const ember_model_card *card,
                                  const char *effort, int client_max_tokens);

#endif  // EMBER_MODEL_CARD_H
