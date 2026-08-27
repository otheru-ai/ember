// C-compatible thinking-budget state used by the C++ engine bridge today and
// by the future C orchestration layer. Keep this header allocation-free: the
// caller owns token storage and passes spans explicitly.
#ifndef DFLASH_COMMON_THINKING_BUDGET_H
#define DFLASH_COMMON_THINKING_BUDGET_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool in_thinking;
} dflash_thinking_budget_state;

#define DFLASH_THINKING_BUDGET_INITIALIZER { true }

static inline void dflash_thinking_budget_init(
        dflash_thinking_budget_state *state, bool starts_in_thinking) {
    state->in_thinking = starts_in_thinking;
}

static inline bool dflash_token_span_equal(
        const int32_t *left, const int32_t *right, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (left[i] != right[i]) return false;
    }
    return true;
}

// Observe a complete already-generated prefix. A natural </think> anywhere in
// it permanently disarms forced close, preventing a second server-authored
// close from being injected into visible answer text.
static inline void dflash_thinking_budget_observe_existing(
        dflash_thinking_budget_state *state,
        const int32_t *generated, size_t generated_count,
        const int32_t *natural_close_ids, size_t natural_close_count) {
    if (!state->in_thinking || !generated || !natural_close_ids ||
        natural_close_count == 0 || generated_count < natural_close_count) {
        return;
    }
    const size_t last = generated_count - natural_close_count;
    for (size_t i = 0; i <= last; ++i) {
        if (dflash_token_span_equal(generated + i, natural_close_ids,
                                   natural_close_count)) {
            state->in_thinking = false;
            return;
        }
    }
}

// Fast path for resident decoding: only the newest suffix can introduce the
// first natural close, so compare the tail rather than rescanning the history.
static inline void dflash_thinking_budget_observe_latest(
        dflash_thinking_budget_state *state,
        const int32_t *generated, size_t generated_count,
        const int32_t *natural_close_ids, size_t natural_close_count) {
    if (!state->in_thinking || !generated || !natural_close_ids ||
        natural_close_count == 0 || generated_count < natural_close_count) {
        return;
    }
    if (dflash_token_span_equal(
            generated + generated_count - natural_close_count,
            natural_close_ids, natural_close_count)) {
        state->in_thinking = false;
    }
}

static inline bool dflash_thinking_budget_should_force_close(
        const dflash_thinking_budget_state *state,
        int generation_limit, size_t generated_tokens,
        int hard_limit_remaining, bool has_forced_close_sequence) {
    if (!state->in_thinking || !has_forced_close_sequence ||
        generation_limit < 0 || hard_limit_remaining <= 0 ||
        generated_tokens > (size_t)generation_limit) {
        return false;
    }
    return (size_t)generation_limit - generated_tokens <=
           (size_t)hard_limit_remaining;
}

static inline void dflash_thinking_budget_mark_closed(
        dflash_thinking_budget_state *state) {
    state->in_thinking = false;
}

static inline bool dflash_thinking_budget_in_thinking(
        const dflash_thinking_budget_state *state) {
    return state->in_thinking;
}

#ifdef __cplusplus
}
#endif

#endif
