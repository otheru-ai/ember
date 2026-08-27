#include "thinking_budget.h"

#include <stdint.h>
#include <stdio.h>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(expr) do {                                                   \
    if (expr) { ++g_pass; }                                                \
    else { ++g_fail; fprintf(stderr, "FAIL %s:%d: %s\n",                 \
                             __FILE__, __LINE__, #expr); }                 \
} while (0)

int main(void) {
    const int32_t natural_close[] = {17, 18, 19};
    const size_t n_close = sizeof(natural_close) / sizeof(natural_close[0]);

    {
        dflash_thinking_budget_state state =
            DFLASH_THINKING_BUDGET_INITIALIZER;
        CHECK(dflash_thinking_budget_in_thinking(&state));
        CHECK(!dflash_thinking_budget_should_force_close(
            &state, 16384, 767, 15616, true));
        CHECK(dflash_thinking_budget_should_force_close(
            &state, 16384, 768, 15616, true));
        CHECK(!dflash_thinking_budget_should_force_close(
            &state, 16384, 768, 15616, false));
    }

    {
        // A natural </think> before the boundary permanently disarms forced
        // close while the visible answer grows past that boundary.
        dflash_thinking_budget_state state =
            DFLASH_THINKING_BUDGET_INITIALIZER;
        const int32_t generated[] = {1, 2, 17, 18, 19, 20, 21};
        dflash_thinking_budget_observe_existing(
            &state, generated, sizeof(generated) / sizeof(generated[0]),
            natural_close, n_close);
        CHECK(!dflash_thinking_budget_in_thinking(&state));
        CHECK(!dflash_thinking_budget_should_force_close(
            &state, 16384, 768, 15616, true));
        CHECK(!dflash_thinking_budget_should_force_close(
            &state, 16384, 16000, 15616, true));
    }

    {
        // Resident batching observes one committed token at a time.
        dflash_thinking_budget_state state =
            DFLASH_THINKING_BUDGET_INITIALIZER;
        int32_t generated[] = {1, 17, 18, 19};
        dflash_thinking_budget_observe_latest(
            &state, generated, 2, natural_close, n_close);
        CHECK(dflash_thinking_budget_in_thinking(&state));
        dflash_thinking_budget_observe_latest(
            &state, generated, 3, natural_close, n_close);
        CHECK(dflash_thinking_budget_in_thinking(&state));
        dflash_thinking_budget_observe_latest(
            &state, generated, 4, natural_close, n_close);
        CHECK(!dflash_thinking_budget_in_thinking(&state));
        CHECK(!dflash_thinking_budget_should_force_close(
            &state, 100, 90, 10, true));
    }

    {
        // An actual runaway-thinking generation still closes at its boundary.
        dflash_thinking_budget_state state =
            DFLASH_THINKING_BUDGET_INITIALIZER;
        int32_t generated[768];
        for (size_t i = 0; i < sizeof(generated) / sizeof(generated[0]); ++i)
            generated[i] = 42;
        dflash_thinking_budget_observe_existing(
            &state, generated, sizeof(generated) / sizeof(generated[0]),
            natural_close, n_close);
        CHECK(dflash_thinking_budget_in_thinking(&state));
        CHECK(dflash_thinking_budget_should_force_close(
            &state, 16384, 768, 15616, true));
        dflash_thinking_budget_mark_closed(&state);
        CHECK(!dflash_thinking_budget_in_thinking(&state));
        CHECK(!dflash_thinking_budget_should_force_close(
            &state, 16384, 768, 15616, true));
    }

    {
        // Without a known natural-close tokenization, observation is a no-op;
        // callers must not arm the hook in this configuration.
        dflash_thinking_budget_state state;
        const int32_t generated[] = {1, 2, 3};
        dflash_thinking_budget_init(&state, true);
        dflash_thinking_budget_observe_existing(
            &state, generated, sizeof(generated) / sizeof(generated[0]),
            NULL, 0);
        CHECK(dflash_thinking_budget_in_thinking(&state));
        CHECK(!dflash_thinking_budget_should_force_close(
            &state, 10, 9, 1, false));
        CHECK(!dflash_thinking_budget_should_force_close(
            &state, 10, 11, 1, true));
    }

    printf("thinking budget: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
