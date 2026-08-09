#include "thinking_budget.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using dflash::common::ThinkingBudgetState;

static int passed = 0;
static int failed = 0;

#define CHECK(expr) do { \
    if (expr) { ++passed; } \
    else { ++failed; std::fprintf(stderr, "FAIL %s:%d: %s\n", \
                                  __FILE__, __LINE__, #expr); } \
} while (0)

int main() {
    const std::vector<int32_t> natural_close = {17, 18, 19};

    {
        ThinkingBudgetState state;
        CHECK(state.in_thinking());
        CHECK(!state.should_force_close(16384, 767, 15616, true));
        CHECK(state.should_force_close(16384, 768, 15616, true));
        CHECK(!state.should_force_close(16384, 768, 15616, false));
    }

    {
        // Regression: a natural </think> before the 768-token boundary must
        // permanently disarm the forced close while the visible answer grows
        // past that boundary.
        ThinkingBudgetState state;
        std::vector<int32_t> generated = {1, 2, 17, 18, 19, 20, 21};
        state.observe_existing(generated, natural_close);
        CHECK(!state.in_thinking());
        CHECK(!state.should_force_close(16384, 768, 15616, true));
        CHECK(!state.should_force_close(16384, 16000, 15616, true));
    }

    {
        // Resident batching observes one committed token at a time.
        ThinkingBudgetState state;
        std::vector<int32_t> generated = {1, 17};
        state.observe_latest(generated, natural_close);
        CHECK(state.in_thinking());
        generated.push_back(18);
        state.observe_latest(generated, natural_close);
        CHECK(state.in_thinking());
        generated.push_back(19);
        state.observe_latest(generated, natural_close);
        CHECK(!state.in_thinking());
        CHECK(!state.should_force_close(100, 90, 10, true));
    }

    {
        // An actual runaway-thinking generation still closes at its boundary.
        ThinkingBudgetState state;
        std::vector<int32_t> generated(768, 42);
        state.observe_existing(generated, natural_close);
        CHECK(state.in_thinking());
        CHECK(state.should_force_close(16384, generated.size(), 15616, true));
        state.mark_closed();
        CHECK(!state.in_thinking());
        CHECK(!state.should_force_close(16384, generated.size(), 15616, true));
    }

    {
        // Without a known natural-close tokenization, observation is a no-op;
        // callers must not arm the hook in this configuration.
        ThinkingBudgetState state;
        state.observe_existing({1, 2, 3}, {});
        CHECK(state.in_thinking());
        CHECK(!state.should_force_close(10, 9, 1, false));
        CHECK(!state.should_force_close(10, 11, 1, true));
    }

    std::printf("thinking budget: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
