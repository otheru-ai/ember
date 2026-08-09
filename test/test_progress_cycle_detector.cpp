#include "progress_cycle_detector.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using dflash::common::ProgressCycleDetector;

static int passed = 0;
static int failed = 0;

#define CHECK(expr) do { \
    if (expr) { ++passed; } \
    else { ++failed; std::fprintf(stderr, "FAIL %s:%d: %s\n", \
                                  __FILE__, __LINE__, #expr); } \
} while (0)

static bool feed(ProgressCycleDetector &detector,
                 const std::vector<int32_t> &tokens) {
    bool detected = false;
    for (int32_t token : tokens) detected = detector.observe(token) || detected;
    return detected;
}

int main() {
    {
        // Ordinary iterative repetition in hidden reasoning stays below the
        // deliberately higher reasoning threshold.
        ProgressCycleDetector detector({17, 18, 19});
        std::vector<int32_t> repeated(
            ProgressCycleDetector::kReasoningMinimumRepeatedTokens - 1, 42);
        CHECK(!feed(detector, repeated));
        CHECK(!detector.visible());
        CHECK(!detector.detected());
    }

    {
        // A genuinely sustained hidden-reasoning cycle is now actionable. The
        // old detector ignored it forever, allowing the model to consume the
        // entire remaining context before the harness learned anything useful.
        ProgressCycleDetector detector({17, 18, 19});
        std::vector<int32_t> repeated(
            ProgressCycleDetector::kReasoningMinimumRepeatedTokens, 42);
        CHECK(feed(detector, repeated));
        CHECK(!detector.visible());
        CHECK(detector.reason() ==
              dflash::common::ProgressStopReason::ReasoningCycle);
        CHECK(std::string(detector.reason_name()) ==
              "reasoning_cycle_detected");
    }

    {
        // A natural close activates the detector, without counting the close
        // sequence itself as visible output.
        ProgressCycleDetector detector({17, 18, 19});
        CHECK(!feed(detector, {1, 2, 17, 18}));
        CHECK(!detector.visible());
        CHECK(!detector.observe(19));
        CHECK(detector.visible());
        std::vector<int32_t> repeated(256, 7);
        CHECK(feed(detector, repeated));
        CHECK(detector.detected());
        CHECK(detector.reason() ==
              dflash::common::ProgressStopReason::VisibleCycle);
        CHECK(detector.cycle_period() == 1);
        CHECK(detector.repeated_span() == 256);
    }

    {
        // Three copies are deliberately insufficient; the fourth exact block
        // is the first point at which a long-period cycle may stop generation.
        ProgressCycleDetector detector;
        std::vector<int32_t> block;
        for (int32_t i = 0; i < 80; ++i) block.push_back(1000 + i);
        CHECK(!feed(detector, block));
        CHECK(!feed(detector, block));
        CHECK(!feed(detector, block));
        CHECK(feed(detector, block));
        CHECK(detector.cycle_period() == block.size());
        CHECK(detector.repeated_span() == block.size() * 4);
    }

    {
        // Similar-looking prose with forward progress must not trigger.
        ProgressCycleDetector detector;
        std::vector<int32_t> changing;
        for (int32_t line = 0; line < 1000; ++line) {
            changing.push_back(10);
            changing.push_back(20);
            changing.push_back(line);
            changing.push_back(30);
        }
        CHECK(!feed(detector, changing));
        CHECK(!detector.detected());
    }

    {
        // A server-authored force-close can arm visible monitoring even if a
        // tokenizer did not provide a natural-close sequence.
        ProgressCycleDetector detector({99});
        CHECK(!feed(detector, std::vector<int32_t>(500, 3)));
        detector.begin_visible();
        std::vector<int32_t> pair_cycle;
        for (int i = 0; i < 160; ++i) {
            pair_cycle.push_back(4);
            pair_cycle.push_back(5);
        }
        CHECK(feed(detector, pair_cycle));
        CHECK(detector.cycle_period() <= 2);
    }

    {
        // Cycles wider than the bounded search window are ignored. This keeps
        // the per-token check predictable even for very long generations.
        ProgressCycleDetector detector;
        std::vector<int32_t> wide;
        for (std::size_t copy = 0; copy < 4; ++copy) {
            for (std::size_t i = 0;
                 i < ProgressCycleDetector::kMaximumPeriod + 1; ++i) {
                wide.push_back(static_cast<int32_t>(i + 1));
            }
        }
        CHECK(!feed(detector, wide));
    }

    {
        // Non-periodic prompt copying is not a cycle: it needs a reference
        // check against the original prompt. The rule is VISIBLE-ONLY, because
        // DeepSeek-V4's template replays the model's own <think> content back
        // into the prompt inside a tool loop (chat_template.c:283-290,
        // ds4_server.c:2481-2488, and DeepSeek's API requires it once a tool
        // call is involved). Re-deriving that from the same context is the
        // expected continuation, but it matches the prompt exactly.
        std::vector<int32_t> prompt;
        for (int32_t i = 0; i < 900; ++i) prompt.push_back(10000 + i);
        const std::vector<int32_t> echoed(
            prompt.begin() + 123,
            prompt.begin() + 123 +
                static_cast<std::ptrdiff_t>(
                    ProgressCycleDetector::kPromptEchoTokens));

        // Reasoning phase: an exact copy must NOT stop the turn. Production
        // killed two agent turns this way on 2026-08-07; the 19:30 turn tripped
        // on all three gateway attempts under three different seeds and the
        // conversation died.
        {
            ProgressCycleDetector detector({17, 18, 19}, prompt);
            CHECK(!detector.visible());
            CHECK(!feed(detector, echoed));
            CHECK(!detector.detected());
        }

        // The same bytes after thinking closes are the ANSWER copying the
        // prompt, which is the failure this rule exists for.
        {
            ProgressCycleDetector detector({17, 18, 19}, prompt);
            CHECK(!feed(detector, {17, 18, 19}));
            CHECK(detector.visible());
            CHECK(feed(detector, echoed));
            CHECK(detector.reason() ==
                  dflash::common::ProgressStopReason::PromptEcho);
            CHECK(std::string(detector.reason_name()) == "prompt_echo_detected");
            CHECK(detector.prompt_echo_offset() == 123);
        }

        // No close sequence configured -> visible from the first token, so a
        // non-thinking generation keeps full echo coverage.
        {
            ProgressCycleDetector detector({}, prompt);
            CHECK(detector.visible());
            CHECK(feed(detector, echoed));
            CHECK(detector.reason() ==
                  dflash::common::ProgressStopReason::PromptEcho);
        }

        // Reasoning is not left unguarded: a genuine loop inside <think> still
        // trips the cycle rule at the reasoning thresholds, so dropping the
        // echo rule there does not make a runaway decode unstoppable.
        {
            ProgressCycleDetector detector({17, 18, 19}, prompt);
            CHECK(!detector.visible());
            std::vector<int32_t> loop;
            const std::size_t period = 8;
            while (loop.size() <
                   ProgressCycleDetector::kReasoningMinimumRepeatedTokens * 2)
                for (std::size_t i = 0; i < period; ++i)
                    loop.push_back(static_cast<int32_t>(70000 + i));
            CHECK(feed(detector, loop));
            CHECK(detector.reason() ==
                  dflash::common::ProgressStopReason::ReasoningCycle);
        }
    }

    {
        // A near-copy containing one changed token must not trip the exact
        // prompt-echo guard.
        std::vector<int32_t> prompt;
        for (int32_t i = 0; i < 700; ++i) prompt.push_back(20000 + i);
        std::vector<int32_t> changed(
            prompt.begin(),
            prompt.begin() + static_cast<std::ptrdiff_t>(
                ProgressCycleDetector::kPromptEchoTokens));
        changed[changed.size() / 2] = -7;
        ProgressCycleDetector detector({}, prompt);
        CHECK(!feed(detector, changed));
        CHECK(!detector.detected());
    }

    {
        // An agent re-sending a large tool argument copies the prompt verbatim
        // by design: writing back a file it just read, or retrying a rejected
        // call after a validation error. Production lost whole turns to this
        // (a skill_manage retry re-sending an unchanged skill body), so the
        // prompt-echo rule is disarmed inside a tool-call region.
        const std::vector<int32_t> open{900, 901};
        const std::vector<int32_t> close{902, 903};
        std::vector<int32_t> prompt;
        for (int32_t i = 0; i < 700; ++i) prompt.push_back(20000 + i);
        const std::vector<int32_t> echo(
            prompt.begin(),
            prompt.begin() + static_cast<std::ptrdiff_t>(
                ProgressCycleDetector::kPromptEchoTokens));

        // Same copied span, no tool markers configured: still a stuck decode.
        {
            ProgressCycleDetector detector({}, prompt);
            CHECK(feed(detector, echo));
            CHECK(detector.reason() ==
                  dflash::common::ProgressStopReason::PromptEcho);
        }

        // Inside a tool call the identical span is legitimate.
        {
            ProgressCycleDetector detector({}, prompt, open, close);
            CHECK(!feed(detector, open));
            CHECK(detector.in_tool_region());
            CHECK(!feed(detector, echo));
            CHECK(!detector.detected());

            // Once the call closes the protection is armed again.
            CHECK(!feed(detector, close));
            CHECK(!detector.in_tool_region());
            CHECK(feed(detector, echo));
            CHECK(detector.reason() ==
                  dflash::common::ProgressStopReason::PromptEcho);
        }

        // Closing the block must not immediately re-arm the rule against a
        // window still holding the copied payload. Measured in production as
        // entries=1, since_exit=0, with the close marker as the last tokens.
        {
            ProgressCycleDetector detector({}, prompt, open, close);
            CHECK(!feed(detector, open));
            CHECK(!feed(detector, echo));      // legitimate copy inside a call
            CHECK(!feed(detector, close));     // <-- used to fire right here
            CHECK(!detector.detected());
            CHECK(!detector.in_tool_region());

            // Still held off while the window drains. Filler must be distinct
            // tokens: a run of one repeated id is a period-1 cycle and would
            // trip the cycle detector instead, proving nothing about echo.
            std::vector<int32_t> filler;
            for (int32_t i = 0;
                 i < (int32_t)ProgressCycleDetector::kPromptEchoTokens - 1; ++i)
                filler.push_back(40000 + i);
            CHECK(!feed(detector, filler));
            CHECK(!detector.detected());

            // ...and armed again once it is entirely outside the region.
            CHECK(feed(detector, echo));
            CHECK(detector.reason() ==
                  dflash::common::ProgressStopReason::PromptEcho);
        }

        // Disarming echo must not disarm cycle detection: a genuine runaway
        // inside a tool call is still caught, since it does not depend on
        // matching the prompt.
        {
            ProgressCycleDetector detector({}, prompt, open, close);
            CHECK(!feed(detector, open));
            detector.begin_visible();
            std::vector<int32_t> loop(
                ProgressCycleDetector::kMinimumRepeatedTokens * 4, 4242);
            CHECK(feed(detector, loop));
            CHECK(detector.reason() ==
                  dflash::common::ProgressStopReason::VisibleCycle);
        }
    }

    std::printf("progress cycle detector: %d passed, %d failed\n",
                passed, failed);
    return failed == 0 ? 0 : 1;
}
