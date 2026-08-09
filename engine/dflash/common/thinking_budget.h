#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dflash::common {

// Tracks whether a generation that started inside <think> has naturally
// emitted its closing token sequence.  A thinking-budget controller must never
// inject another close after this state flips: doing so would place the
// server-authored directive in visible content and can make the model restart
// an answer it has already begun.
class ThinkingBudgetState {
public:
    explicit ThinkingBudgetState(bool starts_in_thinking = true)
        : in_thinking_(starts_in_thinking) {}

    void observe_existing(
            const std::vector<int32_t> &generated,
            const std::vector<int32_t> &natural_close_ids) {
        if (!in_thinking_ || natural_close_ids.empty() ||
            generated.size() < natural_close_ids.size())
            return;
        if (std::search(generated.begin(), generated.end(),
                        natural_close_ids.begin(), natural_close_ids.end()) !=
            generated.end())
            in_thinking_ = false;
    }

    void observe_latest(
            const std::vector<int32_t> &generated,
            const std::vector<int32_t> &natural_close_ids) {
        if (!in_thinking_ || natural_close_ids.empty() ||
            generated.size() < natural_close_ids.size())
            return;
        if (std::equal(natural_close_ids.rbegin(), natural_close_ids.rend(),
                       generated.rbegin()))
            in_thinking_ = false;
    }

    bool should_force_close(int generation_limit,
                            std::size_t generated_tokens,
                            int hard_limit_remaining,
                            bool has_forced_close_sequence) const {
        if (!in_thinking_ || !has_forced_close_sequence ||
            generation_limit < 0 || hard_limit_remaining <= 0 ||
            generated_tokens > static_cast<std::size_t>(generation_limit))
            return false;
        const std::size_t remaining =
            static_cast<std::size_t>(generation_limit) - generated_tokens;
        return remaining <= static_cast<std::size_t>(hard_limit_remaining);
    }

    void mark_closed() { in_thinking_ = false; }
    bool in_thinking() const { return in_thinking_; }

private:
    bool in_thinking_;
};

}  // namespace dflash::common
