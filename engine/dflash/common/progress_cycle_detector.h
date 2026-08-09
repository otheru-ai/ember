#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace dflash::common {

// Why generation stopped making progress. These names are carried through the
// backend ABI so an agent harness can distinguish a model loop from an ordinary
// max-token truncation and choose an informed retry policy.
enum class ProgressStopReason {
    None,
    VisibleCycle,
    ReasoningCycle,
    PromptEcho,
};

constexpr const char *progress_stop_reason_name(ProgressStopReason reason) {
    switch (reason) {
    case ProgressStopReason::VisibleCycle:   return "repetition_detected";
    case ProgressStopReason::ReasoningCycle: return "reasoning_cycle_detected";
    case ProgressStopReason::PromptEcho:     return "prompt_echo_detected";
    case ProgressStopReason::None:           return "none";
    }
    return "none";
}

// Progress-aware decode watchdog. Visible output retains the original
// conservative exact-cycle rule (four copies / 256 repeated tokens). Hidden
// reasoning is now monitored too, but only after eight copies / 1024 repeated
// tokens so ordinary iterative reasoning is not mistaken for a stuck decode.
//
// Independently, a generated suffix that copies 512 consecutive prompt tokens
// is stopped -- but only once generation is past the reasoning phase. Inside
// <think> the template legitimately replays the model's own prior reasoning, so
// an exact prompt match there is expected rather than degenerate; see observe().
// This catches the non-periodic failure where a model starts reproducing a
// summarization prompt and its placeholders: a pure cycle detector cannot see
// that because the copied text continues to change.
// Hashes are only an index; every match is verified token-for-token before a
// generation is terminated, so collisions cannot cause false positives.
class ProgressCycleDetector {
public:
    static constexpr std::size_t kMinimumCopies = 4;
    static constexpr std::size_t kMinimumRepeatedTokens = 256;
    static constexpr std::size_t kMaximumPeriod = 512;
    static constexpr std::size_t kReasoningMinimumCopies = 8;
    static constexpr std::size_t kReasoningMinimumRepeatedTokens = 1024;
    static constexpr std::size_t kPromptEchoTokens = 512;

    explicit ProgressCycleDetector(
            const std::vector<int32_t> &natural_close_ids = {},
            const std::vector<int32_t> &prompt_tokens = {},
            const std::vector<int32_t> &tool_open_ids = {},
            const std::vector<int32_t> &tool_close_ids = {})
        : natural_close_ids_(natural_close_ids),
          prompt_tokens_(prompt_tokens),
          tool_open_ids_(tool_open_ids),
          tool_close_ids_(tool_close_ids),
          visible_(natural_close_ids.empty()) {
        reset_phase_hash();
        all_prefix_hash_.assign(1, 0);
        all_hash_power_.assign(1, 1);
        build_prompt_index();
    }

    bool observe(int32_t token) {
        if (detected()) return true;

        append_hashed(token, all_tokens_, all_prefix_hash_, all_hash_power_);
        append_hashed(token, phase_tokens_, phase_prefix_hash_, phase_hash_power_);

        track_tool_region(token);

        // The echo rule needs kPromptEchoTokens tokens that are ALL outside a
        // tool region. Suppressing only while inside is not enough: on the
        // token that closes the block the trailing window still holds the
        // legitimately copied payload, so the check fires the instant the
        // region ends. Measured exactly that way -- entries=1, since_exit=0,
        // with the close marker as the final tokens.
        //
        // VISIBLE-ONLY. The echo rule asks "is the model reproducing its input
        // instead of answering?", and that question is only meaningful about an
        // ANSWER. Inside reasoning, restating the input is what reasoning is
        // for -- and worse, DeepSeek-V4's template deliberately feeds the
        // model's own prior <think> content back:
        //
        //   {%- if ns.has_tools or loop.index0 > ns.last_user_idx -%}
        //     {{- thinking_start_token + (message['reasoning_content'] or '')
        //         + thinking_end_token -}}
        //
        // which ember mirrors at chat_template.c:283-290 and ds4 at
        // ds4_server.c:2481-2488. DeepSeek's API documents the same rule:
        // reasoning_content MUST be passed back once a tool call is involved
        // (and 400s when it is not). So in a tool loop the prompt legitimately
        // contains the model's earlier thinking, and re-deriving it is expected
        // -- while matching the prompt exactly, which is what this rule fires
        // on. Production killed two agent turns that way before this gate.
        //
        // Reasoning is NOT left unguarded: detect_cycle still runs every token
        // at the reasoning thresholds (see the class block above) and the
        // thinking budget force-closes a runaway <think>. Only the "matches the
        // prompt" rule is dropped, in the one phase where that is legitimate.
        //
        // Not gated on a think marker: `visible_` is already the phase bit the
        // detector maintains, and it starts TRUE when there is no close
        // sequence, so non-thinking generations keep full echo coverage.
        if (visible_ && !in_tool_region_ &&
            all_tokens_.size() >= echo_block_until_ &&
            detect_prompt_echo()) {
            reason_ = ProgressStopReason::PromptEcho;
            return true;
        }

        const bool cycle = visible_
            ? detect_cycle(kMinimumCopies, kMinimumRepeatedTokens)
            : detect_cycle(kReasoningMinimumCopies,
                           kReasoningMinimumRepeatedTokens);
        if (cycle) {
            reason_ = visible_ ? ProgressStopReason::VisibleCycle
                               : ProgressStopReason::ReasoningCycle;
            return true;
        }

        if (!visible_) {
            close_window_.push_back(token);
            if (close_window_.size() > natural_close_ids_.size()) {
                close_window_.erase(close_window_.begin());
            }
            if (close_window_.size() == natural_close_ids_.size() &&
                std::equal(close_window_.begin(), close_window_.end(),
                           natural_close_ids_.begin())) {
                begin_visible();
            }
        }
        return false;
    }

    // Forced-close sequences are server-authored and normally end in the same
    // natural close sequence. Callers invoke this explicitly as a fallback so
    // template/tokenizer changes cannot leave visible monitoring disarmed.
    void begin_visible() {
        visible_ = true;
        close_window_.clear();
        reset_phase_hash();
    }

    bool visible() const { return visible_; }
    bool detected() const { return reason_ != ProgressStopReason::None; }
    ProgressStopReason reason() const { return reason_; }
    const char *reason_name() const { return progress_stop_reason_name(reason_); }
    std::size_t cycle_period() const { return cycle_period_; }
    std::size_t repeated_span() const { return repeated_span_; }
    std::size_t prompt_echo_offset() const { return prompt_echo_offset_; }
    bool in_tool_region() const { return in_tool_region_; }
    // Diagnostics for tool-region scoping. region_entries()==0 at an echo means
    // the open marker was never matched -- the model's token split for the
    // marker differs from the server's encode(). A small
    // tokens_since_region_exit() instead means the region WAS entered and the
    // echo fired on the stale window just after it closed.
    std::size_t region_entries() const { return region_entries_; }
    std::size_t tokens_since_region_exit() const {
        return last_exit_index_ ? all_tokens_.size() - last_exit_index_ : 0;
    }
    const std::vector<int32_t> &all_tokens() const { return all_tokens_; }
    const std::vector<int32_t> &tool_open_ids() const { return tool_open_ids_; }

private:
    static constexpr uint64_t kHashBase = UINT64_C(1099511628211);

    static uint64_t token_value(int32_t token) {
        return static_cast<uint64_t>(static_cast<uint32_t>(token)) +
            UINT64_C(0x9e3779b97f4a7c15);
    }

    static void append_hashed(
            int32_t token,
            std::vector<int32_t> &tokens,
            std::vector<uint64_t> &prefix,
            std::vector<uint64_t> &powers) {
        tokens.push_back(token);
        prefix.push_back(prefix.back() * kHashBase + token_value(token));
        powers.push_back(powers.back() * kHashBase);
    }

    static uint64_t range_hash(
            const std::vector<uint64_t> &prefix,
            const std::vector<uint64_t> &powers,
            std::size_t begin, std::size_t end) {
        return prefix[end] - prefix[begin] * powers[end - begin];
    }

    void reset_phase_hash() {
        phase_tokens_.clear();
        phase_prefix_hash_.assign(1, 0);
        phase_hash_power_.assign(1, 1);
    }

    void build_prompt_index() {
        if (prompt_tokens_.size() < kPromptEchoTokens) return;
        std::vector<uint64_t> prefix(1, 0);
        std::vector<uint64_t> powers(1, 1);
        prefix.reserve(prompt_tokens_.size() + 1);
        powers.reserve(prompt_tokens_.size() + 1);
        for (int32_t token : prompt_tokens_) {
            prefix.push_back(prefix.back() * kHashBase + token_value(token));
            powers.push_back(powers.back() * kHashBase);
        }
        for (std::size_t i = 0;
             i + kPromptEchoTokens <= prompt_tokens_.size(); ++i) {
            prompt_windows_.emplace(
                range_hash(prefix, powers, i, i + kPromptEchoTokens), i);
        }
    }

    bool detect_prompt_echo() {
        if (all_tokens_.size() < kPromptEchoTokens || prompt_windows_.empty())
            return false;
        const std::size_t begin = all_tokens_.size() - kPromptEchoTokens;
        const uint64_t hash = range_hash(
            all_prefix_hash_, all_hash_power_, begin, all_tokens_.size());
        const auto matches = prompt_windows_.equal_range(hash);
        for (auto it = matches.first; it != matches.second; ++it) {
            const std::size_t prompt_begin = it->second;
            if (std::equal(
                    all_tokens_.begin() + static_cast<std::ptrdiff_t>(begin),
                    all_tokens_.end(),
                    prompt_tokens_.begin() +
                        static_cast<std::ptrdiff_t>(prompt_begin))) {
                prompt_echo_offset_ = prompt_begin;
                repeated_span_ = kPromptEchoTokens;
                return true;
            }
        }
        return false;
    }

    bool detect_cycle(std::size_t minimum_copies,
                      std::size_t minimum_repeated_tokens) {
        const std::size_t count = phase_tokens_.size();
        const std::size_t period_limit =
            std::min(kMaximumPeriod, count / minimum_copies);
        for (std::size_t period = 1; period <= period_limit; ++period) {
            const std::size_t copies = std::max(
                minimum_copies,
                (minimum_repeated_tokens + period - 1) / period);
            const std::size_t span = copies * period;
            if (count < span) continue;

            const std::size_t left_begin = count - span;
            const std::size_t left_end = count - period;
            const std::size_t right_begin = left_begin + period;
            if (range_hash(phase_prefix_hash_, phase_hash_power_,
                           left_begin, left_end) !=
                range_hash(phase_prefix_hash_, phase_hash_power_,
                           right_begin, count)) {
                continue;
            }
            if (!std::equal(
                    phase_tokens_.begin() +
                        static_cast<std::ptrdiff_t>(left_begin),
                    phase_tokens_.begin() +
                        static_cast<std::ptrdiff_t>(left_end),
                    phase_tokens_.begin() +
                        static_cast<std::ptrdiff_t>(right_begin))) {
                continue;
            }
            cycle_period_ = period;
            repeated_span_ = span;
            return true;
        }
        return false;
    }

    static bool ends_with(const std::vector<int32_t> &hay,
                          const std::vector<int32_t> &needle) {
        if (needle.empty() || hay.size() < needle.size()) return false;
        return std::equal(needle.begin(), needle.end(),
                          hay.end() -
                              static_cast<std::ptrdiff_t>(needle.size()));
    }

    // Reproducing prompt text inside a tool call is legitimate, not a stuck
    // decode: an agent writes back a file it just read, or retries a rejected
    // call with the same large payload after a validation error. Both copy far
    // more than kPromptEchoTokens verbatim. Production lost whole turns to
    // exactly that -- a skill_manage retry after "Description is 297 chars"
    // re-sent the unchanged skill body and tripped the watchdog. So the
    // prompt-echo rule is disarmed inside a tool-call region. Genuine runaway
    // repetition there is still caught by the cycle detector, which does not
    // depend on matching the prompt.
    void track_tool_region(int32_t token) {
        if (tool_open_ids_.empty() && tool_close_ids_.empty()) return;
        tool_window_.push_back(token);
        const std::size_t cap =
            std::max(tool_open_ids_.size(), tool_close_ids_.size());
        if (cap && tool_window_.size() > cap)
            tool_window_.erase(tool_window_.begin());
        if (!in_tool_region_) {
            if (ends_with(tool_window_, tool_open_ids_)) {
                in_tool_region_ = true;
                ++region_entries_;
                tool_window_.clear();
            }
        } else if (ends_with(tool_window_, tool_close_ids_)) {
            in_tool_region_ = false;
            last_exit_index_ = all_tokens_.size();
            tool_window_.clear();
        }
        // Hold the echo rule off until the window has refilled with tokens
        // generated entirely outside the region.
        if (in_tool_region_ || last_exit_index_ == all_tokens_.size())
            echo_block_until_ = all_tokens_.size() + kPromptEchoTokens;
    }

    std::vector<int32_t> natural_close_ids_;
    std::vector<int32_t> close_window_;
    std::vector<int32_t> prompt_tokens_;
    std::unordered_multimap<uint64_t, std::size_t> prompt_windows_;
    std::vector<int32_t> tool_open_ids_;
    std::vector<int32_t> tool_close_ids_;
    std::vector<int32_t> tool_window_;
    bool in_tool_region_ = false;
    std::size_t region_entries_ = 0;
    std::size_t last_exit_index_ = 0;
    std::size_t echo_block_until_ = 0;
    std::vector<int32_t> phase_tokens_;
    std::vector<uint64_t> phase_prefix_hash_;
    std::vector<uint64_t> phase_hash_power_;
    std::vector<int32_t> all_tokens_;
    std::vector<uint64_t> all_prefix_hash_;
    std::vector<uint64_t> all_hash_power_;
    bool visible_ = false;
    ProgressStopReason reason_ = ProgressStopReason::None;
    std::size_t cycle_period_ = 0;
    std::size_t repeated_span_ = 0;
    std::size_t prompt_echo_offset_ = 0;
};

}  // namespace dflash::common
