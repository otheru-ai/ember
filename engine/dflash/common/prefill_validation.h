#pragma once

// The Qwen production-prefill validator compares two deliberately different
// quantized-matmul families. Token disagreement is acceptable only when the
// authoritative q1 top-two decision is less stable than the measured logit
// perturbation between paths. This is self-calibrating: there is no model- or
// hardware-specific epsilon to tune.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace dflash::common {

struct PrefillMarginDecision {
    bool streams_exact = false;
    bool margin_checked = false;
    bool accepted = false;
    size_t mismatch_index = std::numeric_limits<size_t>::max();
    size_t numerics_index = std::numeric_limits<size_t>::max();
    int32_t expected_token = -1;
    int32_t actual_token = -1;
    float q1_top2_margin = 0.0f;
    float max_abs_logit_delta = 0.0f;
};

inline bool prefill_top2_margin(const std::vector<float> & logits,
                                float & margin) {
    if (logits.size() < 2) return false;
    size_t first = 0;
    size_t second = 1;
    if (!std::isfinite(logits[first]) || !std::isfinite(logits[second]))
        return false;
    if (logits[second] > logits[first]) std::swap(first, second);
    for (size_t index = 2; index < logits.size(); ++index) {
        if (!std::isfinite(logits[index])) return false;
        if (logits[index] > logits[first]) {
            second = first;
            first = index;
        } else if (logits[index] > logits[second]) {
            second = index;
        }
    }
    margin = logits[first] - logits[second];
    return std::isfinite(margin) && margin >= 0.0f;
}

inline bool prefill_row_metrics(const std::vector<float> & reference,
                                const std::vector<float> & actual,
                                float & margin, float & max_abs) {
    if (reference.size() != actual.size() || reference.size() < 2 ||
        !prefill_top2_margin(reference, margin)) return false;
    max_abs = 0.0f;
    for (size_t index = 0; index < reference.size(); ++index) {
        if (!std::isfinite(actual[index])) return false;
        const float delta = std::fabs(reference[index] - actual[index]);
        if (!std::isfinite(delta)) return false;
        max_abs = std::max(max_abs, delta);
    }
    return true;
}

inline PrefillMarginDecision validate_prefill_margin(
        const std::vector<int32_t> & q1_tokens,
        const std::vector<int32_t> & production_tokens,
        const std::vector<std::vector<float>> & q1_logits,
        const std::vector<std::vector<float>> & production_logits) {
    PrefillMarginDecision decision;
    const size_t common = std::min(q1_tokens.size(), production_tokens.size());
    size_t mismatch = common;
    for (size_t index = 0; index < common; ++index) {
        if (q1_tokens[index] != production_tokens[index]) {
            mismatch = index;
            break;
        }
    }
    if (mismatch == common && q1_tokens.size() == production_tokens.size()) {
        decision.streams_exact = true;
        decision.accepted = true;
        if (q1_logits.size() != production_logits.size()) return decision;
        for (size_t row = 0; row < q1_logits.size(); ++row) {
            float margin = 0.0f;
            float max_abs = 0.0f;
            if (!prefill_row_metrics(q1_logits[row], production_logits[row],
                                     margin, max_abs)) return decision;
            if (!decision.margin_checked ||
                max_abs > decision.max_abs_logit_delta) {
                decision.margin_checked = true;
                decision.numerics_index = row;
                decision.q1_top2_margin = margin;
                decision.max_abs_logit_delta = max_abs;
            }
        }
        return decision;
    }

    decision.mismatch_index = mismatch;
    decision.expected_token =
        mismatch < q1_tokens.size() ? q1_tokens[mismatch] : -1;
    decision.actual_token =
        mismatch < production_tokens.size() ? production_tokens[mismatch] : -1;
    if (mismatch >= q1_logits.size() ||
        mismatch >= production_logits.size()) return decision;
    if (!prefill_row_metrics(q1_logits[mismatch], production_logits[mismatch],
                             decision.q1_top2_margin,
                             decision.max_abs_logit_delta)) return decision;
    decision.margin_checked = true;
    decision.numerics_index = mismatch;
    decision.accepted =
        decision.q1_top2_margin < decision.max_abs_logit_delta;
    return decision;
}

} // namespace dflash::common
