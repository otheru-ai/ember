#pragma once

// The production-prefill validator compares two deliberately different
// quantized-matmul families. Token disagreement is acceptable only when the
// authoritative q1 top-two decision is less stable than the measured logit
// perturbation between paths. This is self-calibrating: there is no model- or
// hardware-specific epsilon to tune.
//
// That test alone is NOT sufficient, and the failure is measured rather than
// hypothetical. A width-4 dense-MMQ control returned token-exact and accepted
// while its logit vectors correlated only 0.556/0.601 with q1 -- both argmax
// tokens happened to survive, so nothing token-level could see it. See
// the retained performance evidence.
//
// The reason no margin rule can catch that at any threshold: both quantities it
// compares (top-2 margin, max abs delta) are top-of-distribution order
// statistics, and greedy argmax is a single order statistic that does not
// constrain the distribution beneath it. What needs bounding is DISTRIBUTIONAL
// agreement, because the server samples (--default-temperature 0.6) rather than
// taking argmax.
//
// So acceptance additionally requires a total-variation bound between the two
// softmaxed distributions. TV is the maximum probability difference over any
// event, so it states directly how differently the two paths could sample:
// TV = 0.5 means up to ~50% of sampled tokens could differ. Observed on retained
// rows: genuinely-equivalent 0.0000-0.0002, the token-exact-but-broken control
// 0.139-0.507, known-red widths 0.71-0.94. The default threshold sits two orders
// of magnitude clear of both sides, so it is a separation, not a tuned epsilon.
//
// TV is evaluated at BOTH the canonical T=1.0 and the serving temperature, and
// the larger is used: neither dominates (the control is worse at 1.0, the
// width-6 rows worse at 0.6), so checking one alone is not conservative.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// Total-variation acceptance bound between the q1 and production distributions.
// Two orders of magnitude clear of both the equivalent and the broken rows.
#ifndef DFLASH_PREFILL_TV_THRESHOLD
#define DFLASH_PREFILL_TV_THRESHOLD 0.01f
#endif

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
    bool  tv_checked = false;
    bool  tv_within_bound = false;
    float tv_distance = 0.0f;
    float tv_threshold = DFLASH_PREFILL_TV_THRESHOLD;
    size_t tv_index = std::numeric_limits<size_t>::max();
};

// Softmax into `out` at temperature `t`, max-subtracted so that a large logit
// cannot overflow. Returns false on a non-finite input or a degenerate sum.
inline bool prefill_softmax(const std::vector<float> & logits, float t,
                            std::vector<double> & out) {
    if (logits.empty() || !(t > 0.0f)) return false;
    out.resize(logits.size());
    float peak = logits[0];
    for (const float value : logits) {
        if (!std::isfinite(value)) return false;
        peak = std::max(peak, value);
    }
    double total = 0.0;
    for (size_t index = 0; index < logits.size(); ++index) {
        const double weight =
            std::exp((static_cast<double>(logits[index]) -
                      static_cast<double>(peak)) / static_cast<double>(t));
        out[index] = weight;
        total += weight;
    }
    if (!(total > 0.0) || !std::isfinite(total)) return false;
    for (double & weight : out) weight /= total;
    return true;
}

// Total variation distance: half the L1 distance between two distributions.
inline bool prefill_total_variation(const std::vector<float> & reference,
                                    const std::vector<float> & actual,
                                    float temperature, float & distance) {
    if (reference.size() != actual.size() || reference.empty()) return false;
    std::vector<double> p, q;
    if (!prefill_softmax(reference, temperature, p)) return false;
    if (!prefill_softmax(actual, temperature, q)) return false;
    double sum = 0.0;
    for (size_t index = 0; index < p.size(); ++index)
        sum += std::fabs(p[index] - q[index]);
    const double result = 0.5 * sum;
    if (!std::isfinite(result)) return false;
    distance = static_cast<float>(result);
    return true;
}

// The larger TV over the canonical and serving temperatures. Neither dominates,
// so the maximum is the only conservative choice.
inline bool prefill_total_variation_worst(const std::vector<float> & reference,
                                          const std::vector<float> & actual,
                                          float serving_temperature,
                                          float & distance) {
    float canonical = 0.0f;
    if (!prefill_total_variation(reference, actual, 1.0f, canonical))
        return false;
    distance = canonical;
    if (serving_temperature > 0.0f && serving_temperature != 1.0f) {
        float served = 0.0f;
        if (!prefill_total_variation(reference, actual, serving_temperature,
                                     served))
            return false;
        distance = std::max(distance, served);
    }
    return true;
}

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
        const std::vector<std::vector<float>> & production_logits,
        float serving_temperature = 0.6f,
        float tv_threshold = DFLASH_PREFILL_TV_THRESHOLD) {
    PrefillMarginDecision decision;
    decision.tv_threshold = tv_threshold;

    // Distributional agreement over every captured row, independent of tokens.
    // Runs first and unconditionally: the defect this exists to catch is
    // token-exact, so gating it behind a token mismatch would reproduce the
    // hole. A row whose TV cannot be computed leaves tv_checked false and
    // therefore fails closed below.
    const size_t rows = std::min(q1_logits.size(), production_logits.size());
    if (rows > 0 && q1_logits.size() == production_logits.size()) {
        bool all_rows_ok = true;
        for (size_t row = 0; row < rows; ++row) {
            float distance = 0.0f;
            if (!prefill_total_variation_worst(
                    q1_logits[row], production_logits[row],
                    serving_temperature, distance)) {
                all_rows_ok = false;
                break;
            }
            if (!decision.tv_checked || distance > decision.tv_distance) {
                decision.tv_distance = distance;
                decision.tv_index = row;
            }
            decision.tv_checked = true;
        }
        if (!all_rows_ok) {
            decision.tv_checked = false;
        }
    }
    decision.tv_within_bound =
        decision.tv_checked && decision.tv_distance <= tv_threshold;
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
        // Token-exact is NOT sufficient. The measured counter-example was
        // token-exact with TV up to 0.507, i.e. up to ~51% of sampled tokens
        // could have differed. When logits were captured, exactness is accepted
        // only alongside distributional agreement; with no logits at all there
        // is nothing to check and the token result stands as before.
        decision.accepted =
            q1_logits.empty() ? true : decision.tv_within_bound;
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
    // Both conditions: the flip must be attributable to an unstable q1 decision
    // (the original, user-decided margin rule) AND the distributions must agree
    // closely enough that sampling behaviour is preserved.
    decision.accepted =
        decision.q1_top2_margin < decision.max_abs_logit_delta &&
        decision.tv_within_bound;
    return decision;
}

} // namespace dflash::common
