#pragma once

namespace dflash::common {

// Numerics/performance policy for model-specific prefill attention. Exact is
// the tokenwise reference. Dense changes the execution/reduction topology and
// Sparse additionally prunes model-specific cache entries; neither optimized
// mode promises byte-identical logits or generated tokens.
enum class PrefillAttentionMode {
    Exact,
    Dense,
    Sparse,
};

inline const char * prefill_attention_mode_name(PrefillAttentionMode mode) {
    switch (mode) {
        case PrefillAttentionMode::Exact:  return "exact";
        case PrefillAttentionMode::Dense:  return "dense";
        case PrefillAttentionMode::Sparse: return "sparse";
    }
    return "unknown";
}

inline bool prefill_attention_mode_is_approximate(PrefillAttentionMode mode) {
    return mode != PrefillAttentionMode::Exact;
}

// Select the attention policy for one prompt chunk. DSpark feature capture has
// an exact q=1 contract, but only its final capture window needs that policy;
// the prefix remains on the configured batched path. Decode policy is absent
// by design: target-only AR decode must not implicitly select exact prefill.
inline PrefillAttentionMode select_prefill_step_mode(
        PrefillAttentionMode configured,
        bool force_exact_prefill,
        bool capture_spec_features,
        int relative_offset,
        int capture_from) {
    if (force_exact_prefill ||
        (capture_spec_features && relative_offset >= capture_from))
        return PrefillAttentionMode::Exact;
    return configured;
}

} // namespace dflash::common
