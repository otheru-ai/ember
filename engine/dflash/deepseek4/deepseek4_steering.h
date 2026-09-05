#pragma once

#include "deepseek4_internal.h"

namespace dflash::common {

// Shared by target prefill, AR and speculative verification graph builders.
// Project only the complete sublayer output, before HC post. In particular,
// do not project shared/routed or gathered vision/text FFN pieces separately.
// No nodes are added for absent policy, zero scale, or an excluded layer.
// The dot product uses sum_rows' width-only reduction tree, not a matmul whose
// implementation changes with speculative batch width.
inline ggml_tensor * ds4_directional_steering(
        ggml_context *ctx, ggml_tensor *output, const DeepSeek4Weights &w,
        int layer, bool ffn) {
    const auto *policy = w.directional_steering.get();
    if (!output || !policy) return output;
    GGML_ASSERT(layer >= 0 && layer < EMBER_STEERING_LAYERS);
    const float scale = ffn ? policy->ffn_scale : policy->attn_scale;
    if (scale == 0.0f || !policy->nonzero[layer]) return output;
    GGML_ASSERT(w.steering_rows.size() == EMBER_STEERING_LAYERS);
    auto *direction = w.steering_rows[static_cast<size_t>(layer)];
    GGML_ASSERT(direction && output->ne[0] == EMBER_STEERING_WIDTH);
    auto *dot = ggml_sum_rows(ctx, ggml_mul(ctx, output, direction));
    auto *coefficient = ggml_scale(ctx, dot, scale);
    auto *correction = ggml_mul(ctx, ggml_repeat(ctx, direction, output), coefficient);
    auto *result = ggml_sub(ctx, output, correction);
    ggml_format_name(result, "steering.%s.blk.%d", ffn ? "ffn" : "attn", layer);
    return result;
}

} // namespace dflash::common
