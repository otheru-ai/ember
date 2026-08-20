// Ember's fixed DeepSeek-V4-Flash backend construction contract.
//
// The vendored upstream factory supported multiple architectures, remote
// drafters, and GPU layer splitting. Ember is a single-gfx1151 appliance, so
// this header exposes only the arguments consumed by ember_create_backend.cpp.

#pragma once

#include "model_backend.h"
#include "placement/placement_config.h"
#include "prefill_attention_mode.h"

#include <memory>
#include <string>

namespace dflash::common {

// ─── Backend creation arguments ─────────────────────────────────────────
struct BackendArgs {
    // Required
    const char *    model_path   = nullptr;   // target .gguf

    // The only supported device is local gfx1151 GPU 0.
    DevicePlacement device;

    // Chunked prefill
    int                  chunk             = 512;
    PrefillAttentionMode ds4_prefill_mode = PrefillAttentionMode::Exact;
    bool                 ds4_prefill_mode_set = false;

    // deepseek4-specific decode options
    int             ds4_expert_top_k = 0;  // 0 = model default
    bool            ds4_fused_decode = false;

};

// ─── Factory function ───────────────────────────────────────────────────
// Constructs DeepSeek4Backend and calls init().
std::unique_ptr<ModelBackend> create_backend(const BackendArgs & args);

}  // namespace dflash::common
