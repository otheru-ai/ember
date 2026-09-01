// Ember model-backend construction contract.
//
// Architecture selection happens from `general.architecture`; model-specific
// options remain explicit below.  A recognized architecture whose runtime is
// not implemented must fail closed after structural validation.

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
    // Optional operator-owned DeepSeek native tower. It remains unopened until
    // the first image request and is never sourced from request metadata.
    const char *    vision_mmproj_path = nullptr;

    // The only supported device is local gfx1151 GPU 0.
    DevicePlacement device;

    // Chunked prefill
    int                  chunk             = 512;
    PrefillAttentionMode ds4_prefill_mode = PrefillAttentionMode::Exact;
    bool                 ds4_prefill_mode_set = false;

    // deepseek4-specific decode options
    int             ds4_expert_top_k = 0;  // 0 = model default
    bool            ds4_fused_decode = false;

    // Qwen3.8-Flash-Next operator override. The factory rejects this for
    // non-Qwen architectures; the Qwen loader resolves the exact factor-4,
    // 1,000,000-token recipe and still applies the 128-GiB residency gate.
    bool            qwen_yarn = false;

};

// ─── Factory function ───────────────────────────────────────────────────
// Inspects `general.architecture`, validates the selected model contract, then
// constructs the implemented backend. Qwen4Exp currently supports ordinary
// single-session autoregressive text generation; vision, MTP, and resident
// batching remain explicit unsupported capabilities.
std::unique_ptr<ModelBackend> create_backend(const BackendArgs & args);

}  // namespace dflash::common
