// ember_create_backend.cpp — DeepSeek4-only backend factory for Ember's
// standalone engine, replacing lucebox's multi-arch backend_factory.cpp (which
// #includes every architecture: qwen/gemma/laguna). Ember only ever serves
// DeepSeek-V4-Flash on a single local HIP device, so this always constructs the
// monolithic DeepSeek4Backend — no GGUF arch detection, no layer-split /
// remote-shard paths. The BackendArgs -> DeepSeek4BackendConfig mapping mirrors
// backend_factory.cpp's monolithic branch exactly.
#include "backend_factory.h"
#include "deepseek4_backend.h"
#include "deepseek4_internal.h"

#include <cstdio>

namespace dflash::common {

std::unique_ptr<ModelBackend> create_backend(const BackendArgs & args) {
    DeepSeek4BackendConfig cfg;
    cfg.model_path   = args.model_path;
    cfg.device       = args.device;
    cfg.stream_fd    = args.stream_fd;
    cfg.max_ctx      = args.device.max_ctx;
    cfg.chunk        = args.chunk;
    cfg.expert_top_k = args.ds4_expert_top_k;
    cfg.fused_decode = args.ds4_fused_decode;
    cfg.prefill_mode = args.ds4_prefill_mode;

    auto backend = std::make_unique<DeepSeek4Backend>(cfg);
    if (!backend->init()) {
        std::fprintf(stderr, "[ember] DeepSeek4Backend init failed\n");
        return nullptr;
    }
    return backend;
}

// Ember serves a single fixed arch; these satisfy backend_factory.h's remaining
// declarations (the Ember bridge doesn't call them, but define them so nothing
// under-links if a vendored TU references them).
std::string detect_arch(const char *) { return "deepseek4"; }
bool arch_supports_remote_draft(const std::string &) { return false; }
bool arch_supports_pflash_compression(const std::string &) { return false; }

}  // namespace dflash::common
