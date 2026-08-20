// ember_create_backend.cpp — DeepSeek4-only backend factory for Ember's
// standalone engine. Ember serves DeepSeek-V4-Flash on one local gfx1151 HIP
// device, so no architecture detection, remote transport, or layer splitting
// belongs in this factory.
#include "backend_factory.h"
#include "deepseek4_backend.h"
#include "deepseek4_internal.h"

#include <cstdio>

namespace dflash::common {

std::unique_ptr<ModelBackend> create_backend(const BackendArgs & args) {
    DeepSeek4BackendConfig cfg;
    cfg.model_path   = args.model_path;
    cfg.device       = args.device;
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

}  // namespace dflash::common
