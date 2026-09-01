// ember_create_backend.cpp — architecture-selecting backend factory for
// Ember's standalone engine. Device placement remains the local gfx1151 seam;
// architecture selection is independent of transport/layer splitting.
#include "backend_factory.h"
#include "errors.h"
#include "deepseek4_backend.h"
#include "deepseek4_internal.h"
#include "qwen4exp_backend.h"
#include "qwen4exp_model.h"

#include <cstdio>
#include <string>

namespace dflash::common {

std::unique_ptr<ModelBackend> create_backend(const BackendArgs & args) {
    std::string architecture_error;
    const ModelArchitecture architecture =
        inspect_gguf_architecture(args.model_path, architecture_error);
    if (architecture == ModelArchitecture::UNKNOWN) {
        set_last_error(architecture_error);
        std::fprintf(stderr, "[ember] backend selection failed: %s\n",
                     architecture_error.c_str());
        return nullptr;
    }

    if (architecture == ModelArchitecture::QWEN4_EXP) {
        if (args.allow_single_layer_control) {
            set_last_error("--allow-single-layer-control is valid only for DeepSeek4 diagnostics");
            std::fprintf(stderr, "[ember] backend selection failed: %s\n",
                         last_error());
            return nullptr;
        }
        if (args.vision_mmproj_path && args.vision_mmproj_path[0]) {
            set_last_error("--vision-mmproj is valid only for DeepSeek4 native vision");
            std::fprintf(stderr, "[ember] backend selection failed: %s\n",
                         last_error());
            return nullptr;
        }
        Qwen4ExpBackendConfig config;
        config.model_path = args.model_path;
        config.device = args.device;
        config.max_ctx = args.device.max_ctx;
        config.enable_yarn = args.qwen_yarn;
        auto backend = std::make_unique<Qwen4ExpBackend>(config);
        if (!backend->init()) {
            const char * diagnostic = last_error();
            std::fprintf(stderr, "[ember] Qwen4ExpBackend init failed: %s\n",
                         diagnostic && diagnostic[0]
                             ? diagnostic : "no backend diagnostic");
            return nullptr;
        }
        return backend;
    }

    if (args.qwen_yarn) {
        set_last_error("--qwen-yarn is valid only for Qwen3.8-Flash-Next");
        std::fprintf(stderr, "[ember] backend selection failed: %s\n",
                     last_error());
        return nullptr;
    }

    DeepSeek4BackendConfig cfg;
    cfg.model_path   = args.model_path;
    if (args.vision_mmproj_path)
        cfg.vision_mmproj_path = args.vision_mmproj_path;
    cfg.device       = args.device;
    cfg.max_ctx      = args.device.max_ctx;
    cfg.chunk        = args.chunk;
    cfg.expert_top_k = args.ds4_expert_top_k;
    cfg.fused_decode = args.ds4_fused_decode;
    cfg.prefill_mode = args.ds4_prefill_mode;
    cfg.allow_single_layer_control = args.allow_single_layer_control;

    auto backend = std::make_unique<DeepSeek4Backend>(cfg);
    if (!backend->init()) {
        const char * diagnostic = last_error();
        std::fprintf(stderr, "[ember] DeepSeek4Backend init failed: %s\n",
                     diagnostic && diagnostic[0]
                         ? diagnostic : "no backend diagnostic");
        return nullptr;
    }
    return backend;
}

}  // namespace dflash::common
