// XDNA2 selected-expert provider ABI.
//
// Ryzen AI's XDNA runtime and kernel toolchain move independently from ROCm.
// Loading a provider through this narrow C ABI keeps the vendored HIP engine
// buildable without XRT while giving an IRON/XRT implementation the exact
// token grouping produced by Ember's MoE scheduler. This follows AMD's
// GPT-OSS QMoE split: CPU control/routing, accelerator expert matmuls.
#pragma once

#include "moe_expert_compute.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#define EMBER_XDNA_MOE_PROVIDER_ABI_VERSION 1u
#define EMBER_XDNA_MOE_PROVIDER_SYMBOL "ember_xdna_moe_get_provider_v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ember_xdna_moe_config_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char * model_path;
    int32_t n_layer;
    int32_t n_expert;
    int32_t n_expert_used;
    int32_t n_embd;
    int32_t n_ff_exp;
    float swiglu_clamp;
} ember_xdna_moe_config_v1;

typedef enum ember_xdna_moe_weight_format_v1 {
    EMBER_XDNA_MOE_WEIGHT_NONE = 0,
    EMBER_XDNA_MOE_WEIGHT_ROCMFP2 = 1,
} ember_xdna_moe_weight_format_v1;

// One placement-local selected expert. The pointers remain valid only for the
// duration of compute(). Keeping the views in the request lets an XRT provider
// lazily pre-tile the exact mmap-backed expert selected by Ember without
// independently reparsing GGUF or duplicating the 85 GiB model mapping. XRT
// still owns its BO mappings; UMA does not imply ROCm/XRT allocation coherence.
typedef struct ember_xdna_moe_weight_view_v1 {
    uint32_t struct_size;
    uint32_t fused_gate_up;
    const void * gate;
    size_t gate_bytes;
    const void * up;
    size_t up_bytes;
    const void * down;
    size_t down_bytes;
    const void * gate_up;
    size_t gate_up_bytes;
    uint32_t gate_format;
    uint32_t up_format;
    uint32_t down_format;
    uint32_t gate_up_format;
    float gate_scale;
    float up_scale;
    float down_scale;
    float gate_up_scale;
} ember_xdna_moe_weight_view_v1;

typedef struct ember_xdna_moe_batch_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t layer_idx;
    int32_t n_tokens;
    int32_t n_selected;
    int32_t n_embd;
    int32_t n_ff_exp;
    const float * input;          // [n_tokens, n_embd]
    const int32_t * expert_ids;   // global ids [n_tokens, n_selected]
    const float * router_weights; // [n_tokens, n_selected]
    float * output;               // weighted sum [n_tokens, n_embd]
    // Same slot order as expert_ids. Added at the end of ABI v1 so providers
    // that only use routing metadata remain source-compatible.
    const ember_xdna_moe_weight_view_v1 * expert_weights;
} ember_xdna_moe_batch_v1;

typedef struct ember_xdna_moe_provider_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char * name;
    void * (*create)(const ember_xdna_moe_config_v1 * config,
                     char * error, size_t error_capacity);
    int (*compute)(void * context, const ember_xdna_moe_batch_v1 * batch,
                   char * error, size_t error_capacity);
    int (*healthy)(void * context);
    void (*destroy)(void * context);
} ember_xdna_moe_provider_v1;

typedef const ember_xdna_moe_provider_v1 *
    (*ember_xdna_moe_provider_entry_v1)(void);

#ifdef __cplusplus
}
#endif

namespace dflash::common {

struct XdnaMoeExpertComputeConfig {
    std::string plugin_path;
    std::string model_path;
    int n_layer = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    int n_embd = 0;
    int n_ff_exp = 0;
    float swiglu_clamp = 0.0f;
    int min_tokens = 1;
    bool required = false;
};

std::unique_ptr<MoeExpertCompute> make_xdna_moe_expert_compute(
    const XdnaMoeExpertComputeConfig & config,
    std::string * error = nullptr);

}  // namespace dflash::common
