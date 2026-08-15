#include "moe_expert_compute_xdna.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

struct MockContext {
    int n_layer;
    int n_expert;
    int n_embd;
    int n_ff_exp;
};

void set_error(char * error, size_t capacity, const char * message) {
    if (!error || capacity == 0) return;
    std::strncpy(error, message, capacity - 1);
    error[capacity - 1] = '\0';
}

void * create(const ember_xdna_moe_config_v1 * config,
              char * error, size_t capacity) {
    if (!config || config->abi_version != EMBER_XDNA_MOE_PROVIDER_ABI_VERSION ||
        !config->model_path || std::strcmp(config->model_path, "mock.gguf") != 0 ||
        config->n_layer != 2 || config->n_expert != 32 || config->n_embd != 4 ||
        config->n_ff_exp != 8 || config->swiglu_clamp != 7.0f) {
        set_error(error, capacity, "unexpected mock configuration");
        return nullptr;
    }
    return new MockContext{config->n_layer, config->n_expert,
                           config->n_embd, config->n_ff_exp};
}

int compute(void * raw, const ember_xdna_moe_batch_v1 * batch,
            char * error, size_t capacity) {
    auto * context = static_cast<MockContext *>(raw);
    if (!context || !batch || batch->abi_version != EMBER_XDNA_MOE_PROVIDER_ABI_VERSION ||
        batch->struct_size < sizeof(ember_xdna_moe_batch_v1) ||
        batch->layer_idx < 0 || batch->layer_idx >= context->n_layer ||
        batch->n_embd != context->n_embd || batch->n_ff_exp != context->n_ff_exp ||
        !batch->expert_weights) {
        set_error(error, capacity, "invalid mock batch");
        return 0;
    }
    for (int token = 0; token < batch->n_tokens; ++token) {
        float routed = 0.0f;
        for (int slot = 0; slot < batch->n_selected; ++slot) {
            const size_t index = (size_t)token * (size_t)batch->n_selected +
                                 (size_t)slot;
            const auto & view = batch->expert_weights[index];
            if (view.struct_size < sizeof(ember_xdna_moe_weight_view_v1) ||
                !view.gate || !view.up || !view.down || view.gate_bytes != 4 ||
                view.up_bytes != 4 || view.down_bytes != 4 ||
                view.gate_format != EMBER_XDNA_MOE_WEIGHT_ROCMFP2 ||
                view.up_format != EMBER_XDNA_MOE_WEIGHT_ROCMFP2 ||
                view.down_format != EMBER_XDNA_MOE_WEIGHT_ROCMFP2 ||
                view.gate_scale != 0.5f || view.up_scale != 0.25f ||
                view.down_scale != 2.0f) {
                set_error(error, capacity, "invalid mock expert weight view");
                return 0;
            }
            const auto expected = static_cast<uint8_t>(batch->expert_ids[index]);
            if (*static_cast<const uint8_t *>(view.gate) != expected ||
                *static_cast<const uint8_t *>(view.up) != expected ||
                *static_cast<const uint8_t *>(view.down) != expected) {
                set_error(error, capacity, "mock expert weight offset is wrong");
                return 0;
            }
            routed += (float)(batch->expert_ids[index] + 1) *
                      batch->router_weights[index];
        }
        for (int column = 0; column < batch->n_embd; ++column) {
            const size_t index = (size_t)token * (size_t)batch->n_embd +
                                 (size_t)column;
            batch->output[index] = batch->input[index] + routed;
        }
    }
    return 1;
}

int healthy(void * raw) { return raw != nullptr; }
void destroy(void * raw) { delete static_cast<MockContext *>(raw); }

const ember_xdna_moe_provider_v1 provider = {
    EMBER_XDNA_MOE_PROVIDER_ABI_VERSION,
    sizeof(ember_xdna_moe_provider_v1),
    "mock-xdna",
    create,
    compute,
    healthy,
    destroy,
};

}  // namespace

extern "C" const ember_xdna_moe_provider_v1 * ember_xdna_moe_get_provider_v1() {
    return &provider;
}
