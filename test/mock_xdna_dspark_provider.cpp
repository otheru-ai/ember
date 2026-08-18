#include "dspark_draft_compute_xdna.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace {

struct MockContext {
    int n_embd;
    int n_target_layers;
    int block_size;
    int n_swa;
    int head_dim;
};

struct MockJob {
    int committed;
    int ctx_len;
    int n_embd;
    int n_target_layers;
    int block_size;
    std::vector<float> noise;
    std::vector<float> features;
    std::vector<float> main_context;
    std::vector<float> context_kv;
    bool cancelled = false;
};

void set_error(char * error, size_t capacity, const char * message) {
    if (!error || capacity == 0) return;
    std::strncpy(error, message, capacity - 1);
    error[capacity - 1] = '\0';
}

void * create(const ember_xdna_dspark_config_v1 * config,
              char * error, size_t capacity) {
    if (!config ||
        config->abi_version != EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION ||
        config->struct_size < sizeof(*config) || !config->draft_model_path ||
        std::strcmp(config->draft_model_path, "mock-draft.gguf") != 0 ||
        config->n_embd != 4 || config->n_target_layers != 3 ||
        config->block_size != 5 || config->n_swa != 8 ||
        config->head_dim != 2) {
        set_error(error, capacity, "unexpected mock DSpark configuration");
        return nullptr;
    }
    return new MockContext{config->n_embd, config->n_target_layers,
                           config->block_size, config->n_swa,
                           config->head_dim};
}

void * submit(void * raw, const ember_xdna_dspark_request_v1 * request,
              char * error, size_t capacity) {
    auto * context = static_cast<MockContext *>(raw);
    if (!context || !request ||
        request->abi_version != EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION ||
        request->struct_size < sizeof(*request) || request->committed < 0 ||
        request->ctx_len < 0 || request->ctx_len > context->n_swa ||
        request->n_embd != context->n_embd ||
        request->n_target_layers != context->n_target_layers ||
        request->block_size != context->block_size || !request->noise_embed ||
        (request->ctx_len > 0 && !request->ctx_features &&
         !request->main_context && !request->context_kv)) {
        set_error(error, capacity, "invalid mock DSpark request");
        return nullptr;
    }
    auto * job = new MockJob{};
    job->committed = request->committed;
    job->ctx_len = request->ctx_len;
    job->n_embd = request->n_embd;
    job->n_target_layers = request->n_target_layers;
    job->block_size = request->block_size;
    const size_t noise_count = static_cast<size_t>(request->n_embd) *
                               static_cast<size_t>(request->block_size);
    job->noise.assign(request->noise_embed, request->noise_embed + noise_count);
    const size_t feature_count = static_cast<size_t>(request->ctx_len) *
                                 static_cast<size_t>(request->n_target_layers) *
                                 static_cast<size_t>(request->n_embd);
    if (feature_count && request->ctx_features) {
        job->features.assign(request->ctx_features,
                             request->ctx_features + feature_count);
    }
    const size_t main_count = static_cast<size_t>(request->ctx_len) *
                              static_cast<size_t>(request->n_embd);
    if (main_count && request->main_context) {
        job->main_context.assign(request->main_context,
                                 request->main_context + main_count);
    }
    const size_t context_kv_count =
        static_cast<size_t>(request->ctx_len) *
        static_cast<size_t>(request->n_target_layers) *
        static_cast<size_t>(context->head_dim);
    if (context_kv_count && request->context_kv) {
        job->context_kv.assign(request->context_kv,
                               request->context_kv + context_kv_count);
    }
    return job;
}

int wait(void *, void * raw_job, ember_xdna_dspark_result_v1 * result,
         char * error, size_t capacity) {
    auto * job = static_cast<MockJob *>(raw_job);
    const size_t count = job ? static_cast<size_t>(job->n_embd) *
                               static_cast<size_t>(job->block_size) : 0;
    if (!job || job->cancelled || !result ||
        result->abi_version != EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION ||
        result->struct_size < sizeof(*result) || !result->hidden ||
        result->hidden_capacity < count || !result->confidence_hidden ||
        result->confidence_capacity < count) {
        set_error(error, capacity, "invalid mock DSpark wait");
        return 0;
    }
    const float feature = !job->context_kv.empty()
        ? job->context_kv.back()
        : !job->main_context.empty()
        ? job->main_context.back()
        : (job->features.empty() ? 0.0f : job->features.back());
    for (size_t i = 0; i < count; ++i) {
        result->hidden[i] = job->noise[i] + feature + (float)job->committed;
        result->confidence_hidden[i] = result->hidden[i] + 0.5f;
    }
    return 1;
}

void cancel(void *, void * raw_job) {
    if (auto * job = static_cast<MockJob *>(raw_job)) job->cancelled = true;
}
void destroy_job(void *, void * raw_job) {
    delete static_cast<MockJob *>(raw_job);
}
int healthy(void * raw) { return raw != nullptr; }
void destroy(void * raw) { delete static_cast<MockContext *>(raw); }

const ember_xdna_dspark_provider_v1 provider = {
    EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION,
    sizeof(ember_xdna_dspark_provider_v1),
    "mock-xdna-dspark",
    create,
    submit,
    wait,
    cancel,
    destroy_job,
    healthy,
    destroy,
};

}  // namespace

extern "C" const ember_xdna_dspark_provider_v1 *
ember_xdna_dspark_get_provider_v1() {
    return &provider;
}
