#include "qwen4exp_frontier.h"

#include "qwen4exp_internal.h"
#include "qwen4exp_mtp.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>

namespace dflash::common {
namespace {

constexpr size_t kGraphContextBytes = 512U * 1024U;

bool env_enabled(const char * name, bool fallback) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) return fallback;
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "off") != 0;
}

bool tensor_shape(const ggml_tensor * tensor, int dimensions,
                  int64_t ne0, int64_t ne1, int64_t ne2 = 1) {
    const int actual_dimensions = tensor ? ggml_n_dims(tensor) : 0;
    return tensor && tensor->buffer && actual_dimensions <= dimensions &&
           actual_dimensions >= (dimensions == 3 ? 3 : 1) &&
           tensor->ne[0] == ne0 && tensor->ne[1] == ne1 &&
           (dimensions < 3 || tensor->ne[2] == ne2);
}

void set_layer_name(ggml_tensor * tensor, int layer, const char * suffix) {
    if (!tensor) return;
    char name[GGML_MAX_NAME];
    if (layer >= 0) {
        std::snprintf(name, sizeof(name), "qwen_moe_l%02d_%s", layer, suffix);
    } else {
        std::snprintf(name, sizeof(name), "qwen_moe_test_%s", suffix);
    }
    ggml_set_name(tensor, name);
}

struct RoctxApi {
    using Push = int (*)(const char *);
    using Pop = int (*)();
    void * handle = nullptr;
    Push push = nullptr;
    Pop pop = nullptr;

    RoctxApi() {
        if (!env_enabled("EMBER_QWEN_PROFILE_RANGES", false)) return;
        handle = dlopen("libroctx64.so", RTLD_LAZY | RTLD_LOCAL);
        if (!handle) handle = dlopen("libroctx64.so.4", RTLD_LAZY | RTLD_LOCAL);
        if (!handle) return;
        push = reinterpret_cast<Push>(dlsym(handle, "roctxRangePushA"));
        pop = reinterpret_cast<Pop>(dlsym(handle, "roctxRangePop"));
        if (!push || !pop) {
            dlclose(handle);
            handle = nullptr;
            push = nullptr;
            pop = nullptr;
        }
    }

    ~RoctxApi() {
        if (handle) dlclose(handle);
    }
};

RoctxApi & roctx_api() {
    static RoctxApi api;
    return api;
}

class ProfileRange {
public:
    explicit ProfileRange(const char * label) : api_(roctx_api()) {
        active_ = api_.push && api_.pop;
        if (active_) api_.push(label);
    }
    ~ProfileRange() {
        if (active_) api_.pop();
    }

private:
    RoctxApi & api_;
    bool active_ = false;
};

} // namespace

struct Qwen4ExpFrontierMoeGraph {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    Qwen4ExpFrontierMoeSpec spec{};
    int n_tokens = 1;
    int layer = -1;
    uint64_t calls = 0;
    uint64_t compute_us = 0;
    size_t arena_bytes = 0;
    char profile_label[64]{};
};

struct Qwen4ExpFrontierRuntime {
    using LayerGraphs = std::array<Qwen4ExpFrontierMoeGraph *,
        static_cast<size_t>(kQwen4ExpFrontierMoeMaxBatch + 1)>;
    std::vector<LayerGraphs> moe;
    bool stats = false;
};

int qwen4exp_frontier_moe_cached_width(int n_tokens) {
    if (n_tokens == 1) return 1;
    if (n_tokens >= 2 && n_tokens <= kQwen4ExpFrontierMoeMtpBatch)
        return kQwen4ExpFrontierMoeMtpBatch;
    if (n_tokens > kQwen4ExpFrontierMoeMtpBatch &&
        n_tokens <= kQwen4ExpFrontierMoeMaxBatch)
        return kQwen4ExpFrontierMoeMaxBatch;
    return 0;
}

Qwen4ExpFrontierMoeGraph * qwen4exp_frontier_moe_create(
        ggml_backend_t backend, const Qwen4ExpFrontierMoeSpec & spec,
        const Qwen4ExpFrontierMoeWeights & weights, int layer,
        std::string & error) {
    return qwen4exp_frontier_moe_create_batch(
        backend, spec, weights, layer, 1, error);
}

Qwen4ExpFrontierMoeGraph * qwen4exp_frontier_moe_create_batch(
        ggml_backend_t backend, const Qwen4ExpFrontierMoeSpec & spec,
        const Qwen4ExpFrontierMoeWeights & weights, int layer, int n_tokens,
        std::string & error) {
    error.clear();
    if (!backend || spec.n_embd <= 0 || spec.n_expert <= 0 ||
        spec.n_expert_used <= 0 || spec.n_expert_used > spec.n_expert ||
        spec.n_ff <= 0 || n_tokens <= 0 ||
        n_tokens > kQwen4ExpFrontierMoeMaxBatch ||
        !tensor_shape(weights.router, 2, spec.n_embd, spec.n_expert) ||
        !tensor_shape(weights.experts_gate_up, 3, spec.n_embd,
                      2LL * spec.n_ff, spec.n_expert) ||
        !tensor_shape(weights.experts_down, 3, spec.n_ff,
                      spec.n_embd, spec.n_expert) ||
        !tensor_shape(weights.shared_gate_input, 2, spec.n_embd, 1) ||
        !tensor_shape(weights.shared_gate, 2, spec.n_embd, spec.n_ff) ||
        !tensor_shape(weights.shared_up, 2, spec.n_embd, spec.n_ff) ||
        !tensor_shape(weights.shared_down, 2, spec.n_ff, spec.n_embd)) {
        error = "invalid Qwen4Exp frontier MoE tensor contract";
        return nullptr;
    }

    std::unique_ptr<Qwen4ExpFrontierMoeGraph> result(
        new Qwen4ExpFrontierMoeGraph());
    result->backend = backend;
    result->spec = spec;
    result->layer = layer;
    result->n_tokens = n_tokens;
    std::snprintf(result->profile_label, sizeof(result->profile_label),
                  "qwen4exp/moe/layer_%02d/q%d", layer, n_tokens);

    ggml_init_params params{};
    params.mem_size = kGraphContextBytes;
    params.no_alloc = true;
    result->ctx = ggml_init(params);
    if (!result->ctx) {
        error = "Qwen4Exp frontier MoE context allocation failed";
        return nullptr;
    }

    ggml_context * ctx = result->ctx;
    result->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                       spec.n_embd, n_tokens);
    ggml_set_input(result->input);
    set_layer_name(result->input, layer, "input");

    ggml_tensor * logits = ggml_mul_mat(ctx, weights.router, result->input);
    set_layer_name(logits, layer, "router");
    ggml_tensor * probabilities = ggml_soft_max(ctx, logits);
    ggml_tensor * selected = ggml_top_k(ctx, logits, spec.n_expert_used);
    set_layer_name(selected, layer, "topk");
    ggml_tensor * probabilities_3d =
        ggml_reshape_3d(ctx, probabilities, 1, spec.n_expert, n_tokens);
    ggml_tensor * selected_weights =
        ggml_get_rows(ctx, probabilities_3d, selected);
    selected_weights = ggml_reshape_2d(ctx, selected_weights,
                                       spec.n_expert_used, n_tokens);
    ggml_tensor * selected_sum = ggml_sum_rows(ctx, selected_weights);
    selected_sum = ggml_clamp(ctx, selected_sum,
                              std::numeric_limits<float>::min(), INFINITY);
    selected_weights = ggml_div(ctx, selected_weights, selected_sum);
    set_layer_name(selected_weights, layer, "weights");

    ggml_tensor * input_3d =
        ggml_reshape_3d(ctx, result->input, spec.n_embd, 1, n_tokens);
    ggml_tensor * gate_up = ggml_mul_mat_id(
        ctx, weights.experts_gate_up, input_3d, selected);
    set_layer_name(gate_up, layer, "gate_up");
    ggml_tensor * gate = ggml_view_3d(
        ctx, gate_up, spec.n_ff, gate_up->ne[1], gate_up->ne[2],
        gate_up->nb[1], gate_up->nb[2], 0);
    ggml_tensor * up = ggml_view_3d(
        ctx, gate_up, spec.n_ff, gate_up->ne[1], gate_up->ne[2],
        gate_up->nb[1], gate_up->nb[2],
        static_cast<size_t>(spec.n_ff) * ggml_element_size(gate_up));
    gate = ggml_cont(ctx, gate);
    up = ggml_cont(ctx, up);
    ggml_tensor * activated = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * expert_output = ggml_mul_mat_id(
        ctx, weights.experts_down, activated, selected);
    set_layer_name(expert_output, layer, "down");
    expert_output = ggml_reshape_3d(
        ctx, expert_output, spec.n_embd, spec.n_expert_used, n_tokens);
    ggml_tensor * weighted = ggml_mul(
        ctx, expert_output,
        ggml_reshape_3d(ctx, selected_weights, 1, spec.n_expert_used,
                        n_tokens));
    weighted = ggml_cont(ctx, ggml_permute(ctx, weighted, 1, 0, 2, 3));
    ggml_tensor * routed = ggml_sum_rows(ctx, weighted);
    routed = ggml_reshape_2d(ctx, routed, spec.n_embd, n_tokens);

    ggml_tensor * shared_gate =
        ggml_mul_mat(ctx, weights.shared_gate, result->input);
    ggml_tensor * shared_up =
        ggml_mul_mat(ctx, weights.shared_up, result->input);
    ggml_tensor * shared = ggml_mul_mat(
        ctx, weights.shared_down,
        ggml_swiglu_split(ctx, shared_gate, shared_up));
    // Avoid the known M=1 batched GEMM failure mode: the scalar shared gate is
    // a broadcast multiply and row reduction, exactly as in the measured DS4
    // cached FFN graph.
    ggml_tensor * shared_scale = ggml_sum_rows(
        ctx, ggml_mul(ctx, result->input, weights.shared_gate_input));
    shared = ggml_mul(ctx, shared, ggml_sigmoid(ctx, shared_scale));
    set_layer_name(shared, layer, "shared");

    result->output = ggml_add(ctx, routed, shared);
    set_layer_name(result->output, layer, "output");
    ggml_set_output(result->output);
    result->graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(result->graph, result->output);
    result->allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!result->allocator ||
        !ggml_gallocr_alloc_graph(result->allocator, result->graph)) {
        error = "Qwen4Exp frontier MoE graph allocation failed";
        qwen4exp_frontier_moe_destroy(result.release());
        return nullptr;
    }
    result->arena_bytes =
        ggml_gallocr_get_buffer_size(result->allocator, 0);
    return result.release();
}

void qwen4exp_frontier_moe_destroy(Qwen4ExpFrontierMoeGraph * graph) {
    if (!graph) return;
    if (graph->allocator) ggml_gallocr_free(graph->allocator);
    if (graph->ctx) ggml_free(graph->ctx);
    delete graph;
}

bool qwen4exp_frontier_moe_eval(Qwen4ExpFrontierMoeGraph * graph,
                                const float * input, size_t input_count,
                                std::vector<float> & output,
                                std::string & error) {
    if (!graph || !graph->backend || !graph->input || !graph->output ||
        !input || input_count != static_cast<size_t>(graph->spec.n_embd) *
                                  static_cast<size_t>(graph->n_tokens)) {
        error = "invalid Qwen4Exp frontier MoE evaluation";
        return false;
    }
    ggml_backend_tensor_set(graph->input, input, 0,
                            input_count * sizeof(float));
    const auto begin = std::chrono::steady_clock::now();
    const ProfileRange range(graph->profile_label);
    const ggml_status status =
        ggml_backend_graph_compute(graph->backend, graph->graph);
    const auto end = std::chrono::steady_clock::now();
    if (status != GGML_STATUS_SUCCESS) {
        error = "Qwen4Exp frontier MoE graph execution failed";
        return false;
    }
    output.resize(input_count);
    ggml_backend_tensor_get(graph->output, output.data(), 0,
                            output.size() * sizeof(float));
    ++graph->calls;
    graph->compute_us += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    return true;
}

bool qwen4exp_frontier_create(Qwen4ExpWeights & weights, std::string & error) {
    qwen4exp_frontier_destroy(weights);
    if (!env_enabled("EMBER_QWEN_FRONTIER_MOE", true)) {
        std::fprintf(stderr,
                     "[qwen-frontier] event=disabled component=moe source=env\n");
        return true;
    }
    if (!weights.backend || weights.layers.size() != 48) {
        error = "invalid Qwen4Exp weights for frontier initialization";
        return false;
    }
    std::unique_ptr<Qwen4ExpFrontierRuntime> runtime(
        new Qwen4ExpFrontierRuntime());
    runtime->stats = env_enabled("EMBER_QWEN_FRONTIER_STATS", false);
    runtime->moe.resize(weights.layers.size());
    const Qwen4ExpFrontierMoeSpec spec{2560, 512, 10, 640};
    for (size_t index = 0; index < weights.layers.size(); ++index) {
        const Qwen4ExpLayer & layer = weights.layers[index];
        const Qwen4ExpFrontierMoeWeights graph_weights{
            layer.router, layer.experts_gate_up_tensor,
            layer.experts_down_tensor, layer.shared_gate_input,
            layer.shared_gate, layer.shared_up, layer.shared_down};
        Qwen4ExpFrontierMoeGraph * graph = qwen4exp_frontier_moe_create(
            weights.backend, spec, graph_weights, static_cast<int>(index), error);
        if (!graph) {
            for (const Qwen4ExpFrontierRuntime::LayerGraphs & layer_graphs :
                 runtime->moe) {
                for (Qwen4ExpFrontierMoeGraph * built : layer_graphs)
                    qwen4exp_frontier_moe_destroy(built);
            }
            return false;
        }
        runtime->moe[index][1] = graph;
    }
    weights.frontier = runtime.release();
    std::fprintf(stderr,
                 "[qwen-frontier] event=ready component=moe graphs=48 "
                 "tokens_per_graph=1 lazy_batch_widths=5,16 "
                 "cached_graphs_per_layer=3 graph_replay=off\n");
    return true;
}

void qwen4exp_frontier_destroy(Qwen4ExpWeights & weights) {
    if (!weights.frontier) return;
    Qwen4ExpFrontierRuntime * runtime = weights.frontier;
    if (runtime->stats) {
        uint64_t calls = 0;
        uint64_t compute_us = 0;
        for (const Qwen4ExpFrontierRuntime::LayerGraphs & layer_graphs :
             runtime->moe) {
            for (const Qwen4ExpFrontierMoeGraph * graph : layer_graphs) {
                if (!graph) continue;
                calls += graph->calls;
                compute_us += graph->compute_us;
            }
        }
        const double average = calls ? static_cast<double>(compute_us) / calls : 0.0;
        std::fprintf(stderr,
                     "[qwen-frontier] event=summary component=moe calls=%llu "
                     "compute_us=%llu avg_compute_us=%.3f\n",
                     static_cast<unsigned long long>(calls),
                     static_cast<unsigned long long>(compute_us), average);
    }
    for (const Qwen4ExpFrontierRuntime::LayerGraphs & layer_graphs :
         runtime->moe) {
        for (Qwen4ExpFrontierMoeGraph * graph : layer_graphs)
            qwen4exp_frontier_moe_destroy(graph);
    }
    delete runtime;
    weights.frontier = nullptr;
}

bool qwen4exp_frontier_moe_q1(const Qwen4ExpWeights & weights, int layer,
                              const float * input, size_t input_count,
                              std::vector<float> & output,
                              std::string & error) {
    if (!weights.frontier || layer < 0 ||
        static_cast<size_t>(layer) >= weights.frontier->moe.size()) {
        error = "Qwen4Exp frontier MoE graph is unavailable";
        return false;
    }
    return qwen4exp_frontier_moe_eval(
        weights.frontier->moe[static_cast<size_t>(layer)][1], input,
        input_count, output, error);
}

bool qwen4exp_frontier_moe_batch(const Qwen4ExpWeights & weights, int layer,
                                 const float * input, size_t input_count,
                                 int n_tokens, std::vector<float> & output,
                                 std::string & error) {
    if (!weights.frontier || layer < 0 ||
        static_cast<size_t>(layer) >= weights.frontier->moe.size() || !input ||
        input_count != static_cast<size_t>(n_tokens) * 2560U) {
        error = "Qwen4Exp frontier batched MoE graph is unavailable";
        return false;
    }
    const int graph_width = qwen4exp_frontier_moe_cached_width(n_tokens);
    if (graph_width == 0) {
        error = "Qwen4Exp frontier batched MoE width is unsupported";
        return false;
    }
    Qwen4ExpFrontierRuntime::LayerGraphs & layer_graphs =
        weights.frontier->moe[static_cast<size_t>(layer)];
    Qwen4ExpFrontierMoeGraph *& graph =
        layer_graphs[static_cast<size_t>(graph_width)];
    if (!graph) {
        const Qwen4ExpLayer & model_layer =
            weights.layers[static_cast<size_t>(layer)];
        const Qwen4ExpFrontierMoeSpec spec{2560, 512, 10, 640};
        const Qwen4ExpFrontierMoeWeights graph_weights{
            model_layer.router, model_layer.experts_gate_up_tensor,
            model_layer.experts_down_tensor, model_layer.shared_gate_input,
            model_layer.shared_gate, model_layer.shared_up,
            model_layer.shared_down};
        graph = qwen4exp_frontier_moe_create_batch(
            weights.backend, spec, graph_weights, layer, graph_width, error);
        if (!graph) return false;
    }
    if (graph_width == n_tokens)
        return qwen4exp_frontier_moe_eval(
            graph, input, input_count, output, error);
    std::vector<float> padded(
        static_cast<size_t>(graph_width) * 2560U, 0.0f);
    std::copy_n(input, input_count, padded.data());
    if (!qwen4exp_frontier_moe_eval(
            graph, padded.data(), padded.size(), output, error)) return false;
    output.resize(input_count);
    return true;
}

bool qwen4exp_frontier_mtp_create(Qwen4ExpMtpWeights & weights,
                                  std::string & error) {
    qwen4exp_frontier_mtp_destroy(weights);
    if (!env_enabled("EMBER_QWEN_FRONTIER_MOE", true)) {
        error = "Qwen4Exp MTP requires the GPU frontier MoE graph";
        return false;
    }
    if (!weights.backend) {
        error = "invalid Qwen4Exp MTP weights for frontier initialization";
        return false;
    }
    const Qwen4ExpLayer & layer = weights.layer;
    const Qwen4ExpFrontierMoeSpec spec{2560, 512, 10, 640};
    const Qwen4ExpFrontierMoeWeights graph_weights{
        layer.router, layer.experts_gate_up_tensor,
        layer.experts_down_tensor, layer.shared_gate_input,
        layer.shared_gate, layer.shared_up, layer.shared_down};
    weights.frontier_moe = qwen4exp_frontier_moe_create(
        weights.backend, spec, graph_weights, 48, error);
    if (!weights.frontier_moe) return false;
    std::fprintf(stderr,
                 "[qwen-frontier] event=ready component=mtp_moe graphs=1 "
                 "tokens_per_graph=1 arena_bytes=%zu weight_copies=0 "
                 "graph_replay=off\n",
                 weights.frontier_moe->arena_bytes);
    return true;
}

void qwen4exp_frontier_mtp_destroy(Qwen4ExpMtpWeights & weights) {
    qwen4exp_frontier_moe_destroy(weights.frontier_moe);
    weights.frontier_moe = nullptr;
}

bool qwen4exp_frontier_mtp_moe_q1(const Qwen4ExpMtpWeights & weights,
                                  const float * input, size_t input_count,
                                  std::vector<float> & output,
                                  std::string & error) {
    if (!weights.frontier_moe) {
        error = "Qwen4Exp MTP frontier MoE graph is unavailable";
        return false;
    }
    return qwen4exp_frontier_moe_eval(weights.frontier_moe, input,
                                      input_count, output, error);
}

} // namespace dflash::common
