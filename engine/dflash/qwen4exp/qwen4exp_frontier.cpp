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
#include <unordered_map>

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

bool rocmi4_dispatch_evidence_enabled() {
    static const bool enabled =
        env_enabled("DFLASH_ROCMI4_W4A8_DISPATCH_EVIDENCE", false);
    return enabled;
}

void log_rocmi4_dispatch_control(const char * control_id, const char * op,
                                 int logical_q, const char * phase,
                                 const ggml_tensor * target_weight) {
    if (!rocmi4_dispatch_evidence_enabled()) return;
    std::fprintf(stderr,
                 "[rocmi4-w4a8-dispatch] event=control control_id=%s "
                 "op=%s logical_q=%d target_weight=%s phase=%s\n",
                 control_id, op, logical_q,
                 target_weight ? target_weight->name : "none", phase);
}

void log_rocmi4_dispatch_post_compute(const char * control_id,
                                      const char * op, int logical_q,
                                      int physical_q,
                                      const ggml_tensor * target_weight) {
    if (!rocmi4_dispatch_evidence_enabled()) return;
    std::fprintf(stderr,
                 "[rocmi4-w4a8-dispatch] event=post_compute control_id=%s "
                 "op=%s logical_q=%d physical_q=%d target_weight=%s "
                 "execution=completed\n",
                 control_id, op, logical_q, physical_q,
                 target_weight ? target_weight->name : "none");
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

struct Qwen4ExpFrontierDenseGraph {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    int input_count = 0;
    int n_tokens = 0;
    std::vector<float> padded_input;
};

struct Qwen4ExpFrontierHcGraph {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * mixed = nullptr;
    ggml_tensor * injection = nullptr;
    ggml_tensor * projected = nullptr;
    Qwen4ExpFrontierHcSpec spec{};
    int n_tokens = 0;
    size_t arena_bytes = 0;
    std::vector<float> padded_input;
    char profile_label[96]{};
};

struct Qwen4ExpFrontierGdnGraph {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * conv_history = nullptr;
    ggml_tensor * recurrent_state = nullptr;
    ggml_tensor * qkv = nullptr;
    ggml_tensor * gdn = nullptr;
    ggml_tensor * output = nullptr;
    Qwen4ExpFrontierGdnSpec spec{};
    int n_tokens = 1;
    int layer = -1;
    size_t arena_bytes = 0;
    std::vector<float> conv_window;
    char profile_label[64]{};
};

struct Qwen4ExpFrontierQsaSubgraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    size_t arena_bytes = 0;
};

struct Qwen4ExpFrontierQsaAttentionGraph : Qwen4ExpFrontierQsaSubgraph {
    ggml_tensor * query = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * key = nullptr;
    ggml_tensor * value = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * output = nullptr;
    int width = 0;
    std::vector<float> padded_key;
    std::vector<float> padded_value;
    std::vector<ggml_fp16_t> padded_mask;
    char profile_label[64]{};
};

struct Qwen4ExpFrontierQsaGraph {
    ggml_backend_t backend = nullptr;
    Qwen4ExpFrontierQsaSpec spec{};
    Qwen4ExpFrontierQsaWeights weights{};
    int layer = -1;
    Qwen4ExpFrontierQsaSubgraph projection;
    ggml_tensor * projection_input = nullptr;
    ggml_tensor * projected_query_gate = nullptr;
    ggml_tensor * projected_key = nullptr;
    ggml_tensor * projected_value = nullptr;
    ggml_tensor * projected_index_query = nullptr;
    ggml_tensor * projected_index_key = nullptr;
    Qwen4ExpFrontierQsaSubgraph rotation;
    ggml_tensor * rotation_query_key = nullptr;
    ggml_tensor * rotation_value = nullptr;
    ggml_tensor * rotated_query_key = nullptr;
    ggml_tensor * rotated_value = nullptr;
    std::unordered_map<int, Qwen4ExpFrontierQsaAttentionGraph *> attention;
    char projection_profile_label[64]{};
    char rotation_profile_label[64]{};
};

struct Qwen4ExpFrontierDenseKey {
    ggml_tensor * weight = nullptr;
    int n_tokens = 0;

    bool operator==(const Qwen4ExpFrontierDenseKey & other) const {
        return weight == other.weight && n_tokens == other.n_tokens;
    }
};

struct Qwen4ExpFrontierDenseKeyHash {
    size_t operator()(const Qwen4ExpFrontierDenseKey & key) const {
        const size_t pointer_hash = std::hash<ggml_tensor *>{}(key.weight);
        const size_t width_hash = std::hash<int>{}(key.n_tokens);
        return pointer_hash ^ (width_hash + 0x9e3779b9U +
                               (pointer_hash << 6U) +
                               (pointer_hash >> 2U));
    }
};

struct Qwen4ExpFrontierHcKey {
    ggml_tensor * norm = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * inject = nullptr;
    ggml_tensor * projection = nullptr;
    int n_tokens = 0;

    bool operator==(const Qwen4ExpFrontierHcKey & other) const {
        return norm == other.norm && down == other.down && up == other.up &&
               inject == other.inject && projection == other.projection &&
               n_tokens == other.n_tokens;
    }
};

struct Qwen4ExpFrontierHcKeyHash {
    size_t operator()(const Qwen4ExpFrontierHcKey & key) const {
        size_t value = std::hash<ggml_tensor *>{}(key.norm);
        const auto combine = [&value](size_t item) {
            value ^= item + 0x9e3779b9U + (value << 6U) + (value >> 2U);
        };
        combine(std::hash<ggml_tensor *>{}(key.down));
        combine(std::hash<ggml_tensor *>{}(key.up));
        combine(std::hash<ggml_tensor *>{}(key.inject));
        combine(std::hash<ggml_tensor *>{}(key.projection));
        combine(std::hash<int>{}(key.n_tokens));
        return value;
    }
};

struct Qwen4ExpFrontierDenseCache {
    std::unordered_map<Qwen4ExpFrontierDenseKey,
                       Qwen4ExpFrontierDenseGraph *,
                       Qwen4ExpFrontierDenseKeyHash> graphs;
    std::unordered_map<Qwen4ExpFrontierHcKey,
                       Qwen4ExpFrontierHcGraph *,
                       Qwen4ExpFrontierHcKeyHash> hc_graphs;
    std::unordered_map<ggml_tensor *, std::vector<float>> static_f32;
};

struct Qwen4ExpFrontierRuntime {
    using LayerGraphs = std::array<Qwen4ExpFrontierMoeGraph *,
        static_cast<size_t>(kQwen4ExpFrontierMoeMaxBatch + 1)>;
    std::vector<LayerGraphs> moe;
    using GdnLayerGraphs = std::array<Qwen4ExpFrontierGdnGraph *,
        static_cast<size_t>(kQwen4ExpFrontierMoeMaxBatch + 1)>;
    std::vector<GdnLayerGraphs> gdn;
    std::vector<Qwen4ExpFrontierQsaGraph *> qsa;
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

int qwen4exp_frontier_dense_cached_width(int n_tokens) {
    return qwen4exp_frontier_moe_cached_width(n_tokens);
}

int qwen4exp_frontier_qsa_cached_width(int selected_tokens) {
    // Decode never exposes more than the released 2048-token budget plus the
    // incomplete four-token causal tail. Power-of-four buckets keep short
    // contexts from paying for a max-width arena; 2051 is the exact maximum.
    if (selected_tokens <= 0 || selected_tokens > 2051) return 0;
    if (selected_tokens <= 16) return 16;
    if (selected_tokens <= 64) return 64;
    if (selected_tokens <= 256) return 256;
    if (selected_tokens <= 1024) return 1024;
    if (selected_tokens <= 2048) return 2048;
    return 2051;
}

namespace {

void dense_graph_destroy(Qwen4ExpFrontierDenseGraph * graph) {
    if (!graph) return;
    if (graph->allocator) ggml_gallocr_free(graph->allocator);
    if (graph->ctx) ggml_free(graph->ctx);
    delete graph;
}

Qwen4ExpFrontierDenseGraph * dense_graph_create(
        ggml_backend_t backend, ggml_tensor * weight, int input_count,
        int n_tokens, std::string & error) {
    if (!backend || !weight || !weight->buffer || input_count <= 0 ||
        n_tokens <= 0 || weight->ne[0] != input_count ||
        ggml_n_dims(weight) != 2) {
        error = "invalid Qwen4Exp persistent dense graph shape";
        return nullptr;
    }

    std::unique_ptr<Qwen4ExpFrontierDenseGraph> result(
        new Qwen4ExpFrontierDenseGraph());
    result->backend = backend;
    result->weight = weight;
    result->input_count = input_count;
    result->n_tokens = n_tokens;
    result->padded_input.assign(static_cast<size_t>(input_count) *
                                    static_cast<size_t>(n_tokens),
                                0.0f);

    ggml_init_params params{};
    // A dense frontier has only its input, mul_mat output, and a 16-node
    // graph. Keep metadata storage bounded per `(weight,width)` entry; tensor
    // payloads live in the gallocr backend arena below.
    params.mem_size = 64U * 1024U;
    params.no_alloc = true;
    result->ctx = ggml_init(params);
    if (!result->ctx) {
        error = "Qwen4Exp persistent dense context allocation failed";
        return nullptr;
    }
    result->input = ggml_new_tensor_2d(
        result->ctx, GGML_TYPE_F32, input_count, n_tokens);
    ggml_set_input(result->input);
    result->output = ggml_mul_mat(result->ctx, weight, result->input);
    ggml_set_output(result->output);
    result->graph = ggml_new_graph_custom(result->ctx, 16, false);
    ggml_build_forward_expand(result->graph, result->output);
    result->allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!result->allocator ||
        !ggml_gallocr_alloc_graph(result->allocator, result->graph)) {
        error = "Qwen4Exp persistent dense graph allocation failed";
        dense_graph_destroy(result.release());
        return nullptr;
    }
    return result.release();
}

void hc_graph_destroy(Qwen4ExpFrontierHcGraph * graph) {
    if (!graph) return;
    if (graph->allocator) ggml_gallocr_free(graph->allocator);
    if (graph->ctx) ggml_free(graph->ctx);
    delete graph;
}

Qwen4ExpFrontierHcGraph * hc_graph_create(
        ggml_backend_t backend, const Qwen4ExpFrontierHcSpec & spec,
        ggml_tensor * norm, ggml_tensor * down, ggml_tensor * up,
        ggml_tensor * inject, ggml_tensor * projection, int n_tokens,
        std::string & error) {
    const int64_t hc_dim = static_cast<int64_t>(spec.stream_width) *
                           static_cast<int64_t>(spec.streams);
    const bool inject_valid = !inject ||
        (inject->buffer && ggml_n_dims(inject) == 2 &&
         inject->ne[0] == hc_dim && inject->ne[1] == spec.streams);
    const bool projection_valid = !projection ||
        (projection->buffer && ggml_n_dims(projection) == 2 &&
         projection->ne[0] == spec.stream_width && projection->ne[1] > 0);
    if (!backend || spec.stream_width <= 0 || spec.streams <= 0 ||
        !std::isfinite(spec.epsilon) || spec.epsilon <= 0.0f ||
        n_tokens <= 0 || hc_dim <= 0 || !norm || !norm->buffer ||
        ggml_nelements(norm) != hc_dim || !down || !down->buffer ||
        ggml_n_dims(down) != 2 || down->ne[0] != hc_dim ||
        down->ne[1] <= 0 || !up || !up->buffer ||
        ggml_n_dims(up) != 2 || up->ne[0] != down->ne[1] ||
        up->ne[1] != hc_dim || !inject_valid || !projection_valid) {
        error = "invalid Qwen4Exp persistent HC graph shape";
        return nullptr;
    }
    std::unique_ptr<Qwen4ExpFrontierHcGraph> result(
        new Qwen4ExpFrontierHcGraph());
    result->backend = backend;
    result->spec = spec;
    result->n_tokens = n_tokens;
    result->padded_input.assign(
        static_cast<size_t>(hc_dim) * static_cast<size_t>(n_tokens), 0.0f);
    std::snprintf(result->profile_label, sizeof(result->profile_label),
                  "qwen4exp/hc/%s/q%d", down->name, n_tokens);

    ggml_init_params params{};
    params.mem_size = 256U * 1024U;
    params.no_alloc = true;
    result->ctx = ggml_init(params);
    if (!result->ctx) {
        error = "Qwen4Exp persistent HC context allocation failed";
        return nullptr;
    }
    ggml_context * ctx = result->ctx;
    result->input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(result->input);
    ggml_tensor * streams = ggml_reshape_3d(
        ctx, result->input, spec.stream_width, spec.streams, n_tokens);
    ggml_tensor * normalized = ggml_rms_norm(ctx, streams, spec.epsilon);
    normalized = ggml_reshape_2d(ctx, normalized, hc_dim, n_tokens);
    ggml_tensor * norm_f32 = norm->type == GGML_TYPE_F32
        ? norm : ggml_cast(ctx, norm, GGML_TYPE_F32);
    normalized = ggml_mul(ctx, normalized, norm_f32);
    ggml_tensor * low = ggml_mul_mat(ctx, down, normalized);
    low = ggml_silu(ctx, ggml_scale(
        ctx, low, 1.0f / static_cast<float>(spec.streams)));
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, up, low));
    ggml_tensor * weighted = ggml_mul(ctx, normalized, gate);
    weighted = ggml_reshape_3d(
        ctx, weighted, spec.stream_width, spec.streams, n_tokens);
    weighted = ggml_cont(ctx, ggml_permute(ctx, weighted, 1, 0, 2, 3));
    result->mixed = ggml_scale(
        ctx, ggml_sum_rows(ctx, weighted),
        1.0f / static_cast<float>(spec.streams));
    result->mixed = ggml_reshape_2d(
        ctx, result->mixed, spec.stream_width, n_tokens);
    if (inject) result->injection = ggml_mul_mat(ctx, inject, normalized);
    if (projection)
        result->projected = ggml_mul_mat(ctx, projection, result->mixed);
    ggml_set_output(result->projected ? result->projected : result->mixed);
    if (result->injection) ggml_set_output(result->injection);
    result->graph = ggml_new_graph_custom(ctx, 96, false);
    ggml_build_forward_expand(
        result->graph, result->projected ? result->projected : result->mixed);
    if (result->injection)
        ggml_build_forward_expand(result->graph, result->injection);
    result->allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!result->allocator ||
        !ggml_gallocr_alloc_graph(result->allocator, result->graph)) {
        error = "Qwen4Exp persistent HC graph allocation failed";
        hc_graph_destroy(result.release());
        return nullptr;
    }
    result->arena_bytes =
        ggml_gallocr_get_buffer_size(result->allocator, 0);
    return result.release();
}

bool download_tensor_f32(ggml_tensor * tensor, std::vector<float> & decoded,
                         std::string & error) {
    if (!tensor || !tensor->buffer || tensor->ne[0] <= 0) {
        error = "Qwen4Exp F32 download received an unbound tensor";
        return false;
    }
    const int64_t elements = ggml_nelements(tensor);
    if (elements <= 0 || static_cast<uint64_t>(elements) >
                             std::numeric_limits<size_t>::max()) {
        error = "Qwen4Exp tensor element count overflow";
        return false;
    }
    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (!traits || (!traits->to_float && tensor->type != GGML_TYPE_F32)) {
        error = "Qwen4Exp tensor cannot be decoded as F32";
        return false;
    }
    decoded.resize(static_cast<size_t>(elements));
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(decoded.data(), raw.data(),
                    decoded.size() * sizeof(float));
        return true;
    }
    const int64_t row = tensor->ne[0];
    const size_t row_bytes = ggml_row_size(tensor->type, row);
    const int64_t rows = elements / row;
    for (int64_t index = 0; index < rows; ++index) {
        traits->to_float(raw.data() + static_cast<size_t>(index) * row_bytes,
                         decoded.data() + static_cast<size_t>(index * row),
                         row);
    }
    return true;
}

} // namespace

Qwen4ExpFrontierDenseCache * qwen4exp_frontier_dense_cache_create() {
    return new Qwen4ExpFrontierDenseCache();
}

void qwen4exp_frontier_dense_cache_destroy(
        Qwen4ExpFrontierDenseCache * cache) {
    if (!cache) return;
    for (const auto & entry : cache->graphs)
        dense_graph_destroy(entry.second);
    for (const auto & entry : cache->hc_graphs)
        hc_graph_destroy(entry.second);
    delete cache;
}

bool qwen4exp_frontier_dense_eval(
        Qwen4ExpFrontierDenseCache * cache, ggml_backend_t backend,
        ggml_tensor * weight, const float * input, int input_count,
        int n_tokens, std::vector<float> & output, std::string & error) {
    const int graph_width = qwen4exp_frontier_dense_cached_width(n_tokens);
    if (!cache || !backend || !weight || !input || input_count <= 0 ||
        graph_width == 0 || weight->ne[0] != input_count ||
        weight->ne[1] <= 0 || ggml_n_dims(weight) != 2) {
        error = "invalid Qwen4Exp persistent dense evaluation";
        return false;
    }
    const Qwen4ExpFrontierDenseKey key{weight, graph_width};
    Qwen4ExpFrontierDenseGraph *& graph = cache->graphs[key];
    if (!graph) {
        graph = dense_graph_create(backend, weight, input_count, graph_width,
                                   error);
        if (!graph) {
            cache->graphs.erase(key);
            return false;
        }
    }
    if (graph->backend != backend || graph->input_count != input_count) {
        error = "Qwen4Exp persistent dense cache owner mismatch";
        return false;
    }
    const size_t real_input_values = static_cast<size_t>(input_count) *
                                     static_cast<size_t>(n_tokens);
    std::copy_n(input, real_input_values, graph->padded_input.data());
    std::fill_n(graph->padded_input.data() + real_input_values,
                graph->padded_input.size() - real_input_values, 0.0f);
    ggml_backend_tensor_set(graph->input, graph->padded_input.data(), 0,
                            graph->padded_input.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph->graph) !=
        GGML_STATUS_SUCCESS) {
        error = "Qwen4Exp persistent dense graph execution failed";
        return false;
    }
    if (rocmi4_dispatch_evidence_enabled() &&
        weight->type == GGML_TYPE_Q4_0_ROCMI4) {
        std::fprintf(stderr,
                     "[rocmi4-w4a8-dispatch] event=logical_scope op=dense "
                     "logical_q=%d physical_q=%d type=%s execution=completed "
                     "weight=%s\n",
                     n_tokens, graph_width, ggml_type_name(weight->type),
                     weight->name);
    }
    const size_t full_output_values = static_cast<size_t>(weight->ne[1]) *
                                      static_cast<size_t>(graph_width);
    output.resize(full_output_values);
    ggml_backend_tensor_get(graph->output, output.data(), 0,
                            full_output_values * sizeof(float));
    output.resize(static_cast<size_t>(weight->ne[1]) *
                  static_cast<size_t>(n_tokens));
    return true;
}

bool qwen4exp_frontier_dense_eval_rows(
        Qwen4ExpFrontierDenseCache * cache, ggml_backend_t backend,
        ggml_tensor * weight, const float * input, int input_count,
        int n_tokens, std::vector<float> & output, std::string & error) {
    if (!input || input_count <= 0 || n_tokens <= 0 ||
        static_cast<size_t>(n_tokens) >
            std::numeric_limits<size_t>::max() /
                static_cast<size_t>(input_count)) {
        error = "invalid Qwen4Exp persistent dense row evaluation";
        return false;
    }
    std::vector<float> assembled;
    for (int offset = 0; offset < n_tokens;
         offset += kQwen4ExpFrontierMoeMaxBatch) {
        const int rows = std::min(kQwen4ExpFrontierMoeMaxBatch,
                                  n_tokens - offset);
        std::vector<float> chunk;
        const float * chunk_input =
            input + static_cast<size_t>(offset) *
                        static_cast<size_t>(input_count);
        if (!qwen4exp_frontier_dense_eval(
                cache, backend, weight, chunk_input, input_count, rows,
                chunk, error)) return false;
        assembled.insert(assembled.end(), chunk.begin(), chunk.end());
    }
    output = std::move(assembled);
    return true;
}

namespace {
bool hc_eval(
        Qwen4ExpFrontierDenseCache * cache, ggml_backend_t backend,
        const Qwen4ExpFrontierHcSpec & spec, ggml_tensor * norm,
        ggml_tensor * down, ggml_tensor * up, ggml_tensor * inject,
        ggml_tensor * projection,
        const float * input, size_t input_count, int n_tokens,
        std::vector<float> * mixed, std::vector<float> * injection,
        std::vector<float> * projected,
        std::string & error) {
    const int graph_width = qwen4exp_frontier_dense_cached_width(n_tokens);
    const int64_t hc_dim = static_cast<int64_t>(spec.stream_width) *
                           static_cast<int64_t>(spec.streams);
    if (!cache || !backend || !input || n_tokens <= 0 || graph_width == 0 ||
        hc_dim <= 0 || static_cast<uint64_t>(hc_dim) >
                           std::numeric_limits<size_t>::max() ||
        input_count != static_cast<size_t>(hc_dim) *
                           static_cast<size_t>(n_tokens) ||
        (inject == nullptr) != (injection == nullptr) ||
        (projection == nullptr) != (projected == nullptr) ||
        (!mixed && !projected)) {
        error = "invalid Qwen4Exp persistent HC evaluation";
        return false;
    }
    const Qwen4ExpFrontierHcKey key{
        norm, down, up, inject, projection, graph_width};
    Qwen4ExpFrontierHcGraph *& graph = cache->hc_graphs[key];
    if (!graph) {
        graph = hc_graph_create(backend, spec, norm, down, up, inject,
                                projection, graph_width, error);
        if (!graph) {
            cache->hc_graphs.erase(key);
            return false;
        }
        std::fprintf(stderr,
                     "[qwen-frontier] event=graph_ready component=hc "
                     "weight=%s projection=%s logical_q=%d arena_width=%d "
                     "arena_bytes=%zu graph_replay=off\n",
                     down ? down->name : "none",
                     projection ? projection->name : "none", n_tokens,
                     graph_width,
                     graph->arena_bytes);
    }
    if (graph->backend != backend || graph->spec.stream_width !=
            spec.stream_width || graph->spec.streams != spec.streams ||
        graph->spec.epsilon != spec.epsilon) {
        error = "Qwen4Exp persistent HC cache owner mismatch";
        return false;
    }
    std::copy_n(input, input_count, graph->padded_input.data());
    std::fill(graph->padded_input.begin() +
                  static_cast<std::ptrdiff_t>(input_count),
              graph->padded_input.end(), 0.0f);
    ggml_backend_tensor_set(graph->input, graph->padded_input.data(), 0,
                            graph->padded_input.size() * sizeof(float));
    const ProfileRange range(graph->profile_label);
    if (ggml_backend_graph_compute(backend, graph->graph) !=
        GGML_STATUS_SUCCESS) {
        error = "Qwen4Exp persistent HC graph execution failed";
        return false;
    }
    if (mixed) {
        const size_t mixed_values = static_cast<size_t>(spec.stream_width) *
                                    static_cast<size_t>(graph_width);
        mixed->resize(mixed_values);
        ggml_backend_tensor_get(graph->mixed, mixed->data(), 0,
                                mixed_values * sizeof(float));
        mixed->resize(static_cast<size_t>(spec.stream_width) *
                      static_cast<size_t>(n_tokens));
    }
    if (injection) {
        const size_t injection_values = static_cast<size_t>(spec.streams) *
                                        static_cast<size_t>(graph_width);
        injection->resize(injection_values);
        ggml_backend_tensor_get(graph->injection, injection->data(), 0,
                                injection_values * sizeof(float));
        injection->resize(static_cast<size_t>(spec.streams) *
                          static_cast<size_t>(n_tokens));
    }
    if (projected) {
        if (!projection || projection->ne[1] <= 0 ||
            static_cast<uint64_t>(projection->ne[1]) >
                std::numeric_limits<size_t>::max() /
                    static_cast<size_t>(graph_width)) {
            error = "invalid Qwen4Exp persistent HC output projection";
            return false;
        }
        const size_t output_width = static_cast<size_t>(projection->ne[1]);
        projected->resize(output_width * static_cast<size_t>(graph_width));
        ggml_backend_tensor_get(graph->projected, projected->data(), 0,
                                projected->size() * sizeof(float));
        projected->resize(output_width * static_cast<size_t>(n_tokens));
    }
    return true;
}
} // namespace

bool qwen4exp_frontier_hc_eval(
        Qwen4ExpFrontierDenseCache * cache, ggml_backend_t backend,
        const Qwen4ExpFrontierHcSpec & spec, ggml_tensor * norm,
        ggml_tensor * down, ggml_tensor * up, ggml_tensor * inject,
        const float * input, size_t input_count, int n_tokens,
        std::vector<float> & mixed, std::vector<float> * injection,
        std::string & error) {
    return hc_eval(cache, backend, spec, norm, down, up, inject, nullptr,
                   input, input_count, n_tokens, &mixed, injection, nullptr,
                   error);
}

bool qwen4exp_frontier_hc_output_eval(
        Qwen4ExpFrontierDenseCache * cache, ggml_backend_t backend,
        const Qwen4ExpFrontierHcSpec & spec, ggml_tensor * norm,
        ggml_tensor * down, ggml_tensor * up, ggml_tensor * projection,
        const float * input, size_t input_count, int n_tokens,
        std::vector<float> & output, std::string & error) {
    return hc_eval(cache, backend, spec, norm, down, up, nullptr, projection,
                   input, input_count, n_tokens, nullptr, nullptr, &output,
                   error);
}

size_t qwen4exp_frontier_hc_graph_count(
        const Qwen4ExpFrontierDenseCache * cache) {
    return cache ? cache->hc_graphs.size() : 0;
}

bool qwen4exp_frontier_static_f32(
        Qwen4ExpFrontierDenseCache * cache, ggml_tensor * tensor,
        std::vector<float> & output, std::string & error) {
    if (!cache || !tensor || !tensor->buffer || tensor->ne[0] <= 0) {
        error = "Qwen4Exp static tensor cache received an unbound tensor";
        return false;
    }
    const auto found = cache->static_f32.find(tensor);
    if (found != cache->static_f32.end()) {
        output = found->second;
        return true;
    }
    std::vector<float> decoded;
    if (!download_tensor_f32(tensor, decoded, error)) return false;
    output = decoded;
    cache->static_f32.emplace(tensor, std::move(decoded));
    return true;
}

size_t qwen4exp_frontier_dense_graph_count(
        const Qwen4ExpFrontierDenseCache * cache) {
    return cache ? cache->graphs.size() : 0U;
}

size_t qwen4exp_frontier_static_f32_count(
        const Qwen4ExpFrontierDenseCache * cache) {
    return cache ? cache->static_f32.size() : 0U;
}

namespace {

bool gdn_spec_valid(const Qwen4ExpFrontierGdnSpec & spec) {
    return spec.n_embd > 0 && spec.n_heads > 0 && spec.n_key_heads > 0 &&
           spec.n_heads % spec.n_key_heads == 0 && spec.head_dim > 0 &&
           spec.conv_width >= 2 && spec.epsilon >= 0.0f;
}

int64_t gdn_conv_channels(const Qwen4ExpFrontierGdnSpec & spec) {
    return static_cast<int64_t>(2 * spec.n_key_heads + spec.n_heads) *
           spec.head_dim;
}

bool gdn_weight_shape(const ggml_tensor * tensor, int64_t ne0,
                      int64_t ne1) {
    return tensor && tensor->buffer && ggml_n_dims(tensor) == 2 &&
           tensor->ne[0] == ne0 && tensor->ne[1] == ne1;
}

bool gdn_vector_shape(const ggml_tensor * tensor, int64_t elements) {
    return tensor && tensor->buffer && ggml_nelements(tensor) == elements;
}

ggml_tensor * exact_l2_norm(ggml_context * ctx, ggml_tensor * tensor,
                            float epsilon) {
    // Q/K arrive as strided views into the fused QKV convolution output.
    // CUDA/HIP unary kernels require a contiguous source even though the CPU
    // backend accepts the view strides, so materialize the logical tensor
    // before the first unary operation.  Keep the normalized result in that
    // same logical layout for all following broadcast operations.
    tensor = ggml_cont(ctx, tensor);
    ggml_tensor * sum = ggml_sum_rows(ctx, ggml_sqr(ctx, tensor));
    ggml_tensor * denominator = ggml_sqrt(
        ctx, ggml_scale_bias(ctx, sum, 1.0f, epsilon));
    return ggml_div(ctx, tensor, ggml_repeat(ctx, denominator, tensor));
}

void set_gdn_name(ggml_tensor * tensor, int layer, const char * suffix) {
    if (!tensor) return;
    char name[GGML_MAX_NAME];
    if (layer >= 0) {
        std::snprintf(name, sizeof(name), "qwen_gdn_l%02d_%s", layer, suffix);
    } else {
        std::snprintf(name, sizeof(name), "qwen_gdn_test_%s", suffix);
    }
    ggml_set_name(tensor, name);
}

} // namespace

Qwen4ExpFrontierGdnGraph * qwen4exp_frontier_gdn_create_q1(
        ggml_backend_t backend, const Qwen4ExpFrontierGdnSpec & spec,
        const Qwen4ExpFrontierGdnWeights & weights, int layer,
        std::string & error) {
    return qwen4exp_frontier_gdn_create_batch(
        backend, spec, weights, layer, 1, error);
}

Qwen4ExpFrontierGdnGraph * qwen4exp_frontier_gdn_create_batch(
        ggml_backend_t backend, const Qwen4ExpFrontierGdnSpec & spec,
        const Qwen4ExpFrontierGdnWeights & weights, int layer, int n_tokens,
        std::string & error) {
    error.clear();
    const int64_t conv_channels = gdn_conv_channels(spec);
    const int64_t recurrent_values = static_cast<int64_t>(spec.n_heads) *
                                     spec.head_dim * spec.head_dim;
    const int64_t core_values = static_cast<int64_t>(spec.n_heads) *
                                spec.head_dim;
    if (!backend || !gdn_spec_valid(spec) || n_tokens <= 0 ||
        n_tokens > kQwen4ExpFrontierMoeMaxBatch || conv_channels <= 0 ||
        recurrent_values <= 0 || core_values <= 0 ||
        !gdn_weight_shape(weights.qkv, spec.n_embd, conv_channels) ||
        !gdn_weight_shape(weights.gate, spec.n_embd, core_values) ||
        !gdn_weight_shape(weights.alpha, spec.n_embd, spec.n_heads) ||
        !gdn_weight_shape(weights.beta, spec.n_embd, spec.n_heads) ||
        !gdn_weight_shape(weights.conv, spec.conv_width, conv_channels) ||
        !gdn_vector_shape(weights.a, spec.n_heads) ||
        !gdn_vector_shape(weights.dt, spec.n_heads) ||
        !gdn_vector_shape(weights.norm, spec.head_dim) ||
        !gdn_weight_shape(weights.output, core_values, spec.n_embd)) {
        error = "invalid Qwen4Exp persistent GDN tensor contract";
        return nullptr;
    }

    std::vector<float> conv_f32, a_f32, dt_f32, norm_f32;
    if (!download_tensor_f32(weights.conv, conv_f32, error) ||
        !download_tensor_f32(weights.a, a_f32, error) ||
        !download_tensor_f32(weights.dt, dt_f32, error) ||
        !download_tensor_f32(weights.norm, norm_f32, error)) {
        return nullptr;
    }

    std::unique_ptr<Qwen4ExpFrontierGdnGraph> result(
        new Qwen4ExpFrontierGdnGraph());
    result->backend = backend;
    result->spec = spec;
    result->n_tokens = n_tokens;
    result->layer = layer;
    result->conv_window.resize(
        static_cast<size_t>(spec.conv_width - 1) *
        static_cast<size_t>(conv_channels));
    std::snprintf(result->profile_label, sizeof(result->profile_label),
                  "qwen4exp/gdn/layer_%02d/q%d", layer, n_tokens);

    ggml_init_params params{};
    params.mem_size = 1024U * 1024U;
    params.no_alloc = true;
    result->ctx = ggml_init(params);
    if (!result->ctx) {
        error = "Qwen4Exp persistent GDN context allocation failed";
        return nullptr;
    }
    ggml_context * ctx = result->ctx;
    result->input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, spec.n_embd, n_tokens);
    result->conv_history = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, spec.conv_width - 1, conv_channels, 1);
    result->recurrent_state = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, spec.head_dim, spec.head_dim, spec.n_heads);
    ggml_set_input(result->input);
    ggml_set_input(result->conv_history);
    ggml_set_input(result->recurrent_state);
    set_gdn_name(result->input, layer, "input");
    set_gdn_name(result->conv_history, layer, "conv_state");
    set_gdn_name(result->recurrent_state, layer, "recurrent_state");

    ggml_tensor * conv_weight = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, spec.conv_width, conv_channels);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, spec.n_heads);
    ggml_tensor * dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, spec.n_heads);
    ggml_tensor * norm = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, spec.head_dim);
    ggml_set_input(conv_weight);
    ggml_set_input(a);
    ggml_set_input(dt);
    ggml_set_input(norm);
    // These immutable F32 mirrors live in the graph arena rather than the
    // model's original BF16/quantized buffer. Mark them as outputs as well as
    // inputs so gallocr cannot recycle their storage after first use; the same
    // payload must remain valid across every persistent replay.
    ggml_set_output(conv_weight);
    ggml_set_output(a);
    ggml_set_output(dt);
    ggml_set_output(norm);

    result->qkv = ggml_mul_mat(ctx, weights.qkv, result->input);
    ggml_tensor * gate = ggml_mul_mat(ctx, weights.gate, result->input);
    ggml_tensor * alpha = ggml_mul_mat(ctx, weights.alpha, result->input);
    ggml_tensor * beta = ggml_mul_mat(ctx, weights.beta, result->input);
    set_gdn_name(result->qkv, layer, "qkv");

    ggml_tensor * current = ggml_cont(
        ctx, ggml_transpose(ctx, result->qkv));
    current = ggml_reshape_3d(
        ctx, current, n_tokens, conv_channels, 1);
    ggml_tensor * conv_input = ggml_concat(
        ctx, result->conv_history, current, 0);
    ggml_tensor * convolved = ggml_silu(
        ctx, ggml_ssm_conv(ctx, conv_input, conv_weight));
    set_gdn_name(convolved, layer, "conv_silu");

    const size_t element_bytes = sizeof(float);
    const size_t key_values = static_cast<size_t>(spec.n_key_heads) *
                              static_cast<size_t>(spec.head_dim);
    ggml_tensor * q = ggml_view_4d(
        ctx, convolved, spec.head_dim, spec.n_key_heads, n_tokens, 1,
        static_cast<size_t>(spec.head_dim) * element_bytes,
        static_cast<size_t>(conv_channels) * element_bytes,
        static_cast<size_t>(conv_channels) *
            static_cast<size_t>(n_tokens) * element_bytes, 0);
    ggml_tensor * k = ggml_view_4d(
        ctx, convolved, spec.head_dim, spec.n_key_heads, n_tokens, 1,
        static_cast<size_t>(spec.head_dim) * element_bytes,
        static_cast<size_t>(conv_channels) * element_bytes,
        static_cast<size_t>(conv_channels) *
            static_cast<size_t>(n_tokens) * element_bytes,
        key_values * element_bytes);
    ggml_tensor * v = ggml_view_4d(
        ctx, convolved, spec.head_dim, spec.n_heads, n_tokens, 1,
        static_cast<size_t>(spec.head_dim) * element_bytes,
        static_cast<size_t>(conv_channels) * element_bytes,
        static_cast<size_t>(conv_channels) *
            static_cast<size_t>(n_tokens) * element_bytes,
        2U * key_values * element_bytes);
    q = exact_l2_norm(ctx, q, spec.epsilon);
    k = exact_l2_norm(ctx, k, spec.epsilon);
    const int repeat = spec.n_heads / spec.n_key_heads;
    q = ggml_reshape_4d(
        ctx, q, spec.head_dim, 1, spec.n_key_heads, n_tokens);
    k = ggml_reshape_4d(
        ctx, k, spec.head_dim, 1, spec.n_key_heads, n_tokens);
    q = ggml_repeat_4d(ctx, q, spec.head_dim, repeat,
                       spec.n_key_heads, n_tokens);
    k = ggml_repeat_4d(ctx, k, spec.head_dim, repeat,
                       spec.n_key_heads, n_tokens);
    q = ggml_reshape_4d(
        ctx, q, spec.head_dim, spec.n_heads, n_tokens, 1);
    k = ggml_reshape_4d(
        ctx, k, spec.head_dim, spec.n_heads, n_tokens, 1);

    ggml_tensor * decay = ggml_mul(
        ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, dt)), a);
    decay = ggml_reshape_4d(
        ctx, decay, 1, spec.n_heads, n_tokens, 1);
    beta = ggml_reshape_4d(
        ctx, ggml_sigmoid(ctx, beta), 1, spec.n_heads, n_tokens, 1);
    result->gdn = ggml_gated_delta_net(
        ctx, q, k, v, decay, beta, result->recurrent_state);
    ggml_gated_delta_net_set_skip_intermediate(result->gdn, true);
    set_gdn_name(result->gdn, layer, "recurrent");

    ggml_tensor * core = ggml_view_3d(
        ctx, result->gdn, spec.head_dim, spec.n_heads, n_tokens,
        static_cast<size_t>(spec.head_dim) * element_bytes,
        static_cast<size_t>(core_values) * element_bytes, 0);
    core = ggml_rms_norm(ctx, core, spec.epsilon);
    core = ggml_mul(ctx, core, norm);
    gate = ggml_reshape_3d(
        ctx, gate, spec.head_dim, spec.n_heads, n_tokens);
    core = ggml_mul(ctx, core, ggml_sigmoid(ctx, gate));
    core = ggml_reshape_2d(ctx, core, core_values, n_tokens);
    result->output = ggml_mul_mat(ctx, weights.output, core);
    set_gdn_name(result->output, layer, "output");

    // qkv is the next causal-conv row and GDN contains the next recurrent
    // state. Keeping both live makes the host snapshot boundary explicit and
    // atomic after one graph compute.
    ggml_set_output(result->qkv);
    ggml_set_output(result->gdn);
    ggml_set_output(result->output);
    result->graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(result->graph, result->qkv);
    ggml_build_forward_expand(result->graph, result->gdn);
    ggml_build_forward_expand(result->graph, result->output);
    result->allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!result->allocator ||
        !ggml_gallocr_alloc_graph(result->allocator, result->graph)) {
        error = "Qwen4Exp persistent GDN graph allocation failed";
        qwen4exp_frontier_gdn_destroy(result.release());
        return nullptr;
    }
    result->arena_bytes =
        ggml_gallocr_get_buffer_size(result->allocator, 0);
    ggml_backend_tensor_set(conv_weight, conv_f32.data(), 0,
                            conv_f32.size() * sizeof(float));
    ggml_backend_tensor_set(a, a_f32.data(), 0,
                            a_f32.size() * sizeof(float));
    ggml_backend_tensor_set(dt, dt_f32.data(), 0,
                            dt_f32.size() * sizeof(float));
    ggml_backend_tensor_set(norm, norm_f32.data(), 0,
                            norm_f32.size() * sizeof(float));
    return result.release();
}

void qwen4exp_frontier_gdn_destroy(Qwen4ExpFrontierGdnGraph * graph) {
    if (!graph) return;
    if (graph->allocator) ggml_gallocr_free(graph->allocator);
    if (graph->ctx) ggml_free(graph->ctx);
    delete graph;
}

bool qwen4exp_frontier_gdn_sqr_inputs_contiguous(
        const Qwen4ExpFrontierGdnGraph * graph) {
    if (!graph || !graph->graph) return false;
    bool found = false;
    const int nodes = ggml_graph_n_nodes(graph->graph);
    for (int index = 0; index < nodes; ++index) {
        const ggml_tensor * node = ggml_graph_node(graph->graph, index);
        if (!node || node->op != GGML_OP_SQR) continue;
        found = true;
        if (!node->src[0] || !ggml_is_contiguous(node->src[0])) return false;
    }
    return found;
}

bool qwen4exp_frontier_gdn_eval_q1(
        Qwen4ExpFrontierGdnGraph * graph, const float * input,
        size_t input_count, const float * conv_state,
        size_t conv_state_count, const float * recurrent_state,
        size_t recurrent_state_count, std::vector<float> & output,
        std::vector<float> & next_conv_state,
        std::vector<float> & next_recurrent_state, std::string & error) {
    if (!graph || graph->n_tokens != 1) {
        error = "invalid Qwen4Exp persistent q1 GDN graph width";
        return false;
    }
    return qwen4exp_frontier_gdn_eval_batch(
        graph, input, input_count, conv_state, conv_state_count,
        recurrent_state, recurrent_state_count, output, next_conv_state,
        next_recurrent_state, error);
}

bool qwen4exp_frontier_gdn_eval_batch(
        Qwen4ExpFrontierGdnGraph * graph, const float * input,
        size_t input_count, const float * conv_state,
        size_t conv_state_count, const float * recurrent_state,
        size_t recurrent_state_count, std::vector<float> & output,
        std::vector<float> & next_conv_state,
        std::vector<float> & next_recurrent_state, std::string & error) {
    if (!graph || !graph->backend || !input || !conv_state ||
        !recurrent_state) {
        error = "invalid Qwen4Exp persistent GDN evaluation";
        return false;
    }
    const Qwen4ExpFrontierGdnSpec & spec = graph->spec;
    const size_t conv_channels = static_cast<size_t>(gdn_conv_channels(spec));
    const size_t history = static_cast<size_t>(spec.conv_width - 1);
    const size_t expected_conv = history * conv_channels;
    const size_t expected_recurrent = static_cast<size_t>(spec.n_heads) *
                                      static_cast<size_t>(spec.head_dim) *
                                      static_cast<size_t>(spec.head_dim);
    const size_t n_tokens = static_cast<size_t>(graph->n_tokens);
    if (input_count != static_cast<size_t>(spec.n_embd) * n_tokens ||
        conv_state_count != expected_conv ||
        recurrent_state_count != expected_recurrent) {
        error = "invalid Qwen4Exp persistent GDN state shape";
        return false;
    }
    for (size_t channel = 0; channel < conv_channels; ++channel) {
        for (size_t tap = 0; tap < history; ++tap) {
            graph->conv_window[channel * history + tap] =
                conv_state[tap * conv_channels + channel];
        }
    }
    ggml_backend_tensor_set(graph->input, input, 0,
                            input_count * sizeof(float));
    ggml_backend_tensor_set(graph->conv_history, graph->conv_window.data(), 0,
                            graph->conv_window.size() * sizeof(float));
    ggml_backend_tensor_set(graph->recurrent_state, recurrent_state, 0,
                            recurrent_state_count * sizeof(float));
    const ProfileRange range(graph->profile_label);
    if (ggml_backend_graph_compute(graph->backend, graph->graph) !=
        GGML_STATUS_SUCCESS) {
        error = "Qwen4Exp persistent GDN graph execution failed";
        return false;
    }

    output.resize(static_cast<size_t>(spec.n_embd) * n_tokens);
    ggml_backend_tensor_get(graph->output, output.data(), 0,
                            output.size() * sizeof(float));
    std::vector<float> qkv(conv_channels * n_tokens);
    ggml_backend_tensor_get(graph->qkv, qkv.data(), 0,
                            qkv.size() * sizeof(float));
    next_recurrent_state.resize(expected_recurrent);
    const size_t attention_values = static_cast<size_t>(spec.n_heads) *
                                    static_cast<size_t>(spec.head_dim) *
                                    n_tokens;
    ggml_backend_tensor_get(graph->gdn, next_recurrent_state.data(),
                            attention_values * sizeof(float),
                            next_recurrent_state.size() * sizeof(float));

    next_conv_state.resize(expected_conv);
    const size_t retained_history = n_tokens >= history
        ? 0U : history - n_tokens;
    if (retained_history > 0U) {
        std::copy_n(conv_state + n_tokens * conv_channels,
                    retained_history * conv_channels,
                    next_conv_state.data());
    }
    const size_t qkv_rows = std::min(history, n_tokens);
    std::copy_n(qkv.data() + (n_tokens - qkv_rows) * conv_channels,
                qkv_rows * conv_channels,
                next_conv_state.data() + retained_history * conv_channels);
    return true;
}

uint64_t qwen4exp_frontier_gdn_state_transfer_bytes_q1(
        const Qwen4ExpFrontierGdnSpec & spec) {
    return qwen4exp_frontier_gdn_state_transfer_bytes_batch(spec, 1);
}

uint64_t qwen4exp_frontier_gdn_state_transfer_bytes_batch(
        const Qwen4ExpFrontierGdnSpec & spec, int n_tokens) {
    if (!gdn_spec_valid(spec) || n_tokens <= 0 ||
        n_tokens > kQwen4ExpFrontierMoeMaxBatch) return 0;
    const uint64_t conv_channels = static_cast<uint64_t>(
        gdn_conv_channels(spec));
    const uint64_t recurrent = static_cast<uint64_t>(spec.n_heads) *
                               static_cast<uint64_t>(spec.head_dim) *
                               static_cast<uint64_t>(spec.head_dim);
    const uint64_t history = static_cast<uint64_t>(spec.conv_width - 1);
    // Upload old conv/recurrent state; download every qkv row needed to advance
    // conv state and the final recurrent state. Activation I/O is not counted.
    return (history * conv_channels + recurrent +
            static_cast<uint64_t>(n_tokens) * conv_channels + recurrent) *
           sizeof(float);
}

namespace {

bool qsa_spec_valid(const Qwen4ExpFrontierQsaSpec & spec) {
    return spec.n_embd > 0 && spec.n_heads > 0 && spec.n_kv_heads > 0 &&
           spec.n_heads % spec.n_kv_heads == 0 && spec.head_dim > 0 &&
           spec.n_index_heads > 0 && spec.index_dim > 0;
}

bool qsa_matrix(const ggml_tensor * tensor, int64_t input, int64_t output) {
    return tensor && tensor->buffer && ggml_n_dims(tensor) == 2 &&
           tensor->ne[0] == input && tensor->ne[1] == output;
}

void qsa_subgraph_destroy(Qwen4ExpFrontierQsaSubgraph & graph) {
    if (graph.allocator) ggml_gallocr_free(graph.allocator);
    if (graph.ctx) ggml_free(graph.ctx);
    graph = {};
}

void qsa_attention_destroy(Qwen4ExpFrontierQsaAttentionGraph * graph) {
    if (!graph) return;
    if (graph->allocator) ggml_gallocr_free(graph->allocator);
    if (graph->ctx) ggml_free(graph->ctx);
    delete graph;
}

bool qsa_allocate(Qwen4ExpFrontierQsaSubgraph & subgraph,
                  ggml_backend_t backend, std::string & error,
                  const char * failure) {
    subgraph.allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!subgraph.allocator ||
        !ggml_gallocr_alloc_graph(subgraph.allocator, subgraph.graph)) {
        error = failure;
        return false;
    }
    subgraph.arena_bytes =
        ggml_gallocr_get_buffer_size(subgraph.allocator, 0);
    return true;
}

void set_qsa_name(ggml_tensor * tensor, int layer, const char * suffix) {
    if (!tensor) return;
    char name[GGML_MAX_NAME];
    if (layer >= 0) {
        std::snprintf(name, sizeof(name), "qwen_qsa_l%02d_%s", layer,
                      suffix);
    } else {
        std::snprintf(name, sizeof(name), "qwen_qsa_test_%s", suffix);
    }
    ggml_set_name(tensor, name);
}

Qwen4ExpFrontierQsaAttentionGraph * qsa_attention_create(
        Qwen4ExpFrontierQsaGraph * owner, int width, std::string & error) {
    const Qwen4ExpFrontierQsaSpec & spec = owner->spec;
    std::unique_ptr<Qwen4ExpFrontierQsaAttentionGraph> result(
        new Qwen4ExpFrontierQsaAttentionGraph());
    result->width = width;
    const size_t cache_values = static_cast<size_t>(spec.head_dim) *
                                static_cast<size_t>(width) *
                                static_cast<size_t>(spec.n_kv_heads);
    result->padded_key.resize(cache_values);
    result->padded_value.resize(cache_values);
    result->padded_mask.resize(static_cast<size_t>(width));
    std::snprintf(result->profile_label, sizeof(result->profile_label),
                  "qwen4exp/qsa/layer_%02d/attention_q1/k%d", owner->layer,
                  width);

    ggml_init_params params{};
    params.mem_size = 512U * 1024U;
    params.no_alloc = true;
    result->ctx = ggml_init(params);
    if (!result->ctx) {
        error = "Qwen4Exp QSA attention context allocation failed";
        return nullptr;
    }
    ggml_context * ctx = result->ctx;
    result->query = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, spec.head_dim, spec.n_heads);
    result->gate = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, spec.head_dim, spec.n_heads);
    result->key = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, spec.head_dim, width, spec.n_kv_heads);
    result->value = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, spec.head_dim, width, spec.n_kv_heads);
    result->mask = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F16, width, 1, 1);
    ggml_set_input(result->query);
    ggml_set_input(result->gate);
    ggml_set_input(result->key);
    ggml_set_input(result->value);
    ggml_set_input(result->mask);
    set_qsa_name(result->query, owner->layer, "query");
    set_qsa_name(result->key, owner->layer, "selected_key");
    set_qsa_name(result->value, owner->layer, "selected_value");

    ggml_tensor * query = ggml_reshape_4d(
        ctx, result->query, spec.head_dim, 1, spec.n_heads, 1);
    ggml_tensor * key = ggml_reshape_4d(
        ctx, result->key, spec.head_dim, width, spec.n_kv_heads, 1);
    ggml_tensor * value = ggml_reshape_4d(
        ctx, result->value, spec.head_dim, width, spec.n_kv_heads, 1);
    ggml_tensor * attended = ggml_flash_attn_ext(
        ctx, query, key, value, result->mask,
        1.0f / std::sqrt(static_cast<float>(spec.head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
    attended = ggml_reshape_2d(
        ctx, attended, spec.head_dim, spec.n_heads);
    // #27774 uses a self-inverse V rotation on both cache ingress and
    // attention egress. Ingress is part of the host-selection boundary;
    // egress remains fused with attention and output projection here.
    if (owner->weights.value_rotation) {
        attended = ggml_mul_mat(
            ctx, owner->weights.value_rotation, attended);
    }
    attended = ggml_mul(ctx, attended, ggml_sigmoid(ctx, result->gate));
    attended = ggml_reshape_1d(
        ctx, attended,
        static_cast<int64_t>(spec.n_heads) * spec.head_dim);
    result->output = ggml_mul_mat(ctx, owner->weights.output, attended);
    set_qsa_name(result->output, owner->layer, "output");
    ggml_set_output(result->output);
    result->graph = ggml_new_graph_custom(ctx, 96, false);
    ggml_build_forward_expand(result->graph, result->output);
    if (!qsa_allocate(*result, owner->backend, error,
                      "Qwen4Exp QSA attention graph allocation failed")) {
        qsa_attention_destroy(result.release());
        return nullptr;
    }
    return result.release();
}

} // namespace

Qwen4ExpFrontierQsaGraph * qwen4exp_frontier_qsa_create_q1(
        ggml_backend_t backend, const Qwen4ExpFrontierQsaSpec & spec,
        const Qwen4ExpFrontierQsaWeights & weights, int layer,
        std::string & error) {
    error.clear();
    const int64_t q_values = static_cast<int64_t>(spec.n_heads) *
                             spec.head_dim;
    const int64_t kv_values = static_cast<int64_t>(spec.n_kv_heads) *
                              spec.head_dim;
    const int64_t iq_values = static_cast<int64_t>(spec.n_index_heads) *
                              spec.index_dim;
    const bool key_rotation_valid = !weights.key_rotation ||
        qsa_matrix(weights.key_rotation, spec.head_dim, spec.head_dim);
    const bool value_rotation_valid = !weights.value_rotation ||
        qsa_matrix(weights.value_rotation, spec.head_dim, spec.head_dim);
    if (!backend || !qsa_spec_valid(spec) ||
        !qsa_matrix(weights.query, spec.n_embd, 2 * q_values) ||
        !qsa_matrix(weights.key, spec.n_embd, kv_values) ||
        !qsa_matrix(weights.value, spec.n_embd, kv_values) ||
        !qsa_matrix(weights.index_query, spec.n_embd, iq_values) ||
        !qsa_matrix(weights.index_key, spec.n_embd, spec.index_dim) ||
        !qsa_matrix(weights.output, q_values, spec.n_embd) ||
        !key_rotation_valid || !value_rotation_valid) {
        error = "invalid Qwen4Exp persistent QSA tensor contract";
        return nullptr;
    }
    std::unique_ptr<Qwen4ExpFrontierQsaGraph> result(
        new Qwen4ExpFrontierQsaGraph());
    result->backend = backend;
    result->spec = spec;
    result->weights = weights;
    result->layer = layer;
    std::snprintf(result->projection_profile_label,
                  sizeof(result->projection_profile_label),
                  "qwen4exp/qsa/layer_%02d/projection_q1", layer);
    std::snprintf(result->rotation_profile_label,
                  sizeof(result->rotation_profile_label),
                  "qwen4exp/qsa/layer_%02d/rotation_q1", layer);

    ggml_init_params params{};
    params.mem_size = 256U * 1024U;
    params.no_alloc = true;
    result->projection.ctx = ggml_init(params);
    if (!result->projection.ctx) {
        error = "Qwen4Exp QSA projection context allocation failed";
        return nullptr;
    }
    ggml_context * ctx = result->projection.ctx;
    result->projection_input = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, spec.n_embd);
    ggml_set_input(result->projection_input);
    result->projected_query_gate = ggml_mul_mat(
        ctx, weights.query, result->projection_input);
    result->projected_key = ggml_mul_mat(
        ctx, weights.key, result->projection_input);
    result->projected_value = ggml_mul_mat(
        ctx, weights.value, result->projection_input);
    result->projected_index_query = ggml_mul_mat(
        ctx, weights.index_query, result->projection_input);
    result->projected_index_key = ggml_mul_mat(
        ctx, weights.index_key, result->projection_input);
    ggml_set_output(result->projected_query_gate);
    ggml_set_output(result->projected_key);
    ggml_set_output(result->projected_value);
    ggml_set_output(result->projected_index_query);
    ggml_set_output(result->projected_index_key);
    result->projection.graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(result->projection.graph,
                              result->projected_query_gate);
    ggml_build_forward_expand(result->projection.graph,
                              result->projected_key);
    ggml_build_forward_expand(result->projection.graph,
                              result->projected_value);
    ggml_build_forward_expand(result->projection.graph,
                              result->projected_index_query);
    ggml_build_forward_expand(result->projection.graph,
                              result->projected_index_key);
    if (!qsa_allocate(result->projection, backend, error,
                      "Qwen4Exp QSA projection graph allocation failed")) {
        qwen4exp_frontier_qsa_destroy(result.release());
        return nullptr;
    }

    if (weights.key_rotation || weights.value_rotation) {
        params.mem_size = 128U * 1024U;
        result->rotation.ctx = ggml_init(params);
        if (!result->rotation.ctx) {
            error = "Qwen4Exp QSA rotation context allocation failed";
            qwen4exp_frontier_qsa_destroy(result.release());
            return nullptr;
        }
        ctx = result->rotation.ctx;
        result->rotation_query_key = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, spec.head_dim,
            spec.n_heads + spec.n_kv_heads);
        result->rotation_value = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, spec.head_dim, spec.n_kv_heads);
        ggml_set_input(result->rotation_query_key);
        ggml_set_input(result->rotation_value);
        result->rotated_query_key = weights.key_rotation
            ? ggml_mul_mat(ctx, weights.key_rotation,
                           result->rotation_query_key)
            : ggml_dup(ctx, result->rotation_query_key);
        result->rotated_value = weights.value_rotation
            ? ggml_mul_mat(ctx, weights.value_rotation,
                           result->rotation_value)
            : ggml_dup(ctx, result->rotation_value);
        ggml_set_output(result->rotated_query_key);
        ggml_set_output(result->rotated_value);
        result->rotation.graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(result->rotation.graph,
                                  result->rotated_query_key);
        ggml_build_forward_expand(result->rotation.graph,
                                  result->rotated_value);
        if (!qsa_allocate(result->rotation, backend, error,
                          "Qwen4Exp QSA rotation graph allocation failed")) {
            qwen4exp_frontier_qsa_destroy(result.release());
            return nullptr;
        }
    }
    return result.release();
}

void qwen4exp_frontier_qsa_destroy(Qwen4ExpFrontierQsaGraph * graph) {
    if (!graph) return;
    for (const auto & entry : graph->attention)
        qsa_attention_destroy(entry.second);
    qsa_subgraph_destroy(graph->rotation);
    qsa_subgraph_destroy(graph->projection);
    delete graph;
}

bool qwen4exp_frontier_qsa_project_q1(
        Qwen4ExpFrontierQsaGraph * graph, const float * input,
        size_t input_count, std::vector<float> & query_gate,
        std::vector<float> & key, std::vector<float> & value,
        std::vector<float> & index_query, std::vector<float> & index_key,
        std::string & error) {
    if (!graph || !graph->backend || !input ||
        input_count != static_cast<size_t>(graph->spec.n_embd)) {
        error = "invalid Qwen4Exp QSA projection evaluation";
        return false;
    }
    ggml_backend_tensor_set(graph->projection_input, input, 0,
                            input_count * sizeof(float));
    const ProfileRange range(graph->projection_profile_label);
    if (ggml_backend_graph_compute(graph->backend, graph->projection.graph) !=
        GGML_STATUS_SUCCESS) {
        error = "Qwen4Exp QSA projection graph execution failed";
        return false;
    }
    const Qwen4ExpFrontierQsaSpec & spec = graph->spec;
    query_gate.resize(static_cast<size_t>(2 * spec.n_heads * spec.head_dim));
    key.resize(static_cast<size_t>(spec.n_kv_heads * spec.head_dim));
    value.resize(key.size());
    index_query.resize(
        static_cast<size_t>(spec.n_index_heads * spec.index_dim));
    index_key.resize(static_cast<size_t>(spec.index_dim));
    ggml_backend_tensor_get(graph->projected_query_gate, query_gate.data(), 0,
                            query_gate.size() * sizeof(float));
    ggml_backend_tensor_get(graph->projected_key, key.data(), 0,
                            key.size() * sizeof(float));
    ggml_backend_tensor_get(graph->projected_value, value.data(), 0,
                            value.size() * sizeof(float));
    ggml_backend_tensor_get(graph->projected_index_query, index_query.data(),
                            0, index_query.size() * sizeof(float));
    ggml_backend_tensor_get(graph->projected_index_key, index_key.data(), 0,
                            index_key.size() * sizeof(float));
    return true;
}

bool qwen4exp_frontier_qsa_rotate_q1(
        Qwen4ExpFrontierQsaGraph * graph, std::vector<float> & query,
        std::vector<float> & key, std::vector<float> & value,
        std::string & error) {
    if (!graph) {
        error = "invalid Qwen4Exp QSA rotation evaluation";
        return false;
    }
    const Qwen4ExpFrontierQsaSpec & spec = graph->spec;
    const size_t q_values = static_cast<size_t>(spec.n_heads) *
                            static_cast<size_t>(spec.head_dim);
    const size_t kv_values = static_cast<size_t>(spec.n_kv_heads) *
                             static_cast<size_t>(spec.head_dim);
    if (query.size() != q_values || key.size() != kv_values ||
        value.size() != kv_values) {
        error = "invalid Qwen4Exp QSA rotation shape";
        return false;
    }
    if (!graph->rotation.ctx) return true;
    std::vector<float> query_key;
    query_key.reserve(q_values + kv_values);
    query_key.insert(query_key.end(), query.begin(), query.end());
    query_key.insert(query_key.end(), key.begin(), key.end());
    ggml_backend_tensor_set(graph->rotation_query_key, query_key.data(), 0,
                            query_key.size() * sizeof(float));
    ggml_backend_tensor_set(graph->rotation_value, value.data(), 0,
                            value.size() * sizeof(float));
    const ProfileRange range(graph->rotation_profile_label);
    if (ggml_backend_graph_compute(graph->backend, graph->rotation.graph) !=
        GGML_STATUS_SUCCESS) {
        error = "Qwen4Exp QSA rotation graph execution failed";
        return false;
    }
    ggml_backend_tensor_get(graph->rotated_query_key, query_key.data(), 0,
                            query_key.size() * sizeof(float));
    ggml_backend_tensor_get(graph->rotated_value, value.data(), 0,
                            value.size() * sizeof(float));
    std::copy_n(query_key.data(), q_values, query.data());
    std::copy_n(query_key.data() + q_values, kv_values, key.data());
    return true;
}

bool qwen4exp_frontier_qsa_attend_q1(
        Qwen4ExpFrontierQsaGraph * graph, const float * query,
        size_t query_count, const float * gate, size_t gate_count,
        const float * selected_key, const float * selected_value,
        int selected_tokens, std::vector<float> & output,
        std::string & error) {
    if (!graph || !query || !gate || !selected_key || !selected_value) {
        error = "invalid Qwen4Exp QSA attention evaluation";
        return false;
    }
    const Qwen4ExpFrontierQsaSpec & spec = graph->spec;
    const size_t q_values = static_cast<size_t>(spec.n_heads) *
                            static_cast<size_t>(spec.head_dim);
    const int width = qwen4exp_frontier_qsa_cached_width(selected_tokens);
    if (query_count != q_values || gate_count != q_values || width == 0) {
        error = "invalid Qwen4Exp QSA attention shape";
        return false;
    }
    Qwen4ExpFrontierQsaAttentionGraph *& attention =
        graph->attention[width];
    if (!attention) {
        attention = qsa_attention_create(graph, width, error);
        if (!attention) {
            graph->attention.erase(width);
            return false;
        }
        std::fprintf(
            stderr,
            "[qwen-frontier] event=graph_ready component=qsa layer=%d "
            "selected_tokens=%d arena_width=%d arena_bytes=%zu "
            "transfer_bytes=%llu selected_state_bytes=%llu "
            "state_owner=host_snapshot "
            "graph_replay=off\n",
            graph->layer, selected_tokens, width, attention->arena_bytes,
            static_cast<unsigned long long>(
                qwen4exp_frontier_qsa_transfer_bytes_q1(spec,
                                                        selected_tokens)),
            static_cast<unsigned long long>(
                qwen4exp_frontier_qsa_selected_state_bytes_q1(
                    spec, selected_tokens)));
    }
    const size_t head_dim = static_cast<size_t>(spec.head_dim);
    const size_t real_head_values =
        static_cast<size_t>(selected_tokens) * head_dim;
    const size_t padded_head_values = static_cast<size_t>(width) * head_dim;
    for (int head = 0; head < spec.n_kv_heads; ++head) {
        const size_t source = static_cast<size_t>(head) * real_head_values;
        const size_t target = static_cast<size_t>(head) * padded_head_values;
        std::copy_n(selected_key + source, real_head_values,
                    attention->padded_key.data() + target);
        std::copy_n(selected_value + source, real_head_values,
                    attention->padded_value.data() + target);
        std::fill_n(attention->padded_key.data() + target + real_head_values,
                    padded_head_values - real_head_values, 0.0f);
        std::fill_n(attention->padded_value.data() + target + real_head_values,
                    padded_head_values - real_head_values, 0.0f);
    }
    std::fill_n(attention->padded_mask.data(),
                static_cast<size_t>(selected_tokens),
                ggml_fp32_to_fp16(0.0f));
    std::fill(attention->padded_mask.begin() + selected_tokens,
              attention->padded_mask.end(), ggml_fp32_to_fp16(-INFINITY));
    ggml_backend_tensor_set(attention->query, query, 0,
                            query_count * sizeof(float));
    ggml_backend_tensor_set(attention->gate, gate, 0,
                            gate_count * sizeof(float));
    ggml_backend_tensor_set(attention->key, attention->padded_key.data(), 0,
                            attention->padded_key.size() * sizeof(float));
    ggml_backend_tensor_set(attention->value, attention->padded_value.data(), 0,
                            attention->padded_value.size() * sizeof(float));
    ggml_backend_tensor_set(attention->mask, attention->padded_mask.data(), 0,
                            attention->padded_mask.size() *
                                sizeof(ggml_fp16_t));
    const ProfileRange range(attention->profile_label);
    if (ggml_backend_graph_compute(graph->backend, attention->graph) !=
        GGML_STATUS_SUCCESS) {
        error = "Qwen4Exp QSA attention graph execution failed";
        return false;
    }
    output.resize(static_cast<size_t>(spec.n_embd));
    ggml_backend_tensor_get(attention->output, output.data(), 0,
                            output.size() * sizeof(float));
    return true;
}

size_t qwen4exp_frontier_qsa_arena_bytes(
        const Qwen4ExpFrontierQsaGraph * graph) {
    if (!graph) return 0;
    size_t total = graph->projection.arena_bytes + graph->rotation.arena_bytes;
    for (const auto & entry : graph->attention)
        total += entry.second->arena_bytes;
    return total;
}

uint64_t qwen4exp_frontier_qsa_transfer_bytes_q1(
        const Qwen4ExpFrontierQsaSpec & spec, int selected_tokens) {
    const int width = qwen4exp_frontier_qsa_cached_width(selected_tokens);
    if (!qsa_spec_valid(spec) || width == 0) return 0;
    const uint64_t query = static_cast<uint64_t>(spec.n_heads) *
                           static_cast<uint64_t>(spec.head_dim);
    const uint64_t kv = static_cast<uint64_t>(spec.n_kv_heads) *
                        static_cast<uint64_t>(spec.head_dim);
    const uint64_t index_query = static_cast<uint64_t>(spec.n_index_heads) *
                                 static_cast<uint64_t>(spec.index_dim);
    const uint64_t projected = 2U * query + 2U * kv + index_query +
                               static_cast<uint64_t>(spec.index_dim);
    const uint64_t rotation = 2U * (query + 2U * kv);
    const uint64_t attention_inputs = 2U * query +
        2U * static_cast<uint64_t>(width) * kv;
    // Activation upload + projection downloads, rotation upload/download,
    // attention/gate/padded-cache upload, F16 validity mask upload, and final
    // activation download. This is the exact synchronized transfer count for
    // the selected width; selected-state bytes are reported separately.
    return (static_cast<uint64_t>(spec.n_embd) + projected + rotation +
            attention_inputs + static_cast<uint64_t>(spec.n_embd)) *
               sizeof(float) +
           static_cast<uint64_t>(width) * sizeof(ggml_fp16_t);
}

uint64_t qwen4exp_frontier_qsa_selected_state_bytes_q1(
        const Qwen4ExpFrontierQsaSpec & spec, int selected_tokens) {
    if (!qsa_spec_valid(spec) ||
        qwen4exp_frontier_qsa_cached_width(selected_tokens) == 0) return 0;
    return 2U * static_cast<uint64_t>(selected_tokens) *
           static_cast<uint64_t>(spec.n_kv_heads) *
           static_cast<uint64_t>(spec.head_dim) * sizeof(float);
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
    const bool fused_experts = tensor_shape(
        weights.experts_gate_up, 3, spec.n_embd,
        2LL * spec.n_ff, spec.n_expert);
    const bool split_experts =
        tensor_shape(weights.experts_gate, 3, spec.n_embd,
                     spec.n_ff, spec.n_expert) &&
        tensor_shape(weights.experts_up, 3, spec.n_embd,
                     spec.n_ff, spec.n_expert);
    if (!backend || spec.n_embd <= 0 || spec.n_expert <= 0 ||
        spec.n_expert_used <= 0 || spec.n_expert_used > spec.n_expert ||
        spec.n_ff <= 0 || n_tokens <= 0 ||
        n_tokens > kQwen4ExpFrontierMoeMaxBatch ||
        !tensor_shape(weights.router, 2, spec.n_embd, spec.n_expert) ||
        fused_experts == split_experts ||
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
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    if (fused_experts) {
        ggml_tensor * gate_up = ggml_mul_mat_id(
            ctx, weights.experts_gate_up, input_3d, selected);
        set_layer_name(gate_up, layer, "gate_up");
        gate = ggml_view_3d(
            ctx, gate_up, spec.n_ff, gate_up->ne[1], gate_up->ne[2],
            gate_up->nb[1], gate_up->nb[2], 0);
        up = ggml_view_3d(
            ctx, gate_up, spec.n_ff, gate_up->ne[1], gate_up->ne[2],
            gate_up->nb[1], gate_up->nb[2],
            static_cast<size_t>(spec.n_ff) * ggml_element_size(gate_up));
        gate = ggml_cont(ctx, gate);
        up = ggml_cont(ctx, up);
    } else {
        gate = ggml_mul_mat_id(ctx, weights.experts_gate, input_3d, selected);
        up = ggml_mul_mat_id(ctx, weights.experts_up, input_3d, selected);
        set_layer_name(gate, layer, "gate");
        set_layer_name(up, layer, "up");
    }
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
    // Qwen's canonical shared-expert gate is BF16.  HIP's broadcast binary
    // kernel accepts an F32 activation only when the broadcast operand is F32
    // or F16, so make the weight-side conversion explicit.  The tensor is one
    // row (10 KiB in the production model), and keeping the multiply in F32
    // also preserves the scalar gate accumulation used by the CPU reference.
    ggml_tensor * shared_gate_input = weights.shared_gate_input;
    if (shared_gate_input->type != GGML_TYPE_F32) {
        shared_gate_input = ggml_cast(
            ctx, shared_gate_input, GGML_TYPE_F32);
    }
    ggml_tensor * shared_scale = ggml_sum_rows(
        ctx, ggml_mul(ctx, result->input, shared_gate_input));
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
    weights.dense_cache = qwen4exp_frontier_dense_cache_create();
    if (!weights.backend || weights.layers.size() != 48) {
        error = "invalid Qwen4Exp weights for frontier initialization";
        qwen4exp_frontier_destroy(weights);
        return false;
    }
    const bool moe_enabled = env_enabled("EMBER_QWEN_FRONTIER_MOE", true);
    const bool gdn_enabled = env_enabled("EMBER_QWEN_FRONTIER_GDN", true);
    const bool qsa_enabled = env_enabled("EMBER_QWEN_FRONTIER_QSA", true);
    std::unique_ptr<Qwen4ExpFrontierRuntime> runtime(
        new Qwen4ExpFrontierRuntime());
    runtime->stats = env_enabled("EMBER_QWEN_FRONTIER_STATS", false);
    runtime->moe.resize(weights.layers.size());
    runtime->gdn.resize(weights.layers.size());
    runtime->qsa.resize(weights.layers.size(), nullptr);
    const Qwen4ExpFrontierMoeSpec moe_spec{2560, 512, 10, 640};
    const Qwen4ExpFrontierGdnSpec gdn_spec{
        2560, 48, 16, 128, 4, 1.0e-6f};
    const Qwen4ExpFrontierQsaSpec qsa_spec{2560, 24, 2, 256, 4, 128};
    for (size_t index = 0; index < weights.layers.size(); ++index) {
        const Qwen4ExpLayer & layer = weights.layers[index];
        if (moe_enabled) {
            const Qwen4ExpFrontierMoeWeights graph_weights{
                layer.router, layer.experts_gate_up_tensor,
                layer.experts_gate_tensor, layer.experts_up_tensor,
                layer.experts_down_tensor, layer.shared_gate_input,
                layer.shared_gate, layer.shared_up, layer.shared_down};
            runtime->moe[index][1] = qwen4exp_frontier_moe_create(
                weights.backend, moe_spec, graph_weights,
                static_cast<int>(index), error);
        }
        if (moe_enabled && !runtime->moe[index][1]) {
            for (Qwen4ExpFrontierQsaGraph * built : runtime->qsa)
                qwen4exp_frontier_qsa_destroy(built);
            for (const Qwen4ExpFrontierRuntime::GdnLayerGraphs & layer_graphs :
                 runtime->gdn)
                for (Qwen4ExpFrontierGdnGraph * built : layer_graphs)
                    qwen4exp_frontier_gdn_destroy(built);
            for (const Qwen4ExpFrontierRuntime::LayerGraphs & layer_graphs :
                 runtime->moe) {
                for (Qwen4ExpFrontierMoeGraph * built : layer_graphs)
                    qwen4exp_frontier_moe_destroy(built);
            }
            qwen4exp_frontier_dense_cache_destroy(weights.dense_cache);
            weights.dense_cache = nullptr;
            return false;
        }
        if (gdn_enabled && (index + 1U) % 4U != 0U) {
            const Qwen4ExpFrontierGdnWeights graph_weights{
                layer.attn_qkv, layer.attn_gate, layer.ssm_alpha,
                layer.ssm_beta, layer.ssm_conv, layer.ssm_a, layer.ssm_dt,
                layer.ssm_norm, layer.ssm_out};
            runtime->gdn[index][1] = qwen4exp_frontier_gdn_create_q1(
                weights.backend, gdn_spec, graph_weights,
                static_cast<int>(index), error);
            if (!runtime->gdn[index][1]) {
                for (Qwen4ExpFrontierQsaGraph * built : runtime->qsa)
                    qwen4exp_frontier_qsa_destroy(built);
                for (const Qwen4ExpFrontierRuntime::GdnLayerGraphs & layer_graphs :
                     runtime->gdn)
                    for (Qwen4ExpFrontierGdnGraph * built : layer_graphs)
                        qwen4exp_frontier_gdn_destroy(built);
                for (const Qwen4ExpFrontierRuntime::LayerGraphs & layer_graphs :
                     runtime->moe) {
                    for (Qwen4ExpFrontierMoeGraph * built : layer_graphs)
                        qwen4exp_frontier_moe_destroy(built);
                }
                qwen4exp_frontier_dense_cache_destroy(weights.dense_cache);
                weights.dense_cache = nullptr;
                return false;
            }
        }
        if (qsa_enabled && (index + 1U) % 4U == 0U) {
            const Qwen4ExpFrontierQsaWeights graph_weights{
                layer.attn_q, layer.attn_k, layer.attn_v, layer.index_q,
                layer.index_k, layer.attn_output, layer.self_k_rot,
                layer.self_v_rot};
            runtime->qsa[index] = qwen4exp_frontier_qsa_create_q1(
                weights.backend, qsa_spec, graph_weights,
                static_cast<int>(index), error);
            if (!runtime->qsa[index]) {
                for (Qwen4ExpFrontierQsaGraph * built : runtime->qsa)
                    qwen4exp_frontier_qsa_destroy(built);
                for (const Qwen4ExpFrontierRuntime::GdnLayerGraphs & layer_graphs :
                     runtime->gdn)
                    for (Qwen4ExpFrontierGdnGraph * built : layer_graphs)
                        qwen4exp_frontier_gdn_destroy(built);
                for (const Qwen4ExpFrontierRuntime::LayerGraphs & layer_graphs :
                     runtime->moe) {
                    for (Qwen4ExpFrontierMoeGraph * built : layer_graphs)
                        qwen4exp_frontier_moe_destroy(built);
                }
                qwen4exp_frontier_dense_cache_destroy(weights.dense_cache);
                weights.dense_cache = nullptr;
                return false;
            }
        }
    }
    weights.frontier = runtime.release();
    size_t gdn_arena_bytes = 0;
    for (const Qwen4ExpFrontierRuntime::GdnLayerGraphs & layer_graphs :
         weights.frontier->gdn)
        for (const Qwen4ExpFrontierGdnGraph * graph : layer_graphs)
            if (graph) gdn_arena_bytes += graph->arena_bytes;
    size_t qsa_arena_bytes = 0;
    for (const Qwen4ExpFrontierQsaGraph * graph : weights.frontier->qsa)
        qsa_arena_bytes += qwen4exp_frontier_qsa_arena_bytes(graph);
    std::fprintf(stderr,
                 "[qwen-frontier] event=ready component=moe enabled=%s "
                 "graphs=%d tokens_per_graph=1 lazy_batch_widths=5,16 "
                 "component_gdn_enabled=%s gdn_graphs=%d gdn_width=q1 "
                 "gdn_lazy_exact_batch_widths=2..16 gdn_arena_bytes=%zu "
                 "gdn_state_boundary_bytes_per_layer=%llu "
                 "gdn_state_boundary_bytes_q16=%llu "
                 "gdn_state_owner=host_snapshot component_qsa_enabled=%s "
                 "qsa_graphs=%d qsa_attention_widths=lazy:16,64,256,1024,2048,2051 "
                 "qsa_base_arena_bytes=%zu qsa_state_owner=host_snapshot "
                 "qsa_transfer_bytes_k2050=%llu "
                 "qsa_selected_state_bytes_k2050=%llu graph_replay=off\n",
                 moe_enabled ? "true" : "false", moe_enabled ? 48 : 0,
                 gdn_enabled ? "true" : "false", gdn_enabled ? 36 : 0,
                 gdn_arena_bytes,
                 static_cast<unsigned long long>(
                     qwen4exp_frontier_gdn_state_transfer_bytes_q1(gdn_spec)),
                 static_cast<unsigned long long>(
                     qwen4exp_frontier_gdn_state_transfer_bytes_batch(
                         gdn_spec, 16)),
                 qsa_enabled ? "true" : "false", qsa_enabled ? 12 : 0,
                 qsa_arena_bytes,
                 static_cast<unsigned long long>(
                     qwen4exp_frontier_qsa_transfer_bytes_q1(qsa_spec, 2050)),
                 static_cast<unsigned long long>(
                     qwen4exp_frontier_qsa_selected_state_bytes_q1(
                         qsa_spec, 2050)));
    return true;
}

void qwen4exp_frontier_destroy(Qwen4ExpWeights & weights) {
    Qwen4ExpFrontierRuntime * runtime = weights.frontier;
    if (runtime && runtime->stats) {
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
        const double average = calls
            ? static_cast<double>(compute_us) / static_cast<double>(calls)
            : 0.0;
        std::fprintf(stderr,
                     "[qwen-frontier] event=summary component=moe calls=%llu "
                     "compute_us=%llu avg_compute_us=%.3f\n",
                     static_cast<unsigned long long>(calls),
                     static_cast<unsigned long long>(compute_us), average);
    }
    if (runtime) {
        for (Qwen4ExpFrontierQsaGraph * graph : runtime->qsa)
            qwen4exp_frontier_qsa_destroy(graph);
        for (const Qwen4ExpFrontierRuntime::GdnLayerGraphs & layer_graphs :
             runtime->gdn)
            for (Qwen4ExpFrontierGdnGraph * graph : layer_graphs)
                qwen4exp_frontier_gdn_destroy(graph);
        for (const Qwen4ExpFrontierRuntime::LayerGraphs & layer_graphs :
             runtime->moe) {
            for (Qwen4ExpFrontierMoeGraph * graph : layer_graphs)
                qwen4exp_frontier_moe_destroy(graph);
        }
        delete runtime;
    }
    weights.frontier = nullptr;
    qwen4exp_frontier_dense_cache_destroy(weights.dense_cache);
    weights.dense_cache = nullptr;
}

bool qwen4exp_frontier_moe_available(const Qwen4ExpWeights & weights,
                                     int layer) {
    return weights.frontier && layer >= 0 &&
           static_cast<size_t>(layer) < weights.frontier->moe.size() &&
           weights.frontier->moe[static_cast<size_t>(layer)][1] != nullptr;
}

bool qwen4exp_frontier_gdn_available(const Qwen4ExpWeights & weights,
                                     int layer) {
    return weights.frontier && layer >= 0 &&
           static_cast<size_t>(layer) < weights.frontier->gdn.size() &&
           weights.frontier->gdn[static_cast<size_t>(layer)][1] != nullptr;
}

bool qwen4exp_frontier_qsa_available(const Qwen4ExpWeights & weights,
                                     int layer) {
    return weights.frontier && layer >= 0 &&
           static_cast<size_t>(layer) < weights.frontier->qsa.size() &&
           weights.frontier->qsa[static_cast<size_t>(layer)] != nullptr;
}

Qwen4ExpFrontierQsaGraph * qwen4exp_frontier_qsa_q1(
        const Qwen4ExpWeights & weights, int layer) {
    return qwen4exp_frontier_qsa_available(weights, layer)
        ? weights.frontier->qsa[static_cast<size_t>(layer)] : nullptr;
}

bool qwen4exp_frontier_moe_q1(const Qwen4ExpWeights & weights, int layer,
                              const float * input, size_t input_count,
                              std::vector<float> & output,
                              std::string & error) {
    if (!qwen4exp_frontier_moe_available(weights, layer)) {
        error = "Qwen4Exp frontier MoE graph is unavailable";
        return false;
    }
    return qwen4exp_frontier_moe_eval(
        weights.frontier->moe[static_cast<size_t>(layer)][1], input,
        input_count, output, error);
}

bool qwen4exp_frontier_run_rocmi4_dispatch_controls(
        const Qwen4ExpWeights & weights, std::string & error) {
    if (!weights.backend || !weights.dense_cache || !weights.frontier) {
        error = "Qwen4Exp ROCMI4 dispatch controls require initialized frontiers";
        return false;
    }
    ggml_tensor * dense_weight = nullptr;
    int moe_layer = -1;
    for (size_t index = 0; index < weights.layers.size(); ++index) {
        const Qwen4ExpLayer & layer = weights.layers[index];
        if (!dense_weight && layer.attn_qkv &&
            layer.attn_qkv->type == GGML_TYPE_Q4_0_ROCMI4) {
            dense_weight = layer.attn_qkv;
        }
        const ggml_tensor * expert = layer.experts_gate_up_tensor
            ? layer.experts_gate_up_tensor : layer.experts_gate_tensor;
        if (moe_layer < 0 && expert &&
            expert->type == GGML_TYPE_Q4_0_ROCMI4 &&
            qwen4exp_frontier_moe_available(
                weights, static_cast<int>(index))) {
            moe_layer = static_cast<int>(index);
        }
    }
    const ggml_tensor * expert_weight = nullptr;
    if (moe_layer >= 0) {
        const Qwen4ExpLayer & layer =
            weights.layers[static_cast<size_t>(moe_layer)];
        expert_weight = layer.experts_gate_up_tensor
            ? layer.experts_gate_up_tensor : layer.experts_gate_tensor;
    }
    if (!dense_weight && !expert_weight) {
        std::fprintf(stderr,
                     "[rocmi4-w4a8-dispatch] event=control_suite "
                     "capability=no_eligible_rocmi4_mmq dense_q=none "
                     "routed_expert_q=none execution=completed\n");
        return true;
    }
    if (!dense_weight && expert_weight) {
        error = "Qwen4Exp ROCMI4 routed-only dispatch capability is unsupported";
        return false;
    }

    std::vector<float> input(16U * 2560U, 0.0f);
    std::vector<float> output;
    for (const int q : {1, 4, 5, 16}) {
        const std::string control_id = "dense-q" + std::to_string(q);
        log_rocmi4_dispatch_control(
            control_id.c_str(), "dense", q, "begin", dense_weight);
        if (!qwen4exp_frontier_dense_eval(
                weights.dense_cache, weights.backend, dense_weight,
                input.data(), 2560, q, output, error)) {
            error = "Qwen4Exp dense dispatch control q=" +
                    std::to_string(q) + " failed: " + error;
            return false;
        }
        log_rocmi4_dispatch_post_compute(
            control_id.c_str(), "dense", q,
            qwen4exp_frontier_dense_cached_width(q), dense_weight);
        log_rocmi4_dispatch_control(
            control_id.c_str(), "dense", q, "completed", dense_weight);
    }
    if (expert_weight) {
        log_rocmi4_dispatch_control(
            "routed-expert-q1", "routed_expert", 1, "begin",
            expert_weight);
        if (!qwen4exp_frontier_moe_q1(
                weights, moe_layer, input.data(), 2560, output, error)) {
            error = "Qwen4Exp routed-expert dispatch control q=1 failed: " + error;
            return false;
        }
        log_rocmi4_dispatch_post_compute(
            "routed-expert-q1", "routed_expert", 1, 1, expert_weight);
        log_rocmi4_dispatch_control(
            "routed-expert-q1", "routed_expert", 1, "completed",
            expert_weight);
        for (const int q : {5, 16}) {
            const std::string control_id =
                "routed-expert-q" + std::to_string(q);
            log_rocmi4_dispatch_control(
                control_id.c_str(), "routed_expert", q, "begin",
                expert_weight);
            if (!qwen4exp_frontier_moe_batch(
                    weights, moe_layer, input.data(),
                    static_cast<size_t>(q) * 2560U, q, output, error)) {
                error = "Qwen4Exp routed-expert dispatch control q=" +
                        std::to_string(q) + " failed: " + error;
                return false;
            }
            log_rocmi4_dispatch_post_compute(
                control_id.c_str(), "routed_expert", q, q, expert_weight);
            log_rocmi4_dispatch_control(
                control_id.c_str(), "routed_expert", q, "completed",
                expert_weight);
        }
    }
    std::fprintf(stderr,
                 "[rocmi4-w4a8-dispatch] event=control_suite "
                 "capability=%s dense_q=1,4,5,16 routed_expert_q=%s "
                 "execution=completed\n",
                 expert_weight ? "rocmi4_dense_and_routed" : "rocmi4_dense_only",
                 expert_weight ? "1,5,16" : "none");
    return true;
}

bool qwen4exp_frontier_run_projection_numerics_control(
        const Qwen4ExpWeights & weights, std::string & error) {
    if (!weights.backend || !weights.dense_cache || !weights.frontier) {
        error = "Qwen4Exp projection numerics control requires initialized frontiers";
        return false;
    }
    ggml_tensor * weight = nullptr;
    for (const Qwen4ExpLayer & layer : weights.layers) {
        if (layer.attn_qkv &&
            layer.attn_qkv->type == GGML_TYPE_Q4_0_ROCMFP4_FAST) {
            weight = layer.attn_qkv;
            break;
        }
    }
    if (!weight || ggml_n_dims(weight) != 2 || weight->ne[0] <= 0 ||
        weight->ne[0] > std::numeric_limits<int>::max() ||
        weight->ne[1] <= 1) {
        error = "Qwen4Exp projection numerics control found no eligible type-101 weight";
        return false;
    }

    constexpr int kRows = 16;
    const int input_count = static_cast<int>(weight->ne[0]);
    const size_t row_values = static_cast<size_t>(weight->ne[1]);
    std::vector<float> input(static_cast<size_t>(kRows) *
                             static_cast<size_t>(input_count));
    for (int row = 0; row < kRows; ++row) {
        for (int column = 0; column < input_count; ++column) {
            const float x = static_cast<float>((row + 1) * (column + 3));
            input[static_cast<size_t>(row) *
                      static_cast<size_t>(input_count) +
                  static_cast<size_t>(column)] =
                0.25f * std::sin(x * 0.013f) +
                0.05f * std::cos(x * 0.037f);
        }
    }

    std::vector<float> reference;
    reference.reserve(static_cast<size_t>(kRows) * row_values);
    for (int row = 0; row < kRows; ++row) {
        std::vector<float> q1;
        if (!qwen4exp_frontier_dense_eval(
                weights.dense_cache, weights.backend, weight,
                input.data() + static_cast<size_t>(row) *
                                   static_cast<size_t>(input_count),
                input_count, 1, q1, error)) {
            error = "Qwen4Exp projection numerics q1 failed: " + error;
            return false;
        }
        if (q1.size() != row_values) {
            error = "Qwen4Exp projection numerics q1 returned the wrong shape";
            return false;
        }
        reference.insert(reference.end(), q1.begin(), q1.end());
    }

    for (const int logical_q : {5, 16}) {
        std::vector<float> batch;
        if (!qwen4exp_frontier_dense_eval(
                weights.dense_cache, weights.backend, weight, input.data(),
                input_count, logical_q, batch, error)) {
            error = "Qwen4Exp projection numerics q=" +
                    std::to_string(logical_q) + " failed: " + error;
            return false;
        }
        const size_t values = static_cast<size_t>(logical_q) * row_values;
        if (batch.size() != values || reference.size() < values) {
            error = "Qwen4Exp projection numerics batch returned the wrong shape";
            return false;
        }
        double squared_error = 0.0;
        double squared_reference = 0.0;
        double signed_error = 0.0;
        float max_abs = 0.0f;
        for (size_t index = 0; index < values; ++index) {
            const float delta = batch[index] - reference[index];
            const float absolute = std::fabs(delta);
            max_abs = std::max(max_abs, absolute);
            squared_error += static_cast<double>(delta) * delta;
            squared_reference +=
                static_cast<double>(reference[index]) * reference[index];
            signed_error += delta;
        }
        const double rms = std::sqrt(squared_error /
                                     static_cast<double>(values));
        const double reference_rms = std::sqrt(
            squared_reference / static_cast<double>(values));
        const double normalized_rms = reference_rms > 0.0
            ? rms / reference_rms : 0.0;
        std::fprintf(stderr,
                     "[qwen-numerics] event=projection_compare weight=%s "
                     "type=%s logical_q=%d physical_q=%d values=%zu "
                     "max_abs=%.9g rms=%.9g reference_rms=%.9g "
                     "normalized_rms=%.9g mean_error=%.9g\n",
                     weight->name, ggml_type_name(weight->type), logical_q,
                     qwen4exp_frontier_dense_cached_width(logical_q), values,
                     static_cast<double>(max_abs), rms, reference_rms,
                     normalized_rms,
                     signed_error / static_cast<double>(values));
    }
    return true;
}

bool qwen4exp_frontier_gdn_q1(
        const Qwen4ExpWeights & weights, int layer, const float * input,
        size_t input_count, const float * conv_state, size_t conv_state_count,
        const float * recurrent_state, size_t recurrent_state_count,
        std::vector<float> & output, std::vector<float> & next_conv_state,
        std::vector<float> & next_recurrent_state, std::string & error) {
    if (!qwen4exp_frontier_gdn_available(weights, layer)) {
        error = "Qwen4Exp frontier GDN graph is unavailable";
        return false;
    }
    return qwen4exp_frontier_gdn_eval_q1(
        weights.frontier->gdn[static_cast<size_t>(layer)][1], input,
        input_count,
        conv_state, conv_state_count, recurrent_state, recurrent_state_count,
        output, next_conv_state, next_recurrent_state, error);
}

bool qwen4exp_frontier_gdn_batch(
        const Qwen4ExpWeights & weights, int layer, const float * input,
        size_t input_count, int n_tokens, const float * conv_state,
        size_t conv_state_count, const float * recurrent_state,
        size_t recurrent_state_count, std::vector<float> & output,
        std::vector<float> & next_conv_state,
        std::vector<float> & next_recurrent_state, std::string & error) {
    if (!weights.frontier || layer < 0 || n_tokens < 2 ||
        n_tokens > kQwen4ExpFrontierMoeMaxBatch || !input ||
        static_cast<size_t>(layer) >= weights.frontier->gdn.size() ||
        !weights.frontier->gdn[static_cast<size_t>(layer)][1] ||
        input_count != static_cast<size_t>(n_tokens) * 2560U) {
        error = "Qwen4Exp frontier batched GDN graph is unavailable";
        return false;
    }
    Qwen4ExpFrontierRuntime::GdnLayerGraphs & layer_graphs =
        weights.frontier->gdn[static_cast<size_t>(layer)];
    Qwen4ExpFrontierGdnGraph *& graph =
        layer_graphs[static_cast<size_t>(n_tokens)];
    if (!graph) {
        const Qwen4ExpLayer & model_layer =
            weights.layers[static_cast<size_t>(layer)];
        const Qwen4ExpFrontierGdnSpec spec{
            2560, 48, 16, 128, 4, 1.0e-6f};
        const Qwen4ExpFrontierGdnWeights graph_weights{
            model_layer.attn_qkv, model_layer.attn_gate,
            model_layer.ssm_alpha, model_layer.ssm_beta,
            model_layer.ssm_conv, model_layer.ssm_a, model_layer.ssm_dt,
            model_layer.ssm_norm, model_layer.ssm_out};
        graph = qwen4exp_frontier_gdn_create_batch(
            weights.backend, spec, graph_weights, layer, n_tokens, error);
        if (!graph) return false;
        std::fprintf(
            stderr,
            "[qwen-frontier] event=graph_ready component=gdn layer=%d "
            "tokens=%d arena_bytes=%zu state_boundary_bytes=%llu "
            "state_owner=host_snapshot graph_replay=off\n",
            layer, n_tokens, graph->arena_bytes,
            static_cast<unsigned long long>(
                qwen4exp_frontier_gdn_state_transfer_bytes_batch(
                    spec, n_tokens)));
    }
    return qwen4exp_frontier_gdn_eval_batch(
        graph, input, input_count, conv_state, conv_state_count,
        recurrent_state, recurrent_state_count, output, next_conv_state,
        next_recurrent_state, error);
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
            model_layer.experts_gate_tensor, model_layer.experts_up_tensor,
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
    weights.dense_cache = qwen4exp_frontier_dense_cache_create();
    const Qwen4ExpLayer & layer = weights.layer;
    const Qwen4ExpFrontierMoeSpec spec{2560, 512, 10, 640};
    const Qwen4ExpFrontierMoeWeights graph_weights{
        layer.router, layer.experts_gate_up_tensor,
        layer.experts_gate_tensor, layer.experts_up_tensor,
        layer.experts_down_tensor, layer.shared_gate_input,
        layer.shared_gate, layer.shared_up, layer.shared_down};
    weights.frontier_moe = qwen4exp_frontier_moe_create(
        weights.backend, spec, graph_weights, 48, error);
    if (!weights.frontier_moe) {
        qwen4exp_frontier_dense_cache_destroy(weights.dense_cache);
        weights.dense_cache = nullptr;
        return false;
    }
    const bool qsa_enabled = env_enabled("EMBER_QWEN_FRONTIER_QSA", true);
    const Qwen4ExpFrontierQsaSpec qsa_spec{2560, 24, 2, 256, 4, 128};
    if (qsa_enabled) {
        const Qwen4ExpFrontierQsaWeights qsa_weights{
            layer.attn_q, layer.attn_k, layer.attn_v, layer.index_q,
            layer.index_k, layer.attn_output, layer.self_k_rot,
            layer.self_v_rot};
        weights.frontier_qsa = qwen4exp_frontier_qsa_create_q1(
            weights.backend, qsa_spec, qsa_weights, 48, error);
        if (!weights.frontier_qsa) {
            qwen4exp_frontier_mtp_destroy(weights);
            return false;
        }
    }
    std::fprintf(stderr,
                 "[qwen-frontier] event=ready component=mtp_moe graphs=1 "
                 "tokens_per_graph=1 arena_bytes=%zu "
                 "component_mtp_qsa_enabled=%s qsa_graphs=%d "
                 "qsa_base_arena_bytes=%zu weight_copies=0 "
                 "graph_replay=off\n",
                 weights.frontier_moe->arena_bytes,
                 qsa_enabled ? "true" : "false", qsa_enabled ? 1 : 0,
                 qwen4exp_frontier_qsa_arena_bytes(weights.frontier_qsa));
    return true;
}

void qwen4exp_frontier_mtp_destroy(Qwen4ExpMtpWeights & weights) {
    qwen4exp_frontier_qsa_destroy(weights.frontier_qsa);
    weights.frontier_qsa = nullptr;
    qwen4exp_frontier_moe_destroy(weights.frontier_moe);
    weights.frontier_moe = nullptr;
    qwen4exp_frontier_dense_cache_destroy(weights.dense_cache);
    weights.dense_cache = nullptr;
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
