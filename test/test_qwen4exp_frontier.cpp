#include "qwen4exp_frontier.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

using dflash::common::Qwen4ExpFrontierMoeGraph;
using dflash::common::Qwen4ExpFrontierMoeSpec;
using dflash::common::Qwen4ExpFrontierMoeWeights;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(condition, message) do {                                      \
    if (condition) { ++g_pass; } else {                                     \
        ++g_fail; std::fprintf(stderr, "FAIL: %s\n", message);            \
    }                                                                        \
} while (0)

static float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

static float silu(float value) {
    return value * sigmoid(value);
}

static std::vector<float> matvec(const std::vector<float> & matrix,
                                 int rows, int columns,
                                 const std::vector<float> & input,
                                 size_t base = 0) {
    std::vector<float> output(static_cast<size_t>(rows), 0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            output[static_cast<size_t>(row)] +=
                matrix[base + static_cast<size_t>(row * columns + column)] *
                input[static_cast<size_t>(column)];
        }
    }
    return output;
}

static std::vector<float> reference_moe(
        const Qwen4ExpFrontierMoeSpec & spec,
        const std::vector<float> & router,
        const std::vector<float> & gate_up,
        const std::vector<float> & down,
        const std::vector<float> & shared_gate_input,
        const std::vector<float> & shared_gate,
        const std::vector<float> & shared_up,
        const std::vector<float> & shared_down,
        const std::vector<float> & input) {
    const std::vector<float> logits =
        matvec(router, spec.n_expert, spec.n_embd, input);
    std::vector<int32_t> ids(static_cast<size_t>(spec.n_expert));
    std::iota(ids.begin(), ids.end(), int32_t{0});
    std::partial_sort(ids.begin(), ids.begin() + spec.n_expert_used, ids.end(),
        [&](int32_t a, int32_t b) {
            if (logits[static_cast<size_t>(a)] !=
                logits[static_cast<size_t>(b)]) {
                return logits[static_cast<size_t>(a)] >
                       logits[static_cast<size_t>(b)];
            }
            return a < b;
        });
    const float selected_max = logits[static_cast<size_t>(ids[0])];
    std::vector<float> weights(static_cast<size_t>(spec.n_expert_used));
    float sum = 0.0f;
    for (int slot = 0; slot < spec.n_expert_used; ++slot) {
        weights[static_cast<size_t>(slot)] = std::exp(
            logits[static_cast<size_t>(ids[static_cast<size_t>(slot)])] -
            selected_max);
        sum += weights[static_cast<size_t>(slot)];
    }
    for (float & weight : weights) weight /= sum;

    std::vector<float> output(static_cast<size_t>(spec.n_embd), 0.0f);
    const size_t gate_up_stride = static_cast<size_t>(spec.n_embd) *
                                  static_cast<size_t>(2 * spec.n_ff);
    const size_t down_stride = static_cast<size_t>(spec.n_ff) *
                               static_cast<size_t>(spec.n_embd);
    for (int slot = 0; slot < spec.n_expert_used; ++slot) {
        const int expert = ids[static_cast<size_t>(slot)];
        std::vector<float> projected = matvec(
            gate_up, 2 * spec.n_ff, spec.n_embd, input,
            static_cast<size_t>(expert) * gate_up_stride);
        std::vector<float> hidden(static_cast<size_t>(spec.n_ff));
        for (int channel = 0; channel < spec.n_ff; ++channel) {
            hidden[static_cast<size_t>(channel)] =
                silu(projected[static_cast<size_t>(channel)]) *
                projected[static_cast<size_t>(spec.n_ff + channel)];
        }
        const std::vector<float> expert_output = matvec(
            down, spec.n_embd, spec.n_ff, hidden,
            static_cast<size_t>(expert) * down_stride);
        for (int channel = 0; channel < spec.n_embd; ++channel) {
            output[static_cast<size_t>(channel)] +=
                weights[static_cast<size_t>(slot)] *
                expert_output[static_cast<size_t>(channel)];
        }
    }

    std::vector<float> shared_g =
        matvec(shared_gate, spec.n_ff, spec.n_embd, input);
    const std::vector<float> shared_u =
        matvec(shared_up, spec.n_ff, spec.n_embd, input);
    for (int channel = 0; channel < spec.n_ff; ++channel) {
        shared_g[static_cast<size_t>(channel)] =
            silu(shared_g[static_cast<size_t>(channel)]) *
            shared_u[static_cast<size_t>(channel)];
    }
    const std::vector<float> shared =
        matvec(shared_down, spec.n_embd, spec.n_ff, shared_g);
    float gate = 0.0f;
    for (int channel = 0; channel < spec.n_embd; ++channel) {
        gate += shared_gate_input[static_cast<size_t>(channel)] *
                input[static_cast<size_t>(channel)];
    }
    gate = sigmoid(gate);
    for (int channel = 0; channel < spec.n_embd; ++channel) {
        output[static_cast<size_t>(channel)] +=
            gate * shared[static_cast<size_t>(channel)];
    }
    return output;
}

static bool close_vectors(const std::vector<float> & actual,
                          const std::vector<float> & expected,
                          float tolerance = 2.0e-5f) {
    if (actual.size() != expected.size()) return false;
    for (size_t index = 0; index < actual.size(); ++index) {
        if (std::fabs(actual[index] - expected[index]) > tolerance) {
            std::fprintf(stderr, "mismatch[%zu]: actual=%g expected=%g\n",
                         index, actual[index], expected[index]);
            return false;
        }
    }
    return true;
}

int main() {
    const Qwen4ExpFrontierMoeSpec spec{4, 5, 2, 3};
    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr, "CPU backend initializes");
    if (!backend) return 1;

    ggml_init_params params{};
    params.mem_size = 1024U * 1024U;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr, "weight metadata context initializes");
    if (!ctx) {
        ggml_backend_free(backend);
        return 1;
    }
    Qwen4ExpFrontierMoeWeights weights;
    weights.router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5);
    weights.experts_gate_up =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 6, 5);
    weights.experts_down =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 5);
    weights.shared_gate_input =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    weights.shared_gate =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    weights.shared_up =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    weights.shared_down =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr, "weight buffer allocates");
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> router(20), gate_up(120), down(60);
    std::vector<float> shared_gate_input = {0.07f, -0.04f, 0.02f, 0.09f};
    std::vector<float> shared_gate(12), shared_up(12), shared_down(12);
    for (size_t index = 0; index < router.size(); ++index)
        router[index] = 0.013f * static_cast<float>(static_cast<int>(index % 9) - 4);
    for (size_t index = 0; index < gate_up.size(); ++index)
        gate_up[index] = 0.009f * static_cast<float>(static_cast<int>(index % 13) - 6);
    for (size_t index = 0; index < down.size(); ++index)
        down[index] = 0.011f * static_cast<float>(static_cast<int>(index % 11) - 5);
    for (size_t index = 0; index < shared_gate.size(); ++index) {
        shared_gate[index] = 0.017f *
            static_cast<float>(static_cast<int>(index % 7) - 3);
        shared_up[index] = 0.014f *
            static_cast<float>(static_cast<int>(index % 5) - 2);
        shared_down[index] = 0.012f *
            static_cast<float>(static_cast<int>(index % 9) - 4);
    }
    ggml_backend_tensor_set(weights.router, router.data(), 0,
                            router.size() * sizeof(float));
    ggml_backend_tensor_set(weights.experts_gate_up, gate_up.data(), 0,
                            gate_up.size() * sizeof(float));
    ggml_backend_tensor_set(weights.experts_down, down.data(), 0,
                            down.size() * sizeof(float));
    ggml_backend_tensor_set(weights.shared_gate_input,
                            shared_gate_input.data(), 0,
                            shared_gate_input.size() * sizeof(float));
    ggml_backend_tensor_set(weights.shared_gate, shared_gate.data(), 0,
                            shared_gate.size() * sizeof(float));
    ggml_backend_tensor_set(weights.shared_up, shared_up.data(), 0,
                            shared_up.size() * sizeof(float));
    ggml_backend_tensor_set(weights.shared_down, shared_down.data(), 0,
                            shared_down.size() * sizeof(float));

    std::string error;
    Qwen4ExpFrontierMoeGraph * graph =
        dflash::common::qwen4exp_frontier_moe_create(
            backend, spec, weights, -1, error);
    if (!graph) std::fprintf(stderr, "frontier build error: %s\n", error.c_str());
    CHECK(graph != nullptr, "persistent frontier graph builds");
    if (graph) {
        const std::array<std::vector<float>, 2> inputs = {
            std::vector<float>{0.5f, -1.25f, 0.75f, 2.0f},
            std::vector<float>{-0.2f, 0.4f, 1.1f, -0.7f},
        };
        for (size_t sample = 0; sample < inputs.size(); ++sample) {
            std::vector<float> actual;
            const bool ok = dflash::common::qwen4exp_frontier_moe_eval(
                graph, inputs[sample].data(), inputs[sample].size(), actual,
                error);
            CHECK(ok, sample == 0 ? "first frontier evaluation succeeds" :
                                   "cached frontier evaluation succeeds");
            const std::vector<float> expected = reference_moe(
                spec, router, gate_up, down, shared_gate_input, shared_gate,
                shared_up, shared_down, inputs[sample]);
            CHECK(ok && close_vectors(actual, expected),
                  sample == 0 ? "fused graph matches scalar MoE reference" :
                                "reused graph matches scalar MoE reference");
        }
        std::vector<float> wrong;
        CHECK(!dflash::common::qwen4exp_frontier_moe_eval(
                  graph, inputs[0].data(), inputs[0].size() - 1, wrong, error),
              "frontier evaluation rejects a wrong input width");
    }

    dflash::common::qwen4exp_frontier_moe_destroy(graph);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::printf("qwen4exp frontier: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
