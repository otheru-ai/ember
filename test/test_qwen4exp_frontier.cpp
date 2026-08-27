#include "qwen4exp_frontier.h"
#include "qwen4exp_internal.h"
#include "qwen4exp_mtp.h"

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

static void test_causal_attention_stateless_ffn_batching() {
    constexpr size_t kRows = 7;
    constexpr size_t kLayers = 6;
    const std::array<float, kRows> seed = {
        0.2f, -0.7f, 1.1f, 0.4f, -0.3f, 0.8f, 0.05f};
    const auto attention = [](float input, float & recurrent,
                              size_t row, size_t layer) {
        // Represents a causal GDN/QSA update: only the same layer's prior rows
        // enter recurrent state. Row/layer terms stand in for M-RoPE position.
        recurrent = 0.73f * recurrent + input +
            0.01f * static_cast<float>(3 * row + layer);
        return input + 0.19f * recurrent;
    };
    const auto ffn = [](float input, size_t layer) {
        // Stateless per row, like routed/shared MoE after expert selection.
        return input + 0.11f * std::tanh(
            input + 0.03f * static_cast<float>(layer));
    };

    std::array<float, kLayers> token_state{};
    std::array<float, kRows> token_rows{};
    for (size_t row = 0; row < kRows; ++row) {
        float value = seed[row];
        for (size_t layer = 0; layer < kLayers; ++layer) {
            value = ffn(attention(value, token_state[layer], row, layer),
                        layer);
        }
        token_rows[row] = value;
    }

    std::array<float, kLayers> batch_state{};
    std::array<float, kRows> batch_rows = seed;
    for (size_t layer = 0; layer < kLayers; ++layer) {
        std::array<float, kRows> ffn_boundary{};
        for (size_t row = 0; row < kRows; ++row) {
            ffn_boundary[row] = attention(
                batch_rows[row], batch_state[layer], row, layer);
        }
        for (size_t row = 0; row < kRows; ++row) {
            batch_rows[row] = ffn(ffn_boundary[row], layer);
        }
    }
    CHECK(close_vectors(
              std::vector<float>(batch_rows.begin(), batch_rows.end()),
              std::vector<float>(token_rows.begin(), token_rows.end()),
              1.0e-6f),
          "batching only stateless FFN rows matches token-major outputs");
    CHECK(close_vectors(
              std::vector<float>(batch_state.begin(), batch_state.end()),
              std::vector<float>(token_state.begin(), token_state.end()),
              1.0e-6f),
          "causal layer state matches token-major snapshot frontier");
}

static void test_bounded_cache_and_prefill_policy() {
    using dflash::common::kQwen4ExpFrontierMoeCachedGraphsPerLayer;
    using dflash::common::kQwen4ExpFrontierMoeMaxBatch;
    using dflash::common::kQwen4ExpFrontierMoeMtpBatch;
    using dflash::common::qwen4exp_frontier_moe_cached_width;
    using dflash::common::qwen4exp_prefill_chunk_rows;

    CHECK(kQwen4ExpFrontierMoeCachedGraphsPerLayer == 3,
          "frontier cache has a fixed q1, MTP and prefill arena bound");
    CHECK(qwen4exp_frontier_moe_cached_width(1) == 1,
          "q1 retains its exact graph");
    bool bounded_remainders = true;
    for (int width = 2; width <= kQwen4ExpFrontierMoeMaxBatch; ++width) {
        const int expected = width <= kQwen4ExpFrontierMoeMtpBatch
            ? kQwen4ExpFrontierMoeMtpBatch
            : kQwen4ExpFrontierMoeMaxBatch;
        bounded_remainders = bounded_remainders &&
            qwen4exp_frontier_moe_cached_width(width) == expected;
    }
    CHECK(bounded_remainders,
          "every bounded remainder reuses q5 or q16 instead of a new arena");
    CHECK(qwen4exp_frontier_moe_cached_width(0) == 0 &&
              qwen4exp_frontier_moe_cached_width(17) == 0,
          "cache policy rejects widths outside the bounded contract");

    CHECK(qwen4exp_prefill_chunk_rows(40, 0, -1, false) == 16,
          "ordinary text prefill uses the maximum causal chunk");
    CHECK(qwen4exp_prefill_chunk_rows(16, 20, 25, false) == 5,
          "snapshot frontier is a hard chunk boundary");
    CHECK(qwen4exp_prefill_chunk_rows(16, 20, 21, false) == 1,
          "one-row snapshot frontier falls back to q1");
    CHECK(qwen4exp_prefill_chunk_rows(16, 20, -1, true) == 1,
          "force-exact prefill remains q1");
    CHECK(qwen4exp_prefill_chunk_rows(1, 20, -1, false) == 1 &&
              qwen4exp_prefill_chunk_rows(0, 20, -1, false) == 0,
          "vision and activation barriers cannot be crossed");

    // Deterministic miniature of the production schedule. The reference runs
    // every row q1. Ordinary prefill stops at a snapshot and batches only
    // stateless FFNs layer-major. Activation extraction remains wholly q1.
    constexpr size_t kRows = 7;
    constexpr size_t kLayers = 6;
    const std::array<float, kRows> seed = {
        0.2f, -0.7f, 1.1f, 0.4f, -0.3f, 0.8f, 0.05f};
    const auto attention = [](float input, float & recurrent,
                              size_t row, size_t layer) {
        recurrent = 0.73f * recurrent + input +
            0.01f * static_cast<float>(3 * row + layer);
        return input + 0.19f * recurrent;
    };
    const auto ffn = [](float input, size_t layer) {
        return input + 0.11f * std::tanh(
            input + 0.03f * static_cast<float>(layer));
    };
    const auto q1 = [&](float input, size_t row,
                        std::array<float, kLayers> & state,
                        std::vector<float> * capture) {
        float value = input;
        for (size_t layer = 0; layer < kLayers; ++layer) {
            if (capture) capture->push_back(value);
            value = ffn(attention(value, state[layer], row, layer), layer);
        }
        return value;
    };
    const auto batch = [&](std::array<float, kRows> & values, size_t first,
                           size_t count,
                           std::array<float, kLayers> & state) {
        for (size_t layer = 0; layer < kLayers; ++layer) {
            std::array<float, kRows> boundary{};
            for (size_t row = first; row < first + count; ++row) {
                boundary[row] = attention(values[row], state[layer], row,
                                          layer);
            }
            for (size_t row = first; row < first + count; ++row)
                values[row] = ffn(boundary[row], layer);
        }
    };

    std::array<float, kLayers> reference_state{};
    std::array<float, kLayers> reference_snapshot{};
    std::array<float, kRows> reference_rows{};
    std::vector<float> reference_capture;
    for (size_t row = 0; row < kRows; ++row) {
        reference_rows[row] = q1(seed[row], row, reference_state,
                                 row + 1 == kRows ? &reference_capture
                                                  : nullptr);
        if (row == 2) reference_snapshot = reference_state;
    }

    std::array<float, kLayers> candidate_state{};
    std::array<float, kRows> candidate_rows = seed;
    const size_t first_chunk = qwen4exp_prefill_chunk_rows(
        kRows, 0, 3, false);
    batch(candidate_rows, 0, first_chunk, candidate_state);
    const std::array<float, kLayers> candidate_snapshot = candidate_state;
    const size_t second_chunk = qwen4exp_prefill_chunk_rows(
        kRows - first_chunk, static_cast<int>(first_chunk), 3, false);
    batch(candidate_rows, first_chunk, second_chunk, candidate_state);

    std::array<float, kLayers> capture_state{};
    std::array<float, kRows> capture_rows{};
    std::vector<float> candidate_capture;
    for (size_t row = 0; row < kRows; ++row) {
        capture_rows[row] = q1(seed[row], row, capture_state,
                               row + 1 == kRows ? &candidate_capture
                                                : nullptr);
    }

    CHECK(close_vectors(
              std::vector<float>(candidate_rows.begin(), candidate_rows.end()),
              std::vector<float>(reference_rows.begin(), reference_rows.end()),
              1.0e-6f),
          "chunked prefill logits and raw HC match deterministic q1");
    CHECK(close_vectors(
              std::vector<float>(candidate_state.begin(),
                                 candidate_state.end()),
              std::vector<float>(reference_state.begin(),
                                 reference_state.end()),
              1.0e-6f),
          "chunked prefill final recurrent state matches deterministic q1");
    CHECK(close_vectors(
              std::vector<float>(candidate_snapshot.begin(),
                                 candidate_snapshot.end()),
              std::vector<float>(reference_snapshot.begin(),
                                 reference_snapshot.end()),
              1.0e-6f),
          "chunk boundary snapshot matches the q1 frontier");
    CHECK(candidate_capture.size() == kLayers &&
              close_vectors(candidate_capture, reference_capture, 1.0e-6f) &&
              close_vectors(
                  std::vector<float>(capture_rows.begin(), capture_rows.end()),
                  std::vector<float>(reference_rows.begin(),
                                     reference_rows.end()),
                  1.0e-6f),
          "activation extraction keeps the stock all-q1 layer records");
}

int main() {
    test_causal_attention_stateless_ffn_batching();
    test_bounded_cache_and_prefill_policy();
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
            backend, spec, weights, 48, error);
    if (!graph) std::fprintf(stderr, "frontier build error: %s\n", error.c_str());
    CHECK(graph != nullptr, "persistent frontier graph builds");
    if (graph) {
        dflash::common::Qwen4ExpMtpWeights mtp;
        mtp.frontier_moe = graph;
        const std::array<std::vector<float>, 2> inputs = {
            std::vector<float>{0.5f, -1.25f, 0.75f, 2.0f},
            std::vector<float>{-0.2f, 0.4f, 1.1f, -0.7f},
        };
        for (size_t sample = 0; sample < inputs.size(); ++sample) {
            std::vector<float> actual;
            const bool ok = dflash::common::qwen4exp_frontier_mtp_moe_q1(
                mtp, inputs[sample].data(), inputs[sample].size(), actual,
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
        CHECK(!dflash::common::qwen4exp_frontier_mtp_moe_q1(
                  mtp, inputs[0].data(), inputs[0].size() - 1, wrong, error),
              "MTP frontier evaluation rejects a wrong input width");
        dflash::common::qwen4exp_frontier_mtp_destroy(mtp);
        graph = nullptr;
        CHECK(mtp.frontier_moe == nullptr,
              "MTP companion destroys its graph before weight storage");
        CHECK(!dflash::common::qwen4exp_frontier_mtp_moe_q1(
                  mtp, inputs[0].data(), inputs[0].size(), wrong, error),
              "MTP execution fails closed without its GPU frontier graph");
    }

    dflash::common::qwen4exp_frontier_moe_destroy(graph);

    Qwen4ExpFrontierMoeGraph * batch_graph =
        dflash::common::qwen4exp_frontier_moe_create_batch(
            backend, spec, weights, -1, 3, error);
    if (!batch_graph)
        std::fprintf(stderr, "frontier batch build error: %s\n", error.c_str());
    CHECK(batch_graph != nullptr, "persistent three-row frontier graph builds");
    if (batch_graph) {
        const std::array<std::vector<float>, 3> rows = {
            std::vector<float>{0.5f, -1.25f, 0.75f, 2.0f},
            std::vector<float>{-0.2f, 0.4f, 1.1f, -0.7f},
            std::vector<float>{1.25f, 0.1f, -0.3f, 0.8f},
        };
        std::vector<float> batch_input;
        std::vector<float> expected;
        for (const std::vector<float> & row : rows) {
            batch_input.insert(batch_input.end(), row.begin(), row.end());
            const std::vector<float> reference = reference_moe(
                spec, router, gate_up, down, shared_gate_input, shared_gate,
                shared_up, shared_down, row);
            expected.insert(expected.end(), reference.begin(), reference.end());
        }
        std::vector<float> actual;
        const bool first_ok = dflash::common::qwen4exp_frontier_moe_eval(
            batch_graph, batch_input.data(), batch_input.size(), actual, error);
        CHECK(first_ok, "three-row frontier evaluation succeeds");
        CHECK(first_ok && close_vectors(actual, expected),
              "batched graph matches independent row-wise scalar reference");

        std::rotate(batch_input.begin(), batch_input.begin() + 4,
                    batch_input.end());
        expected.clear();
        for (size_t row = 0; row < rows.size(); ++row) {
            const std::vector<float> input_row(
                batch_input.begin() + static_cast<std::ptrdiff_t>(row * 4),
                batch_input.begin() + static_cast<std::ptrdiff_t>((row + 1) * 4));
            const std::vector<float> reference = reference_moe(
                spec, router, gate_up, down, shared_gate_input, shared_gate,
                shared_up, shared_down, input_row);
            expected.insert(expected.end(), reference.begin(), reference.end());
        }
        const bool reuse_ok = dflash::common::qwen4exp_frontier_moe_eval(
            batch_graph, batch_input.data(), batch_input.size(), actual, error);
        CHECK(reuse_ok, "cached three-row frontier evaluation succeeds");
        CHECK(reuse_ok && close_vectors(actual, expected),
              "reused batched graph preserves independent row semantics");
    }
    dflash::common::qwen4exp_frontier_moe_destroy(batch_graph);

    Qwen4ExpFrontierMoeGraph * wide_graph =
        dflash::common::qwen4exp_frontier_moe_create_batch(
            backend, spec, weights, -1,
            dflash::common::kQwen4ExpFrontierMoeMaxBatch, error);
    CHECK(wide_graph != nullptr, "maximum-width prefill frontier graph builds");
    if (wide_graph) {
        std::vector<float> wide_input;
        std::vector<float> wide_expected;
        for (int row = 0;
             row < dflash::common::kQwen4ExpFrontierMoeMaxBatch; ++row) {
            const std::vector<float> input_row = {
                0.1f * static_cast<float>(row + 1),
                -0.03f * static_cast<float>(row + 2),
                0.02f * static_cast<float>(row % 5),
                0.04f * static_cast<float>(7 - row % 7),
            };
            wide_input.insert(wide_input.end(), input_row.begin(),
                              input_row.end());
            const std::vector<float> reference = reference_moe(
                spec, router, gate_up, down, shared_gate_input, shared_gate,
                shared_up, shared_down, input_row);
            wide_expected.insert(wide_expected.end(), reference.begin(),
                                 reference.end());
        }
        std::vector<float> wide_actual;
        const bool wide_ok = dflash::common::qwen4exp_frontier_moe_eval(
            wide_graph, wide_input.data(), wide_input.size(), wide_actual,
            error);
        CHECK(wide_ok && close_vectors(wide_actual, wide_expected),
              "maximum-width graph preserves independent row semantics");

        std::vector<float> padded_input(
            static_cast<size_t>(dflash::common::kQwen4ExpFrontierMoeMaxBatch) *
                static_cast<size_t>(spec.n_embd),
            0.0f);
        std::vector<float> padded_expected;
        for (int row = 0; row < 3; ++row) {
            const std::vector<float> input_row = {
                0.17f * static_cast<float>(row + 1),
                -0.09f * static_cast<float>(row + 2),
                0.05f * static_cast<float>(row),
                0.11f * static_cast<float>(3 - row),
            };
            std::copy(input_row.begin(), input_row.end(),
                      padded_input.begin() +
                          static_cast<std::ptrdiff_t>(row * spec.n_embd));
            const std::vector<float> reference = reference_moe(
                spec, router, gate_up, down, shared_gate_input, shared_gate,
                shared_up, shared_down, input_row);
            padded_expected.insert(padded_expected.end(), reference.begin(),
                                   reference.end());
        }
        std::vector<float> padded_actual;
        const bool padded_ok = dflash::common::qwen4exp_frontier_moe_eval(
            wide_graph, padded_input.data(), padded_input.size(),
            padded_actual, error);
        padded_actual.resize(padded_expected.size());
        CHECK(padded_ok && close_vectors(padded_actual, padded_expected),
              "q16 zero padding preserves every real remainder row");
    }
    dflash::common::qwen4exp_frontier_moe_destroy(wide_graph);

    Qwen4ExpFrontierMoeGraph * oversized =
        dflash::common::qwen4exp_frontier_moe_create_batch(
            backend, spec, weights, -1,
            dflash::common::kQwen4ExpFrontierMoeMaxBatch + 1, error);
    CHECK(oversized == nullptr,
          "frontier rejects a batch wider than its configured bound");
    dflash::common::qwen4exp_frontier_moe_destroy(oversized);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::printf("qwen4exp frontier: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
