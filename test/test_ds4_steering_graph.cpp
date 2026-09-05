// Real ggml CPU execution of the production projection helper. This validates
// tensor broadcasting, per-token reductions and scale semantics; GPU model
// AR/speculative parity and behaviour require the separate hardware gate.
#include "../engine/dflash/deepseek4/deepseek4_steering.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace dflash::common;
static int failures;
static void check(bool ok, const char *message) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
}

static std::vector<float> run(int tokens, float scale, bool ffn) {
    constexpr int width = EMBER_STEERING_WIDTH;
    auto policy = std::make_shared<ember_directional_steering>();
    policy->attn_scale = ffn ? 0.0f : scale;
    policy->ffn_scale = ffn ? scale : 0.0f;
    policy->nonzero[39] = true;
    // Non-unit row proves there is no hidden normalization. Dyadic values
    // keep the expected projection exact and test sign/scaling independently.
    policy->rows[39][0] = 1.0f;
    policy->rows[39][1] = -0.5f;
    DeepSeek4Weights w;
    w.directional_steering = policy;
    ggml_init_params params{ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
                            nullptr, true};
    auto *ctx = ggml_init(params);
    auto backend = ggml_backend_cpu_init();
    auto *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, tokens);
    auto *direction = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    ggml_set_input(input);
    ggml_set_input(direction);
    w.steering_rows.resize(EMBER_STEERING_LAYERS, nullptr);
    w.steering_rows[39] = direction;
    check(ds4_directional_steering(ctx, input, w, 38, ffn) == input,
          "excluded layer adds no graph nodes");
    check(ds4_directional_steering(ctx, input, w, 39, !ffn) == input,
          "zero scale adds no graph nodes");
    DeepSeek4Weights drafter;
    drafter.n_layer = 3;
    check(ds4_directional_steering(ctx, input, drafter, 0, ffn) == input,
          "proposal model has no target direction policy");
    auto *output = ds4_directional_steering(ctx, input, w, 39, ffn);
    check((output == input) == (scale == 0), "active projection cannot be silently skipped");
    auto *graph = ggml_new_graph(ctx);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) std::abort();
    std::vector<float> values(static_cast<size_t>(width * tokens), 0.0f);
    for (int token = 0; token < tokens; ++token) {
        values[static_cast<size_t>(token * width)] = 4.0f + static_cast<float>(token);
        values[static_cast<size_t>(token * width + 1)] = 4.0f;
        values[static_cast<size_t>(token * width + 2)] = 7.0f;
    }
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    if (output != input) ggml_backend_tensor_set(direction, policy->rows[39], 0,
                                           sizeof(policy->rows[39]));
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
          "projection graph executes");
    std::vector<float> result(values.size());
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    for (int token = 0; token < tokens; ++token) {
        const size_t base = static_cast<size_t>(token * width);
        const float dot = 2.0f + static_cast<float>(token);
        check(result[base] == values[base] - scale * dot, "projected channel 0");
        check(result[base + 1] == values[base + 1] + 0.5f * scale * dot,
              "projected channel 1");
        check(result[base + 2] == 7.0f, "orthogonal component retained");
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return result;
}

int main() {
    for (float scale : {0.0f, 1.0f, -1.0f, 3.5f}) {
        for (bool ffn : {false, true}) {
            auto single = run(1, scale, ffn);
            for (int tokens : {2, 4, 17, 64}) {
                auto batch = run(tokens, scale, ffn);
                check(std::equal(single.begin(), single.end(), batch.begin()),
                      "first token independent of batch width on CPU");
            }
        }
    }
    std::printf("steering graph: %d failures\n", failures);
    return failures ? 1 : 0;
}
