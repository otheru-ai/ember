#include "moe_expert_compute_xdna.h"

#include <cmath>
#include <cstdio>
#include <string>

using namespace dflash::common;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { std::fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
} while (0)

int main(int argc, char ** argv) {
    CHECK(argc == 2, "mock provider path supplied");
    if (argc != 2) return 1;

    XdnaMoeExpertComputeConfig config;
    config.plugin_path = argv[1];
    config.model_path = "mock.gguf";
    config.n_layer = 2;
    config.n_expert = 32;
    config.n_expert_used = 2;
    config.n_embd = 4;
    config.n_ff_exp = 8;
    config.swiglu_clamp = 7.0f;
    config.min_tokens = 2;

    std::string error;
    auto compute = make_xdna_moe_expert_compute(config, &error);
    CHECK(compute != nullptr, "provider loads");
    CHECK(error.empty(), "provider load has no error");
    if (!compute) return 1;
    CHECK(compute->healthy(), "provider reports healthy");
    CHECK(!compute->accepts(1, 2), "small decode batch stays on fallback");
    CHECK(compute->accepts(2, 2), "prefill batch is accepted");
    CHECK(!compute->failure_is_fatal(), "optional provider permits fallback");

    MoeExpertLayer layer;
    layer.layer_idx = 1;
    layer.cold_global_by_local = {3, 7, 11};
    const uint8_t gate[12] = {};
    const uint8_t up[12] = {};
    const uint8_t down[12] = {};
    layer.gate_data = gate;
    layer.up_data = up;
    layer.down_data = down;
    layer.gate_stride = 4;
    layer.up_stride = 4;
    layer.down_stride = 4;
    layer.gate_type = GGML_TYPE_Q2_0_ROCMFP2;
    layer.up_type = GGML_TYPE_Q2_0_ROCMFP2;
    layer.down_type = GGML_TYPE_Q2_0_ROCMFP2;
    layer.gate_scale = 0.5f;
    layer.up_scale = 0.25f;
    layer.down_scale = 2.0f;
    const float input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int32_t ids[4] = {0, 2, 1, 0};
    const float weights[4] = {0.25f, 0.75f, 0.5f, 0.5f};
    float output[8] = {};
    CHECK(compute->compute_batch(layer, input, ids, weights,
                                 2, 2, 4, 8, output),
          "batched expert compute succeeds");
    // Token 0: (global 3 + 1)*.25 + (global 11 + 1)*.75 = 10.
    // Token 1: (global 7 + 1)*.5 + (global 3 + 1)*.5 = 6.
    for (int i = 0; i < 4; ++i)
        CHECK(std::fabs(output[i] - (input[i] + 10.0f)) < 1e-6f,
              "token zero output matches");
    for (int i = 4; i < 8; ++i)
        CHECK(std::fabs(output[i] - (input[i] + 6.0f)) < 1e-6f,
              "token one output matches");

    XdnaMoeExpertComputeConfig q1_config = config;
    q1_config.min_tokens = 1;
    auto q1_compute = make_xdna_moe_expert_compute(q1_config, &error);
    CHECK(q1_compute && q1_compute->accepts(1, 2),
          "q1 hybrid path is accepted by default");
    const float q1_input[4] = {2, 4, 6, 8};
    const int32_t q1_ids[2] = {2, 0};
    const float q1_weights[2] = {0.5f, 0.5f};
    float q1_output[4] = {};
    CHECK(q1_compute && q1_compute->compute(
              layer, q1_input, q1_ids, q1_weights, 2, 4, 8, q1_output),
          "q1 expert compute succeeds");
    for (int i = 0; i < 4; ++i)
        CHECK(std::fabs(q1_output[i] - (q1_input[i] + 8.0f)) < 1e-6f,
              "q1 output matches");

    XdnaMoeExpertComputeConfig bad = config;
    bad.plugin_path += ".missing";
    error.clear();
    CHECK(!make_xdna_moe_expert_compute(bad, &error),
          "missing provider is rejected");
    CHECK(!error.empty(), "missing provider reports an error");

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
