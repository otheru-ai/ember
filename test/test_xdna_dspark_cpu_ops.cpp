#include "dspark_cpu_ops.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;
#define CHECK(condition, message) do { \
    if (condition) ++g_pass; \
    else { ++g_fail; std::printf("  FAIL: %s\n", message); } \
} while (0)

static bool near(float actual, float expected, float tolerance = 1.0e-5f) {
    return std::fabs(actual - expected) <= tolerance;
}

int main() {
    using namespace ember::xdna2;
    std::printf("ember XDNA DSpark CPU-op tests\n");
    std::string error;

    const float rms_input[] = {3.0f, 4.0f, 0.0f, 2.0f};
    const float rms_weight[] = {2.0f, 0.5f};
    std::vector<float> normalized;
    CHECK(dspark_weighted_rms_norm(rms_input, rms_weight, 2, 2, 0.0f,
                                   normalized, &error),
          "weighted RMSNorm accepts finite rows");
    CHECK(normalized.size() == 4 &&
              near(normalized[0], 3.0f * 2.0f / std::sqrt(12.5f)) &&
              near(normalized[1], 4.0f * 0.5f / std::sqrt(12.5f)) &&
              near(normalized[3], std::sqrt(2.0f) * 0.5f),
          "weighted RMSNorm matches rowwise definition");

    constexpr int tokens = 1;
    constexpr int embd = 2;
    constexpr int hc = 2;
    constexpr int hc_width = embd * hc;
    constexpr int mix_width = 2 * hc + hc * hc;
    const float state[hc_width] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> fn(static_cast<size_t>(mix_width) * hc_width, 0.0f);
    std::vector<float> base(mix_width, 0.0f);
    const float scales[3] = {1.0f, 1.0f, 1.0f};
    std::vector<float> working;
    DsparkHcSplit split;
    CHECK(dspark_hc_pre(state, fn.data(), base.data(), scales, tokens, embd,
                        hc, 3, 0.0f, working, split, &error),
          "HC pre accepts the two-stream fixture");
    CHECK(working.size() == 2 && near(working[0], 2.000004f, 2.0e-5f) &&
              near(working[1], 3.000006f, 2.0e-5f),
          "HC pre applies sigmoid stream gates");
    CHECK(split.values.size() == mix_width &&
              near(split.values[2], 1.0f) && near(split.values[3], 1.0f) &&
              near(split.values[4], 0.5f, 2.0e-5f),
          "HC split retains post gates and Sinkhorn matrix");

    const float block[] = {10.0f, 20.0f};
    std::vector<float> post;
    CHECK(dspark_hc_post(state, block, split, embd, post, &error),
          "HC post accepts HC-pre split");
    CHECK(post.size() == hc_width && near(post[0], 12.0f, 3.0e-5f) &&
              near(post[1], 23.0f, 3.0e-5f) &&
              near(post[2], 12.0f, 3.0e-5f) &&
              near(post[3], 23.0f, 3.0e-5f),
          "HC post combines residual streams and block output");

    std::vector<float> out_fn(static_cast<size_t>(hc) * hc_width, 0.0f);
    const float out_base[hc] = {0.0f, 0.0f};
    std::vector<float> collapsed;
    CHECK(dspark_hc_out(state, out_fn.data(), out_base, 1.0f, tokens, embd,
                        hc, 0.0f, collapsed, &error),
          "HC output accepts finite fixture");
    CHECK(collapsed.size() == 2 && near(collapsed[0], 2.000004f, 2.0e-5f) &&
              near(collapsed[1], 3.000006f, 2.0e-5f),
          "HC output uses trained sigmoid collapse rule");

    const float route_input[] = {1.0f, 0.0f};
    const float route_matrix[] = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f};
    const float route_bias[] = {0.0f, 2.0f, 0.0f};
    std::vector<int32_t> selected;
    std::vector<float> route_weights;
    CHECK(dspark_route_topk(route_input, route_matrix, route_bias, 1, 2, 3,
                            2, 1.5f, selected, route_weights, &error),
          "router accepts score-routed fixture");
    CHECK(selected.size() == 2 && selected[0] == 1 && selected[1] == 0,
          "selection bias changes top-k order only");
    CHECK(route_weights.size() == 2 &&
              near(route_weights[0] + route_weights[1], 1.5f),
          "unbiased selected probabilities normalize to expert scale");

    const float query[] = {1.0f, 0.0f};
    const float keys[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const int32_t query_pos[] = {0};
    const int32_t key_pos[] = {0, 0};
    std::vector<float> attention;
    CHECK(dspark_attention_reduce(query, keys, nullptr, query_pos, key_pos,
                                  1, 1, 1, 2, 2, 10000.0f, attention,
                                  &error),
          "attention reduction accepts one-head fixture");
    const float first = std::exp(1.0f / std::sqrt(2.0f));
    CHECK(attention.size() == 2 && near(attention[0], first / (first + 1.0f)) &&
              near(attention[1], 1.0f / (first + 1.0f)),
          "attention reduction matches full-visibility softmax");

    CHECK(!dspark_attention_reduce(query, keys, nullptr, query_pos, key_pos,
                                   1, 1, 1, 2, 1, 10000.0f, attention,
                                   &error) && error.find("invalid") != std::string::npos,
          "odd RoPE width fails closed");

    std::printf("──────────────────────────────\n  %d passed, %d failed\n",
                g_pass, g_fail);
    return g_fail ? 1 : 0;
}
