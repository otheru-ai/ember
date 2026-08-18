// Measure the CPU-resident control and reduction work around XDNA2's large
// DSpark projections. This intentionally excludes all AIE execution time; its
// result is the placement gate for HC, routing, RMSNorm, and MLA softmax.

#include "dspark_cpu_ops.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kTokens = 5;
constexpr int kEmbd = 4096;
constexpr int kHc = 4;
constexpr int kMix = 24;
constexpr int kHeads = 64;
constexpr int kHeadDim = 512;
constexpr int kContext = 128;
constexpr int kExperts = 256;
constexpr int kTopK = 6;

void fill(std::vector<float> & values, float scale, uint32_t salt) {
    uint32_t state = salt;
    for (float & value : values) {
        state = state * 1664525u + 1013904223u;
        value = (static_cast<float>((state >> 8) & 0xffffu) / 32768.0f - 1.0f) * scale;
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        int repeats = 5;
        if (argc == 2) {
            char * end = nullptr;
            const long parsed = std::strtol(argv[1], &end, 10);
            if (end == argv[1] || *end != '\0' || parsed < 1 || parsed > 100)
                throw std::runtime_error("repeats must be in [1,100]");
            repeats = static_cast<int>(parsed);
        } else if (argc != 1) {
            throw std::runtime_error("usage: ember-dspark-cpu-bench [REPEATS]");
        }

        std::vector<float> state(static_cast<size_t>(kTokens) * kHc * kEmbd);
        std::vector<float> hc_fn(static_cast<size_t>(kMix) * kHc * kEmbd);
        std::vector<float> hc_base(kMix);
        std::vector<float> norm(kEmbd, 1.0f);
        std::vector<float> router(static_cast<size_t>(kExperts) * kEmbd);
        std::vector<float> bias(kExperts);
        std::vector<float> q_a(static_cast<size_t>(kTokens) * 1024);
        std::vector<float> q(static_cast<size_t>(kTokens) * kHeads * kHeadDim);
        std::vector<float> kv(static_cast<size_t>(kContext + kTokens) * kHeadDim);
        std::vector<float> q_a_norm(1024, 1.0f);
        std::vector<float> head_norm(kHeadDim, 1.0f);
        std::vector<float> kv_norm(kHeadDim, 1.0f);
        std::vector<float> sinks(kHeads);
        std::vector<float> hc_out_fn(static_cast<size_t>(kHc) * kHc * kEmbd);
        std::vector<float> hc_out_base(kHc);
        fill(state, 0.25f, 1u); fill(hc_fn, 0.001f, 2u);
        fill(hc_base, 0.1f, 3u); fill(router, 0.01f, 4u);
        fill(bias, 0.01f, 5u); fill(q_a, 0.05f, 6u);
        fill(q, 0.05f, 7u); fill(kv, 0.05f, 8u);
        fill(sinks, 0.1f, 9u);
        fill(hc_out_fn, 0.001f, 10u); fill(hc_out_base, 0.1f, 11u);
        std::vector<int32_t> query_pos(kTokens), kv_pos(kContext + kTokens);
        for (int i = 0; i < kTokens; ++i) query_pos[static_cast<size_t>(i)] = 1000 + i;
        for (int i = 0; i < kContext + kTokens; ++i)
            kv_pos[static_cast<size_t>(i)] = 1000 - kContext + i;

        const float scales[3] = {0.5f, 0.5f, 0.5f};
        std::vector<double> samples;
        std::string error;
        for (int repeat = 0; repeat < repeats + 1; ++repeat) {
            std::vector<float> current = state;
            const auto begin = Clock::now();
            for (int layer = 0; layer < 3; ++layer) {
                for (int sublayer = 0; sublayer < 2; ++sublayer) {
                    std::vector<float> working, normalized, block;
                    ember::xdna2::DsparkHcSplit split;
                    if (!ember::xdna2::dspark_hc_pre(
                            current.data(), hc_fn.data(), hc_base.data(), scales,
                            kTokens, kEmbd, kHc, 20, 1.0e-6f,
                            working, split, &error) ||
                        !ember::xdna2::dspark_weighted_rms_norm(
                            working.data(), norm.data(), kTokens, kEmbd,
                            1.0e-6f, normalized, &error))
                        throw std::runtime_error(error);
                    if (sublayer == 0) {
                        std::vector<float> normalized_q_a;
                        std::vector<float> normalized_q;
                        std::vector<float> normalized_kv;
                        if (!ember::xdna2::dspark_weighted_rms_norm(
                                q_a.data(), q_a_norm.data(), kTokens, 1024,
                                1.0e-6f, normalized_q_a, &error) ||
                            !ember::xdna2::dspark_weighted_rms_norm(
                                q.data(), head_norm.data(), kTokens * kHeads,
                                kHeadDim, 1.0e-6f, normalized_q, &error) ||
                            !ember::xdna2::dspark_weighted_rms_norm(
                                kv.data(), kv_norm.data(), kContext + kTokens,
                                kHeadDim, 1.0e-6f, normalized_kv, &error))
                            throw std::runtime_error(error);
                        if (!ember::xdna2::dspark_attention_reduce(
                                normalized_q.data(), normalized_kv.data(),
                                sinks.data(),
                                query_pos.data(), kv_pos.data(), kTokens,
                                kContext, kHeads, kHeadDim, 64, 10000.0f,
                                block, &error))
                            throw std::runtime_error(error);
                        // The grouped output projections produce an embedding;
                        // preserve only their placement-equivalent shape here.
                        block.assign(normalized.begin(), normalized.end());
                    } else {
                        std::vector<int32_t> selected;
                        std::vector<float> weights;
                        if (!ember::xdna2::dspark_route_topk(
                                normalized.data(), router.data(), bias.data(),
                                kTokens, kEmbd, kExperts, kTopK, 1.5f,
                                selected, weights, &error))
                            throw std::runtime_error(error);
                        block.assign(normalized.begin(), normalized.end());
                    }
                    std::vector<float> next;
                    if (!ember::xdna2::dspark_hc_post(
                            current.data(), block.data(), split, kEmbd,
                            next, &error))
                        throw std::runtime_error(error);
                    current = std::move(next);
                }
            }
            std::vector<float> collapsed, final_hidden;
            if (!ember::xdna2::dspark_hc_out(
                    current.data(), hc_out_fn.data(), hc_out_base.data(), 0.5f,
                    kTokens, kEmbd, kHc, 1.0e-6f, collapsed, &error) ||
                !ember::xdna2::dspark_weighted_rms_norm(
                    collapsed.data(), norm.data(), kTokens, kEmbd, 1.0e-6f,
                    final_hidden, &error))
                throw std::runtime_error(error);
            const double milliseconds = std::chrono::duration<double, std::milli>(
                Clock::now() - begin).count();
            if (repeat > 0) samples.push_back(milliseconds);
        }
        const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                            static_cast<double>(samples.size());
        std::printf("dspark_cpu_body tokens=5 context=128 layers=3 repeats=%d mean_ms=%.6f\n",
                    repeats, mean);
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "DSpark CPU body benchmark failed: %s\n", exception.what());
        return 1;
    }
}
