// Hardware correctness validator for Ember's XDNA2 ROCMFP2 provider.
//
// This deliberately enters through the public provider ABI. It validates the
// packaged xclbins, XRT BO plumbing, weight pre-tiling, affine ROCMFP2 decode,
// generation-specific projection boundaries, clamped SwiGLU, routing, and the
// persistent weight cache without loading the full model.

#include "moe_expert_compute_xdna.h"
#include "rocmfp2_pack.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

uint16_t float_to_bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_round(float value) {
    const uint32_t bits = static_cast<uint32_t>(float_to_bf16_bits(value)) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float decode(const uint8_t * block, int index) {
    const uint8_t code = static_cast<uint8_t>(
        (block[index >> 2] >> (2 * (index & 3))) & 3u);
    return static_cast<float>(code) * ember::xdna2::ue4m3_to_float(block[8]) -
           ember::xdna2::ue4m3_to_float(block[9]);
}

void fill_projection(std::vector<uint8_t> & raw, int k, int n, uint32_t salt,
                     bool dense) {
    const int blocks_per_row = k / ember::xdna2::kRocmfp2BlockWeights;
    std::fill(raw.begin(), raw.end(), 0);
    for (int output = 0; output < n; ++output) {
        if (dense) {
            for (int block = 0; block < blocks_per_row; ++block) {
                uint8_t * q = raw.data() +
                    (static_cast<size_t>(output) *
                         static_cast<size_t>(blocks_per_row) +
                     static_cast<size_t>(block)) *
                        ember::xdna2::kRocmfp2BlockBytes;
                uint32_t state = salt ^
                    (static_cast<uint32_t>(output) * 0x9e3779b9u) ^
                    (static_cast<uint32_t>(block) * 0x85ebca6bu);
                for (int byte = 0; byte < 8; ++byte) {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    q[byte] = static_cast<uint8_t>(state >> 24);
                }
                q[8] = static_cast<uint8_t>(0x18u + (state & 7u));
                q[9] = static_cast<uint8_t>(0x20u + ((state >> 3) & 7u));
            }
            continue;
        }
        const uint32_t multiplier = ((salt >> 24) % 61u) | 1u;
        const int input = static_cast<int>(
            (static_cast<uint32_t>(output) * multiplier + (salt & 0xffffu)) %
            static_cast<uint32_t>(k));
        const int block = input / ember::xdna2::kRocmfp2BlockWeights;
        const int lane = input % ember::xdna2::kRocmfp2BlockWeights;
        uint8_t * q = raw.data() +
            (static_cast<size_t>(output) * static_cast<size_t>(blocks_per_row) +
             static_cast<size_t>(block)) * ember::xdna2::kRocmfp2BlockBytes;
        q[lane >> 2] = static_cast<uint8_t>(1u << (2 * (lane & 3)));
        q[8] = 0x40;  // UE4M3 1.0
    }
}

bool gemv_kernel_reference(const std::vector<uint8_t> & raw,
                           const std::vector<float> & input,
                           int k, int n, int generation,
                           std::vector<float> & output) {
    const size_t expected = ember::xdna2::rocmfp2_projection_bytes(k, n);
    if (raw.size() != expected || input.size() != static_cast<size_t>(k))
        return false;
    const int blocks_per_row = k / ember::xdna2::kRocmfp2BlockWeights;
    const int tiles = k / ember::xdna2::kGemvTileK;
    std::vector<float> input_bf16(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i)
        input_bf16[static_cast<size_t>(i)] = bf16_round(input[static_cast<size_t>(i)]);

    output.assign(static_cast<size_t>(n), 0.0f);
    for (int out = 0; out < n; ++out) {
        float accumulated = 0.0f;
        for (int tile = 0; tile < tiles; ++tile) {
            float sum = accumulated;
            for (int block = 0;
                 block < ember::xdna2::kGemvTileK /
                             ember::xdna2::kRocmfp2BlockWeights;
                 ++block) {
                const int global_block =
                    tile * (ember::xdna2::kGemvTileK /
                            ember::xdna2::kRocmfp2BlockWeights) + block;
                const uint8_t * q = raw.data() +
                    (static_cast<size_t>(out) * static_cast<size_t>(blocks_per_row) +
                     static_cast<size_t>(global_block)) *
                        ember::xdna2::kRocmfp2BlockBytes;
                for (int i = 0; i < ember::xdna2::kRocmfp2BlockWeights; ++i) {
                    const int input_index =
                        global_block * ember::xdna2::kRocmfp2BlockWeights + i;
                    sum += input_bf16[static_cast<size_t>(input_index)] * decode(q, i);
                }
            }
            accumulated = generation == 1 ? bf16_round(sum) : sum;
        }
        output[static_cast<size_t>(out)] =
            generation >= 3 ? accumulated : bf16_round(accumulated);
    }
    return true;
}

struct ErrorMetrics {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float max_actual_abs = 0.0f;
    float max_expected_abs = 0.0f;
    float actual_at_max = 0.0f;
    float expected_at_max = 0.0f;
    double mean_abs = 0.0;
    double rms = 0.0;
    double cosine = 0.0;
    int max_index = -1;
};

ErrorMetrics compare(const std::vector<float> & actual,
                     const std::vector<float> & expected) {
    ErrorMetrics metrics;
    double squared = 0.0;
    double absolute_sum = 0.0;
    double dot = 0.0;
    double actual_squared = 0.0;
    double expected_squared = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float absolute = std::fabs(actual[i] - expected[i]);
        const float relative = absolute / std::max(std::fabs(expected[i]), 1.0e-6f);
        squared += static_cast<double>(absolute) * static_cast<double>(absolute);
        absolute_sum += absolute;
        dot += static_cast<double>(actual[i]) * static_cast<double>(expected[i]);
        actual_squared += static_cast<double>(actual[i]) * static_cast<double>(actual[i]);
        expected_squared +=
            static_cast<double>(expected[i]) * static_cast<double>(expected[i]);
        metrics.max_actual_abs = std::max(metrics.max_actual_abs, std::fabs(actual[i]));
        metrics.max_expected_abs =
            std::max(metrics.max_expected_abs, std::fabs(expected[i]));
        if (absolute > metrics.max_abs) {
            metrics.max_abs = absolute;
            metrics.max_index = static_cast<int>(i);
            metrics.actual_at_max = actual[i];
            metrics.expected_at_max = expected[i];
        }
        metrics.max_rel = std::max(metrics.max_rel, relative);
    }
    metrics.mean_abs = absolute_sum / static_cast<double>(actual.size());
    metrics.rms = std::sqrt(squared / static_cast<double>(actual.size()));
    if (actual_squared > 0.0 && expected_squared > 0.0)
        metrics.cosine = dot / std::sqrt(actual_squared * expected_squared);
    return metrics;
}

double milliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main(int argc, char ** argv) {
    const char * plugin = argc > 1 ? argv[1] :
        "/usr/local/lib/ember/libember_xdna_moe.so";
    void * library = dlopen(plugin, RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    auto entry = reinterpret_cast<ember_xdna_moe_provider_entry_v1>(
        dlsym(library, EMBER_XDNA_MOE_PROVIDER_SYMBOL));
    if (!entry) {
        std::fprintf(stderr, "provider symbol missing\n");
        dlclose(library);
        return 1;
    }
    const ember_xdna_moe_provider_v1 * provider = entry();
    if (!provider || provider->abi_version != EMBER_XDNA_MOE_PROVIDER_ABI_VERSION ||
        provider->struct_size < sizeof(*provider)) {
        std::fprintf(stderr, "provider ABI mismatch\n");
        dlclose(library);
        return 1;
    }

    constexpr int n_embd = 4096;
    constexpr int n_ff = 2048;
    constexpr float clamp = 10.0f;
    constexpr float gate_scale = 0.75f;
    constexpr float up_scale = 0.625f;
    constexpr float down_scale = 0.5f;
    constexpr float route = 0.8f;
    int generation = 4;
    if (const char * raw = std::getenv("EMBER_XDNA_KERNEL_GEN")) {
        if (std::strcmp(raw, "1") == 0) generation = 1;
        else if (std::strcmp(raw, "3") == 0) generation = 3;
        else if (std::strcmp(raw, "4") == 0) generation = 4;
        else if (std::strcmp(raw, "2") != 0) {
            std::fprintf(stderr,
                         "EMBER_XDNA_KERNEL_GEN must be 1, 2, 3, or 4\n");
            dlclose(library);
            return 1;
        }
    }
    const bool dense = std::getenv("EMBER_XDNA_VALIDATION_DENSE") != nullptr;
    int selected_experts = 1;
    if (const char * raw = std::getenv("EMBER_XDNA_VALIDATION_EXPERTS")) {
        char * end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || *end != '\0' || parsed < 1 || parsed > 6) {
            std::fprintf(stderr,
                         "EMBER_XDNA_VALIDATION_EXPERTS must be in [1,6]\n");
            dlclose(library);
            return 1;
        }
        selected_experts = static_cast<int>(parsed);
    }

    ember_xdna_moe_config_v1 config{};
    config.abi_version = EMBER_XDNA_MOE_PROVIDER_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.n_layer = 1;
    config.n_expert = 256;
    config.n_expert_used = selected_experts;
    config.n_embd = n_embd;
    config.n_ff_exp = n_ff;
    config.swiglu_clamp = clamp;
    char error[512] = {};
    void * context = provider->create(&config, error, sizeof(error));
    if (!context) {
        std::fprintf(stderr, "provider create failed: %s\n", error);
        dlclose(library);
        return 1;
    }

    const size_t gate_bytes = ember::xdna2::rocmfp2_projection_bytes(n_embd, n_ff);
    const size_t down_bytes = ember::xdna2::rocmfp2_projection_bytes(n_ff, n_embd);
    std::vector<uint8_t> gate(gate_bytes), up(gate_bytes), down(down_bytes);
    fill_projection(gate, n_embd, n_ff, 0x13579bdfu, dense);
    fill_projection(up, n_embd, n_ff, 0x2468ace0u, dense);
    fill_projection(down, n_ff, n_embd, 0xdeadbeefu, dense);

    std::vector<float> input(n_embd);
    for (int i = 0; i < n_embd; ++i)
        input[static_cast<size_t>(i)] =
            static_cast<float>((i * 37) % 257 - 128) / 256.0f;

    const auto reference_start = std::chrono::steady_clock::now();
    std::vector<float> gate_out, up_out, hidden(n_ff), expected;
    if (!gemv_kernel_reference(gate, input, n_embd, n_ff,
                               generation, gate_out) ||
        !gemv_kernel_reference(up, input, n_embd, n_ff,
                               generation, up_out)) {
        std::fprintf(stderr, "reference gate/up failed\n");
        provider->destroy(context);
        dlclose(library);
        return 1;
    }
    for (int i = 0; i < n_ff; ++i) {
        float g = gate_out[static_cast<size_t>(i)] * gate_scale;
        float u = up_out[static_cast<size_t>(i)] * up_scale;
        g = std::min(g, clamp);
        u = std::max(-clamp, std::min(u, clamp));
        hidden[static_cast<size_t>(i)] = (g / (1.0f + std::exp(-g))) * u;
    }
    if (!gemv_kernel_reference(down, hidden, n_ff, n_embd,
                               generation, expected)) {
        std::fprintf(stderr, "reference down failed\n");
        provider->destroy(context);
        dlclose(library);
        return 1;
    }
    for (float & value : expected)
        value *= down_scale * route * static_cast<float>(selected_experts);
    const double reference_ms = milliseconds(reference_start);

    ember_xdna_moe_weight_view_v1 view{};
    view.struct_size = sizeof(view);
    view.gate = gate.data();
    view.gate_bytes = gate.size();
    view.up = up.data();
    view.up_bytes = up.size();
    view.down = down.data();
    view.down_bytes = down.size();
    view.gate_format = EMBER_XDNA_MOE_WEIGHT_ROCMFP2;
    view.up_format = EMBER_XDNA_MOE_WEIGHT_ROCMFP2;
    view.down_format = EMBER_XDNA_MOE_WEIGHT_ROCMFP2;
    view.gate_scale = gate_scale;
    view.up_scale = up_scale;
    view.down_scale = down_scale;

    std::vector<int32_t> expert_ids(static_cast<size_t>(selected_experts));
    std::vector<float> router_weights(static_cast<size_t>(selected_experts), route);
    std::vector<ember_xdna_moe_weight_view_v1> views(
        static_cast<size_t>(selected_experts), view);
    for (int slot = 0; slot < selected_experts; ++slot)
        expert_ids[static_cast<size_t>(slot)] = 17 + slot;
    std::vector<float> actual(n_embd, std::numeric_limits<float>::quiet_NaN());
    ember_xdna_moe_batch_v1 batch{};
    batch.abi_version = EMBER_XDNA_MOE_PROVIDER_ABI_VERSION;
    batch.struct_size = sizeof(batch);
    batch.n_tokens = 1;
    batch.n_selected = selected_experts;
    batch.n_embd = n_embd;
    batch.n_ff_exp = n_ff;
    batch.input = input.data();
    batch.expert_ids = expert_ids.data();
    batch.router_weights = router_weights.data();
    batch.output = actual.data();
    batch.expert_weights = views.data();

    const auto cold_start = std::chrono::steady_clock::now();
    const int cold_ok = provider->compute(context, &batch, error, sizeof(error));
    const double cold_ms = milliseconds(cold_start);
    if (!cold_ok) {
        std::fprintf(stderr, "cold provider compute failed: %s\n", error);
        provider->destroy(context);
        dlclose(library);
        return 1;
    }
    const ErrorMetrics cold = compare(actual, expected);

    std::fill(actual.begin(), actual.end(), std::numeric_limits<float>::quiet_NaN());
    error[0] = '\0';
    const auto warm_start = std::chrono::steady_clock::now();
    const int warm_ok = provider->compute(context, &batch, error, sizeof(error));
    const double warm_ms = milliseconds(warm_start);
    if (!warm_ok) {
        std::fprintf(stderr, "warm provider compute failed: %s\n", error);
        provider->destroy(context);
        dlclose(library);
        return 1;
    }
    const ErrorMetrics warm = compare(actual, expected);
    const bool finite = std::all_of(actual.begin(), actual.end(),
        [](float value) { return std::isfinite(value); });
    const bool pass = finite && cold.max_abs <= 1.0e-4f &&
                      warm.max_abs <= 1.0e-4f;

    std::printf("provider=%s generation=%d mode=%s experts=%d outputs=%d "
                "reference_ms=%.3f cold_ms=%.3f "
                "warm_ms=%.3f\n",
                provider->name ? provider->name : "(unnamed)", generation,
                dense ? "dense" : "structured", selected_experts, n_embd,
                reference_ms, cold_ms, warm_ms);
    std::printf("cold max_abs=%.9g max_rel=%.9g mean_abs=%.9g rms=%.9g "
                "cosine=%.9g actual_range=%.9g expected_range=%.9g "
                "index=%d actual=%.9g expected=%.9g\n",
                cold.max_abs, cold.max_rel, cold.mean_abs, cold.rms,
                cold.cosine, cold.max_actual_abs, cold.max_expected_abs,
                cold.max_index, cold.actual_at_max, cold.expected_at_max);
    std::printf("warm max_abs=%.9g max_rel=%.9g mean_abs=%.9g rms=%.9g "
                "cosine=%.9g actual_range=%.9g expected_range=%.9g "
                "index=%d actual=%.9g expected=%.9g\n",
                warm.max_abs, warm.max_rel, warm.mean_abs, warm.rms,
                warm.cosine, warm.max_actual_abs, warm.max_expected_abs,
                warm.max_index, warm.actual_at_max, warm.expected_at_max);
    std::printf("XDNA2_ROCMFP2_VALIDATION_%s\n", pass ? "PASS" : "FAIL");

    provider->destroy(context);
    dlclose(library);
    return pass ? 0 : 1;
}
