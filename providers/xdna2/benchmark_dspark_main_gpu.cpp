// Measure the GPU half of the heterogeneous DSpark placement without loading
// the 85 GiB target model. This uses the trained draft Q8_0 main projection and
// the same pre-RMSNorm -> matmul -> post-RMSNorm graph as
// deepseek4_dspark_project_main_context(). It is a diagnostic, not a runtime
// implementation; the production helper owns the cached generation-worker
// graph.

#include "q8_model_weights.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kInput = 12288;
constexpr int kOutput = 4096;

using Clock = std::chrono::steady_clock;

int parse_repeats(const char * value) {
    if (!value) return 20;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 2 || parsed > 1000)
        throw std::runtime_error("repeats must be in [2,1000]");
    return static_cast<int>(parsed);
}

void benchmark_shape(ggml_backend_t backend,
                     const std::vector<uint8_t> & raw_weight,
                     int rows,
                     int repeats) {
    std::vector<uint8_t> arena(4u * 1024 * 1024);
    ggml_init_params params{};
    params.mem_size = arena.size();
    params.mem_buffer = arena.data();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("ggml_init failed");

    ggml_tensor * weight = ggml_new_tensor_2d(
        ctx, GGML_TYPE_Q8_0, kInput, kOutput);
    ggml_tensor * input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, kInput, rows);
    ggml_tensor * norm_weight = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, kOutput);
    ggml_set_input(input);
    ggml_tensor * input_norm = ggml_rms_norm(ctx, input, 1.0e-6f);
    ggml_tensor * projected = ggml_mul_mat(ctx, weight, input_norm);
    ggml_tensor * output = ggml_mul(
        ctx, ggml_rms_norm(ctx, projected, 1.0e-6f), norm_weight);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        throw std::runtime_error("GPU tensor allocation failed");
    }
    if (ggml_nbytes(weight) != raw_weight.size()) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw std::runtime_error("trained Q8_0 byte count mismatch");
    }
    ggml_backend_tensor_set(
        weight, raw_weight.data(), 0, raw_weight.size());

    std::vector<float> input_values(
        static_cast<size_t>(rows) * kInput);
    uint32_t state = 0x7f4a7c15u;
    for (float & value : input_values) {
        state = state * 1664525u + 1013904223u;
        value = (static_cast<float>((state >> 8) & 0xffffu) / 32768.0f) -
                1.0f;
    }
    std::vector<float> norm_values(kOutput, 1.0f);
    ggml_backend_tensor_set(
        input, input_values.data(), 0,
        sizeof(float) * input_values.size());
    ggml_backend_tensor_set(
        norm_weight, norm_values.data(), 0,
        sizeof(float) * norm_values.size());

    for (int warmup = 0; warmup < 3; ++warmup) {
        if (ggml_backend_graph_compute(backend, graph) !=
            GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            throw std::runtime_error("GPU warmup failed");
        }
    }
    ggml_backend_synchronize(backend);

    std::vector<float> result(
        static_cast<size_t>(rows) * kOutput);
    std::vector<double> upload_samples;
    std::vector<double> compute_samples;
    std::vector<double> download_samples;
    std::vector<double> total_samples;
    upload_samples.reserve(static_cast<size_t>(repeats));
    compute_samples.reserve(static_cast<size_t>(repeats));
    download_samples.reserve(static_cast<size_t>(repeats));
    total_samples.reserve(static_cast<size_t>(repeats));
    for (int iteration = 0; iteration < repeats; ++iteration) {
        const auto begin = Clock::now();
        // Match the production helper: captured target features arrive in a
        // host vector on every proposal, while the weight and graph stay
        // resident. The output readback is the XRT provider handoff.
        ggml_backend_tensor_set(
            input, input_values.data(), 0,
            sizeof(float) * input_values.size());
        const auto uploaded = Clock::now();
        if (ggml_backend_graph_compute(backend, graph) !=
            GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            throw std::runtime_error("GPU benchmark compute failed");
        }
        ggml_backend_synchronize(backend);
        const auto computed = Clock::now();
        ggml_backend_tensor_get(
            output, result.data(), 0, sizeof(float) * result.size());
        const auto end = Clock::now();
        const auto milliseconds = [](auto first, auto last) {
            return std::chrono::duration<double, std::milli>(
                last - first).count();
        };
        upload_samples.push_back(milliseconds(begin, uploaded));
        compute_samples.push_back(milliseconds(uploaded, computed));
        download_samples.push_back(milliseconds(computed, end));
        total_samples.push_back(milliseconds(begin, end));
    }

    if (!std::all_of(result.begin(), result.end(),
                     [](float value) { return std::isfinite(value); })) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw std::runtime_error("GPU projection produced non-finite output");
    }

    std::sort(total_samples.begin(), total_samples.end());
    const auto mean = [](const std::vector<double> & samples) {
        return std::accumulate(samples.begin(), samples.end(), 0.0) /
               samples.size();
    };
    std::printf(
        "dspark_main_gpu M=%d K=%d N=%d repeats=%d "
        "total_min_ms=%.6f total_median_ms=%.6f total_mean_ms=%.6f "
        "upload_mean_ms=%.6f compute_mean_ms=%.6f "
        "download_mean_ms=%.6f\n",
        rows, kInput, kOutput, repeats, total_samples.front(),
        total_samples[total_samples.size() / 2], mean(total_samples),
        mean(upload_samples), mean(compute_samples), mean(download_samples));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s DRAFT_GGUF [REPEATS]\n", argv[0]);
        return 2;
    }
    try {
        const int repeats = parse_repeats(argc == 3 ? argv[2] : nullptr);
        ember::xdna2::Q8ModelProjection projection;
        std::string error;
        if (!ember::xdna2::load_q8_model_projection(
                argv[1], "dflash.fc.weight", kInput, kOutput,
                projection, &error)) {
            throw std::runtime_error(error);
        }
        // The AIE pack is produced by the shared loader for validation tools;
        // this diagnostic intentionally benchmarks the raw GGML Q8_0 tensor.
        projection.packed.clear();
        projection.packed.shrink_to_fit();

        ggml_backend_t backend = ggml_backend_cuda_init(0);
        if (!backend) throw std::runtime_error("cannot initialize HIP backend");
        benchmark_shape(backend, projection.raw, 4, repeats);
        benchmark_shape(backend, projection.raw, 128, repeats);
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "dspark main GPU benchmark failed: %s\n",
                     exception.what());
        return 1;
    }
}
