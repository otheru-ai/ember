// Differential validator for the complete heterogeneous DSpark provider.
//
// This deliberately loads only the ~11-GiB drafter, not the 85-GiB target.
// Synthetic captured features still traverse the trained GPU main projection;
// the resulting main/context-KV tensors feed the CPU/NPU provider, while the
// ordinary HIP drafter consumes the same raw features as the reference.  It is
// therefore the hidden-boundary gate that can run beside an idle resident
// target server on a 128-GiB Strix Halo.

#include "common/dspark_draft_compute_xdna.h"
#include "common/dspark_head.h"
#include "deepseek4/deepseek4_dspark.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Metrics {
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    double cosine = 0.0;
    size_t nonfinite = 0;
};

Metrics compare(const std::vector<float> & actual,
                const std::vector<float> & reference) {
    if (actual.size() != reference.size() || actual.empty())
        throw std::runtime_error("differential output shape mismatch");
    Metrics result;
    double sum_abs = 0.0;
    double dot = 0.0;
    double actual_sq = 0.0;
    double reference_sq = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float a = actual[index];
        const float r = reference[index];
        if (!std::isfinite(a) || !std::isfinite(r)) {
            ++result.nonfinite;
            continue;
        }
        const float error = std::fabs(a - r);
        result.max_abs = std::max(result.max_abs, error);
        sum_abs += error;
        dot += static_cast<double>(a) * r;
        actual_sq += static_cast<double>(a) * a;
        reference_sq += static_cast<double>(r) * r;
    }
    result.mean_abs = sum_abs / static_cast<double>(actual.size());
    if (actual_sq > 0.0 && reference_sq > 0.0)
        result.cosine = dot / std::sqrt(actual_sq * reference_sq);
    return result;
}

bool passes(const Metrics & metrics) {
    return metrics.nonfinite == 0 && metrics.max_abs <= 0.02f &&
           metrics.cosine >= 0.99999;
}

int parse_int(const char * value, int fallback, int minimum, int maximum,
              const char * label) {
    if (!value) return fallback;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < minimum || parsed > maximum)
        throw std::runtime_error(std::string(label) + " is outside its range");
    return static_cast<int>(parsed);
}

void fill(std::vector<float> & values, uint32_t salt, float scale) {
    uint32_t state = salt;
    for (float & value : values) {
        state = state * 1664525u + 1013904223u;
        value = (static_cast<float>((state >> 8) & 0xffffu) / 32768.0f -
                 1.0f) * scale;
    }
}

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

dflash::common::DSparkHeadWeights make_dspark_head_weights(
        const dflash::common::DSparkDrafter & drafter) {
    return {
        drafter.dspark_enabled,
        drafter.core.n_embd,
        drafter.markov_rank,
        drafter.vocab_size,
        drafter.confidence_dim,
        drafter.markov_w1,
        drafter.markov_w2,
        drafter.confidence_w,
        drafter.confidence_b,
    };
}

bool project_markov_chain(
        ggml_backend_t backend,
        const dflash::common::DSparkDrafter & drafter,
        const dflash::common::DSparkHeadWeights & weights,
        ggml_tensor * lm_head,
        const std::vector<float> & hidden,
        const std::vector<float> & confidence_hidden,
        std::vector<int32_t> & tokens,
        std::vector<float> & confidence) {
    const int n_embd = drafter.core.n_embd;
    const int q_len = drafter.block_size + 1;
    const size_t padded_elements = static_cast<size_t>(q_len) * n_embd;
    std::vector<float> padded_hidden(padded_elements, 0.0f);
    std::vector<float> padded_confidence(padded_elements, 0.0f);
    std::copy(hidden.begin(), hidden.end(),
              padded_hidden.begin() + n_embd);
    std::copy(confidence_hidden.begin(), confidence_hidden.end(),
              padded_confidence.begin() + n_embd);
    return dflash::common::dspark_markov_correct_greedy_chain_fused(
        weights, backend, lm_head, padded_hidden.data(), q_len,
        1, tokens, &confidence, padded_confidence.data());
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 8) {
        std::fprintf(stderr,
                     "usage: %s DRAFT_GGUF PROVIDER_SO ARTIFACT_DIR "
                     "TARGET_GGUF "
                     "[CTX_LEN] [REPEATS] [GPU_OVERLAP_REPEATS]\n",
                     argv[0]);
        return 2;
    }

    ggml_backend_t backend = nullptr;
    dflash::common::DSparkDrafter drafter;
    dflash::common::DeepSeek4Weights target_head;
    std::unique_ptr<dflash::common::XdnaDSparkDraftCompute> provider;
    try {
        const int ctx_len = parse_int(
            argc >= 6 ? argv[5] : nullptr, 16, 0, 128, "CTX_LEN");
        const int repeats = parse_int(
            argc >= 7 ? argv[6] : nullptr, 3, 1, 100, "REPEATS");
        const int gpu_overlap_repeats = parse_int(
            argc >= 8 ? argv[7] : nullptr, 0, 0, 32,
            "GPU_OVERLAP_REPEATS");
        if (setenv("DFLASH_DSPARK_XDNA_PLUGIN", argv[2], 1) != 0 ||
            setenv("EMBER_XDNA_ARTIFACT_DIR", argv[3], 1) != 0)
            throw std::runtime_error("cannot set provider environment");

        backend = ggml_backend_cuda_init(0);
        if (!backend) throw std::runtime_error("cannot initialize HIP backend");
        if (!dflash::common::load_deepseek4_dspark_drafter(
                argv[1], backend, drafter)) {
            throw std::runtime_error(std::string("drafter load failed: ") +
                dflash::common::deepseek4_dspark_last_error());
        }
        dflash::common::TargetLoadPlan target_plan;
        target_plan.layer_end = 0;
        target_plan.load_output = true;
        if (!dflash::common::load_deepseek4_gguf_partial(
                argv[4], backend, target_plan, target_head) ||
            !target_head.output) {
            throw std::runtime_error(
                "cannot load the target model's tied LM head");
        }
        if (target_head.n_embd != drafter.core.n_embd ||
            target_head.n_vocab != drafter.vocab_size) {
            throw std::runtime_error(
                "target LM head is incompatible with the drafter");
        }
        ggml_backend_synchronize(backend);
        if (!ggml_backend_cuda_buffer_is_managed(drafter.core.buf)) {
            throw std::runtime_error(
                "drafter weights are not managed UMA; provider cannot share them");
        }

        std::vector<ember_xdna_dspark_tensor_view_v1> views;
        for (ggml_tensor * tensor = ggml_get_first_tensor(drafter.core.ctx);
             tensor != nullptr;
             tensor = ggml_get_next_tensor(drafter.core.ctx, tensor)) {
            if (!tensor->data || !tensor->name[0]) continue;
            ember_xdna_dspark_tensor_view_v1 view{};
            view.abi_version = EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION;
            view.struct_size = sizeof(view);
            view.name = tensor->name;
            view.data = tensor->data;
            view.bytes = ggml_nbytes(tensor);
            view.type = static_cast<int32_t>(tensor->type);
            view.n_dims = GGML_MAX_DIMS;
            for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
                view.dims[dimension] = tensor->ne[dimension];
                view.strides[dimension] = tensor->nb[dimension];
            }
            views.push_back(view);
        }

        dflash::common::XdnaDSparkDraftConfig config;
        config.plugin_path = argv[2];
        config.draft_model_path = argv[1];
        config.n_embd = drafter.core.n_embd;
        config.n_target_layers = drafter.n_target_layers;
        config.block_size = drafter.block_size;
        config.n_swa = drafter.core.n_swa;
        config.head_dim = drafter.core.head_dim;
        config.weights_cpu_accessible = true;
        config.weight_views = views.data();
        config.weight_view_count = static_cast<uint32_t>(views.size());
        config.required = true;
        std::string error;
        provider = dflash::common::make_xdna_dspark_draft_compute(
            config, &error);
        if (!provider) throw std::runtime_error(error);

        const size_t hidden_elements =
            static_cast<size_t>(drafter.block_size) * drafter.core.n_embd;
        const size_t feature_elements = static_cast<size_t>(ctx_len) *
            drafter.n_target_layers * drafter.core.n_embd;
        std::vector<float> noise(hidden_elements);
        std::vector<float> features(feature_elements);
        constexpr int committed = 1024;

        std::vector<double> gpu_samples;
        std::vector<double> main_samples;
        std::vector<double> provider_samples;
        std::vector<double> heterogeneous_samples;
        std::vector<double> overlap_samples;
        bool overlap_deterministic = true;
        Metrics hidden_metrics;
        Metrics confidence_metrics;
        Metrics markov_confidence_metrics;
        std::vector<int32_t> gpu_tokens;
        std::vector<int32_t> provider_tokens;
        size_t markov_confidence_count = 0;
        size_t token_cases = 0;
        size_t token_mismatch_cases = 0;
        size_t token_mismatch_positions = 0;
        size_t token_common_prefix_total = 0;
        size_t token_common_prefix_min =
            std::numeric_limits<size_t>::max();
        bool token_match = true;
        bool pass = true;
        const dflash::common::DSparkHeadWeights head_weights =
            make_dspark_head_weights(drafter);

        // One unmeasured pass builds the HIP graphs and warms the resident AIE
        // overlay before the reported samples.
        for (int iteration = -1; iteration < repeats; ++iteration) {
            const uint32_t case_id = static_cast<uint32_t>(iteration + 2);
            fill(noise, 0x1f123bb5u ^ (case_id * 0x9e3779b9u), 0.25f);
            fill(features, 0x9a833f21u ^ (case_id * 0x85ebca6bu), 0.125f);
            std::vector<float> gpu_hidden;
            std::vector<float> gpu_confidence;
            const auto gpu_begin = Clock::now();
            if (!dflash::common::deepseek4_dspark_draft_forward(
                    backend, drafter, noise.data(),
                    ctx_len > 0 ? features.data() : nullptr,
                    ctx_len, committed, gpu_hidden, &gpu_confidence)) {
                throw std::runtime_error("HIP drafter reference failed");
            }
            const auto gpu_end = Clock::now();

            std::vector<float> main_context;
            std::vector<float> context_kv;
            const auto main_begin = Clock::now();
            if (ctx_len > 0 &&
                !dflash::common::deepseek4_dspark_project_main_context(
                    backend, drafter, features.data(), ctx_len,
                    main_context, &context_kv)) {
                throw std::runtime_error("HIP main/context-KV projection failed");
            }
            const auto main_end = Clock::now();

            dflash::common::XdnaDSparkDraftRequest request;
            request.committed = committed;
            request.ctx_len = ctx_len;
            request.noise_embed = noise.data();
            request.main_context = ctx_len > 0 ? main_context.data() : nullptr;
            request.context_kv = ctx_len > 0 ? context_kv.data() : nullptr;
            const auto provider_begin = Clock::now();
            auto job = provider->submit(request, &error);
            dflash::common::XdnaDSparkDraftOutput output;
            if (!job || !job->wait(output, &error))
                throw std::runtime_error(error);
            const auto provider_end = Clock::now();

            hidden_metrics = compare(output.hidden, gpu_hidden);
            confidence_metrics = compare(
                output.confidence_hidden, gpu_confidence);
            std::vector<float> gpu_markov_confidence;
            std::vector<float> provider_markov_confidence;
            if (!project_markov_chain(
                    backend, drafter, head_weights,
                    target_head.output,
                    gpu_hidden, gpu_confidence,
                    gpu_tokens, gpu_markov_confidence) ||
                !project_markov_chain(
                    backend, drafter, head_weights,
                    target_head.output,
                    output.hidden, output.confidence_hidden,
                    provider_tokens, provider_markov_confidence)) {
                throw std::runtime_error("DSpark Markov head comparison failed");
            }
            ++token_cases;
            size_t common_prefix = 0;
            const size_t common = std::min(
                gpu_tokens.size(), provider_tokens.size());
            while (common_prefix < common &&
                   gpu_tokens[common_prefix] == provider_tokens[common_prefix])
                ++common_prefix;
            token_common_prefix_total += common_prefix;
            token_common_prefix_min = std::min(
                token_common_prefix_min, common_prefix);
            if (gpu_tokens != provider_tokens) {
                ++token_mismatch_cases;
                for (size_t token = 0; token < common; ++token)
                    if (gpu_tokens[token] != provider_tokens[token])
                        ++token_mismatch_positions;
                token_mismatch_positions +=
                    std::max(gpu_tokens.size(), provider_tokens.size()) - common;
                if (std::getenv("DFLASH_DS4_DSPARK_DEBUG") ||
                    std::getenv("DFLASH_DS4_DSPARK_ROUTE_DEBUG")) {
                    std::fprintf(stderr,
                                 "[dspark-provider-case] case=%u gpu=",
                                 case_id);
                    for (size_t token = 0; token < gpu_tokens.size(); ++token)
                        std::fprintf(stderr, "%s%d", token ? "," : "",
                                     gpu_tokens[token]);
                    std::fprintf(stderr, " provider=");
                    for (size_t token = 0; token < provider_tokens.size(); ++token)
                        std::fprintf(stderr, "%s%d", token ? "," : "",
                                     provider_tokens[token]);
                    std::fputc('\n', stderr);
                }
            }
            token_match = token_match && gpu_tokens == provider_tokens;
            markov_confidence_metrics = compare(
                provider_markov_confidence, gpu_markov_confidence);
            markov_confidence_count = gpu_markov_confidence.size();
            pass = pass && passes(hidden_metrics) &&
                passes(confidence_metrics) && token_match;

            if (gpu_overlap_repeats > 0) {
                const auto overlap_begin = Clock::now();
                auto overlap_job = provider->submit(request, &error);
                std::vector<float> overlap_gpu_hidden;
                std::vector<float> overlap_gpu_confidence;
                for (int repeat = 0; repeat < gpu_overlap_repeats; ++repeat) {
                    if (!dflash::common::deepseek4_dspark_draft_forward(
                            backend, drafter, noise.data(),
                            ctx_len > 0 ? features.data() : nullptr,
                            ctx_len, committed, overlap_gpu_hidden,
                            &overlap_gpu_confidence)) {
                        throw std::runtime_error(
                            "overlap HIP drafter workload failed");
                    }
                }
                dflash::common::XdnaDSparkDraftOutput overlap_output;
                if (!overlap_job ||
                    !overlap_job->wait(overlap_output, &error)) {
                    throw std::runtime_error(error);
                }
                const auto overlap_end = Clock::now();
                overlap_deterministic = overlap_deterministic &&
                    overlap_output.hidden == output.hidden &&
                    overlap_output.confidence_hidden ==
                        output.confidence_hidden;
                if (iteration >= 0) {
                    overlap_samples.push_back(
                        milliseconds(overlap_begin, overlap_end));
                }
            }
            if (iteration >= 0) {
                gpu_samples.push_back(milliseconds(gpu_begin, gpu_end));
                main_samples.push_back(milliseconds(main_begin, main_end));
                provider_samples.push_back(
                    milliseconds(provider_begin, provider_end));
                heterogeneous_samples.push_back(
                    milliseconds(main_begin, provider_end));
            }
        }

        const auto mean = [](const std::vector<double> & samples) {
            return std::accumulate(samples.begin(), samples.end(), 0.0) /
                   static_cast<double>(samples.size());
        };
        std::printf(
            "dspark_provider_diff boundary=normalized_hidden n=%zu "
            "nonfinite=%zu max_abs=%.9g mean_abs=%.9g cosine=%.10f\n",
            hidden_elements, hidden_metrics.nonfinite,
            hidden_metrics.max_abs, hidden_metrics.mean_abs,
            hidden_metrics.cosine);
        std::printf(
            "dspark_provider_diff boundary=confidence_hidden n=%zu "
            "nonfinite=%zu max_abs=%.9g mean_abs=%.9g cosine=%.10f\n",
            hidden_elements, confidence_metrics.nonfinite,
            confidence_metrics.max_abs, confidence_metrics.mean_abs,
            confidence_metrics.cosine);
        std::printf(
            "dspark_provider_diff boundary=markov_head tokens=%zu "
            "cases=%zu mismatch_cases=%zu mismatch_positions=%zu "
            "common_prefix_mean=%.6f common_prefix_min=%zu token_match=%d "
            "confidence_n=%zu confidence_max_abs=%.9g "
            "confidence_mean_abs=%.9g confidence_cosine=%.10f\n",
            gpu_tokens.size(), token_cases, token_mismatch_cases,
            token_mismatch_positions,
            token_cases > 0
                ? static_cast<double>(token_common_prefix_total) /
                    static_cast<double>(token_cases)
                : 0.0,
            token_common_prefix_min == std::numeric_limits<size_t>::max()
                ? 0 : token_common_prefix_min,
            token_match ? 1 : 0,
            markov_confidence_count,
            markov_confidence_metrics.max_abs,
            markov_confidence_metrics.mean_abs,
            markov_confidence_metrics.cosine);
        std::printf(
            "dspark_provider_diff ctx=%d repeats=%d gpu_mean_ms=%.6f "
            "gpu_main_mean_ms=%.6f provider_mean_ms=%.6f "
            "heterogeneous_mean_ms=%.6f provider=%s result=%s\n",
            ctx_len, repeats, mean(gpu_samples), mean(main_samples),
            mean(provider_samples), mean(heterogeneous_samples),
            provider->name(), pass ? "PASS" : "FAIL");
        if (gpu_overlap_repeats > 0) {
            const double isolated_ms = mean(provider_samples) +
                static_cast<double>(gpu_overlap_repeats) * mean(gpu_samples);
            const double overlap_ms = mean(overlap_samples);
            std::printf(
                "dspark_provider_overlap gpu_repeats=%d mean_ms=%.6f "
                "isolated_sum_ms=%.6f speedup=%.6f deterministic=%d\n",
                gpu_overlap_repeats, overlap_ms, isolated_ms,
                overlap_ms > 0.0 ? isolated_ms / overlap_ms : 0.0,
                overlap_deterministic ? 1 : 0);
        }

        provider.reset();
        dflash::common::reset_dspark_head_runtime_cache();
        dflash::common::reset_deepseek4_dspark_runtime_cache();
        dflash::common::free_deepseek4_weights(target_head);
        dflash::common::free_deepseek4_dspark_drafter(drafter);
        ggml_backend_free(backend);
        return pass ? 0 : 1;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "DSpark provider differential failed: %s\n",
                     exception.what());
        provider.reset();
        dflash::common::reset_dspark_head_runtime_cache();
        dflash::common::reset_deepseek4_dspark_runtime_cache();
        dflash::common::free_deepseek4_weights(target_head);
        dflash::common::free_deepseek4_dspark_drafter(drafter);
        if (backend) ggml_backend_free(backend);
        return 1;
    }
}
