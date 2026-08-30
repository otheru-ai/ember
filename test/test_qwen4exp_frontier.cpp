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
#include <cstdlib>
#include <cstring>
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

static float softplus(float value) {
    return value > 20.0f ? value : std::log1p(std::exp(value));
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

struct TinyGdnResult {
    std::vector<float> output;
    std::vector<float> conv;
    std::vector<float> recurrent;
};

static TinyGdnResult reference_gdn_q1(
        const dflash::common::Qwen4ExpFrontierGdnSpec & spec,
        const std::vector<float> & qkv_weight,
        const std::vector<float> & gate_weight,
        const std::vector<float> & alpha_weight,
        const std::vector<float> & beta_weight,
        const std::vector<float> & conv_weight,
        const std::vector<float> & a,
        const std::vector<float> & dt,
        const std::vector<float> & norm,
        const std::vector<float> & output_weight,
        const std::vector<float> & input,
        const std::vector<float> & conv_state,
        const std::vector<float> & recurrent_state) {
    const int channels =
        (2 * spec.n_key_heads + spec.n_heads) * spec.head_dim;
    const int core_values = spec.n_heads * spec.head_dim;
    std::vector<float> qkv = matvec(
        qkv_weight, channels, spec.n_embd, input);
    const std::vector<float> gate = matvec(
        gate_weight, core_values, spec.n_embd, input);
    const std::vector<float> alpha = matvec(
        alpha_weight, spec.n_heads, spec.n_embd, input);
    const std::vector<float> beta = matvec(
        beta_weight, spec.n_heads, spec.n_embd, input);

    std::vector<float> convolved(static_cast<size_t>(channels));
    for (int channel = 0; channel < channels; ++channel) {
        float value = conv_weight[static_cast<size_t>(
            channel * spec.conv_width + spec.conv_width - 1)] *
            qkv[static_cast<size_t>(channel)];
        for (int tap = 0; tap < spec.conv_width - 1; ++tap) {
            value += conv_weight[static_cast<size_t>(
                         channel * spec.conv_width + tap)] *
                     conv_state[static_cast<size_t>(tap * channels + channel)];
        }
        convolved[static_cast<size_t>(channel)] = silu(value);
    }

    TinyGdnResult result;
    result.conv.resize(conv_state.size());
    const size_t channels_size = static_cast<size_t>(channels);
    if (spec.conv_width > 2) {
        std::copy_n(conv_state.data() + channels_size,
                    static_cast<size_t>(spec.conv_width - 2) * channels_size,
                    result.conv.data());
    }
    std::copy(qkv.begin(), qkv.end(),
              result.conv.data() +
                  static_cast<size_t>(spec.conv_width - 2) * channels_size);
    result.recurrent = recurrent_state;
    std::vector<float> core(static_cast<size_t>(core_values));
    const int repeats = spec.n_heads / spec.n_key_heads;
    for (int head = 0; head < spec.n_heads; ++head) {
        const int key_head = head / repeats;
        std::vector<float> q(
            convolved.begin() + key_head * spec.head_dim,
            convolved.begin() + (key_head + 1) * spec.head_dim);
        std::vector<float> k(
            convolved.begin() + spec.n_key_heads * spec.head_dim +
                key_head * spec.head_dim,
            convolved.begin() + spec.n_key_heads * spec.head_dim +
                (key_head + 1) * spec.head_dim);
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (int index = 0; index < spec.head_dim; ++index) {
            q_sum += q[static_cast<size_t>(index)] *
                     q[static_cast<size_t>(index)];
            k_sum += k[static_cast<size_t>(index)] *
                     k[static_cast<size_t>(index)];
        }
        const float q_scale = 1.0f / std::sqrt(q_sum + spec.epsilon);
        const float k_scale = 1.0f / std::sqrt(k_sum + spec.epsilon);
        for (int index = 0; index < spec.head_dim; ++index) {
            q[static_cast<size_t>(index)] *= q_scale;
            k[static_cast<size_t>(index)] *= k_scale;
        }
        const float decay = std::exp(
            softplus(alpha[static_cast<size_t>(head)] +
                     dt[static_cast<size_t>(head)]) *
            a[static_cast<size_t>(head)]);
        const float mix = sigmoid(beta[static_cast<size_t>(head)]);
        float * recurrent = result.recurrent.data() +
            static_cast<size_t>(head * spec.head_dim * spec.head_dim);
        const float * value = convolved.data() +
            2 * spec.n_key_heads * spec.head_dim + head * spec.head_dim;
        for (int column = 0; column < spec.head_dim; ++column) {
            float dot = 0.0f;
            for (int row = 0; row < spec.head_dim; ++row) {
                recurrent[column * spec.head_dim + row] *= decay;
                dot += recurrent[column * spec.head_dim + row] *
                       k[static_cast<size_t>(row)];
            }
            const float delta = (value[column] - dot) * mix;
            float attention = 0.0f;
            for (int row = 0; row < spec.head_dim; ++row) {
                recurrent[column * spec.head_dim + row] +=
                    delta * k[static_cast<size_t>(row)];
                attention += recurrent[column * spec.head_dim + row] *
                             q[static_cast<size_t>(row)];
            }
            core[static_cast<size_t>(head * spec.head_dim + column)] =
                attention / std::sqrt(static_cast<float>(spec.head_dim));
        }
        float square_sum = 0.0f;
        for (int column = 0; column < spec.head_dim; ++column) {
            const float value_at = core[static_cast<size_t>(
                head * spec.head_dim + column)];
            square_sum += value_at * value_at;
        }
        const float rms_scale = 1.0f / std::sqrt(
            square_sum / static_cast<float>(spec.head_dim) + spec.epsilon);
        for (int column = 0; column < spec.head_dim; ++column) {
            const size_t offset = static_cast<size_t>(
                head * spec.head_dim + column);
            core[offset] *= rms_scale * norm[static_cast<size_t>(column)] *
                            sigmoid(gate[offset]);
        }
    }
    result.output = matvec(
        output_weight, spec.n_embd, core_values, core);
    return result;
}

// Double-precision reference for the GDN recurrent state alone.
//
// The batched path keeps the recurrent state in registers across the token
// loop; three sequential q1 steps round-trip it through memory between tokens.
// Those are different accumulation orders, so they cannot be bit-identical,
// and codex measured their first divergence at 1.1920929e-07 -- exactly 2^-23,
// one ULP of float32 near 1.0, which is the *floor* for two valid roundings
// rather than evidence of a defect.
//
// Which of them is right is not decidable by comparing them to each other.
// This computes the same recurrence in double so both can be measured against
// it, the way the rope oracle settled the same question for M-RoPE -- where
// the graph path turned out closer to exact than the host scalar it was being
// compared against.
//
// Only the recurrent state is reproduced here; that is where the divergence
// starts, and the output path adds arithmetic without adding evidence.
static std::vector<double> reference_gdn_recurrent_double(
        const dflash::common::Qwen4ExpFrontierGdnSpec & spec,
        const std::vector<float> & qkv_weight,
        const std::vector<float> & alpha_weight,
        const std::vector<float> & beta_weight,
        const std::vector<float> & conv_weight,
        const std::vector<float> & a,
        const std::vector<float> & dt,
        const std::vector<float> & input,
        const std::vector<double> & conv_state,
        const std::vector<double> & recurrent_state,
        std::vector<double> & next_conv_state) {
    const int channels =
        (2 * spec.n_key_heads + spec.n_heads) * spec.head_dim;
    auto matvec_d = [&](const std::vector<float> & weight, int rows,
                        int columns) {
        std::vector<double> out(static_cast<size_t>(rows), 0.0);
        for (int row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (int column = 0; column < columns; ++column) {
                sum += static_cast<double>(
                           weight[static_cast<size_t>(row * columns + column)]) *
                       static_cast<double>(input[static_cast<size_t>(column)]);
            }
            out[static_cast<size_t>(row)] = sum;
        }
        return out;
    };
    const std::vector<double> qkv = matvec_d(qkv_weight, channels, spec.n_embd);
    const std::vector<double> alpha =
        matvec_d(alpha_weight, spec.n_heads, spec.n_embd);
    const std::vector<double> beta =
        matvec_d(beta_weight, spec.n_heads, spec.n_embd);

    std::vector<double> convolved(static_cast<size_t>(channels));
    for (int channel = 0; channel < channels; ++channel) {
        double value = static_cast<double>(
                conv_weight[static_cast<size_t>(
                    channel * spec.conv_width + spec.conv_width - 1)]) *
            qkv[static_cast<size_t>(channel)];
        for (int tap = 0; tap < spec.conv_width - 1; ++tap) {
            value += static_cast<double>(
                         conv_weight[static_cast<size_t>(
                             channel * spec.conv_width + tap)]) *
                     conv_state[static_cast<size_t>(tap * channels + channel)];
        }
        convolved[static_cast<size_t>(channel)] = value / (1.0 + std::exp(-value));
    }

    const size_t channels_size = static_cast<size_t>(channels);
    next_conv_state.assign(conv_state.size(), 0.0);
    if (spec.conv_width > 2) {
        std::copy_n(conv_state.data() + channels_size,
                    static_cast<size_t>(spec.conv_width - 2) * channels_size,
                    next_conv_state.data());
    }
    std::copy(qkv.begin(), qkv.end(),
              next_conv_state.data() +
                  static_cast<size_t>(spec.conv_width - 2) * channels_size);

    std::vector<double> recurrent = recurrent_state;
    const int repeats = spec.n_heads / spec.n_key_heads;
    for (int head = 0; head < spec.n_heads; ++head) {
        const int key_head = head / repeats;
        std::vector<double> q(
            convolved.begin() + key_head * spec.head_dim,
            convolved.begin() + (key_head + 1) * spec.head_dim);
        std::vector<double> k(
            convolved.begin() + spec.n_key_heads * spec.head_dim +
                key_head * spec.head_dim,
            convolved.begin() + spec.n_key_heads * spec.head_dim +
                (key_head + 1) * spec.head_dim);
        double q_sum = 0.0;
        double k_sum = 0.0;
        for (int i = 0; i < spec.head_dim; ++i) {
            q_sum += q[static_cast<size_t>(i)] * q[static_cast<size_t>(i)];
            k_sum += k[static_cast<size_t>(i)] * k[static_cast<size_t>(i)];
        }
        const double q_scale = 1.0 / std::sqrt(q_sum + spec.epsilon);
        const double k_scale = 1.0 / std::sqrt(k_sum + spec.epsilon);
        for (int i = 0; i < spec.head_dim; ++i) {
            q[static_cast<size_t>(i)] *= q_scale;
            k[static_cast<size_t>(i)] *= k_scale;
        }
        const double sp = alpha[static_cast<size_t>(head)] +
                          static_cast<double>(dt[static_cast<size_t>(head)]);
        const double softplus_d = sp > 20.0 ? sp : std::log1p(std::exp(sp));
        const double decay = std::exp(
            softplus_d * static_cast<double>(a[static_cast<size_t>(head)]));
        const double bv = beta[static_cast<size_t>(head)];
        const double mix = 1.0 / (1.0 + std::exp(-bv));
        double * rec = recurrent.data() +
            static_cast<size_t>(head * spec.head_dim * spec.head_dim);
        const double * value = convolved.data() +
            2 * spec.n_key_heads * spec.head_dim + head * spec.head_dim;
        for (int column = 0; column < spec.head_dim; ++column) {
            double dot = 0.0;
            for (int row = 0; row < spec.head_dim; ++row) {
                rec[column * spec.head_dim + row] *= decay;
                dot += rec[column * spec.head_dim + row] *
                       k[static_cast<size_t>(row)];
            }
            const double delta = (value[column] - dot) * mix;
            for (int row = 0; row < spec.head_dim; ++row) {
                rec[column * spec.head_dim + row] +=
                    delta * k[static_cast<size_t>(row)];
            }
        }
    }
    return recurrent;
}

static std::vector<float> patterned_values(size_t count, float scale,
                                           int modulus) {
    std::vector<float> values(count);
    for (size_t index = 0; index < count; ++index) {
        values[index] = scale * static_cast<float>(
            static_cast<int>(index % static_cast<size_t>(modulus)) -
            modulus / 2);
    }
    return values;
}

static bool close_vectors(const std::vector<float> & actual,
                          const std::vector<float> & expected,
                          float tolerance = 2.0e-5f);

static void test_persistent_hc_mixer() {
    using dflash::common::Qwen4ExpFrontierHcSpec;
    const Qwen4ExpFrontierHcSpec spec{4, 3, 1.0e-6f};
    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr, "HC CPU backend initializes");
    if (!backend) return;
    ggml_init_params params{};
    params.mem_size = 128U * 1024U;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr, "HC weight context initializes");
    if (!ctx) {
        ggml_backend_free(backend);
        return;
    }
    ggml_tensor * norm = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 12);
    ggml_tensor * down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 12, 5);
    ggml_tensor * up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 12);
    ggml_tensor * inject = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 12, 3);
    ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 6);
    ggml_set_name(down, "tiny_hc_down");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr, "HC weight buffer allocates");
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }
    const std::vector<float> norm_f32 = patterned_values(12, 0.07f, 9);
    std::vector<ggml_bf16_t> norm_bf16(norm_f32.size());
    std::vector<float> decoded_norm(norm_f32.size());
    for (size_t index = 0; index < norm_f32.size(); ++index) {
        norm_bf16[index] = ggml_fp32_to_bf16(norm_f32[index]);
        decoded_norm[index] = ggml_bf16_to_fp32(norm_bf16[index]);
    }
    const std::vector<float> down_weight = patterned_values(60, 0.019f, 11);
    const std::vector<float> up_weight = patterned_values(60, 0.017f, 13);
    const std::vector<float> inject_weight =
        patterned_values(36, 0.023f, 7);
    const std::vector<float> output_weight =
        patterned_values(24, 0.029f, 11);
    ggml_backend_tensor_set(norm, norm_bf16.data(), 0,
                            norm_bf16.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(down, down_weight.data(), 0,
                            down_weight.size() * sizeof(float));
    ggml_backend_tensor_set(up, up_weight.data(), 0,
                            up_weight.size() * sizeof(float));
    ggml_backend_tensor_set(inject, inject_weight.data(), 0,
                            inject_weight.size() * sizeof(float));
    ggml_backend_tensor_set(output, output_weight.data(), 0,
                            output_weight.size() * sizeof(float));

    std::vector<float> input = patterned_values(36, 0.031f, 17);
    std::vector<float> expected_mixed;
    std::vector<float> expected_injection;
    for (int token = 0; token < 3; ++token) {
        std::vector<float> normalized(
            input.begin() + token * 12, input.begin() + (token + 1) * 12);
        for (int stream = 0; stream < 3; ++stream) {
            float sum = 0.0f;
            for (int channel = 0; channel < 4; ++channel) {
                const float value = normalized[static_cast<size_t>(
                    stream * 4 + channel)];
                sum += value * value;
            }
            const float scale = 1.0f / std::sqrt(sum / 4.0f + spec.epsilon);
            for (int channel = 0; channel < 4; ++channel) {
                const size_t index = static_cast<size_t>(
                    stream * 4 + channel);
                normalized[index] *= scale * decoded_norm[index];
            }
        }
        std::vector<float> low = matvec(down_weight, 5, 12, normalized);
        for (float & value : low) value = silu(value / 3.0f);
        std::vector<float> gate = matvec(up_weight, 12, 5, low);
        for (float & value : gate) value = sigmoid(value);
        for (int channel = 0; channel < 4; ++channel) {
            float mixed = 0.0f;
            for (int stream = 0; stream < 3; ++stream) {
                const size_t index = static_cast<size_t>(
                    stream * 4 + channel);
                mixed += normalized[index] * gate[index] / 3.0f;
            }
            expected_mixed.push_back(mixed);
        }
        const std::vector<float> row_injection =
            matvec(inject_weight, 3, 12, normalized);
        expected_injection.insert(expected_injection.end(),
                                  row_injection.begin(), row_injection.end());
    }

    dflash::common::Qwen4ExpFrontierDenseCache * cache =
        dflash::common::qwen4exp_frontier_dense_cache_create();
    std::vector<float> mixed, injection_values;
    std::string error;
    const bool batch_ok = dflash::common::qwen4exp_frontier_hc_eval(
        cache, backend, spec, norm, down, up, inject, input.data(),
        input.size(), 3, mixed, &injection_values, error);
    CHECK(batch_ok && close_vectors(mixed, expected_mixed, 3.0e-5f) &&
              close_vectors(injection_values, expected_injection, 3.0e-5f),
          "persistent q5 HC graph matches scalar normalization and gating");
    CHECK(dflash::common::qwen4exp_frontier_hc_graph_count(cache) == 1U,
          "three HC rows allocate one bounded q5 graph");
    std::vector<float> q1_mixed, q1_injection;
    const bool q1_ok = dflash::common::qwen4exp_frontier_hc_eval(
        cache, backend, spec, norm, down, up, inject, input.data(), 12, 1,
        q1_mixed, &q1_injection, error);
    CHECK(q1_ok && close_vectors(
              q1_mixed,
              std::vector<float>(expected_mixed.begin(),
                                 expected_mixed.begin() + 4), 3.0e-5f) &&
              dflash::common::qwen4exp_frontier_hc_graph_count(cache) == 2U,
          "persistent q1 HC graph preserves the decode row and cache bound");
    std::vector<float> expected_output;
    for (int token = 0; token < 3; ++token) {
        const std::vector<float> row(
            expected_mixed.begin() + token * 4,
            expected_mixed.begin() + (token + 1) * 4);
        const std::vector<float> projected =
            matvec(output_weight, 6, 4, row);
        expected_output.insert(expected_output.end(), projected.begin(),
                               projected.end());
    }
    std::vector<float> actual_output;
    const bool output_ok =
        dflash::common::qwen4exp_frontier_hc_output_eval(
            cache, backend, spec, norm, down, up, output, input.data(),
            input.size(), 3, actual_output, error);
    CHECK(output_ok && close_vectors(actual_output, expected_output, 3.0e-5f),
          "persistent HC output graph matches scalar mixer and projection");
    CHECK(dflash::common::qwen4exp_frontier_hc_graph_count(cache) == 3U,
          "final projection owns one additional bounded HC graph");
    dflash::common::qwen4exp_frontier_dense_cache_destroy(cache);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void test_persistent_gdn_q1() {
    using dflash::common::Qwen4ExpFrontierGdnGraph;
    using dflash::common::Qwen4ExpFrontierGdnSpec;
    using dflash::common::Qwen4ExpFrontierGdnWeights;
    // Two key heads repeated three times distinguish Qwen's repeat-interleave
    // mapping from the generic modulo broadcast that would be wrong here.
    const Qwen4ExpFrontierGdnSpec spec{4, 6, 2, 4, 4, 1.0e-6f};
    const int channels = 40;
    const int core_values = 24;

    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr, "GDN CPU backend initializes");
    if (!backend) return;
    ggml_init_params params{};
    params.mem_size = 256U * 1024U;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr, "GDN weight context initializes");
    if (!ctx) {
        ggml_backend_free(backend);
        return;
    }
    Qwen4ExpFrontierGdnWeights weights;
    weights.qkv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, channels);
    weights.gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, core_values);
    weights.alpha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 6);
    weights.beta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 6);
    weights.conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, channels);
    weights.a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6);
    weights.dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6);
    weights.norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    weights.output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, core_values, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr, "GDN weight buffer allocates");
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    const std::vector<float> qkv_weight = patterned_values(160, 0.017f, 11);
    const std::vector<float> gate_weight = patterned_values(96, 0.013f, 9);
    const std::vector<float> alpha_weight = patterned_values(24, 0.019f, 7);
    const std::vector<float> beta_weight = patterned_values(24, 0.023f, 5);
    const std::vector<float> conv_weight = patterned_values(160, 0.021f, 13);
    const std::vector<float> a{
        -0.31f, -0.27f, -0.19f, -0.23f, -0.29f, -0.17f};
    const std::vector<float> dt{
        0.07f, -0.04f, 0.11f, 0.03f, -0.08f, 0.05f};
    const std::vector<float> norm{0.91f, 1.07f, 0.96f, 1.03f};
    const std::vector<float> output_weight = patterned_values(96, 0.015f, 9);
    const std::array<std::pair<ggml_tensor *, const std::vector<float> *>, 9>
        uploads{{
            {weights.qkv, &qkv_weight},
            {weights.gate, &gate_weight},
            {weights.alpha, &alpha_weight},
            {weights.beta, &beta_weight},
            {weights.conv, &conv_weight},
            {weights.a, &a},
            {weights.dt, &dt},
            {weights.norm, &norm},
            {weights.output, &output_weight},
        }};
    for (const auto & upload : uploads) {
        ggml_backend_tensor_set(upload.first, upload.second->data(), 0,
                                upload.second->size() * sizeof(float));
    }

    std::string error;
    Qwen4ExpFrontierGdnGraph * graph =
        dflash::common::qwen4exp_frontier_gdn_create_q1(
            backend, spec, weights, -1, error);
    if (!graph)
        std::fprintf(stderr, "GDN frontier build error: %s\n", error.c_str());
    CHECK(graph != nullptr, "persistent fused q1 GDN graph builds");
    CHECK(dflash::common::qwen4exp_frontier_gdn_sqr_inputs_contiguous(graph),
          "q1 GDN materializes strided Q/K before HIP unary kernels");
    CHECK(dflash::common::qwen4exp_frontier_gdn_state_transfer_bytes_q1(
              spec) == 1408U,
          "q1 GDN synchronized host-state boundary is explicitly bounded");
    if (graph) {
        std::vector<float> conv_state = patterned_values(120, 0.009f, 7);
        std::vector<float> recurrent_state =
            patterned_values(96, 0.004f, 9);
        const std::array<std::vector<float>, 2> inputs{{
            {0.35f, -0.28f, 0.17f, 0.42f},
            {-0.11f, 0.26f, 0.39f, -0.31f},
        }};
        for (size_t step = 0; step < inputs.size(); ++step) {
            const TinyGdnResult expected = reference_gdn_q1(
                spec, qkv_weight, gate_weight, alpha_weight, beta_weight,
                conv_weight, a, dt, norm, output_weight, inputs[step],
                conv_state, recurrent_state);
            std::vector<float> output, next_conv, next_recurrent;
            const bool ok = dflash::common::qwen4exp_frontier_gdn_eval_q1(
                graph, inputs[step].data(), inputs[step].size(),
                conv_state.data(), conv_state.size(), recurrent_state.data(),
                recurrent_state.size(), output, next_conv, next_recurrent,
                error);
            CHECK(ok && close_vectors(output, expected.output, 2.0e-5f),
                  step == 0 ? "fused GDN output matches scalar oracle" :
                              "reused fused GDN output matches scalar oracle");
            CHECK(ok && close_vectors(next_conv, expected.conv, 2.0e-5f),
                  "fused GDN advances causal convolution state exactly");
            CHECK(ok && close_vectors(next_recurrent, expected.recurrent,
                                      2.0e-5f),
                  "fused GDN recurrent state matches scalar oracle");
            conv_state = next_conv;
            recurrent_state = next_recurrent;
        }

        const std::vector<float> saved_conv = conv_state;
        const std::vector<float> saved_recurrent = recurrent_state;
        const std::vector<float> branch_input{0.23f, 0.05f, -0.37f, 0.19f};
        std::vector<float> first_output, first_conv, first_recurrent;
        std::vector<float> replay_output, replay_conv, replay_recurrent;
        const bool first_ok = dflash::common::qwen4exp_frontier_gdn_eval_q1(
            graph, branch_input.data(), branch_input.size(), saved_conv.data(),
            saved_conv.size(), saved_recurrent.data(), saved_recurrent.size(),
            first_output, first_conv, first_recurrent, error);
        const bool replay_ok = dflash::common::qwen4exp_frontier_gdn_eval_q1(
            graph, branch_input.data(), branch_input.size(), saved_conv.data(),
            saved_conv.size(), saved_recurrent.data(), saved_recurrent.size(),
            replay_output, replay_conv, replay_recurrent, error);
        CHECK(first_ok && replay_ok &&
                  close_vectors(first_output, replay_output, 0.0f) &&
                  close_vectors(first_conv, replay_conv, 0.0f) &&
                  close_vectors(first_recurrent, replay_recurrent, 0.0f),
              "saved host GDN state replays a rejected branch exactly");
        CHECK(!dflash::common::qwen4exp_frontier_gdn_eval_q1(
                  graph, branch_input.data(), branch_input.size(),
                  saved_conv.data(), saved_conv.size() - 1U,
                  saved_recurrent.data(), saved_recurrent.size(),
                  replay_output, replay_conv, replay_recurrent, error),
              "GDN graph rejects an incomplete snapshot state");
    }
    dflash::common::qwen4exp_frontier_gdn_destroy(graph);

    Qwen4ExpFrontierGdnGraph * batch_graph =
        dflash::common::qwen4exp_frontier_gdn_create_batch(
            backend, spec, weights, -1, 3, error);
    if (!batch_graph)
        std::fprintf(stderr, "GDN batch build error: %s\n", error.c_str());
    CHECK(batch_graph != nullptr,
          "persistent fused three-row causal GDN graph builds");
    CHECK(dflash::common::qwen4exp_frontier_gdn_sqr_inputs_contiguous(
              batch_graph),
          "batched GDN materializes strided Q/K before HIP unary kernels");
    CHECK(dflash::common::qwen4exp_frontier_gdn_state_transfer_bytes_batch(
              spec, 3) == 1728U,
          "batched GDN transfers one initial and one final recurrent state");
    if (batch_graph) {
        const std::array<std::vector<float>, 3> batch_inputs{{
            {0.35f, -0.28f, 0.17f, 0.42f},
            {-0.11f, 0.26f, 0.39f, -0.31f},
            {0.23f, 0.05f, -0.37f, 0.19f},
        }};
        std::vector<float> batch_input;
        std::vector<float> expected_output;
        std::vector<float> expected_conv =
            patterned_values(120, 0.009f, 7);
        std::vector<float> expected_recurrent =
            patterned_values(96, 0.004f, 9);
        const std::vector<float> initial_conv = expected_conv;
        const std::vector<float> initial_recurrent = expected_recurrent;
        for (const std::vector<float> & row : batch_inputs) {
            batch_input.insert(batch_input.end(), row.begin(), row.end());
            const TinyGdnResult expected = reference_gdn_q1(
                spec, qkv_weight, gate_weight, alpha_weight, beta_weight,
                conv_weight, a, dt, norm, output_weight, row, expected_conv,
                expected_recurrent);
            expected_output.insert(expected_output.end(),
                                   expected.output.begin(),
                                   expected.output.end());
            expected_conv = expected.conv;
            expected_recurrent = expected.recurrent;
        }
        std::vector<float> actual_output, actual_conv, actual_recurrent;
        const bool batch_ok =
            dflash::common::qwen4exp_frontier_gdn_eval_batch(
                batch_graph, batch_input.data(), batch_input.size(),
                initial_conv.data(), initial_conv.size(),
                initial_recurrent.data(), initial_recurrent.size(),
                actual_output, actual_conv, actual_recurrent, error);
        CHECK(batch_ok &&
                  close_vectors(actual_output, expected_output, 2.0e-5f),
              "batched GDN outputs match three sequential scalar rows");
        CHECK(batch_ok && close_vectors(actual_conv, expected_conv, 2.0e-5f),
              "batched GDN causal convolution frontier matches scalar rows");
        CHECK(batch_ok &&
                  close_vectors(actual_recurrent, expected_recurrent,
                                2.0e-5f),
              "batched GDN final recurrent state matches scalar rows");
        std::vector<float> replay_output, replay_conv, replay_recurrent;
        const bool replay_ok =
            dflash::common::qwen4exp_frontier_gdn_eval_batch(
                batch_graph, batch_input.data(), batch_input.size(),
                initial_conv.data(), initial_conv.size(),
                initial_recurrent.data(), initial_recurrent.size(),
                replay_output, replay_conv, replay_recurrent, error);
        CHECK(batch_ok && replay_ok &&
                  close_vectors(actual_output, replay_output, 0.0f) &&
                  close_vectors(actual_conv, replay_conv, 0.0f) &&
                  close_vectors(actual_recurrent, replay_recurrent, 0.0f),
              "saved GDN state replays a whole rejected batch exactly");
    }
    dflash::common::qwen4exp_frontier_gdn_destroy(batch_graph);

    Qwen4ExpFrontierGdnGraph * wide_gdn =
        dflash::common::qwen4exp_frontier_gdn_create_batch(
            backend, spec, weights, -1, 16, error);
    CHECK(wide_gdn != nullptr,
          "persistent q16 GDN graph builds for ordinary prefill chunks");
    if (wide_gdn) {
        std::vector<float> wide_input;
        std::vector<float> wide_expected;
        std::vector<float> expected_conv =
            patterned_values(120, 0.009f, 7);
        std::vector<float> expected_recurrent =
            patterned_values(96, 0.004f, 9);
        const std::vector<float> initial_conv = expected_conv;
        const std::vector<float> initial_recurrent = expected_recurrent;
        for (int row = 0; row < 16; ++row) {
            const std::vector<float> input{
                0.03f * static_cast<float>(row + 1),
                -0.02f * static_cast<float>(row % 5),
                0.04f * static_cast<float>(3 - row % 7),
                0.01f * static_cast<float>(row - 8),
            };
            wide_input.insert(wide_input.end(), input.begin(), input.end());
            const TinyGdnResult expected = reference_gdn_q1(
                spec, qkv_weight, gate_weight, alpha_weight, beta_weight,
                conv_weight, a, dt, norm, output_weight, input, expected_conv,
                expected_recurrent);
            wide_expected.insert(wide_expected.end(), expected.output.begin(),
                                 expected.output.end());
            expected_conv = expected.conv;
            expected_recurrent = expected.recurrent;
        }
        std::vector<float> output, next_conv, next_recurrent;
        const bool ok = dflash::common::qwen4exp_frontier_gdn_eval_batch(
            wide_gdn, wide_input.data(), wide_input.size(),
            initial_conv.data(), initial_conv.size(), initial_recurrent.data(),
            initial_recurrent.size(), output, next_conv, next_recurrent,
            error);
        CHECK(ok && close_vectors(output, wide_expected, 2.0e-5f) &&
                  close_vectors(next_conv, expected_conv, 2.0e-5f) &&
                  close_vectors(next_recurrent, expected_recurrent, 2.0e-5f),
              "q16 GDN matches scalar rows beyond the conv-history width");
    }
    dflash::common::qwen4exp_frontier_gdn_destroy(wide_gdn);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static std::vector<float> reference_qsa_attention(
        const dflash::common::Qwen4ExpFrontierQsaSpec & spec,
        const std::vector<float> & query, const std::vector<float> & gate,
        const std::vector<float> & selected_key,
        const std::vector<float> & selected_value, int selected_tokens,
        const std::vector<float> & value_rotation,
        const std::vector<float> & output_weight) {
    std::vector<float> attended(
        static_cast<size_t>(spec.n_heads * spec.head_dim));
    const int repeat = spec.n_heads / spec.n_kv_heads;
    for (int head = 0; head < spec.n_heads; ++head) {
        const int kv_head = head / repeat;
        std::vector<float> scores(static_cast<size_t>(selected_tokens));
        float maximum = -INFINITY;
        for (int token = 0; token < selected_tokens; ++token) {
            float dot = 0.0f;
            const size_t cache =
                (static_cast<size_t>(kv_head * selected_tokens + token) *
                 static_cast<size_t>(spec.head_dim));
            for (int dimension = 0; dimension < spec.head_dim; ++dimension) {
                dot += query[static_cast<size_t>(head * spec.head_dim +
                                                  dimension)] *
                       selected_key[cache + static_cast<size_t>(dimension)];
            }
            scores[static_cast<size_t>(token)] =
                dot / std::sqrt(static_cast<float>(spec.head_dim));
            maximum = std::max(maximum, scores[static_cast<size_t>(token)]);
        }
        float denominator = 0.0f;
        for (float & score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        std::vector<float> raw(static_cast<size_t>(spec.head_dim));
        for (int token = 0; token < selected_tokens; ++token) {
            const size_t cache =
                (static_cast<size_t>(kv_head * selected_tokens + token) *
                 static_cast<size_t>(spec.head_dim));
            const float probability =
                scores[static_cast<size_t>(token)] / denominator;
            for (int dimension = 0; dimension < spec.head_dim; ++dimension) {
                raw[static_cast<size_t>(dimension)] += probability *
                    selected_value[cache + static_cast<size_t>(dimension)];
            }
        }
        if (!value_rotation.empty()) {
            raw = matvec(value_rotation, spec.head_dim, spec.head_dim, raw);
        }
        for (int dimension = 0; dimension < spec.head_dim; ++dimension) {
            const size_t index = static_cast<size_t>(
                head * spec.head_dim + dimension);
            attended[index] = raw[static_cast<size_t>(dimension)] *
                              sigmoid(gate[index]);
        }
    }
    return matvec(output_weight, spec.n_embd,
                  spec.n_heads * spec.head_dim, attended);
}

// The GDN batch-versus-sequential coverage above runs spec {4, 6, 2, 4, 4},
// whose convolution channel count is (2*2 + 6) * 4 = 40. HIP's `supports_op`
// for SSM_CONV requires `src0->ne[1] % 128 == 0` (ggml-cuda.cu:5526-5528,
// "assumes d_inner % threads == 0"), so 40 is a shape HIP *refuses*: run that
// fixture on a HIP backend and SSM_CONV silently falls back to CPU. It can
// therefore never exercise the production dispatch, whatever backend it is
// pointed at.
//
// Production is n_heads 48, n_key_heads 16, head_dim 128
// (qwen4exp_runtime.cpp:21-23), giving (2*16 + 48) * 128 = 10240, which passes.
//
// This runs the same batch-versus-sequential comparison at the smallest spec
// that satisfies the HIP predicate: (2*2 + 4) * 16 = 128. It is still the CPU
// backend and still not the 128-wide production kernel, but unlike the fixture
// above it is a shape HIP would accept, so pointing it at a HIP build later
// exercises the real dispatch instead of a fallback.
// Which accumulation order is closer to exact?
//
// The hardware divergence between batched GDN and three sequential q1 steps
// measures 1.1920929e-07 -- exactly 2^-23, one float32 ULP near 1.0. That is
// the floor for two valid roundings, so it says the two paths round
// differently and nothing about which is right. Comparing them to each other
// never can.
//
// The structural difference is where the recurrent state lives between tokens:
// the batched kernel carries it across the token loop without storing it,
// while q1 writes it to a float buffer and reads it back on the next step. The
// control above cannot expose that, because on the CPU backend both paths
// store to the same std::vector<float>.
//
// So model the two orders directly and measure both against a double chain:
//
//   carried  -- state stays in double across the three tokens, rounded to
//               float once at the end
//   rounded  -- state is rounded to float after every token
//
// **This does not model the two HIP paths, and must not be read as doing so.**
// The grouped kernel holds `state_shard` in *float* registers
// (gated_delta_net.cu:253), so the batched path rounds to float per element
// too; the real difference between it and q1 is FMA contraction and scheduling
// at a different loop trip count, not float versus double. A faithful model of
// that cannot be written portably in C++.
//
// What this does establish is the recurrence's *sensitivity*: how much error a
// rounding at the token boundary actually costs, which bounds whether a
// one-ULP divergence compounding over three tokens and 48 layers is plausible
// or needs another explanation. Which of Ember's two paths is closer to exact
// is still decided by running the control below on HIP.
// A reduction's arithmetic must not depend on how many independent reductions
// are launched alongside it.
//
// It did. `ggml_cuda_op_sum_rows` selected its block width from the row count
// -- 512 threads when `(nrows / nsm) < 2`, otherwise 32 -- and
// `reduce_rows_f32` strides by exactly that width, so the same row summed
// through a different tree depending on how many rows shared the launch. On
// gfx1151's 20 CUs that made `exact_l2_norm` reduce identical 128-value rows
// one way at q=1 (nrows 16) and another at q=3 (nrows 48), which is what made
// batched prefill disagree with q=1 and blocked the release.
//
// This is the guard for the class, not the instance: identical row data must
// produce bit-identical sums whichever batch it arrives in. It passes trivially
// on the CPU backend and is meaningful under DFLASH_QWEN_GDN_TEST_HIP=1, where
// it fails on any build predating the fix.
static void test_sum_rows_shape_invariance() {
    const int ncols = 128;
    const int narrow = 16;   // q1 shape:  n_key_heads * 1
    const int wide   = 48;   // q3 shape:  n_key_heads * 3

    const char * hip_value = std::getenv("DFLASH_QWEN_GDN_TEST_HIP");
    const bool use_hip = hip_value && std::strcmp(hip_value, "1") == 0;
    ggml_backend_t backend = use_hip
        ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)
        : ggml_backend_cpu_init();
    if (use_hip && !backend) {
        backend = ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    if (!backend) {
        CHECK(false, "sum_rows shape-invariance backend initializes");
        return;
    }

    // Values that straddle zero and do not sum exactly, so a different
    // accumulation tree actually shows.
    auto row_value = [&](int row, int col) {
        return 0.7071f * std::sin(static_cast<float>((row + 1) * 31 + col) * 0.113f) +
               0.3313f * std::cos(static_cast<float>(col * 7 + row) * 0.037f);
    };

    auto run = [&](int nrows, std::vector<float> & out) {
        ggml_init_params params{};
        params.mem_size = 8U * 1024U * 1024U;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return false;
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
        ggml_tensor * sum = ggml_sum_rows(ctx, x);
        ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) { ggml_free(ctx); return false; }

        std::vector<float> host(static_cast<size_t>(ncols) * nrows);
        for (int row = 0; row < nrows; ++row) {
            for (int col = 0; col < ncols; ++col) {
                host[static_cast<size_t>(row) * ncols + col] =
                    row_value(row, col);
            }
        }
        ggml_backend_tensor_set(x, host.data(), 0, host.size() * sizeof(float));

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, sum);
        const bool ok =
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        if (ok) {
            out.resize(static_cast<size_t>(nrows));
            ggml_backend_tensor_get(sum, out.data(), 0,
                                    out.size() * sizeof(float));
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return ok;
    };

    std::vector<float> narrow_out, wide_out;
    const bool ok = run(narrow, narrow_out) && run(wide, wide_out);
    CHECK(ok, "sum_rows runs at both row counts");

    if (ok) {
        // Rows 0..15 carry identical data in both tensors, so their sums must
        // be bit-identical. Not close -- identical.
        size_t first_diff = static_cast<size_t>(narrow);
        for (int row = 0; row < narrow; ++row) {
            if (narrow_out[static_cast<size_t>(row)] !=
                wide_out[static_cast<size_t>(row)]) {
                first_diff = static_cast<size_t>(row);
                break;
            }
        }
        std::fprintf(stderr,
                     "[sum-rows-invariance] backend=%s nrows %d vs %d "
                     "first_diff_row=%zd\n",
                     use_hip ? "hip" : "cpu", narrow, wide,
                     first_diff == static_cast<size_t>(narrow)
                         ? -1 : static_cast<ssize_t>(first_diff));
        CHECK(first_diff == static_cast<size_t>(narrow),
              "identical rows sum bit-identically regardless of how many rows "
              "share the launch");
    }
    ggml_backend_free(backend);
}

// The external Q4_K checkpoint puts its expert matrices through MUL_MAT_ID.
// Source support is not enough: this HIP-only case instantiates the production
// dimensions, asks the selected device whether it supports the concrete node,
// and computes it directly on that backend. Width 16 is above gfx1151's Q4_K
// MMVQ ceiling, so DFLASH_MMID_TELEMETRY must report path=mmq.
static void test_q4k_mul_mat_id_hip() {
    const char * hip_value = std::getenv("DFLASH_QWEN_Q4K_MMID_TEST_HIP");
    if (!hip_value || std::strcmp(hip_value, "1") != 0) return;

    ggml_backend_t backend =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        backend = ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    CHECK(backend != nullptr, "Q4_K MUL_MAT_ID HIP backend initializes");
    if (!backend) return;

    const enum ggml_backend_dev_type device_type =
        ggml_backend_dev_type(ggml_backend_get_device(backend));
    CHECK(device_type == GGML_BACKEND_DEVICE_TYPE_GPU ||
              device_type == GGML_BACKEND_DEVICE_TYPE_IGPU,
          "Q4_K MUL_MAT_ID runs on a GPU device, not CPU fallback");

    constexpr int64_t embedding = 2560;
    constexpr int64_t expert_ff = 640;
    constexpr int64_t experts = 512;
    constexpr int64_t experts_used = 10;
    constexpr int64_t tokens = 16;
    ggml_init_params params{};
    params.mem_size = 1024U * 1024U;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr, "Q4_K MUL_MAT_ID metadata context initializes");
    if (!ctx) {
        ggml_backend_free(backend);
        return;
    }

    ggml_tensor * weights = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q4_K, embedding, expert_ff, experts);
    ggml_tensor * input = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, embedding, 1, tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, experts_used, tokens);
    ggml_tensor * output = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_set_name(output, "q4k_expert_gate_test");

    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr, "Q4_K expert-shape HIP tensors allocate");
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // Deliberately degenerate routing: the zeroed ids send every row to expert
    // zero. This test proves support and dispatch, not expert distribution.
    ggml_backend_buffer_clear(buffer, 0);

    const bool supported = ggml_backend_supports_op(backend, output);
    CHECK(supported, "HIP supports Q4_K MUL_MAT_ID at Qwen expert shapes");
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = supported
        ? ggml_backend_graph_compute(backend, graph)
        : GGML_STATUS_FAILED;
    CHECK(status == GGML_STATUS_SUCCESS,
          "Q4_K MUL_MAT_ID computes directly on the HIP backend");

    std::array<float, 16> sample{};
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, sample.data(), 0,
                                sample.size() * sizeof(float));
    }
    // Zero Q4_K blocks make this only a finiteness/no-fault assertion. Numeric
    // equivalence to a CPU reference is outside this dispatch test's claim.
    CHECK(status == GGML_STATUS_SUCCESS &&
              std::all_of(sample.begin(), sample.end(),
                          [](float value) { return value == 0.0f; }),
          "zero Q4_K expert blocks produce finite zero HIP output");
    std::fprintf(stderr,
                 "[q4k-mmid] backend=%s device_type=%d supports=%s "
                 "status=%d weights_bytes=%zu width=%lld expected_path=mmq\n",
                 ggml_backend_name(backend), static_cast<int>(device_type),
                 supported ? "true" : "false", static_cast<int>(status),
                 ggml_nbytes(weights), static_cast<long long>(tokens));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void test_gdn_recurrent_accumulation_order() {
    using dflash::common::Qwen4ExpFrontierGdnSpec;
    const Qwen4ExpFrontierGdnSpec spec{8, 4, 2, 128, 4, 1.0e-6f};
    const int channels =
        (2 * spec.n_key_heads + spec.n_heads) * spec.head_dim;
    const int core_values = spec.n_heads * spec.head_dim;

    const std::vector<float> qkv_weight =
        patterned_values(static_cast<size_t>(spec.n_embd) * channels, 0.017f, 11);
    const std::vector<float> alpha_weight =
        patterned_values(static_cast<size_t>(spec.n_embd) * spec.n_heads, 0.019f, 7);
    const std::vector<float> beta_weight =
        patterned_values(static_cast<size_t>(spec.n_embd) * spec.n_heads, 0.023f, 5);
    const std::vector<float> conv_weight =
        patterned_values(static_cast<size_t>(spec.conv_width) * channels, 0.021f, 13);
    const std::vector<float> a = patterned_values(spec.n_heads, -0.031f, 5);
    const std::vector<float> dt = patterned_values(spec.n_heads, 0.007f, 3);
    (void) core_values;

    const std::vector<float> initial_conv = patterned_values(
        static_cast<size_t>(spec.conv_width - 1) * channels, 0.009f, 7);
    const std::vector<float> initial_recurrent = patterned_values(
        static_cast<size_t>(spec.n_heads) * spec.head_dim * spec.head_dim,
        0.004f, 9);

    std::vector<std::vector<float>> rows;
    for (int row = 0; row < 3; ++row) {
        rows.push_back(patterned_values(spec.n_embd, 0.037f, row + 2));
    }

    auto chain = [&](bool round_between_tokens) {
        std::vector<double> conv(initial_conv.begin(), initial_conv.end());
        std::vector<double> rec(initial_recurrent.begin(),
                                initial_recurrent.end());
        for (const std::vector<float> & row : rows) {
            std::vector<double> next_conv;
            rec = reference_gdn_recurrent_double(
                spec, qkv_weight, alpha_weight, beta_weight, conv_weight,
                a, dt, row, conv, rec, next_conv);
            conv = next_conv;
            if (round_between_tokens) {
                for (double & value : rec)
                    value = static_cast<double>(static_cast<float>(value));
                for (double & value : conv)
                    value = static_cast<double>(static_cast<float>(value));
            }
        }
        return rec;
    };

    const std::vector<double> exact   = chain(false);
    const std::vector<double> carried = chain(false);
    const std::vector<double> rounded = chain(true);

    auto worst = [&](const std::vector<double> & candidate) {
        double m = 0.0;
        for (size_t i = 0; i < candidate.size() && i < exact.size(); ++i) {
            m = std::max(m, std::fabs(candidate[i] - exact[i]));
        }
        return m;
    };

    // `carried` is the same computation as `exact`; rounding to float once at
    // the end is what a non-spilling kernel does, so its only error is that
    // final rounding.
    std::vector<double> carried_f32 = carried;
    for (double & value : carried_f32)
        value = static_cast<double>(static_cast<float>(value));

    const double carried_error = worst(carried_f32);
    const double rounded_error = worst(rounded);

    std::fprintf(stderr,
                 "[gdn-accumulation] carried_vs_exact=%.9g "
                 "rounded_vs_exact=%.9g ratio=%.4g\n",
                 carried_error, rounded_error,
                 carried_error > 0.0 ? rounded_error / carried_error : 0.0);

    CHECK(rounded_error > 0.0,
          "rounding the recurrent state between tokens loses precision");
    CHECK(carried_error <= rounded_error,
          "carrying the recurrent state across tokens is at least as close to "
          "exact as rounding it between them");
    CHECK(rounded_error < 1.0e-6,
          "token-boundary rounding stays within a few ULP over three tokens, "
          "so a compounding one-ULP divergence is the expected scale");
}

static void test_gdn_batch_at_hip_legal_conv_channels() {
    using dflash::common::Qwen4ExpFrontierGdnGraph;
    using dflash::common::Qwen4ExpFrontierGdnSpec;
    using dflash::common::Qwen4ExpFrontierGdnWeights;
    // head_dim MUST be 128. `launch_gated_delta_net` switches on S_v
    // (gated_delta_net.cu:397-440) and head_dim 16 lands in
    // `gated_delta_net_cuda<16, ...>`, a different template instantiation that
    // shares no code with the S_v=128 path production runs. A control at any
    // other head_dim proves nothing about the kernel we ship.
    const Qwen4ExpFrontierGdnSpec spec{8, 4, 2, 128, 4, 1.0e-6f};
    const int channels = (2 * spec.n_key_heads + spec.n_heads) * spec.head_dim;
    const int core_values = spec.n_heads * spec.head_dim;
    CHECK(channels % 128 == 0,
          "GDN control spec satisfies the HIP SSM_CONV channel predicate");
    CHECK(spec.head_dim == 128,
          "GDN control runs the S_v=128 kernel production dispatches to");

    const char * hip_value = std::getenv("DFLASH_QWEN_GDN_TEST_HIP");
    const bool use_hip = hip_value && std::strcmp(hip_value, "1") == 0;
    ggml_backend_t backend = use_hip
        ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)
        : ggml_backend_cpu_init();
    if (use_hip && !backend) {
        backend = ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    if (!backend) {
        CHECK(false, use_hip ? "GDN HIP control backend initializes" :
                              "GDN control backend initializes");
        return;
    }
    ggml_init_params params{};
    params.mem_size = 4U * 1024U * 1024U;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        CHECK(false, "GDN control context initializes");
        ggml_backend_free(backend);
        return;
    }
    Qwen4ExpFrontierGdnWeights weights;
    weights.qkv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.n_embd, channels);
    weights.gate =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.n_embd, core_values);
    weights.alpha =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.n_embd, spec.n_heads);
    weights.beta =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.n_embd, spec.n_heads);
    weights.conv =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.conv_width, channels);
    weights.a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, spec.n_heads);
    weights.dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, spec.n_heads);
    weights.norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, spec.head_dim);
    weights.output =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, core_values, spec.n_embd);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        CHECK(false, "GDN control weight buffer allocates");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    const std::vector<float> qkv_weight =
        patterned_values(spec.n_embd * channels, 0.017f, 11);
    const std::vector<float> gate_weight =
        patterned_values(spec.n_embd * core_values, 0.013f, 9);
    const std::vector<float> alpha_weight =
        patterned_values(spec.n_embd * spec.n_heads, 0.019f, 7);
    const std::vector<float> beta_weight =
        patterned_values(spec.n_embd * spec.n_heads, 0.023f, 5);
    const std::vector<float> conv_weight =
        patterned_values(spec.conv_width * channels, 0.021f, 13);
    const std::vector<float> a = patterned_values(spec.n_heads, -0.031f, 5);
    const std::vector<float> dt = patterned_values(spec.n_heads, 0.007f, 3);
    const std::vector<float> norm = patterned_values(spec.head_dim, 0.011f, 7);
    const std::vector<float> output_weight =
        patterned_values(core_values * spec.n_embd, 0.015f, 9);
    const std::array<std::pair<ggml_tensor *, const std::vector<float> *>, 9>
        uploads{{
            {weights.qkv, &qkv_weight},     {weights.gate, &gate_weight},
            {weights.alpha, &alpha_weight}, {weights.beta, &beta_weight},
            {weights.conv, &conv_weight},   {weights.a, &a},
            {weights.dt, &dt},              {weights.norm, &norm},
            {weights.output, &output_weight},
        }};
    for (const auto & upload : uploads) {
        ggml_backend_tensor_set(upload.first, upload.second->data(), 0,
                                upload.second->size() * sizeof(float));
    }

    std::string error;
    setenv("DFLASH_QWEN_GDN_BATCH_COMPARE", "1", 1);
    Qwen4ExpFrontierGdnGraph * batch_graph =
        dflash::common::qwen4exp_frontier_gdn_create_batch(
            backend, spec, weights, -1, 3, error);
    if (!batch_graph) {
        std::fprintf(stderr, "GDN control build error: %s\n", error.c_str());
        CHECK(false, "HIP-legal GDN batch graph builds");
    } else {
        const std::vector<float> initial_conv = patterned_values(
            (spec.conv_width - 1) * channels, 0.009f, 7);
        const std::vector<float> initial_recurrent = patterned_values(
            spec.n_heads * spec.head_dim * spec.head_dim, 0.004f, 9);
        std::vector<float> batch_input, expected_output;
        std::vector<float> expected_conv = initial_conv;
        std::vector<float> expected_recurrent = initial_recurrent;
        for (int row = 0; row < 3; ++row) {
            const std::vector<float> input_row =
                patterned_values(spec.n_embd, 0.037f, row + 2);
            batch_input.insert(batch_input.end(), input_row.begin(),
                               input_row.end());
            const TinyGdnResult step = reference_gdn_q1(
                spec, qkv_weight, gate_weight, alpha_weight, beta_weight,
                conv_weight, a, dt, norm, output_weight, input_row,
                expected_conv, expected_recurrent);
            expected_output.insert(expected_output.end(), step.output.begin(),
                                   step.output.end());
            expected_conv = step.conv;
            expected_recurrent = step.recurrent;
        }
        std::vector<float> output, next_conv, next_recurrent;
        const bool ok = dflash::common::qwen4exp_frontier_gdn_eval_batch(
            batch_graph, batch_input.data(), batch_input.size(),
            initial_conv.data(), initial_conv.size(),
            initial_recurrent.data(), initial_recurrent.size(), output,
            next_conv, next_recurrent, error);
        CHECK(ok && close_vectors(output, expected_output, 2.0e-5f),
              "HIP-legal GDN batch outputs match three sequential scalar rows");
        CHECK(ok && close_vectors(next_conv, expected_conv, 2.0e-5f),
              "HIP-legal GDN batch advances convolution state exactly");
        CHECK(ok && close_vectors(next_recurrent, expected_recurrent, 2.0e-5f),
              "HIP-legal GDN batch final recurrent state matches scalar rows");
        dflash::common::Qwen4ExpFrontierGdnInputs captured;
        const bool capture_ok =
            dflash::common::qwen4exp_frontier_gdn_capture_inputs(
                batch_graph, captured, error);
        CHECK(capture_ok &&
                  captured.convolved.size() ==
                      static_cast<size_t>(3 * channels) &&
                  captured.q.size() ==
                      static_cast<size_t>(
                          3 * spec.n_key_heads * spec.head_dim) &&
                  captured.k.size() ==
                      static_cast<size_t>(
                          3 * spec.n_key_heads * spec.head_dim) &&
                  captured.decay.size() ==
                      static_cast<size_t>(3 * spec.n_heads) &&
                  captured.beta.size() ==
                      static_cast<size_t>(3 * spec.n_heads),
              "GDN comparator captures every recurrence input tensor");

        // Which side is right? The batched path keeps the recurrent state in
        // registers across the token loop; three sequential q1 steps round-trip
        // it through memory. Different accumulation orders cannot be
        // bit-identical, and on hardware their first divergence measures
        // 1.1920929e-07 -- exactly one float32 ULP near 1.0, the floor for two
        // valid roundings.
        //
        // Comparing them to each other cannot say which is correct. Comparing
        // both to a double-precision chain can, and that is the question the
        // release criterion turns on.
        Qwen4ExpFrontierGdnGraph * q1_graph =
            dflash::common::qwen4exp_frontier_gdn_create_batch(
                backend, spec, weights, -1, 1, error);
        CHECK(q1_graph != nullptr, "q1 GDN graph builds for the precision control");
        if (ok && q1_graph) {
            std::vector<float> serial_conv = initial_conv;
            std::vector<float> serial_recurrent = initial_recurrent;
            for (int row = 0; row < 3; ++row) {
                std::vector<float> row_out, next_c, next_r;
                if (!dflash::common::qwen4exp_frontier_gdn_eval_batch(
                        q1_graph,
                        batch_input.data() +
                            static_cast<size_t>(row) * spec.n_embd,
                        static_cast<size_t>(spec.n_embd),
                        serial_conv.data(), serial_conv.size(),
                        serial_recurrent.data(), serial_recurrent.size(),
                        row_out, next_c, next_r, error)) {
                    CHECK(false, "q1 GDN step succeeds in the precision control");
                    break;
                }
                serial_conv = next_c;
                serial_recurrent = next_r;
            }

            std::vector<double> exact_conv(initial_conv.begin(),
                                           initial_conv.end());
            std::vector<double> exact_recurrent(initial_recurrent.begin(),
                                                initial_recurrent.end());
            for (int row = 0; row < 3; ++row) {
                const std::vector<float> row_input =
                    patterned_values(spec.n_embd, 0.037f, row + 2);
                std::vector<double> next_c;
                exact_recurrent = reference_gdn_recurrent_double(
                    spec, qkv_weight, alpha_weight, beta_weight, conv_weight,
                    a, dt, row_input, exact_conv, exact_recurrent, next_c);
                exact_conv = next_c;
            }

            auto worst = [&](const std::vector<float> & candidate) {
                double m = 0.0;
                const size_t n = std::min(candidate.size(),
                                          exact_recurrent.size());
                for (size_t i = 0; i < n; ++i) {
                    m = std::max(m, std::fabs(static_cast<double>(candidate[i]) -
                                              exact_recurrent[i]));
                }
                return m;
            };
            const double batched_error = worst(next_recurrent);
            const double serial_error = worst(serial_recurrent);
            std::fprintf(stderr,
                         "[gdn-precision] backend=%s batched_vs_exact=%.9g "
                         "serial_q1_vs_exact=%.9g batched_closer=%s\n",
                         use_hip ? "hip" : "cpu", batched_error, serial_error,
                         batched_error <= serial_error ? "true" : "false");
            CHECK(batched_error > 0.0 && serial_error > 0.0,
                  "both float paths differ from the double reference, as "
                  "different accumulation orders must");
        }
        dflash::common::qwen4exp_frontier_gdn_destroy(q1_graph);
        dflash::common::qwen4exp_frontier_gdn_destroy(batch_graph);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    unsetenv("DFLASH_QWEN_GDN_BATCH_COMPARE");
}

static void test_persistent_qsa_q1() {
    using dflash::common::Qwen4ExpFrontierQsaGraph;
    using dflash::common::Qwen4ExpFrontierQsaSpec;
    using dflash::common::Qwen4ExpFrontierQsaWeights;
    const Qwen4ExpFrontierQsaSpec spec{4, 4, 2, 4, 2, 2};
    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr, "QSA CPU backend initializes");
    if (!backend) return;
    ggml_init_params params{};
    params.mem_size = 256U * 1024U;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr, "QSA weight context initializes");
    if (!ctx) {
        ggml_backend_free(backend);
        return;
    }
    Qwen4ExpFrontierQsaWeights weights;
    weights.query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 32);
    weights.key = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
    weights.value = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
    weights.index_query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    weights.index_key = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    weights.output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4);
    weights.key_rotation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    weights.value_rotation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr, "QSA weight buffer allocates");
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    const std::vector<float> query_weight =
        patterned_values(128, 0.013f, 11);
    const std::vector<float> key_weight =
        patterned_values(32, 0.019f, 7);
    const std::vector<float> value_weight =
        patterned_values(32, 0.017f, 9);
    const std::vector<float> index_query_weight =
        patterned_values(16, 0.023f, 5);
    const std::vector<float> index_key_weight =
        patterned_values(8, 0.029f, 7);
    const std::vector<float> output_weight =
        patterned_values(64, 0.021f, 13);
    const std::vector<float> rotation{
         0.5f,  0.5f,  0.5f,  0.5f,
         0.5f, -0.5f,  0.5f, -0.5f,
         0.5f,  0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,  0.5f,
    };
    const std::array<std::pair<ggml_tensor *, const std::vector<float> *>, 8>
        uploads{{
            {weights.query, &query_weight},
            {weights.key, &key_weight},
            {weights.value, &value_weight},
            {weights.index_query, &index_query_weight},
            {weights.index_key, &index_key_weight},
            {weights.output, &output_weight},
            {weights.key_rotation, &rotation},
            {weights.value_rotation, &rotation},
        }};
    for (const auto & upload : uploads) {
        ggml_backend_tensor_set(upload.first, upload.second->data(), 0,
                                upload.second->size() * sizeof(float));
    }
    std::string error;
    Qwen4ExpFrontierQsaGraph * graph =
        dflash::common::qwen4exp_frontier_qsa_create_q1(
            backend, spec, weights, -1, error);
    if (!graph)
        std::fprintf(stderr, "QSA frontier build error: %s\n", error.c_str());
    CHECK(graph != nullptr, "persistent q1 QSA graph builds");
    CHECK(dflash::common::qwen4exp_frontier_qsa_cached_width(2048) ==
              2048 &&
          dflash::common::qwen4exp_frontier_qsa_cached_width(2050) ==
              2051 &&
          dflash::common::qwen4exp_frontier_qsa_cached_width(2052) == 0,
          "QSA width policy covers dense boundary and 2074-token tail");
        CHECK(dflash::common::qwen4exp_frontier_qsa_transfer_bytes_q1(
              spec, 2050) == 135998U &&
              dflash::common::qwen4exp_frontier_qsa_selected_state_bytes_q1(
                  spec, 2050) == 131200U,
          "QSA transfer and selected-state boundaries have exact byte bounds");
    if (graph) {
        const std::vector<float> input{0.31f, -0.27f, 0.43f, 0.16f};
        std::vector<float> qfull, key, value, index_query, index_key;
        const bool projected =
            dflash::common::qwen4exp_frontier_qsa_project_q1(
                graph, input.data(), input.size(), qfull, key, value,
                index_query, index_key, error);
        CHECK(projected && close_vectors(
                  qfull, matvec(query_weight, 32, 4, input)),
              "fused QSA query/gate projection matches scalar oracle");
        CHECK(projected &&
                  close_vectors(key, matvec(key_weight, 8, 4, input)) &&
                  close_vectors(value,
                                matvec(value_weight, 8, 4, input)) &&
                  close_vectors(index_query,
                                matvec(index_query_weight, 4, 4, input)) &&
                  close_vectors(index_key,
                                matvec(index_key_weight, 2, 4, input)),
              "fused QSA K/V/index projections match scalar oracle");

        std::vector<float> query(16), gate(16);
        for (int head = 0; head < 4; ++head) {
            std::copy_n(qfull.data() + head * 8, 4,
                        query.data() + head * 4);
            std::copy_n(qfull.data() + head * 8 + 4, 4,
                        gate.data() + head * 4);
        }
        const std::vector<float> raw_query = query;
        const std::vector<float> raw_key = key;
        const std::vector<float> raw_value = value;
        const bool rotated = dflash::common::qwen4exp_frontier_qsa_rotate_q1(
            graph, query, key, value, error);
        std::vector<float> expected_query;
        for (int head = 0; head < 4; ++head) {
            const std::vector<float> row(
                raw_query.begin() + head * 4,
                raw_query.begin() + (head + 1) * 4);
            const std::vector<float> transformed =
                matvec(rotation, 4, 4, row);
            expected_query.insert(expected_query.end(), transformed.begin(),
                                  transformed.end());
        }
        std::vector<float> expected_key, expected_value;
        for (int head = 0; head < 2; ++head) {
            const std::vector<float> key_row(raw_key.begin() + head * 4,
                                             raw_key.begin() + (head + 1) * 4);
            const std::vector<float> value_row(
                raw_value.begin() + head * 4,
                raw_value.begin() + (head + 1) * 4);
            const std::vector<float> transformed_key =
                matvec(rotation, 4, 4, key_row);
            const std::vector<float> transformed_value =
                matvec(rotation, 4, 4, value_row);
            expected_key.insert(expected_key.end(), transformed_key.begin(),
                                transformed_key.end());
            expected_value.insert(expected_value.end(),
                                  transformed_value.begin(),
                                  transformed_value.end());
        }
        CHECK(rotated && close_vectors(query, expected_query) &&
                  close_vectors(key, expected_key) &&
                  close_vectors(value, expected_value),
              "QSA fused rotation preserves #27774 cache representation");

        constexpr int dense_tokens = 2048;
        std::vector<float> dense_key(
            static_cast<size_t>(2 * dense_tokens * 4));
        std::vector<float> dense_value(dense_key.size());
        for (int head = 0; head < 2; ++head) {
            for (int token = 0; token < dense_tokens; ++token) {
                for (int dimension = 0; dimension < 4; ++dimension) {
                    const size_t offset = static_cast<size_t>(
                        (head * dense_tokens + token) * 4 + dimension);
                    dense_key[offset] = 0.001f * static_cast<float>(
                        ((token + 3 * head + dimension) % 17) - 8);
                    dense_value[offset] = 0.002f * static_cast<float>(
                        ((2 * token + head + dimension) % 19) - 9);
                }
            }
        }
        const std::vector<float> dense_expected = reference_qsa_attention(
            spec, query, gate, dense_key, dense_value, dense_tokens,
            rotation, output_weight);
        std::vector<float> dense_actual;
        const bool dense_ok = dflash::common::qwen4exp_frontier_qsa_attend_q1(
            graph, query.data(), query.size(), gate.data(), gate.size(),
            dense_key.data(), dense_value.data(), dense_tokens, dense_actual,
            error);
        CHECK(dense_ok && close_vectors(dense_actual, dense_expected, 4.0e-5f),
              "QSA GPU graph matches scalar attention at dense 2048 boundary");

        // Exact 2074-token frontier: 518 complete blocks, retain the best 512,
        // and append the two-token causal tail. The last six blocks point away
        // from every index-query head and must be absent.
        constexpr int frontier_tokens = 2074;
        std::vector<float> raw_index(
            static_cast<size_t>(frontier_tokens * 128), 0.0f);
        for (int block = 0; block < 518; ++block) {
            const float direction = block < 512 ? 1.0f : -1.0f;
            for (int member = 0; member < 4; ++member) {
                raw_index[static_cast<size_t>((block * 4 + member) * 128)] =
                    direction;
            }
        }
        std::array<float, 512> index_heads{};
        for (int head = 0; head < 4; ++head)
            index_heads[static_cast<size_t>(head * 128)] = 1.0f;
        const std::vector<int32_t> selected =
            dflash::common::qwen4exp_qsa_selected_tokens(
                raw_index, index_heads.data(), frontier_tokens);
        CHECK(selected.size() == 2050U && selected.front() == 0 &&
                  selected[2047] == 2047 && selected[2048] == 2072 &&
                  selected[2049] == 2073,
              "synthetic sparse selector preserves blocks and causal tail");
        std::vector<float> sparse_key(static_cast<size_t>(2 * 2050 * 4));
        std::vector<float> sparse_value(sparse_key.size());
        for (int head = 0; head < 2; ++head) {
            for (size_t slot = 0; slot < selected.size(); ++slot) {
                for (int dimension = 0; dimension < 4; ++dimension) {
                    const size_t offset =
                        (static_cast<size_t>(head) * selected.size() + slot) *
                        4U + static_cast<size_t>(dimension);
                    sparse_key[offset] = 0.0013f * static_cast<float>(
                        (selected[slot] + 5 * head + dimension) % 23 - 11);
                    sparse_value[offset] = 0.0017f * static_cast<float>(
                        (2 * selected[slot] + head + dimension) % 29 - 14);
                }
            }
        }
        const std::vector<float> sparse_expected = reference_qsa_attention(
            spec, query, gate, sparse_key, sparse_value, 2050, rotation,
            output_weight);
        std::vector<float> sparse_actual;
        const bool sparse_ok =
            dflash::common::qwen4exp_frontier_qsa_attend_q1(
                graph, query.data(), query.size(), gate.data(), gate.size(),
                sparse_key.data(), sparse_value.data(), 2050, sparse_actual,
                error);
        CHECK(sparse_ok &&
                  close_vectors(sparse_actual, sparse_expected, 4.0e-5f),
              "QSA graph matches scalar sparse-selected 2074 frontier");

        // Snapshot/replay seam: append-only host state is copied before a new
        // rotated row. Replaying the same branch must publish the same cache
        // and output without mutating the saved frontier.
        dflash::common::Qwen4ExpLayerState saved;
        saved.key.append(key.data(), key.size());
        saved.value.append(value.data(), value.size());
        saved.index_key.append(index_key.data(), index_key.size());
        const size_t saved_key_size = saved.key.size();
        const auto run_branch = [&](dflash::common::Qwen4ExpLayerState branch,
                                    std::vector<float> & branch_output) {
            std::vector<float> branch_query = query;
            std::vector<float> branch_key = key;
            std::vector<float> branch_value = value;
            std::vector<float> packed_key;
            std::vector<float> packed_value;
            packed_key.reserve(16U);
            packed_value.reserve(16U);
            for (int head = 0; head < 2; ++head) {
                for (int dimension = 0; dimension < 4; ++dimension) {
                    packed_key.push_back(branch.key.at(
                        static_cast<size_t>(head * 4 + dimension)));
                    packed_value.push_back(branch.value.at(
                        static_cast<size_t>(head * 4 + dimension)));
                }
                packed_key.insert(packed_key.end(),
                                  branch_key.begin() + head * 4,
                                  branch_key.begin() + (head + 1) * 4);
                packed_value.insert(packed_value.end(),
                                    branch_value.begin() + head * 4,
                                    branch_value.begin() + (head + 1) * 4);
            }
            if (!dflash::common::qwen4exp_frontier_qsa_attend_q1(
                    graph, branch_query.data(), branch_query.size(),
                    gate.data(), gate.size(), packed_key.data(),
                    packed_value.data(), 2, branch_output, error)) return branch;
            branch.key.append(branch_key.data(), branch_key.size());
            branch.value.append(branch_value.data(), branch_value.size());
            branch.index_key.append(index_key.data(), index_key.size());
            return branch;
        };
        std::vector<float> first_output, replay_output;
        const dflash::common::Qwen4ExpLayerState first =
            run_branch(saved, first_output);
        const dflash::common::Qwen4ExpLayerState replay =
            run_branch(saved, replay_output);
        bool cache_equal = first.key.size() == replay.key.size() &&
                           first.value.size() == replay.value.size();
        for (size_t index = 0; cache_equal && index < first.key.size(); ++index)
            cache_equal = first.key.at(index) == replay.key.at(index) &&
                          first.value.at(index) == replay.value.at(index);
        CHECK(saved.key.size() == saved_key_size && cache_equal &&
                  close_vectors(first_output, replay_output, 0.0f),
              "saved QSA host state replays rotated cache branch exactly");
        std::vector<float> rejected;
        CHECK(!dflash::common::qwen4exp_frontier_qsa_attend_q1(
                  graph, query.data(), query.size() - 1U, gate.data(),
                  gate.size(), key.data(), value.data(), 1, rejected, error) &&
                  saved.key.size() == saved_key_size,
              "failed QSA evaluation leaves saved host cache unpublished");
    }
    dflash::common::qwen4exp_frontier_qsa_destroy(graph);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
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
                          float tolerance) {
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

static void test_persistent_qsa_prepared_resident() {
    using dflash::common::Qwen4ExpFrontierQsaGraph;
    using dflash::common::Qwen4ExpFrontierQsaSpec;
    using dflash::common::Qwen4ExpFrontierQsaWeights;
    Qwen4ExpFrontierQsaSpec spec{64, 2, 1, 64, 1, 64};
    char yarn_error[192];
    CHECK(ember_qwen_yarn_configure(
              false, EMBER_QWEN_NATIVE_CONTEXT, &spec.yarn, yarn_error,
              sizeof(yarn_error)),
          "resident QSA test configures native M-RoPE");
    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr, "resident QSA CPU backend initializes");
    if (!backend) return;
    ggml_init_params params{};
    params.mem_size = 2U * 1024U * 1024U;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr, "resident QSA weight context initializes");
    if (!ctx) {
        ggml_backend_free(backend);
        return;
    }
    Qwen4ExpFrontierQsaWeights weights;
    weights.query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 256);
    weights.key = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    weights.value = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    weights.index_query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    weights.index_key = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    weights.output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 64);
    weights.query_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    weights.key_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    weights.index_query_norm =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr, "resident QSA weight buffer allocates");
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }
    const std::vector<float> query_weight =
        patterned_values(64U * 256U, 0.0017f, 29);
    const std::vector<float> key_weight =
        patterned_values(64U * 64U, 0.0021f, 23);
    const std::vector<float> value_weight =
        patterned_values(64U * 64U, 0.0019f, 31);
    const std::vector<float> index_query_weight =
        patterned_values(64U * 64U, 0.0013f, 19);
    const std::vector<float> index_key_weight =
        patterned_values(64U * 64U, 0.0011f, 17);
    const std::vector<float> output_weight =
        patterned_values(128U * 64U, 0.0015f, 27);
    std::vector<float> norm(64U);
    for (size_t index = 0; index < norm.size(); ++index)
        norm[index] = 0.75f + 0.01f * static_cast<float>(index % 23U);
    const std::array<std::pair<ggml_tensor *, const std::vector<float> *>, 9>
        uploads{{
            {weights.query, &query_weight},
            {weights.key, &key_weight},
            {weights.value, &value_weight},
            {weights.index_query, &index_query_weight},
            {weights.index_key, &index_key_weight},
            {weights.output, &output_weight},
            {weights.query_norm, &norm},
            {weights.key_norm, &norm},
            {weights.index_query_norm, &norm},
        }};
    for (const auto & upload : uploads) {
        ggml_backend_tensor_set(upload.first, upload.second->data(), 0,
                                upload.second->size() * sizeof(float));
    }
    std::string error;
    Qwen4ExpFrontierQsaGraph * graph =
        dflash::common::qwen4exp_frontier_qsa_create_q1(
            backend, spec, weights, -1, error);
    if (!graph)
        std::fprintf(stderr, "resident QSA build error: %s\n", error.c_str());
    CHECK(graph != nullptr &&
              dflash::common::qwen4exp_frontier_qsa_can_keep_prepared(graph),
          "resident QSA graph exposes prepared projection handoff");
    if (graph) {
        const std::vector<float> input = patterned_values(64U, 0.013f, 21);
        const int32_t position[3] = {37, 37, 37};
        std::vector<float> query, gate, key, value, index_query, index_key;
        const bool host_ok =
            dflash::common::qwen4exp_frontier_qsa_project_prepared_q1(
                graph, input.data(), input.size(), position, false, query,
                gate, key, value, index_query, index_key, error);
        std::vector<float> host_output;
        const bool host_attention_ok = host_ok &&
            dflash::common::qwen4exp_frontier_qsa_attend_q1(
                graph, query.data(), query.size(), gate.data(), gate.size(),
                key.data(), value.data(), 1, host_output, error);

        std::vector<float> resident_query, resident_gate, resident_key,
                           resident_value, resident_index_query,
                           resident_index_key;
        const bool resident_ok =
            dflash::common::qwen4exp_frontier_qsa_project_prepared_q1(
                graph, input.data(), input.size(), position, true,
                resident_query, resident_gate, resident_key, resident_value,
                resident_index_query, resident_index_key, error);
        std::vector<float> zero_key(key.size(), 0.0f);
        std::vector<float> zero_value(value.size(), 0.0f);
        std::vector<float> current_key, current_value, resident_output;
        const bool resident_attention_ok = resident_ok &&
            dflash::common::qwen4exp_frontier_qsa_attend_prepared_q1(
                graph, zero_key.data(), zero_value.data(), 1, current_key,
                current_value, resident_output, error);
        CHECK(host_ok && resident_ok && resident_query.empty() &&
                  resident_gate.empty() && resident_key.empty() &&
                  resident_value.empty() && resident_index_query.empty() &&
                  close_vectors(resident_index_key, index_key, 0.0f),
              "dense prepared projection downloads only raw index-K");
        CHECK(host_attention_ok && resident_attention_ok,
              "host and resident QSA attention graphs execute");
        CHECK(close_vectors(current_key, key, 0.0f) &&
                  close_vectors(current_value, value, 0.0f),
              "resident current K/V cache rows match prepared projections");
        CHECK(close_vectors(resident_output, host_output, 2.0e-5f),
              "resident Q/gate/current K/V handoff matches host staging");

        std::vector<float> selected_key =
            patterned_values(3U * 64U, 0.007f, 25);
        std::vector<float> selected_value =
            patterned_values(3U * 64U, 0.009f, 33);
        std::copy(key.begin(), key.end(), selected_key.end() - 64);
        std::copy(value.begin(), value.end(), selected_value.end() - 64);
        std::vector<float> host_q3_output;
        const bool host_q3_ok =
            dflash::common::qwen4exp_frontier_qsa_attend_q1(
                graph, query.data(), query.size(), gate.data(), gate.size(),
                selected_key.data(), selected_value.data(), 3,
                host_q3_output, error);
        std::fill(selected_key.end() - 64, selected_key.end(), 0.0f);
        std::fill(selected_value.end() - 64, selected_value.end(), 0.0f);
        const bool resident_q3_projected =
            dflash::common::qwen4exp_frontier_qsa_project_prepared_q1(
                graph, input.data(), input.size(), position, true,
                resident_query, resident_gate, resident_key, resident_value,
                resident_index_query, resident_index_key, error);
        std::vector<float> resident_q3_output;
        const bool resident_q3_ok = resident_q3_projected &&
            dflash::common::qwen4exp_frontier_qsa_attend_prepared_q1(
                graph, selected_key.data(), selected_value.data(), 3,
                current_key, current_value, resident_q3_output, error);
        CHECK(host_q3_ok && resident_q3_ok &&
                  close_vectors(resident_q3_output, host_q3_output, 2.0e-5f),
              "right-aligned resident current row matches q3 host staging");
    }
    dflash::common::qwen4exp_frontier_qsa_destroy(graph);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
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
    const auto project = [](float input, size_t row, size_t layer,
                            size_t boundary) {
        // Miniature of the stateless HC norm/down/up/injection projections.
        // Their result depends on this row's HC frontier but never another
        // row, so all rows may cross each boundary in one matrix call.
        return input * (1.0f + 0.01f * static_cast<float>(layer)) +
            0.001f * static_cast<float>(row) +
            0.02f * static_cast<float>(boundary);
    };
    const auto qsa_project = [](float input, size_t row, size_t layer) {
        // Five QSA input matrices are independent across rows. Model their
        // combined deterministic projection separately from the causal
        // selector/attention update that consumes prior rows below.
        return input * (0.94f + 0.003f * static_cast<float>(layer)) -
            0.002f * static_cast<float>(row);
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
            const float attention_input = qsa_project(
                project(value, row, layer, 0), row, layer);
            const float attention_output = attention(
                attention_input, token_state[layer], row, layer);
            value = ffn(project(attention_output, row, layer, 1), layer);
        }
        token_rows[row] = value;
    }

    std::array<float, kLayers> batch_state{};
    std::array<float, kRows> batch_rows = seed;
    for (size_t layer = 0; layer < kLayers; ++layer) {
        std::array<float, kRows> attention_input{};
        for (size_t row = 0; row < kRows; ++row) {
            attention_input[row] = qsa_project(
                project(batch_rows[row], row, layer, 0), row, layer);
        }
        std::array<float, kRows> ffn_boundary{};
        for (size_t row = 0; row < kRows; ++row) {
            ffn_boundary[row] = attention(
                attention_input[row], batch_state[layer], row, layer);
        }
        std::array<float, kRows> ffn_input{};
        for (size_t row = 0; row < kRows; ++row)
            ffn_input[row] = project(ffn_boundary[row], row, layer, 1);
        for (size_t row = 0; row < kRows; ++row) {
            batch_rows[row] = ffn(ffn_input[row], layer);
        }
    }
    CHECK(close_vectors(
              std::vector<float>(batch_rows.begin(), batch_rows.end()),
              std::vector<float>(token_rows.begin(), token_rows.end()),
              1.0e-6f),
          "batching HC/QSA projections and stateless FFN rows matches token-major outputs");
    CHECK(close_vectors(
              std::vector<float>(batch_state.begin(), batch_state.end()),
              std::vector<float>(token_state.begin(), token_state.end()),
              1.0e-6f),
          "causal layer state matches token-major snapshot frontier");
    std::array<float, kRows> token_logits{};
    std::array<float, kRows> batch_logits{};
    for (size_t row = 0; row < kRows; ++row) {
        // The verifier needs every row's final HC mix and vocabulary logits.
        // Both are row-independent after the last causal layer, so a q5/q16
        // matrix boundary is equivalent to the former q1 loop.
        token_logits[row] = project(token_rows[row], row, kLayers, 2);
        batch_logits[row] = project(batch_rows[row], row, kLayers, 2);
    }
    CHECK(close_vectors(
              std::vector<float>(batch_logits.begin(), batch_logits.end()),
              std::vector<float>(token_logits.begin(), token_logits.end()),
              1.0e-6f),
          "batched final HC and vocabulary projection preserves every verifier row");
}

static void test_causal_ple_projection_batching() {
    constexpr size_t kRows = 7;
    const std::array<int32_t, kRows> tokens = {19, 7, 31, 5, 11, 23, 3};
    const std::array<float, kRows> inputs = {
        0.3f, -0.8f, 1.2f, 0.05f, -0.4f, 0.9f, 0.17f};
    const auto project = [](float input, int32_t token,
                            const std::array<int32_t, 2> & history) {
        // Miniature of PLE row selection followed by the independent key/value
        // matrices. The selected identity is fixed entirely by token history.
        return input + 0.01f * static_cast<float>(token) +
            0.003f * static_cast<float>(history[0]) -
            0.002f * static_cast<float>(history[1]);
    };
    const auto apply = [](float projected, float input, float & conv) {
        // PLE gating and its dilated convolution remain causal and ordered.
        conv = 0.61f * conv + projected;
        return input + 0.13f * projected + 0.07f * conv;
    };

    std::array<int32_t, 2> q1_history{-1, -1};
    float q1_conv = 0.0f;
    std::array<float, kRows> q1_rows{};
    for (size_t row = 0; row < kRows; ++row) {
        const float projected = project(inputs[row], tokens[row], q1_history);
        q1_history = {q1_history[1], tokens[row]};
        q1_rows[row] = apply(projected, inputs[row], q1_conv);
    }

    std::array<int32_t, 2> batch_history{-1, -1};
    std::array<float, kRows> projected{};
    for (size_t row = 0; row < kRows; ++row) {
        projected[row] = project(inputs[row], tokens[row], batch_history);
        batch_history = {batch_history[1], tokens[row]};
    }
    float batch_conv = 0.0f;
    std::array<float, kRows> batch_rows{};
    for (size_t row = 0; row < kRows; ++row)
        batch_rows[row] = apply(projected[row], inputs[row], batch_conv);

    CHECK(close_vectors(
              std::vector<float>(batch_rows.begin(), batch_rows.end()),
              std::vector<float>(q1_rows.begin(), q1_rows.end()), 0.0f) &&
              batch_history == q1_history && batch_conv == q1_conv,
          "batched PLE key/value projections preserve token and convolution frontiers");
}

static void test_bounded_cache_and_prefill_policy() {
    using dflash::common::kQwen4ExpFrontierMoeCachedGraphsPerLayer;
    using dflash::common::kQwen4ExpFrontierMoeMaxBatch;
    using dflash::common::kQwen4ExpFrontierMoeMtpBatch;
    using dflash::common::qwen4exp_frontier_dense_cached_width;
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
    bool dense_policy_matches = true;
    for (int width = 0; width <= kQwen4ExpFrontierMoeMaxBatch + 1; ++width) {
        dense_policy_matches = dense_policy_matches &&
            qwen4exp_frontier_dense_cached_width(width) ==
                qwen4exp_frontier_moe_cached_width(width);
    }
    CHECK(dense_policy_matches,
          "dense projection cache shares the bounded q1/q5/q16 policy");

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
    // MMID telemetry is latched by a function-local static on first dispatch.
    // Arm it before any backend work so the opt-in Q4_K route check cannot be
    // silently voided by an earlier test. The HIP case allocates a production-
    // shape expert tensor (roughly half a GiB), so it remains explicitly gated.
    const char * q4k_hip = std::getenv("DFLASH_QWEN_Q4K_MMID_TEST_HIP");
    if (q4k_hip && std::strcmp(q4k_hip, "1") == 0)
        setenv("DFLASH_MMID_TELEMETRY", "1", 1);
    test_persistent_hc_mixer();
    test_persistent_gdn_q1();
    test_sum_rows_shape_invariance();
    test_q4k_mul_mat_id_hip();
    test_gdn_recurrent_accumulation_order();
    test_gdn_batch_at_hip_legal_conv_channels();
    test_persistent_qsa_q1();
    test_persistent_qsa_prepared_resident();
    test_causal_attention_stateless_ffn_batching();
    test_causal_ple_projection_batching();
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
    ggml_tensor * experts_gate =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 3, 5);
    ggml_tensor * experts_up =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 3, 5);
    weights.experts_down =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 5);
    weights.shared_gate_input =
        ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 4, 1);
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

    std::vector<float> router(20), gate_up(120), gate(60), up(60), down(60);
    std::vector<float> shared_gate_input = {0.07f, -0.04f, 0.02f, 0.09f};
    std::vector<ggml_bf16_t> shared_gate_input_bf16(
        shared_gate_input.size());
    for (size_t index = 0; index < shared_gate_input.size(); ++index) {
        shared_gate_input_bf16[index] =
            ggml_fp32_to_bf16(shared_gate_input[index]);
        shared_gate_input[index] =
            ggml_bf16_to_fp32(shared_gate_input_bf16[index]);
    }
    std::vector<float> shared_gate(12), shared_up(12), shared_down(12);
    for (size_t index = 0; index < router.size(); ++index)
        router[index] = 0.013f * static_cast<float>(static_cast<int>(index % 9) - 4);
    for (size_t index = 0; index < gate_up.size(); ++index)
        gate_up[index] = 0.009f * static_cast<float>(static_cast<int>(index % 13) - 6);
    for (size_t expert = 0; expert < 5; ++expert) {
        for (size_t channel = 0; channel < 3; ++channel) {
            const size_t split = (expert * 3U + channel) * 4U;
            const size_t fused_gate = (expert * 6U + channel) * 4U;
            const size_t fused_up = (expert * 6U + 3U + channel) * 4U;
            std::copy_n(gate_up.begin() + static_cast<std::ptrdiff_t>(fused_gate),
                        4, gate.begin() + static_cast<std::ptrdiff_t>(split));
            std::copy_n(gate_up.begin() + static_cast<std::ptrdiff_t>(fused_up),
                        4, up.begin() + static_cast<std::ptrdiff_t>(split));
        }
    }
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
    ggml_backend_tensor_set(experts_gate, gate.data(), 0,
                            gate.size() * sizeof(float));
    ggml_backend_tensor_set(experts_up, up.data(), 0,
                            up.size() * sizeof(float));
    ggml_backend_tensor_set(weights.experts_down, down.data(), 0,
                            down.size() * sizeof(float));
    ggml_backend_tensor_set(weights.shared_gate_input,
                            shared_gate_input_bf16.data(), 0,
                            shared_gate_input_bf16.size() *
                                sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(weights.shared_gate, shared_gate.data(), 0,
                            shared_gate.size() * sizeof(float));
    ggml_backend_tensor_set(weights.shared_up, shared_up.data(), 0,
                            shared_up.size() * sizeof(float));
    ggml_backend_tensor_set(weights.shared_down, shared_down.data(), 0,
                            shared_down.size() * sizeof(float));

    std::string error;
    dflash::common::Qwen4ExpFrontierDenseCache * dense_cache =
        dflash::common::qwen4exp_frontier_dense_cache_create();
    CHECK(dense_cache != nullptr, "persistent dense cache initializes");
    if (dense_cache) {
        const std::vector<float> q1_input{0.5f, -1.25f, 0.75f, 2.0f};
        std::vector<float> actual;
        bool dense_ok = dflash::common::qwen4exp_frontier_dense_eval(
            dense_cache, backend, weights.router, q1_input.data(), 4, 1,
            actual, error);
        CHECK(dense_ok && close_vectors(actual, matvec(router, 5, 4, q1_input)),
              "persistent q1 dense graph matches scalar matvec");
        CHECK(dflash::common::qwen4exp_frontier_dense_graph_count(dense_cache) ==
                  1U,
              "first dense weight and width creates one graph");

        std::vector<float> q3_input;
        std::vector<float> q3_expected;
        for (int row = 0; row < 3; ++row) {
            const std::vector<float> input_row{
                0.2f * static_cast<float>(row + 1),
                -0.1f * static_cast<float>(row + 2),
                0.05f * static_cast<float>(row),
                0.03f * static_cast<float>(5 - row),
            };
            q3_input.insert(q3_input.end(), input_row.begin(), input_row.end());
            const std::vector<float> expected_row =
                matvec(router, 5, 4, input_row);
            q3_expected.insert(q3_expected.end(), expected_row.begin(),
                               expected_row.end());
        }
        dense_ok = dflash::common::qwen4exp_frontier_dense_eval(
            dense_cache, backend, weights.router, q3_input.data(), 4, 3,
            actual, error);
        CHECK(dense_ok && close_vectors(actual, q3_expected),
              "q3 dense rows zero-pad through the persistent q5 graph");
        CHECK(dflash::common::qwen4exp_frontier_dense_graph_count(dense_cache) ==
                  2U,
              "q1 and q3 use exactly the q1 and q5 cache keys");
        dense_ok = dflash::common::qwen4exp_frontier_dense_eval(
            dense_cache, backend, weights.router, q3_input.data(), 4, 3,
            actual, error);
        CHECK(dense_ok &&
                  dflash::common::qwen4exp_frontier_dense_graph_count(
                      dense_cache) == 2U,
              "repeated dense shape reuses its graph and allocator");

        std::vector<float> q6_input;
        std::vector<float> q6_expected;
        for (int row = 0; row < 6; ++row) {
            q6_input.insert(q6_input.end(), q1_input.begin(), q1_input.end());
            const std::vector<float> expected_row =
                matvec(router, 5, 4, q1_input);
            q6_expected.insert(q6_expected.end(), expected_row.begin(),
                               expected_row.end());
        }
        dense_ok = dflash::common::qwen4exp_frontier_dense_eval(
            dense_cache, backend, weights.router, q6_input.data(), 4, 6,
            actual, error);
        CHECK(dense_ok && close_vectors(actual, q6_expected) &&
                  dflash::common::qwen4exp_frontier_dense_graph_count(
                      dense_cache) == 3U,
              "q6 dense rows use the persistent q16 cache entry");

        dense_ok = dflash::common::qwen4exp_frontier_dense_eval(
            dense_cache, backend, weights.shared_gate, q1_input.data(), 4, 1,
            actual, error);
        CHECK(dense_ok &&
                  close_vectors(actual, matvec(shared_gate, 3, 4, q1_input)) &&
                  dflash::common::qwen4exp_frontier_dense_graph_count(
                      dense_cache) == 4U,
              "dense cache key includes the borrowed weight descriptor");
        CHECK(!dflash::common::qwen4exp_frontier_dense_eval(
                  dense_cache, backend, weights.router, q1_input.data(), 4, 17,
                  actual, error) &&
                  dflash::common::qwen4exp_frontier_dense_graph_count(
                      dense_cache) == 4U,
              "unsupported dense width fails without growing the cache");

        std::vector<float> q60_input;
        std::vector<float> q60_expected;
        for (int row = 0; row < 60; ++row) {
            const std::vector<float> input_row{
                0.01f * static_cast<float>(row + 1),
                -0.02f * static_cast<float>(row + 2),
                0.03f * static_cast<float>(row % 7),
                0.04f * static_cast<float>(row % 11),
            };
            q60_input.insert(q60_input.end(), input_row.begin(), input_row.end());
            const std::vector<float> expected_row =
                matvec(router, 5, 4, input_row);
            q60_expected.insert(q60_expected.end(), expected_row.begin(),
                                expected_row.end());
        }
        dense_ok = dflash::common::qwen4exp_frontier_dense_eval_rows(
            dense_cache, backend, weights.router, q60_input.data(), 4, 60,
            actual, error);
        CHECK(dense_ok && close_vectors(actual, q60_expected) &&
                  dflash::common::qwen4exp_frontier_dense_graph_count(
                      dense_cache) == 4U,
              "q60 MTP HC rows chunk through bounded q16 graphs exactly");

        std::vector<float> first_static;
        bool static_ok = dflash::common::qwen4exp_frontier_static_f32(
            dense_cache, weights.router, first_static, error);
        std::vector<float> changed(router.size(), 123.0f);
        ggml_backend_tensor_set(weights.router, changed.data(), 0,
                                changed.size() * sizeof(float));
        std::vector<float> second_static;
        static_ok = static_ok && dflash::common::qwen4exp_frontier_static_f32(
            dense_cache, weights.router, second_static, error);
        CHECK(static_ok && close_vectors(first_static, router) &&
                  close_vectors(second_static, router),
              "immutable F32 cache avoids a repeated backend download");
        CHECK(dflash::common::qwen4exp_frontier_static_f32_count(dense_cache) ==
                  1U,
              "one immutable tensor owns one decoded host cache entry");
        ggml_backend_tensor_set(weights.router, router.data(), 0,
                                router.size() * sizeof(float));

        // Pad independence, at the widths the HIP differential fails on.
        //
        // qwen4exp_frontier.h:104-107 asserts that q2-q5 reuse the q5 graph and
        // q6-q16 the q16 graph with "zero-padded independent rows", so padding
        // "cannot change a real row". That is the load-bearing assumption
        // behind the bounded cache and it has been a comment, not a test.
        //
        // This compares each row evaluated inside a padded batch against the
        // same row evaluated alone at q=1 -- the same comparison the failing
        // differential makes. It runs on the CPU backend with F32 weights, so
        // a pass here does not clear the HIP quantized path; it narrows the
        // search to what differs from this one.
        const int pad_widths[] = {1, 2, 3, 4, 5, 6, 16, 17};
        bool pad_independent = true;
        int first_bad_width = 0;
        for (int width : pad_widths) {
            std::vector<float> batched_input;
            batched_input.reserve(static_cast<size_t>(width) * 4);
            for (int row = 0; row < width; ++row) {
                const std::vector<float> row_input{
                    0.5f + 0.125f * static_cast<float>(row),
                    -1.25f * static_cast<float>(row % 3 + 1),
                    0.75f - 0.0625f * static_cast<float>(row),
                    2.0f * static_cast<float>(row % 5 - 2),
                };
                batched_input.insert(batched_input.end(), row_input.begin(),
                                     row_input.end());
            }

            std::vector<float> batched;
            if (!dflash::common::qwen4exp_frontier_dense_eval_rows(
                    dense_cache, backend, weights.router, batched_input.data(),
                    4, width, batched, error)) {
                pad_independent = false;
                first_bad_width = width;
                break;
            }

            for (int row = 0; row < width && pad_independent; ++row) {
                std::vector<float> alone;
                if (!dflash::common::qwen4exp_frontier_dense_eval(
                        dense_cache, backend, weights.router,
                        batched_input.data() + static_cast<size_t>(row) * 4, 4,
                        1, alone, error)) {
                    pad_independent = false;
                    first_bad_width = width;
                    break;
                }
                const std::vector<float> from_batch(
                    batched.begin() + static_cast<std::ptrdiff_t>(row) * 5,
                    batched.begin() + static_cast<std::ptrdiff_t>(row + 1) * 5);
                if (!close_vectors(from_batch, alone)) {
                    pad_independent = false;
                    first_bad_width = width;
                    std::fprintf(stderr,
                                 "[pad-independence] width %d row %d differs "
                                 "from its q1 evaluation\n",
                                 width, row);
                }
            }
        }
        if (!pad_independent) {
            std::fprintf(stderr,
                         "[pad-independence] first failing width %d\n",
                         first_bad_width);
        }
        CHECK(pad_independent,
              "every real row in a zero-padded batch equals its q1 evaluation");
    }
    dflash::common::qwen4exp_frontier_dense_cache_destroy(dense_cache);
    dense_cache = nullptr;
    CHECK(dense_cache == nullptr,
          "dense graph and static cache lifecycle ends before weight storage");

    Qwen4ExpFrontierMoeGraph * graph =
        dflash::common::qwen4exp_frontier_moe_create(
            backend, spec, weights, 48, error);
    if (!graph) std::fprintf(stderr, "frontier build error: %s\n", error.c_str());
    CHECK(graph != nullptr,
          "persistent frontier graph builds with a BF16 shared gate");
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
        CHECK(mtp.frontier_moe == nullptr && mtp.frontier_qsa == nullptr,
              "MTP companion destroys its graphs before weight storage");
        CHECK(!dflash::common::qwen4exp_frontier_mtp_moe_q1(
                  mtp, inputs[0].data(), inputs[0].size(), wrong, error),
              "MTP execution fails closed without its GPU frontier graph");
    }

    dflash::common::qwen4exp_frontier_moe_destroy(graph);

    // The MoE half of the pad-independence claim.
    //
    // qwen4exp_frontier.h:104-107 says padding "cannot change a real row"
    // because "MoE rows are independent". The dense half of that sentence is
    // tested above through dense_eval_rows. This is the other half, and it is
    // the one with a plausible coupling: routing picks top-k experts per row,
    // and anything that reduced across the batch axis would let a padded row
    // change which experts a real row is dispatched to. That is a
    // whole-logit-scale error, not a rounding one.
    //
    // Widths cover both physical buckets and both bands the HIP differential
    // fails on. CPU backend with F32 weights, so this constrains the graph
    // algebra and not the quantized or HIP paths.
    {
        Qwen4ExpFrontierMoeGraph * q1_graph =
            dflash::common::qwen4exp_frontier_moe_create(
                backend, spec, weights, 0, error);
        CHECK(q1_graph != nullptr, "q1 MoE graph builds for the pad control");

        const int moe_pad_widths[] = {2, 3, 5, 6, 16};
        bool moe_pad_independent = q1_graph != nullptr;
        for (int width : moe_pad_widths) {
            if (!moe_pad_independent) break;
            const int physical =
                dflash::common::qwen4exp_frontier_moe_cached_width(width);
            Qwen4ExpFrontierMoeGraph * batch_graph =
                dflash::common::qwen4exp_frontier_moe_create_batch(
                    backend, spec, weights, 0, physical, error);
            if (!batch_graph) {
                moe_pad_independent = false;
                std::fprintf(stderr,
                             "[moe-pad] width %d physical %d: graph build "
                             "failed: %s\n",
                             width, physical, error.c_str());
                break;
            }

            // `width` real rows followed by zero padding out to `physical`.
            std::vector<float> padded(
                static_cast<size_t>(physical) * spec.n_embd, 0.0f);
            for (int row = 0; row < width; ++row) {
                for (int d = 0; d < spec.n_embd; ++d) {
                    padded[static_cast<size_t>(row) * spec.n_embd +
                           static_cast<size_t>(d)] =
                        0.4f * static_cast<float>((row + 1) * (d + 1) % 7) -
                        1.1f * static_cast<float>((row + d) % 3);
                }
            }

            std::vector<float> batched;
            if (!dflash::common::qwen4exp_frontier_moe_eval(
                    batch_graph, padded.data(), padded.size(), batched,
                    error)) {
                moe_pad_independent = false;
                std::fprintf(stderr, "[moe-pad] width %d eval failed: %s\n",
                             width, error.c_str());
                dflash::common::qwen4exp_frontier_moe_destroy(batch_graph);
                break;
            }

            for (int row = 0; row < width; ++row) {
                std::vector<float> alone;
                const float * row_input =
                    padded.data() + static_cast<size_t>(row) * spec.n_embd;
                if (!dflash::common::qwen4exp_frontier_moe_eval(
                        q1_graph, row_input,
                        static_cast<size_t>(spec.n_embd), alone, error)) {
                    moe_pad_independent = false;
                    break;
                }
                const std::vector<float> from_batch(
                    batched.begin() +
                        static_cast<std::ptrdiff_t>(row) * spec.n_embd,
                    batched.begin() +
                        static_cast<std::ptrdiff_t>(row + 1) * spec.n_embd);
                if (!close_vectors(from_batch, alone)) {
                    moe_pad_independent = false;
                    std::fprintf(stderr,
                                 "[moe-pad] width %d (physical %d) row %d "
                                 "differs from its q1 evaluation\n",
                                 width, physical, row);
                }
            }
            dflash::common::qwen4exp_frontier_moe_destroy(batch_graph);
        }
        CHECK(moe_pad_independent,
              "every real MoE row in a zero-padded batch equals its q1 "
              "evaluation");
        dflash::common::qwen4exp_frontier_moe_destroy(q1_graph);
    }

    Qwen4ExpFrontierMoeWeights split_weights = weights;
    split_weights.experts_gate_up = nullptr;
    split_weights.experts_gate = experts_gate;
    split_weights.experts_up = experts_up;
    Qwen4ExpFrontierMoeGraph * split_graph =
        dflash::common::qwen4exp_frontier_moe_create(
            backend, spec, split_weights, 0, error);
    CHECK(split_graph != nullptr,
          "canonical split gate/up frontier graph builds");
    if (split_graph) {
        const std::vector<float> input{0.5f, -1.25f, 0.75f, 2.0f};
        std::vector<float> actual;
        const bool ok = dflash::common::qwen4exp_frontier_moe_eval(
            split_graph, input.data(), input.size(), actual, error);
        const std::vector<float> expected = reference_moe(
            spec, router, gate_up, down, shared_gate_input, shared_gate,
            shared_up, shared_down, input);
        CHECK(ok && close_vectors(actual, expected),
              "split gate/up graph matches fused scalar MoE reference");
    }
    dflash::common::qwen4exp_frontier_moe_destroy(split_graph);

    Qwen4ExpFrontierMoeWeights incomplete_split = split_weights;
    incomplete_split.experts_up = nullptr;
    CHECK(dflash::common::qwen4exp_frontier_moe_create(
              backend, spec, incomplete_split, 0, error) == nullptr,
          "frontier rejects an incomplete split expert contract");
    Qwen4ExpFrontierMoeWeights ambiguous = split_weights;
    ambiguous.experts_gate_up = weights.experts_gate_up;
    CHECK(dflash::common::qwen4exp_frontier_moe_create(
              backend, spec, ambiguous, 0, error) == nullptr,
          "frontier rejects simultaneous fused and split experts");

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
