#include "qwen4exp_internal.h"
#include "qwen4exp_frontier.h"
#include "qwen4exp_mtp.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

namespace dflash::common {
namespace {

constexpr int kEmbedding = 2560;
constexpr int kHc = 4;
constexpr int kHcDim = kEmbedding * kHc;
constexpr int kGdnHeads = 48;
constexpr int kGdnKeyHeads = 16;
constexpr int kGdnDim = 128;
constexpr int kQsaHeads = 24;
constexpr int kQsaKvHeads = 2;
constexpr int kQsaDim = 256;
constexpr int kIndexerHeads = 4;
constexpr int kIndexerDim = 128;
constexpr int kExpertFf = 640;
constexpr int kExpertCount = 512;
constexpr int kExpertUsed = 10;
constexpr float kEpsilon = 1.0e-6f;

float sigmoid(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
    const float e = std::exp(value);
    return e / (1.0f + e);
}

float silu(float value) { return value * sigmoid(value); }

float softplus(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return std::exp(value);
    return std::log1p(std::exp(value));
}

void rms_norm(float * values, int count, const float * weight) {
    double sum = 0.0;
    for (int i = 0; i < count; ++i) sum += values[i] * values[i];
    const float scale = 1.0f / std::sqrt(static_cast<float>(sum / count) + kEpsilon);
    for (int i = 0; i < count; ++i) values[i] *= scale * (weight ? weight[i] : 1.0f);
}

void l2_norm(float * values, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; ++i) sum += values[i] * values[i];
    const float scale = 1.0f / std::sqrt(static_cast<float>(sum) + kEpsilon);
    for (int i = 0; i < count; ++i) values[i] *= scale;
}

bool tensor_f32(Qwen4ExpFrontierDenseCache * cache, ggml_tensor * tensor,
                std::vector<float> & out, std::string & error) {
    return qwen4exp_frontier_static_f32(cache, tensor, out, error);
}

bool matvec(Qwen4ExpFrontierDenseCache * cache, ggml_backend_t backend,
            ggml_tensor * weight,
            const float * input, int input_count,
            std::vector<float> & output, std::string & error) {
    return qwen4exp_frontier_dense_eval(cache, backend, weight, input,
                                        input_count, 1, output, error);
}

bool matmul_rows(Qwen4ExpFrontierDenseCache * cache, ggml_backend_t backend,
                 ggml_tensor * weight,
                 const float * input, int input_count, int rows,
                 std::vector<float> & output, std::string & error) {
    return qwen4exp_frontier_dense_eval_rows(
        cache, backend, weight, input, input_count, rows, output, error);
}

bool rotate_optional(Qwen4ExpFrontierDenseCache * cache,
                     ggml_backend_t backend, ggml_tensor * rotation,
                     std::vector<float> & values, std::string & error) {
    if (!rotation) return true;
    if (ggml_n_dims(rotation) != 2 || rotation->ne[0] != rotation->ne[1] ||
        rotation->ne[0] <= 0 || values.size() % static_cast<size_t>(rotation->ne[0])) {
        error = "invalid Qwen4Exp optional Hadamard rotation shape";
        return false;
    }
    const int width = static_cast<int>(rotation->ne[0]);
    std::vector<float> rotated;
    const size_t rows = values.size() / static_cast<size_t>(width);
    if (rows == 0 || rows >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        !matmul_rows(cache, backend, rotation, values.data(), width,
                     static_cast<int>(rows), rotated, error)) return false;
    values = std::move(rotated);
    return true;
}

bool hc_mix(const Qwen4ExpWeights & weights, const std::vector<float> & hc,
            ggml_tensor * norm, ggml_tensor * down, ggml_tensor * up,
            ggml_tensor * inject_weight, std::vector<float> & mixed,
            std::array<float, kHc> * inject, std::string & error,
            Qwen4ExpFrontierDenseCache * cache_override = nullptr) {
    Qwen4ExpFrontierDenseCache * cache =
        cache_override ? cache_override : weights.dense_cache;
    if (hc.size() != static_cast<size_t>(kHcDim)) {
        error = "invalid Qwen4Exp HC state";
        return false;
    }
    const Qwen4ExpFrontierHcSpec spec{
        kEmbedding, kHc, kEpsilon};
    std::vector<float> raw_injection;
    if (!qwen4exp_frontier_hc_eval(
            cache, weights.backend, spec, norm, down, up, inject_weight,
            hc.data(), hc.size(), 1, mixed,
            inject ? &raw_injection : nullptr, error)) return false;
    if (inject) {
        if (raw_injection.size() != static_cast<size_t>(kHc)) {
            error = "Qwen4Exp HC graph returned an invalid injection";
            return false;
        }
        std::copy(raw_injection.begin(), raw_injection.end(), inject->begin());
    }
    if (mixed.size() != static_cast<size_t>(kEmbedding)) {
        error = "Qwen4Exp HC graph returned an invalid mixed row";
        return false;
    }
    return true;
}

bool hc_output_rows(
        const Qwen4ExpWeights & weights, Qwen4ExpFrontierDenseCache * cache,
        ggml_tensor * norm, ggml_tensor * down, ggml_tensor * up,
        ggml_tensor * projection, const float * hc, size_t hc_count,
        int n_tokens, std::vector<float> & output, std::string & error) {
    if (!cache || !hc || n_tokens <= 0 || hc_count !=
            static_cast<size_t>(kHcDim) * static_cast<size_t>(n_tokens)) {
        error = "invalid Qwen4Exp final HC rows";
        return false;
    }
    const Qwen4ExpFrontierHcSpec spec{kEmbedding, kHc, kEpsilon};
    return qwen4exp_frontier_hc_output_eval(
        cache, weights.backend, spec, norm, down, up, projection, hc,
        hc_count, n_tokens, output, error);
}

void hc_combine(std::vector<float> & hc, const std::vector<float> & block,
                const std::array<float, kHc> & inject) {
    for (int stream = 0; stream < kHc; ++stream) {
        const float weight = 2.0f * sigmoid(inject[stream] / static_cast<float>(kHc));
        for (int i = 0; i < kEmbedding; ++i)
            hc[stream * kEmbedding + i] += weight * block[i];
    }
}

bool rope(const Qwen4ExpWeights & weights, float * vector, int dimension,
          const int32_t positions[3], std::string & error) {
    if (!ember_qwen_yarn_apply(vector, static_cast<size_t>(dimension),
                               &weights.yarn, positions)) {
        error = "invalid Qwen4Exp YaRN/M-RoPE application";
        return false;
    }
    return true;
}

Qwen4ExpMappedTensor expert_slice(const Qwen4ExpMappedTensor & tensor,
                                  int expert) {
    Qwen4ExpMappedTensor slice = tensor;
    if (!tensor.valid() || tensor.n_dims != 3 || expert < 0 ||
        expert >= tensor.ne[2]) return {};
    slice.n_dims = 2;
    slice.bytes = tensor.bytes / static_cast<size_t>(tensor.ne[2]);
    slice.data = tensor.data + static_cast<size_t>(expert) * slice.bytes;
    slice.ne[2] = 0;
    return slice;
}

bool mapped_matvec(const Qwen4ExpMappedTensor & matrix, const float * input,
                   std::vector<float> & output, std::string & error) {
    if (!matrix.valid() || matrix.n_dims != 2 || !input) {
        error = "invalid mapped Qwen4Exp matvec"; return false;
    }
    std::vector<float> row(static_cast<size_t>(matrix.ne[0]));
    output.assign(static_cast<size_t>(matrix.ne[1]), 0.0f);
    for (int64_t r = 0; r < matrix.ne[1]; ++r) {
        if (!qwen4exp_mapped_row_f32(matrix, r, row.data(), row.size(), &error))
            return false;
        double sum = 0.0;
        for (int64_t c = 0; c < matrix.ne[0]; ++c) sum += row[c] * input[c];
        output[static_cast<size_t>(r)] = static_cast<float>(sum);
    }
    return true;
}

bool run_ple(const Qwen4ExpWeights & weights, Qwen4ExpState & state,
             const Qwen4ExpLayer & layer, int32_t token,
             std::string & error) {
    const auto rows = qwen4exp_ple_rows(token, state.ple_tokens);
    std::vector<float> embedded(kEmbedding);
    std::vector<float> one(160);
    for (int head = 0; head < 16; ++head) {
        if (!qwen4exp_mapped_row_f32(weights.ple_table, rows[head], one.data(),
                                    one.size(), &error)) return false;
        std::copy(one.begin(), one.end(), embedded.begin() + head * 160);
    }
    std::vector<float> key, value;
    if (!matvec(weights.dense_cache, weights.backend, layer.ple_key,
                embedded.data(), kEmbedding, key, error) ||
        !matvec(weights.dense_cache, weights.backend, layer.ple_value,
                embedded.data(), kEmbedding, value, error)) return false;
    std::vector<float> key_norm, query_norm, conv_norm, conv_weight;
    if (!tensor_f32(weights.dense_cache, layer.ple_norm_key, key_norm, error) ||
        !tensor_f32(weights.dense_cache, layer.ple_norm_query, query_norm,
                    error) ||
        !tensor_f32(weights.dense_cache, layer.ple_norm_conv, conv_norm,
                    error) ||
        !tensor_f32(weights.dense_cache, layer.ple_conv, conv_weight, error))
        return false;
    std::vector<float> gated(kHcDim), normalized(kHcDim);
    for (int stream = 0; stream < kHc; ++stream) {
        float * kp = key.data() + stream * kEmbedding;
        std::vector<float> query(state.hc.begin() + stream * kEmbedding,
                                 state.hc.begin() + (stream + 1) * kEmbedding);
        rms_norm(kp, kEmbedding, key_norm.data() + stream * kEmbedding);
        rms_norm(query.data(), kEmbedding,
                 query_norm.data() + stream * kEmbedding);
        double dot = 0.0;
        for (int i = 0; i < kEmbedding; ++i) dot += kp[i] * query[i];
        const float score = static_cast<float>(dot / std::sqrt(kEmbedding));
        const float gate = sigmoid(std::copysign(std::sqrt(std::max(std::fabs(score), 1e-6f)), score));
        for (int i = 0; i < kEmbedding; ++i)
            gated[stream * kEmbedding + i] = value[i] * gate;
        std::copy(gated.begin() + stream * kEmbedding,
                  gated.begin() + (stream + 1) * kEmbedding,
                  normalized.begin() + stream * kEmbedding);
        rms_norm(normalized.data() + stream * kEmbedding, kEmbedding,
                 conv_norm.data() + stream * kEmbedding);
    }
    if (state.ple_conv.empty()) state.ple_conv.assign(9 * kHcDim, 0.0f);
    std::vector<float> conv(kHcDim, 0.0f);
    for (int channel = 0; channel < kHcDim; ++channel) {
        for (int tap = 0; tap < 4; ++tap) {
            const int back = (3 - tap) * 3;
            const float input = back == 0
                ? normalized[channel]
                : state.ple_conv[static_cast<size_t>(9 - back) * kHcDim + channel];
            conv[channel] += conv_weight[channel * 4 + tap] * input;
        }
        conv[channel] = silu(conv[channel]);
    }
    std::move(state.ple_conv.begin() + kHcDim, state.ple_conv.end(),
              state.ple_conv.begin());
    std::copy(normalized.begin(), normalized.end(),
              state.ple_conv.end() - kHcDim);
    for (int i = 0; i < kHcDim; ++i) state.hc[i] += gated[i] + conv[i];
    state.ple_tokens = {state.ple_tokens[1], token};
    return true;
}

bool run_ple_batch(const Qwen4ExpWeights & weights, Qwen4ExpState & state,
                   const Qwen4ExpLayer & layer,
                   const std::vector<int32_t> & tokens,
                   std::vector<std::vector<float>> & hc_rows,
                   std::string & error) {
    const size_t rows = tokens.size();
    if (rows == 0 || rows > 16 || hc_rows.size() != rows) {
        error = "invalid Qwen4Exp batched PLE input";
        return false;
    }
    for (const std::vector<float> & hc : hc_rows) {
        if (hc.size() != static_cast<size_t>(kHcDim)) {
            error = "invalid Qwen4Exp batched PLE HC row";
            return false;
        }
    }

    // PLE row identities depend only on the prior two token ids. Resolve that
    // small causal chain first, then cross the large key/value projection
    // boundary once for the complete q5/q16 chunk.
    std::array<int32_t, 2> token_history = state.ple_tokens;
    std::vector<float> embedded(rows * static_cast<size_t>(kEmbedding));
    std::vector<float> one(160);
    for (size_t row = 0; row < rows; ++row) {
        const auto selected = qwen4exp_ple_rows(tokens[row], token_history);
        for (int head = 0; head < 16; ++head) {
            if (!qwen4exp_mapped_row_f32(
                    weights.ple_table, selected[head], one.data(), one.size(),
                    &error)) return false;
            std::copy(one.begin(), one.end(),
                      embedded.begin() + static_cast<std::ptrdiff_t>(
                          row * static_cast<size_t>(kEmbedding) +
                          static_cast<size_t>(head * 160)));
        }
        token_history = {token_history[1], tokens[row]};
    }
    std::vector<float> key, value;
    if (!matmul_rows(weights.dense_cache, weights.backend, layer.ple_key,
                     embedded.data(), kEmbedding, static_cast<int>(rows),
                     key, error) ||
        !matmul_rows(weights.dense_cache, weights.backend, layer.ple_value,
                     embedded.data(), kEmbedding, static_cast<int>(rows),
                     value, error) ||
        key.size() != rows * static_cast<size_t>(kHcDim) ||
        value.size() != rows * static_cast<size_t>(kEmbedding)) {
        if (error.empty()) error = "Qwen4Exp batched PLE projection mismatch";
        return false;
    }
    std::vector<float> key_norm, query_norm, conv_norm, conv_weight;
    if (!tensor_f32(weights.dense_cache, layer.ple_norm_key, key_norm,
                    error) ||
        !tensor_f32(weights.dense_cache, layer.ple_norm_query, query_norm,
                    error) ||
        !tensor_f32(weights.dense_cache, layer.ple_norm_conv, conv_norm,
                    error) ||
        !tensor_f32(weights.dense_cache, layer.ple_conv, conv_weight, error))
        return false;
    if (state.ple_conv.empty()) state.ple_conv.assign(9 * kHcDim, 0.0f);
    if (state.ple_conv.size() != 9U * static_cast<size_t>(kHcDim)) {
        error = "invalid Qwen4Exp batched PLE convolution frontier";
        return false;
    }
    for (size_t row = 0; row < rows; ++row) {
        float * row_key = key.data() + row * static_cast<size_t>(kHcDim);
        const float * row_value =
            value.data() + row * static_cast<size_t>(kEmbedding);
        std::vector<float> gated(static_cast<size_t>(kHcDim));
        std::vector<float> normalized(static_cast<size_t>(kHcDim));
        for (int stream = 0; stream < kHc; ++stream) {
            float * key_stream = row_key + stream * kEmbedding;
            std::vector<float> query(
                hc_rows[row].begin() + stream * kEmbedding,
                hc_rows[row].begin() + (stream + 1) * kEmbedding);
            rms_norm(key_stream, kEmbedding,
                     key_norm.data() + stream * kEmbedding);
            rms_norm(query.data(), kEmbedding,
                     query_norm.data() + stream * kEmbedding);
            double dot = 0.0;
            for (int channel = 0; channel < kEmbedding; ++channel)
                dot += key_stream[channel] * query[channel];
            const float score =
                static_cast<float>(dot / std::sqrt(kEmbedding));
            const float gate = sigmoid(std::copysign(
                std::sqrt(std::max(std::fabs(score), 1e-6f)), score));
            for (int channel = 0; channel < kEmbedding; ++channel) {
                gated[static_cast<size_t>(stream * kEmbedding + channel)] =
                    row_value[channel] * gate;
            }
            std::copy_n(
                gated.data() + stream * kEmbedding, kEmbedding,
                normalized.data() + stream * kEmbedding);
            rms_norm(normalized.data() + stream * kEmbedding, kEmbedding,
                     conv_norm.data() + stream * kEmbedding);
        }
        std::vector<float> conv(static_cast<size_t>(kHcDim), 0.0f);
        for (int channel = 0; channel < kHcDim; ++channel) {
            for (int tap = 0; tap < 4; ++tap) {
                const int back = (3 - tap) * 3;
                const float input = back == 0
                    ? normalized[static_cast<size_t>(channel)]
                    : state.ple_conv[static_cast<size_t>(9 - back) *
                                         static_cast<size_t>(kHcDim) +
                                     static_cast<size_t>(channel)];
                conv[static_cast<size_t>(channel)] +=
                    conv_weight[static_cast<size_t>(channel * 4 + tap)] *
                    input;
            }
            conv[static_cast<size_t>(channel)] =
                silu(conv[static_cast<size_t>(channel)]);
        }
        std::move(state.ple_conv.begin() + kHcDim, state.ple_conv.end(),
                  state.ple_conv.begin());
        std::copy(normalized.begin(), normalized.end(),
                  state.ple_conv.end() - kHcDim);
        for (int channel = 0; channel < kHcDim; ++channel) {
            hc_rows[row][static_cast<size_t>(channel)] +=
                gated[static_cast<size_t>(channel)] +
                conv[static_cast<size_t>(channel)];
        }
    }
    state.ple_tokens = token_history;
    return true;
}

bool run_gdn_scalar(const Qwen4ExpWeights & weights,
                    Qwen4ExpLayerState & state,
                    const Qwen4ExpLayer & layer,
                    const std::vector<float> & input,
                    std::vector<float> & output, std::string & error) {
    std::vector<float> qkv, z, alpha, beta;
    if (!matvec(weights.dense_cache, weights.backend, layer.attn_qkv,
                input.data(), kEmbedding, qkv, error) ||
        !matvec(weights.dense_cache, weights.backend, layer.attn_gate,
                input.data(), kEmbedding, z, error) ||
        !matvec(weights.dense_cache, weights.backend, layer.ssm_alpha,
                input.data(), kEmbedding, alpha, error) ||
        !matvec(weights.dense_cache, weights.backend, layer.ssm_beta,
                input.data(), kEmbedding, beta, error)) return false;
    std::vector<float> conv_weight, a, dt, norm;
    if (!tensor_f32(weights.dense_cache, layer.ssm_conv, conv_weight, error) ||
        !tensor_f32(weights.dense_cache, layer.ssm_a, a, error) ||
        !tensor_f32(weights.dense_cache, layer.ssm_dt, dt, error) ||
        !tensor_f32(weights.dense_cache, layer.ssm_norm, norm, error))
        return false;
    const size_t conv_size = 3U * 10240U;
    const size_t recurrent_size =
        static_cast<size_t>(kGdnHeads) * kGdnDim * kGdnDim;
    if (!state.conv) {
        state.conv = std::make_shared<std::vector<float>>(conv_size, 0.0f);
    } else if (!state.conv.unique()) {
        state.conv = std::make_shared<std::vector<float>>(*state.conv);
    }
    if (!state.recurrent) {
        state.recurrent =
            std::make_shared<std::vector<float>>(recurrent_size, 0.0f);
    } else if (!state.recurrent.unique()) {
        state.recurrent =
            std::make_shared<std::vector<float>>(*state.recurrent);
    }
    std::vector<float> & conv_state = *state.conv;
    std::vector<float> & recurrent_state = *state.recurrent;
    std::vector<float> convolved(10240);
    for (int channel = 0; channel < 10240; ++channel) {
        float value = conv_weight[channel * 4 + 3] * qkv[channel];
        for (int tap = 0; tap < 3; ++tap)
            value += conv_weight[channel * 4 + tap] *
                     conv_state[static_cast<size_t>(tap) * 10240U +
                                static_cast<size_t>(channel)];
        convolved[channel] = silu(value);
    }
    std::move(conv_state.begin() + 10240, conv_state.end(), conv_state.begin());
    std::copy(qkv.begin(), qkv.end(), conv_state.end() - 10240);
    std::vector<float> core(6144);
    for (int head = 0; head < kGdnHeads; ++head) {
        const int key_head = head / 3; // HF repeat_interleave(48/16)
        std::vector<float> q(convolved.begin() + key_head * kGdnDim,
                             convolved.begin() + (key_head + 1) * kGdnDim);
        std::vector<float> k(convolved.begin() + 2048 + key_head * kGdnDim,
                             convolved.begin() + 2048 + (key_head + 1) * kGdnDim);
        l2_norm(q.data(), kGdnDim); l2_norm(k.data(), kGdnDim);
        const float decay = std::exp(softplus(alpha[head] + dt[head]) * a[head]);
        const float b = sigmoid(beta[head]);
        float * recurrent = recurrent_state.data() +
            static_cast<size_t>(head) * kGdnDim * kGdnDim;
        const float * v = convolved.data() + 4096 + head * kGdnDim;
        std::array<float, kGdnDim> delta{};
        for (int j = 0; j < kGdnDim; ++j) {
            float dot = 0.0f;
            for (int i = 0; i < kGdnDim; ++i) {
                recurrent[j * kGdnDim + i] *= decay;
                dot += recurrent[j * kGdnDim + i] * k[i];
            }
            delta[j] = (v[j] - dot) * b;
            for (int i = 0; i < kGdnDim; ++i)
                recurrent[j * kGdnDim + i] += delta[j] * k[i];
            float result = 0.0f;
            for (int i = 0; i < kGdnDim; ++i)
                result += recurrent[j * kGdnDim + i] * q[i];
            core[head * kGdnDim + j] = result / std::sqrt(static_cast<float>(kGdnDim));
        }
        rms_norm(core.data() + head * kGdnDim, kGdnDim, norm.data());
        for (int j = 0; j < kGdnDim; ++j)
            core[head * kGdnDim + j] *= sigmoid(z[head * kGdnDim + j]);
    }
    return matvec(weights.dense_cache, weights.backend, layer.ssm_out,
                  core.data(), 6144, output, error);
}

bool run_gdn(const Qwen4ExpWeights & weights, Qwen4ExpLayerState & state,
             const Qwen4ExpLayer & layer, int layer_index,
             const std::vector<float> & input,
             std::vector<float> & output, std::string & error) {
    if (!qwen4exp_frontier_gdn_available(weights, layer_index)) {
        return run_gdn_scalar(weights, state, layer, input, output, error);
    }
    constexpr size_t kConvValues = 3U * 10240U;
    constexpr size_t kRecurrentValues =
        static_cast<size_t>(kGdnHeads) * kGdnDim * kGdnDim;
    const std::vector<float> zero_conv(
        state.conv ? 0U : kConvValues, 0.0f);
    const std::vector<float> zero_recurrent(
        state.recurrent ? 0U : kRecurrentValues, 0.0f);
    const std::vector<float> & conv = state.conv ? *state.conv : zero_conv;
    const std::vector<float> & recurrent =
        state.recurrent ? *state.recurrent : zero_recurrent;
    std::vector<float> next_conv, next_recurrent;
    if (!qwen4exp_frontier_gdn_q1(
            weights, layer_index, input.data(), input.size(), conv.data(),
            conv.size(), recurrent.data(), recurrent.size(), output, next_conv,
            next_recurrent, error)) {
        return false;
    }
    // Commit only after graph compute and every synchronized state download
    // succeeded. Snapshot-shared vectors remain untouched, so speculative
    // rejection can restore the exact host frontier.
    state.conv = std::make_shared<std::vector<float>>(std::move(next_conv));
    state.recurrent =
        std::make_shared<std::vector<float>>(std::move(next_recurrent));
    return true;
}

bool run_gdn_batch(const Qwen4ExpWeights & weights,
                   Qwen4ExpLayerState & state,
                   const Qwen4ExpLayer & layer, int layer_index,
                   const std::vector<float> & input, int n_tokens,
                   std::vector<float> & output, std::string & error) {
    if (n_tokens < 2 || n_tokens > kQwen4ExpFrontierMoeMaxBatch ||
        input.size() != static_cast<size_t>(n_tokens) * kEmbedding) {
        error = "invalid Qwen4Exp batched GDN input";
        return false;
    }
    if (!qwen4exp_frontier_gdn_available(weights, layer_index)) {
        output.resize(input.size());
        for (int row = 0; row < n_tokens; ++row) {
            const auto begin = input.begin() +
                static_cast<std::ptrdiff_t>(row * kEmbedding);
            const std::vector<float> row_input(
                begin, begin + static_cast<std::ptrdiff_t>(kEmbedding));
            std::vector<float> row_output;
            if (!run_gdn_scalar(weights, state, layer, row_input, row_output,
                                error)) return false;
            std::copy(row_output.begin(), row_output.end(),
                      output.begin() +
                          static_cast<std::ptrdiff_t>(row * kEmbedding));
        }
        return true;
    }
    constexpr size_t kConvValues = 3U * 10240U;
    constexpr size_t kRecurrentValues =
        static_cast<size_t>(kGdnHeads) * kGdnDim * kGdnDim;
    const std::vector<float> zero_conv(
        state.conv ? 0U : kConvValues, 0.0f);
    const std::vector<float> zero_recurrent(
        state.recurrent ? 0U : kRecurrentValues, 0.0f);
    const std::vector<float> & conv = state.conv ? *state.conv : zero_conv;
    const std::vector<float> & recurrent =
        state.recurrent ? *state.recurrent : zero_recurrent;
    std::vector<float> next_conv, next_recurrent;
    if (!qwen4exp_frontier_gdn_batch(
            weights, layer_index, input.data(), input.size(), n_tokens,
            conv.data(), conv.size(), recurrent.data(), recurrent.size(),
            output, next_conv, next_recurrent, error)) return false;
    // As with q1, publish only the final causal frontier after graph compute
    // and all downloads succeed. Saved shared states remain exact rollback
    // points for a rejected verifier/prefill chunk.
    state.conv = std::make_shared<std::vector<float>>(std::move(next_conv));
    state.recurrent =
        std::make_shared<std::vector<float>>(std::move(next_recurrent));
    return true;
}

bool append_qsa_cache(const Qwen4ExpWeights & weights,
                      Qwen4ExpLayerState & state,
                      const Qwen4ExpLayer & layer,
                      const std::vector<float> & input,
                      const std::array<int32_t, 3> & position,
                      std::string & error,
                      Qwen4ExpFrontierDenseCache * cache_override = nullptr) {
    Qwen4ExpFrontierDenseCache * cache =
        cache_override ? cache_override : weights.dense_cache;
    std::vector<float> k, v, ik;
    if (!matvec(cache, weights.backend, layer.attn_k, input.data(), kEmbedding,
                k, error) ||
        !matvec(cache, weights.backend, layer.attn_v, input.data(), kEmbedding,
                v, error) ||
        !matvec(cache, weights.backend, layer.index_k, input.data(),
                kEmbedding, ik, error)) return false;
    std::vector<float> knorm;
    if (!tensor_f32(cache, layer.attn_k_norm, knorm, error)) return false;
    for (int head = 0; head < kQsaKvHeads; ++head) {
        rms_norm(k.data() + head * kQsaDim, kQsaDim, knorm.data());
        if (!rope(weights, k.data() + head * kQsaDim, kQsaDim,
                  position.data(), error)) return false;
    }
    // PR #27774 (abdc7a0b over #27742 035e2273): cache-side Hadamard
    // transforms are part of the stored representation, not query output.
    if (!rotate_optional(cache, weights.backend, layer.self_k_rot, k, error) ||
        !rotate_optional(cache, weights.backend, layer.self_v_rot, v, error))
        return false;
    state.key.append(k.data(), k.size());
    state.value.append(v.data(), v.size());
    state.index_key.append(ik.data(), ik.size());
    return true;
}

bool run_qsa_scalar(
             const Qwen4ExpWeights & weights, Qwen4ExpLayerState & state,
             const Qwen4ExpLayer & layer, const std::vector<float> & input,
             const std::array<int32_t, 3> & position,
             const std::array<std::vector<int32_t>, 3> & position_history,
             std::vector<float> & output, std::string & error,
             Qwen4ExpFrontierDenseCache * cache_override = nullptr) {
    Qwen4ExpFrontierDenseCache * cache =
        cache_override ? cache_override : weights.dense_cache;
    std::vector<float> qfull, iq;
    const int tokens = static_cast<int>(state.index_key.size() / kIndexerDim) + 1;
    std::vector<int32_t> selected;
    const bool dense_selection =
        qwen4exp_qsa_dense_selection(tokens, selected);
    if (!matvec(cache, weights.backend, layer.attn_q, input.data(), kEmbedding,
                qfull, error)) return false;
    if (!dense_selection &&
        !matvec(cache, weights.backend, layer.index_q, input.data(), kEmbedding,
                iq, error)) return false;
    std::vector<float> qnorm;
    if (!tensor_f32(cache, layer.attn_q_norm, qnorm, error)) return false;
    std::vector<float> iqnorm, iknorm;
    if (!dense_selection &&
        (!tensor_f32(cache, layer.index_q_norm, iqnorm, error) ||
         !tensor_f32(cache, layer.index_k_norm, iknorm, error))) return false;
    std::vector<float> q(kQsaHeads * kQsaDim), gate(kQsaHeads * kQsaDim);
    for (int head = 0; head < kQsaHeads; ++head) {
        std::copy_n(qfull.data() + head * 2 * kQsaDim, kQsaDim,
                    q.data() + head * kQsaDim);
        std::copy_n(qfull.data() + head * 2 * kQsaDim + kQsaDim, kQsaDim,
                    gate.data() + head * kQsaDim);
        rms_norm(q.data() + head * kQsaDim, kQsaDim, qnorm.data());
        if (!rope(weights, q.data() + head * kQsaDim, kQsaDim,
                  position.data(), error)) return false;
    }
    if (!dense_selection) {
        for (int head = 0; head < kIndexerHeads; ++head) {
            rms_norm(iq.data() + head * kIndexerDim, kIndexerDim,
                     iqnorm.data());
            if (!rope(weights, iq.data() + head * kIndexerDim, kIndexerDim,
                      position.data(), error)) return false;
        }
    }
    // Q uses the same self-inverse K transform as the stored K cache.
    if (!rotate_optional(cache, weights.backend, layer.self_k_rot, q, error) ||
        !append_qsa_cache(weights, state, layer, input, position, error,
                          cache))
        return false;
    // Score complete four-token blocks from the raw index-K cache. Pooling is
    // deliberately before learned RMSNorm and RoPE, matching #27742/HF. The
    // selection is host-side because the released 2048-token budget means
    // top-512 blocks and must not pass through a 1023-element-capped primitive.
    if (!dense_selection) {
        const int complete = tokens / 4;
        const int keep = std::min(complete, 512);
        std::vector<std::pair<float, int32_t>> scored;
        scored.reserve(static_cast<size_t>(complete));
        for (int block_index = 0; block_index < complete; ++block_index) {
            std::array<float, kIndexerDim> pooled{};
            for (int member = 0; member < 4; ++member)
                for (int d = 0; d < kIndexerDim; ++d)
                    pooled[d] += state.index_key.at(static_cast<size_t>(
                        (block_index * 4 + member) * kIndexerDim + d)) * 0.25f;
            rms_norm(pooled.data(), kIndexerDim, iknorm.data());
            const size_t group_start = static_cast<size_t>(block_index * 4);
            if (group_start >= position_history[0].size() ||
                position_history[1].size() != position_history[0].size() ||
                position_history[2].size() != position_history[0].size()) {
                error = "Qwen4Exp M-RoPE history does not cover QSA block";
                return false;
            }
            const int32_t group_position[3] = {
                position_history[0][group_start],
                position_history[1][group_start],
                position_history[2][group_start],
            };
            if (!rope(weights, pooled.data(), kIndexerDim, group_position,
                      error)) return false;
            float score = 0.0f;
            for (int head = 0; head < kIndexerHeads; ++head) {
                float dot = 0.0f;
                for (int d = 0; d < kIndexerDim; ++d)
                    dot += iq[head * kIndexerDim + d] * pooled[d];
                score += std::max(dot, 0.0f);
            }
            scored.emplace_back(
                score / std::sqrt(static_cast<float>(kIndexerDim)),
                block_index);
        }
        if (keep < complete)
            std::partial_sort(scored.begin(), scored.begin() + keep,
                              scored.end(), [](const auto & a, const auto & b) {
                                  return a.first != b.first
                                      ? a.first > b.first : a.second < b.second;
                              });
        selected.reserve(static_cast<size_t>(keep * 4 + tokens % 4));
        for (int i = 0; i < keep; ++i)
            for (int member = 0; member < 4; ++member)
                selected.push_back(scored[i].second * 4 + member);
        for (int tail = complete * 4; tail < tokens; ++tail)
            selected.push_back(tail);
    }
    std::vector<float> attended(kQsaHeads * kQsaDim);
    for (int head = 0; head < kQsaHeads; ++head) {
        const int kv_head = head / 12;
        std::vector<float> scores(selected.size());
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t s = 0; s < selected.size(); ++s) {
            const size_t cached =
                (static_cast<size_t>(selected[s]) * kQsaKvHeads + kv_head) * kQsaDim;
            float dot = 0.0f;
            for (int d = 0; d < kQsaDim; ++d)
                dot += q[head * kQsaDim + d] * state.key.at(cached + d);
            scores[s] = dot / std::sqrt(static_cast<float>(kQsaDim));
            max_score = std::max(max_score, scores[s]);
        }
        float denominator = 0.0f;
        for (float & score : scores) { score = std::exp(score - max_score); denominator += score; }
        for (size_t s = 0; s < selected.size(); ++s) {
            const size_t cached =
                (static_cast<size_t>(selected[s]) * kQsaKvHeads + kv_head) * kQsaDim;
            const float probability = scores[s] / denominator;
            for (int d = 0; d < kQsaDim; ++d)
                attended[head * kQsaDim + d] +=
                    probability * state.value.at(cached + d);
        }
    }
    // The same V rotation is self-inverse and applies again after attention.
    if (!rotate_optional(cache, weights.backend, layer.self_v_rot, attended,
                         error)) return false;
    for (size_t i = 0; i < attended.size(); ++i)
        attended[i] *= sigmoid(gate[i]);
    return matvec(cache, weights.backend, layer.attn_output, attended.data(),
                  static_cast<int>(attended.size()), output, error);
}

bool prepare_qsa_row(
        const Qwen4ExpWeights & weights,
        const std::array<int32_t, 3> & position,
        const std::vector<float> & qnorm, const std::vector<float> & knorm,
        const std::vector<float> & iqnorm, const float * qfull,
        const float * key, const float * value, const float * index_query,
        const float * index_key, std::vector<float> & query,
        std::vector<float> & gate, std::vector<float> & prepared_key,
        std::vector<float> & prepared_value,
        std::vector<float> & prepared_index_query,
        std::vector<float> & prepared_index_key, std::string & error) {
    if (!qfull || !key || !value || !index_query || !index_key ||
        qnorm.size() != static_cast<size_t>(kQsaDim) ||
        knorm.size() != static_cast<size_t>(kQsaDim) ||
        iqnorm.size() != static_cast<size_t>(kIndexerDim)) {
        error = "invalid Qwen4Exp prepared QSA row";
        return false;
    }
    query.resize(static_cast<size_t>(kQsaHeads * kQsaDim));
    gate.resize(query.size());
    for (int head = 0; head < kQsaHeads; ++head) {
        std::copy_n(qfull + head * 2 * kQsaDim, kQsaDim,
                    query.data() + head * kQsaDim);
        std::copy_n(qfull + head * 2 * kQsaDim + kQsaDim, kQsaDim,
                    gate.data() + head * kQsaDim);
        rms_norm(query.data() + head * kQsaDim, kQsaDim, qnorm.data());
        if (!rope(weights, query.data() + head * kQsaDim, kQsaDim,
                  position.data(), error)) return false;
    }
    prepared_key.assign(key, key + kQsaKvHeads * kQsaDim);
    prepared_value.assign(value, value + kQsaKvHeads * kQsaDim);
    for (int head = 0; head < kQsaKvHeads; ++head) {
        rms_norm(prepared_key.data() + head * kQsaDim, kQsaDim,
                 knorm.data());
        if (!rope(weights, prepared_key.data() + head * kQsaDim, kQsaDim,
                  position.data(), error)) return false;
    }
    prepared_index_query.assign(
        index_query, index_query + kIndexerHeads * kIndexerDim);
    for (int head = 0; head < kIndexerHeads; ++head) {
        rms_norm(prepared_index_query.data() + head * kIndexerDim,
                 kIndexerDim, iqnorm.data());
        if (!rope(weights,
                  prepared_index_query.data() + head * kIndexerDim,
                  kIndexerDim, position.data(), error)) return false;
    }
    prepared_index_key.assign(index_key, index_key + kIndexerDim);
    return true;
}

bool finish_qsa_row(
        const Qwen4ExpWeights & weights, Qwen4ExpLayerState & state,
        const Qwen4ExpLayer & layer, Qwen4ExpFrontierQsaGraph * graph,
        Qwen4ExpFrontierDenseCache * cache,
        const std::vector<float> & query, const std::vector<float> & gate,
        const std::vector<float> & key, const std::vector<float> & value,
        const std::vector<float> & index_query,
        const std::vector<float> & index_key,
        const std::array<std::vector<int32_t>, 3> & position_history,
        std::vector<float> & output, std::string & error) {
    if (!graph || query.size() != static_cast<size_t>(kQsaHeads * kQsaDim) ||
        gate.size() != query.size() ||
        key.size() != static_cast<size_t>(kQsaKvHeads * kQsaDim) ||
        value.size() != key.size() ||
        index_query.size() !=
            static_cast<size_t>(kIndexerHeads * kIndexerDim) ||
        index_key.size() != static_cast<size_t>(kIndexerDim)) {
        error = "invalid Qwen4Exp finalized QSA row";
        return false;
    }
    const int prior_tokens = static_cast<int>(
        state.index_key.size() / static_cast<size_t>(kIndexerDim));
    const int tokens = prior_tokens + 1;
    std::vector<int32_t> selected;
    const bool dense_selection =
        qwen4exp_qsa_dense_selection(tokens, selected);
    if (!dense_selection) {
        std::vector<float> iknorm;
        if (!tensor_f32(cache, layer.index_k_norm, iknorm, error) ||
            iknorm.size() != static_cast<size_t>(kIndexerDim)) return false;
        const int complete = tokens / 4;
        const int keep = std::min(complete, 512);
        std::vector<std::pair<float, int32_t>> scored;
        scored.reserve(static_cast<size_t>(complete));
        const auto raw_index = [&](int token, int dimension) {
            return token == prior_tokens
                ? index_key[static_cast<size_t>(dimension)]
                : state.index_key.at(static_cast<size_t>(
                    token * kIndexerDim + dimension));
        };
        for (int block_index = 0; block_index < complete; ++block_index) {
            std::array<float, kIndexerDim> pooled{};
            for (int member = 0; member < 4; ++member)
                for (int dimension = 0; dimension < kIndexerDim;
                     ++dimension) {
                    pooled[static_cast<size_t>(dimension)] += raw_index(
                        block_index * 4 + member, dimension) * 0.25f;
                }
            rms_norm(pooled.data(), kIndexerDim, iknorm.data());
            const size_t group_start = static_cast<size_t>(block_index * 4);
            if (group_start >= position_history[0].size() ||
                position_history[1].size() != position_history[0].size() ||
                position_history[2].size() != position_history[0].size()) {
                error = "Qwen4Exp M-RoPE history does not cover QSA block";
                return false;
            }
            const int32_t group_position[3] = {
                position_history[0][group_start],
                position_history[1][group_start],
                position_history[2][group_start],
            };
            if (!rope(weights, pooled.data(), kIndexerDim, group_position,
                      error)) return false;
            float score = 0.0f;
            for (int head = 0; head < kIndexerHeads; ++head) {
                float dot = 0.0f;
                for (int dimension = 0; dimension < kIndexerDim;
                     ++dimension) {
                    dot += index_query[static_cast<size_t>(
                               head * kIndexerDim + dimension)] *
                           pooled[static_cast<size_t>(dimension)];
                }
                score += std::max(dot, 0.0f);
            }
            scored.emplace_back(
                score / std::sqrt(static_cast<float>(kIndexerDim)),
                block_index);
        }
        if (keep < complete) {
            std::partial_sort(
                scored.begin(), scored.begin() + keep, scored.end(),
                [](const auto & a, const auto & b) {
                    return a.first != b.first ? a.first > b.first
                                              : a.second < b.second;
                });
        }
        selected.reserve(static_cast<size_t>(keep * 4 + tokens % 4));
        for (int index = 0; index < keep; ++index)
            for (int member = 0; member < 4; ++member)
                selected.push_back(scored[static_cast<size_t>(index)].second *
                                   4 + member);
        for (int tail = complete * 4; tail < tokens; ++tail)
            selected.push_back(tail);
    }

    // Pack exact causal rows only after selection. The current projected row
    // remains private until attention and its output projection both finish.
    const size_t selected_count = selected.size();
    std::vector<float> selected_key(
        static_cast<size_t>(kQsaKvHeads * kQsaDim) * selected_count);
    std::vector<float> selected_value(selected_key.size());
    for (int head = 0; head < kQsaKvHeads; ++head) {
        for (size_t slot = 0; slot < selected_count; ++slot) {
            const int token = selected[slot];
            const size_t destination =
                (static_cast<size_t>(head) * selected_count + slot) * kQsaDim;
            if (token == prior_tokens) {
                std::copy_n(key.data() + head * kQsaDim, kQsaDim,
                            selected_key.data() + destination);
                std::copy_n(value.data() + head * kQsaDim, kQsaDim,
                            selected_value.data() + destination);
            } else {
                const size_t source =
                    (static_cast<size_t>(token) * kQsaKvHeads +
                     static_cast<size_t>(head)) * kQsaDim;
                for (int dimension = 0; dimension < kQsaDim; ++dimension) {
                    selected_key[destination +
                                 static_cast<size_t>(dimension)] =
                        state.key.at(source + static_cast<size_t>(dimension));
                    selected_value[destination +
                                   static_cast<size_t>(dimension)] =
                        state.value.at(source +
                                       static_cast<size_t>(dimension));
                }
            }
        }
    }
    if (!qwen4exp_frontier_qsa_attend_q1(
            graph, query.data(), query.size(), gate.data(), gate.size(),
            selected_key.data(), selected_value.data(),
            static_cast<int>(selected_count), output, error)) return false;
    state.key.append(key.data(), key.size());
    state.value.append(value.data(), value.size());
    state.index_key.append(index_key.data(), index_key.size());
    return true;
}

bool run_qsa(const Qwen4ExpWeights & weights, Qwen4ExpLayerState & state,
             const Qwen4ExpLayer & layer, int layer_index,
             const std::vector<float> & input,
             const std::array<int32_t, 3> & position,
             const std::array<std::vector<int32_t>, 3> & position_history,
             std::vector<float> & output, std::string & error,
             Qwen4ExpFrontierDenseCache * cache_override = nullptr,
             Qwen4ExpFrontierQsaGraph * graph_override = nullptr) {
    Qwen4ExpFrontierDenseCache * cache =
        cache_override ? cache_override : weights.dense_cache;
    Qwen4ExpFrontierQsaGraph * graph = graph_override
        ? graph_override
        : (cache_override ? nullptr
                          : qwen4exp_frontier_qsa_q1(weights, layer_index));
    if (!graph) {
        return run_qsa_scalar(weights, state, layer, input, position,
                              position_history, output, error,
                              cache_override);
    }

    std::vector<float> qfull, k, v, iq, ik;
    if (!qwen4exp_frontier_qsa_project_q1(
            graph, input.data(), input.size(), qfull, k, v, iq, ik,
            error)) return false;

    std::vector<float> qnorm, knorm, iqnorm;
    if (!tensor_f32(cache, layer.attn_q_norm, qnorm, error) ||
        !tensor_f32(cache, layer.attn_k_norm, knorm, error) ||
        !tensor_f32(cache, layer.index_q_norm, iqnorm, error))
        return false;
    std::vector<float> q, gate, prepared_k, prepared_v, prepared_iq,
                       prepared_ik;
    if (!prepare_qsa_row(weights, position, qnorm, knorm, iqnorm,
                         qfull.data(), k.data(), v.data(), iq.data(),
                         ik.data(), q, gate, prepared_k, prepared_v,
                         prepared_iq, prepared_ik, error)) return false;
    // #27774 rotations are executed as one persistent graph for all Q/K/V
    // heads. The rotated current K/V are not published until attention and
    // output projection have both completed successfully.
    if (!qwen4exp_frontier_qsa_rotate_q1(
            graph, q, prepared_k, prepared_v, error)) return false;
    return finish_qsa_row(weights, state, layer, graph, cache, q, gate,
                          prepared_k, prepared_v, prepared_iq, prepared_ik,
                          position_history, output, error);
}

bool run_qsa_batch(
        const Qwen4ExpWeights & weights, Qwen4ExpLayerState & state,
        const Qwen4ExpLayer & layer, int layer_index,
        const std::vector<float> & input,
        const std::vector<std::array<int32_t, 3>> & positions,
        const std::array<std::vector<int32_t>, 3> & position_history,
        std::vector<float> & output, std::string & error) {
    const size_t rows = positions.size();
    if (rows == 0 || rows > 16 ||
        input.size() != rows * static_cast<size_t>(kEmbedding)) {
        error = "invalid Qwen4Exp batched QSA input";
        return false;
    }
    Qwen4ExpFrontierQsaGraph * graph =
        qwen4exp_frontier_qsa_q1(weights, layer_index);
    if (!graph) {
        output.resize(rows * static_cast<size_t>(kEmbedding));
        for (size_t row = 0; row < rows; ++row) {
            const auto begin = input.begin() + static_cast<std::ptrdiff_t>(
                row * static_cast<size_t>(kEmbedding));
            const std::vector<float> row_input(
                begin, begin + static_cast<std::ptrdiff_t>(kEmbedding));
            std::vector<float> row_output;
            if (!run_qsa(weights, state, layer, layer_index, row_input,
                         positions[row], position_history, row_output,
                         error)) return false;
            std::copy(row_output.begin(), row_output.end(),
                      output.begin() + static_cast<std::ptrdiff_t>(
                          row * static_cast<size_t>(kEmbedding)));
        }
        return true;
    }

    constexpr size_t kQueryGateValues =
        static_cast<size_t>(2 * kQsaHeads * kQsaDim);
    constexpr size_t kKvValues =
        static_cast<size_t>(kQsaKvHeads * kQsaDim);
    constexpr size_t kIndexQueryValues =
        static_cast<size_t>(kIndexerHeads * kIndexerDim);
    constexpr size_t kIndexKeyValues = static_cast<size_t>(kIndexerDim);
    std::vector<float> qfull, key, value, index_query, index_key;
    if (!matmul_rows(weights.dense_cache, weights.backend, layer.attn_q,
                     input.data(), kEmbedding, static_cast<int>(rows),
                     qfull, error) ||
        !matmul_rows(weights.dense_cache, weights.backend, layer.attn_k,
                     input.data(), kEmbedding, static_cast<int>(rows), key,
                     error) ||
        !matmul_rows(weights.dense_cache, weights.backend, layer.attn_v,
                     input.data(), kEmbedding, static_cast<int>(rows), value,
                     error) ||
        !matmul_rows(weights.dense_cache, weights.backend, layer.index_q,
                     input.data(), kEmbedding, static_cast<int>(rows),
                     index_query, error) ||
        !matmul_rows(weights.dense_cache, weights.backend, layer.index_k,
                     input.data(), kEmbedding, static_cast<int>(rows),
                     index_key, error) ||
        qfull.size() != rows * kQueryGateValues ||
        key.size() != rows * kKvValues || value.size() != key.size() ||
        index_query.size() != rows * kIndexQueryValues ||
        index_key.size() != rows * kIndexKeyValues) {
        if (error.empty()) error = "Qwen4Exp batched QSA projection mismatch";
        return false;
    }
    std::vector<float> qnorm, knorm, iqnorm;
    if (!tensor_f32(weights.dense_cache, layer.attn_q_norm, qnorm, error) ||
        !tensor_f32(weights.dense_cache, layer.attn_k_norm, knorm, error) ||
        !tensor_f32(weights.dense_cache, layer.index_q_norm, iqnorm, error))
        return false;

    output.resize(rows * static_cast<size_t>(kEmbedding));
    for (size_t row = 0; row < rows; ++row) {
        std::vector<float> query_row, gate_row, key_row, value_row,
                           index_query_row, index_key_row;
        if (!prepare_qsa_row(
                weights, positions[row], qnorm, knorm, iqnorm,
                qfull.data() + row * kQueryGateValues,
                key.data() + row * kKvValues,
                value.data() + row * kKvValues,
                index_query.data() + row * kIndexQueryValues,
                index_key.data() + row * kIndexKeyValues,
                query_row, gate_row, key_row, value_row, index_query_row,
                index_key_row, error) ||
            !qwen4exp_frontier_qsa_rotate_q1(
                graph, query_row, key_row, value_row, error)) return false;
        std::vector<float> row_output;
        if (!finish_qsa_row(weights, state, layer, graph,
                            weights.dense_cache, query_row, gate_row, key_row,
                            value_row, index_query_row, index_key_row,
                            position_history, row_output, error) ||
            row_output.size() != static_cast<size_t>(kEmbedding)) {
            if (error.empty()) error = "Qwen4Exp batched QSA output mismatch";
            return false;
        }
        std::copy(row_output.begin(), row_output.end(),
                  output.begin() + static_cast<std::ptrdiff_t>(
                      row * static_cast<size_t>(kEmbedding)));
    }
    return true;
}

bool run_moe(const Qwen4ExpWeights & weights, int layer_index,
             const Qwen4ExpLayer & layer,
             const std::vector<float> & input, std::vector<float> & output,
             std::string & error) {
    if (layer_index < 0 || layer_index >= 48) {
        error = "Qwen4Exp target MoE layer index is out of range";
        return false;
    }
    if (qwen4exp_frontier_moe_available(weights, layer_index)) {
        return qwen4exp_frontier_moe_q1(weights, layer_index, input.data(),
                                        input.size(), output, error);
    }
    std::vector<float> router;
    if (!matvec(weights.dense_cache, weights.backend, layer.router,
                input.data(), kEmbedding, router, error) ||
        router.size() != kExpertCount) return false;
    std::array<int32_t, kExpertCount> ids{};
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + kExpertUsed, ids.end(),
        [&](int32_t a, int32_t b) { return router[a] > router[b]; });
    const float max_logit = *std::max_element(router.begin(), router.end());
    double all_sum = 0.0;
    for (float value : router) all_sum += std::exp(value - max_logit);
    std::array<float, kExpertUsed> selected_weight{};
    float selected_sum = 0.0f;
    for (int i = 0; i < kExpertUsed; ++i) {
        selected_weight[i] = static_cast<float>(
            std::exp(router[ids[i]] - max_logit) / all_sum);
        selected_sum += selected_weight[i];
    }
    // Released config normalizes top-k probabilities.
    for (float & value : selected_weight) value /= selected_sum;
    output.assign(kEmbedding, 0.0f);
    for (int i = 0; i < kExpertUsed; ++i) {
        std::vector<float> gate, up, down;
        const Qwen4ExpMappedTensor dw = expert_slice(layer.experts_down, ids[i]);
        if (layer.experts_gate.valid() && layer.experts_up.valid()) {
            const Qwen4ExpMappedTensor gw =
                expert_slice(layer.experts_gate, ids[i]);
            const Qwen4ExpMappedTensor uw =
                expert_slice(layer.experts_up, ids[i]);
            if (!mapped_matvec(gw, input.data(), gate, error) ||
                !mapped_matvec(uw, input.data(), up, error) ||
                gate.size() != kExpertFf || up.size() != kExpertFf) return false;
        } else {
            std::vector<float> gate_up;
            const Qwen4ExpMappedTensor gu =
                expert_slice(layer.experts_gate_up, ids[i]);
            if (!mapped_matvec(gu, input.data(), gate_up, error) ||
                gate_up.size() != 2 * kExpertFf) return false;
            gate.assign(gate_up.begin(), gate_up.begin() + kExpertFf);
            up.assign(gate_up.begin() + kExpertFf, gate_up.end());
        }
        std::vector<float> intermediate(kExpertFf);
        for (int j = 0; j < kExpertFf; ++j)
            intermediate[j] = silu(gate[j]) * up[j];
        if (!mapped_matvec(dw, intermediate.data(), down, error) ||
            down.size() != kEmbedding) return false;
        for (int j = 0; j < kEmbedding; ++j)
            output[j] += selected_weight[i] * down[j];
    }
    std::vector<float> shared_gate, shared_up, shared_down, shared_scale;
    if (!matvec(weights.dense_cache, weights.backend, layer.shared_gate,
                input.data(), kEmbedding, shared_gate, error) ||
        !matvec(weights.dense_cache, weights.backend, layer.shared_up,
                input.data(), kEmbedding, shared_up, error) ||
        shared_gate.size() != kExpertFf ||
        shared_up.size() != kExpertFf) return false;
    for (int i = 0; i < kExpertFf; ++i) shared_gate[i] = silu(shared_gate[i]) * shared_up[i];
    if (!matvec(weights.dense_cache, weights.backend, layer.shared_down,
                shared_gate.data(), kExpertFf, shared_down, error) ||
        !matvec(weights.dense_cache, weights.backend, layer.shared_gate_input,
                input.data(), kEmbedding, shared_scale, error) ||
        shared_scale.size() != 1)
        return false;
    const float scale = sigmoid(shared_scale[0]);
    for (int i = 0; i < kEmbedding; ++i) output[i] += scale * shared_down[i];
    return true;
}

} // namespace

static bool step_q1_embedding(const Qwen4ExpWeights & weights,
                              Qwen4ExpState & state, int32_t token,
                              const float * supplied_embedding,
                              const std::array<int32_t, 3> & position,
                              std::vector<float> & logits,
                              std::vector<float> * writer_output_capture,
                              std::string & error) {
    if (token < 0 || token >= 248320 || state.cur_pos < 0 ||
        state.cur_pos >= weights.max_ctx || weights.layers.size() != 48) {
        error = "invalid Qwen4Exp q=1 token/frontier"; return false;
    }
    if (writer_output_capture) {
        writer_output_capture->clear();
        writer_output_capture->reserve(48 * kEmbedding);
    }
    std::vector<float> embedding(kEmbedding);
    if (supplied_embedding) {
        std::copy_n(supplied_embedding, kEmbedding, embedding.data());
    } else if (!weights.embedder.embed(&token, 1, embedding.data())) {
        error = "Qwen4Exp token embedding lookup failed"; return false;
    }
    for (size_t axis = 0; axis < 3; ++axis)
        state.mrope_positions[axis].push_back(position[axis]);
    if (state.hc.empty()) {
        state.hc.resize(kHcDim);
        for (int stream = 0; stream < kHc; ++stream)
            std::copy(embedding.begin(), embedding.end(),
                      state.hc.begin() + stream * kEmbedding);
    } else {
        // A q=1 step consumes one new token embedding, so the incoming residual
        // starts from that embedding just as a one-row full graph does.
        for (int stream = 0; stream < kHc; ++stream)
            std::copy(embedding.begin(), embedding.end(),
                      state.hc.begin() + stream * kEmbedding);
    }
    for (int layer_index = 0; layer_index < 48; ++layer_index) {
        const Qwen4ExpLayer & layer = weights.layers[static_cast<size_t>(layer_index)];
        if (layer_index == 1 && !run_ple(weights, state, layer, token, error)) return false;
        std::vector<float> mixed, block;
        std::array<float, kHc> inject{};
        if (!hc_mix(weights, state.hc, layer.hc_attn_norm, layer.hc_attn_down,
                    layer.hc_attn_up, layer.hc_attn_inject, mixed, &inject,
                    error)) return false;
        const bool qsa = (layer_index + 1) % 4 == 0;
        if (qsa) {
            if (!run_qsa(weights, state.layers[static_cast<size_t>(layer_index)],
                          layer, layer_index, mixed, position,
                          state.mrope_positions,
                          block, error)) return false;
        } else if (!run_gdn(
                       weights,
                       state.layers[static_cast<size_t>(layer_index)], layer,
                       layer_index, mixed, block, error)) return false;
        // Heretic applies each direction to this layer's residual writer
        // (linear_attn.out_proj/ssm_out or self_attn.o_proj/attn_output).
        // Capture that writer's 2560-wide output, before HC injection, rather
        // than the same-width attention input.  The output already contains
        // GDN/QSA context, including at layer zero.
        if (writer_output_capture)
            writer_output_capture->insert(writer_output_capture->end(),
                                           block.begin(), block.end());
        hc_combine(state.hc, block, inject);
        if (!hc_mix(weights, state.hc, layer.hc_ffn_norm, layer.hc_ffn_down,
                    layer.hc_ffn_up, layer.hc_ffn_inject, mixed, &inject,
                    error) || !run_moe(weights, layer_index, layer, mixed,
                                       block, error)) return false;
        hc_combine(state.hc, block, inject);
    }
    if (!hc_output_rows(
            weights, weights.dense_cache, weights.output_hc_norm,
            weights.output_hc_down, weights.output_hc_up, weights.output,
            state.hc.data(), state.hc.size(), 1, logits, error)) return false;
    ++state.cur_pos;
    state.last_token = token;
    return true;
}

bool qwen4exp_step_q1(const Qwen4ExpWeights & weights,
                      Qwen4ExpState & state, int32_t token,
                      std::vector<float> & logits, std::string & error) {
    const int32_t p = state.cur_pos;
    return step_q1_embedding(weights, state, token, nullptr, {p, p, p},
                             logits, nullptr, error);
}

bool qwen4exp_step_q1_embedding(
        const Qwen4ExpWeights & weights, Qwen4ExpState & state, int32_t token,
        const float * embedding, size_t embedding_count,
        const std::array<int32_t, 3> & mrope_position,
        std::vector<float> & logits, std::string & error) {
    if (!embedding || embedding_count != kEmbedding) {
        error = "Qwen4Exp image embedding row must contain 2560 values";
        return false;
    }
    return step_q1_embedding(weights, state, token, embedding, mrope_position,
                             logits, nullptr, error);
}

bool qwen4exp_step_q1_mrope(
        const Qwen4ExpWeights & weights, Qwen4ExpState & state, int32_t token,
        const std::array<int32_t, 3> & mrope_position,
        std::vector<float> & logits, std::string & error) {
    return step_q1_embedding(weights, state, token, nullptr, mrope_position,
                             logits, nullptr, error);
}

bool qwen4exp_step_q1_mrope_capture(
        const Qwen4ExpWeights & weights, Qwen4ExpState & state, int32_t token,
        const std::array<int32_t, 3> & mrope_position,
        std::vector<float> & logits, std::vector<float> & writer_output_capture,
        std::string & error) {
    return step_q1_embedding(weights, state, token, nullptr, mrope_position,
                             logits, &writer_output_capture, error);
}

namespace {
bool prepare_mtp_hc(const Qwen4ExpWeights & target,
                    const Qwen4ExpMtpWeights & mtp, int32_t token,
                    const float * next_embedding, size_t next_embedding_count,
                    const float * target_hc, size_t target_hc_count,
                    std::vector<float> & hc, std::string & error) {
    if (token < 0 || token >= 248320 || !target_hc ||
        target_hc_count != static_cast<size_t>(kHcDim) ||
        !mtp.pre_embedding_norm || !mtp.pre_hc_norm || !mtp.fc_embedding ||
        !mtp.fc_hc) {
        error = "invalid Qwen4Exp MTP token/frontier contract";
        return false;
    }
    std::vector<float> embedding(kEmbedding);
    if (next_embedding) {
        if (next_embedding_count != static_cast<size_t>(kEmbedding)) {
            error = "Qwen4Exp MTP supplied embedding must contain 2560 values";
            return false;
        }
        std::copy_n(next_embedding, kEmbedding, embedding.data());
    } else if (next_embedding_count != 0 ||
               !target.embedder.embed(&token, 1, embedding.data())) {
        error = "Qwen4Exp MTP token embedding lookup failed";
        return false;
    }
    std::vector<float> embedding_norm, hc_norm;
    if (!tensor_f32(mtp.dense_cache, mtp.pre_embedding_norm, embedding_norm,
                    error) ||
        embedding_norm.size() != static_cast<size_t>(kEmbedding) ||
        !tensor_f32(mtp.dense_cache, mtp.pre_hc_norm, hc_norm, error) ||
        hc_norm.size() != static_cast<size_t>(kHcDim)) return false;

    rms_norm(embedding.data(), kEmbedding, embedding_norm.data());
    std::vector<float> projected_embedding;
    if (!matvec(mtp.dense_cache, target.backend, mtp.fc_embedding,
                embedding.data(), kEmbedding, projected_embedding, error) ||
        projected_embedding.size() != static_cast<size_t>(kEmbedding))
        return false;

    // Qwen4Exp MTP normalizes h_p globally across all four HC streams before
    // applying one shared 2560x2560 projection to each stream. This differs
    // from the stream-local normalization used by ordinary HC mixers.
    std::vector<float> normalized_hc(target_hc, target_hc + kHcDim);
    rms_norm(normalized_hc.data(), kHcDim, hc_norm.data());
    std::vector<float> projected_hidden;
    if (!matmul_rows(mtp.dense_cache, target.backend, mtp.fc_hc,
                     normalized_hc.data(), kEmbedding, kHc,
                     projected_hidden, error) ||
        projected_hidden.size() != static_cast<size_t>(kHcDim)) return false;
    hc.resize(kHcDim);
    for (int stream = 0; stream < kHc; ++stream) {
        for (int channel = 0; channel < kEmbedding; ++channel) {
            hc[static_cast<size_t>(stream * kEmbedding + channel)] =
                projected_hidden[static_cast<size_t>(
                    stream * kEmbedding + channel)] +
                projected_embedding[static_cast<size_t>(channel)];
        }
    }
    return true;
}

bool prepare_mtp_hc_batch(
        const Qwen4ExpWeights & target, const Qwen4ExpMtpWeights & mtp,
        const std::vector<int32_t> & tokens,
        const std::vector<std::vector<float>> & target_hc_rows,
        std::vector<float> & hc_rows, std::string & error) {
    Qwen4ExpMtpCacheBatchShape shape;
    if (!qwen4exp_mtp_cache_batch_shape(tokens.size(), shape, error) ||
        target_hc_rows.size() != shape.rows || !mtp.pre_embedding_norm ||
        !mtp.pre_hc_norm || !mtp.fc_embedding || !mtp.fc_hc) {
        if (error.empty()) error = "invalid Qwen4Exp MTP cache batch inputs";
        return false;
    }
    for (size_t row = 0; row < shape.rows; ++row) {
        if (tokens[row] < 0 || tokens[row] >= 248320 ||
            target_hc_rows[row].size() != 10240U) {
            error = "invalid Qwen4Exp MTP cache batch row";
            return false;
        }
    }

    std::vector<float> embeddings(shape.embedding_values);
    if (!target.embedder.embed(tokens.data(), static_cast<int>(shape.rows),
                               embeddings.data())) {
        error = "Qwen4Exp MTP cache batch embedding lookup failed";
        return false;
    }
    std::vector<float> embedding_norm, hc_norm;
    if (!tensor_f32(mtp.dense_cache, mtp.pre_embedding_norm, embedding_norm,
                    error) ||
        embedding_norm.size() != 2560U ||
        !tensor_f32(mtp.dense_cache, mtp.pre_hc_norm, hc_norm, error) ||
        hc_norm.size() != 10240U) return false;
    for (size_t row = 0; row < shape.rows; ++row) {
        rms_norm(embeddings.data() + row * 2560U, kEmbedding,
                 embedding_norm.data());
    }
    std::vector<float> projected_embeddings;
    if (!matmul_rows(mtp.dense_cache, target.backend, mtp.fc_embedding,
                     embeddings.data(), kEmbedding,
                     static_cast<int>(shape.rows),
                     projected_embeddings, error) ||
        projected_embeddings.size() != shape.embedding_values) return false;

    std::vector<float> normalized_hc(shape.target_hc_values);
    for (size_t row = 0; row < shape.rows; ++row) {
        float * destination = normalized_hc.data() + row * 10240U;
        std::copy(target_hc_rows[row].begin(), target_hc_rows[row].end(),
                  destination);
        rms_norm(destination, kHcDim, hc_norm.data());
    }
    std::vector<float> projected_hidden;
    if (!matmul_rows(mtp.dense_cache, target.backend, mtp.fc_hc,
                     normalized_hc.data(), kEmbedding,
                     static_cast<int>(shape.hc_projection_rows),
                     projected_hidden, error) ||
        projected_hidden.size() != shape.target_hc_values) return false;

    hc_rows.resize(shape.target_hc_values);
    for (size_t row = 0; row < shape.rows; ++row) {
        for (size_t stream = 0; stream < 4U; ++stream) {
            for (size_t channel = 0; channel < 2560U; ++channel) {
                const size_t offset = (row * 4U + stream) * 2560U + channel;
                hc_rows[offset] = projected_hidden[offset] +
                    projected_embeddings[row * 2560U + channel];
            }
        }
    }
    return true;
}

bool hc_mix_rows(const Qwen4ExpWeights & target,
                 Qwen4ExpFrontierDenseCache * cache,
                 ggml_tensor * norm, ggml_tensor * down, ggml_tensor * up,
                 ggml_tensor * inject_weight, size_t rows,
                 const std::vector<float> & hc_rows,
                 std::vector<float> & mixed_rows,
                 std::vector<std::array<float, kHc>> * inject_rows,
                 std::string & error) {
    if (!norm || !down || !up || (inject_rows && !inject_weight) ||
        rows == 0 || rows > 16 ||
        hc_rows.size() != rows * static_cast<size_t>(kHcDim)) {
        error = "invalid Qwen4Exp HC row batch shape";
        return false;
    }
    const Qwen4ExpFrontierHcSpec spec{
        kEmbedding, kHc, kEpsilon};
    std::vector<float> raw_injection;
    if (!qwen4exp_frontier_hc_eval(
            cache, target.backend, spec, norm, down, up, inject_weight,
            hc_rows.data(), hc_rows.size(), static_cast<int>(rows),
            mixed_rows, inject_rows ? &raw_injection : nullptr, error))
        return false;
    if (mixed_rows.size() != rows * static_cast<size_t>(kEmbedding)) {
        error = "Qwen4Exp HC graph returned invalid batch rows";
        return false;
    }
    if (inject_rows) {
        if (raw_injection.size() != rows * static_cast<size_t>(kHc)) {
            error = "Qwen4Exp HC graph returned invalid batch injections";
            return false;
        }
        inject_rows->resize(rows);
        for (size_t row = 0; row < rows; ++row) {
            std::copy_n(raw_injection.data() +
                            row * static_cast<size_t>(kHc), kHc,
                        (*inject_rows)[row].begin());
        }
    }
    return true;
}

bool hc_mix_batch(const Qwen4ExpWeights & target,
                  Qwen4ExpFrontierDenseCache * cache,
                  const Qwen4ExpLayer & layer, size_t rows,
                  const std::vector<float> & hc_rows,
                  std::vector<float> & mixed_rows, std::string & error) {
    return hc_mix_rows(target, cache, layer.hc_attn_norm,
                       layer.hc_attn_down, layer.hc_attn_up, nullptr, rows,
                       hc_rows, mixed_rows, nullptr, error);
}

bool rotate_optional_batch(Qwen4ExpFrontierDenseCache * cache,
                           ggml_backend_t backend, ggml_tensor * rotation,
                           std::vector<float> & values,
                           std::string & error) {
    if (!rotation) return true;
    if (ggml_n_dims(rotation) != 2 || rotation->ne[0] != rotation->ne[1] ||
        rotation->ne[0] <= 0 ||
        rotation->ne[0] > std::numeric_limits<int>::max() ||
        values.size() % static_cast<size_t>(rotation->ne[0]) != 0) {
        error = "invalid Qwen4Exp batched optional Hadamard rotation shape";
        return false;
    }
    const int width = static_cast<int>(rotation->ne[0]);
    const size_t row_count = values.size() / static_cast<size_t>(width);
    if (row_count == 0 ||
        row_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error = "invalid Qwen4Exp batched optional Hadamard rotation rows";
        return false;
    }
    std::vector<float> rotated;
    if (!matmul_rows(cache, backend, rotation, values.data(), width,
                     static_cast<int>(row_count), rotated, error)) return false;
    values = std::move(rotated);
    return true;
}

bool append_qsa_cache_batch(
        const Qwen4ExpWeights & target, Qwen4ExpMtpState & state,
        Qwen4ExpFrontierDenseCache * cache,
        const Qwen4ExpLayer & layer, const std::vector<float> & mixed_rows,
        const std::vector<std::array<int32_t, 3>> & positions,
        std::string & error) {
    const size_t rows = positions.size();
    if (rows == 0 || rows > 16 || mixed_rows.size() != rows * 2560U) {
        error = "invalid Qwen4Exp MTP QSA cache batch shape";
        return false;
    }
    std::vector<float> key, value, index_key;
    if (!matmul_rows(cache, target.backend, layer.attn_k, mixed_rows.data(),
                     kEmbedding, static_cast<int>(rows), key, error) ||
        !matmul_rows(cache, target.backend, layer.attn_v, mixed_rows.data(),
                     kEmbedding, static_cast<int>(rows), value, error) ||
        !matmul_rows(cache, target.backend, layer.index_k, mixed_rows.data(),
                     kEmbedding, static_cast<int>(rows), index_key, error) ||
        key.size() != rows * 512U || value.size() != rows * 512U ||
        index_key.size() != rows * 128U) return false;
    std::vector<float> key_norm;
    if (!tensor_f32(cache, layer.attn_k_norm, key_norm, error) ||
        key_norm.size() != 256U) return false;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t head = 0; head < 2U; ++head) {
            float * head_key = key.data() + (row * 2U + head) * 256U;
            rms_norm(head_key, kQsaDim, key_norm.data());
            if (!rope(target, head_key, kQsaDim, positions[row].data(),
                      error)) return false;
        }
    }
    if (!rotate_optional_batch(cache, target.backend, layer.self_k_rot, key,
                               error) ||
        !rotate_optional_batch(cache, target.backend, layer.self_v_rot, value,
                               error))
        return false;

    // Commit only after every stateless batch operation succeeded. The
    // append sequence matches qwen4exp_mtp_sync_cache_q1 exactly: K/V/raw
    // index-K first, then the corresponding M-RoPE row and cur_pos.
    for (size_t row = 0; row < rows; ++row) {
        state.qsa.key.append(key.data() + row * 512U, 512U);
        state.qsa.value.append(value.data() + row * 512U, 512U);
        state.qsa.index_key.append(index_key.data() + row * 128U, 128U);
        for (size_t axis = 0; axis < state.mrope_positions.size(); ++axis)
            state.mrope_positions[axis].push_back(positions[row][axis]);
        ++state.cur_pos;
    }
    return true;
}
} // namespace

bool qwen4exp_mtp_step_q1(
        const Qwen4ExpWeights & target, const Qwen4ExpMtpWeights & mtp,
        Qwen4ExpMtpState & state, int32_t token,
        const float * next_embedding, size_t next_embedding_count,
        const float * target_hc,
        size_t target_hc_count,
        const std::array<int32_t, 3> & mrope_position,
        std::vector<float> & logits, std::vector<float> & draft_hc,
        std::string & error) {
    if (state.cur_pos < 0 || state.cur_pos >= target.max_ctx ||
        !mtp.output_hc_norm || !mtp.output_hc_down || !mtp.output_hc_up) {
        error = "invalid Qwen4Exp MTP token/frontier contract";
        return false;
    }
    if (!prepare_mtp_hc(target, mtp, token, next_embedding,
                        next_embedding_count, target_hc, target_hc_count,
                        state.hc, error))
        return false;

    for (size_t axis = 0; axis < state.mrope_positions.size(); ++axis)
        state.mrope_positions[axis].push_back(mrope_position[axis]);
    std::vector<float> mixed, block;
    std::array<float, kHc> inject{};
    if (!hc_mix(target, state.hc, mtp.layer.hc_attn_norm,
                mtp.layer.hc_attn_down, mtp.layer.hc_attn_up,
                mtp.layer.hc_attn_inject, mixed, &inject, error,
                mtp.dense_cache) ||
        !run_qsa(target, state.qsa, mtp.layer, -1, mixed, mrope_position,
                 state.mrope_positions, block, error, mtp.dense_cache,
                 mtp.frontier_qsa))
        return false;
    hc_combine(state.hc, block, inject);
    if (!hc_mix(target, state.hc, mtp.layer.hc_ffn_norm,
                mtp.layer.hc_ffn_down, mtp.layer.hc_ffn_up,
                mtp.layer.hc_ffn_inject, mixed, &inject, error,
                mtp.dense_cache) ||
        !qwen4exp_frontier_mtp_moe_q1(mtp, mixed.data(), mixed.size(),
                                      block, error)) return false;
    hc_combine(state.hc, block, inject);

    draft_hc = state.hc;
    if (!hc_output_rows(
            target, mtp.dense_cache, mtp.output_hc_norm,
            mtp.output_hc_down, mtp.output_hc_up, target.output,
            state.hc.data(), state.hc.size(), 1, logits, error)) return false;
    ++state.cur_pos;
    return true;
}

bool qwen4exp_mtp_sync_cache_q1(
        const Qwen4ExpWeights & target, const Qwen4ExpMtpWeights & mtp,
        Qwen4ExpMtpState & state, int32_t token,
        const float * next_embedding, size_t next_embedding_count,
        const float * target_hc, size_t target_hc_count,
        const std::array<int32_t, 3> & mrope_position,
        std::string & error) {
    if (state.cur_pos < 0 || state.cur_pos >= target.max_ctx) {
        error = "invalid Qwen4Exp MTP cache synchronization frontier";
        return false;
    }
    std::vector<float> hc;
    if (!prepare_mtp_hc(target, mtp, token, next_embedding,
                        next_embedding_count, target_hc, target_hc_count, hc,
                        error)) return false;
    std::vector<float> mixed;
    if (!hc_mix(target, hc, mtp.layer.hc_attn_norm,
                mtp.layer.hc_attn_down, mtp.layer.hc_attn_up, nullptr, mixed,
                nullptr, error, mtp.dense_cache)) return false;
    if (!append_qsa_cache(target, state.qsa, mtp.layer, mixed,
                          mrope_position, error, mtp.dense_cache)) return false;
    for (size_t axis = 0; axis < state.mrope_positions.size(); ++axis)
        state.mrope_positions[axis].push_back(mrope_position[axis]);
    ++state.cur_pos;
    return true;
}

bool qwen4exp_mtp_sync_cache_batch(
        const Qwen4ExpWeights & target, const Qwen4ExpMtpWeights & mtp,
        Qwen4ExpMtpState & state, const std::vector<int32_t> & tokens,
        const std::vector<std::vector<float>> & target_hc_rows,
        const std::vector<std::array<int32_t, 3>> & mrope_positions,
        std::string & error) {
    Qwen4ExpMtpCacheBatchShape shape;
    if (!qwen4exp_mtp_cache_batch_shape(tokens.size(), shape, error) ||
        target_hc_rows.size() != shape.rows ||
        mrope_positions.size() != shape.rows || state.cur_pos < 0 ||
        state.cur_pos > target.max_ctx ||
        shape.rows > static_cast<size_t>(target.max_ctx - state.cur_pos)) {
        if (error.empty())
            error = "invalid Qwen4Exp MTP cache batch frontier";
        return false;
    }
    std::vector<float> hc_rows, mixed_rows;
    if (!prepare_mtp_hc_batch(target, mtp, tokens, target_hc_rows, hc_rows,
                              error) ||
        !hc_mix_batch(target, mtp.dense_cache, mtp.layer, shape.rows, hc_rows,
                      mixed_rows, error) ||
        !append_qsa_cache_batch(target, state, mtp.dense_cache, mtp.layer,
                                mixed_rows, mrope_positions, error))
        return false;
    return true;
}

namespace {
enum Qwen4ExpBatchQ1Mask : int {
    kBatchQ1Ple = 1,
    kBatchQ1AttentionHc = 2,
    kBatchQ1Attention = 4,
    kBatchQ1FfnHc = 8,
    kBatchQ1Moe = 16,
    kBatchQ1All = 31,
};

int batch_q1_numerics_mask() {
    static const int mask = []() {
        const char * force_all =
            std::getenv("DFLASH_QWEN_BATCH_FORCE_Q1_NUMERICS");
        if (force_all && std::strcmp(force_all, "1") == 0)
            return static_cast<int>(kBatchQ1All);
        const char * value = std::getenv("DFLASH_QWEN_BATCH_Q1_MASK");
        if (!value || !*value) return 0;
        const long parsed = std::strtol(value, nullptr, 10);
        return parsed >= 0 && parsed <= kBatchQ1All
            ? static_cast<int>(parsed) : 0;
    }();
    return mask;
}

bool qwen4exp_batch_layer_q1(
        const Qwen4ExpWeights & weights, Qwen4ExpState & state,
        const std::vector<int32_t> & tokens,
        const std::vector<std::array<int32_t, 3>> & positions,
        std::vector<std::vector<float>> & hc_rows, int layer_index,
        std::string & error) {
    const Qwen4ExpLayer & layer =
        weights.layers[static_cast<size_t>(layer_index)];
    for (size_t row = 0; row < tokens.size(); ++row) {
        state.hc = std::move(hc_rows[row]);
        if (layer_index == 1 &&
            !run_ple(weights, state, layer, tokens[row], error)) return false;
        std::vector<float> mixed, block;
        std::array<float, kHc> inject{};
        if (!hc_mix(weights, state.hc, layer.hc_attn_norm,
                    layer.hc_attn_down, layer.hc_attn_up,
                    layer.hc_attn_inject, mixed, &inject, error)) return false;
        const bool qsa = (layer_index + 1) % 4 == 0;
        if (qsa) {
            if (!run_qsa(
                    weights,
                    state.layers[static_cast<size_t>(layer_index)], layer,
                    layer_index, mixed, positions[row], state.mrope_positions,
                    block, error)) return false;
        } else if (!run_gdn(
                       weights,
                       state.layers[static_cast<size_t>(layer_index)], layer,
                       layer_index, mixed, block, error)) {
            return false;
        }
        hc_combine(state.hc, block, inject);
        if (!hc_mix(weights, state.hc, layer.hc_ffn_norm,
                    layer.hc_ffn_down, layer.hc_ffn_up,
                    layer.hc_ffn_inject, mixed, &inject, error) ||
            !run_moe(weights, layer_index, layer, mixed, block, error)) {
            return false;
        }
        hc_combine(state.hc, block, inject);
        hc_rows[row] = std::move(state.hc);
    }
    return true;
}

bool qwen4exp_batch_layer(
        const Qwen4ExpWeights & weights, Qwen4ExpState & state,
        const std::vector<int32_t> & tokens,
        const std::vector<std::array<int32_t, 3>> & positions,
        std::vector<std::vector<float>> & hc_rows, int layer_index,
        std::string & error) {
    const size_t rows = tokens.size();
    const Qwen4ExpLayer & layer =
        weights.layers[static_cast<size_t>(layer_index)];
    const int q1_mask = batch_q1_numerics_mask();
    // Diagnostic-only: retain the layer-major schedule and causal state order
    // while forcing every normally batched subsystem through its q=1 graph.
    // This separates a scheduling/composition defect from MMQ-vs-MMVQ
    // arithmetic without changing the production path.
    if (q1_mask == kBatchQ1All) {
        return qwen4exp_batch_layer_q1(
            weights, state, tokens, positions, hc_rows, layer_index, error);
    }
    if (layer_index == 1) {
        if (q1_mask & kBatchQ1Ple) {
            for (size_t row = 0; row < rows; ++row) {
                state.hc = std::move(hc_rows[row]);
                if (!run_ple(weights, state, layer, tokens[row], error))
                    return false;
                hc_rows[row] = std::move(state.hc);
            }
        } else if (!run_ple_batch(
                       weights, state, layer, tokens, hc_rows, error)) {
            return false;
        }
    }
    std::vector<float> attention_hc(rows * static_cast<size_t>(kHcDim));
    for (size_t row = 0; row < rows; ++row) {
        state.hc = std::move(hc_rows[row]);
        std::copy(state.hc.begin(), state.hc.end(),
                  attention_hc.begin() +
                      static_cast<std::ptrdiff_t>(row * kHcDim));
        hc_rows[row] = std::move(state.hc);
    }

    std::vector<float> attention_inputs;
    std::vector<std::array<float, kHc>> attention_inject;
    if (q1_mask & kBatchQ1AttentionHc) {
        attention_inputs.resize(rows * static_cast<size_t>(kEmbedding));
        attention_inject.resize(rows);
        for (size_t row = 0; row < rows; ++row) {
            std::vector<float> mixed;
            if (!hc_mix(weights, hc_rows[row], layer.hc_attn_norm,
                        layer.hc_attn_down, layer.hc_attn_up,
                        layer.hc_attn_inject, mixed,
                        &attention_inject[row], error)) return false;
            std::copy(mixed.begin(), mixed.end(),
                      attention_inputs.begin() + static_cast<std::ptrdiff_t>(
                          row * static_cast<size_t>(kEmbedding)));
        }
    } else if (!hc_mix_rows(
                   weights, weights.dense_cache, layer.hc_attn_norm,
                   layer.hc_attn_down, layer.hc_attn_up,
                   layer.hc_attn_inject, rows, attention_hc,
                   attention_inputs, &attention_inject, error)) {
        return false;
    }

    const bool qsa = (layer_index + 1) % 4 == 0;
    std::vector<float> attention_outputs;
    if ((q1_mask & kBatchQ1Attention) && qsa) {
        attention_outputs.resize(rows * static_cast<size_t>(kEmbedding));
        for (size_t row = 0; row < rows; ++row) {
            const auto begin = attention_inputs.begin() +
                static_cast<std::ptrdiff_t>(
                    row * static_cast<size_t>(kEmbedding));
            const std::vector<float> row_input(
                begin, begin + static_cast<std::ptrdiff_t>(kEmbedding));
            std::vector<float> block;
            if (!run_qsa(
                    weights,
                    state.layers[static_cast<size_t>(layer_index)], layer,
                    layer_index, row_input, positions[row],
                    state.mrope_positions, block, error)) return false;
            std::copy(block.begin(), block.end(),
                      attention_outputs.begin() +
                          static_cast<std::ptrdiff_t>(
                              row * static_cast<size_t>(kEmbedding)));
        }
    } else if (qsa) {
        // QSA selection depends on every newly appended raw index-K row. Its
        // state update remains strictly row ordered. The independent Q/K/V
        // and index projections cross one q5/q16 boundary before that loop;
        // selection, attention, and cache publication do not.
        if (!run_qsa_batch(
                weights, state.layers[static_cast<size_t>(layer_index)],
                layer, layer_index, attention_inputs, positions,
                state.mrope_positions, attention_outputs, error)) return false;
    } else if (q1_mask & kBatchQ1Attention) {
        // The diagnostic attention bit covers both attention families.  QSA
        // already takes its q=1 path above; mirror that behavior for GDN so a
        // mask-4 run does not silently leave three quarters of the layers on
        // the batched recurrent kernel.
        attention_outputs.resize(rows * static_cast<size_t>(kEmbedding));
        for (size_t row = 0; row < rows; ++row) {
            const auto begin = attention_inputs.begin() +
                static_cast<std::ptrdiff_t>(
                    row * static_cast<size_t>(kEmbedding));
            const std::vector<float> row_input(
                begin, begin + static_cast<std::ptrdiff_t>(kEmbedding));
            std::vector<float> block;
            if (!run_gdn(
                    weights,
                    state.layers[static_cast<size_t>(layer_index)], layer,
                    layer_index, row_input, block, error)) return false;
            std::copy(block.begin(), block.end(),
                      attention_outputs.begin() +
                          static_cast<std::ptrdiff_t>(
                              row * static_cast<size_t>(kEmbedding)));
        }
    } else {
        // GDN is causal internally and executes the exact q2-q16 recurrent
        // sequence in one persistent graph/state boundary.
        if (!run_gdn_batch(
                weights, state.layers[static_cast<size_t>(layer_index)],
                layer, layer_index, attention_inputs,
                static_cast<int>(rows), attention_outputs, error))
            return false;
    }
    if (attention_outputs.size() !=
        rows * static_cast<size_t>(kEmbedding)) {
        error = qsa ? "Qwen4Exp batched QSA returned the wrong output shape"
                    : "Qwen4Exp batched GDN returned the wrong output shape";
        return false;
    }
    for (size_t row = 0; row < rows; ++row) {
        state.hc = std::move(hc_rows[row]);
        const auto block_begin = attention_outputs.begin() +
            static_cast<std::ptrdiff_t>(row * kEmbedding);
        const std::vector<float> block(
            block_begin,
            block_begin + static_cast<std::ptrdiff_t>(kEmbedding));
        hc_combine(state.hc, block, attention_inject[row]);
        hc_rows[row] = std::move(state.hc);
    }

    std::vector<float> ffn_hc(rows * static_cast<size_t>(kHcDim));
    for (size_t row = 0; row < rows; ++row) {
        std::copy(hc_rows[row].begin(), hc_rows[row].end(),
                  ffn_hc.begin() +
                      static_cast<std::ptrdiff_t>(row * kHcDim));
    }
    std::vector<float> ffn_inputs;
    std::vector<std::array<float, kHc>> ffn_inject;
    if (q1_mask & kBatchQ1FfnHc) {
        ffn_inputs.resize(rows * static_cast<size_t>(kEmbedding));
        ffn_inject.resize(rows);
        for (size_t row = 0; row < rows; ++row) {
            std::vector<float> mixed;
            if (!hc_mix(weights, hc_rows[row], layer.hc_ffn_norm,
                        layer.hc_ffn_down, layer.hc_ffn_up,
                        layer.hc_ffn_inject, mixed, &ffn_inject[row],
                        error)) return false;
            std::copy(mixed.begin(), mixed.end(),
                      ffn_inputs.begin() + static_cast<std::ptrdiff_t>(
                          row * static_cast<size_t>(kEmbedding)));
        }
    } else if (!hc_mix_rows(
                   weights, weights.dense_cache, layer.hc_ffn_norm,
                   layer.hc_ffn_down, layer.hc_ffn_up,
                   layer.hc_ffn_inject, rows, ffn_hc, ffn_inputs,
                   &ffn_inject, error)) {
        return false;
    }

    std::vector<float> ffn_outputs;
    if (!(q1_mask & kBatchQ1Moe) &&
        qwen4exp_frontier_moe_available(weights, layer_index)) {
        if (!qwen4exp_frontier_moe_batch(
                weights, layer_index, ffn_inputs.data(), ffn_inputs.size(),
                static_cast<int>(rows), ffn_outputs, error)) return false;
    } else {
        ffn_outputs.resize(rows * static_cast<size_t>(kEmbedding));
        for (size_t row = 0; row < rows; ++row) {
            std::vector<float> block;
            const auto begin = ffn_inputs.begin() +
                static_cast<std::ptrdiff_t>(row * kEmbedding);
            const std::vector<float> row_input(
                begin, begin + static_cast<std::ptrdiff_t>(kEmbedding));
            if (!run_moe(weights, layer_index, layer,
                         row_input,
                         block, error)) return false;
            std::copy(block.begin(), block.end(),
                      ffn_outputs.begin() +
                          static_cast<std::ptrdiff_t>(row * kEmbedding));
        }
    }
    if (ffn_outputs.size() != rows * static_cast<size_t>(kEmbedding)) {
        error = "Qwen4Exp batched MoE returned the wrong output shape";
        return false;
    }
    for (size_t row = 0; row < rows; ++row) {
        state.hc = std::move(hc_rows[row]);
        const auto begin = ffn_outputs.begin() +
            static_cast<std::ptrdiff_t>(row * kEmbedding);
        const std::vector<float> block(
            begin, begin + static_cast<std::ptrdiff_t>(kEmbedding));
        hc_combine(state.hc, block, ffn_inject[row]);
        hc_rows[row] = std::move(state.hc);
    }
    return true;
}
} // namespace

namespace {
bool qwen4exp_step_batch_mrope_impl(
        const Qwen4ExpWeights & weights, Qwen4ExpState & state,
        const std::vector<int32_t> & tokens,
        const std::vector<std::array<int32_t, 3>> & mrope_positions,
        std::vector<std::vector<float>> * row_logits,
        std::vector<std::vector<float>> * row_hc,
        std::vector<float> * final_logits, std::string & error) {
    if (row_logits) row_logits->clear();
    if (row_hc) row_hc->clear();
    if (final_logits) final_logits->clear();
    const size_t rows = tokens.size();
    if (rows < 2 || rows >
            static_cast<size_t>(kQwen4ExpFrontierMoeMaxBatch) ||
        mrope_positions.size() != rows ||
        state.cur_pos < 0 || state.cur_pos > weights.max_ctx ||
        weights.layers.size() != 48 ||
        rows > static_cast<size_t>(weights.max_ctx - state.cur_pos)) {
        error = "invalid Qwen4Exp bounded q>1 verifier frontier";
        return false;
    }
    for (int32_t token : tokens) {
        if (token < 0 || token >= 248320) {
            error = "invalid Qwen4Exp bounded verifier token";
            return false;
        }
    }

    std::vector<float> embeddings(rows * static_cast<size_t>(kEmbedding));
    if (!weights.embedder.embed(tokens.data(), static_cast<int>(rows),
                                embeddings.data())) {
        error = "Qwen4Exp bounded verifier embedding lookup failed";
        return false;
    }
    std::vector<std::vector<float>> hc_rows(
        rows, std::vector<float>(static_cast<size_t>(kHcDim)));
    for (size_t row = 0; row < rows; ++row) {
        const float * embedding = embeddings.data() + row * kEmbedding;
        for (int stream = 0; stream < kHc; ++stream) {
            std::copy_n(embedding, kEmbedding,
                        hc_rows[row].data() + stream * kEmbedding);
        }
        for (size_t axis = 0; axis < state.mrope_positions.size(); ++axis)
            state.mrope_positions[axis].push_back(mrope_positions[row][axis]);
    }

    // Layer-major execution is causally equivalent to q=1 token-major
    // execution: a later row at layer L depends only on earlier rows at L and
    // its own output from L-1. The inner row order is never parallelized across
    // PLE/GDN/QSA state mutations.
    for (int layer_index = 0; layer_index < 48; ++layer_index) {
        if (!qwen4exp_batch_layer(weights, state, tokens, mrope_positions,
                                  hc_rows, layer_index, error)) return false;
    }

    if (row_hc) *row_hc = hc_rows;
    if (row_logits) {
        // The bounded verifier consumes every row, but the final HC
        // The down/up mixer and vocabulary projection are stateless across
        // rows. Keep them in one q5/q16 frontier instead of returning to a
        // host boundary between the final HC row and vocabulary head. This is
        // local vendored-engine divergence; the authoritative replay below the MTP
        // verifier still commits accepted tokens through q=1.
        std::vector<float> final_hc(rows * static_cast<size_t>(kHcDim));
        for (size_t row = 0; row < rows; ++row) {
            std::copy(hc_rows[row].begin(), hc_rows[row].end(),
                      final_hc.begin() + static_cast<std::ptrdiff_t>(
                          row * static_cast<size_t>(kHcDim)));
        }
        if (!weights.output || weights.output->ne[1] <= 0 ||
            static_cast<uint64_t>(weights.output->ne[1]) >
                static_cast<uint64_t>(
                    std::numeric_limits<size_t>::max() / rows)) {
            error = "invalid Qwen4Exp batched output projection shape";
            return false;
        }
        const size_t vocabulary = static_cast<size_t>(weights.output->ne[1]);
        std::vector<float> logits;
        if (!hc_output_rows(
                weights, weights.dense_cache, weights.output_hc_norm,
                weights.output_hc_down, weights.output_hc_up, weights.output,
                final_hc.data(), final_hc.size(), static_cast<int>(rows),
                logits, error) ||
            logits.size() != rows * vocabulary) {
            if (error.empty())
                error = "Qwen4Exp batched output projection shape mismatch";
            return false;
        }
        row_logits->resize(rows);
        for (size_t row = 0; row < rows; ++row) {
            const auto begin = logits.begin() +
                static_cast<std::ptrdiff_t>(row * vocabulary);
            (*row_logits)[row].assign(
                begin, begin + static_cast<std::ptrdiff_t>(vocabulary));
        }
    } else if (final_logits) {
        std::vector<float> logits;
        if (!hc_output_rows(
                weights, weights.dense_cache, weights.output_hc_norm,
                weights.output_hc_down, weights.output_hc_up, weights.output,
                hc_rows.back().data(), hc_rows.back().size(), 1, logits,
                error)) return false;
        *final_logits = std::move(logits);
    }
    state.hc = hc_rows.back();
    state.cur_pos += static_cast<int>(rows);
    state.last_token = tokens.back();
    return true;
}
} // namespace

bool qwen4exp_step_batch_mrope(
        const Qwen4ExpWeights & weights, Qwen4ExpState & state,
        const std::vector<int32_t> & tokens,
        const std::vector<std::array<int32_t, 3>> & mrope_positions,
        std::vector<std::vector<float>> & row_logits,
        std::vector<std::vector<float>> & row_hc, std::string & error) {
    return qwen4exp_step_batch_mrope_impl(
        weights, state, tokens, mrope_positions, &row_logits, &row_hc,
        nullptr, error);
}

bool qwen4exp_step_prefill_batch_mrope(
        const Qwen4ExpWeights & weights, Qwen4ExpState & state,
        const std::vector<int32_t> & tokens,
        const std::vector<std::array<int32_t, 3>> & mrope_positions,
        std::vector<float> & logits, std::vector<std::vector<float>> & row_hc,
        std::string & error) {
    return qwen4exp_step_batch_mrope_impl(
        weights, state, tokens, mrope_positions, nullptr, &row_hc, &logits,
        error);
}

size_t qwen4exp_prefill_chunk_rows(size_t batchable_rows, int current_pos,
                                   int snapshot_pos, bool force_q1) {
    if (batchable_rows == 0) return 0;
    if (force_q1) return 1;
    size_t rows = std::min(
        batchable_rows,
        static_cast<size_t>(kQwen4ExpFrontierMoeMaxBatch));
    if (snapshot_pos > current_pos) {
        const int distance = snapshot_pos - current_pos;
        rows = std::min(rows, static_cast<size_t>(distance));
    }
    return rows < 2 ? 1 : rows;
}

} // namespace dflash::common
