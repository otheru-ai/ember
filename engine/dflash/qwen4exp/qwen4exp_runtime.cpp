#include "qwen4exp_internal.h"
#include "qwen4exp_frontier.h"
#include "qwen4exp_mtp.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
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

bool tensor_f32(ggml_tensor * tensor, std::vector<float> & out,
                std::string & error) {
    if (!tensor || !tensor->buffer) {
        error = "Qwen4Exp runtime received an unbound tensor";
        return false;
    }
    const int64_t elements = ggml_nelements(tensor);
    if (elements < 0 || static_cast<uint64_t>(elements) >
                            std::numeric_limits<size_t>::max()) {
        error = "Qwen4Exp tensor element count overflow";
        return false;
    }
    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (!traits || !traits->to_float) {
        error = "Qwen4Exp runtime cannot decode tensor type";
        return false;
    }
    out.resize(static_cast<size_t>(elements));
    const int64_t row = tensor->ne[0];
    const size_t row_bytes = ggml_row_size(tensor->type, row);
    const int64_t rows = elements / row;
    for (int64_t i = 0; i < rows; ++i) {
        traits->to_float(raw.data() + static_cast<size_t>(i) * row_bytes,
                         out.data() + static_cast<size_t>(i * row), row);
    }
    return true;
}

bool matvec(ggml_backend_t backend, ggml_tensor * weight,
            const float * input, int input_count,
            std::vector<float> & output, std::string & error) {
    if (!backend || !weight || !input || weight->ne[0] != input_count ||
        ggml_n_dims(weight) != 2) {
        error = "invalid Qwen4Exp matvec shape";
        return false;
    }
    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { error = "Qwen4Exp matvec context allocation failed"; return false; }
    ggml_tensor * in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_count);
    ggml_set_input(in);
    ggml_tensor * out = ggml_mul_mat(ctx, weight, in);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(ctx); error = "Qwen4Exp matvec graph allocation failed"; return false;
    }
    ggml_backend_tensor_set(in, input, 0,
                            sizeof(float) * static_cast<size_t>(input_count));
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (ok) {
        output.resize(static_cast<size_t>(weight->ne[1]));
        ggml_backend_tensor_get(out, output.data(), 0,
                                output.size() * sizeof(float));
    } else {
        error = "Qwen4Exp matvec graph execution failed";
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return ok;
}

bool rotate_optional(ggml_backend_t backend, ggml_tensor * rotation,
                     std::vector<float> & values, std::string & error) {
    if (!rotation) return true;
    if (ggml_n_dims(rotation) != 2 || rotation->ne[0] != rotation->ne[1] ||
        rotation->ne[0] <= 0 || values.size() % static_cast<size_t>(rotation->ne[0])) {
        error = "invalid Qwen4Exp optional Hadamard rotation shape";
        return false;
    }
    const int width = static_cast<int>(rotation->ne[0]);
    std::vector<float> rotated;
    for (size_t offset = 0; offset < values.size(); offset += static_cast<size_t>(width)) {
        if (!matvec(backend, rotation, values.data() + offset, width,
                    rotated, error)) return false;
        std::copy(rotated.begin(), rotated.end(), values.begin() +
                  static_cast<std::ptrdiff_t>(offset));
    }
    return true;
}

bool hc_mix(const Qwen4ExpWeights & weights, const std::vector<float> & hc,
            ggml_tensor * norm, ggml_tensor * down, ggml_tensor * up,
            ggml_tensor * inject_weight, std::vector<float> & mixed,
            std::array<float, kHc> * inject, std::string & error) {
    if (hc.size() != kHcDim) { error = "invalid Qwen4Exp HC state"; return false; }
    std::vector<float> norm_weight;
    if (!tensor_f32(norm, norm_weight, error) || norm_weight.size() != kHcDim)
        return false;
    std::vector<float> xn = hc;
    for (int stream = 0; stream < kHc; ++stream)
        rms_norm(xn.data() + stream * kEmbedding, kEmbedding, nullptr);
    for (int i = 0; i < kHcDim; ++i) xn[i] *= norm_weight[i];

    std::vector<float> low;
    if (!matvec(weights.backend, down, xn.data(), kHcDim, low, error)) return false;
    for (float & value : low) value = silu(value / static_cast<float>(kHc));
    std::vector<float> gate;
    if (!matvec(weights.backend, up, low.data(), static_cast<int>(low.size()),
                gate, error) || gate.size() != kHcDim) return false;
    for (float & value : gate) value = sigmoid(value);
    mixed.assign(kEmbedding, 0.0f);
    for (int stream = 0; stream < kHc; ++stream)
        for (int i = 0; i < kEmbedding; ++i)
            mixed[i] += xn[stream * kEmbedding + i] *
                        gate[stream * kEmbedding + i] / static_cast<float>(kHc);
    if (inject) {
        std::vector<float> raw;
        if (!matvec(weights.backend, inject_weight, xn.data(), kHcDim,
                    raw, error) || raw.size() != kHc) return false;
        std::copy(raw.begin(), raw.end(), inject->begin());
    }
    return true;
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
    if (!matvec(weights.backend, layer.ple_key, embedded.data(), kEmbedding,
                key, error) ||
        !matvec(weights.backend, layer.ple_value, embedded.data(), kEmbedding,
                value, error)) return false;
    std::vector<float> key_norm, query_norm, conv_norm, conv_weight;
    if (!tensor_f32(layer.ple_norm_key, key_norm, error) ||
        !tensor_f32(layer.ple_norm_query, query_norm, error) ||
        !tensor_f32(layer.ple_norm_conv, conv_norm, error) ||
        !tensor_f32(layer.ple_conv, conv_weight, error)) return false;
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

bool run_gdn(const Qwen4ExpWeights & weights, Qwen4ExpLayerState & state,
             const Qwen4ExpLayer & layer, const std::vector<float> & input,
             std::vector<float> & output, std::string & error) {
    std::vector<float> qkv, z, alpha, beta;
    if (!matvec(weights.backend, layer.attn_qkv, input.data(), kEmbedding, qkv, error) ||
        !matvec(weights.backend, layer.attn_gate, input.data(), kEmbedding, z, error) ||
        !matvec(weights.backend, layer.ssm_alpha, input.data(), kEmbedding, alpha, error) ||
        !matvec(weights.backend, layer.ssm_beta, input.data(), kEmbedding, beta, error)) return false;
    std::vector<float> conv_weight, a, dt, norm;
    if (!tensor_f32(layer.ssm_conv, conv_weight, error) ||
        !tensor_f32(layer.ssm_a, a, error) ||
        !tensor_f32(layer.ssm_dt, dt, error) ||
        !tensor_f32(layer.ssm_norm, norm, error)) return false;
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
    return matvec(weights.backend, layer.ssm_out, core.data(), 6144, output, error);
}

bool run_qsa(const Qwen4ExpWeights & weights, Qwen4ExpLayerState & state,
             const Qwen4ExpLayer & layer, const std::vector<float> & input,
             const std::array<int32_t, 3> & position,
             const std::array<std::vector<int32_t>, 3> & position_history,
             std::vector<float> & output, std::string & error) {
    std::vector<float> qfull, k, v, iq, ik;
    if (!matvec(weights.backend, layer.attn_q, input.data(), kEmbedding, qfull, error) ||
        !matvec(weights.backend, layer.attn_k, input.data(), kEmbedding, k, error) ||
        !matvec(weights.backend, layer.attn_v, input.data(), kEmbedding, v, error) ||
        !matvec(weights.backend, layer.index_q, input.data(), kEmbedding, iq, error) ||
        !matvec(weights.backend, layer.index_k, input.data(), kEmbedding, ik, error)) return false;
    std::vector<float> qnorm, knorm, iqnorm, iknorm;
    if (!tensor_f32(layer.attn_q_norm, qnorm, error) ||
        !tensor_f32(layer.attn_k_norm, knorm, error) ||
        !tensor_f32(layer.index_q_norm, iqnorm, error) ||
        !tensor_f32(layer.index_k_norm, iknorm, error)) return false;
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
    for (int head = 0; head < kQsaKvHeads; ++head) {
        rms_norm(k.data() + head * kQsaDim, kQsaDim, knorm.data());
        if (!rope(weights, k.data() + head * kQsaDim, kQsaDim,
                  position.data(), error)) return false;
    }
    for (int head = 0; head < kIndexerHeads; ++head) {
        rms_norm(iq.data() + head * kIndexerDim, kIndexerDim, iqnorm.data());
        if (!rope(weights, iq.data() + head * kIndexerDim, kIndexerDim,
                  position.data(), error)) return false;
    }
    // PR #27774 (abdc7a0b over #27742 035e2273): K-cache Hadamard
    // applies to Q/K and V-cache Hadamard to V before sparse attention.
    if (!rotate_optional(weights.backend, layer.self_k_rot, q, error) ||
        !rotate_optional(weights.backend, layer.self_k_rot, k, error) ||
        !rotate_optional(weights.backend, layer.self_v_rot, v, error)) return false;
    state.key.append(k.data(), k.size());
    state.value.append(v.data(), v.size());
    state.index_key.append(ik.data(), ik.size());
    const int tokens = static_cast<int>(state.index_key.size() / kIndexerDim);
    // Score complete four-token blocks from the raw index-K cache. Pooling is
    // deliberately before learned RMSNorm and RoPE, matching #27742/HF. The
    // selection is host-side because the released 2048-token budget means
    // top-512 blocks and must not pass through a 1023-element-capped primitive.
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
        scored.emplace_back(score / std::sqrt(static_cast<float>(kIndexerDim)),
                            block_index);
    }
    if (keep < complete)
        std::partial_sort(scored.begin(), scored.begin() + keep, scored.end(),
                          [](const auto & a, const auto & b) {
                              return a.first != b.first ? a.first > b.first
                                                        : a.second < b.second;
                          });
    std::vector<int32_t> selected;
    selected.reserve(static_cast<size_t>(keep * 4 + tokens % 4));
    for (int i = 0; i < keep; ++i)
        for (int member = 0; member < 4; ++member)
            selected.push_back(scored[i].second * 4 + member);
    for (int tail = complete * 4; tail < tokens; ++tail) selected.push_back(tail);
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
    if (!rotate_optional(weights.backend, layer.self_v_rot, attended, error)) return false;
    for (size_t i = 0; i < attended.size(); ++i)
        attended[i] *= sigmoid(gate[i]);
    return matvec(weights.backend, layer.attn_output, attended.data(),
                  static_cast<int>(attended.size()), output, error);
}

bool run_moe(const Qwen4ExpWeights & weights, int layer_index,
             const Qwen4ExpLayer & layer,
             const std::vector<float> & input, std::vector<float> & output,
             std::string & error) {
    if (weights.frontier && layer_index >= 0 && layer_index < 48) {
        return qwen4exp_frontier_moe_q1(weights, layer_index, input.data(),
                                        input.size(), output, error);
    }
    std::vector<float> router;
    if (!matvec(weights.backend, layer.router, input.data(), kEmbedding,
                router, error) || router.size() != kExpertCount) return false;
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
        std::vector<float> gate_up, down;
        const Qwen4ExpMappedTensor gu = expert_slice(layer.experts_gate_up, ids[i]);
        const Qwen4ExpMappedTensor dw = expert_slice(layer.experts_down, ids[i]);
        if (!mapped_matvec(gu, input.data(), gate_up, error) ||
            gate_up.size() != 2 * kExpertFf) return false;
        std::vector<float> intermediate(kExpertFf);
        for (int j = 0; j < kExpertFf; ++j)
            intermediate[j] = silu(gate_up[j]) * gate_up[kExpertFf + j];
        if (!mapped_matvec(dw, intermediate.data(), down, error) ||
            down.size() != kEmbedding) return false;
        for (int j = 0; j < kEmbedding; ++j)
            output[j] += selected_weight[i] * down[j];
    }
    std::vector<float> shared_gate, shared_up, shared_down, shared_scale;
    if (!matvec(weights.backend, layer.shared_gate, input.data(), kEmbedding,
                shared_gate, error) ||
        !matvec(weights.backend, layer.shared_up, input.data(), kEmbedding,
                shared_up, error) || shared_gate.size() != kExpertFf ||
        shared_up.size() != kExpertFf) return false;
    for (int i = 0; i < kExpertFf; ++i) shared_gate[i] = silu(shared_gate[i]) * shared_up[i];
    if (!matvec(weights.backend, layer.shared_down, shared_gate.data(),
                kExpertFf, shared_down, error) ||
        !matvec(weights.backend, layer.shared_gate_input, input.data(),
                kEmbedding, shared_scale, error) || shared_scale.size() != 1)
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
                              std::vector<float> * attn_mixed_capture,
                              std::string & error) {
    if (token < 0 || token >= 248320 || state.cur_pos < 0 ||
        state.cur_pos >= weights.max_ctx || weights.layers.size() != 48) {
        error = "invalid Qwen4Exp q=1 token/frontier"; return false;
    }
    if (attn_mixed_capture) {
        attn_mixed_capture->clear();
        attn_mixed_capture->reserve(48 * kEmbedding);
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
        if (attn_mixed_capture)
            attn_mixed_capture->insert(attn_mixed_capture->end(),
                                       mixed.begin(), mixed.end());
        const bool qsa = (layer_index + 1) % 4 == 0;
        if (qsa) {
            if (!run_qsa(weights, state.layers[static_cast<size_t>(layer_index)],
                          layer, mixed, position, state.mrope_positions,
                          block, error)) return false;
        } else if (!run_gdn(weights, state.layers[static_cast<size_t>(layer_index)],
                            layer, mixed, block, error)) return false;
        hc_combine(state.hc, block, inject);
        if (!hc_mix(weights, state.hc, layer.hc_ffn_norm, layer.hc_ffn_down,
                    layer.hc_ffn_up, layer.hc_ffn_inject, mixed, &inject,
                    error) || !run_moe(weights, layer_index, layer, mixed,
                                       block, error)) return false;
        hc_combine(state.hc, block, inject);
    }
    std::vector<float> final;
    if (!hc_mix(weights, state.hc, weights.output_hc_norm,
                weights.output_hc_down, weights.output_hc_up, nullptr,
                final, nullptr, error) ||
        !matvec(weights.backend, weights.output, final.data(), kEmbedding,
                logits, error)) return false;
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
        std::vector<float> & logits, std::vector<float> & attn_mixed_capture,
        std::string & error) {
    return step_q1_embedding(weights, state, token, nullptr, mrope_position,
                             logits, &attn_mixed_capture, error);
}

bool qwen4exp_mtp_step_q1(
        const Qwen4ExpWeights & target, const Qwen4ExpMtpWeights & mtp,
        Qwen4ExpMtpState & state, int32_t token,
        const float * next_embedding, size_t next_embedding_count,
        const float * target_hc,
        size_t target_hc_count,
        const std::array<int32_t, 3> & mrope_position,
        std::vector<float> & logits, std::vector<float> & draft_hc,
        std::string & error) {
    if (token < 0 || token >= 248320 || !target_hc ||
        target_hc_count != static_cast<size_t>(kHcDim) ||
        state.cur_pos < 0 || state.cur_pos >= target.max_ctx ||
        !mtp.pre_embedding_norm || !mtp.pre_hc_norm || !mtp.fc_embedding ||
        !mtp.fc_hc || !mtp.output_hc_norm || !mtp.output_hc_down ||
        !mtp.output_hc_up) {
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
    if (!tensor_f32(mtp.pre_embedding_norm, embedding_norm, error) ||
        embedding_norm.size() != static_cast<size_t>(kEmbedding) ||
        !tensor_f32(mtp.pre_hc_norm, hc_norm, error) ||
        hc_norm.size() != static_cast<size_t>(kHcDim)) return false;

    rms_norm(embedding.data(), kEmbedding, embedding_norm.data());
    std::vector<float> projected_embedding;
    if (!matvec(target.backend, mtp.fc_embedding, embedding.data(), kEmbedding,
                projected_embedding, error) ||
        projected_embedding.size() != static_cast<size_t>(kEmbedding))
        return false;

    // Qwen4Exp MTP normalizes h_p globally across all four HC streams before
    // applying one shared 2560x2560 projection to each stream. This differs
    // from the stream-local normalization used by ordinary HC mixers.
    std::vector<float> normalized_hc(target_hc, target_hc + kHcDim);
    rms_norm(normalized_hc.data(), kHcDim, hc_norm.data());
    state.hc.resize(kHcDim);
    for (int stream = 0; stream < kHc; ++stream) {
        std::vector<float> projected_hidden;
        if (!matvec(target.backend, mtp.fc_hc,
                    normalized_hc.data() + stream * kEmbedding, kEmbedding,
                    projected_hidden, error) ||
            projected_hidden.size() != static_cast<size_t>(kEmbedding))
            return false;
        for (int channel = 0; channel < kEmbedding; ++channel) {
            state.hc[static_cast<size_t>(stream * kEmbedding + channel)] =
                projected_hidden[static_cast<size_t>(channel)] +
                projected_embedding[static_cast<size_t>(channel)];
        }
    }

    for (size_t axis = 0; axis < state.mrope_positions.size(); ++axis)
        state.mrope_positions[axis].push_back(mrope_position[axis]);
    std::vector<float> mixed, block;
    std::array<float, kHc> inject{};
    if (!hc_mix(target, state.hc, mtp.layer.hc_attn_norm,
                mtp.layer.hc_attn_down, mtp.layer.hc_attn_up,
                mtp.layer.hc_attn_inject, mixed, &inject, error) ||
        !run_qsa(target, state.qsa, mtp.layer, mixed, mrope_position,
                 state.mrope_positions, block, error)) return false;
    hc_combine(state.hc, block, inject);
    if (!hc_mix(target, state.hc, mtp.layer.hc_ffn_norm,
                mtp.layer.hc_ffn_down, mtp.layer.hc_ffn_up,
                mtp.layer.hc_ffn_inject, mixed, &inject, error) ||
        !run_moe(target, 48, mtp.layer, mixed, block, error)) return false;
    hc_combine(state.hc, block, inject);

    draft_hc = state.hc;
    std::vector<float> final;
    if (!hc_mix(target, state.hc, mtp.output_hc_norm, mtp.output_hc_down,
                mtp.output_hc_up, nullptr, final, nullptr, error) ||
        !matvec(target.backend, target.output, final.data(), kEmbedding,
                logits, error)) return false;
    ++state.cur_pos;
    return true;
}

namespace {
struct Qwen4ExpBatchLayerContext {
    const Qwen4ExpWeights & weights;
    Qwen4ExpState & state;
    const std::vector<int32_t> & tokens;
    const std::vector<std::array<int32_t, 3>> & positions;
    std::vector<std::vector<float>> & hc_rows;
};

bool qwen4exp_batch_layer_step(void * opaque, size_t layer_number,
                               size_t row, std::string & error) {
    auto & context = *static_cast<Qwen4ExpBatchLayerContext *>(opaque);
    const int layer_index = static_cast<int>(layer_number);
    const Qwen4ExpLayer & layer = context.weights.layers[layer_number];
    context.state.hc = std::move(context.hc_rows[row]);
    if (layer_index == 1 && !run_ple(context.weights, context.state, layer,
                                     context.tokens[row], error)) return false;
    std::vector<float> mixed, block;
    std::array<float, kHc> inject{};
    if (!hc_mix(context.weights, context.state.hc, layer.hc_attn_norm,
                layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
                mixed, &inject, error)) return false;
    const bool qsa = (layer_index + 1) % 4 == 0;
    if (qsa) {
        if (!run_qsa(context.weights,
                     context.state.layers[layer_number], layer, mixed,
                     context.positions[row], context.state.mrope_positions,
                     block, error)) return false;
    } else if (!run_gdn(context.weights, context.state.layers[layer_number],
                        layer, mixed, block, error)) {
        return false;
    }
    hc_combine(context.state.hc, block, inject);
    if (!hc_mix(context.weights, context.state.hc, layer.hc_ffn_norm,
                layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
                mixed, &inject, error) ||
        !run_moe(context.weights, layer_index, layer, mixed, block, error))
        return false;
    hc_combine(context.state.hc, block, inject);
    context.hc_rows[row] = std::move(context.state.hc);
    return true;
}
} // namespace

bool qwen4exp_step_batch_mrope(
        const Qwen4ExpWeights & weights, Qwen4ExpState & state,
        const std::vector<int32_t> & tokens,
        const std::vector<std::array<int32_t, 3>> & mrope_positions,
        std::vector<std::vector<float>> & row_logits,
        std::vector<std::vector<float>> & row_hc, std::string & error) {
    row_logits.clear();
    row_hc.clear();
    const size_t rows = tokens.size();
    if (rows < 2 || rows > 5 || mrope_positions.size() != rows ||
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
    Qwen4ExpBatchLayerContext context{
        weights, state, tokens, mrope_positions, hc_rows};
    if (!qwen4exp_run_layer_major(rows, 48, qwen4exp_batch_layer_step,
                                  &context, error)) return false;

    row_logits.reserve(rows);
    row_hc.reserve(rows);
    for (size_t row = 0; row < rows; ++row) {
        std::vector<float> final, logits;
        if (!hc_mix(weights, hc_rows[row], weights.output_hc_norm,
                    weights.output_hc_down, weights.output_hc_up, nullptr,
                    final, nullptr, error) ||
            !matvec(weights.backend, weights.output, final.data(), kEmbedding,
                    logits, error)) return false;
        row_hc.push_back(hc_rows[row]);
        row_logits.push_back(std::move(logits));
    }
    state.hc = row_hc.back();
    state.cur_pos += static_cast<int>(rows);
    state.last_token = tokens.back();
    return true;
}

} // namespace dflash::common
