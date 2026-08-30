#include "qwen4exp_internal.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace dflash::common {
namespace {
constexpr uint64_t kCapacityBytes = 128ULL << 30;
constexpr uint64_t kRuntimeReserveBytes = 8ULL << 30;
constexpr uint64_t kQsaBytesPerToken = 12ULL * 1152ULL * sizeof(float);
constexpr uint64_t kGdnStateBytes =
    36ULL * (3ULL * 10240ULL + 48ULL * 128ULL * 128ULL) * sizeof(float);

uint64_t add_saturating(uint64_t a, uint64_t b) {
    return b > UINT64_MAX - a ? UINT64_MAX : a + b;
}

uint64_t vector_bytes(const std::vector<float> & values) {
    return static_cast<uint64_t>(values.capacity()) * sizeof(float);
}
} // namespace

bool qwen4exp_weight_type_supported(ggml_type type, bool vector_or_norm) {
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 ||
        type == GGML_TYPE_BF16) return true;
    if (vector_or_norm) return false;
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q6_K ||
           type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q3_0_ROCMFPX ||
           type == GGML_TYPE_Q4_0_ROCMI4 ||
           type == GGML_TYPE_Q4_0_ROCMFP4_FAST;
}

float Qwen4ExpCowBuffer::at(size_t index) const {
    if (index >= size_) return 0.0f;
    return (*slabs_[index / kSlabFloats])[index % kSlabFloats];
}

void Qwen4ExpCowBuffer::append(const float * values, size_t count) {
    while (count > 0) {
        const size_t used = size_ % kSlabFloats;
        if (slabs_.empty() || used == 0) {
            auto slab = std::make_shared<std::vector<float>>();
            slab->reserve(kSlabFloats);
            slabs_.push_back(std::move(slab));
        } else if (!slabs_.back().unique()) {
            slabs_.back() =
                std::make_shared<std::vector<float>>(*slabs_.back());
        }
        std::vector<float> & slab = *slabs_.back();
        const size_t available = kSlabFloats - slab.size();
        const size_t take = std::min(available, count);
        slab.insert(slab.end(), values, values + take);
        values += take;
        count -= take;
        size_ += take;
    }
}

bool Qwen4ExpCowBuffer::copy_to(float * values, size_t count) const {
    if ((!values && count != 0) || count != size_) return false;
    size_t copied = 0;
    for (const auto & slab : slabs_) {
        if (!slab || slab->size() > count - copied) return false;
        std::copy(slab->begin(), slab->end(), values + copied);
        copied += slab->size();
    }
    return copied == count;
}

void Qwen4ExpCowBuffer::clear() {
    slabs_.clear();
    size_ = 0;
}

uint64_t Qwen4ExpCowBuffer::account_bytes(
        std::unordered_set<const void *> & seen) const {
    uint64_t total = 0;
    for (const auto & slab : slabs_) {
        if (slab && seen.insert(slab.get()).second)
            total = add_saturating(total, vector_bytes(*slab));
    }
    return total;
}

void Qwen4ExpState::clear() { *this = {}; ple_tokens = {248044, 248044}; }

uint64_t Qwen4ExpState::account_bytes(
        std::unordered_set<const void *> & seen) const {
    uint64_t total = add_saturating(vector_bytes(hc), vector_bytes(ple_conv));
    for (const auto & axis : mrope_positions)
        total = add_saturating(total,
            static_cast<uint64_t>(axis.capacity()) * sizeof(int32_t));
    for (const Qwen4ExpLayerState & layer : layers) {
        if (layer.conv && seen.insert(layer.conv.get()).second)
            total = add_saturating(total, vector_bytes(*layer.conv));
        if (layer.recurrent && seen.insert(layer.recurrent.get()).second)
            total = add_saturating(total, vector_bytes(*layer.recurrent));
        total = add_saturating(total, layer.key.account_bytes(seen));
        total = add_saturating(total, layer.value.account_bytes(seen));
        total = add_saturating(total, layer.index_key.account_bytes(seen));
    }
    return total;
}

Qwen4ExpMemoryPlan qwen4exp_memory_plan(uint64_t resident_weight_bytes,
                                        int max_ctx) {
    Qwen4ExpMemoryPlan plan;
    plan.resident_weight_bytes = resident_weight_bytes;
    plan.capacity_bytes = kCapacityBytes;
    plan.runtime_reserve_bytes = kRuntimeReserveBytes;
    // Policy accepts the native range or the exact official factor-4 target.
    // The latter is still expected to fail `fits` for the released weights on
    // 128-GiB UMA; calculating it here produces an honest component budget
    // instead of an opaque invalid-context sentinel.
    if (max_ctx <= 0 ||
        (max_ctx > EMBER_QWEN_NATIVE_CONTEXT &&
         max_ctx != EMBER_QWEN_YARN_MAX_CONTEXT)) {
        plan.total_bytes = UINT64_MAX;
        return plan;
    }
    plan.qsa_cache_bytes = static_cast<uint64_t>(max_ctx) * kQsaBytesPerToken;
    plan.recurrent_state_bytes = kGdnStateBytes;
    plan.total_bytes = add_saturating(resident_weight_bytes,
        add_saturating(plan.qsa_cache_bytes,
        add_saturating(plan.recurrent_state_bytes, plan.runtime_reserve_bytes)));
    plan.fits = plan.total_bytes <= plan.capacity_bytes;
    return plan;
}

std::array<int32_t, 16> qwen4exp_ple_rows(
        int32_t token, const std::array<int32_t, 2> & history) {
    constexpr int32_t eos = 248044;
    constexpr uint64_t multipliers[3] = {
        23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
    constexpr uint64_t vocab[16] = {
        20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,
        20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171};
    std::array<int32_t, 16> rows{};
    uint64_t offset = 0;
    for (int head = 0; head < 16; ++head) {
        const int order = head < 8 ? 2 : 3;
        uint64_t mixed = static_cast<uint64_t>(token) * multipliers[0] ^
                         static_cast<uint64_t>(history[1]) * multipliers[1];
        if (order == 3) {
            // PLE resets predecessor context strictly after EOS. Thus the
            // older member of a trigram is EOS too when the immediate
            // predecessor is EOS (HF Qwen4ExpTextPLE.forward and llama.cpp
            // #27742 llm_graph_input_ple::set_input).
            const int32_t older = history[1] == eos ? eos : history[0];
            mixed ^= static_cast<uint64_t>(older) * multipliers[2];
        }
        rows[head] = static_cast<int32_t>(mixed % vocab[head] + offset);
        offset += vocab[head];
    }
    return rows;
}

std::vector<int32_t> qwen4exp_qsa_selected_tokens(
        const std::vector<float> & raw_index_keys,
        const float * query_heads, int n_tokens) {
    constexpr int dimension = 128;
    constexpr int heads = 4;
    if (!query_heads || n_tokens <= 0 ||
        raw_index_keys.size() != static_cast<size_t>(n_tokens * dimension)) return {};
    const int complete = n_tokens / 4;
    const int selected_blocks = std::min(complete, 512);
    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(static_cast<size_t>(complete));
    for (int block = 0; block < complete; ++block) {
        std::array<float, dimension> pooled{};
        for (int member = 0; member < 4; ++member)
            for (int d = 0; d < dimension; ++d)
                pooled[d] += raw_index_keys[(block * 4 + member) * dimension + d] * 0.25f;
        double norm = 0.0;
        for (float value : pooled) norm += value * value;
        const float inv = 1.0f / std::sqrt(static_cast<float>(norm) + 1.0e-6f);
        for (float & value : pooled) value *= inv;
        float score = 0.0f;
        for (int head = 0; head < heads; ++head) {
            float dot = 0.0f;
            for (int d = 0; d < dimension; ++d)
                dot += query_heads[head * dimension + d] * pooled[d];
            score += std::max(dot, 0.0f);
        }
        scored.emplace_back(score / std::sqrt(static_cast<float>(dimension)), block);
    }
    if (selected_blocks < complete)
        std::partial_sort(scored.begin(), scored.begin() + selected_blocks,
                          scored.end(), [](const auto & a, const auto & b) {
                              return a.first != b.first ? a.first > b.first
                                                        : a.second < b.second;
                          });
    std::vector<int32_t> selected;
    selected.reserve(static_cast<size_t>(selected_blocks * 4 + n_tokens % 4));
    for (int i = 0; i < selected_blocks; ++i)
        for (int member = 0; member < 4; ++member)
            selected.push_back(scored[i].second * 4 + member);
    for (int token = complete * 4; token < n_tokens; ++token) selected.push_back(token);
    return selected;
}

} // namespace dflash::common
