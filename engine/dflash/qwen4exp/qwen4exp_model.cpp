#include "qwen4exp_model.h"

#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace dflash::common {
namespace {

constexpr uint32_t kLayers = 48;
constexpr uint32_t kEmbedding = 2560;
constexpr uint32_t kVocab = 248320;
constexpr uint32_t kContext = 262144;
constexpr uint32_t kHeads = 24;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kExperts = 512;
constexpr uint32_t kExpertsUsed = 10;
constexpr uint32_t kExpertFf = 640;
constexpr uint32_t kHc = 4;
constexpr uint32_t kHcDim = kHc * kEmbedding;
constexpr uint32_t kHcLowRank = 320;
constexpr uint32_t kIndexerHeads = 4;
constexpr uint32_t kIndexerDim = 128;
// Official text_config.indexer_budget. With ratio four this selects the best
// 512 complete blocks (2048 tokens), plus the incomplete causal tail.
constexpr uint32_t kIndexerTopK = 2048;
constexpr uint32_t kPleRows = 320001536;
constexpr uint32_t kPleHeadDim = 160;

constexpr std::array<uint64_t, 16> kPleVocabSizes = {
    20000003, 20000023, 20000033, 20000047,
    20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147,
    20000153, 20000159, 20000161, 20000171,
};

bool get_required_u32(const gguf_context * gctx, const char * key,
                      uint32_t expected, std::string & error) {
    const int64_t id = gguf_find_key(gctx, key);
    if (id < 0 || gguf_get_kv_type(gctx, id) != GGUF_TYPE_UINT32) {
        error = std::string("missing or non-u32 Qwen4Exp metadata: ") + key;
        return false;
    }
    const uint32_t actual = gguf_get_val_u32(gctx, id);
    if (actual != expected) {
        error = std::string("unsupported Qwen4Exp metadata ") + key +
                "=" + std::to_string(actual) + " (expected " +
                std::to_string(expected) + ")";
        return false;
    }
    return true;
}

bool get_required_f32(const gguf_context * gctx, const char * key,
                      float expected, std::string & error) {
    const int64_t id = gguf_find_key(gctx, key);
    if (id < 0 || gguf_get_kv_type(gctx, id) != GGUF_TYPE_FLOAT32) {
        error = std::string("missing or non-f32 Qwen4Exp metadata: ") + key;
        return false;
    }
    const float actual = gguf_get_val_f32(gctx, id);
    const float tolerance = std::fabs(expected) * 1.0e-6f + 1.0e-12f;
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        error = std::string("unsupported Qwen4Exp metadata ") + key;
        return false;
    }
    return true;
}

bool get_required_u32_array(const gguf_context * gctx, const char * key,
                            const std::vector<uint32_t> & expected,
                            std::string & error) {
    const int64_t id = gguf_find_key(gctx, key);
    const gguf_type element_type =
        id >= 0 && gguf_get_kv_type(gctx, id) == GGUF_TYPE_ARRAY
            ? gguf_get_arr_type(gctx, id)
            : GGUF_TYPE_COUNT;
    if (id < 0 || gguf_get_kv_type(gctx, id) != GGUF_TYPE_ARRAY ||
        (element_type != GGUF_TYPE_UINT32 &&
         element_type != GGUF_TYPE_INT32) ||
        gguf_get_arr_n(gctx, id) != expected.size()) {
        error = std::string("missing or incompatible Qwen4Exp array: ") + key;
        return false;
    }
    const void * data = gguf_get_arr_data(gctx, id);
    if (!data) {
        error = std::string("Qwen4Exp array has no data: ") + key;
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        // llama.cpp conversion/qwen4exp.py passes ordinary Python integers to
        // GGUFWriter.add_array(), which serializes these three architectural
        // arrays as INT32. Older artifacts may carry UINT32. Accept either
        // exact 32-bit representation, but never reinterpret a negative value
        // or relax the architecture contract.
        const bool matches = element_type == GGUF_TYPE_UINT32
            ? static_cast<const uint32_t *>(data)[i] == expected[i]
            : static_cast<const int32_t *>(data)[i] >= 0 &&
                  static_cast<uint32_t>(
                      static_cast<const int32_t *>(data)[i]) == expected[i];
        if (!matches) {
            error = std::string("unsupported Qwen4Exp array value: ") + key +
                    "[" + std::to_string(i) + "]";
            return false;
        }
    }
    return true;
}

bool get_required_u64_array(const gguf_context * gctx, const char * key,
                            const std::vector<uint64_t> & expected,
                            std::string & error) {
    const int64_t id = gguf_find_key(gctx, key);
    if (id < 0 || gguf_get_kv_type(gctx, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(gctx, id) != GGUF_TYPE_UINT64 ||
        gguf_get_arr_n(gctx, id) != expected.size()) {
        error = std::string("missing or incompatible Qwen4Exp u64 array: ") + key;
        return false;
    }
    const auto * values = static_cast<const uint64_t *>(gguf_get_arr_data(gctx, id));
    if (!values) {
        error = std::string("Qwen4Exp array has no data: ") + key;
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (values[i] != expected[i]) {
            error = std::string("unsupported Qwen4Exp array value: ") + key +
                    "[" + std::to_string(i) + "]";
            return false;
        }
    }
    return true;
}

using TensorMap = std::unordered_map<std::string, ggml_tensor *>;

bool require_tensor(const TensorMap & tensors, const std::string & name,
                    std::initializer_list<int64_t> shape,
                    std::string & error) {
    const auto found = tensors.find(name);
    if (found == tensors.end() || !found->second) {
        error = "missing required Qwen4Exp tensor: " + name;
        return false;
    }
    ggml_tensor * tensor = found->second;
    if (ggml_n_dims(tensor) != static_cast<int>(shape.size())) {
        error = "wrong rank for Qwen4Exp tensor: " + name;
        return false;
    }
    size_t dim = 0;
    for (int64_t expected : shape) {
        if (tensor->ne[dim] != expected) {
            error = "wrong shape for Qwen4Exp tensor: " + name +
                    " at ne[" + std::to_string(dim) + "]=" +
                    std::to_string(tensor->ne[dim]) + " (expected " +
                    std::to_string(expected) + ")";
            return false;
        }
        ++dim;
    }
    return true;
}

std::string block_name(uint32_t layer, const char * suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

bool validate_tensors(const TensorMap & tensors, std::string & error) {
    if (!require_tensor(tensors, "token_embd.weight", {kEmbedding, kVocab}, error) ||
        !require_tensor(tensors, "output.weight", {kEmbedding, kVocab}, error) ||
        !require_tensor(tensors, "output_hc_norm.weight", {kHcDim}, error) ||
        !require_tensor(tensors, "output_hc_down.weight", {kHcDim, kHcLowRank}, error) ||
        !require_tensor(tensors, "output_hc_up.weight", {kHcLowRank, kHcDim}, error) ||
        !require_tensor(tensors, "per_layer_token_embd.weight",
                        {kPleHeadDim, kPleRows}, error)) {
        return false;
    }

    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        auto req = [&](const char * suffix,
                       std::initializer_list<int64_t> shape) {
            return require_tensor(tensors, block_name(layer, suffix), shape, error);
        };

        if (!req("hc_attn_norm.weight", {kHcDim}) ||
            !req("hc_attn_down.weight", {kHcDim, kHcLowRank}) ||
            !req("hc_attn_up.weight", {kHcLowRank, kHcDim}) ||
            !req("hc_attn_inject.weight", {kHcDim, kHc}) ||
            !req("hc_ffn_norm.weight", {kHcDim}) ||
            !req("hc_ffn_down.weight", {kHcDim, kHcLowRank}) ||
            !req("hc_ffn_up.weight", {kHcLowRank, kHcDim}) ||
            !req("hc_ffn_inject.weight", {kHcDim, kHc}) ||
            !req("ffn_gate_inp.weight", {kEmbedding, kExperts}) ||
            !req("ffn_gate_up_exps.weight", {kEmbedding, 2 * kExpertFf, kExperts}) ||
            !req("ffn_down_exps.weight", {kExpertFf, kEmbedding, kExperts}) ||
            !req("ffn_gate_inp_shexp.weight", {kEmbedding}) ||
            !req("ffn_gate_shexp.weight", {kEmbedding, kExpertFf}) ||
            !req("ffn_up_shexp.weight", {kEmbedding, kExpertFf}) ||
            !req("ffn_down_shexp.weight", {kExpertFf, kEmbedding})) {
            return false;
        }

        const bool qsa = (layer + 1) % 4 == 0;
        if (qsa) {
            if (!req("attn_q.weight", {kEmbedding, 2 * kHeads * kHeadDim}) ||
                !req("attn_k.weight", {kEmbedding, kKvHeads * kHeadDim}) ||
                !req("attn_v.weight", {kEmbedding, kKvHeads * kHeadDim}) ||
                !req("attn_output.weight", {kHeads * kHeadDim, kEmbedding}) ||
                !req("attn_q_norm.weight", {kHeadDim}) ||
                !req("attn_k_norm.weight", {kHeadDim}) ||
                !req("indexer.q_proj.weight", {kEmbedding, kIndexerHeads * kIndexerDim}) ||
                !req("indexer.k_proj.weight", {kEmbedding, kIndexerDim}) ||
                !req("indexer.q_norm.weight", {kIndexerDim}) ||
                !req("indexer.k_norm.weight", {kIndexerDim})) {
                return false;
            }
        } else if (!req("attn_qkv.weight", {kEmbedding, 10240}) ||
                   !req("attn_gate.weight", {kEmbedding, 6144}) ||
                   !req("ssm_conv1d.weight", {4, 10240}) ||
                   !req("ssm_a", {48}) ||
                   !req("ssm_alpha.weight", {kEmbedding, 48}) ||
                   !req("ssm_beta.weight", {kEmbedding, 48}) ||
                   !req("ssm_dt.bias", {48}) ||
                   !req("ssm_norm.weight", {128}) ||
                   !req("ssm_out.weight", {6144, kEmbedding})) {
            return false;
        }

        if (layer == 1 &&
            (!req("ple_key.weight", {kEmbedding, kHcDim}) ||
             !req("ple_value.weight", {kEmbedding, kEmbedding}) ||
             !req("ple_norm_key.weight", {kHcDim}) ||
             !req("ple_norm_query.weight", {kHcDim}) ||
             !req("ple_norm_conv.weight", {kHcDim}) ||
             !req("ple_conv1d.weight", {4, kHcDim}))) {
            return false;
        }
    }
    return true;
}

bool is_split_key(const char * key) {
    return key && (std::strcmp(key, "split.no") == 0 ||
                   std::strcmp(key, "split.count") == 0 ||
                   std::strcmp(key, "split.tensors.count") == 0);
}

size_t gguf_scalar_size(gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

bool metadata_value_equal(const gguf_context * left, int64_t left_id,
                          const gguf_context * right, int64_t right_id) {
    const gguf_type type = gguf_get_kv_type(left, left_id);
    if (type != gguf_get_kv_type(right, right_id)) return false;
    if (type == GGUF_TYPE_STRING) {
        const char * a = gguf_get_val_str(left, left_id);
        const char * b = gguf_get_val_str(right, right_id);
        return a && b && std::strcmp(a, b) == 0;
    }
    if (type == GGUF_TYPE_ARRAY) {
        const gguf_type element = gguf_get_arr_type(left, left_id);
        const size_t count = gguf_get_arr_n(left, left_id);
        if (element != gguf_get_arr_type(right, right_id) ||
            count != gguf_get_arr_n(right, right_id)) return false;
        if (element == GGUF_TYPE_STRING) {
            for (size_t i = 0; i < count; ++i) {
                const char * a = gguf_get_arr_str(left, left_id, i);
                const char * b = gguf_get_arr_str(right, right_id, i);
                if (!a || !b || std::strcmp(a, b) != 0) return false;
            }
            return true;
        }
        const size_t element_size = gguf_scalar_size(element);
        if (element_size == 0 ||
            count > std::numeric_limits<size_t>::max() / element_size) {
            return false;
        }
        const size_t bytes = count * element_size;
        const void * a = gguf_get_arr_data(left, left_id);
        const void * b = gguf_get_arr_data(right, right_id);
        return bytes == 0 || (a && b && std::memcmp(a, b, bytes) == 0);
    }
    const size_t bytes = gguf_scalar_size(type);
    const void * a = gguf_get_val_data(left, left_id);
    const void * b = gguf_get_val_data(right, right_id);
    return bytes != 0 && a && b && std::memcmp(a, b, bytes) == 0;
}

bool continuation_metadata_matches_reference(
        const gguf_context * reference, const gguf_context * candidate,
        std::string & error) {
    // Pinned llama.cpp writes the complete model metadata only to shard zero.
    // Ember's streaming quantizer copies each input shard, so continuations
    // retain only the split locators plus quantization/mode evidence. Treat
    // shard zero as authoritative while requiring every repeated continuation
    // key to exist there with a byte-exact value. This accepts both canonical
    // first-only metadata and older fully replicated sets, but never an
    // injected or contradictory continuation value.
    for (int64_t i = 0; i < gguf_get_n_kv(candidate); ++i) {
        const char * key = gguf_get_key(candidate, i);
        if (is_split_key(key)) continue;
        const int64_t authoritative = key ? gguf_find_key(reference, key) : -1;
        if (authoritative < 0 ||
            !metadata_value_equal(reference, authoritative, candidate, i)) {
            error = "Qwen4Exp continuation metadata differs from "
                    "authoritative shard: " +
                    std::string(key ? key : "(null)");
            return false;
        }
    }
    return true;
}

struct MetadataShard {
    std::string path;
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;
    size_t file_size = 0;

    ~MetadataShard() {
        if (gguf) gguf_free(gguf);
        if (meta) ggml_free(meta);
    }
};

struct SplitInfo {
    bool present = false;
    uint16_t number = 0;
    uint16_t count = 1;
    int32_t tensors = 0;
};

bool read_split_info(const gguf_context * context, SplitInfo & result,
                     std::string & error) {
    const int64_t number = gguf_find_key(context, "split.no");
    const int64_t count = gguf_find_key(context, "split.count");
    const int64_t tensors = gguf_find_key(context, "split.tensors.count");
    if (number < 0 && count < 0 && tensors < 0) return true;
    if (number < 0 || count < 0 || tensors < 0) {
        error = "incomplete GGUF split metadata";
        return false;
    }
    if (gguf_get_kv_type(context, number) != GGUF_TYPE_UINT16 ||
        gguf_get_kv_type(context, count) != GGUF_TYPE_UINT16 ||
        gguf_get_kv_type(context, tensors) != GGUF_TYPE_INT32) {
        error = "split.no/count must be UINT16 and split.tensors.count must be INT32";
        return false;
    }
    result.present = true;
    result.number = gguf_get_val_u16(context, number);
    result.count = gguf_get_val_u16(context, count);
    result.tensors = gguf_get_val_i32(context, tensors);
    if (result.count == 0 || result.number >= result.count || result.tensors < 0) {
        error = "invalid GGUF split metadata values";
        return false;
    }
    return true;
}

bool load_metadata_set(const std::vector<std::string> & paths,
                       std::vector<std::unique_ptr<MetadataShard>> & shards,
                       TensorMap & tensors, std::string & error) {
    shards.clear();
    tensors.clear();
    int32_t declared_tensors = -1;
    size_t aggregate_tensors = 0;
    for (size_t shard_index = 0; shard_index < paths.size(); ++shard_index) {
        auto shard = std::make_unique<MetadataShard>();
        shard->path = paths[shard_index];
        gguf_init_params params{/*.no_alloc=*/true, /*.ctx=*/&shard->meta};
        shard->gguf = gguf_init_from_file(shard->path.c_str(), params);
        if (!shard->gguf || !shard->meta) {
            error = "failed to open Qwen4Exp GGUF shard metadata: " + shard->path;
            return false;
        }
        struct stat status{};
        if (::stat(shard->path.c_str(), &status) != 0 || status.st_size <= 0 ||
            static_cast<uintmax_t>(status.st_size) >
                std::numeric_limits<size_t>::max()) {
            error = "failed to stat Qwen4Exp GGUF shard: " + shard->path;
            return false;
        }
        shard->file_size = static_cast<size_t>(status.st_size);

        SplitInfo split;
        if (!read_split_info(shard->gguf, split, error)) return false;
        if (paths.size() > 1) {
            if (!split.present || split.count != paths.size() ||
                split.number != shard_index) {
                error = "GGUF split.no/count does not match ordered shard filenames";
                return false;
            }
            if (declared_tensors < 0) declared_tensors = split.tensors;
            else if (declared_tensors != split.tensors) {
                error = "inconsistent split.tensors.count across shards";
                return false;
            }
        } else if (split.present &&
                   (split.count != 1 || split.number != 0)) {
            error = "GGUF split metadata requires a complete ordered shard set";
            return false;
        } else if (split.present) {
            declared_tensors = split.tensors;
        }
        if (shard_index != 0 &&
            !continuation_metadata_matches_reference(
                shards.front()->gguf, shard->gguf, error)) {
            return false;
        }

        const int64_t local_count = gguf_get_n_tensors(shard->gguf);
        if (local_count < 0 ||
            static_cast<uint64_t>(local_count) >
                std::numeric_limits<size_t>::max() - aggregate_tensors) {
            error = "Qwen4Exp aggregate tensor count overflow";
            return false;
        }
        aggregate_tensors += static_cast<size_t>(local_count);
        const size_t data_offset = gguf_get_data_offset(shard->gguf);
        for (int64_t tensor_index = 0; tensor_index < local_count; ++tensor_index) {
            const char * name = gguf_get_tensor_name(shard->gguf, tensor_index);
            ggml_tensor * tensor = name ? ggml_get_tensor(shard->meta, name) : nullptr;
            if (!name || !tensor || !tensors.emplace(name, tensor).second) {
                error = "duplicate or invalid tensor across GGUF shards: " +
                        std::string(name ? name : "(null)");
                return false;
            }
            const size_t offset = gguf_get_tensor_offset(shard->gguf, tensor_index);
            const size_t bytes = gguf_get_tensor_size(shard->gguf, tensor_index);
            if (data_offset > shard->file_size ||
                offset > shard->file_size - data_offset ||
                bytes > shard->file_size - data_offset - offset) {
                error = "Qwen4Exp tensor extends past end of GGUF shard: " +
                        std::string(name);
                return false;
            }
        }
        shards.push_back(std::move(shard));
    }
    if (declared_tensors >= 0 &&
        static_cast<size_t>(declared_tensors) != aggregate_tensors) {
        error = "split.tensors.count does not match global tensor inventory";
        return false;
    }
    return true;
}

}  // namespace

ModelArchitecture inspect_gguf_architecture(const char * model_path,
                                            std::string & error) {
    error.clear();
    std::vector<std::string> paths;
    if (!qwen4exp_discover_gguf_shards(model_path, paths, error))
        return ModelArchitecture::UNKNOWN;
    gguf_init_params params = {/*.no_alloc=*/true, /*.ctx=*/nullptr};
    gguf_context * gctx = gguf_init_from_file(paths.front().c_str(), params);
    if (!gctx) {
        error = std::string("failed to open GGUF: ") + model_path;
        return ModelArchitecture::UNKNOWN;
    }
    const int64_t id = gguf_find_key(gctx, "general.architecture");
    if (id < 0 || gguf_get_kv_type(gctx, id) != GGUF_TYPE_STRING) {
        error = "missing or non-string general.architecture";
        gguf_free(gctx);
        return ModelArchitecture::UNKNOWN;
    }
    const char * name = gguf_get_val_str(gctx, id);
    const ModelArchitecture architecture = model_architecture_from_name(name);
    if (architecture == ModelArchitecture::UNKNOWN) {
        error = std::string("unsupported general.architecture: ") +
                (name ? name : "(null)");
    }
    gguf_free(gctx);
    return architecture;
}

bool validate_qwen4exp_gguf(const char * model_path, std::string & error) {
    error.clear();
    std::vector<std::string> paths;
    if (!qwen4exp_discover_gguf_shards(model_path, paths, error)) return false;
    std::vector<std::unique_ptr<MetadataShard>> shards;
    TensorMap tensors;
    if (!load_metadata_set(paths, shards, tensors, error)) return false;
    gguf_context * gctx = shards.front()->gguf;

    const int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gctx, arch_id) != GGUF_TYPE_STRING ||
        model_architecture_from_name(gguf_get_val_str(gctx, arch_id)) !=
            ModelArchitecture::QWEN4_EXP) {
        error = "Qwen4Exp validator received a non-qwen4exp GGUF";
    }

    const struct RequiredU32 { const char * key; uint32_t value; } required[] = {
        {"qwen4exp.context_length", kContext},
        {"qwen4exp.embedding_length", kEmbedding},
        {"qwen4exp.block_count", kLayers},
        {"qwen4exp.attention.head_count", kHeads},
        {"qwen4exp.attention.head_count_kv", kKvHeads},
        {"qwen4exp.attention.key_length", kHeadDim},
        {"qwen4exp.rope.dimension_count", 64},
        {"qwen4exp.expert_count", kExperts},
        {"qwen4exp.expert_used_count", kExpertsUsed},
        {"qwen4exp.expert_feed_forward_length", kExpertFf},
        {"qwen4exp.expert_shared_feed_forward_length", kExpertFf},
        {"qwen4exp.ssm.conv_kernel", 4},
        {"qwen4exp.ssm.inner_size", 6144},
        {"qwen4exp.ssm.state_size", 128},
        {"qwen4exp.ssm.time_step_rank", 48},
        {"qwen4exp.ssm.group_count", 16},
        {"qwen4exp.hyper_connection.count", kHc},
        {"qwen4exp.hyper_connection.low_rank", kHcLowRank},
        {"qwen4exp.attention.indexer.head_count", kIndexerHeads},
        {"qwen4exp.attention.indexer.key_length", kIndexerDim},
        {"qwen4exp.attention.indexer.top_k", kIndexerTopK},
        {"qwen4exp.ple.ngram_size", 3},
        {"qwen4exp.ple.heads_per_ngram", 8},
        {"qwen4exp.ple.conv_kernel", 4},
        {"qwen4exp.ple.eos_token_id", 248044},
        {"qwen4exp.ple.image_token_id", 248056},
        {"qwen4exp.embedding_length_per_layer_input", kPleHeadDim},
    };
    for (const auto & item : required) {
        if (!error.empty() ||
            !get_required_u32(gctx, item.key, item.value, error)) break;
    }
    if (error.empty()) {
        (void)get_required_f32(gctx, "qwen4exp.rope.freq_base", 10000000.0f, error);
    }
    if (error.empty()) {
        (void)get_required_f32(gctx,
                               "qwen4exp.attention.layer_norm_rms_epsilon",
                               1.0e-6f, error);
    }

    if (error.empty()) {
        std::vector<uint32_t> ratios(kLayers, 0);
        for (uint32_t i = 3; i < kLayers; i += 4) ratios[i] = 4;
        (void)get_required_u32_array(gctx, "qwen4exp.attention.compress_ratios",
                                     ratios, error);
    }
    if (error.empty()) {
        (void)get_required_u32_array(gctx, "qwen4exp.rope.dimension_sections",
                                     {11, 11, 10}, error);
    }
    if (error.empty()) {
        (void)get_required_u32_array(gctx, "qwen4exp.ple.layers", {1}, error);
    }
    if (error.empty()) {
        std::vector<uint64_t> offsets;
        offsets.reserve(kPleVocabSizes.size());
        uint64_t offset = 0;
        for (uint64_t size : kPleVocabSizes) {
            offsets.push_back(offset);
            offset += size;
        }
        (void)get_required_u64_array(
            gctx, "qwen4exp.ple.head_offsets", offsets, error);
    }
    if (error.empty()) {
        (void)get_required_u64_array(
            gctx, "qwen4exp.ple.head_vocab_sizes",
            std::vector<uint64_t>(kPleVocabSizes.begin(), kPleVocabSizes.end()),
            error);
    }
    if (error.empty()) {
        // Read directly from the released I64 safetensor buffers. These values
        // must never pass through float32 (which rounds their 45-bit payloads).
        (void)get_required_u64_array(
            gctx, "qwen4exp.ple.layer_multipliers",
            {23703573157769ULL, 20109073645365ULL, 8052911324071ULL},
            error);
    }

    const int64_t tokens_id = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (error.empty() &&
        (tokens_id < 0 || gguf_get_kv_type(gctx, tokens_id) != GGUF_TYPE_ARRAY ||
         gguf_get_arr_type(gctx, tokens_id) != GGUF_TYPE_STRING ||
         gguf_get_arr_n(gctx, tokens_id) != kVocab)) {
        error = "Qwen4Exp tokenizer vocabulary must contain 248320 entries";
    }
    const int64_t merges_id = gguf_find_key(gctx, "tokenizer.ggml.merges");
    if (error.empty() &&
        (merges_id < 0 || gguf_get_kv_type(gctx, merges_id) != GGUF_TYPE_ARRAY ||
         gguf_get_arr_type(gctx, merges_id) != GGUF_TYPE_STRING ||
         gguf_get_arr_n(gctx, merges_id) != 247587)) {
        error = "Qwen4Exp tokenizer must contain the 247587 released BPE merges";
    }
    if (error.empty()) {
        const int64_t pre_id = gguf_find_key(gctx, "tokenizer.ggml.pre");
        if (pre_id < 0 || gguf_get_kv_type(gctx, pre_id) != GGUF_TYPE_STRING ||
            std::strcmp(gguf_get_val_str(gctx, pre_id), "qwen2") != 0) {
            error = "Qwen4Exp requires tokenizer.ggml.pre=qwen2";
        }
    }
    if (error.empty()) {
        (void)get_required_u32(gctx, "tokenizer.ggml.bos_token_id", 248044, error);
    }
    if (error.empty()) {
        (void)get_required_u32(gctx, "tokenizer.ggml.eos_token_id", 248046, error);
    }
    if (error.empty()) {
        (void)get_required_u32(gctx, "tokenizer.ggml.padding_token_id", 248044,
                               error);
    }
    if (error.empty()) {
        static const std::array<const char *, 33> special_tokens = {
            "<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
            "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
            "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
            "<|image_pad|>", "<|video_pad|>", "<tool_call>",
            "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>",
            "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>",
            "<|file_sep|>", "<tool_response>", "</tool_response>",
            "<think>", "</think>", "<|audio_start|>", "<|audio_end|>",
            "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>",
            "<tts_text_bos_single>", "<|audio_pad|>",
        };
        for (size_t i = 0; i < special_tokens.size(); ++i) {
            const char * actual = gguf_get_arr_str(gctx, tokens_id, 248044 + i);
            if (!actual || std::strcmp(actual, special_tokens[i]) != 0) {
                error = "Qwen4Exp special token mismatch at id " +
                        std::to_string(248044 + i);
                break;
            }
        }
    }
    if (error.empty()) {
        const int64_t types_id = gguf_find_key(gctx, "tokenizer.ggml.token_type");
        if (types_id < 0 || gguf_get_kv_type(gctx, types_id) != GGUF_TYPE_ARRAY ||
            (gguf_get_arr_type(gctx, types_id) != GGUF_TYPE_UINT32 &&
             gguf_get_arr_type(gctx, types_id) != GGUF_TYPE_INT32) ||
            gguf_get_arr_n(gctx, types_id) != kVocab ||
            !gguf_get_arr_data(gctx, types_id)) {
            error = "Qwen4Exp requires a full tokenizer token-type array";
        } else {
            const gguf_type type = gguf_get_arr_type(gctx, types_id);
            const void * data = gguf_get_arr_data(gctx, types_id);
            for (uint32_t id = 248044; id <= 248076; ++id) {
                const uint32_t value = type == GGUF_TYPE_UINT32
                    ? static_cast<const uint32_t *>(data)[id]
                    : static_cast<uint32_t>(static_cast<const int32_t *>(data)[id]);
                if (value != 3 && value != 4) {
                    error = "Qwen4Exp added token is not marked special at id " +
                            std::to_string(id);
                    break;
                }
            }
        }
    }
    if (error.empty()) (void)validate_tensors(tensors, error);
    return error.empty();
}

}  // namespace dflash::common
