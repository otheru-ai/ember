#include "qwen4exp_mtp.h"

#include "qwen4exp_frontier.h"

#include "gguf.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <initializer_list>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace dflash::common {
namespace {

struct RequiredTensor {
    const char * name;
    std::initializer_list<int64_t> shape;
};

const RequiredTensor kRequired[] = {
    {"mtp_hc_norm.weight", {10240}},
    {"mtp_hc_down.weight", {10240, 320}},
    {"mtp_hc_up.weight", {320, 10240}},
    {"mtp.hc_attn_inject.weight", {10240, 4}},
    {"mtp.hc_attn_norm.weight", {10240}},
    {"mtp.hc_attn_down.weight", {10240, 320}},
    {"mtp.hc_attn_up.weight", {320, 10240}},
    {"mtp.ffn_gate_up_exps.weight", {2560, 1280, 512}},
    {"mtp.ffn_down_exps.weight", {640, 2560, 512}},
    {"mtp.ffn_gate_inp.weight", {2560, 512}},
    {"mtp.ffn_gate_inp_shexp.weight", {2560, 1}},
    {"mtp.ffn_gate_shexp.weight", {2560, 640}},
    {"mtp.ffn_up_shexp.weight", {2560, 640}},
    {"mtp.ffn_down_shexp.weight", {640, 2560}},
    {"mtp.hc_ffn_inject.weight", {10240, 4}},
    {"mtp.hc_ffn_norm.weight", {10240}},
    {"mtp.hc_ffn_down.weight", {10240, 320}},
    {"mtp.hc_ffn_up.weight", {320, 10240}},
    {"mtp.indexer.q_proj.weight", {2560, 512}},
    {"mtp.indexer.k_proj.weight", {2560, 128}},
    {"mtp.indexer.k_norm.weight", {128}},
    {"mtp.indexer.q_norm.weight", {128}},
    {"mtp.attn_k_norm.weight", {256}},
    {"mtp.attn_output.weight", {6144, 2560}},
    {"mtp.attn_k.weight", {2560, 512}},
    {"mtp.attn_q.weight", {2560, 12288}},
    {"mtp.attn_v.weight", {2560, 512}},
    {"mtp.attn_q_norm.weight", {256}},
    {"mtp_pre_emb_norm.weight", {2560}},
    {"mtp_pre_hc_norm.weight", {10240}},
    {"mtp_fc_emb.weight", {2560, 2560}},
    {"mtp_fc_hc.weight", {2560, 2560}},
};

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0 || value > SIZE_MAX - (alignment - 1)) return SIZE_MAX;
    return (value + alignment - 1) / alignment * alignment;
}

bool key_u32(const gguf_context * ctx, const char * name, uint32_t expected,
             std::string & error) {
    const int64_t id = gguf_find_key(ctx, name);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32 ||
        gguf_get_val_u32(ctx, id) != expected) {
        error = std::string("invalid Qwen4Exp MTP metadata: ") + name;
        return false;
    }
    return true;
}

bool key_string(const gguf_context * ctx, const char * name,
                const char * expected, std::string & error) {
    const int64_t id = gguf_find_key(ctx, name);
    const char * value = id >= 0 && gguf_get_kv_type(ctx, id) == GGUF_TYPE_STRING
        ? gguf_get_val_str(ctx, id) : nullptr;
    if (!value || std::strcmp(value, expected) != 0) {
        error = std::string("invalid Qwen4Exp MTP metadata: ") + name;
        return false;
    }
    return true;
}

bool validate_contract(gguf_context * gctx, ggml_context * meta,
                       std::string & error) {
    if (!key_string(gctx, "general.architecture", "qwen4exp-mtp", error) ||
        !key_string(gctx, "qwen4exp-mtp.source_revision",
                    "f5d08274bafd880402bd16f5e3e6c514136ec06c", error) ||
        !key_u32(gctx, "qwen4exp-mtp.block_count", 1, error) ||
        !key_u32(gctx, "qwen4exp-mtp.embedding_length", 2560, error) ||
        !key_u32(gctx, "qwen4exp-mtp.hyper_connection_count", 4, error) ||
        !key_u32(gctx, "qwen4exp-mtp.hyper_connection_low_rank", 320, error) ||
        !key_u32(gctx, "qwen4exp-mtp.attention.head_count", 24, error) ||
        !key_u32(gctx, "qwen4exp-mtp.attention.head_count_kv", 2, error) ||
        !key_u32(gctx, "qwen4exp-mtp.attention.key_length", 256, error) ||
        !key_u32(gctx, "qwen4exp-mtp.indexer.head_count", 4, error) ||
        !key_u32(gctx, "qwen4exp-mtp.indexer.key_length", 128, error) ||
        !key_u32(gctx, "qwen4exp-mtp.indexer.top_k", 2048, error) ||
        !key_u32(gctx, "qwen4exp-mtp.indexer.compress_ratio", 4, error) ||
        !key_u32(gctx, "qwen4exp-mtp.expert_count", 512, error) ||
        !key_u32(gctx, "qwen4exp-mtp.expert_used_count", 10, error)) return false;

    const int64_t shared_id =
        gguf_find_key(gctx, "qwen4exp-mtp.shared_main_weights");
    if (shared_id < 0 || gguf_get_kv_type(gctx, shared_id) != GGUF_TYPE_BOOL ||
        !gguf_get_val_bool(gctx, shared_id)) {
        error = "Qwen4Exp MTP companion must share the target embedding/head";
        return false;
    }
    if (gguf_get_n_tensors(gctx) !=
        static_cast<int64_t>(sizeof(kRequired) / sizeof(kRequired[0]))) {
        error = "Qwen4Exp MTP companion tensor count is not exact";
        return false;
    }
    for (const RequiredTensor & required : kRequired) {
        ggml_tensor * tensor = ggml_get_tensor(meta, required.name);
        if (!tensor || ggml_n_dims(tensor) !=
                           static_cast<int>(required.shape.size())) {
            error = std::string("missing/wrong-rank Qwen4Exp MTP tensor: ") +
                    required.name;
            return false;
        }
        size_t dimension = 0;
        for (int64_t expected : required.shape) {
            if (tensor->ne[dimension++] != expected) {
                error = std::string("wrong-shape Qwen4Exp MTP tensor: ") +
                        required.name;
                return false;
            }
        }
        const bool vector = required.shape.size() == 1;
        if (!qwen4exp_weight_type_supported(tensor->type, vector)) {
            error = std::string("unsupported Qwen4Exp MTP tensor type: ") +
                    required.name;
            return false;
        }
    }
    return true;
}

void bind_layer(Qwen4ExpLayer & layer, const char * name,
                ggml_tensor * tensor) {
    if (std::strcmp(name, "mtp.hc_attn_norm.weight") == 0) layer.hc_attn_norm = tensor;
    else if (std::strcmp(name, "mtp.hc_attn_down.weight") == 0) layer.hc_attn_down = tensor;
    else if (std::strcmp(name, "mtp.hc_attn_up.weight") == 0) layer.hc_attn_up = tensor;
    else if (std::strcmp(name, "mtp.hc_attn_inject.weight") == 0) layer.hc_attn_inject = tensor;
    else if (std::strcmp(name, "mtp.hc_ffn_norm.weight") == 0) layer.hc_ffn_norm = tensor;
    else if (std::strcmp(name, "mtp.hc_ffn_down.weight") == 0) layer.hc_ffn_down = tensor;
    else if (std::strcmp(name, "mtp.hc_ffn_up.weight") == 0) layer.hc_ffn_up = tensor;
    else if (std::strcmp(name, "mtp.hc_ffn_inject.weight") == 0) layer.hc_ffn_inject = tensor;
    else if (std::strcmp(name, "mtp.attn_q.weight") == 0) layer.attn_q = tensor;
    else if (std::strcmp(name, "mtp.attn_k.weight") == 0) layer.attn_k = tensor;
    else if (std::strcmp(name, "mtp.attn_v.weight") == 0) layer.attn_v = tensor;
    else if (std::strcmp(name, "mtp.attn_output.weight") == 0) layer.attn_output = tensor;
    else if (std::strcmp(name, "mtp.attn_q_norm.weight") == 0) layer.attn_q_norm = tensor;
    else if (std::strcmp(name, "mtp.attn_k_norm.weight") == 0) layer.attn_k_norm = tensor;
    else if (std::strcmp(name, "mtp.indexer.q_proj.weight") == 0) layer.index_q = tensor;
    else if (std::strcmp(name, "mtp.indexer.k_proj.weight") == 0) layer.index_k = tensor;
    else if (std::strcmp(name, "mtp.indexer.q_norm.weight") == 0) layer.index_q_norm = tensor;
    else if (std::strcmp(name, "mtp.indexer.k_norm.weight") == 0) layer.index_k_norm = tensor;
    else if (std::strcmp(name, "mtp.ffn_gate_inp.weight") == 0) layer.router = tensor;
    else if (std::strcmp(name, "mtp.ffn_gate_inp_shexp.weight") == 0) layer.shared_gate_input = tensor;
    else if (std::strcmp(name, "mtp.ffn_gate_shexp.weight") == 0) layer.shared_gate = tensor;
    else if (std::strcmp(name, "mtp.ffn_up_shexp.weight") == 0) layer.shared_up = tensor;
    else if (std::strcmp(name, "mtp.ffn_down_shexp.weight") == 0) layer.shared_down = tensor;
    else if (std::strcmp(name, "mtp.ffn_gate_up_exps.weight") == 0) layer.experts_gate_up_tensor = tensor;
    else if (std::strcmp(name, "mtp.ffn_down_exps.weight") == 0) layer.experts_down_tensor = tensor;
}

} // namespace

bool load_qwen4exp_mtp_gguf(const std::string & path,
                            ggml_backend_t backend,
                            Qwen4ExpMtpWeights & out,
                            std::string & error) {
    error.clear();
    if (!backend || path.empty()) {
        error = "invalid Qwen4Exp MTP companion/backend";
        return false;
    }
    ggml_context * meta = nullptr;
    gguf_init_params params{true, &meta};
    gguf_context * gctx = gguf_init_from_file(path.c_str(), params);
    if (!gctx || !meta || !validate_contract(gctx, meta, error)) {
        if (gctx) gguf_free(gctx);
        if (meta) ggml_free(meta);
        if (error.empty()) error = "failed to open Qwen4Exp MTP companion";
        return false;
    }
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    struct stat status{};
    if (fd < 0 || fstat(fd, &status) != 0 || status.st_size <= 0 ||
        static_cast<uintmax_t>(status.st_size) > SIZE_MAX) {
        if (fd >= 0) ::close(fd);
        gguf_free(gctx); ggml_free(meta);
        error = "failed to stat Qwen4Exp MTP companion";
        return false;
    }
    const size_t file_size = static_cast<size_t>(status.st_size);
    void * mapping = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        ::close(fd); gguf_free(gctx); ggml_free(meta);
        error = "failed to mmap Qwen4Exp MTP companion";
        return false;
    }
    out.backend = backend;
    out.shards.push_back({meta, mapping, file_size, fd, path});
    const auto fail = [&]() {
        gguf_free(gctx);
        free_qwen4exp_mtp_weights(out);
        return false;
    };

    struct Allocation {
        ggml_tensor * tensor;
        size_t file_offset;
        size_t bytes;
        size_t buffer_offset;
    };
    std::vector<Allocation> allocations;
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t data_offset = gguf_get_data_offset(gctx);
    size_t total = 0;
    const int64_t count = gguf_get_n_tensors(gctx);
    for (int64_t index = 0; index < count; ++index) {
        const char * name = gguf_get_tensor_name(gctx, index);
        ggml_tensor * tensor = ggml_get_tensor(meta, name);
        const size_t offset = gguf_get_tensor_offset(gctx, index);
        const size_t bytes = gguf_get_tensor_size(gctx, index);
        if (!tensor || data_offset > file_size ||
            offset > file_size - data_offset ||
            bytes > file_size - data_offset - offset) {
            error = "Qwen4Exp MTP tensor exceeds companion file";
            return fail();
        }
        total = align_up(total, alignment);
        const size_t allocated = ggml_backend_buft_get_alloc_size(buft, tensor);
        if (total == SIZE_MAX || allocated > SIZE_MAX - total) {
            error = "Qwen4Exp MTP dense size overflow";
            return fail();
        }
        allocations.push_back({tensor, data_offset + offset, bytes, total});
        total += allocated;
    }
    out.buf = ggml_backend_alloc_buffer(backend, total);
    if (!out.buf) {
        error = "failed to allocate Qwen4Exp MTP dense weights";
        return fail();
    }
    ggml_backend_buffer_set_usage(out.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const Allocation & allocation : allocations) {
        char * base = static_cast<char *>(ggml_backend_buffer_get_base(out.buf));
        if (ggml_backend_tensor_alloc(out.buf, allocation.tensor,
                                      base + allocation.buffer_offset) !=
            GGML_STATUS_SUCCESS) {
            error = "failed to bind Qwen4Exp MTP tensor buffer";
            return fail();
        }
        ggml_backend_tensor_set(allocation.tensor,
            static_cast<const uint8_t *>(mapping) + allocation.file_offset,
            0, allocation.bytes);
    }
    out.resident_weight_bytes = static_cast<uint64_t>(total);
    out.pre_embedding_norm = ggml_get_tensor(meta, "mtp_pre_emb_norm.weight");
    out.pre_hc_norm = ggml_get_tensor(meta, "mtp_pre_hc_norm.weight");
    out.fc_embedding = ggml_get_tensor(meta, "mtp_fc_emb.weight");
    out.fc_hc = ggml_get_tensor(meta, "mtp_fc_hc.weight");
    out.output_hc_norm = ggml_get_tensor(meta, "mtp_hc_norm.weight");
    out.output_hc_down = ggml_get_tensor(meta, "mtp_hc_down.weight");
    out.output_hc_up = ggml_get_tensor(meta, "mtp_hc_up.weight");
    for (int64_t index = 0; index < count; ++index) {
        const char * name = gguf_get_tensor_name(gctx, index);
        bind_layer(out.layer, name, ggml_get_tensor(meta, name));
    }
    if (!out.layer.experts_gate_up_tensor ||
        !out.layer.experts_down_tensor) {
        error = "Qwen4Exp MTP backend expert tensors were not bound";
        return fail();
    }
    // Every companion tensor now lives in the backend buffer. Unlike the
    // target GGUF, there is no borrowed embedding payload, so retain neither
    // an mmap nor clean file-cache pages as a second expert-weight copy.
    if (madvise(mapping, file_size, MADV_DONTNEED) != 0) {
        error = "madvise(DONTNEED) failed after Qwen4Exp MTP upload: " +
                std::string(std::strerror(errno));
        return fail();
    }
    const int advice = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (advice != 0) {
        error = "posix_fadvise(DONTNEED) failed after Qwen4Exp MTP upload: " +
                std::string(std::strerror(advice));
        return fail();
    }
    munmap(mapping, file_size);
    ::close(fd);
    out.shards[0].mmap_addr = nullptr;
    out.shards[0].mmap_size = 0;
    out.shards[0].mmap_fd = -1;
    gguf_free(gctx);
    return true;
}

void free_qwen4exp_mtp_weights(Qwen4ExpMtpWeights & weights) {
    qwen4exp_frontier_mtp_destroy(weights);
    if (weights.buf) ggml_backend_buffer_free(weights.buf);
    for (Qwen4ExpWeightShard & shard : weights.shards) {
        if (shard.ctx) ggml_free(shard.ctx);
        if (shard.mmap_addr && shard.mmap_size)
            munmap(shard.mmap_addr, shard.mmap_size);
        if (shard.mmap_fd >= 0) ::close(shard.mmap_fd);
    }
    weights = {};
}

} // namespace dflash::common
