#include "qwen4exp_internal.h"
#include "qwen4exp_frontier.h"
#include "qwen4exp_model.h"

#include "gguf.h"

#include <cerrno>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace dflash::common {
namespace {

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0 || value > SIZE_MAX - (alignment - 1)) return SIZE_MAX;
    return (value + alignment - 1) / alignment * alignment;
}

bool evict_dense_source_pages(void * mapping, size_t mapping_size, int fd,
                              size_t file_offset, size_t bytes,
                              std::string & error) {
    if (!mapping || fd < 0 || bytes == 0 || file_offset > mapping_size ||
        bytes > mapping_size - file_offset) {
        error = "invalid Qwen4Exp dense-source eviction range";
        return false;
    }
    const long page_value = sysconf(_SC_PAGESIZE);
    if (page_value <= 0) {
        error = "failed to query page size for Qwen4Exp load eviction";
        return false;
    }
    const size_t page = static_cast<size_t>(page_value);
    const size_t begin = file_offset / page * page;
    const size_t raw_end = file_offset + bytes;
    size_t end = align_up(raw_end, page);
    if (end == SIZE_MAX || end > mapping_size) end = mapping_size;

    // ggml_backend_tensor_set is synchronous for both the managed gfx1151
    // buffer (memcpy) and ordinary HIP buffer (copy + stream sync). Drop the
    // clean source pages immediately so loading does not transiently retain a
    // second dense-weight copy in the file cache on 128-GiB UMA.
    if (madvise(static_cast<uint8_t *>(mapping) + begin, end - begin,
                MADV_DONTNEED) != 0) {
        error = "madvise(DONTNEED) failed after Qwen4Exp weight upload: " +
                std::string(std::strerror(errno));
        return false;
    }
    const int advice = posix_fadvise(fd, static_cast<off_t>(file_offset),
                                     static_cast<off_t>(bytes),
                                     POSIX_FADV_DONTNEED);
    if (advice != 0) {
        error = "posix_fadvise(DONTNEED) failed after Qwen4Exp weight upload: " +
                std::string(std::strerror(advice));
        return false;
    }
    return true;
}

bool is_mapped_weight(const char * name) {
    if (std::strcmp(name, "token_embd.weight") == 0 ||
        std::strcmp(name, "per_layer_token_embd.weight") == 0) return true;
    return false;
}

bool parse_layer_suffix(const char * name, int & layer, std::string & suffix) {
    if (!name || std::strncmp(name, "blk.", 4) != 0) return false;
    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(name + 4, &end, 10);
    if (errno || end == name + 4 || !end || *end != '.' || value < 0 ||
        value >= 48) return false;
    layer = static_cast<int>(value);
    suffix.assign(end + 1);
    return true;
}

bool is_text_weight(const char * name) {
    if (!name) return false;
    if (std::strcmp(name, "token_embd.weight") == 0 ||
        std::strcmp(name, "output.weight") == 0 ||
        std::strcmp(name, "output_hc_norm.weight") == 0 ||
        std::strcmp(name, "output_hc_down.weight") == 0 ||
        std::strcmp(name, "output_hc_up.weight") == 0 ||
        std::strcmp(name, "per_layer_token_embd.weight") == 0) return true;
    int layer = -1;
    std::string suffix;
    return parse_layer_suffix(name, layer, suffix);
}

Qwen4ExpMappedTensor mapped_tensor(const uint8_t * base, size_t file_size,
                                  size_t data_offset, gguf_context * gctx,
                                  ggml_context * meta, int64_t id,
                                  std::string & error) {
    Qwen4ExpMappedTensor result;
    const size_t offset = gguf_get_tensor_offset(gctx, id);
    const size_t bytes = gguf_get_tensor_size(gctx, id);
    if (data_offset > file_size || offset > file_size - data_offset ||
        bytes > file_size - data_offset - offset) {
        error = "Qwen4Exp tensor extends past end of GGUF: " +
                std::string(gguf_get_tensor_name(gctx, id));
        return result;
    }
    result.data = base + data_offset + offset;
    result.bytes = bytes;
    ggml_tensor * tensor = ggml_get_tensor(meta, gguf_get_tensor_name(gctx, id));
    if (!tensor) {
        error = "Qwen4Exp mapped tensor metadata is absent";
        return {};
    }
    result.type = tensor->type;
    result.n_dims = ggml_n_dims(tensor);
    for (int i = 0; i < result.n_dims && i < 4; ++i) {
        result.ne[static_cast<size_t>(i)] = tensor->ne[i];
    }
    return result;
}

ggml_tensor * find(ggml_context * ctx, const char * name) {
    return ggml_get_tensor(ctx, name);
}

void bind_layer_tensor(Qwen4ExpLayer & layer, const std::string & suffix,
                       ggml_tensor * tensor) {
    if (suffix == "hc_attn_norm.weight") layer.hc_attn_norm = tensor;
    else if (suffix == "hc_attn_down.weight") layer.hc_attn_down = tensor;
    else if (suffix == "hc_attn_up.weight") layer.hc_attn_up = tensor;
    else if (suffix == "hc_attn_inject.weight") layer.hc_attn_inject = tensor;
    else if (suffix == "hc_ffn_norm.weight") layer.hc_ffn_norm = tensor;
    else if (suffix == "hc_ffn_down.weight") layer.hc_ffn_down = tensor;
    else if (suffix == "hc_ffn_up.weight") layer.hc_ffn_up = tensor;
    else if (suffix == "hc_ffn_inject.weight") layer.hc_ffn_inject = tensor;
    else if (suffix == "attn_q.weight") layer.attn_q = tensor;
    else if (suffix == "attn_k.weight") layer.attn_k = tensor;
    else if (suffix == "attn_v.weight") layer.attn_v = tensor;
    else if (suffix == "attn_output.weight") layer.attn_output = tensor;
    else if (suffix == "attn_q_norm.weight") layer.attn_q_norm = tensor;
    else if (suffix == "attn_k_norm.weight") layer.attn_k_norm = tensor;
    else if (suffix == "indexer.q_proj.weight") layer.index_q = tensor;
    else if (suffix == "indexer.k_proj.weight") layer.index_k = tensor;
    else if (suffix == "indexer.q_norm.weight") layer.index_q_norm = tensor;
    else if (suffix == "indexer.k_norm.weight") layer.index_k_norm = tensor;
    else if (suffix == "attn_k_rot.weight") layer.self_k_rot = tensor;
    else if (suffix == "attn_v_rot.weight") layer.self_v_rot = tensor;
    else if (suffix == "attn_qkv.weight") layer.attn_qkv = tensor;
    else if (suffix == "attn_gate.weight") layer.attn_gate = tensor;
    else if (suffix == "ssm_conv1d.weight") layer.ssm_conv = tensor;
    else if (suffix == "ssm_a") layer.ssm_a = tensor;
    else if (suffix == "ssm_alpha.weight") layer.ssm_alpha = tensor;
    else if (suffix == "ssm_beta.weight") layer.ssm_beta = tensor;
    else if (suffix == "ssm_dt.bias") layer.ssm_dt = tensor;
    else if (suffix == "ssm_norm.weight") layer.ssm_norm = tensor;
    else if (suffix == "ssm_out.weight") layer.ssm_out = tensor;
    else if (suffix == "ple_key.weight") layer.ple_key = tensor;
    else if (suffix == "ple_value.weight") layer.ple_value = tensor;
    else if (suffix == "ple_norm_key.weight") layer.ple_norm_key = tensor;
    else if (suffix == "ple_norm_query.weight") layer.ple_norm_query = tensor;
    else if (suffix == "ple_norm_conv.weight") layer.ple_norm_conv = tensor;
    else if (suffix == "ple_conv1d.weight") layer.ple_conv = tensor;
    else if (suffix == "ffn_gate_inp.weight") layer.router = tensor;
    else if (suffix == "ffn_gate_inp_shexp.weight") layer.shared_gate_input = tensor;
    else if (suffix == "ffn_gate_shexp.weight") layer.shared_gate = tensor;
    else if (suffix == "ffn_up_shexp.weight") layer.shared_up = tensor;
    else if (suffix == "ffn_down_shexp.weight") layer.shared_down = tensor;
}

} // namespace

bool qwen4exp_mapped_row_f32(const Qwen4ExpMappedTensor & tensor,
                            int64_t row, float * out, size_t out_count,
                            std::string * error) {
    if (!tensor.valid() || tensor.n_dims < 2 || row < 0 ||
        row >= tensor.ne[1] || !out ||
        out_count != static_cast<size_t>(tensor.ne[0])) {
        if (error) *error = "invalid mapped Qwen4Exp row request";
        return false;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(tensor.type);
    if (!traits || !traits->to_float) {
        if (error) *error = "mapped Qwen4Exp tensor type cannot dequantize";
        return false;
    }
    const size_t row_bytes = ggml_row_size(tensor.type, tensor.ne[0]);
    const size_t offset = static_cast<size_t>(row) * row_bytes;
    if (offset > tensor.bytes || row_bytes > tensor.bytes - offset) {
        if (error) *error = "mapped Qwen4Exp row is out of bounds";
        return false;
    }
    traits->to_float(tensor.data + offset, out, tensor.ne[0]);
    return true;
}

bool load_qwen4exp_gguf(const std::string & path, ggml_backend_t backend,
                        int max_ctx, bool enable_yarn, Qwen4ExpWeights & out,
                        std::string & error) {
    error.clear();
    if (!backend) {
        error = "invalid Qwen4Exp backend";
        return false;
    }
    char yarn_error[192];
    if (!ember_qwen_yarn_configure(enable_yarn, max_ctx, &out.yarn,
                                   yarn_error, sizeof(yarn_error))) {
        error = yarn_error;
        return false;
    }
    std::vector<std::string> paths;
    if (!qwen4exp_discover_gguf_shards(path.c_str(), paths, error) ||
        !validate_qwen4exp_gguf(path.c_str(), error)) return false;

    struct Allocation {
        ggml_tensor * tensor;
        size_t shard;
        size_t file_offset;
        size_t bytes;
        size_t buffer_offset;
    };
    std::vector<Allocation> allocations;
    std::vector<gguf_context *> contexts(paths.size(), nullptr);
    std::unordered_map<std::string, ggml_tensor *> tensors;
    const auto lookup = [&tensors](const char * name) -> ggml_tensor * {
        const auto found = tensors.find(name);
        return found == tensors.end() ? nullptr : found->second;
    };
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t total = 0;
    uint64_t mapped_total = 0;
    out.backend = backend;
    out.shards.reserve(paths.size());

    for (size_t shard_index = 0; shard_index < paths.size(); ++shard_index) {
        ggml_context * meta = nullptr;
        gguf_init_params params{/*.no_alloc=*/true, /*.ctx=*/&meta};
        gguf_context * gctx = gguf_init_from_file(paths[shard_index].c_str(), params);
        if (!gctx || !meta) {
            if (gctx) gguf_free(gctx);
            if (meta) ggml_free(meta);
            error = "failed to reopen Qwen4Exp tensor metadata shard: " +
                    paths[shard_index];
            goto fail;
        }
        contexts[shard_index] = gctx;
        const int fd = ::open(paths[shard_index].c_str(), O_RDONLY | O_CLOEXEC);
        struct stat status{};
        if (fd < 0 || fstat(fd, &status) != 0 || status.st_size <= 0 ||
            static_cast<uintmax_t>(status.st_size) > SIZE_MAX) {
            if (fd >= 0) ::close(fd);
            ggml_free(meta);
            error = "failed to stat Qwen4Exp GGUF shard: " +
                    std::string(std::strerror(errno));
            goto fail;
        }
        const size_t file_size = static_cast<size_t>(status.st_size);
        void * mapping = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapping == MAP_FAILED) {
            ::close(fd);
            ggml_free(meta);
            error = "failed to mmap Qwen4Exp GGUF shard: " +
                    std::string(std::strerror(errno));
            goto fail;
        }
        out.shards.push_back({meta, mapping, file_size, fd, paths[shard_index]});

        const int64_t count = gguf_get_n_tensors(gctx);
        const size_t data_offset = gguf_get_data_offset(gctx);
        for (int64_t i = 0; i < count; ++i) {
            const char * name = gguf_get_tensor_name(gctx, i);
            ggml_tensor * tensor = name ? find(meta, name) : nullptr;
            if (!name || !tensor || !tensors.emplace(name, tensor).second) {
                error = "duplicate or missing Qwen4Exp tensor metadata: " +
                        std::string(name ? name : "(null)");
                goto fail;
            }
            // The official repository also describes vision and MTP stacks.
            // The text loader leaves those companion tensors untouched.
            if (!is_text_weight(name)) continue;
            const bool vector = ggml_n_dims(tensor) == 1;
            if (!qwen4exp_weight_type_supported(tensor->type, vector)) {
                error = "unsupported Qwen4Exp tensor type " +
                        std::string(ggml_type_name(tensor->type)) + ": " + name;
                goto fail;
            }
            const size_t tensor_offset = gguf_get_tensor_offset(gctx, i);
            const size_t bytes = gguf_get_tensor_size(gctx, i);
            if (data_offset > file_size || tensor_offset > file_size - data_offset ||
                bytes > file_size - data_offset - tensor_offset) {
                error = "Qwen4Exp GGUF tensor is out of file bounds: " +
                        std::string(name);
                goto fail;
            }
            if (is_mapped_weight(name)) {
                if (bytes > UINT64_MAX - mapped_total) {
                    error = "Qwen4Exp mapped weight size overflow";
                    goto fail;
                }
                mapped_total += static_cast<uint64_t>(bytes);
                continue;
            }
            total = align_up(total, alignment);
            const size_t allocated = ggml_backend_buft_get_alloc_size(buft, tensor);
            if (total == SIZE_MAX || allocated > SIZE_MAX - total) {
                error = "Qwen4Exp dense weight allocation size overflow";
                goto fail;
            }
            allocations.push_back(
                {tensor, shard_index, data_offset + tensor_offset, bytes, total});
            total += allocated;
        }
    }

    {
        const uint64_t dense_total = static_cast<uint64_t>(total);
        const uint64_t resident = mapped_total > UINT64_MAX - dense_total
            ? UINT64_MAX : mapped_total + dense_total;
        const Qwen4ExpMemoryPlan plan =
            qwen4exp_memory_plan(resident, max_ctx);
        if (!plan.fits) {
            error = "Qwen4Exp 128-GiB UMA plan exceeds capacity: weights=" +
                    std::to_string(plan.resident_weight_bytes) +
                    " qsa=" + std::to_string(plan.qsa_cache_bytes) +
                    " recurrent=" +
                    std::to_string(plan.recurrent_state_bytes) +
                    " reserve=" +
                    std::to_string(plan.runtime_reserve_bytes) +
                    " total=" + std::to_string(plan.total_bytes);
            goto fail;
        }
        out.resident_weight_bytes = resident;
        out.state_budget_bytes = plan.capacity_bytes - resident -
                                 plan.runtime_reserve_bytes;
    }

    out.buf = ggml_backend_alloc_buffer(backend, total);
    if (!out.buf) {
        error = "failed to allocate Qwen4Exp dense weight buffer (" +
                std::to_string(total) + " bytes)";
        goto fail;
    }
    ggml_backend_buffer_set_usage(out.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const Allocation & allocation : allocations) {
        Qwen4ExpWeightShard & shard = out.shards[allocation.shard];
        char * base = static_cast<char *>(ggml_backend_buffer_get_base(out.buf));
        if (ggml_backend_tensor_alloc(out.buf, allocation.tensor,
                                      base + allocation.buffer_offset) !=
            GGML_STATUS_SUCCESS) {
            error = "failed to bind Qwen4Exp dense tensor buffer";
            goto fail;
        }
        ggml_backend_tensor_set(allocation.tensor,
                                static_cast<const uint8_t *>(shard.mmap_addr) +
                                    allocation.file_offset,
                                0, allocation.bytes);
        if (!evict_dense_source_pages(shard.mmap_addr, shard.mmap_size,
                                      shard.mmap_fd,
                                      allocation.file_offset,
                                      allocation.bytes, error)) goto fail;
    }

    out.max_ctx = max_ctx;
    out.layers.resize(48);
    out.output = lookup("output.weight");
    out.output_hc_norm = lookup("output_hc_norm.weight");
    out.output_hc_down = lookup("output_hc_down.weight");
    out.output_hc_up = lookup("output_hc_up.weight");

    for (size_t shard_index = 0; shard_index < contexts.size(); ++shard_index) {
        gguf_context * gctx = contexts[shard_index];
        ggml_context * meta = out.shards[shard_index].ctx;
        const size_t data_offset = gguf_get_data_offset(gctx);
        const int64_t count = gguf_get_n_tensors(gctx);
        for (int64_t i = 0; i < count; ++i) {
            const char * name = gguf_get_tensor_name(gctx, i);
            std::string suffix;
            int layer_index = -1;
            Qwen4ExpWeightShard & shard = out.shards[shard_index];
            if (std::strcmp(name, "token_embd.weight") == 0) {
                const Qwen4ExpMappedTensor mapped = mapped_tensor(
                    static_cast<const uint8_t *>(shard.mmap_addr), shard.mmap_size,
                    data_offset, gctx, meta, i, error);
                if (!mapped.valid()) goto fail;
                out.embedder.tok_embd_bytes = mapped.data;
                out.embedder.tok_embd_type = mapped.type;
                out.embedder.n_embd = 2560;
                out.embedder.n_vocab = 248320;
                out.embedder.row_bytes = ggml_row_size(mapped.type, 2560);
            } else if (std::strcmp(name, "per_layer_token_embd.weight") == 0) {
                out.ple_table = mapped_tensor(
                    static_cast<const uint8_t *>(shard.mmap_addr), shard.mmap_size,
                    data_offset, gctx, meta, i, error);
                if (!out.ple_table.valid()) goto fail;
            } else if (parse_layer_suffix(name, layer_index, suffix)) {
                Qwen4ExpLayer & layer = out.layers[static_cast<size_t>(layer_index)];
                if (suffix == "ffn_gate_up_exps.weight") {
                    layer.experts_gate_up_tensor = find(meta, name);
                    layer.experts_gate_up = mapped_tensor(
                        static_cast<const uint8_t *>(shard.mmap_addr), shard.mmap_size,
                        data_offset, gctx, meta, i, error);
                    if (!layer.experts_gate_up.valid()) goto fail;
                } else if (suffix == "ffn_gate_exps.weight") {
                    layer.experts_gate_tensor = find(meta, name);
                    layer.experts_gate = mapped_tensor(
                        static_cast<const uint8_t *>(shard.mmap_addr), shard.mmap_size,
                        data_offset, gctx, meta, i, error);
                    if (!layer.experts_gate.valid()) goto fail;
                } else if (suffix == "ffn_up_exps.weight") {
                    layer.experts_up_tensor = find(meta, name);
                    layer.experts_up = mapped_tensor(
                        static_cast<const uint8_t *>(shard.mmap_addr), shard.mmap_size,
                        data_offset, gctx, meta, i, error);
                    if (!layer.experts_up.valid()) goto fail;
                } else if (suffix == "ffn_down_exps.weight") {
                    layer.experts_down_tensor = find(meta, name);
                    layer.experts_down = mapped_tensor(
                        static_cast<const uint8_t *>(shard.mmap_addr), shard.mmap_size,
                        data_offset, gctx, meta, i, error);
                    if (!layer.experts_down.valid()) goto fail;
                } else {
                    bind_layer_tensor(layer, suffix, find(meta, name));
                }
            }
        }
    }

    if (!out.embedder.tok_embd_bytes || !out.ple_table.valid()) {
        error = "Qwen4Exp mapped embedding tensors were not bound";
        goto fail;
    }
    if (!qwen4exp_frontier_create(out, error)) goto fail;
    for (gguf_context * context : contexts) gguf_free(context);
    return true;

fail:
    for (gguf_context * context : contexts) {
        if (context) gguf_free(context);
    }
    free_qwen4exp_weights(out);
    return false;
}

void free_qwen4exp_weights(Qwen4ExpWeights & weights) {
    qwen4exp_frontier_destroy(weights);
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
