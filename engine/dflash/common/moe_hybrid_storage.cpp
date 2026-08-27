#include "moe_hybrid_storage.h"
#include "moe_hybrid_types.h"

#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include <sys/mman.h>

namespace dflash::common {

void CachedFfnGraph::free() {
    if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    gf = nullptr;
    inp = nullptr;
    ids = nullptr;
    weights = nullptr;
    output = nullptr;
    n_hot = 0;
    n_tokens = 1;
}

void CachedHotBatchedGraph::free() {
    if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    gf = nullptr;
    inp = nullptr;
    sel = nullptr;
    wts = nullptr;
    output = nullptr;
    n_tokens = 0;
}

namespace {

static bool read_expert_slices_from_mem(const uint8_t * tensor_data,
                                        size_t tensor_size,
                                        const std::vector<int32_t> & expert_ids,
                                        size_t expert_bytes,
                                        std::vector<uint8_t> & out,
                                        std::string * err) {
    if (expert_ids.empty()) {
        out.clear();
        return true;
    }
    if (!tensor_data || expert_bytes == 0) {
        if (err) *err = "missing expert tensor file data";
        return false;
    }
    if (expert_ids.size() >
        std::numeric_limits<size_t>::max() / expert_bytes) {
        if (err) *err = "expert slice allocation overflow";
        return false;
    }
    out.resize(expert_bytes * expert_ids.size());
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        const int32_t expert_id = expert_ids[i];
        if (expert_id < 0 ||
            static_cast<size_t>(expert_id) >
                std::numeric_limits<size_t>::max() / expert_bytes) {
            if (err) *err = "expert id or slice offset out of range";
            return false;
        }
        const size_t offset = expert_bytes * (size_t)expert_id;
        if (offset > tensor_size || expert_bytes > tensor_size - offset) {
            if (err) *err = "expert slice out of bounds in file";
            return false;
        }
        std::memcpy(out.data() + expert_bytes * i, tensor_data + offset, expert_bytes);
    }
    return true;
}

static bool validate_expert_tensor(ggml_tensor * tensor, int n_expert, size_t * expert_bytes, std::string * err) {
    if (!tensor) {
        *expert_bytes = 0;
        return true;
    }
    if (tensor->ne[2] != n_expert) {
        if (err) *err = "tensor expert dimension mismatch";
        return false;
    }
    if ((int64_t)tensor->nb[2] <= 0) {
        if (err) *err = "tensor expert stride invalid";
        return false;
    }
    *expert_bytes = (size_t)tensor->nb[2];
    if (static_cast<size_t>(n_expert) >
            std::numeric_limits<size_t>::max() / *expert_bytes ||
        static_cast<size_t>(n_expert) * *expert_bytes > ggml_nbytes(tensor)) {
        if (err) *err = "tensor expert storage is truncated";
        return false;
    }
    return true;
}

static ggml_tensor * new_like_with_expert_count(ggml_context * ctx, ggml_tensor * src, int hot_count) {
    if (!src || hot_count <= 0) return nullptr;
    const int64_t ne[4] = { src->ne[0], src->ne[1], hot_count, 1 };
    return ggml_new_tensor(ctx, src->type, 4, ne);
}

} // namespace

MoeHybridStorage::~MoeHybridStorage() {
    for (auto & layer : layers) {
        layer.hot_graph.free();
        layer.cold_graph.free();
        layer.hot_batched_graph.free();
        for (auto & g : layer.hot_batched_mixed) g.free();
        for (auto & g : layer.cold_batched_mixed) g.free();
        layer.shared_batched_graph.free();
        if (layer.hot_buf) {
            ggml_backend_buffer_free(layer.hot_buf);
            layer.hot_buf = nullptr;
        }
        if (layer.hot_ctx) {
            ggml_free(layer.hot_ctx);
            layer.hot_ctx = nullptr;
        }
        if (layer.cold_buf) {
            ggml_backend_buffer_free(layer.cold_buf);
            layer.cold_buf = nullptr;
        }
        if (layer.cold_ctx) {
            ggml_free(layer.cold_ctx);
            layer.cold_ctx = nullptr;
        }
        layer.gate_hot = nullptr;
        layer.up_hot = nullptr;
        layer.down_hot = nullptr;
        layer.gate_up_hot = nullptr;
        layer.gate_cold = nullptr;
        layer.up_cold = nullptr;
        layer.down_cold = nullptr;
        layer.gate_up_cold = nullptr;
    }
    if (cpu_backend) {
        ggml_backend_free(cpu_backend);
        cpu_backend = nullptr;
    }
    if (mmap_data) {
        ::munmap(const_cast<void *>(mmap_data), mmap_size);
        mmap_data = nullptr;
        mmap_size = 0;
    }
}

bool MoeHybridStorage::matches(const MoeHybridConfig & cfg) const {
    return placement.matches(cfg) &&
           (int)layers.size() == cfg.n_layer &&
           cold_backend_kind == cfg.cold_expert_backend &&
           materialized_cold_experts == cfg.materialize_cold_experts;
}

bool MoeHybridStorage::empty() const {
    return layers.empty();
}

bool build_moe_hybrid_storage_from_file(
    const MoeHybridConfig & cfg,
    ggml_backend_t gpu_backend,
    const MoeHybridPlacement & placement,
    const std::vector<MoeLayerDesc> & layer_descs,
    const std::vector<LayerExpertFileData> & file_data,
    MoeHybridStorage & out,
    std::string * err) {

    if (!placement.matches(cfg)) {
        if (err) *err = "placement does not match config";
        return false;
    }
    if ((int)layer_descs.size() != cfg.n_layer || (int)file_data.size() != cfg.n_layer) {
        if (err) *err = "layer_descs/file_data size does not match n_layer";
        return false;
    }

    out.placement = placement;
    out.layers.resize((size_t)cfg.n_layer);
    out.cpu_backend = ggml_backend_cpu_init();
    if (!out.cpu_backend) {
        if (err) *err = "failed to init cpu backend";
        return false;
    }
    ggml_backend_cpu_set_n_threads(out.cpu_backend, std::max(1, std::min(cfg.n_expert_used, 8)));
    out.cold_backend_kind = cfg.cold_expert_backend;
    out.materialized_cold_experts = cfg.materialize_cold_experts;
    out.cold_backend = (cfg.cold_expert_backend == MoeHybridColdBackend::Gpu) ? gpu_backend : out.cpu_backend;
    if (!out.cold_backend) {
        if (err) *err = "failed to select cold expert backend";
        return false;
    }

    for (int il = 0; il < cfg.n_layer; ++il) {
        const MoeLayerDesc & desc = layer_descs[(size_t)il];
        const LayerExpertFileData & fd = file_data[(size_t)il];
        MoeHybridLayerStorage & dst = out.layers[(size_t)il];
        dst.cold_backend = out.cold_backend;
        dst.cold_backend_kind = out.cold_backend_kind;

        // Skip dense layers (no experts)
        if (!desc.ffn_gate_exps && !desc.ffn_up_exps && !desc.ffn_down_exps && !desc.ffn_gate_up_exps) {
            continue;
        }

        dst.hot_expert_ids = placement.hot_expert_ids[(size_t)il];
        dst.hot_local_by_global.assign((size_t)cfg.n_expert, -1);
        dst.cold_local_by_global.assign((size_t)cfg.n_expert, -1);

        std::vector<uint8_t> is_hot((size_t)cfg.n_expert, 0);
        for (size_t i = 0; i < dst.hot_expert_ids.size(); ++i) {
            const int32_t expert = dst.hot_expert_ids[i];
            if (expert < 0 || expert >= cfg.n_expert) {
                if (err) *err = "hot expert id out of range";
                return false;
            }
            dst.hot_local_by_global[(size_t)expert] = (int32_t)i;
            is_hot[(size_t)expert] = 1;
        }
        for (int expert = 0; expert < cfg.n_expert; ++expert) {
            if (!is_hot[(size_t)expert]) {
                dst.cold_local_by_global[(size_t)expert] = (int32_t)dst.cold_expert_ids.size();
                dst.cold_expert_ids.push_back((int32_t)expert);
            }
        }

        // Populate VRAM bitmask from hot expert IDs
        std::memset(dst.expert_vram_mask, 0, sizeof(dst.expert_vram_mask));
        for (int32_t eid : dst.hot_expert_ids) {
            if (eid >= 0 && eid < 256)
                dst.expert_vram_mask[eid >> 6] |= (1ULL << (eid & 63));
        }

        dst.fused_gate_up = desc.has_fused_gate_up();
        if (!validate_expert_tensor(desc.ffn_gate_exps, cfg.n_expert, &dst.gate_expert_bytes, err) ||
            !validate_expert_tensor(desc.ffn_up_exps, cfg.n_expert, &dst.up_expert_bytes, err) ||
            !validate_expert_tensor(desc.ffn_down_exps, cfg.n_expert, &dst.down_expert_bytes, err) ||
            !validate_expert_tensor(desc.ffn_gate_up_exps, cfg.n_expert, &dst.gate_up_expert_bytes, err)) {
            return false;
        }

        const int hot_count = (int)dst.hot_expert_ids.size();
        const int cold_count = (int)dst.cold_expert_ids.size();
        const int hot_alloc = hot_count;

        // Allocate hot expert tensors on GPU
        if (hot_count > 0) {
            ggml_init_params ip{};
            ip.mem_size   = 16 * ggml_tensor_overhead();
            ip.mem_buffer = nullptr;
            ip.no_alloc   = true;
            dst.hot_ctx = ggml_init(ip);
            if (!dst.hot_ctx) {
                if (err) *err = "failed to init hot_ctx";
                return false;
            }
            if (dst.fused_gate_up) {
                dst.gate_up_hot = new_like_with_expert_count(dst.hot_ctx, desc.ffn_gate_up_exps, hot_alloc);
                dst.down_hot    = new_like_with_expert_count(dst.hot_ctx, desc.ffn_down_exps, hot_alloc);
            } else {
                dst.gate_hot = new_like_with_expert_count(dst.hot_ctx, desc.ffn_gate_exps, hot_alloc);
                dst.up_hot   = new_like_with_expert_count(dst.hot_ctx, desc.ffn_up_exps, hot_alloc);
                dst.down_hot = new_like_with_expert_count(dst.hot_ctx, desc.ffn_down_exps, hot_alloc);
            }
            dst.hot_buf = ggml_backend_alloc_ctx_tensors(dst.hot_ctx, gpu_backend);
            if (!dst.hot_buf) {
                char msg[128];
                std::snprintf(msg, sizeof(msg),
                    "failed to allocate hot expert GPU buffer (layer %d, %d hot experts)", il, hot_count);
                if (err) *err = msg;
                return false;
            }

            std::vector<uint8_t> slice_buf;
            if (dst.fused_gate_up) {
                if (!read_expert_slices_from_mem(fd.gate_up_exps.data, fd.gate_up_exps.size,
                                                 dst.hot_expert_ids, dst.gate_up_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.gate_up_hot, slice_buf.data(), 0, slice_buf.size());
                if (!read_expert_slices_from_mem(fd.down_exps.data, fd.down_exps.size,
                                                 dst.hot_expert_ids, dst.down_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.down_hot, slice_buf.data(), 0, slice_buf.size());
            } else {
                if (!read_expert_slices_from_mem(fd.gate_exps.data, fd.gate_exps.size,
                                                 dst.hot_expert_ids, dst.gate_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.gate_hot, slice_buf.data(), 0, slice_buf.size());
                if (!read_expert_slices_from_mem(fd.up_exps.data, fd.up_exps.size,
                                                 dst.hot_expert_ids, dst.up_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.up_hot, slice_buf.data(), 0, slice_buf.size());
                if (!read_expert_slices_from_mem(fd.down_exps.data, fd.down_exps.size,
                                                 dst.hot_expert_ids, dst.down_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.down_hot, slice_buf.data(), 0, slice_buf.size());
            }
        }

        // Allocate cold expert tensors on the selected cold backend.
        if (cold_count > 0 && cfg.materialize_cold_experts) {
            ggml_init_params ip{};
            ip.mem_size   = 16 * ggml_tensor_overhead();
            ip.mem_buffer = nullptr;
            ip.no_alloc   = true;
            dst.cold_ctx = ggml_init(ip);
            if (!dst.cold_ctx) {
                if (err) *err = "failed to init cold_ctx";
                return false;
            }
            if (dst.fused_gate_up) {
                dst.gate_up_cold = new_like_with_expert_count(dst.cold_ctx, desc.ffn_gate_up_exps, cold_count);
                dst.down_cold    = new_like_with_expert_count(dst.cold_ctx, desc.ffn_down_exps, cold_count);
            } else {
                dst.gate_cold = new_like_with_expert_count(dst.cold_ctx, desc.ffn_gate_exps, cold_count);
                dst.up_cold   = new_like_with_expert_count(dst.cold_ctx, desc.ffn_up_exps, cold_count);
                dst.down_cold = new_like_with_expert_count(dst.cold_ctx, desc.ffn_down_exps, cold_count);
            }
            dst.cold_buf = ggml_backend_alloc_ctx_tensors(dst.cold_ctx, out.cold_backend);
            if (!dst.cold_buf) {
                if (err) {
                    *err = (out.cold_backend_kind == MoeHybridColdBackend::Gpu)
                        ? "failed to allocate cold expert GPU buffer"
                        : "failed to allocate cold expert CPU buffer";
                }
                return false;
            }

            std::vector<uint8_t> slice_buf;
            if (dst.fused_gate_up) {
                if (!read_expert_slices_from_mem(fd.gate_up_exps.data, fd.gate_up_exps.size,
                                                 dst.cold_expert_ids, dst.gate_up_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.gate_up_cold, slice_buf.data(), 0, slice_buf.size());
                if (!read_expert_slices_from_mem(fd.down_exps.data, fd.down_exps.size,
                                                 dst.cold_expert_ids, dst.down_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.down_cold, slice_buf.data(), 0, slice_buf.size());
            } else {
                if (!read_expert_slices_from_mem(fd.gate_exps.data, fd.gate_exps.size,
                                                 dst.cold_expert_ids, dst.gate_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.gate_cold, slice_buf.data(), 0, slice_buf.size());
                if (!read_expert_slices_from_mem(fd.up_exps.data, fd.up_exps.size,
                                                 dst.cold_expert_ids, dst.up_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.up_cold, slice_buf.data(), 0, slice_buf.size());
                if (!read_expert_slices_from_mem(fd.down_exps.data, fd.down_exps.size,
                                                 dst.cold_expert_ids, dst.down_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.down_cold, slice_buf.data(), 0, slice_buf.size());
            }
        }
    }

    return true;
}


bool build_moe_hybrid_storage_from_file_with_mmap(
    const MoeHybridConfig & cfg,
    ggml_backend_t gpu_backend,
    const MoeHybridPlacement & placement,
    const std::vector<MoeLayerDesc> & layer_descs,
    const std::vector<LayerExpertFileData> & file_data,
    const void * mmap_base,
    size_t mmap_total_size,
    MoeHybridStorage & out,
    std::string * err) {

    if (!mmap_base || mmap_total_size == 0) {
        if (err) *err = "invalid expert mmap";
        return false;
    }

    // First build storage normally (hot GPU + cold CPU buffers).
    if (!build_moe_hybrid_storage_from_file(cfg, gpu_backend, placement, layer_descs, file_data, out, err)) {
        return false;
    }

    // Compute per-layer expert file regions (offsets relative to mmap base).
    const auto * base = static_cast<const uint8_t *>(mmap_base);
    const uintptr_t base_addr = reinterpret_cast<uintptr_t>(base);
    if (mmap_total_size >
        std::numeric_limits<uintptr_t>::max() - base_addr) {
        if (err) *err = "expert mmap address range overflows";
        return false;
    }
    const uintptr_t map_end = base_addr + mmap_total_size;
    out.layer_regions.resize((size_t)cfg.n_layer);
    for (int il = 0; il < cfg.n_layer; ++il) {
        const auto & fd = file_data[(size_t)il];
        auto & reg = out.layer_regions[(size_t)il];

        auto set_region = [&](const ExpertTensorFileData & src,
                              ExpertFileRegion & dst) -> bool {
            if (!src.data && src.size == 0) return true;
            if (!src.data || src.size == 0) return false;
            const uintptr_t begin = reinterpret_cast<uintptr_t>(src.data);
            if (begin < base_addr || begin > map_end ||
                src.size > map_end - begin) {
                return false;
            }
            dst.offset = static_cast<size_t>(begin - base_addr);
            dst.size = src.size;
            return true;
        };
        if (!set_region(fd.gate_exps, reg.gate_exps) ||
            !set_region(fd.up_exps, reg.up_exps) ||
            !set_region(fd.down_exps, reg.down_exps) ||
            !set_region(fd.gate_up_exps, reg.gate_up_exps)) {
            if (err) *err = "expert tensor region lies outside mmap";
            return false;
        }

        // Copy per-expert byte sizes from layer storage (already computed)
        const auto & ls = out.layers[(size_t)il];
        reg.expert_bytes_gate    = ls.gate_expert_bytes;
        reg.expert_bytes_up      = ls.up_expert_bytes;
        reg.expert_bytes_down    = ls.down_expert_bytes;
        reg.expert_bytes_gate_up = ls.gate_up_expert_bytes;
        reg.fused_gate_up        = ls.fused_gate_up;
    }

    // Transfer mapping ownership only after every region has validated. The
    // caller retains ownership on any false return.
    out.mmap_data = mmap_base;
    out.mmap_size = mmap_total_size;
    return true;
}

}  // namespace dflash::common
