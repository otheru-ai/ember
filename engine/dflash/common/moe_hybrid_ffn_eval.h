// Common MoE hybrid FFN evaluation — hot experts on GPU, cold on CPU, concurrent.

#pragma once

#include "moe_hybrid_types.h"
#include "moe_hybrid_storage.h"

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct MoeHybridFfnTelemetry {
    uint64_t ffn_wall_us = 0;
    uint64_t partition_us = 0;
    uint64_t hot_us = 0;
    uint64_t cold_us = 0;
    uint64_t shared_us = 0;
    uint64_t combine_us = 0;
    uint64_t hot_graph_build_us = 0;
    uint64_t hot_input_us = 0;
    uint64_t hot_compute_us = 0;
    uint64_t hot_read_us = 0;
    uint64_t cold_graph_build_us = 0;
    uint64_t cold_input_us = 0;
    uint64_t cold_compute_us = 0;
    uint64_t cold_read_us = 0;
    uint64_t hot_graph_builds = 0;
    uint64_t hot_graph_hits = 0;
    uint64_t cold_graph_builds = 0;
    uint64_t cold_graph_hits = 0;
    int hot_selected = 0;
    int cold_selected = 0;
};

int moe_hybrid_prefill_hot_sub_batch_limit();

// Single-token hybrid FFN: hot on GPU, cold on CPU, combine on host.
bool eval_moe_hybrid_ffn_single(
    ggml_backend_t                  gpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    ggml_backend_t                  cpu_backend,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_selected,
    std::vector<float> &            out,
    MoeHybridFfnTelemetry *         telemetry = nullptr,
    std::string *                   err = nullptr);

// Batched hybrid prefill FFN: hot on GPU, cold on CPU concurrently.
bool eval_moe_hybrid_ffn_batched(
    ggml_backend_t                  gpu_backend,
    ggml_backend_t                  cpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err = nullptr,
    ggml_gallocr_t *                p_hot_alloc = nullptr,
    ggml_gallocr_t *                p_cold_alloc = nullptr,
    MoeHybridFfnTelemetry *         telemetry = nullptr);

// Hot-only batched prefill: all selected experts are in VRAM.
// Skips cold graph build, CPU compute, and merge — pure GPU path.
bool eval_moe_hot_only_batched(
    ggml_backend_t                  gpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err = nullptr,
    ggml_gallocr_t *                p_hot_alloc = nullptr);

struct CachedHotGraphOptions {
    float swiglu_clamp = 0.0f;
};

// Build/rebuild cached hot FFN graph.
bool build_cached_hot_graph(
    CachedFfnGraph & out,
    ggml_backend_t backend,
    ggml_tensor * gate_tensor,
    ggml_tensor * up_tensor,
    ggml_tensor * down_tensor,
    ggml_tensor * gate_up_tensor,
    float gate_scale,
    float up_scale,
    float down_scale,
    float gate_up_scale,
    const MoeLayerDesc & desc,
    int n_embd,
    int n_ff_exp,
    int n_hot,
    CachedHotGraphOptions options = {});

// Build/rebuild cached MoE expert compute graph.
bool build_cached_cold_graph(
    CachedFfnGraph & out,
    ggml_backend_t cpu_backend,
    ggml_tensor * gate_tensor,
    ggml_tensor * up_tensor,
    ggml_tensor * down_tensor,
    ggml_tensor * gate_up_tensor,
    float gate_scale,
    float up_scale,
    float down_scale,
    float gate_up_scale,
    int n_embd,
    int n_ff_exp,
    int n_cold,
    float swiglu_clamp = 0.0f);

// Build cached hot-only batched graph for prefill (n_tokens=MMQ_SAFE_SUB_BATCH).
bool build_cached_hot_batched_graph(
    CachedHotBatchedGraph & out,
    ggml_backend_t gpu_backend,
    MoeHybridLayerStorage & storage,
    const MoeLayerDesc & desc,
    const MoeHybridConfig & cfg,
    int n_tokens);

}  // namespace dflash::common
