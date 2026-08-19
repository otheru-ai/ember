// CPU control/reduction half of the heterogeneous DSpark provider.
//
// XDNA2 owns the large trained projections and expert graphs. The CPU keeps
// the small, latency-sensitive operations whose shapes are a poor AIE fit:
// weighted RMSNorm, 4-stream hyper-connections, router top-k, and MLA
// score/softmax/context reduction. These routines are independent of XRT so
// their numerical contract remains GPU-free testable.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ember::xdna2 {

struct DsparkHcSplit {
    std::vector<float> values;  // token-major [pre, post, combine]
    int tokens = 0;
    int n_hc = 0;
};

// Shared persistent worker pool used by CPU control and routed-expert work.
// Calls are serialized at the pool boundary, while tasks within one phase run
// in parallel. This avoids creating 15-30 host threads per draft layer.
void dspark_parallel_for(int count, const std::function<void(int)> & function);

bool dspark_weighted_rms_norm(const float * input,
                              const float * weight,
                              int rows,
                              int width,
                              float epsilon,
                              std::vector<float> & output,
                              std::string * error = nullptr);

bool dspark_hc_pre(const float * state,
                   const float * fn_weight,
                   const float * base,
                   const float scale[3],
                   int tokens,
                   int n_embd,
                   int n_hc,
                   int sinkhorn_iterations,
                   float epsilon,
                   std::vector<float> & working,
                   DsparkHcSplit & split,
                   std::string * error = nullptr);

bool dspark_hc_post(const float * residual,
                    const float * block_output,
                    const DsparkHcSplit & split,
                    int n_embd,
                    std::vector<float> & output,
                    std::string * error = nullptr);

bool dspark_hc_out(const float * state,
                   const float * fn_weight,
                   const float * base,
                   float scale,
                   int tokens,
                   int n_embd,
                   int n_hc,
                   float epsilon,
                   std::vector<float> & output,
                   std::string * error = nullptr);

// Distance between the last selected route and the best rejected route.
// Small positive margins identify the discontinuous top-k boundary where
// otherwise tiny heterogeneous numerical drift can change expert membership.
struct DsparkRouteBoundary {
    float margin = 0.0f;
    int32_t selected_expert = -1;
    int32_t rejected_expert = -1;
};

bool dspark_route_topk(const float * normalized,
                       const float * router_weight,
                       const float * selection_bias,
                       int tokens,
                       int n_embd,
                       int n_experts,
                       int top_k,
                       float expert_scale,
                       std::vector<int32_t> & selected,
                       std::vector<float> & weights,
                       std::string * error = nullptr,
                       std::vector<DsparkRouteBoundary> * boundaries = nullptr);

// q is [tokens, heads, head_dim], kv is [context+tokens, head_dim]. Positions
// match those rows. The result is [tokens, heads, head_dim] after forward RoPE,
// sink softmax, context reduction, and inverse RoPE.
bool dspark_attention_reduce(const float * q,
                             const float * kv,
                             const float * sinks,
                             const int32_t * query_positions,
                             const int32_t * kv_positions,
                             int tokens,
                             int context_rows,
                             int heads,
                             int head_dim,
                             int rope_dims,
                             float rope_base,
                             std::vector<float> & output,
                             std::string * error = nullptr);

}  // namespace ember::xdna2
