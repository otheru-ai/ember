// Persistent Qwen4Exp MoE frontier graph.
//
// Qwen's routed FFN is a single mathematical unit: router, top-k selection,
// selected-probability renormalization, fused gate/up experts, down experts,
// and the gated shared expert.  Keeping those operations in one reusable ggml
// graph removes the correctness-first runtime's host router round-trip and ten
// scalar expert evaluations per layer.  The graph is backend-neutral so its
// exact semantics can be tested with the CPU backend; production uses the
// existing gfx1151 MMVQ/MMQ mul_mat_id kernels.

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen4ExpWeights;

struct Qwen4ExpFrontierMoeSpec {
    int n_embd = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    int n_ff = 0;
};

struct Qwen4ExpFrontierMoeWeights {
    ggml_tensor * router = nullptr;
    ggml_tensor * experts_gate_up = nullptr;
    ggml_tensor * experts_down = nullptr;
    ggml_tensor * shared_gate_input = nullptr;
    ggml_tensor * shared_gate = nullptr;
    ggml_tensor * shared_up = nullptr;
    ggml_tensor * shared_down = nullptr;
};

struct Qwen4ExpFrontierMoeGraph;

constexpr int kQwen4ExpFrontierMoeMaxBatch = 16;
constexpr int kQwen4ExpFrontierMoeMtpBatch = 5;
constexpr int kQwen4ExpFrontierMoeCachedGraphsPerLayer = 3;

// The runtime permanently owns only q=1, q=5 (the maximum native MTP verify
// window), and q=16 arenas. Short bounded batches are zero-padded to the next
// cached width and their padding rows are discarded. MoE rows are independent,
// so padding cannot change a real row, and arbitrary prompt/MTP remainders
// cannot grow 48 separate sets of graph arenas.
int qwen4exp_frontier_moe_cached_width(int n_tokens);

Qwen4ExpFrontierMoeGraph * qwen4exp_frontier_moe_create(
    ggml_backend_t backend, const Qwen4ExpFrontierMoeSpec & spec,
    const Qwen4ExpFrontierMoeWeights & weights, int layer,
    std::string & error);
Qwen4ExpFrontierMoeGraph * qwen4exp_frontier_moe_create_batch(
    ggml_backend_t backend, const Qwen4ExpFrontierMoeSpec & spec,
    const Qwen4ExpFrontierMoeWeights & weights, int layer, int n_tokens,
    std::string & error);
void qwen4exp_frontier_moe_destroy(Qwen4ExpFrontierMoeGraph * graph);
bool qwen4exp_frontier_moe_eval(Qwen4ExpFrontierMoeGraph * graph,
                                const float * input, size_t input_count,
                                std::vector<float> & output,
                                std::string & error);

struct Qwen4ExpFrontierRuntime;

bool qwen4exp_frontier_create(Qwen4ExpWeights & weights, std::string & error);
void qwen4exp_frontier_destroy(Qwen4ExpWeights & weights);
bool qwen4exp_frontier_moe_q1(const Qwen4ExpWeights & weights, int layer,
                              const float * input, size_t input_count,
                              std::vector<float> & output,
                              std::string & error);
bool qwen4exp_frontier_moe_batch(const Qwen4ExpWeights & weights, int layer,
                                 const float * input, size_t input_count,
                                 int n_tokens, std::vector<float> & output,
                                 std::string & error);

} // namespace dflash::common
