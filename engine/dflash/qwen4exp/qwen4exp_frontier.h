// Persistent Qwen4Exp MoE frontier graph.
//
// Qwen's routed FFN is a single mathematical unit: router, top-k selection,
// selected-probability renormalization, fused or separate gate/up experts,
// down experts,
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
struct Qwen4ExpMtpWeights;

struct Qwen4ExpFrontierMoeSpec {
    int n_embd = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    int n_ff = 0;
};

struct Qwen4ExpFrontierMoeWeights {
    ggml_tensor * router = nullptr;
    ggml_tensor * experts_gate_up = nullptr;
    ggml_tensor * experts_gate = nullptr;
    ggml_tensor * experts_up = nullptr;
    ggml_tensor * experts_down = nullptr;
    ggml_tensor * shared_gate_input = nullptr;
    ggml_tensor * shared_gate = nullptr;
    ggml_tensor * shared_up = nullptr;
    ggml_tensor * shared_down = nullptr;
};

struct Qwen4ExpFrontierMoeGraph;
struct Qwen4ExpFrontierDenseCache;
struct Qwen4ExpFrontierGdnGraph;
struct Qwen4ExpFrontierQsaGraph;

struct Qwen4ExpFrontierGdnSpec {
    int n_embd = 0;
    int n_heads = 0;
    int n_key_heads = 0;
    int head_dim = 0;
    int conv_width = 4;
    float epsilon = 1.0e-6f;
};

struct Qwen4ExpFrontierGdnWeights {
    ggml_tensor * qkv = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * alpha = nullptr;
    ggml_tensor * beta = nullptr;
    ggml_tensor * conv = nullptr;
    ggml_tensor * a = nullptr;
    ggml_tensor * dt = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * output = nullptr;
};

struct Qwen4ExpFrontierQsaSpec {
    int n_embd = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int n_index_heads = 0;
    int index_dim = 0;
};

struct Qwen4ExpFrontierQsaWeights {
    ggml_tensor * query = nullptr;
    ggml_tensor * key = nullptr;
    ggml_tensor * value = nullptr;
    ggml_tensor * index_query = nullptr;
    ggml_tensor * index_key = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * key_rotation = nullptr;
    ggml_tensor * value_rotation = nullptr;
};

constexpr int kQwen4ExpFrontierMoeMaxBatch = 16;
constexpr int kQwen4ExpFrontierMoeMtpBatch = 5;
constexpr int kQwen4ExpFrontierMoeCachedGraphsPerLayer = 3;

// The runtime permanently owns only q=1, q=5 (the maximum native MTP verify
// window), and q=16 arenas. Short bounded batches are zero-padded to the next
// cached width and their padding rows are discarded. MoE rows are independent,
// so padding cannot change a real row, and arbitrary prompt/MTP remainders
// cannot grow 48 separate sets of graph arenas.
int qwen4exp_frontier_moe_cached_width(int n_tokens);

// Dense projections share the same bounded width policy as the MoE frontier.
// The cache key is `(weight descriptor, cached width)`: q2-q5 reuse q5 and
// q6-q16 reuse q16 with zero-padded independent rows.  A cache belongs to one
// loaded weight set and its generation-worker thread; callers must destroy it
// before freeing the tensors or backend it borrows.
int qwen4exp_frontier_dense_cached_width(int n_tokens);
Qwen4ExpFrontierDenseCache * qwen4exp_frontier_dense_cache_create();
void qwen4exp_frontier_dense_cache_destroy(
    Qwen4ExpFrontierDenseCache * cache);
bool qwen4exp_frontier_dense_eval(
    Qwen4ExpFrontierDenseCache * cache, ggml_backend_t backend,
    ggml_tensor * weight, const float * input, int input_count, int n_tokens,
    std::vector<float> & output, std::string & error);
bool qwen4exp_frontier_static_f32(
    Qwen4ExpFrontierDenseCache * cache, ggml_tensor * tensor,
    std::vector<float> & output, std::string & error);
size_t qwen4exp_frontier_dense_graph_count(
    const Qwen4ExpFrontierDenseCache * cache);
size_t qwen4exp_frontier_static_f32_count(
    const Qwen4ExpFrontierDenseCache * cache);

// The q1 GDN graph fuses all four input projections, depthwise convolution,
// normalized Q/K preparation, recurrent delta-net update, output norm/gate,
// and output projection. The current snapshot representation is authoritative
// host COW state, so evaluation synchronizes convolution and recurrent state at
// one explicit layer boundary. This is intentionally conservative: rejected
// speculative rows can replay any saved state exactly, while a later resident
// mirror can remove the bounded transfer without changing the API contract.
Qwen4ExpFrontierGdnGraph * qwen4exp_frontier_gdn_create_q1(
    ggml_backend_t backend, const Qwen4ExpFrontierGdnSpec & spec,
    const Qwen4ExpFrontierGdnWeights & weights, int layer,
    std::string & error);
Qwen4ExpFrontierGdnGraph * qwen4exp_frontier_gdn_create_batch(
    ggml_backend_t backend, const Qwen4ExpFrontierGdnSpec & spec,
    const Qwen4ExpFrontierGdnWeights & weights, int layer, int n_tokens,
    std::string & error);
void qwen4exp_frontier_gdn_destroy(Qwen4ExpFrontierGdnGraph * graph);
bool qwen4exp_frontier_gdn_eval_q1(
    Qwen4ExpFrontierGdnGraph * graph, const float * input,
    size_t input_count, const float * conv_state, size_t conv_state_count,
    const float * recurrent_state, size_t recurrent_state_count,
    std::vector<float> & output, std::vector<float> & next_conv_state,
    std::vector<float> & next_recurrent_state, std::string & error);
bool qwen4exp_frontier_gdn_eval_batch(
    Qwen4ExpFrontierGdnGraph * graph, const float * input,
    size_t input_count, const float * conv_state, size_t conv_state_count,
    const float * recurrent_state, size_t recurrent_state_count,
    std::vector<float> & output, std::vector<float> & next_conv_state,
    std::vector<float> & next_recurrent_state, std::string & error);
uint64_t qwen4exp_frontier_gdn_state_transfer_bytes_q1(
    const Qwen4ExpFrontierGdnSpec & spec);
uint64_t qwen4exp_frontier_gdn_state_transfer_bytes_batch(
    const Qwen4ExpFrontierGdnSpec & spec, int n_tokens);

// QSA has an unavoidable data-dependent boundary: the current raw index-K
// projection participates in host top-block selection before attention can be
// built. The persistent implementation therefore uses one fused projection
// graph, one optional fused K/Q/V rotation graph, and a lazily-created
// width-bucketed flash-attention/output graph. Host code remains authoritative
// for exact RMSNorm, M-RoPE, block selection, and snapshot publication.
int qwen4exp_frontier_qsa_cached_width(int selected_tokens);
Qwen4ExpFrontierQsaGraph * qwen4exp_frontier_qsa_create_q1(
    ggml_backend_t backend, const Qwen4ExpFrontierQsaSpec & spec,
    const Qwen4ExpFrontierQsaWeights & weights, int layer,
    std::string & error);
void qwen4exp_frontier_qsa_destroy(Qwen4ExpFrontierQsaGraph * graph);
bool qwen4exp_frontier_qsa_project_q1(
    Qwen4ExpFrontierQsaGraph * graph, const float * input,
    size_t input_count, std::vector<float> & query_gate,
    std::vector<float> & key, std::vector<float> & value,
    std::vector<float> & index_query, std::vector<float> & index_key,
    std::string & error);
bool qwen4exp_frontier_qsa_rotate_q1(
    Qwen4ExpFrontierQsaGraph * graph, std::vector<float> & query,
    std::vector<float> & key, std::vector<float> & value,
    std::string & error);
// selected K/V are head-major [kv_head, selected_token, head_dim].
bool qwen4exp_frontier_qsa_attend_q1(
    Qwen4ExpFrontierQsaGraph * graph, const float * query,
    size_t query_count, const float * gate, size_t gate_count,
    const float * selected_key, const float * selected_value,
    int selected_tokens, std::vector<float> & output, std::string & error);
size_t qwen4exp_frontier_qsa_arena_bytes(
    const Qwen4ExpFrontierQsaGraph * graph);
uint64_t qwen4exp_frontier_qsa_transfer_bytes_q1(
    const Qwen4ExpFrontierQsaSpec & spec, int selected_tokens);
uint64_t qwen4exp_frontier_qsa_selected_state_bytes_q1(
    const Qwen4ExpFrontierQsaSpec & spec, int selected_tokens);

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
bool qwen4exp_frontier_moe_available(const Qwen4ExpWeights & weights,
                                     int layer);
bool qwen4exp_frontier_gdn_available(const Qwen4ExpWeights & weights,
                                     int layer);
bool qwen4exp_frontier_qsa_available(const Qwen4ExpWeights & weights,
                                     int layer);
bool qwen4exp_frontier_moe_q1(const Qwen4ExpWeights & weights, int layer,
                              const float * input, size_t input_count,
                              std::vector<float> & output,
                              std::string & error);
bool qwen4exp_frontier_gdn_q1(
    const Qwen4ExpWeights & weights, int layer, const float * input,
    size_t input_count, const float * conv_state, size_t conv_state_count,
    const float * recurrent_state, size_t recurrent_state_count,
    std::vector<float> & output, std::vector<float> & next_conv_state,
    std::vector<float> & next_recurrent_state, std::string & error);
bool qwen4exp_frontier_gdn_batch(
    const Qwen4ExpWeights & weights, int layer, const float * input,
    size_t input_count, int n_tokens, const float * conv_state,
    size_t conv_state_count, const float * recurrent_state,
    size_t recurrent_state_count, std::vector<float> & output,
    std::vector<float> & next_conv_state,
    std::vector<float> & next_recurrent_state, std::string & error);
Qwen4ExpFrontierQsaGraph * qwen4exp_frontier_qsa_q1(
    const Qwen4ExpWeights & weights, int layer);
bool qwen4exp_frontier_moe_batch(const Qwen4ExpWeights & weights, int layer,
                                 const float * input, size_t input_count,
                                 int n_tokens, std::vector<float> & output,
                                 std::string & error);

// The separately loaded MTP companion owns one q=1 frontier graph. Its expert
// tensors remain in the companion's existing backend buffer; the graph never
// clones weight payloads and is destroyed before that buffer.
bool qwen4exp_frontier_mtp_create(Qwen4ExpMtpWeights & weights,
                                  std::string & error);
void qwen4exp_frontier_mtp_destroy(Qwen4ExpMtpWeights & weights);
bool qwen4exp_frontier_mtp_moe_q1(const Qwen4ExpMtpWeights & weights,
                                  const float * input, size_t input_count,
                                  std::vector<float> & output,
                                  std::string & error);

} // namespace dflash::common
