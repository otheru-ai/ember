// Qwen4Exp text-runtime ownership types.
//
// The released model is too large to duplicate its PLE table and routed
// experts in the backend allocation. Dense tensors are uploaded once, while
// token embeddings, PLE rows, and selected expert slices remain in a
// read-only mmap and are dequantized on demand. This is a correctness-first
// q=1 path; it intentionally exposes no vision or MTP state.

#pragma once

#include "common/cpu_embedder.h"
#include "common/qwen_yarn.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace dflash::common {

struct Qwen4ExpFrontierRuntime;
struct Qwen4ExpFrontierDenseCache;

struct Qwen4ExpMappedTensor {
    const uint8_t * data = nullptr;
    size_t bytes = 0;
    ggml_type type = GGML_TYPE_COUNT;
    std::array<int64_t, 4> ne{};
    int n_dims = 0;

    bool valid() const { return data && bytes && n_dims > 0; }
};

struct Qwen4ExpLayer {
    ggml_tensor * hc_attn_norm = nullptr;
    ggml_tensor * hc_attn_down = nullptr;
    ggml_tensor * hc_attn_up = nullptr;
    ggml_tensor * hc_attn_inject = nullptr;
    ggml_tensor * hc_ffn_norm = nullptr;
    ggml_tensor * hc_ffn_down = nullptr;
    ggml_tensor * hc_ffn_up = nullptr;
    ggml_tensor * hc_ffn_inject = nullptr;

    ggml_tensor * attn_q = nullptr;
    ggml_tensor * attn_k = nullptr;
    ggml_tensor * attn_v = nullptr;
    ggml_tensor * attn_output = nullptr;
    ggml_tensor * attn_q_norm = nullptr;
    ggml_tensor * attn_k_norm = nullptr;
    ggml_tensor * index_q = nullptr;
    ggml_tensor * index_k = nullptr;
    ggml_tensor * index_q_norm = nullptr;
    ggml_tensor * index_k_norm = nullptr;
    ggml_tensor * self_k_rot = nullptr; // optional quantization rotation
    ggml_tensor * self_v_rot = nullptr; // optional quantization rotation

    ggml_tensor * attn_qkv = nullptr;
    ggml_tensor * attn_gate = nullptr;
    ggml_tensor * ssm_conv = nullptr;
    ggml_tensor * ssm_a = nullptr;
    ggml_tensor * ssm_alpha = nullptr;
    ggml_tensor * ssm_beta = nullptr;
    ggml_tensor * ssm_dt = nullptr;
    ggml_tensor * ssm_norm = nullptr;
    ggml_tensor * ssm_out = nullptr;

    ggml_tensor * ple_key = nullptr;
    ggml_tensor * ple_value = nullptr;
    ggml_tensor * ple_norm_key = nullptr;
    ggml_tensor * ple_norm_query = nullptr;
    ggml_tensor * ple_norm_conv = nullptr;
    ggml_tensor * ple_conv = nullptr;

    ggml_tensor * router = nullptr;
    ggml_tensor * shared_gate_input = nullptr;
    ggml_tensor * shared_gate = nullptr;
    ggml_tensor * shared_up = nullptr;
    ggml_tensor * shared_down = nullptr;
    ggml_tensor * experts_gate_up_tensor = nullptr;
    ggml_tensor * experts_down_tensor = nullptr;
    Qwen4ExpMappedTensor experts_gate_up;
    Qwen4ExpMappedTensor experts_down;
};

// Every split file owns an independent ggml metadata context and mmap. Tensor
// descriptors and mapped expert/embedding pointers retain those owners for the
// complete model lifetime; coalescing only the dense payload into one backend
// buffer must not invalidate metadata from later shards.
struct Qwen4ExpWeightShard {
    ggml_context * ctx = nullptr;
    void * mmap_addr = nullptr;
    size_t mmap_size = 0;
    int mmap_fd = -1;
    std::string path;
};

// Persistent append-only cache used by QSA snapshots. Completed slabs are
// immutable and shared by every prefix snapshot. Appending after a snapshot
// copies only the current partial slab, never the full prefix.
class Qwen4ExpCowBuffer {
public:
    static constexpr size_t kSlabFloats = 65536;
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    float at(size_t index) const;
    void append(const float * values, size_t count);
    void clear();
    size_t shared_slab_count() const { return slabs_.size(); }
    uint64_t account_bytes(std::unordered_set<const void *> & seen) const;

private:
    std::vector<std::shared_ptr<std::vector<float>>> slabs_;
    size_t size_ = 0;
};

struct Qwen4ExpWeights {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<Qwen4ExpWeightShard> shards;

    CpuEmbedder embedder;
    Qwen4ExpMappedTensor ple_table;
    ggml_tensor * output = nullptr;
    ggml_tensor * output_hc_norm = nullptr;
    ggml_tensor * output_hc_down = nullptr;
    ggml_tensor * output_hc_up = nullptr;
    std::vector<Qwen4ExpLayer> layers;

    int max_ctx = 8192;
    int32_t eos_id = 248046;
    int32_t eot_id = 248044;
    ember_qwen_yarn_config yarn{};
    uint64_t resident_weight_bytes = 0;
    uint64_t state_budget_bytes = 0;
    Qwen4ExpFrontierRuntime * frontier = nullptr;
    // Persistent dense projection graphs and host copies of immutable small
    // tensors. Both borrow this weight set and are freed before `buf`.
    Qwen4ExpFrontierDenseCache * dense_cache = nullptr;
};

struct Qwen4ExpLayerState {
    // GDN state is fixed-size and much smaller than the QSA prefix, but a
    // deep copy per saved prefix is still wasteful.  Snapshots share these
    // vectors and the live state detaches once, on its first later write.
    std::shared_ptr<std::vector<float>> conv;       // GDN: [3,10240]
    std::shared_ptr<std::vector<float>> recurrent;  // GDN: [48,128,128], transposed
    Qwen4ExpCowBuffer key;        // QSA: appended [2,256]
    Qwen4ExpCowBuffer value;      // QSA: appended [2,256]
    Qwen4ExpCowBuffer index_key;  // QSA: appended raw [128]
};

struct Qwen4ExpState {
    int cur_pos = 0;
    int32_t last_token = -1;
    std::vector<float> hc; // current [2560,4] stream state
    std::array<Qwen4ExpLayerState, 48> layers;
    std::array<int32_t, 2> ple_tokens{{248044, 248044}};
    std::vector<float> ple_conv; // [9,10240], dilation=3, kernel=4
    // Full T/H/W history is required by the sparse indexer: pooled raw keys
    // are rotated at the first token of each four-token block, which is not
    // recoverable from a scalar cache index after an image span.
    std::array<std::vector<int32_t>, 3> mrope_positions;

    void clear();
    uint64_t account_bytes(std::unordered_set<const void *> & seen) const;
};

struct Qwen4ExpMemoryPlan {
    uint64_t resident_weight_bytes = 0;
    uint64_t qsa_cache_bytes = 0;
    uint64_t recurrent_state_bytes = 0;
    uint64_t runtime_reserve_bytes = 0;
    uint64_t capacity_bytes = 0;
    uint64_t total_bytes = 0;
    bool fits = false;
};

Qwen4ExpMemoryPlan qwen4exp_memory_plan(uint64_t resident_weight_bytes,
                                        int max_ctx);

struct Qwen4ExpSnapshot {
    bool used = false;
    Qwen4ExpState state;
    std::vector<float> logits;
};

// With at most the released 2048-token QSA budget visible, every complete
// four-token block is selected and the causal tail follows it.  The general
// scorer therefore cannot change the result: its insertion order already is
// block order and no partial sort runs when every block is retained.  Keep the
// boundary explicit so longer contexts continue through the scored path.
inline bool qwen4exp_qsa_dense_selection(int n_tokens,
                                         std::vector<int32_t> & selected) {
    constexpr int kDenseTokenLimit = 2048;
    if (n_tokens <= 0 || n_tokens > kDenseTokenLimit) return false;
    selected.resize(static_cast<size_t>(n_tokens));
    for (int token = 0; token < n_tokens; ++token) {
        selected[static_cast<size_t>(token)] = token;
    }
    return true;
}

bool qwen4exp_weight_type_supported(ggml_type type, bool vector_or_norm);
bool qwen4exp_mapped_row_f32(const Qwen4ExpMappedTensor & tensor,
                            int64_t row, float * out, size_t out_count,
                            std::string * error = nullptr);
std::vector<int32_t> qwen4exp_qsa_selected_tokens(
    const std::vector<float> & raw_index_keys,
    const float * query_heads, int n_tokens);
std::array<int32_t, 16> qwen4exp_ple_rows(
    int32_t token, const std::array<int32_t, 2> & history);

bool load_qwen4exp_gguf(const std::string & path, ggml_backend_t backend,
                        int max_ctx, bool enable_yarn, Qwen4ExpWeights & out,
                        std::string & error);
void free_qwen4exp_weights(Qwen4ExpWeights & weights);

// Executes one autoregressive row. `token` is the input row; returned logits
// predict the next token. A backend execution failure invalidates the partial
// frontier; callers must reset before reuse. Avoiding a transactional copy is
// essential because the GDN recurrent state is about 108 MiB.
bool qwen4exp_step_q1(const Qwen4ExpWeights & weights,
                      Qwen4ExpState & state, int32_t token,
                      std::vector<float> & logits, std::string & error);

// Same language-model step with an externally projected image row. `token`
// remains image_pad for PLE; only the ordinary input embedding is replaced.
bool qwen4exp_step_q1_embedding(
    const Qwen4ExpWeights & weights, Qwen4ExpState & state, int32_t token,
    const float * embedding, size_t embedding_count,
    const std::array<int32_t, 3> & mrope_position,
    std::vector<float> & logits, std::string & error);

bool qwen4exp_step_q1_mrope(
    const Qwen4ExpWeights & weights, Qwen4ExpState & state, int32_t token,
    const std::array<int32_t, 3> & mrope_position,
    std::vector<float> & logits, std::string & error);

// Direction-extraction seam used only for a prompt frontier. The returned
// record contains the 2560-wide attention hyper-connection mixed input for
// all 48 layers in numeric order. It is deliberately separate from ordinary
// generation so a capture can never become an accidental decode hot-path cost.
bool qwen4exp_step_q1_mrope_capture(
    const Qwen4ExpWeights & weights, Qwen4ExpState & state, int32_t token,
    const std::array<int32_t, 3> & mrope_position,
    std::vector<float> & logits, std::vector<float> & attn_mixed_capture,
    std::string & error);

// Native bounded verifier entry. Rows are evaluated layer-major so target
// weights/frontier graphs stay hot across the depth-1..4 MTP window, while
// PLE, GDN and QSA state still advance causally in row order. The final state
// covers every input row; callers must restore/replay on partial acceptance.
bool qwen4exp_step_batch_mrope(
    const Qwen4ExpWeights & weights,
    Qwen4ExpState & state,
    const std::vector<int32_t> & tokens,
    const std::vector<std::array<int32_t, 3>> & mrope_positions,
    std::vector<std::vector<float>> & row_logits,
    std::vector<std::vector<float>> & row_hc,
    std::string & error);

// Ordinary prefill variant of the bounded layer-major step. It exposes every
// raw target HC row for causal MTP cache synchronization, preserves the final
// one in `state.hc`, and computes only the final row's logits. MTP verification
// uses qwen4exp_step_batch_mrope above because it needs every logit row too.
bool qwen4exp_step_prefill_batch_mrope(
    const Qwen4ExpWeights & weights,
    Qwen4ExpState & state,
    const std::vector<int32_t> & tokens,
    const std::vector<std::array<int32_t, 3>> & mrope_positions,
    std::vector<float> & logits,
    std::vector<std::vector<float>> & row_hc,
    std::string & error);

// Pure scheduling seam shared by the backend and GPU-free deterministic
// tests. `batchable_rows` stops at the next vision/capture/MTP barrier.
size_t qwen4exp_prefill_chunk_rows(size_t batchable_rows, int current_pos,
                                   int snapshot_pos, bool force_q1);

} // namespace dflash::common
