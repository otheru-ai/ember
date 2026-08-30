// Qwen3.8-Flash-Next MTP state and strict target-replay seam.
//
// The released checkpoint contains one trained MTP block, but neither the
// pinned Transformers implementation nor the audited llama.cpp Qwen4Exp PRs
// execute or export it.  This header therefore defines only the parts Ember
// can make correct independently of the eventual HIP graph:
//
//   * the target row handed to the draft is the pre-output-mixer four-stream
//     HC state h_p [10240], paired with token embedding x_{p+1} [2560];
//   * the draft owns an independent QSA K/V/index cache.  It must never borrow
//     or mutate the target's QSA, GDN, PLE, HC, or M-RoPE state;
//   * after a partial verification acceptance, the target is restored to the
//     checkpoint before the bounded draft and only the accepted input rows are
//     replayed through the ordinary q=1 target step.
//
// The last rule ports the state boundary from HaloSpecKV commit 60ff854b
// (tools/server/server-context.cpp:3598-3636), adapted to Ember's complete
// Qwen4ExpState snapshot.  Full acceptance retains the verified fast-path
// state.  A replay failure restores the checkpoint transactionally.

#pragma once

#include "qwen4exp_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen4ExpFrontierMoeGraph;
struct Qwen4ExpFrontierDenseCache;
struct Qwen4ExpFrontierQsaGraph;

struct Qwen4ExpMtpInput {
    const float * target_hc = nullptr;       // h_p, exactly 10240 values
    size_t target_hc_count = 0;
    const float * next_embedding = nullptr;  // x_{p+1}, exactly 2560 values
    size_t next_embedding_count = 0;
    std::array<int32_t, 3> mrope_position{};
};

// State owned by the one-layer full-attention MTP head.  Its cache shape is
// deliberately the same QSA state type as a target QSA layer, but the object
// is separate so draft rejection cannot alter target cache coverage.
struct Qwen4ExpMtpState {
    int cur_pos = 0;
    Qwen4ExpLayerState qsa;
    std::vector<float> hc; // draft four-stream residual, [10240]
    std::array<std::vector<int32_t>, 3> mrope_positions;

    void clear() { *this = {}; }
    uint64_t account_bytes(std::unordered_set<const void *> & seen) const;
};

// A prompt chunk advances the target over every row, but the draft cache
// intentionally trails it by one row: token x_(p+1) is paired with target
// h_p and uses h_p's position p. `preceding_target_hc_row == -1` selects the
// committed pre-chunk HC and its position; otherwise it selects the same row
// from the target batch output and input positions. The pristine first token
// has no preceding target HC and is the only row omitted from the plan.
struct Qwen4ExpMtpPromptSyncRow {
    size_t token_row = 0;
    int preceding_target_hc_row = -1;
};

bool qwen4exp_mtp_prompt_sync_plan(
    int target_pos_before, int mtp_pos_before, size_t target_rows,
    std::vector<Qwen4ExpMtpPromptSyncRow> & plan, std::string & error);
bool qwen4exp_mtp_prompt_sync_position(
    const Qwen4ExpMtpPromptSyncRow & row,
    const std::array<int32_t, 3> & pre_chunk_target_position,
    const std::vector<std::array<int32_t, 3>> & target_batch_positions,
    std::array<int32_t, 3> & position, std::string & error);

// MTP shifts token IDs forward by one without shifting target hidden-state
// positions: (h_p, x_(p+1)) executes at p. Recursive proposal depth d executes
// at p+d. This is the Qwen4Exp contract in vLLM PR 53896,
// vllm/v1/spec_decode/llm_base_proposer.py:set_inputs_first_pass(). Read the
// authoritative last target M-RoPE row rather than deriving p from token count,
// which also preserves non-scalar image positions.
bool qwen4exp_mtp_chain_position(
    const Qwen4ExpState & target, size_t draft_depth,
    std::array<int32_t, 3> & position, std::string & error);

struct Qwen4ExpMtpCacheBatchShape {
    size_t rows = 0;
    size_t embedding_values = 0;    // [rows, 2560]
    size_t target_hc_values = 0;    // [rows, 10240]
    size_t hc_projection_rows = 0;  // four 2560-wide streams per row
    size_t key_values = 0;          // [rows, 2, 256]
    size_t value_values = 0;        // [rows, 2, 256]
    size_t index_key_values = 0;    // [rows, 128]
};

// Pure bounded-layout contract used before constructing GPU batch inputs.
// Widths beyond the target q16 frontier are rejected rather than silently
// split, because each target chunk is one causal synchronization boundary.
bool qwen4exp_mtp_cache_batch_shape(
    size_t rows, Qwen4ExpMtpCacheBatchShape & shape, std::string & error);

// Companion metadata is authoritative: controlled matrix tensors must match
// its exact Ember storage type while vectors and routers stay floating-point.
// Unknown contracts and mixed encodings fail closed before backend upload.
bool qwen4exp_mtp_matrix_quant_type_valid(
    const char * contract, const char * tensor_name, int dimensions,
    ggml_type type, std::string & error);

// Snapshot validity is stronger than a scalar position check. The MTP QSA
// K/V/raw-index caches and all M-RoPE axes must cover exactly the trailing,
// target-aligned draft frontier, while target_hc remains the target's
// authoritative h_p.
bool qwen4exp_mtp_frontier_valid(
    const Qwen4ExpState & target, const Qwen4ExpMtpState & mtp,
    const std::vector<float> & target_hc, std::string & error);

// The MTP companion owns only its trained block. Token embedding and output
// head remain borrowed from the matching target, exactly as the released
// `mtp_use_dedicated_embeddings=false` contract requires.
struct Qwen4ExpMtpWeights {
    ggml_backend_t backend = nullptr; // borrowed target backend
    ggml_backend_buffer_t buf = nullptr;
    std::vector<Qwen4ExpWeightShard> shards;

    ggml_tensor * pre_embedding_norm = nullptr;
    ggml_tensor * pre_hc_norm = nullptr;
    ggml_tensor * fc_embedding = nullptr;
    ggml_tensor * fc_hc = nullptr;
    Qwen4ExpLayer layer;
    ggml_tensor * output_hc_norm = nullptr;
    ggml_tensor * output_hc_down = nullptr;
    ggml_tensor * output_hc_up = nullptr;
    uint64_t resident_weight_bytes = 0;
    // Persistent q=1 graphs own only their compute arenas. All weights are
    // borrowed from `buf`, so the trained payload has one resident copy.
    Qwen4ExpFrontierMoeGraph * frontier_moe = nullptr;
    Qwen4ExpFrontierQsaGraph * frontier_qsa = nullptr;
    // Dense graphs and immutable host tensor copies are companion-owned and
    // destroyed before the companion backend buffer.
    Qwen4ExpFrontierDenseCache * dense_cache = nullptr;
};

struct Qwen4ExpMtpStats {
    uint64_t rounds = 0;
    uint64_t proposed = 0;
    uint64_t accepted = 0;
    uint64_t partial_replays = 0;

    float accept_rate() const {
        return proposed ? static_cast<float>(accepted) /
                              static_cast<float>(proposed) : 0.0f;
    }
};

// Depth is bounded by the trained one-layer chain contract and by the number
// of output slots remaining after the ordinary base token. Returning zero is
// the explicit q=1 fallback for an invalid/empty window.
int qwen4exp_mtp_effective_depth(int configured_depth,
                                 int remaining_output_slots);
bool qwen4exp_mtp_record_round(Qwen4ExpMtpStats & stats, uint64_t proposed,
                               uint64_t accepted, bool partial_replay);

using Qwen4ExpLayerMajorStep = bool (*)(
    void * opaque, size_t layer, size_t row, std::string & error);
bool qwen4exp_run_layer_major(size_t rows, size_t layers,
                              Qwen4ExpLayerMajorStep step, void * opaque,
                              std::string & error);

bool load_qwen4exp_mtp_gguf(const std::string & path,
                            ggml_backend_t backend,
                            Qwen4ExpMtpWeights & out,
                            std::string & error);
void free_qwen4exp_mtp_weights(Qwen4ExpMtpWeights & weights);

// One trained-head call. `target_hc` is h_p; `token` supplies x_(p+1).
// `draft_hc` receives the raw four-stream output used to chain depths 2-4.
bool qwen4exp_mtp_step_q1(
    const Qwen4ExpWeights & target,
    const Qwen4ExpMtpWeights & mtp,
    Qwen4ExpMtpState & state,
    int32_t token,
    const float * next_embedding,
    size_t next_embedding_count,
    const float * target_hc,
    size_t target_hc_count,
    const std::array<int32_t, 3> & mrope_position,
    std::vector<float> & logits,
    std::vector<float> & draft_hc,
    std::string & error);

// Prompt/AR synchronization needs only the persistent attention inputs for a
// future draft step. It appends MTP QSA K/V/raw-indexK, M-RoPE history and
// cur_pos, deliberately skipping query/attention output, FFN/MoE, HC output
// mixing, and the vocabulary head.
bool qwen4exp_mtp_sync_cache_q1(
    const Qwen4ExpWeights & target,
    const Qwen4ExpMtpWeights & mtp,
    Qwen4ExpMtpState & state,
    int32_t token,
    const float * next_embedding,
    size_t next_embedding_count,
    const float * target_hc,
    size_t target_hc_count,
    const std::array<int32_t, 3> & mrope_position,
    std::string & error);

// Bounded text-prompt synchronization. All projection and HC-mix rows are
// evaluated as q<=16 matrices; only the final K/V/raw-indexK and position
// commits are row-ordered. `target_hc_rows[r]` is the preceding authoritative
// target HC selected by qwen4exp_mtp_prompt_sync_plan().
bool qwen4exp_mtp_sync_cache_batch(
    const Qwen4ExpWeights & target,
    const Qwen4ExpMtpWeights & mtp,
    Qwen4ExpMtpState & state,
    const std::vector<int32_t> & tokens,
    const std::vector<std::vector<float>> & target_hc_rows,
    const std::vector<std::array<int32_t, 3>> & mrope_positions,
    std::string & error);

struct Qwen4ExpReplayRow {
    int32_t token = -1;
    std::array<int32_t, 3> mrope_position{};
};

using Qwen4ExpReplayStep = bool (*)(
    void * opaque, Qwen4ExpState & state, const Qwen4ExpReplayRow & row,
    std::vector<float> & logits, std::string & error);

struct Qwen4ExpMtpVerifyOutput {
    // One next-token logit row and raw target HC row for every consumed input
    // row. The final entry is retained when every candidate is accepted.
    std::vector<std::vector<float>> row_logits;
    std::vector<std::vector<float>> row_hc;
};

using Qwen4ExpVerifyBatch = bool (*)(
    void * opaque, Qwen4ExpState & state,
    const std::vector<Qwen4ExpReplayRow> & rows,
    Qwen4ExpMtpVerifyOutput & output, std::string & error);

// Adapter for the production q=1 target step. Pass `Qwen4ExpWeights *` as
// opaque to qwen4exp_replay_accepted_prefix(). Kept explicit so tests can use
// a deterministic state mutator without a HIP backend.
bool qwen4exp_replay_target_q1(
    void * opaque, Qwen4ExpState & state, const Qwen4ExpReplayRow & row,
    std::vector<float> & logits, std::string & error);

// Production verifier adapter. Its bounded contract is independent of the
// execution provider: the current correctness fallback executes q=1 rows;
// the gfx1151 provider may replace the body with one native q>1 graph without
// changing acceptance/replay scheduling.
bool qwen4exp_verify_target_batch(
    void * opaque, Qwen4ExpState & state,
    const std::vector<Qwen4ExpReplayRow> & rows,
    Qwen4ExpMtpVerifyOutput & output, std::string & error);

enum class Qwen4ExpReplayDisposition {
    FullAcceptance,
    RestoredCheckpoint,
    ReplayedAcceptedPrefix,
};

struct Qwen4ExpReplayResult {
    Qwen4ExpReplayDisposition disposition =
        Qwen4ExpReplayDisposition::RestoredCheckpoint;
    size_t rows_replayed = 0;
};

struct Qwen4ExpMtpVerifyResult {
    size_t accepted_predictions = 0;
    size_t committed_input_rows = 0;
    bool terminal_prediction = false;
    Qwen4ExpReplayResult replay;
};

// Runs one bounded target batch over [base, candidate_0, ...], compares each
// row's next-token logits to the corresponding candidate, and reconciles the
// complete target state. Batched acceptance is tentative: q=1 replay confirms
// every candidate before it can be committed, so reduction-order argmax drift
// cannot change the generated token stream.
bool qwen4exp_verify_bounded_batch(
    Qwen4ExpState & target_state,
    std::vector<float> & target_logits,
    const std::vector<Qwen4ExpReplayRow> & input_rows,
    const std::vector<int32_t> & candidates,
    int32_t eos_id,
    int32_t eot_id,
    Qwen4ExpVerifyBatch verify_batch,
    Qwen4ExpReplayStep replay_step,
    void * opaque,
    Qwen4ExpMtpVerifyOutput & output,
    Qwen4ExpMtpVerifyResult & result,
    std::string & error);

// Reconcile a speculative target verification with its committed checkpoint.
// `verified_state`/`verified_logits` are in/out and may contain mutations from
// all proposed rows. `accepted_rows` counts accepted members of `draft_rows`.
// The caller must checkpoint before *any* member of `draft_rows` is evaluated.
bool qwen4exp_replay_accepted_prefix(
    const Qwen4ExpState & committed_state,
    const std::vector<float> & committed_logits,
    Qwen4ExpState & verified_state,
    std::vector<float> & verified_logits,
    const std::vector<Qwen4ExpReplayRow> & draft_rows,
    size_t accepted_rows,
    Qwen4ExpReplayStep step,
    void * opaque,
    Qwen4ExpReplayResult & result,
    std::string & error);

} // namespace dflash::common
