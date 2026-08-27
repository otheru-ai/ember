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

struct Qwen4ExpReplayRow {
    int32_t token = -1;
    std::array<int32_t, 3> mrope_position{};
};

using Qwen4ExpReplayStep = bool (*)(
    void * opaque, Qwen4ExpState & state, const Qwen4ExpReplayRow & row,
    std::vector<float> & logits, std::string & error);

// Adapter for the production q=1 target step. Pass `Qwen4ExpWeights *` as
// opaque to qwen4exp_replay_accepted_prefix(). Kept explicit so tests can use
// a deterministic state mutator without a HIP backend.
bool qwen4exp_replay_target_q1(
    void * opaque, Qwen4ExpState & state, const Qwen4ExpReplayRow & row,
    std::vector<float> & logits, std::string & error);

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
