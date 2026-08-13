// DeepSeek4Backend — ModelBackend for DeepSeek V4 Flash MLA+MoE models.
//
// Architecture: Multi-head Latent Attention (MLA), KV compression with
// learned compressors, Hierarchical Controller (HC), MoE with hash routing
// (first 3 layers) + top-k routing + shared expert.

#pragma once

#include "common/model_backend.h"
#include "common/progress_cycle_detector.h"
#include "common/sampler.h"
#include "../common/moe_hybrid_placement.h"
#include "../common/moe_hybrid_routing_stats.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_hybrid_stream.h"
#include "../common/moe_expert_compute_xdna.h"
#include "deepseek4_internal.h"
#include "deepseek4_dspark.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace dflash::common {

class DeepSeek4Backend : public ModelBackend, public ResidentBatchBackend {
public:
    explicit DeepSeek4Backend(const DeepSeek4BackendConfig & cfg);
    ~DeepSeek4Backend() override;

    DeepSeek4Backend(const DeepSeek4Backend &) = delete;
    DeepSeek4Backend & operator=(const DeepSeek4Backend &) = delete;

    bool init();

    // ModelBackend interface
    void print_ready_banner() const override;

    bool park(ParkTarget target) override;
    bool unpark(ParkTarget target) override;
    bool is_target_parked() const override { return parked_; }

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;

    void release_idle_graphs() override;
    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;
    // Ondisk prefix cache. Without these the base-class no-ops make every
    // disk save fail silently (save() needs ref.ctx to enumerate tensors).
    SnapshotRef snapshot_ref(int slot) const override;
    bool snapshot_adopt(int slot, ggml_context * ctx,
                        ggml_backend_buffer_t buf, int cur_pos,
                        int32_t last_tok = -1) override;

    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    // Resident continuous-batch sessions. The first implementation executes
    // decode rows sequentially while preserving independent cache state; the
    // scheduler/executor contract remains the same when a native fused batch
    // replaces it.
    bool resident_session_create(ContinuousBatchSessionId id,
                                 const GenerateRequest &request,
                                 const DaemonIO &io,
                                 int restore_slot,
                                 std::string *error) override;
    bool resident_session_destroy(ContinuousBatchSessionId id) override;
    bool resident_session_cancel(ContinuousBatchSessionId id) override;
    bool resident_session_decode_ready(
        ContinuousBatchSessionId id) const override;
    ResidentBatchBackend::SessionStatus resident_session_status(
        ContinuousBatchSessionId id) const override;
    GenerateResult resident_session_result(
        ContinuousBatchSessionId id) const override;
    bool resident_session_snapshot(
        ContinuousBatchSessionId id, int slot) override;
    ContinuousBatchPrefillCompletion prefill(
        ContinuousBatchSessionId id, int requested_tokens) override;
    std::vector<ContinuousBatchDecodeCompletion> decode_batch(
        const std::vector<ContinuousBatchSessionId> &sessions) override;
    bool supports_native_mixed() const override { return true; }
    ContinuousBatchMixedCompletion execute_mixed(
        ContinuousBatchSessionId prefill_session,
        int requested_tokens,
        const std::vector<ContinuousBatchSessionId> &decode_sessions) override;

    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;

    void shutdown() override;

private:
    DeepSeek4BackendConfig cfg_;
    ggml_backend_t         backend_      = nullptr;
    ggml_backend_t         snap_backend_ = nullptr;
    DeepSeek4Weights       w_;
    DeepSeek4Cache         cache_;
    bool                   parked_       = false;

    // Sampler
    SamplerCfg             sampler_;
    std::mt19937_64        sampler_rng_{std::random_device{}()};

    // Snapshots
    static constexpr int PREFIX_SLOTS = 64;
    DeepSeek4Snapshot      snapshots_[PREFIX_SLOTS];
    std::vector<float>     snapshot_logits_[PREFIX_SLOTS];
    std::vector<float>     snapshot_spec_features_[PREFIX_SLOTS];
    std::vector<float>     last_logits_;
    // Set by do_prefill when it actually persists the inline snapshot for the
    // request's snap_slot; surfaced into GenerateResult::snapshot_saved so the
    // server only commits its logical prefix entry on a real save (see #2).
    bool                   inline_snapshot_saved_ = false;

    // DSpark speculative decode (opt-in: DFLASH_DS4_SPEC=1 + DFLASH_DS4_DRAFT=<gguf>).
    bool                           spec_enabled_ = false;
    bool                           spec_drafter_parked_ = false;
    std::string                    spec_draft_path_;
    std::unique_ptr<DSparkDrafter> spec_drafter_;
    std::vector<float>             spec_feat_window_;

    bool load_spec_drafter();
    void release_spec_drafter(bool mark_parked);

    struct ResidentSession;
    std::unordered_map<ContinuousBatchSessionId,
                       std::unique_ptr<ResidentSession>> resident_sessions_;
    std::vector<DeepSeek4Cache> resident_cache_pool_;
    void swap_resident_state(ResidentSession &session);
    bool resident_sample_next(ResidentSession &session);
    void recycle_resident_cache(DeepSeek4Cache &cache) noexcept;
    void free_resident_sessions();

    // Prefill prompt tokens in chunks, return absolute committed position.
    // Clamp a proposed chunk at stateful boundaries. `capture_from` is
    // relative to the supplied token span and negative when DSpark capture is
    // disabled.
    static int clamp_prefill_chunk(int proposed_tokens,
                                   int relative_offset,
                                   int absolute_pos,
                                   int snap_pos,
                                   int capture_from);

    int do_prefill(const std::vector<int32_t> & tokens, const DaemonIO & io,
                   int kv_offset = 0,
                   int snap_pos = -1, int snap_slot = -1,
                   bool allow_spec_capture = true,
                   bool force_exact_prefill = false);

    // Autoregressive decode loop.
    // resume_from: number of tokens already present in `out_tokens` (and
    // already emitted) when AR takes over from speculative decode. The loop
    // counter doubles as the index into out_tokens, so it must start there or
    // both the force-close trigger and the position computation are wrong.
    // Requires the spec seam invariant: every out_token except the last is in
    // KV, which is what run_deepseek4_dspark_spec_decode leaves behind.
    bool do_decode(int committed, int n_gen,
                   const std::vector<int32_t> & history_prefix,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io,
                   const BudgetHook & budget_hook = {},
                   bool * forced_close_out = nullptr,
                   bool * degenerate_close_out = nullptr,
                   std::string * termination_reason_out = nullptr,
                   const std::function<bool()> & force_greedy = {},
                   int resume_from = 0,
                   // Disarm the prompt-echo watchdog inside a DSML tool call;
                   // see GenerateRequest::tool_region_open_ids.
                   const std::vector<int32_t> & tool_open_ids = {},
                   const std::vector<int32_t> & tool_close_ids = {},
                   // Constrained decoding; null leaves sampling untouched.
                   const std::shared_ptr<dflash::common::TokenMask> &
                       token_mask = {});

    bool load_model();
    bool init_hybrid_model();
    bool init_xdna_moe_provider();
    bool requires_monolithic_model() const;
    bool validate_prefill_mode() const;
    bool compute_uniform_hybrid_placement(const DeepSeek4Weights & w,
                                          int max_ctx,
                                          MoeHybridPlacement & out,
                                          std::string * err) const;
    void maybe_save_routing_stats();

    std::shared_ptr<MoeHybridStorage> moe_hybrid_;
    MoeHybridPlacement                moe_placement_;
    MoeHybridStreamEngine             stream_engine_;
    std::unique_ptr<MoeExpertCompute> xdna_expert_compute_;
    std::vector<MoeExpertLayer>       xdna_expert_layers_;
    // Expert IPC removed — layer split replaces expert split.
    // Kept for compilation compatibility; init_hybrid_model() is no longer called
    // from the layer-split path.
    std::shared_ptr<MoeHybridRoutingStats> routing_stats_;
    std::string                       routing_stats_out_path_;
};

}  // namespace dflash::common
