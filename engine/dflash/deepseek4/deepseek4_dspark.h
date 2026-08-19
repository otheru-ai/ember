// DeepSeek-V4-Flash "DSpark" speculative-decode drafter.
//
// The DSpark drafter is a small (n_layer≈3) DeepSeek-V4 block stack stored under
// the checkpoint's mtp.* namespace and converted to a GGUF with arch
// "deepseek4-dflash-draft" (see scripts/convert_ds4_dspark_draft_to_gguf.py).
//
// Reference forward: deepseek-ai/DeepSeek-V4-Flash-DSpark inference/model.py
// (DSparkBlock / DSparkAttention / DSparkMarkovHead / DSparkConfidenceHead,
//  Transformer.forward_spec). Key facts encoded here:
//   - The target captures h.mean(dim=2) (mean over the hc_mult HC copies) after
//     each of dspark_target_layer_ids (=[40,41,42]) -> main_hidden [.., 3*n_embd].
//   - forward_embed (stage 0): main_x = main_norm(main_proj(main_hidden)); the
//     noise block = embed([seed]+[MASK]*(block_size-1)), HC-expanded.
//   - DSparkAttention: compress_ratio==0 (no compressor/indexer). Each layer
//     projects the shared main_x -> main_kv via its own wkv, and the block's
//     block_size query positions attend BIDIRECTIONALLY over
//     [sliding-window main-context KV] ++ [block KV].
//   - forward_head (last stage): hc_head collapse -> out_norm -> tied target
//     lm_head -> per-position Markov correction + confidence gate.
//
// The drafter's token embedding and lm_head are TIED to the target, so the
// drafter GGUF carries neither token_embd nor output; embedding + projection
// go through the DeepSeek4DFlashTarget adapter.

#pragma once

#include "deepseek4_internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

class XdnaDSparkDraftCompute;
class XdnaDSparkDraftJob;

// The drafter weights. `core` reuses DeepSeek4Weights for the n_layer decoder
// blocks + per-layer tensors + metadata + out_norm + output_hc_* tail; its
// tok_embd/output stay null (tied to target). The DSpark-specific tensors below
// live in the same ggml context / backend buffer as `core`.
struct DSparkDrafter {
    DeepSeek4Weights core;

    // Captured-feature fusion (stage 0 only in the checkpoint, but stored global).
    ggml_tensor * main_proj    = nullptr;  // dflash.fc.weight            [n_tgt*n_embd, n_embd]
    ggml_tensor * main_norm    = nullptr;  // dflash.hidden_norm.weight   [n_embd]

    // DSpark heads (last stage).
    ggml_tensor * markov_w1    = nullptr;  // dflash.dspark.markov.w1        [markov_rank, vocab]
    ggml_tensor * markov_w2    = nullptr;  // dflash.dspark.markov.w2        [markov_rank, vocab]
    ggml_tensor * confidence_w = nullptr;  // dflash.dspark.confidence.weight [confidence_dim, 1]
    ggml_tensor * confidence_b = nullptr;  // dflash.dspark.confidence.bias   [1]

    int block_size      = 5;
    int n_target_layers = 3;
    int markov_rank     = 256;
    int vocab_size      = 129280;
    int confidence_dim  = 0;
    int mask_token_id   = 128799;
    bool dspark_enabled  = false;
    bool head_hc_enabled = false;
    std::vector<int> capture_layer_ids;  // [40,41,42]
};

// Load a "deepseek4-dflash-draft" GGUF into `out`. Returns false on error;
// deepseek4_dspark_last_error() has the message.
bool load_deepseek4_dspark_drafter(const std::string & path,
                                   ggml_backend_t backend,
                                   DSparkDrafter & out);

void free_deepseek4_dspark_drafter(DSparkDrafter & d);

// Drop thread-local graph/allocation state that retains drafter tensor and
// backend references. Called before releasing the drafter weights.
void reset_deepseek4_dspark_runtime_cache();

const char * deepseek4_dspark_last_error();

// GPU stage-0 projection for the heterogeneous DSpark split. Converts raw
// concatenated target features [ctx_len, n_target_layers*n_embd] into the
// post-main_norm [ctx_len, n_embd] context consumed by every draft layer and,
// when requested, the normalized pre-RoPE context KV rows for all draft layers
// [n_layer, ctx_len, head_dim].
// The graph/allocation are generation-worker TLS, like the monolithic draft
// graph. This is an opt-in placement primitive; the ordinary GPU drafter still
// accepts raw features and remains the fallback.
bool deepseek4_dspark_project_main_context(
    ggml_backend_t backend,
    const DSparkDrafter & d,
    const float * ctx_features,
    int ctx_len,
    std::vector<float> & main_context,
    std::vector<float> * context_kv = nullptr);

// One drafter forward. Produces block_size normed hidden states (the input to
// the tied lm_head + Markov head), conditioned on a window of captured target
// features. When requested, also returns the HC-collapsed state before the
// output RMSNorm, which is the trained input to the DSpark confidence head.
// All host-side f32 for a simple v1 (GPU feature-ring plumbing can come later).
//
//   noise_embed     : [n_embd * block_size] embeds of [seed]+[MASK]*(block_size-1);
//                     its first block element is the committed seed embedding
//   ctx_features    : [n_target_layers*n_embd * ctx_len] captured features,
//                     ordered oldest..newest, absolute positions
//                     [committed-ctx_len .. committed-1]
//   ctx_len         : number of context feature columns (<= n_swa)
//   committed       : absolute position of the seed (block position 0)
//   out_hidden      : filled with [n_embd * block_size] = out_norm(hc_head(block))
//   confidence_hidden: optional [n_embd * block_size] = hc_head(block), before
//                      out_norm; never use it for the tied lm_head
//
// Defined in deepseek4_graph.cpp (needs the static DS4 sub-builders).
bool deepseek4_dspark_draft_forward(ggml_backend_t backend,
                                    const DSparkDrafter & d,
                                    const float * noise_embed,
                                    const float * ctx_features,
                                    int ctx_len,
                                    int committed,
                                    std::vector<float> & out_hidden,
                                    std::vector<float> * confidence_hidden = nullptr);

// Target verify forward WITH feature capture. q=1 can reuse the ordinary fused
// decode graph with passive capture outputs; the opt-in approximate verifier
// can still send q=2..4 through the batched graph. Returns:
//   argmax_out  : per-position argmax token id (size n_tokens)
//   logits_out  : if non-null, full [n_tokens * n_vocab] f32 logits
//   capture_out : [n_target_layers*n_embd * n_tokens] f32, per-position mean
//                 over the hc_mult HC copies after each capture layer
//                 (concatenated in capture_layer_ids order) — the drafter's
//                 main_hidden feed.
// Advances/updates the target cache exactly like a decode of these tokens.
bool deepseek4_dspark_verify_forward(ggml_backend_t backend,
                                     int device,
                                     const DeepSeek4Weights & w,
                                     DeepSeek4Cache & cache,
                                     const std::vector<int> & capture_layer_ids,
                                     const float * embed,
                                     const int32_t * token_ids,
                                     int n_tokens,
                                     int kv_start,
                                     std::vector<int32_t> & argmax_out,
                                     std::vector<float> * logits_out,
                                     std::vector<float> & capture_out,
                                     DeepSeek4StepTelemetry * telemetry = nullptr,
                                     bool allow_graph_reuse = false,
                                     bool require_target_graph = false);

// Minimal speculative-decode rollback state. Rejected positions must restore
// the physical SWA rows they overwrote after the ring wraps; otherwise a later
// causal verify reads rejected-token KV as if it were older committed history.
// This remains much smaller than a full target-cache snapshot because q <= 4.
struct DeepSeek4SpecRollback {
    int raw_pos = 0;
    int raw_count = 0;
    struct Layer {
        std::vector<uint8_t> attn_kv, attn_sc, idx_kv, idx_sc;
        std::size_t raw_row_bytes = 0;
        std::vector<uint8_t> raw_rows;
    };
    std::vector<Layer> layers;
    std::vector<uint8_t> hc_state;
};

void deepseek4_spec_rollback_save(const DeepSeek4Cache & cache,
                                  DeepSeek4SpecRollback & rollback,
                                  int raw_pos,
                                  int raw_count);

void deepseek4_spec_rollback_apply(const DeepSeek4SpecRollback & rollback,
                                   const DeepSeek4Weights & weights,
                                   DeepSeek4Cache & cache,
                                   int commit_pos,
                                   bool restore_prev);

// One asynchronous, session-owned DSpark proposal.  Its implementation keeps
// the provider job and all phase-local buffers opaque so ResidentSession does
// not acquire XRT/ggml details.  A proposal is submitted while another
// session's target work is eligible to run, then finished later through the
// exact q=1 verifier.  Moving is allowed; copying would duplicate job ownership.
class DeepSeek4DSparkResidentProposal {
public:
    DeepSeek4DSparkResidentProposal();
    ~DeepSeek4DSparkResidentProposal();
    DeepSeek4DSparkResidentProposal(DeepSeek4DSparkResidentProposal &&) noexcept;
    DeepSeek4DSparkResidentProposal & operator=(
        DeepSeek4DSparkResidentProposal &&) noexcept;
    DeepSeek4DSparkResidentProposal(
        const DeepSeek4DSparkResidentProposal &) = delete;
    DeepSeek4DSparkResidentProposal & operator=(
        const DeepSeek4DSparkResidentProposal &) = delete;

    bool pending() const;
    void cancel() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend bool deepseek4_dspark_resident_prepare(
        ggml_backend_t, const DeepSeek4Weights &, const DSparkDrafter &,
        const std::vector<float> &, int, int32_t, int,
        XdnaDSparkDraftCompute &, DeepSeek4DSparkResidentProposal &,
        std::string *);
    friend bool deepseek4_dspark_resident_finish(
        ggml_backend_t, int, const DeepSeek4Weights &, DeepSeek4Cache &,
        const DSparkDrafter &, std::vector<float> &,
        DeepSeek4DSparkResidentProposal &, std::vector<int32_t> &,
        int32_t &, std::vector<float> &, int &, int &,
        struct DeepSeek4DSparkResidentTiming &, std::string *);
};

// Per-cycle wall phases for the coarse resident pipeline. Provider age includes
// useful queue/compute overlap; provider_block is only the wait still exposed
// when this session reaches the GPU verifier. Keeping both prevents aggregate
// throughput from hiding whether the NPU or target graph is the critical stage.
struct DeepSeek4DSparkResidentTiming {
    double provider_age_s = 0.0;
    double provider_block_s = 0.0;
    double head_s = 0.0;
    double verify_s = 0.0;
};

// Submit the support-model half of one resident speculative cycle.  `committed`
// is the KV frontier and `seed` is the already-sampled token not yet in KV.
// max_commit_tokens includes the seed and bounds both target mutation and the
// number of tokens later reported to the continuous-batch scheduler.
bool deepseek4_dspark_resident_prepare(
    ggml_backend_t backend,
    const DeepSeek4Weights & target_w,
    const DSparkDrafter & drafter,
    const std::vector<float> & feature_window,
    int committed,
    int32_t seed,
    int max_commit_tokens,
    XdnaDSparkDraftCompute & xdna_draft_compute,
    DeepSeek4DSparkResidentProposal & proposal,
    std::string * error = nullptr);

// Collect a submitted proposal, run the tied DSpark head and exact-prefix
// target verifier, append the accepted committed input tokens to `committed`,
// and return the deferred target bonus in `next_token`.  `last_logits` is the
// target distribution that produced next_token, preserving the ordinary AR
// seam if speculation is disabled on the following scheduler turn.
bool deepseek4_dspark_resident_finish(
    ggml_backend_t backend,
    int device,
    const DeepSeek4Weights & target_w,
    DeepSeek4Cache & target_cache,
    const DSparkDrafter & drafter,
    std::vector<float> & feature_window,
    DeepSeek4DSparkResidentProposal & proposal,
    std::vector<int32_t> & committed_tokens,
    int32_t & next_token,
    std::vector<float> & last_logits,
    int & offered_candidates,
    int & accepted_candidates,
    DeepSeek4DSparkResidentTiming & timing,
    std::string * error = nullptr);

// Run DSpark speculative decode: draft block_size candidates with `drafter`,
// verify against the DS4 target in one batched forward, accept the matching
// prefix, and loop. Returns generated tokens via `io.emit`. Mirrors the laguna
// DSpark loop. accept_rate_out (optional) gets mean accepted / block.
struct GenerateRequest;  // fwd (from common/…); the loop only needs n_gen + committed
bool run_deepseek4_dspark_spec_decode(
        ggml_backend_t backend,
        int device,
        const DeepSeek4Weights & target_w,
        DeepSeek4Cache & target_cache,
        const DSparkDrafter & drafter,
        int committed,
        int last_tok,
        int n_gen,
        const float * prompt_feature_window,  // [n_target_layers*n_embd * win_len] captured during prefill
        int win_len,
        std::vector<int32_t> & out_tokens,
        float * accept_rate_out,
        XdnaDSparkDraftCompute * xdna_draft_compute = nullptr,
        const std::function<bool(int32_t)> & on_token = {});

}  // namespace dflash::common
