// GPU-free mathematical oracles for the Qwen4Exp text architecture.
//
// These routines intentionally stop at tensor-free equations: they neither
// build a ggml graph nor claim runtime performance.  Their purpose is to pin
// the checkpoint's integer hashing and the ordering of its sparse-attention,
// gated-residual, and recurrent operations for later backend comparisons.
// Matrix arguments are row-major, with output rows followed by input columns.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dflash::qwen4exp::reference {

constexpr std::size_t kResidualStreams = 4;
constexpr std::size_t kPleNgramSize = 3;
constexpr std::size_t kPleHeadsPerNgram = 8;
constexpr std::size_t kPleHeadCount = 16;
constexpr std::size_t kQsaBlockSize = 4;
constexpr std::size_t kQsaTokenBudget = 512;

struct PleHashParameters {
    std::uint64_t eos_token_id = 0;
    std::array<std::uint64_t, kPleNgramSize> multipliers{};
    std::array<std::uint64_t, kPleHeadCount> head_vocab_sizes{};
    std::array<std::uint64_t, kPleHeadCount> head_offsets{};
};

// Exact I64 buffers and primary EOS from Qwen/Qwen3.8-Flash-Next revision
// f5d08274bafd880402bd16f5e3e6c514136ec06c.  llama.cpp PR #27742 carries
// these buffers losslessly through GGUF metadata rather than float32 tensors.
const PleHashParameters & released_ple_hash_parameters();

// Returns token-major rows: 8 bigram heads followed by 8 trigram heads for
// every input token. previous_context is ordered oldest-to-newest; only its
// newest two entries are relevant. Missing history is padded with EOS.
std::vector<std::uint64_t> ple_hash_indices(
    const std::vector<std::uint64_t> & input_ids,
    const std::vector<std::uint64_t> & previous_context,
    const PleHashParameters & parameters);

// Mean-pool complete visible blocks. The incomplete causal tail is deliberately
// absent here because QSA appends those token indices without scoring it.
std::vector<double> qsa_pool_complete_blocks(
    const std::vector<double> & raw_keys,
    std::size_t key_dimension,
    const std::vector<std::size_t> & visible_token_indices,
    std::size_t block_size = kQsaBlockSize);

// Score already-normalized/rotated pooled keys with the official per-head
// relu(dot) sum divided by sqrt(head_dimension).
std::vector<double> qsa_score_blocks(
    const std::vector<double> & query_heads,
    std::size_t head_count,
    std::size_t head_dimension,
    const std::vector<double> & pooled_keys);

// Select whole blocks, highest score first, expand each selected block to its
// member token indices, then append the incomplete visible tail. Equal scores
// use the lower block index first so the oracle is stable across host libraries.
std::vector<std::size_t> qsa_select_tokens(
    const std::vector<double> & complete_block_scores,
    const std::vector<std::size_t> & visible_token_indices,
    std::size_t token_budget = kQsaTokenBudget,
    std::size_t block_size = kQsaBlockSize);

struct GatedResidualRead {
    std::vector<double> normalized;
    std::vector<double> mix_weights;
    std::vector<double> mixed;
    std::vector<double> injection_weights;
};

// Repeat one hidden vector into the four independent residual streams.
std::vector<double> gated_residual_init(const std::vector<double> & input);

// Apply grouped RMSNorm and the down/SiLU/up read gate. norm_weight_delta is
// the zero-centred Transformers parameter, hence normalization multiplies by
// (1 + weight). An empty injection_matrix requests the final read-only mixer;
// otherwise it must contain four rows and returns 2*sigmoid(write_logits/4).
GatedResidualRead gated_residual_read(
    const std::vector<double> & hyper_input,
    std::size_t hidden_size,
    const std::vector<double> & norm_weight_delta,
    std::size_t low_rank,
    const std::vector<double> & down_matrix,
    const std::vector<double> & up_matrix,
    const std::vector<double> & injection_matrix,
    double epsilon = 1.0e-6);

// Scatter one block output into all four streams through the prepared write
// weights: residual[s,d] + block_output[d] * injection_weights[s].
std::vector<double> gated_residual_inject(
    const std::vector<double> & residual,
    const std::vector<double> & block_output,
    const std::vector<double> & injection_weights);

struct GdnScalarStep {
    double query = 0.0;
    double key = 0.0;
    double value = 0.0;
    double log_decay = 0.0;
    double beta = 0.0;
    double output_gate = 0.0;
};

struct GdnScalarTrace {
    std::vector<double> states;
    std::vector<double> core_outputs;
    std::vector<double> gated_outputs;
};

// Scalar specialization of the official recurrent gated-delta rule. Query and
// key receive the same epsilon-L2 normalization as the kernel path; core output
// then receives RMSNorm and Qwen4Exp's sigmoid output gate.
GdnScalarTrace gdn_scalar_recurrence(
    const std::vector<GdnScalarStep> & steps,
    double initial_state = 0.0,
    double norm_weight = 1.0,
    double l2_epsilon = 1.0e-6,
    double rms_epsilon = 1.0e-6);

}  // namespace dflash::qwen4exp::reference
