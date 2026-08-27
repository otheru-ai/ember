#pragma once

#include "dflash_target.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// Only the state consumed by the shared DSpark projection head. Keeping this
// contract narrow prevents an unrelated legacy draft-model object graph from
// leaking into the DeepSeek4 backend.
struct DSparkHeadWeights {
    bool enabled = false;
    int n_embd = 0;
    int markov_rank = 0;
    int vocab_size = 0;
    int confidence_dim = 0;
    ggml_tensor * markov_w1 = nullptr;
    ggml_tensor * markov_w2 = nullptr;
    ggml_tensor * confidence_w = nullptr;
    ggml_tensor * confidence_b = nullptr;
};

// Release thread-local graph allocators while their owning GPU backend is
// still alive. Safe to call repeatedly on the generation thread.
void reset_dspark_head_runtime_cache();

bool dspark_markov_correct_greedy_chain(const DSparkHeadWeights & weights,
                                        ggml_backend_t backend,
                                        DFlashTarget & target,
                                        const float * local_hidden,
                                        int q_len,
                                        int32_t last_tok,
                                        float confidence_threshold,
                                        std::vector<int32_t> & draft_tok);

// Fused variant: base logits (one lm_head matmul over all candidates) +
// unrolled Markov correction chain + in-graph argmax feeding the next
// step's get_rows, all in ONE graph on the draft backend. No host logits
// round-trip. When confidence_out is non-null and the checkpoint has a
// compatible confidence head, returns one score per candidate from the same
// graph and host synchronization as the token ids. `confidence_hidden`, when
// non-null, has the same padded layout as `local_hidden` and supplies the
// pre-output-norm state expected by the confidence head. Callers without a
// separate state retain the legacy behavior by leaving it null.
bool dspark_markov_correct_greedy_chain_fused(const DSparkHeadWeights & weights,
                                              ggml_backend_t backend,
                                              ggml_tensor * lm_head,
                                              const float * local_hidden,
                                              int q_len,
                                              int32_t last_tok,
                                              std::vector<int32_t> & draft_tok,
                                              std::vector<float> * confidence_out = nullptr,
                                              const float * confidence_hidden = nullptr);

}  // namespace dflash::common
