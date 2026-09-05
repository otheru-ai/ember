// Request-local target sampling for speculative verification.
// Local Ember fork extension. Algorithm: llama.cpp common/sampling.cpp,
// common_sampler_sample_and_accept_n at 6a1a922d269908a29cbd4b49c27e6a8e7fd10fae.
// Only selected output tokens enter history; proposals and KV replay never do.
#pragma once

#include "sampler.h"
#include "token_mask.h"
#include <functional>
#include <memory>
#include <utility>

namespace dflash::common {

inline int32_t sample_request_logits(float * logits, int vocab,
                                    const SamplerCfg & cfg,
                                    const std::vector<int32_t> & history,
                                    std::mt19937_64 & rng,
                                    TokenMask * mask,
                                    const std::function<bool()> & force_greedy) {
    if (!logits || vocab <= 0) return -1;
    if (mask && mask->active()) mask->apply(logits, vocab);
    if (cfg.needs_logit_processing() && !(force_greedy && force_greedy()))
        return sample_logits(logits, vocab, cfg, history, rng);
    int32_t best = 0;
    for (int i = 1; i < vocab; ++i)
        if (logits[i] > logits[best]) best = i;
    return best;
}

class SpeculativeSampler {
public:
    SpeculativeSampler(const SamplerCfg & cfg,
                       const std::vector<int32_t> & history,
                       std::mt19937_64 & rng,
                       std::shared_ptr<TokenMask> mask = {},
                       std::function<bool()> force_greedy = {})
        : cfg_(cfg), history_(history), rng_(rng), mask_(std::move(mask)),
          force_greedy_(std::move(force_greedy)) {}

    int32_t sample(float * logits, int vocab) {
        return sample_request_logits(logits, vocab, cfg_, history_, rng_,
                                     mask_.get(), force_greedy_);
    }

    // Call once per emitted token, including the seed before its deferred KV
    // write. Emit synchronously after accept, before selecting another token:
    // server structural-token hooks observe the output callback's state.
    void accept(int32_t token) {
        history_.push_back(token);
        if (mask_) mask_->accept(token);
    }

    const std::vector<int32_t> & history() const { return history_; }

private:
    const SamplerCfg cfg_;
    std::vector<int32_t> history_;
    std::mt19937_64 & rng_;
    std::shared_ptr<TokenMask> mask_;
    std::function<bool()> force_greedy_;
};

} // namespace dflash::common
