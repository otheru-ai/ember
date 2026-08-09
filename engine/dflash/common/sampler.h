// Shared CPU sampler chain used by both target arches.
//
// dflash::common daemon protocol embeds optional sampler params as a tail on each
// generate command: ` samp=temp,top_p,top_k,rep_pen,seed[,freq_pen,pres_pen]`.
// parse_sampler_token strips the tail in place and fills a SamplerCfg;
// sample_logits applies the chain:
//   rep_penalty -> freq/pres_penalty -> dry -> top_k -> softmax(temp)
//   -> top_p -> min_p -> draw.
//
// All backends (qwen35, qwen3, gemma4, laguna) include this header to keep
// sampling behaviour identical across arches.

#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

struct SamplerCfg {
    float    temp       = 0.0f;
    float    top_p      = 1.0f;
    int      top_k      = 0;
    float    min_p      = 0.0f;     // keep tokens with prob >= min_p * max_prob (0 = off)
    float    rep_pen    = 1.0f;     // multiplicative repetition penalty (HF-style)
    int      rep_window = 256;
    uint64_t seed       = 0;

    // OpenAI-style additive penalties (applied per-token to logits before softmax).
    // frequency_penalty: subtract freq_pen * count(token_in_history) from logit.
    // presence_penalty:  subtract pres_pen * 1(token_appeared_in_history) from logit.
    // Range: [-2.0, 2.0], default 0.0 (no effect).
    float    freq_pen   = 0.0f;
    float    pres_pen   = 0.0f;

    // DRY ("Don't Repeat Yourself") -- penalise a token in proportion to how
    // long a VERBATIM sequence it would extend, rather than how often it has
    // appeared. rep_pen/freq_pen cannot express this: they score a token by its
    // own history, so suppressing a loop with them also suppresses every
    // legitimate reuse of those tokens, which flattens prose and code alike.
    // DRY only bites once the model is genuinely replaying a span.
    //
    // Motivated by ember's progress watchdog being an outlier: ds4 has no such
    // watchdog, and llama.cpp/vLLM steer with penalties instead of aborting a
    // generation. Hard-aborting turns a quality problem into an availability
    // one -- production lost two agent turns to that on 2026-08-07.
    //
    // penalty = dry_multiplier * dry_base ^ (match_len - dry_allowed_length),
    // applied only when match_len >= dry_allowed_length. Defaults follow
    // llama.cpp (base 1.75, allowed length 2); multiplier 0 disables it.
    float    dry_multiplier      = 0.0f;   // 0 = off
    float    dry_base            = 1.75f;
    int      dry_allowed_length  = 2;      // matches this long are free
    int      dry_window          = 1024;   // lookback; <=0 = whole history
    // Token ids that terminate a match (newline, punctuation...). Left empty by
    // default: llama.cpp specifies breakers as TEXT, and resolving those to ids
    // needs a tokenizer the sampler does not have. Callers that know their
    // vocabulary can supply ids; empty means "no breakers", which is the
    // permissive choice and never rejects a legitimate token.
    std::vector<int32_t> dry_breaker_ids;

    // True when any logit modifier is active (penalties or stochastic sampling).
    // Backends should use CPU sample_logits() path when this returns true.
    bool needs_logit_processing() const {
        // NOTE: min_p is intentionally NOT here. It only affects the outcome when
        // sampling (temp>0, which already triggers this); at temp<=0 the argmax
        // is always kept, so including min_p would force the greedy fast path
        // (and DSpark) off for no behavioural change.
        //
        // dry_multiplier IS here, unlike min_p: DRY rewrites logits, so it can
        // change the argmax even at temp=0. Leaving it out would let the greedy
        // fast path silently ignore it.
        return temp > 0.0f || rep_pen != 1.0f || freq_pen != 0.0f ||
               pres_pen != 0.0f || dry_multiplier > 0.0f;
    }
};

// Returns the chosen token id. cfg.temp == 0 -> caller should use argmax;
// the chain assumes a positive temperature and falls back to a small floor.
int sample_logits(const float * logits_in,
                  int vocab,
                  const SamplerCfg & cfg,
                  const std::vector<int32_t> & history,
                  std::mt19937_64 & rng);

// Strip ` samp=...` tail from `line` (in place); return true when one was
// parsed. Out-of-band fields default to a permissive greedy-equivalent (top_p=1,
// top_k=0, rep_pen=1, seed=0).
bool parse_sampler_token(std::string & line, SamplerCfg & out);

}  // namespace dflash::common
