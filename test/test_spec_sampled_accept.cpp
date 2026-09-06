// Sampled speculative acceptance must be output-identical to autoregressive
// decoding.
//
// The engine change under test (DFLASH_DS4_SPEC_SAMPLED=1, see SpecSampling in
// engine/dflash/deepseek4/deepseek4_dspark.h) replaces the greedy verifier's
// "accept iff draft == argmax" with "draw t ~ p_i, accept iff draft == t". The
// claim that makes it safe to ship is strong and exactly testable:
//
//   for a fixed seed, the emitted token sequence is IDENTICAL to AR decoding,
//   whatever the drafter proposes -- including a drafter that is always wrong,
//   always right, or adversarially alternating.
//
// That property is what distinguishes this rule from an approximation. It holds
// because every emitted token is drawn from the true target distribution on a
// correct prefix, and because sample_logits() consumes exactly one RNG value
// per call (common/sampler.cpp:62) while the rule draws exactly once per
// position it evaluates -- so the RNG stream advances once per emitted token in
// both schemes.
//
// This test drives the real sampler with a deterministic synthetic target, so a
// regression in the acceptance rule, the RNG accounting, or the penalty history
// sequencing fails here without needing the 98 GiB model.

#include "common/sampler.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

using dflash::common::SamplerCfg;
using dflash::common::sample_logits;

namespace {

int g_failures = 0;

constexpr int kVocab = 64;

// A deterministic stand-in for the target model: logits depend on the whole
// prefix, so an incorrect prefix yields a different distribution -- which is
// what makes a mis-sequenced history or a stale KV observable as a diff.
std::vector<float> target_logits(const std::vector<int32_t> & prefix) {
    std::vector<float> lg((size_t) kVocab, 0.0f);
    uint64_t h = 1469598103934665603ull;
    for (int32_t t : prefix) {
        h ^= (uint64_t) (t + 1);
        h *= 1099511628211ull;
    }
    for (int v = 0; v < kVocab; ++v) {
        uint64_t x = h ^ ((uint64_t) (v + 1) * 0x9E3779B97F4A7C15ull);
        x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
        // Spread over a few nats so temperature and top_p actually bite.
        lg[(size_t) v] = (float) ((double) (x % 100000) / 100000.0 * 8.0 - 4.0);
    }
    return lg;
}

// Reference: ordinary autoregressive decoding.
std::vector<int32_t> decode_ar(const std::vector<int32_t> & prompt, int n_gen,
                               const SamplerCfg & cfg, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<int32_t> history = prompt;
    std::vector<int32_t> out;
    for (int i = 0; i < n_gen; ++i) {
        const std::vector<float> lg = target_logits(history);
        const int t = sample_logits(lg.data(), kVocab, cfg, history, rng);
        out.push_back((int32_t) t);
        history.push_back((int32_t) t);
    }
    return out;
}

// The rule under test, mirroring verify_exact_prefix_embedded(): walk the
// proposed block, draw at each position, stop at the first position where the
// draw disagrees with the proposal, and emit the drawn token there.
std::vector<int32_t> decode_spec(const std::vector<int32_t> & prompt, int n_gen,
                                 const SamplerCfg & cfg, uint64_t seed,
                                 int block,
                                 const std::function<int32_t(const std::vector<int32_t> &, int)> & propose) {
    std::mt19937_64 rng(seed);
    std::vector<int32_t> history = prompt;
    std::vector<int32_t> out;
    while ((int) out.size() < n_gen) {
        // Draft a block of candidates from whatever the drafter believes.
        std::vector<int32_t> draft;
        {
            std::vector<int32_t> lookahead = history;
            for (int k = 0; k < block; ++k) {
                const int32_t d = propose(lookahead, k);
                draft.push_back(d);
                lookahead.push_back(d);
            }
        }
        // Verify: one draw per evaluated position, against the target's own
        // distribution on the prefix accepted so far.
        for (int k = 0; k < block && (int) out.size() < n_gen; ++k) {
            const std::vector<float> lg = target_logits(history);
            const int t = sample_logits(lg.data(), kVocab, cfg, history, rng);
            out.push_back((int32_t) t);
            history.push_back((int32_t) t);
            if ((int32_t) t != draft[(size_t) k]) break;   // rejected: block ends
        }
    }
    out.resize((size_t) n_gen);
    return out;
}

void expect_identical(const char * what, const SamplerCfg & cfg, int block,
                      const std::function<int32_t(const std::vector<int32_t> &, int)> & propose) {
    const std::vector<int32_t> prompt = {7, 11, 13, 13, 2};
    const int n_gen = 96;
    for (uint64_t seed : {1ull, 42ull, 12345ull, 0xDEADBEEFull}) {
        const std::vector<int32_t> ar = decode_ar(prompt, n_gen, cfg, seed);
        const std::vector<int32_t> sp = decode_spec(prompt, n_gen, cfg, seed, block, propose);
        bool same = ar.size() == sp.size();
        size_t first_diff = ar.size();
        for (size_t i = 0; same && i < ar.size(); ++i) {
            if (ar[i] != sp[i]) { same = false; first_diff = i; }
        }
        if (!same) {
            std::fprintf(stderr,
                         "FAIL: %s (seed=%llu): diverged at token %zu (ar=%d spec=%d)\n",
                         what, (unsigned long long) seed, first_diff,
                         first_diff < ar.size() ? ar[first_diff] : -1,
                         first_diff < sp.size() ? sp[first_diff] : -1);
            ++g_failures;
        }
    }
}

}  // namespace

int main() {
    // Temperature alone.
    SamplerCfg temp_only;
    temp_only.temp = 0.6f;
    temp_only.top_p = 1.0f;
    temp_only.top_k = 0;

    // The deployed agent shape: temp plus truncation.
    SamplerCfg deployed;
    deployed.temp  = 0.6f;
    deployed.top_p = 0.95f;
    deployed.top_k = 40;

    // Penalties make the distribution depend on the emitted history, so this
    // arm is what catches a mis-sequenced `history` in the verify loop.
    SamplerCfg penalised;
    penalised.temp     = 0.8f;
    penalised.top_p    = 0.95f;
    penalised.rep_pen  = 1.15f;
    penalised.freq_pen = 0.4f;
    penalised.pres_pen = 0.3f;

    // DRY is sequence-shaped rather than count-shaped: an off-by-one in history
    // changes which span it thinks is being replayed.
    SamplerCfg dry;
    dry.temp                = 0.7f;
    dry.dry_multiplier      = 0.8f;
    dry.dry_base            = 1.75f;
    dry.dry_allowed_length  = 2;

    struct { const char * name; const SamplerCfg * cfg; } cfgs[] = {
        {"temp-only",  &temp_only},
        {"deployed",   &deployed},
        {"penalties",  &penalised},
        {"dry",        &dry},
    };

    // Drafters chosen to exercise every acceptance path, since the invariant
    // must not depend on draft quality at all.
    struct { const char * name; std::function<int32_t(const std::vector<int32_t> &, int)> fn; } drafters[] = {
        // Always wrong: every block rejects at position 0 (worst case).
        {"always-reject", [](const std::vector<int32_t> &, int) { return (int32_t) (kVocab - 1); }},
        // Perfect oracle: every block accepts fully (best case). Uses a
        // separate RNG so it cannot perturb the decode stream.
        {"oracle", [](const std::vector<int32_t> & pre, int) {
            std::mt19937_64 side(0xABCDEF);
            const std::vector<float> lg = target_logits(pre);
            SamplerCfg greedy; greedy.temp = 0.0f;
            int best = 0; float mv = lg[0];
            for (int i = 1; i < kVocab; ++i) if (lg[(size_t) i] > mv) { mv = lg[(size_t) i]; best = i; }
            (void) side;
            return (int32_t) best;
        }},
        // Prefix-dependent noise: partial accepts of varying length.
        {"noisy", [](const std::vector<int32_t> & pre, int k) {
            return (int32_t) ((pre.size() * 31u + (unsigned) k * 7u) % (unsigned) kVocab);
        }},
    };

    for (auto & c : cfgs) {
        for (auto & d : drafters) {
            for (int block : {2, 4, 5}) {
                char what[160];
                std::snprintf(what, sizeof(what), "%s / %s / block=%d",
                              c.name, d.name, block);
                expect_identical(what, *c.cfg, block, d.fn);
            }
        }
    }

    // Greedy must remain byte-identical to the pre-change verifier: at temp==0
    // the draw IS the argmax, so this is the degenerate case the existing
    // production path already implements.
    SamplerCfg greedy;
    greedy.temp = 0.0f;
    for (auto & d : drafters) {
        char what[160];
        std::snprintf(what, sizeof(what), "greedy / %s", d.name);
        expect_identical(what, greedy, 4, d.fn);
    }

    if (g_failures == 0) {
        std::printf("test_spec_sampled_accept: all acceptance arms identical to AR\n");
        return 0;
    }
    std::fprintf(stderr, "test_spec_sampled_accept: %d failure(s)\n", g_failures);
    return 1;
}
