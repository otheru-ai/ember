#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "sampler.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (cond) {                                                          \
            ++g_pass;                                                        \
        } else {                                                             \
            ++g_fail;                                                        \
            std::printf("  FAIL: %s\n", msg);                              \
        }                                                                    \
    } while (0)

int main() {
    using dflash::common::SamplerCfg;
    using dflash::common::sample_logits;

    std::printf("ember sampler tests\n");
    const float logits[] = {10.0f, 9.0f};
    const std::vector<int32_t> repeated_second = {1};
    std::mt19937_64 rng(1);

    SamplerCfg cfg;
    cfg.temp = 0.0f;
    cfg.rep_window = 256;
    CHECK(!cfg.needs_logit_processing(), "neutral sampler uses argmax path");
    CHECK(sample_logits(logits, 2, cfg, repeated_second, rng) == 0,
          "neutral repetition penalty leaves argmax unchanged");

    cfg.rep_pen = 0.5f;
    CHECK(cfg.needs_logit_processing(),
          "sub-unit repetition penalty takes the logit-processing path");
    CHECK(sample_logits(logits, 2, cfg, repeated_second, rng) == 1,
          "sub-unit repetition penalty promotes a prior positive-logit token");

    cfg.rep_pen = 2.0f;
    const std::vector<int32_t> repeated_first = {0};
    CHECK(sample_logits(logits, 2, cfg, repeated_first, rng) == 1,
          "greater-than-one repetition penalty discourages a prior token");

    // ── DRY: penalise by the LENGTH of the verbatim span a token would extend,
    // not by how often it has appeared. Every case below picks greedily
    // (temp=0), so the chosen id is a direct read-out of which token DRY
    // demoted. Vocab is 8; A/B/C/D are ids 0..3.
    {
        enum { A = 0, B = 1, C = 2, D = 3, E = 4 };
        const int V = 8;
        auto pick = [&](const std::vector<float> &lg, const SamplerCfg &c,
                        const std::vector<int32_t> &hist) {
            std::mt19937_64 r(12345);
            SamplerCfg gc = c;
            gc.temp = 0.0f;
            return sample_logits(lg.data(), V, gc, hist, r);
        };
        std::vector<float> lg(V, 0.0f);
        lg[C] = 1.0f;
        lg[D] = 0.9f;
        const std::vector<int32_t> ab_c_ab{A, B, C, A, B};

        SamplerCfg off;
        off.dry_multiplier = 0.0f;
        CHECK(pick(lg, off, ab_c_ab) == C, "DRY off leaves the argmax alone");

        SamplerCfg on;
        on.dry_multiplier = 5.0f;
        on.dry_base = 1.75f;
        on.dry_allowed_length = 2;
        // Suffix "A,B" was previously followed by C, so C would extend a
        // 2-token verbatim span and must fall below D despite a higher logit.
        CHECK(pick(lg, on, ab_c_ab) == D,
              "DRY demotes the token that would replay a 2-token span");

        SamplerCfg raised = on;
        raised.dry_allowed_length = 3;
        CHECK(pick(lg, raised, ab_c_ab) == C,
              "allowed_length is a floor: a shorter match is untouched");

        // Mere recurrence is NOT repetition -- this is what separates DRY from
        // rep_pen. C occurs in history but the current suffix never preceded it.
        const std::vector<int32_t> recur{C, B, B, D, A};
        std::vector<float> unseen(V, 0.0f);
        unseen[C] = 1.0f;
        unseen[E] = 0.5f;
        CHECK(pick(unseen, on, recur) == C,
              "DRY ignores a token that merely recurs");
        SamplerCfg rep;
        rep.rep_pen = 10.0f;
        rep.rep_window = 256;
        CHECK(pick(unseen, rep, recur) == E,
              "rep_pen demotes it for occurring at all -- the collateral "
              "damage DRY avoids");

        // Penalty grows with match length: a long run beats a huge logit gap.
        std::vector<float> steep(V, 0.0f);
        steep[C] = 100.0f;
        steep[D] = 1.0f;   // unique runner-up; ties go to the lowest id
        SamplerCfg mild;
        mild.dry_multiplier = 1.0f;
        mild.dry_allowed_length = 2;
        CHECK(pick(steep, mild, std::vector<int32_t>(20, C)) == D,
              "penalty grows geometrically with match length");

        SamplerCfg brk = on;
        brk.dry_breaker_ids = {B};
        CHECK(pick(lg, brk, ab_c_ab) == C,
              "a sequence breaker terminates the match");

        SamplerCfg windowed = on;
        windowed.dry_window = 3;
        CHECK(pick(lg, windowed, ab_c_ab) == C,
              "dry_window bounds the lookback");

        SamplerCfg flat = on;
        flat.dry_base = 1.0f;
        CHECK(pick(lg, flat, ab_c_ab) == C, "base <= 1 is inert, not a crash");
        CHECK(pick(lg, on, {}) == C, "empty history is inert");
        CHECK(pick(lg, on, {A}) == C, "single-token history is inert");

        CHECK(!SamplerCfg().needs_logit_processing(),
              "default cfg needs no logit processing");
        SamplerCfg armed;
        armed.dry_multiplier = 1.0f;
        CHECK(armed.needs_logit_processing(),
              "DRY forces the CPU logit path, or the greedy/DSpark fast path "
              "would silently ignore it");

        // THE PRODUCTION CASE. A DSML tool-call marker is the same token
        // sequence every time, so without a breaker DRY scores a CORRECT marker
        // as a runaway and the model drifts to an ASCII lookalike -- the tool
        // then never fires. backend_dflash.cc registers the ｜DSML｜ delimiter
        // as a breaker; PIPE stands for it here.
        {
            const int PIPE = 5, TOOL = 6, CALLS = 7;
            std::vector<float> mk(V, 0.0f);
            mk[CALLS] = 1.0f;
            mk[D] = 0.9f;
            std::vector<int32_t> hist;
            for (int i = 0; i < 4; ++i) {
                hist.push_back(A); hist.push_back(PIPE);
                hist.push_back(TOOL); hist.push_back(CALLS);
            }
            hist.push_back(A); hist.push_back(PIPE); hist.push_back(TOOL);

            SamplerCfg unbroken;
            unbroken.dry_multiplier = 5.0f;
            unbroken.dry_allowed_length = 2;
            CHECK(pick(mk, unbroken, hist) == D,
                  "without a breaker DRY demotes a correct repeated marker "
                  "(reproduces the 2026-08-08 ASCII-marker regression)");

            SamplerCfg broken = unbroken;
            broken.dry_breaker_ids = {PIPE};
            CHECK(pick(mk, broken, hist) == CALLS,
                  "the delimiter breaker keeps the marker emittable however "
                  "often it has occurred");
        }
    }

    // ── DRY differential: optimized implementation vs a naive reference ──────
    //
    // The shipped DRY block is an OPTIMIZED rewrite -- loop-invariant breaker
    // test hoisted, an early-exit when a token cannot beat its recorded best,
    // collapsed bounds, and an apply loop that walks the match set instead of
    // the vocabulary. Each of those was justified by an argument, and the cases
    // above check specific scenarios. Neither is proof of EQUIVALENCE.
    //
    // So: state the semantics in the most obvious possible form, then assert
    // the two agree on randomized inputs. The reference is deliberately dumb --
    // O(win * match * vocab), every bound spelled out, no early exits -- because
    // its only job is to be trivially readable as "what DRY means".
    //
    // Inputs are biased toward the shapes that actually broke things: heavy
    // repetition (the degenerate case the early-exit collapses), breakers
    // present and absent, windows shorter than the history, and allowed_length
    // spanning 0..4.
    {
        auto reference_dry = [](const SamplerCfg &cfg,
                                const std::vector<int32_t> &history,
                                int vocab) {
            std::vector<float> pen(static_cast<size_t>(vocab), 0.0f);
            if (!(cfg.dry_multiplier > 0.0f) || !(cfg.dry_base > 1.0f) ||
                history.size() <= 1) {
                return pen;
            }
            const size_t n = history.size();
            const size_t win = cfg.dry_window > 0
                ? std::min(n, static_cast<size_t>(cfg.dry_window)) : n;
            const size_t from = n - win;
            const size_t kMaxMatch = 64;
            auto is_brk = [&](int32_t t) {
                for (int32_t b : cfg.dry_breaker_ids) if (b == t) return true;
                return false;
            };
            std::vector<int> best(static_cast<size_t>(vocab), 0);
            for (size_t j = from + 1; j < n; ++j) {
                const int z = history[j];
                if (is_brk(z)) continue;
                size_t k = 0;
                while (k < kMaxMatch && j >= 1 + k && n >= 1 + k &&
                       (j - 1 - k) >= from &&
                       history[j - 1 - k] == history[n - 1 - k] &&
                       !is_brk(history[n - 1 - k])) {
                    ++k;
                }
                if (k == 0) continue;
                if (z >= 0 && z < vocab &&
                    static_cast<size_t>(best[static_cast<size_t>(z)]) < k) {
                    best[static_cast<size_t>(z)] = static_cast<int>(k);
                }
            }
            const int allowed =
                cfg.dry_allowed_length > 0 ? cfg.dry_allowed_length : 0;
            for (int z = 0; z < vocab; ++z) {
                const int len = best[static_cast<size_t>(z)];
                if (len < allowed || len == 0) continue;
                pen[static_cast<size_t>(z)] =
                    cfg.dry_multiplier *
                    std::pow(cfg.dry_base, static_cast<float>(len - allowed));
            }
            return pen;
        };

        const int V = 12;
        std::mt19937_64 drng(0xD2ULL);
        int cases = 0, agreed = 0;
        for (int trial = 0; trial < 400; ++trial) {
            // Alphabet 2..5 so repetition is common; length up to 80.
            const int alpha = 2 + (int)(drng() % 4);
            const size_t len = 2 + (size_t)(drng() % 78);
            std::vector<int32_t> hist(len);
            for (size_t i = 0; i < len; ++i)
                hist[i] = (int32_t)(drng() % (uint64_t)alpha);
            // Every ~4th trial is a near-period-1 run: the degenerate shape.
            if (trial % 4 == 0)
                for (size_t i = 0; i < len; ++i)
                    if (drng() % 8) hist[i] = 1;

            SamplerCfg c;
            c.dry_multiplier = 1.0f + (float)(drng() % 5);
            c.dry_base = 1.25f + (float)(drng() % 3) * 0.5f;
            c.dry_allowed_length = (int)(drng() % 5);
            c.dry_window = (int)(drng() % 3) == 0 ? -1 : (int)(1 + drng() % 40);
            if (drng() % 2) c.dry_breaker_ids = {(int32_t)(drng() % (uint64_t)alpha)};

            const std::vector<float> want = reference_dry(c, hist, V);

            // Drive the real implementation: flat logits, greedy, so the
            // returned penalties are observable as the resulting logit vector.
            // sample_logits mutates a copy internally, so recover the applied
            // penalty by comparing argmax behaviour across a probe per token.
            // Simpler and exact: give token t a large unique lead and see how
            // much lead it must have to still win.
            bool ok = true;
            for (int t = 0; t < V && ok; ++t) {
                if (want[(size_t)t] == 0.0f) continue;
                // logit(t) = want[t] - eps means t must LOSE to a 0-penalty
                // token at logit 0; +eps means it must WIN.
                for (int side = 0; side < 2 && ok; ++side) {
                    // RELATIVE epsilon. An absolute one is swallowed by
                    // float32 precision once the penalty is large: at
                    // want=5.5e7 the ULP exceeds 0.05, so want+eps == want and
                    // the probe degenerates into an id-ordered tie. Measured:
                    // 8/400 spurious failures, all with want > 2e6.
                    const float mag = std::fabs(want[(size_t)t]);
                    const float eps = mag > 1.0f ? mag * 1e-3f : 0.05f;
                    std::vector<float> lg((size_t)V, -1000.0f);
                    lg[(size_t)t] = want[(size_t)t] + (side ? eps : -eps);
                    // A neutral rival that DRY never penalises: pick a token id
                    // absent from history (so it has no match).
                    int rival = -1;
                    for (int r = 0; r < V; ++r)
                        if (want[(size_t)r] == 0.0f &&
                            std::find(hist.begin(), hist.end(), (int32_t)r) == hist.end()) {
                            rival = r; break;
                        }
                    if (rival < 0) break;   // no neutral rival available
                    lg[(size_t)rival] = 0.0f;
                    std::mt19937_64 r2(7);
                    SamplerCfg gc = c;
                    gc.temp = 0.0f;
                    const int got = sample_logits(lg.data(), V, gc, hist, r2);
                    const int expect = side ? t : rival;
                    if (got != expect) {
                        ok = false;
                        std::printf("    [diag] trial=%d t=%d want=%.6g "
                                    "eps=%.3g side=%d got=%d expect=%d "
                                    "mult=%.3g base=%.3g allow=%d win=%d "
                                    "hist=%zu brk=%zu\n",
                                    trial, t, (double)want[(size_t)t], (double)eps,
                                    side, got, expect, (double)c.dry_multiplier,
                                    (double)c.dry_base, c.dry_allowed_length,
                                    c.dry_window, hist.size(),
                                    c.dry_breaker_ids.size());
                    }
                }
            }
            ++cases;
            if (ok) ++agreed;
        }
        CHECK(cases == 400, "differential ran every trial");
        CHECK(agreed == cases,
              "optimized DRY agrees with the naive reference on 400 randomized "
              "histories (repetition-heavy, breakers, windows, allowed_length)");
        std::printf("  [dry differential] %d/%d trials agreed\n", agreed, cases);
    }

    std::printf("──────────────────────────────\n  %d passed, %d failed\n",
                g_pass, g_fail);
    return g_fail ? 1 : 0;
}
