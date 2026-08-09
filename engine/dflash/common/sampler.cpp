// Shared CPU sampler chain. See sampler.h for the protocol overview.

#include "sampler.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifdef DFLASH27B_HAVE_GPU_SAMPLER
#include "geometric_sampler_cuda.h"
#endif

namespace dflash::common {

namespace {

// Binary search (quickselect via std::nth_element) for the smallest index
// `cut` such that the descending-by-value prefix cand[0,cut) has cumulative
// mass >= target, where "mass" of an element is given by `mass_of`. At each
// level, partitioning [lo,hi) at its midpoint puts exactly the top (mid-lo)
// elements of that range into [lo,mid) (in some order) — whichever half still
// contains the boundary is recursed into and the other is discarded. Mutates
// `cand` in place; only the final small base-case range ends up sorted, since
// order doesn't matter for the caller's draw, only which elements make the
// cut. Each level's cost is proportional to its (shrinking) range, so total
// work is O(cand.size()), not O(cand.size() log cand.size()) like a full sort,
// regardless of where the cutoff lands.
template <typename MassFn>
size_t nucleus_cutoff(std::vector<std::pair<float, int>> & cand, double target, MassFn mass_of) {
    constexpr size_t kBaseCase = 64;
    size_t lo = 0, hi = cand.size();
    while (hi - lo > kBaseCase) {
        const size_t mid = lo + (hi - lo) / 2;
        std::nth_element(cand.begin() + lo, cand.begin() + mid, cand.begin() + hi,
                         [](auto & a, auto & b){ return a.first > b.first; });
        double mass = 0.0;
        for (size_t i = lo; i < mid; i++) mass += mass_of(cand[i]);
        if (mass >= target) {
            hi = mid;        // cutoff lies within [lo, mid)
        } else {
            target -= mass;  // [lo, mid) fully included; keep searching [mid, hi)
            lo = mid;
        }
    }
    // Base case: small enough range left, sort it and walk the exact cumulative
    // cutoff (cand[0, lo) is already fully confirmed included from above).
    std::sort(cand.begin() + lo, cand.begin() + hi,
             [](auto & a, auto & b){ return a.first > b.first; });
    size_t cut = hi;
    double cum = 0.0;
    for (size_t i = lo; i < hi; i++) {
        cum += mass_of(cand[i]);
        if (cum >= target) { cut = i + 1; break; }
    }
    return cut;
}

// Draws a token from `cand`, whose .first fields are proportional
// probabilities (need not already sum to 1 or be sorted). `r_uniform` is a
// pre-drawn uniform in [0,1) supplied by the caller (drawn once per
// sample_logits call) so every path — GPU, GPU-assisted top_p, or CPU —
// consumes the same single RNG value.
int draw_from_weights(const std::vector<std::pair<float, int>> & cand, double r_uniform) {
    if (cand.empty()) return -1;
    double Z = 0.0;
    for (auto & c : cand) {
        if (std::isfinite(c.first) && c.first > 0.0f) Z += c.first;
    }
    if (!(Z > 0.0) || !std::isfinite(Z)) return cand.front().second;
    const double r = r_uniform * Z;
    double acc = 0.0;
    for (auto & c : cand) {
        if (std::isfinite(c.first) && c.first > 0.0f) acc += c.first;
        if (r <= acc) return c.second;
    }
    return cand.back().second;
}

static bool trace_sampler_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("DFLASH_TRACE_SAMPLER");
        return value && value[0] && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// Off-by-default post-filter trace. Production's top_k=40 configuration takes
// this CPU path, so these are the exact final weights used by its inverse-CDF
// draw—not reconstructed logits or an approximation of a device-side result.
static void trace_sampler_choice(
        const std::vector<std::pair<float, int>> & weighted,
        int chosen, const SamplerCfg & cfg, double r_uniform,
        const char * stage) {
    if (!trace_sampler_enabled()) return;
    std::vector<std::pair<float, int>> top = weighted;
    const size_t count = std::min<size_t>(10, top.size());
    std::partial_sort(top.begin(), top.begin() + count, top.end(),
                      [](const auto &a, const auto &b) {
                          if (a.first != b.first) return a.first > b.first;
                          return a.second < b.second;
                      });
    float chosen_weight = 0.0f;
    for (const auto &candidate : weighted) {
        if (candidate.second == chosen) {
            chosen_weight = candidate.first;
            break;
        }
    }
    std::fprintf(stderr,
                 "[dflash-sampler] stage=%s chosen=%d weight=%.9g "
                 "rng=%.17g temp=%.9g top_p=%.9g top_k=%d min_p=%.9g "
                 "rep_pen=%.9g rep_window=%d freq_pen=%.9g pres_pen=%.9g "
                 "candidates=%zu top=",
                 stage, chosen, chosen_weight, r_uniform, cfg.temp, cfg.top_p,
                 cfg.top_k, cfg.min_p, cfg.rep_pen, cfg.rep_window,
                 cfg.freq_pen, cfg.pres_pen, weighted.size());
    for (size_t i = 0; i < count; ++i) {
        std::fprintf(stderr, "%s%d:%.9g", i ? "," : "",
                     top[i].second, top[i].first);
    }
    std::fprintf(stderr, "\n");
}

#ifdef DFLASH27B_HAVE_GPU_SAMPLER
// Given probabilities the GPU already computed (penalties + softmax(temp)
// applied, summing to ~1) for a pure top_p (no top_k) config, find the
// nucleus and draw. Skips all exp()/Z bookkeeping the raw-logit path needs,
// since the input is already normalized.
//
// Only worth calling for top_p without top_k: top_k's CPU cost is already
// cheap (partial_sort scales with k, not vocab — measured ~270-300us at
// vocab=151936), so a GPU round trip (kernel + D2H copy, ~500-800us) makes it
// slower, not faster. top_p's CPU cost without top_k is dominated by
// nucleus_cutoff's O(vocab) std::nth_element passes regardless of who
// computed the input probabilities, so skipping the CPU-side exp() pass here
// is a net win (measured ~1.4x faster end-to-end at vocab=151936).
int sample_from_gpu_probs(std::vector<float> & probs, double top_p, double r_uniform) {
    std::vector<std::pair<float, int>> cand(probs.size());
    for (size_t i = 0; i < probs.size(); i++) cand[i] = {probs[i], (int)i};

    double Z = 0.0;
    for (auto & c : cand) Z += c.first;
    const double target = top_p * Z;
    const size_t cut = nucleus_cutoff(cand, target, [](auto & c){ return (double)c.first; });
    cand.resize(cut);

    return draw_from_weights(cand, r_uniform);
}
#endif

}  // namespace

int sample_logits(const float * logits_in,
                  int vocab,
                  const SamplerCfg & cfg,
                  const std::vector<int32_t> & history,
                  std::mt19937_64 & rng) {
    if (!logits_in || vocab <= 0) return -1;

    // Draw the single uniform up front, exactly once, and only when we will
    // actually sample (temp>0; greedy returns before any draw). It is then
    // threaded through every path below — the full-GPU draw, the GPU-assisted
    // top_p draw, and the CPU draw all consume this same value — so the RNG
    // stream advances identically no matter which path resolves the token
    // (including a GPU→CPU fallback), keeping decode reproducible whether the
    // GPU sampler is on or off.
    double r_uniform = 0.0;
    if (cfg.temp > 0.0f) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        r_uniform = u(rng);
    }

#ifdef DFLASH27B_HAVE_GPU_SAMPLER
    // GPU path (on by default; set DFLASH_GPU_SAMPLE=0 to disable). top_k>0,
    // top_p in (0,1) (both unsupported on the GPU, see geometric_sampler_cuda.h),
    // and any CUDA error return -1 and fall through to the CPU chain below.
    if (gpu_sampler_enabled() && gpu_sampler_supports(cfg)) {
        const int g = geometric_sample_logits_cuda(logits_in, vocab, cfg, history, r_uniform,
                                         /*logits_on_device=*/false);
        if (g >= 0) return g;
    }
#endif
    const bool need_top_k = cfg.top_k > 0 && cfg.top_k < vocab;
    const bool need_top_p = cfg.top_p > 0.0f && cfg.top_p < 1.0f;

#ifdef DFLASH27B_HAVE_GPU_SAMPLER
    // Reaching here means the GPU either can't fully handle this config
    // (top_k/top_p above) or is disabled. For pure top_p (no top_k), the GPU
    // can still compute the shared, vocab-wide penalty+softmax prefix and
    // hand back normalized probabilities, letting the CPU skip straight to
    // nucleus_cutoff — worth it because that search's O(vocab) cost dominates
    // regardless of who computed its input. top_k (with or without top_p) is
    // deliberately excluded: its CPU cost is already cheap (partial_sort
    // scales with k, not vocab), so the GPU round trip would make it slower,
    // not faster (measured regression, not just "no win").
    if (cfg.temp > 0.0f && need_top_p && !need_top_k && cfg.min_p <= 0.0f && gpu_sampler_enabled()) {
        std::vector<float> gpu_probs(vocab);
        if (geometric_compute_probs_cuda(logits_in, vocab, cfg, history,
                                         gpu_probs.data(), /*logits_on_device=*/false)) {
            return sample_from_gpu_probs(gpu_probs, cfg.top_p, r_uniform);
        }
        // else: fall through to the full CPU chain below.
    }
#endif

    std::vector<std::pair<float, int>> cand(vocab);
    bool any_usable_logit = false;
    for (int i = 0; i < vocab; i++) {
        float logit = logits_in[i];
        if (std::isnan(logit)) logit = -INFINITY;
        else any_usable_logit = true;
        cand[i] = {logit, i};
    }
    if (!any_usable_logit) return 0;

    // Multiplicative repetition penalty (HuggingFace-style).
    // HuggingFace accepts every positive value: >1 discourages prior tokens,
    // while (0,1) promotes them. Lucebox checked only >1 even though its parser
    // accepted (0,1), making those valid request/card values silently inert.
    if (cfg.rep_pen != 1.0f && !history.empty()) {
        const size_t win = cfg.rep_window > 0
            ? std::min(history.size(), static_cast<size_t>(cfg.rep_window))
            : 0;
        const size_t from = history.size() - win;
        std::unordered_set<int> seen;
        for (size_t i = from; i < history.size(); i++) seen.insert(history[i]);
        for (auto & c : cand) {
            if (seen.count(c.second)) {
                c.first = (c.first > 0.0f) ? c.first / cfg.rep_pen
                                           : c.first * cfg.rep_pen;
            }
        }
    }

    // OpenAI-style additive frequency and presence penalties.
    if ((cfg.freq_pen != 0.0f || cfg.pres_pen != 0.0f) && !history.empty()) {
        const size_t win = cfg.rep_window > 0
            ? std::min(history.size(), static_cast<size_t>(cfg.rep_window))
            : 0;
        const size_t from = history.size() - win;
        std::unordered_map<int, int> counts;
        for (size_t i = from; i < history.size(); i++) counts[history[i]]++;
        for (auto & c : cand) {
            auto it = counts.find(c.second);
            if (it != counts.end()) {
                c.first -= cfg.freq_pen * static_cast<float>(it->second);
                c.first -= cfg.pres_pen;
            }
        }
    }

    // DRY: penalise tokens by the LENGTH of the verbatim span they would
    // extend, not by how often they have occurred.
    //
    // For a candidate z, find the longest k such that the current suffix
    // history[n-k .. n-1] followed by z already occurred earlier. Every
    // position j < n is a possible earlier occurrence: match backwards from
    // j-1 and n-1 for as long as the tokens agree, and whatever token sits at
    // history[j] is the one that continuation would pick. Take the longest such
    // match per token.
    //
    // Worked through with concrete values in test_sampler_dry.cpp.
    //
    // Overlapping matches are intentional: a period-1 run [A,A,A,A] yields
    // len 3 for A, which is precisely the runaway this is meant to damp.
    if (cfg.dry_multiplier > 0.0f && cfg.dry_base > 1.0f && history.size() > 1) {
        const size_t n = history.size();
        const size_t win = cfg.dry_window > 0
            ? std::min(n, static_cast<size_t>(cfg.dry_window))
            : n;
        const size_t from = n - win;
        // Bounds the O(win * kMaxMatch) scan. Beyond this the penalty is already
        // astronomically large (1.75^62), so a longer match changes nothing.
        constexpr size_t kMaxMatch = 64;

        // Linear over a vector, not a hash set: the breaker list is one or two
        // protocol ids in practice, so a compare against a register beats a
        // hash + modulo + bucket load, and it costs no allocation per token.
        const auto is_breaker = [&cfg](int32_t t) {
            for (int32_t b : cfg.dry_breaker_ids) if (b == t) return true;
            return false;
        };

        // The suffix side of the comparison does not depend on j, so the
        // breaker test is loop-invariant: resolve it once into the furthest
        // any match may reach, rather than re-testing the same <=64 tokens on
        // every one of the ~win outer iterations.
        size_t suffix_limit = 0;
        while (suffix_limit < kMaxMatch && suffix_limit < n &&
               !is_breaker(history[n - 1 - suffix_limit])) {
            ++suffix_limit;
        }

        // token -> longest verbatim match it would extend. Reused across calls
        // to keep the hot path allocation-free; `touched` bounds the clear to
        // what was actually written.
        static thread_local std::unordered_map<int, int> repeat_len;
        static thread_local std::vector<int> touched;
        repeat_len.clear();
        touched.clear();

        for (size_t j = from + 1; j < n; ++j) {
            const int z = history[j];
            if (is_breaker(z)) continue;
            // `j - from` is the only bound needed: the for-loop guarantees
            // j >= from+1, so it cannot underflow, and it subsumes both the
            // old j >= 1+k and (j-1-k) >= from tests. The old n >= 1+k was
            // dead — the window arithmetic already bounds n-1-k.
            const size_t limit = std::min(suffix_limit, j - from);
            // A j that cannot beat what this token already has is a no-op,
            // because only the maximum is kept. Skipping it collapses the
            // degenerate case this feature exists for: in a period-1 run every
            // j carries the same z, so all but the first few exit here.
            auto it = repeat_len.find(z);
            const size_t best = (it == repeat_len.end())
                ? 0 : static_cast<size_t>(it->second);
            if (best >= limit) continue;

            size_t k = 0;
            while (k < limit && history[j - 1 - k] == history[n - 1 - k]) ++k;
            if (k == 0 || k <= best) continue;
            if (it == repeat_len.end()) touched.push_back(z);
            repeat_len[z] = static_cast<int>(k);
        }

        const int allowed = cfg.dry_allowed_length > 0
            ? cfg.dry_allowed_length : 0;
        // Walk the matches, not the vocabulary. `cand` is still identity-indexed
        // here (built as cand[i]={logit,i}; the first reorder is the partial_sort
        // below), so cand[z] is token z. The old form did a hash lookup for every
        // one of ~129k candidates to find at most a few hundred hits.
        const int vocab_n = vocab;
        for (int z : touched) {
            const int len = repeat_len[z];
            if (len < allowed) continue;
            if (z < 0 || z >= vocab_n) continue;
            const float exponent = static_cast<float>(len - allowed);
            cand[static_cast<size_t>(z)].first -=
                cfg.dry_multiplier * std::pow(cfg.dry_base, exponent);
        }
    }

    // temp=0 → deterministic argmax (after penalties have been applied above).
    // Independent of top_k/top_p (the single highest-logit token is always the
    // answer), so this skips sorting/truncation entirely: an O(vocab) max scan
    // beats the O(vocab log vocab) sort below. Ties go to the lowest token id
    // (max_element returns the first of equal maxima), matching the GPU kernel.
    if (cfg.temp <= 0.0f) {
        const int chosen = std::max_element(
            cand.begin(), cand.end(),
            [](auto & a, auto & b){ return a.first < b.first; })->second;
        trace_sampler_choice(cand, chosen, cfg, r_uniform, "greedy");
        return chosen;
    }

    // Only sort/truncate when top_k or top_p actually need it. Softmax and the
    // inverse-CDF draw below are order-independent, so the common case (plain
    // temperature sampling, no truncation) skips sorting entirely.
    if (need_top_k) {
        std::partial_sort(cand.begin(), cand.begin() + cfg.top_k, cand.end(),
                          [](auto & a, auto & b){ return a.first > b.first; });
        cand.resize(cfg.top_k);
    }

    const float inv_t = 1.0f / std::max(1e-3f, cfg.temp);
    const float maxv_logit = need_top_k
        ? cand.front().first
        : std::max_element(cand.begin(), cand.end(),
                           [](auto & a, auto & b){ return a.first < b.first; })->first;
    if (std::isinf(maxv_logit) && maxv_logit > 0.0f) {
        std::vector<std::pair<float, int>> infinite;
        for (const auto & c : cand)
            if (std::isinf(c.first) && c.first > 0.0f)
                infinite.push_back({1.0f, c.second});
        return draw_from_weights(infinite, r_uniform);
    }
    if (!std::isfinite(maxv_logit)) return cand.front().second;
    const float maxv = maxv_logit * inv_t;

    if (need_top_p && !need_top_k) {
        // Nucleus cutoff over the full (untruncated) vocab. Z is the true
        // full-vocab softmax denominator (one O(vocab) exp() pass, needed
        // regardless to know the absolute mass threshold).
        double Z = 0.0;
        for (auto & c : cand) Z += std::exp((double)c.first * inv_t - maxv);
        const double target = (double)cfg.top_p * Z;
        const size_t cut = nucleus_cutoff(cand, target, [&](auto & c) {
            return std::exp((double)c.first * inv_t - maxv);
        });
        cand.resize(cut);
    }

    double Z = 0.0;
    std::vector<float> probs(cand.size());
    for (size_t i = 0; i < cand.size(); i++) {
        probs[i] = std::exp(cand[i].first * inv_t - maxv);
        Z       += probs[i];
    }
    if (!(Z > 0.0) || !std::isfinite(Z)) {
        return std::max_element(cand.begin(), cand.end(),
                                [](auto & a, auto & b){ return a.first < b.first; })->second;
    }
    for (auto & p : probs) p = (float)(p / Z);

    // top_k+top_p combined: cut the already top_k-truncated (and thus
    // already-sorted) subset further to the top_p nucleus within it.
    if (need_top_p && need_top_k) {
        double cum = 0.0;
        size_t cut = probs.size();
        for (size_t i = 0; i < probs.size(); i++) {
            cum += probs[i];
            if (cum >= cfg.top_p) { cut = i + 1; break; }
        }
        probs.resize(cut); cand.resize(cut);
    }

    // min_p: keep only tokens with prob >= min_p * max_prob (relative floor).
    // Zeroed weights are simply never drawn — draw_from_weights renormalizes
    // over the remaining mass. Applied last so it composes with top_k/top_p.
    if (cfg.min_p > 0.0f && !probs.empty()) {
        float pmax = 0.0f;
        for (float p : probs) pmax = std::max(pmax, p);
        const float thr = cfg.min_p * pmax;
        for (auto & p : probs) if (p < thr) p = 0.0f;
    }

    // Draw from the final candidate set. Same CDF walk as draw_from_weights
    // above (it renormalizes internally, which is a no-op cost here since
    // probs already sums to ~1) — reuse it instead of re-deriving the same
    // loop a second time.
    for (size_t i = 0; i < cand.size(); i++) cand[i].first = probs[i];
    const int chosen = draw_from_weights(cand, r_uniform);
    trace_sampler_choice(cand, chosen, cfg, r_uniform, "cpu_sample");
    return chosen;
}

bool parse_sampler_token(std::string & line, SamplerCfg & out) {
    auto pos = line.find(" samp=");
    if (pos == std::string::npos) return false;
    auto end = line.find(' ', pos + 1);
    std::string tok = (end == std::string::npos)
                          ? line.substr(pos + 6)
                          : line.substr(pos + 6, end - (pos + 6));
    std::vector<std::string> fields;
    size_t field_begin = 0;
    for (;;) {
        const size_t comma = tok.find(',', field_begin);
        fields.push_back(tok.substr(field_begin, comma == std::string::npos
                                                    ? std::string::npos
                                                    : comma - field_begin));
        if (comma == std::string::npos) break;
        field_begin = comma + 1;
    }
    if (fields.empty() || fields.size() > 7) return false;
    for (const auto & field : fields) if (field.empty()) return false;

    auto parse_float = [](const std::string & text, float & value) {
        errno = 0;
        char * tail = nullptr;
        value = std::strtof(text.c_str(), &tail);
        return errno != ERANGE && tail != text.c_str() && *tail == '\0';
    };
    auto parse_int = [](const std::string & text, int & value) {
        errno = 0;
        char * tail = nullptr;
        const long parsed = std::strtol(text.c_str(), &tail, 10);
        if (errno == ERANGE || tail == text.c_str() || *tail != '\0' ||
            parsed < INT_MIN || parsed > INT_MAX) return false;
        value = static_cast<int>(parsed);
        return true;
    };
    auto parse_seed = [](const std::string & text, uint64_t & value) {
        if (!text.empty() && text[0] == '-') return false;
        errno = 0;
        char * tail = nullptr;
        const unsigned long long parsed = std::strtoull(text.c_str(), &tail, 10);
        if (errno == ERANGE || tail == text.c_str() || *tail != '\0' ||
            parsed > std::numeric_limits<uint64_t>::max()) return false;
        value = static_cast<uint64_t>(parsed);
        return true;
    };

    float t = 0.0f, tp = 1.0f, rp = 1.0f, fp = 0.0f, pp = 0.0f;
    int tk = 0;
    uint64_t sd = 0;
    if (!parse_float(fields[0], t) ||
        (fields.size() > 1 && !parse_float(fields[1], tp)) ||
        (fields.size() > 2 && !parse_int(fields[2], tk)) ||
        (fields.size() > 3 && !parse_float(fields[3], rp)) ||
        (fields.size() > 4 && !parse_seed(fields[4], sd)) ||
        (fields.size() > 5 && !parse_float(fields[5], fp)) ||
        (fields.size() > 6 && !parse_float(fields[6], pp))) {
        return false;
    }
    if (!std::isfinite(t) || t < 0.0f ||
        !std::isfinite(tp) || tp < 0.0f || tp > 1.0f ||
        tk < 0 || !std::isfinite(rp) || rp <= 0.0f ||
        !std::isfinite(fp) || fp < -2.0f || fp > 2.0f ||
        !std::isfinite(pp) || pp < -2.0f || pp > 2.0f) {
        return false;
    }
    line.erase(pos, (end == std::string::npos ? std::string::npos : end - pos));
    out.temp     = t;
    out.top_p    = tp;
    out.top_k    = tk;
    out.rep_pen  = rp;
    out.seed     = sd;
    out.freq_pen = fp;
    out.pres_pen = pp;
    return true;
}

}  // namespace dflash::common
