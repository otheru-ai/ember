#include "qwen4exp_reference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

namespace ref = dflash::qwen4exp::reference;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(condition, message) do {                                      \
    if (condition) { ++g_pass; } else {                                     \
        ++g_fail; std::fprintf(stderr, "FAIL: %s\n", message);             \
    }                                                                        \
} while (0)

static bool close_enough(double actual, double expected,
                         double tolerance = 1.0e-10) {
    return std::fabs(actual - expected) <= tolerance;
}

static void test_ple() {
    const ref::PleHashParameters & released = ref::released_ple_hash_parameters();
    CHECK(released.eos_token_id == 248046,
          "released PLE EOS is exact");
    const std::array<std::uint64_t, ref::kPleNgramSize> expected_multipliers = {
        23703573157769ULL, 20109073645365ULL, 8052911324071ULL,
    };
    CHECK(released.multipliers == expected_multipliers,
          "released PLE I64 multipliers are exact");
    const std::array<std::uint64_t, ref::kPleHeadCount> expected_primes = {
        20000003ULL, 20000023ULL, 20000033ULL, 20000047ULL,
        20000059ULL, 20000063ULL, 20000069ULL, 20000077ULL,
        20000081ULL, 20000093ULL, 20000107ULL, 20000147ULL,
        20000153ULL, 20000159ULL, 20000161ULL, 20000171ULL,
    };
    const std::array<std::uint64_t, ref::kPleHeadCount> expected_offsets = {
        0ULL, 20000003ULL, 40000026ULL, 60000059ULL,
        80000106ULL, 100000165ULL, 120000228ULL, 140000297ULL,
        160000374ULL, 180000455ULL, 200000548ULL, 220000655ULL,
        240000802ULL, 260000955ULL, 280001114ULL, 300001275ULL,
    };
    CHECK(released.head_vocab_sizes == expected_primes,
          "all released PLE prime vocabulary sizes are exact");
    CHECK(released.head_offsets == expected_offsets,
          "released PLE I64 offsets are exact");

    const std::vector<std::uint64_t> expected_cold = {
        14203330ULL, 27434421ULL, 47790928ULL, 68479882ULL,
        84389552ULL, 103824216ULL, 133724406ULL, 154987881ULL,
        164840003ULL, 199120592ULL, 215311083ULL, 224054397ULL,
        244064495ULL, 265300291ULL, 285984707ULL, 311448973ULL,
    };
    CHECK(ref::ple_hash_indices({5}, {}, released) == expected_cold,
          "released PLE hash matches the checkpoint constants");

    ref::PleHashParameters tiny{};
    tiny.eos_token_id = 99;
    tiny.multipliers = {3, 5, 7};
    tiny.head_vocab_sizes.fill(11);
    for (std::size_t head = 0; head < ref::kPleHeadCount; ++head) {
        tiny.head_offsets[head] = 11 * head;
    }
    const std::vector<std::uint64_t> sequence = {2, 3, 99, 4, 5};
    const std::vector<std::uint64_t> rows =
        ref::ple_hash_indices(sequence, {}, tiny);
    const std::vector<std::uint64_t> cold_eos =
        ref::ple_hash_indices({99}, {}, tiny);
    CHECK(!std::equal(cold_eos.begin(), cold_eos.end(),
                      rows.begin() + static_cast<std::ptrdiff_t>(2 * ref::kPleHeadCount)),
          "an EOS token still hashes its own preceding context");
    const std::vector<std::uint64_t> cold_after_eos =
        ref::ple_hash_indices({4, 5}, {}, tiny);
    CHECK(std::equal(cold_after_eos.begin(), cold_after_eos.end(),
                     rows.begin() + static_cast<std::ptrdiff_t>(3 * ref::kPleHeadCount)),
          "PLE EOS resets all predecessor positions after the boundary");

    const std::vector<std::uint64_t> continued =
        ref::ple_hash_indices({9}, {7, 8}, tiny);
    const std::vector<std::uint64_t> contiguous =
        ref::ple_hash_indices({7, 8, 9}, {}, tiny);
    CHECK(std::equal(continued.begin(), continued.end(),
                     contiguous.end() - static_cast<std::ptrdiff_t>(ref::kPleHeadCount)),
          "PLE previous context is equivalent to contiguous token history");
}

static void test_qsa() {
    std::vector<double> keys;
    for (std::size_t token = 0; token < 11; ++token) {
        keys.push_back(static_cast<double>(token));
        keys.push_back(2.0 * static_cast<double>(token));
    }
    const std::vector<std::size_t> visible = {1, 2, 3, 4, 6, 7, 8, 9, 10};
    const std::vector<double> pooled =
        ref::qsa_pool_complete_blocks(keys, 2, visible);
    CHECK(pooled.size() == 4 && close_enough(pooled[0], 2.5) &&
              close_enough(pooled[1], 5.0) && close_enough(pooled[2], 7.5) &&
              close_enough(pooled[3], 15.0),
          "QSA mean-pools complete visible blocks of four");

    const std::vector<double> scores =
        ref::qsa_score_blocks({1, -1, -1, 2}, 2, 2, pooled);
    CHECK(scores.size() == 2 &&
              close_enough(scores[0], 7.5 / std::sqrt(2.0)) &&
              close_enough(scores[1], 22.5 / std::sqrt(2.0)),
          "QSA score sums rectified per-head dot products");
    CHECK(ref::qsa_select_tokens(scores, visible, 4) ==
              std::vector<std::size_t>({6, 7, 8, 9, 10}),
          "QSA expands the winning block and appends its incomplete causal tail");

    const std::vector<std::size_t> tie_visible = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    };
    CHECK(ref::qsa_select_tokens({1, 1, 0}, tie_visible, 8) ==
              std::vector<std::size_t>({0, 1, 2, 3, 4, 5, 6, 7, 12}),
          "QSA ties are stable by lower complete-block index");

    std::vector<std::size_t> release_visible(130 * ref::kQsaBlockSize + 3);
    std::iota(release_visible.begin(), release_visible.end(), std::size_t{0});
    std::vector<double> release_scores(130);
    std::iota(release_scores.begin(), release_scores.end(), 0.0);
    const std::vector<std::size_t> release_selected =
        ref::qsa_select_tokens(release_scores, release_visible);
    CHECK(release_selected.size() == ref::kQsaTokenBudget + 3 &&
              release_selected.front() == 516 &&
              release_selected[ref::kQsaTokenBudget - 1] == 11 &&
              release_selected[ref::kQsaTokenBudget] == 520 &&
              release_selected.back() == 522,
          "released top-512 QSA budget selects 128 blocks plus a causal tail");
}

static void test_gated_residual() {
    CHECK(ref::gated_residual_init({2, -1}) ==
              std::vector<double>({2, -1, 2, -1, 2, -1, 2, -1}),
          "gated residual initialization creates four exact stream copies");

    const std::vector<double> hyper = {1, 2, 3, 4};
    const std::vector<double> norm_delta(4, 0.0);
    const std::vector<double> down = {1, 1, 1, 1};
    const std::vector<double> up = {1, 2, 0, -1};
    const std::vector<double> inject = {
        1, 1, 1, 1,
        0, 0, 0, 0,
        -1, -1, -1, -1,
        1, -1, 0, 0,
    };
    const ref::GatedResidualRead read = ref::gated_residual_read(
        hyper, 1, norm_delta, 1, down, up, inject, 1.0e-12);
    const double low = 1.0 / (1.0 + std::exp(-1.0));
    const double expected_mix =
        (1.0 / (1.0 + std::exp(-low)) +
         1.0 / (1.0 + std::exp(-2.0 * low)) + 0.5 +
         1.0 / (1.0 + std::exp(low))) / 4.0;
    CHECK(read.normalized.size() == 4 &&
              close_enough(read.normalized.front(), 1.0, 1.0e-10) &&
              close_enough(read.normalized.back(), 1.0, 1.0e-10),
          "gated residual RMSNorm operates independently per stream");
    CHECK(read.mix_weights.size() == 4 &&
              close_enough(read.mixed[0], expected_mix, 1.0e-10),
          "gated residual read uses down/SiLU/up/sigmoid and stream mean");
    CHECK(read.injection_weights.size() == 4 &&
              close_enough(read.injection_weights[0],
                           2.0 / (1.0 + std::exp(-1.0)), 1.0e-10) &&
              close_enough(read.injection_weights[1], 1.0, 1.0e-10) &&
              close_enough(read.injection_weights[2],
                           2.0 / (1.0 + std::exp(1.0)), 1.0e-10) &&
              close_enough(read.injection_weights[3], 1.0, 1.0e-10),
          "gated residual injection uses two times sigmoid of scaled logits");

    const std::vector<double> scattered =
        ref::gated_residual_inject({1, 2, 3, 4}, {10}, read.injection_weights);
    CHECK(close_enough(scattered[0], 1 + 10 * read.injection_weights[0]) &&
              close_enough(scattered[1], 2 + 10 * read.injection_weights[1]) &&
              close_enough(scattered[2], 3 + 10 * read.injection_weights[2]) &&
              close_enough(scattered[3], 4 + 10 * read.injection_weights[3]),
          "gated residual block output scatters independently to four streams");
}

static void test_gdn() {
    const std::vector<ref::GdnScalarStep> steps = {
        {2, 3, 4, std::log(0.5), 0.25, 0},
        {-1, 2, -3, std::log(0.8), 0.5, std::log(3.0)},
    };
    const ref::GdnScalarTrace trace =
        ref::gdn_scalar_recurrence(steps, 0.25, 2.0);
    CHECK(trace.states.size() == 2 &&
              close_enough(trace.states[0], 1.093749947916671) &&
              close_enough(trace.states[1], -1.0624997239583993),
          "GDN scalar state applies decay before gated delta update");
    CHECK(close_enough(trace.core_outputs[0], 1.0937498111979533) &&
              close_enough(trace.core_outputs[1], 1.0624991927089358),
          "GDN scalar output reads updated state through normalized query");
    CHECK(close_enough(trace.gated_outputs[0], 0.9999995820409341) &&
              close_enough(trace.gated_outputs[1], 1.4999993356395702),
          "Qwen4Exp GDN applies sigmoid output gating after RMSNorm");
}

int main() {
    test_ple();
    test_qsa();
    test_gated_residual();
    test_gdn();
    std::printf("qwen4exp reference: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
