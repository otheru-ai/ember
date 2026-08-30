#include "qwen4exp_internal.h"

#include <cstdio>
#include <numeric>
#include <vector>

using namespace dflash::common;

static int g_pass = 0;
static int g_fail = 0;
#define CHECK(c, m) do { if (c) ++g_pass; else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", m); } } while (0)

int main() {
    for (const int tokens : {1, 3, 4, 2047, 2048}) {
        std::vector<int32_t> selected;
        const bool dense = qwen4exp_qsa_dense_selection(tokens, selected);
        std::vector<int32_t> expected(static_cast<size_t>(tokens));
        std::iota(expected.begin(), expected.end(), int32_t{0});
        CHECK(dense && selected == expected,
              "QSA dense selection preserves sequential block-plus-tail order");
        std::vector<float> raw(static_cast<size_t>(tokens) * 128, 0.0f);
        std::vector<float> query(4 * 128, 1.0f);
        for (int token = 0; token < tokens; ++token) {
            raw[static_cast<size_t>(token) * 128] =
                static_cast<float>(tokens - token);
        }
        CHECK(selected == qwen4exp_qsa_selected_tokens(
                              raw, query.data(), tokens),
              "QSA dense selection is exact against the scored path");
    }
    std::vector<int32_t> boundary_sentinel = {7, 11};
    CHECK(!qwen4exp_qsa_dense_selection(2049, boundary_sentinel) &&
              boundary_sentinel == std::vector<int32_t>({7, 11}),
          "QSA dense selection leaves 2049-token contexts to scored selection");
    std::vector<float> boundary_raw(static_cast<size_t>(2049) * 128, 0.0f);
    std::vector<float> boundary_query(4 * 128, 1.0f);
    const std::vector<int32_t> boundary_scored =
        qwen4exp_qsa_selected_tokens(boundary_raw, boundary_query.data(), 2049);
    CHECK(boundary_scored.size() == 2049 &&
              boundary_scored.front() == 0 && boundary_scored.back() == 2048,
          "QSA scored fallback preserves all blocks plus the 2049-token tail");

    CHECK(qwen4exp_weight_type_supported(GGML_TYPE_F32, true) &&
              qwen4exp_weight_type_supported(GGML_TYPE_F16, true) &&
              qwen4exp_weight_type_supported(GGML_TYPE_BF16, true),
          "Qwen vectors and norms accept only released floating storage");
    CHECK(!qwen4exp_weight_type_supported(GGML_TYPE_Q8_0, true) &&
              !qwen4exp_weight_type_supported(GGML_TYPE_Q4_K, true) &&
              qwen4exp_weight_type_supported(GGML_TYPE_Q8_0, false) &&
              qwen4exp_weight_type_supported(GGML_TYPE_Q6_K, false) &&
              qwen4exp_weight_type_supported(GGML_TYPE_Q4_K, false) &&
              qwen4exp_weight_type_supported(GGML_TYPE_Q3_0_ROCMFPX, false) &&
              qwen4exp_weight_type_supported(GGML_TYPE_Q4_0_ROCMI4, false) &&
              qwen4exp_weight_type_supported(GGML_TYPE_Q4_0_ROCMFP4_FAST, false) &&
              !qwen4exp_weight_type_supported(GGML_TYPE_Q5_0, false),
          "Qwen matrix loader accepts kernel-backed bakeoff types and rejects unknown types");

    const Qwen4ExpMemoryPlan iq4_plan =
        qwen4exp_memory_plan(94000000000ULL, 262144);
    CHECK(iq4_plan.fits && iq4_plan.capacity_bytes == (128ULL << 30) &&
              iq4_plan.qsa_cache_bytes == 14495514624ULL &&
              iq4_plan.runtime_reserve_bytes == (8ULL << 30),
          "94-GB ROCMI4 text weights fit native 262144 with explicit reserve");
    const Qwen4ExpMemoryPlan oversized_plan =
        qwen4exp_memory_plan(120000000000ULL, 262144);
    CHECK(!oversized_plan.fits && oversized_plan.total_bytes >
              oversized_plan.capacity_bytes,
          "loader rejects a native-context plan that exceeds 128-GiB UMA");
    const Qwen4ExpMemoryPlan yarn_plan =
        qwen4exp_memory_plan(94000000000ULL, EMBER_QWEN_YARN_MAX_CONTEXT);
    CHECK(!yarn_plan.fits &&
              yarn_plan.qsa_cache_bytes == 55296000000ULL &&
              yarn_plan.total_bytes > yarn_plan.capacity_bytes,
          "official 1M YaRN policy is costed honestly and rejected above 128 GiB");
    const Qwen4ExpMemoryPlan arbitrary_extension =
        qwen4exp_memory_plan(1, 524288);
    CHECK(!arbitrary_extension.fits && arbitrary_extension.total_bytes == UINT64_MAX,
          "planner rejects unconfigured context-extension targets");

    Qwen4ExpState state;
    state.cur_pos = 17;
    state.last_token = 42;
    state.hc.assign(10240, 1.0f);
    const float key[] = {1.0f, 2.0f};
    const float value[] = {3.0f, 4.0f};
    const float index_key[] = {5.0f, 6.0f};
    state.layers[3].key.append(key, 2);
    state.layers[3].value.append(value, 2);
    state.layers[3].index_key.append(index_key, 2);
    state.layers[0].conv =
        std::make_shared<std::vector<float>>(std::initializer_list<float>{7.0f});
    state.layers[0].recurrent =
        std::make_shared<std::vector<float>>(std::initializer_list<float>{8.0f});
    state.ple_conv = {9.0f};
    Qwen4ExpState snapshot = state;
    std::unordered_set<const void *> live_seen;
    const uint64_t live_bytes = state.account_bytes(live_seen);
    const uint64_t shared_bytes = snapshot.account_bytes(live_seen);
    CHECK(shared_bytes < live_bytes,
          "snapshot accounting does not charge shared QSA/GDN storage twice");
    state.layers[3].index_key.clear();
    CHECK(snapshot.cur_pos == 17 && snapshot.last_token == 42,
          "snapshot preserves the committed frontier");
    CHECK(snapshot.layers[3].key.size() == 2 &&
          snapshot.layers[3].value.size() == 2 &&
          snapshot.layers[3].index_key.size() == 2,
          "snapshot keeps QSA K/V and raw index-K atomically");
    std::vector<float> exported_key(snapshot.layers[3].key.size());
    CHECK(snapshot.layers[3].key.copy_to(exported_key.data(),
                                         exported_key.size()) &&
              exported_key == std::vector<float>({1.0f, 2.0f}) &&
              !snapshot.layers[3].key.copy_to(exported_key.data(), 1),
          "QSA snapshot exports an exact, size-checked disk representation");
    const float live_more[] = {7.0f};
    state.layers[3].key.append(live_more, 1);
    CHECK(snapshot.layers[3].key.size() == 2 &&
              snapshot.layers[3].key.at(1) == 2.0f &&
              state.layers[3].key.size() == 3,
          "QSA snapshot slabs are copy-on-write after the frontier");
    CHECK(snapshot.layers[0].conv == state.layers[0].conv &&
          snapshot.layers[0].recurrent == state.layers[0].recurrent &&
          *snapshot.layers[0].conv == std::vector<float>({7.0f}) &&
          *snapshot.layers[0].recurrent == std::vector<float>({8.0f}) &&
          snapshot.ple_conv == std::vector<float>({9.0f}),
          "snapshot shares GDN state and keeps PLE recurrent state");

    snapshot.clear();
    CHECK(snapshot.cur_pos == 0 && snapshot.hc.empty() &&
          snapshot.ple_tokens[0] == 248044 &&
          snapshot.ple_tokens[1] == 248044,
          "state clear restores the PLE EOS-padded frontier");

    const auto rows_a = qwen4exp_ple_rows(17, {248044, 248044});
    const auto rows_b = qwen4exp_ple_rows(17, {3, 4});
    CHECK(rows_a != rows_b, "PLE hashes include predecessor history");
    const auto rows_after_eos_a = qwen4exp_ple_rows(17, {3, 248044});
    const auto rows_after_eos_b = qwen4exp_ple_rows(17, {9, 248044});
    CHECK(rows_after_eos_a == rows_after_eos_b,
          "PLE EOS boundary hides older trigram history");
    int64_t offset = 0;
    const int64_t vocab[16] = {
        20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,
        20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171};
    bool rows_in_range = true;
    for (int i = 0; i < 16; ++i) {
        rows_in_range = rows_in_range && rows_a[i] >= offset &&
                        rows_a[i] < offset + vocab[i];
        offset += vocab[i];
    }
    CHECK(rows_in_range, "PLE rows stay inside each prime-sized head partition");

    constexpr int tokens = 2405; // 601 complete blocks + one-token tail
    std::vector<float> raw(static_cast<size_t>(tokens) * 128, 0.0f);
    std::vector<float> query(4 * 128, 1.0f);
    for (int token = 0; token < tokens; ++token)
        raw[static_cast<size_t>(token) * 128] = static_cast<float>(token + 1);
    const std::vector<int32_t> selected =
        qwen4exp_qsa_selected_tokens(raw, query.data(), tokens);
    CHECK(selected.size() == 2049,
          "QSA selects 512 whole blocks (2048 tokens) plus causal tail");
    CHECK(!selected.empty() && selected.back() == tokens - 1,
          "QSA incomplete tail is always visible");
    bool whole_blocks = selected.size() >= 2048;
    for (size_t i = 0; i + 3 < 2048; i += 4)
        whole_blocks = whole_blocks && selected[i + 1] == selected[i] + 1 &&
                       selected[i + 2] == selected[i] + 2 &&
                       selected[i + 3] == selected[i] + 3;
    CHECK(whole_blocks, "QSA top-k never cuts through a four-token block");

    std::printf("qwen4exp-state: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
