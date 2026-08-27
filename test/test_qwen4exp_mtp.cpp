#include "qwen4exp_mtp.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace dflash::common;

static int g_pass = 0;
static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) ++g_pass; else { ++g_fail; std::printf("FAIL: %s\n", msg); } \
} while (0)

static Qwen4ExpState seed_state() {
    Qwen4ExpState state;
    state.cur_pos = 7;
    state.last_token = 101;
    state.hc = {1.0f, 2.0f, 3.0f};
    state.ple_tokens = {41, 42};
    state.ple_conv = {4.0f, 5.0f};
    state.mrope_positions[0] = {1, 2};
    state.mrope_positions[1] = {3, 4};
    state.mrope_positions[2] = {5, 6};
    Qwen4ExpLayerState & layer = state.layers[0];
    layer.conv = std::make_shared<std::vector<float>>(
        std::initializer_list<float>{10.0f, 11.0f});
    layer.recurrent = std::make_shared<std::vector<float>>(
        std::initializer_list<float>{12.0f, 13.0f});
    const float key[] = {14.0f, 15.0f};
    const float value[] = {16.0f};
    const float index[] = {17.0f, 18.0f, 19.0f};
    layer.key.append(key, 2);
    layer.value.append(value, 1);
    layer.index_key.append(index, 3);
    return state;
}

static bool same_buffer(const Qwen4ExpCowBuffer & a,
                        const Qwen4ExpCowBuffer & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a.at(i) != b.at(i)) return false;
    return true;
}

static bool same_ptr_vector(const std::shared_ptr<std::vector<float>> & a,
                            const std::shared_ptr<std::vector<float>> & b) {
    if (static_cast<bool>(a) != static_cast<bool>(b)) return false;
    return !a || *a == *b;
}

static bool same_state(const Qwen4ExpState & a, const Qwen4ExpState & b) {
    if (a.cur_pos != b.cur_pos || a.last_token != b.last_token ||
        a.hc != b.hc || a.ple_tokens != b.ple_tokens ||
        a.ple_conv != b.ple_conv || a.mrope_positions != b.mrope_positions)
        return false;
    for (size_t i = 0; i < a.layers.size(); ++i) {
        const Qwen4ExpLayerState & la = a.layers[i];
        const Qwen4ExpLayerState & lb = b.layers[i];
        if (!same_ptr_vector(la.conv, lb.conv) ||
            !same_ptr_vector(la.recurrent, lb.recurrent) ||
            !same_buffer(la.key, lb.key) ||
            !same_buffer(la.value, lb.value) ||
            !same_buffer(la.index_key, lb.index_key)) return false;
    }
    return true;
}

struct StepHarness {
    int calls = 0;
    int fail_at = -1;
};

static bool fake_q1_step(void * opaque, Qwen4ExpState & state,
                         const Qwen4ExpReplayRow & row,
                         std::vector<float> & logits, std::string & error) {
    auto * harness = static_cast<StepHarness *>(opaque);
    if (harness->calls++ == harness->fail_at) {
        error = "injected replay failure";
        return false;
    }
    ++state.cur_pos;
    state.last_token = row.token;
    state.hc.push_back(static_cast<float>(row.token));
    state.ple_tokens = {state.ple_tokens[1], row.token};
    state.ple_conv.push_back(static_cast<float>(row.token + 1));
    for (size_t axis = 0; axis < 3; ++axis)
        state.mrope_positions[axis].push_back(row.mrope_position[axis]);

    Qwen4ExpLayerState & layer = state.layers[0];
    if (!layer.conv.unique())
        layer.conv = std::make_shared<std::vector<float>>(*layer.conv);
    if (!layer.recurrent.unique())
        layer.recurrent = std::make_shared<std::vector<float>>(*layer.recurrent);
    layer.conv->push_back(static_cast<float>(row.token + 2));
    layer.recurrent->push_back(static_cast<float>(row.token + 3));
    const float v = static_cast<float>(row.token);
    layer.key.append(&v, 1);
    layer.value.append(&v, 1);
    layer.index_key.append(&v, 1);
    logits = {v, v + 0.5f};
    return true;
}

static std::vector<Qwen4ExpReplayRow> rows() {
    return {{201, {7, 7, 7}}, {202, {8, 8, 8}}, {203, {9, 9, 9}}};
}

static void poison_verified_state(Qwen4ExpState & state,
                                  std::vector<float> & logits) {
    StepHarness harness;
    std::string error;
    for (const auto & row : rows())
        (void) fake_q1_step(&harness, state, row, logits, error);
    // A rejected verification graph may also have touched later layers.
    state.layers[17].conv =
        std::make_shared<std::vector<float>>(4, 999.0f);
}

static void test_full_accept_keeps_fast_state() {
    const Qwen4ExpState committed = seed_state();
    const std::vector<float> committed_logits = {1.0f};
    Qwen4ExpState verified = committed;
    std::vector<float> verified_logits = committed_logits;
    poison_verified_state(verified, verified_logits);
    const Qwen4ExpState expected = verified;
    const std::vector<float> expected_logits = verified_logits;
    StepHarness harness;
    Qwen4ExpReplayResult result;
    std::string error;
    const auto draft = rows();
    CHECK(qwen4exp_replay_accepted_prefix(
              committed, committed_logits, verified, verified_logits, draft,
              draft.size(), fake_q1_step, &harness, result, error),
          "full acceptance succeeds");
    CHECK(same_state(verified, expected) && verified_logits == expected_logits,
          "full acceptance retains verified state");
    CHECK(harness.calls == 0 && result.rows_replayed == 0 &&
              result.disposition == Qwen4ExpReplayDisposition::FullAcceptance,
          "full acceptance does not replay");
}

static void test_partial_accept_replays_only_prefix() {
    const Qwen4ExpState committed = seed_state();
    const std::vector<float> committed_logits = {1.0f};
    Qwen4ExpState verified = committed;
    std::vector<float> verified_logits = committed_logits;
    poison_verified_state(verified, verified_logits);

    Qwen4ExpState expected = committed;
    std::vector<float> expected_logits = committed_logits;
    StepHarness expected_harness;
    std::string expected_error;
    const auto draft = rows();
    CHECK(fake_q1_step(&expected_harness, expected, draft[0], expected_logits,
                       expected_error),
          "expected replay row builds");
    CHECK(fake_q1_step(&expected_harness, expected, draft[1], expected_logits,
                       expected_error),
          "second expected replay row builds");

    StepHarness harness;
    Qwen4ExpReplayResult result;
    std::string error;
    CHECK(qwen4exp_replay_accepted_prefix(
              committed, committed_logits, verified, verified_logits, draft, 2,
              fake_q1_step, &harness, result, error),
          "partial acceptance succeeds");
    CHECK(same_state(verified, expected) && verified_logits == expected_logits,
          "partial acceptance rebuilds exact q=1 target prefix");
    CHECK(!verified.layers[17].conv,
          "rejected verification mutations do not survive restore");
    CHECK(harness.calls == 2 && result.rows_replayed == 2 &&
              result.disposition ==
                  Qwen4ExpReplayDisposition::ReplayedAcceptedPrefix,
          "partial acceptance replays accepted rows only");
}

static void test_zero_accept_and_failure_are_transactional() {
    const Qwen4ExpState committed = seed_state();
    const std::vector<float> committed_logits = {1.0f};
    const auto draft = rows();
    Qwen4ExpReplayResult result;
    std::string error;

    Qwen4ExpState zero = committed;
    std::vector<float> zero_logits = committed_logits;
    poison_verified_state(zero, zero_logits);
    CHECK(qwen4exp_replay_accepted_prefix(
              committed, committed_logits, zero, zero_logits, draft, 0,
              nullptr, nullptr, result, error),
          "zero acceptance restores without a step callback");
    CHECK(same_state(zero, committed) && zero_logits == committed_logits,
          "zero acceptance restores every target state family");

    Qwen4ExpState failed = committed;
    std::vector<float> failed_logits = committed_logits;
    poison_verified_state(failed, failed_logits);
    StepHarness harness;
    harness.fail_at = 1;
    CHECK(!qwen4exp_replay_accepted_prefix(
              committed, committed_logits, failed, failed_logits, draft, 2,
              fake_q1_step, &harness, result, error),
          "injected replay failure is reported");
    CHECK(same_state(failed, committed) && failed_logits == committed_logits,
          "failed replay returns to the committed frontier");
    CHECK(error == "injected replay failure",
          "replay preserves the concrete step failure");
}

static void test_invalid_accept_count_fails_closed() {
    const Qwen4ExpState committed = seed_state();
    Qwen4ExpState verified = committed;
    std::vector<float> logits = {1.0f};
    Qwen4ExpReplayResult result;
    std::string error;
    const auto draft = rows();
    CHECK(!qwen4exp_replay_accepted_prefix(
              committed, logits, verified, logits, draft, draft.size() + 1,
              nullptr, nullptr, result, error),
          "accepted count beyond draft is rejected");
    CHECK(error.find("exceeds") != std::string::npos,
          "invalid count error is actionable");
}

static void test_depth_and_instrumentation_contract() {
    CHECK(qwen4exp_mtp_effective_depth(1, 4) == 1 &&
              qwen4exp_mtp_effective_depth(4, 2) == 2,
          "MTP depth clips to the configured and remaining-token bounds");
    CHECK(qwen4exp_mtp_effective_depth(0, 3) == 0 &&
              qwen4exp_mtp_effective_depth(5, 3) == 0 &&
              qwen4exp_mtp_effective_depth(3, 0) == 0,
          "invalid or empty MTP windows fall back to q=1");
    Qwen4ExpMtpStats stats;
    CHECK(qwen4exp_mtp_record_round(stats, 4, 3, true) &&
              qwen4exp_mtp_record_round(stats, 2, 2, false),
          "valid MTP rounds update instrumentation");
    CHECK(stats.rounds == 2 && stats.proposed == 6 && stats.accepted == 5 &&
              stats.partial_replays == 1 &&
              stats.accept_rate() > 0.83f && stats.accept_rate() < 0.84f,
          "MTP acceptance telemetry is exact");
    CHECK(!qwen4exp_mtp_record_round(stats, 2, 3, false),
          "impossible MTP acceptance telemetry fails closed");
}

int main() {
    test_full_accept_keeps_fast_state();
    test_partial_accept_replays_only_prefix();
    test_zero_accept_and_failure_are_transactional();
    test_invalid_accept_count_fails_closed();
    test_depth_and_instrumentation_contract();
    std::printf("qwen4exp mtp replay: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
