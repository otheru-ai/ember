#include "qwen4exp_mtp.h"

#include <cstdio>
#include <string>
#include <utility>
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

struct BatchHarness {
    StepHarness step;
    std::vector<int32_t> predictions;
    bool fail = false;
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

static bool fake_verify_batch(
        void * opaque, Qwen4ExpState & state,
        const std::vector<Qwen4ExpReplayRow> & input_rows,
        Qwen4ExpMtpVerifyOutput & output, std::string & error) {
    auto * batch = static_cast<BatchHarness *>(opaque);
    if (batch->fail) {
        state.cur_pos += 100;
        error = "injected batch failure";
        return false;
    }
    if (batch->predictions.size() != input_rows.size()) return false;
    output.row_logits.clear();
    output.row_hc.clear();
    for (size_t i = 0; i < input_rows.size(); ++i) {
        std::vector<float> ignored;
        if (!fake_q1_step(&batch->step, state, input_rows[i], ignored, error))
            return false;
        std::vector<float> logits(16, -1.0f);
        logits[static_cast<size_t>(batch->predictions[i])] = 1.0f;
        output.row_logits.push_back(std::move(logits));
        state.hc.assign(10240, static_cast<float>(state.cur_pos));
        output.row_hc.push_back(state.hc);
    }
    return true;
}

static bool fake_batch_replay(void * opaque, Qwen4ExpState & state,
                              const Qwen4ExpReplayRow & row,
                              std::vector<float> & logits,
                              std::string & error) {
    auto * batch = static_cast<BatchHarness *>(opaque);
    return fake_q1_step(&batch->step, state, row, logits, error);
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

static void test_prompt_sync_plan_is_causally_exact() {
    std::vector<Qwen4ExpMtpPromptSyncRow> plan;
    std::string error;
    CHECK(qwen4exp_mtp_prompt_sync_plan(0, 0, 16, plan, error),
          "pristine q16 prompt synchronization plan builds");
    CHECK(plan.size() == 15 && plan.front().token_row == 1 &&
              plan.front().preceding_target_hc_row == 0 &&
              plan.back().token_row == 15 &&
              plan.back().preceding_target_hc_row == 14,
          "pristine q16 skips only r0 and pairs each token with preceding HC");

    CHECK(qwen4exp_mtp_prompt_sync_plan(16, 15, 5, plan, error),
          "continued prompt synchronization plan builds");
    CHECK(plan.size() == 5 && plan.front().token_row == 0 &&
              plan.front().preceding_target_hc_row == -1 &&
              plan.back().token_row == 4 &&
              plan.back().preceding_target_hc_row == 3,
          "continued chunk uses pre-chunk HC for r0 then preceding batch rows");
    CHECK(!qwen4exp_mtp_prompt_sync_plan(16, 14, 2, plan, error) &&
              error.find("trail") != std::string::npos,
          "stale MTP prompt frontier fails closed before a target batch");

    // Differential scheduling model: arbitrary q16 remainders must produce
    // the same (token, preceding target HC) pairs as token-major q=1 sync.
    std::vector<std::pair<int, int>> batched_pairs;
    int target_pos = 0;
    int mtp_pos = 0;
    for (const size_t rows : {16U, 16U, 5U}) {
        CHECK(qwen4exp_mtp_prompt_sync_plan(
                  target_pos, mtp_pos, rows, plan, error),
              "multi-chunk MTP synchronization plan builds");
        for (const auto & sync : plan) {
            const int token = target_pos + static_cast<int>(sync.token_row);
            const int preceding = sync.preceding_target_hc_row < 0
                ? target_pos - 1
                : target_pos + sync.preceding_target_hc_row;
            batched_pairs.emplace_back(token, preceding);
            ++mtp_pos;
        }
        target_pos += static_cast<int>(rows);
    }
    std::vector<std::pair<int, int>> q1_pairs;
    for (int token = 1; token < 37; ++token)
        q1_pairs.emplace_back(token, token - 1);
    CHECK(batched_pairs == q1_pairs && target_pos == 37 && mtp_pos == 36,
          "q16 MTP synchronization is differential-equivalent to q1 order");
}

static void test_mtp_cache_batch_shape_contract() {
    Qwen4ExpMtpCacheBatchShape shape;
    std::string error;
    CHECK(qwen4exp_mtp_cache_batch_shape(1, shape, error) &&
              shape.rows == 1 && shape.embedding_values == 2560 &&
              shape.target_hc_values == 10240 &&
              shape.hc_projection_rows == 4 && shape.key_values == 512 &&
              shape.value_values == 512 && shape.index_key_values == 128,
          "q1 cache batch shape matches one MTP synchronization row");
    CHECK(qwen4exp_mtp_cache_batch_shape(16, shape, error) &&
              shape.embedding_values == 16U * 2560U &&
              shape.target_hc_values == 16U * 10240U &&
              shape.hc_projection_rows == 64 &&
              shape.key_values == 16U * 512U &&
              shape.index_key_values == 16U * 128U,
          "q16 cache batch shape preserves every independent projection row");
    CHECK(!qwen4exp_mtp_cache_batch_shape(0, shape, error) &&
              shape.rows == 0 && error.find("1 to 16") != std::string::npos,
          "empty MTP cache batch is rejected");
    CHECK(!qwen4exp_mtp_cache_batch_shape(17, shape, error),
          "MTP cache batch cannot exceed the target q16 frontier");
}

static Qwen4ExpMtpState covered_mtp_state(int rows) {
    Qwen4ExpMtpState state;
    state.cur_pos = rows;
    std::vector<float> key(static_cast<size_t>(rows) * 512U, 1.0f);
    std::vector<float> value(static_cast<size_t>(rows) * 512U, 2.0f);
    std::vector<float> index(static_cast<size_t>(rows) * 128U, 3.0f);
    state.qsa.key.append(key.data(), key.size());
    state.qsa.value.append(value.data(), value.size());
    state.qsa.index_key.append(index.data(), index.size());
    for (int axis = 0; axis < 3; ++axis) {
        for (int row = 0; row < rows; ++row)
            state.mrope_positions[static_cast<size_t>(axis)].push_back(row);
    }
    return state;
}

static void test_mtp_snapshot_frontier_invariant() {
    Qwen4ExpState target;
    target.cur_pos = 4;
    target.hc.assign(10240, 7.0f);
    for (auto & axis : target.mrope_positions) axis.assign(4, 1);
    const std::vector<float> target_key(4U * 512U, 1.0f);
    const std::vector<float> target_value(4U * 512U, 2.0f);
    const std::vector<float> target_index(4U * 128U, 3.0f);
    for (size_t layer = 3; layer < target.layers.size(); layer += 4) {
        target.layers[layer].key.append(target_key.data(), target_key.size());
        target.layers[layer].value.append(
            target_value.data(), target_value.size());
        target.layers[layer].index_key.append(
            target_index.data(), target_index.size());
    }
    Qwen4ExpMtpState mtp = covered_mtp_state(3);
    std::vector<float> target_hc = target.hc;
    std::string error;
    CHECK(qwen4exp_mtp_frontier_valid(target, mtp, target_hc, error),
          "complete one-row-trailing MTP snapshot frontier is valid");

    Qwen4ExpMtpState stale = mtp;
    stale.cur_pos = 2;
    CHECK(!qwen4exp_mtp_frontier_valid(target, stale, target_hc, error) &&
              error.find("trail") != std::string::npos,
          "AR fallback that leaves stale MTP position cannot snapshot");
    Qwen4ExpState short_target = target;
    short_target.layers[3].key.clear();
    CHECK(!qwen4exp_mtp_frontier_valid(
              short_target, mtp, target_hc, error) &&
              error.find("target snapshot QSA") != std::string::npos,
          "snapshot rejects incomplete target QSA coverage");
    Qwen4ExpMtpState short_cache = mtp;
    short_cache.qsa.index_key.clear();
    CHECK(!qwen4exp_mtp_frontier_valid(
              target, short_cache, target_hc, error) &&
              error.find("cache coverage") != std::string::npos,
          "snapshot rejects missing raw index-K coverage");
    Qwen4ExpMtpState short_position = mtp;
    short_position.mrope_positions[2].pop_back();
    CHECK(!qwen4exp_mtp_frontier_valid(
              target, short_position, target_hc, error) &&
              error.find("M-RoPE") != std::string::npos,
          "snapshot rejects incomplete M-RoPE history");
    target_hc[0] = 8.0f;
    CHECK(!qwen4exp_mtp_frontier_valid(target, mtp, target_hc, error) &&
              error.find("authoritative") != std::string::npos,
          "snapshot rejects stale target HC even at matching positions");
}

struct LayerMajorHarness {
    size_t rows = 0;
    size_t layers = 0;
    std::vector<float> seed;
    std::vector<float> output;
    std::vector<float> causal;
    std::vector<std::pair<size_t, size_t>> order;
};

static bool fake_layer_major_step(void * opaque, size_t layer, size_t row,
                                  std::string &) {
    auto * harness = static_cast<LayerMajorHarness *>(opaque);
    const float input = layer == 0
        ? harness->seed[row]
        : harness->output[row * harness->layers + layer - 1];
    const float value = input + 0.5f * harness->causal[layer] +
                        static_cast<float>(layer + 1);
    harness->output[row * harness->layers + layer] = value;
    harness->causal[layer] = value;
    harness->order.emplace_back(layer, row);
    return true;
}

static void test_layer_major_matches_token_major_causality() {
    constexpr size_t rows = 5;
    constexpr size_t layers = 7;
    const std::vector<float> seed = {1, 2, 3, 4, 5};
    std::vector<float> expected(rows * layers);
    std::vector<float> expected_causal(layers, 0.0f);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t layer = 0; layer < layers; ++layer) {
            const float input = layer == 0
                ? seed[row] : expected[row * layers + layer - 1];
            const float value = input + 0.5f * expected_causal[layer] +
                                static_cast<float>(layer + 1);
            expected[row * layers + layer] = value;
            expected_causal[layer] = value;
        }
    }
    LayerMajorHarness harness{
        rows, layers, seed, std::vector<float>(rows * layers),
        std::vector<float>(layers, 0.0f), {}};
    std::string error;
    CHECK(qwen4exp_run_layer_major(rows, layers, fake_layer_major_step,
                                   &harness, error),
          "native verifier layer-major schedule completes");
    CHECK(harness.output == expected && harness.causal == expected_causal,
          "layer-major verifier is differential-equivalent to token-major q=1");
    const std::pair<size_t, size_t> first{0, 0};
    const std::pair<size_t, size_t> end_first_layer{0, rows - 1};
    const std::pair<size_t, size_t> start_second_layer{1, 0};
    CHECK(harness.order.front() == first &&
              harness.order[rows - 1] == end_first_layer &&
              harness.order[rows] == start_second_layer,
          "native verifier keeps rows causal inside each hot target layer");
}

static void test_bounded_batch_full_and_partial_rejection() {
    const std::vector<Qwen4ExpReplayRow> input = {
        {210, {7, 7, 7}}, {211, {8, 8, 8}},
        {212, {9, 9, 9}}, {213, {10, 10, 10}}};
    const Qwen4ExpState committed = seed_state();
    const std::vector<float> committed_logits = {1.0f};

    Qwen4ExpState full = committed;
    std::vector<float> full_logits = committed_logits;
    BatchHarness full_batch;
    full_batch.predictions = {3, 4, 5, 6};
    Qwen4ExpMtpVerifyOutput output;
    Qwen4ExpMtpVerifyResult result;
    std::string error;
    CHECK(qwen4exp_verify_bounded_batch(
              full, full_logits, input, {3, 4, 5}, 14, 15,
              fake_verify_batch, fake_batch_replay, &full_batch, output,
              result, error),
          "bounded target batch accepts a complete draft");
    CHECK(result.accepted_predictions == 3 &&
              result.committed_input_rows == 4 &&
              result.replay.disposition ==
                  Qwen4ExpReplayDisposition::FullAcceptance &&
              result.replay.rows_replayed == 0,
          "full bounded acceptance retains the batch state");
    CHECK(output.row_logits.size() == 4 && output.row_hc.size() == 4 &&
              full.cur_pos == committed.cur_pos + 4,
          "bounded verifier exposes every target logit and HC row");

    Qwen4ExpState partial = committed;
    std::vector<float> partial_logits = committed_logits;
    BatchHarness partial_batch;
    partial_batch.predictions = {3, 4, 5, 6};
    CHECK(qwen4exp_verify_bounded_batch(
              partial, partial_logits, input, {3, 9, 5}, 14, 15,
              fake_verify_batch, fake_batch_replay, &partial_batch, output,
              result, error),
          "bounded target batch reconciles a partial rejection");
    Qwen4ExpState expected = committed;
    std::vector<float> expected_logits = committed_logits;
    StepHarness expected_step;
    CHECK(fake_q1_step(&expected_step, expected, input[0], expected_logits,
                       error) &&
              fake_q1_step(&expected_step, expected, input[1], expected_logits,
                           error),
          "partial bounded reference prefix builds");
    CHECK(result.accepted_predictions == 1 &&
              result.committed_input_rows == 2 && same_state(partial, expected) &&
              partial_logits == expected_logits,
          "partial rejection restores and replays exactly base plus acceptance");
    CHECK(result.replay.disposition ==
              Qwen4ExpReplayDisposition::ReplayedAcceptedPrefix &&
              result.replay.rows_replayed == 2,
          "partial bounded rejection reports strict HaloSpecKV replay");

    // A prefix-cache snapshot copied after reconcile must restore the exact
    // accepted target frontier, never the rejected tail evaluated in batch.
    const Qwen4ExpState snapshot = partial;
    const std::vector<float> snapshot_logits = partial_logits;
    poison_verified_state(partial, partial_logits);
    partial = snapshot;
    partial_logits = snapshot_logits;
    CHECK(same_state(partial, expected) && partial_logits == expected_logits,
          "post-rejection snapshot restores target state and seed logits");
}

static void test_bounded_batch_failure_is_transactional() {
    Qwen4ExpState state = seed_state();
    const Qwen4ExpState committed = state;
    std::vector<float> logits = {1.0f};
    const std::vector<float> committed_logits = logits;
    BatchHarness batch;
    batch.fail = true;
    Qwen4ExpMtpVerifyOutput output;
    Qwen4ExpMtpVerifyResult result;
    std::string error;
    const std::vector<Qwen4ExpReplayRow> input = {
        {210, {7, 7, 7}}, {211, {8, 8, 8}}};
    CHECK(!qwen4exp_verify_bounded_batch(
              state, logits, input, {3}, 14, 15, fake_verify_batch,
              fake_batch_replay, &batch, output, result, error),
          "bounded target batch failure is reported");
    CHECK(same_state(state, committed) && logits == committed_logits &&
              output.row_logits.empty() && error == "injected batch failure",
          "failed bounded target batch restores the complete checkpoint");
}

static void test_bounded_terminal_is_observed_not_consumed() {
    const std::vector<Qwen4ExpReplayRow> input = {
        {210, {7, 7, 7}}, {211, {8, 8, 8}},
        {212, {9, 9, 9}}, {213, {10, 10, 10}}};
    Qwen4ExpState state = seed_state();
    const Qwen4ExpState committed = state;
    std::vector<float> logits = {1.0f};
    BatchHarness batch;
    batch.predictions = {3, 14, 5, 6};
    Qwen4ExpMtpVerifyOutput output;
    Qwen4ExpMtpVerifyResult result;
    std::string error;
    CHECK(qwen4exp_verify_bounded_batch(
              state, logits, input, {3, 14, 5}, 14, 15,
              fake_verify_batch, fake_batch_replay, &batch, output, result,
              error),
          "bounded target batch observes an accepted terminal prediction");
    Qwen4ExpState expected = committed;
    std::vector<float> expected_logits = {1.0f};
    StepHarness expected_step;
    CHECK(fake_q1_step(&expected_step, expected, input[0], expected_logits,
                       error) &&
              fake_q1_step(&expected_step, expected, input[1], expected_logits,
                           error),
          "terminal reference frontier builds without consuming EOS");
    CHECK(result.accepted_predictions == 2 && result.terminal_prediction &&
              result.committed_input_rows == 2 && same_state(state, expected),
          "accepted EOS is reported but absent from target state and snapshots");
}

int main() {
    test_full_accept_keeps_fast_state();
    test_partial_accept_replays_only_prefix();
    test_zero_accept_and_failure_are_transactional();
    test_invalid_accept_count_fails_closed();
    test_depth_and_instrumentation_contract();
    test_prompt_sync_plan_is_causally_exact();
    test_mtp_cache_batch_shape_contract();
    test_mtp_snapshot_frontier_invariant();
    test_layer_major_matches_token_major_causality();
    test_bounded_batch_full_and_partial_rejection();
    test_bounded_batch_failure_is_transactional();
    test_bounded_terminal_is_observed_not_consumed();
    std::printf("qwen4exp mtp replay: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
