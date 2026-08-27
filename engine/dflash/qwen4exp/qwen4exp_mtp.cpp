#include "qwen4exp_mtp.h"

#include <algorithm>
#include <limits>

namespace dflash::common {

uint64_t Qwen4ExpMtpState::account_bytes(
        std::unordered_set<const void *> & seen) const {
    const auto vector_bytes = [](const auto & values) {
        return static_cast<uint64_t>(values.capacity()) * sizeof(values[0]);
    };
    uint64_t total = vector_bytes(hc);
    for (const auto & axis : mrope_positions) total += vector_bytes(axis);
    if (qsa.conv && seen.insert(qsa.conv.get()).second)
        total += vector_bytes(*qsa.conv);
    if (qsa.recurrent && seen.insert(qsa.recurrent.get()).second)
        total += vector_bytes(*qsa.recurrent);
    total += qsa.key.account_bytes(seen);
    total += qsa.value.account_bytes(seen);
    total += qsa.index_key.account_bytes(seen);
    return total;
}

bool qwen4exp_mtp_prompt_sync_plan(
        int target_pos_before, int mtp_pos_before, size_t target_rows,
        std::vector<Qwen4ExpMtpPromptSyncRow> & plan, std::string & error) {
    plan.clear();
    error.clear();
    if (target_pos_before < 0 || mtp_pos_before < 0 ||
        target_rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        target_pos_before > std::numeric_limits<int>::max() -
                                static_cast<int>(target_rows)) {
        error = "invalid Qwen4Exp MTP prompt synchronization range";
        return false;
    }
    const bool pristine = target_pos_before == 0 && mtp_pos_before == 0;
    if (!pristine && target_pos_before != mtp_pos_before + 1) {
        error = "Qwen4Exp MTP prompt frontier does not trail target by one row";
        return false;
    }
    const size_t first = pristine && target_rows ? 1U : 0U;
    plan.reserve(target_rows - first);
    for (size_t row = first; row < target_rows; ++row) {
        plan.push_back({row, row == 0 ? -1 : static_cast<int>(row - 1)});
    }
    return true;
}

bool qwen4exp_mtp_cache_batch_shape(
        size_t rows, Qwen4ExpMtpCacheBatchShape & shape,
        std::string & error) {
    shape = {};
    error.clear();
    constexpr size_t kMaxRows = 16;
    if (rows == 0 || rows > kMaxRows) {
        error = "Qwen4Exp MTP cache batch width must be from 1 to 16";
        return false;
    }
    shape.rows = rows;
    shape.embedding_values = rows * 2560U;
    shape.target_hc_values = rows * 10240U;
    shape.hc_projection_rows = rows * 4U;
    shape.key_values = rows * 512U;
    shape.value_values = rows * 512U;
    shape.index_key_values = rows * 128U;
    return true;
}

bool qwen4exp_mtp_frontier_valid(
        const Qwen4ExpState & target, const Qwen4ExpMtpState & mtp,
        const std::vector<float> & target_hc, std::string & error) {
    error.clear();
    if (target.cur_pos <= 0 || mtp.cur_pos < 0 ||
        target.cur_pos != mtp.cur_pos + 1) {
        error = "Qwen4Exp MTP snapshot frontier must trail target by one row";
        return false;
    }
    if (target_hc.size() != 10240 || target.hc.size() != 10240 ||
        target.hc != target_hc) {
        error = "Qwen4Exp MTP snapshot target HC is not authoritative";
        return false;
    }
    const size_t target_rows = static_cast<size_t>(target.cur_pos);
    for (const auto & axis : target.mrope_positions) {
        if (axis.size() != target_rows) {
            error = "Qwen4Exp target snapshot M-RoPE coverage is incomplete";
            return false;
        }
    }
    for (size_t layer = 3; layer < target.layers.size(); layer += 4) {
        const Qwen4ExpLayerState & qsa = target.layers[layer];
        if (qsa.key.size() != target_rows * 512U ||
            qsa.value.size() != target_rows * 512U ||
            qsa.index_key.size() != target_rows * 128U) {
            error = "Qwen4Exp target snapshot QSA cache coverage is incomplete";
            return false;
        }
    }
    const size_t rows = static_cast<size_t>(mtp.cur_pos);
    if (mtp.qsa.key.size() != rows * 512U ||
        mtp.qsa.value.size() != rows * 512U ||
        mtp.qsa.index_key.size() != rows * 128U) {
        error = "Qwen4Exp MTP snapshot QSA cache coverage is incomplete";
        return false;
    }
    for (const auto & axis : mtp.mrope_positions) {
        if (axis.size() != rows) {
            error = "Qwen4Exp MTP snapshot M-RoPE coverage is incomplete";
            return false;
        }
    }
    return true;
}

int qwen4exp_mtp_effective_depth(int configured_depth,
                                 int remaining_output_slots) {
    if (configured_depth < 1 || configured_depth > 4 ||
        remaining_output_slots <= 0) return 0;
    return std::min(configured_depth, remaining_output_slots);
}

bool qwen4exp_mtp_record_round(Qwen4ExpMtpStats & stats, uint64_t proposed,
                               uint64_t accepted, bool partial_replay) {
    if (!proposed || accepted > proposed ||
        proposed > UINT64_MAX - stats.proposed ||
        accepted > UINT64_MAX - stats.accepted || stats.rounds == UINT64_MAX ||
        (partial_replay && stats.partial_replays == UINT64_MAX)) return false;
    ++stats.rounds;
    stats.proposed += proposed;
    stats.accepted += accepted;
    if (partial_replay) ++stats.partial_replays;
    return true;
}

bool qwen4exp_run_layer_major(size_t rows, size_t layers,
                              Qwen4ExpLayerMajorStep step, void * opaque,
                              std::string & error) {
    if (!rows || !layers || !step) {
        error = "invalid Qwen4Exp layer-major batch schedule";
        return false;
    }
    for (size_t layer = 0; layer < layers; ++layer) {
        for (size_t row = 0; row < rows; ++row) {
            if (!step(opaque, layer, row, error)) return false;
        }
    }
    return true;
}

namespace {
int32_t argmax_row(const std::vector<float> & logits) {
    return logits.empty() ? -1 : static_cast<int32_t>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
}
} // namespace

bool qwen4exp_verify_bounded_batch(
        Qwen4ExpState & target_state, std::vector<float> & target_logits,
        const std::vector<Qwen4ExpReplayRow> & input_rows,
        const std::vector<int32_t> & candidates, int32_t eos_id,
        int32_t eot_id, Qwen4ExpVerifyBatch verify_batch,
        Qwen4ExpReplayStep replay_step, void * opaque,
        Qwen4ExpMtpVerifyOutput & output,
        Qwen4ExpMtpVerifyResult & result, std::string & error) {
    result = {};
    output = {};
    error.clear();
    if (!verify_batch || !replay_step || input_rows.empty() ||
        candidates.empty() || input_rows.size() != candidates.size() + 1) {
        error = "invalid Qwen4Exp MTP bounded verifier contract";
        return false;
    }

    const Qwen4ExpState committed_state = target_state;
    const std::vector<float> committed_logits = target_logits;
    if (!verify_batch(opaque, target_state, input_rows, output, error)) {
        target_state = committed_state;
        target_logits = committed_logits;
        output = {};
        if (error.empty()) error = "Qwen4Exp MTP target batch failed";
        return false;
    }
    if (output.row_logits.size() != input_rows.size() ||
        output.row_hc.size() != input_rows.size()) {
        target_state = committed_state;
        target_logits = committed_logits;
        output = {};
        error = "Qwen4Exp MTP target batch returned incomplete rows";
        return false;
    }
    for (size_t row = 0; row < input_rows.size(); ++row) {
        if (output.row_logits[row].empty() ||
            output.row_hc[row].size() != 10240) {
            target_state = committed_state;
            target_logits = committed_logits;
            output = {};
            error = "Qwen4Exp MTP target batch returned invalid row shapes";
            return false;
        }
    }
    if (target_state.cur_pos != committed_state.cur_pos +
                                    static_cast<int>(input_rows.size()) ||
        target_state.hc != output.row_hc.back()) {
        target_state = committed_state;
        target_logits = committed_logits;
        output = {};
        error = "Qwen4Exp MTP target batch returned an inconsistent frontier";
        return false;
    }

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (argmax_row(output.row_logits[i]) != candidates[i]) break;
        ++result.accepted_predictions;
        if (candidates[i] == eos_id || candidates[i] == eot_id) {
            result.terminal_prediction = true;
            break;
        }
    }
    // The base row is always authoritative. Every accepted non-terminal
    // candidate is also an emitted input row; EOS/EOT is observed but is not
    // consumed, matching ordinary q=1 generation semantics.
    result.committed_input_rows = 1 + result.accepted_predictions;
    if (result.terminal_prediction) --result.committed_input_rows;
    target_logits = output.row_logits.back();
    if (!qwen4exp_replay_accepted_prefix(
            committed_state, committed_logits, target_state, target_logits,
            input_rows, result.committed_input_rows, replay_step, opaque,
            result.replay, error)) {
        output = {};
        return false;
    }
    return true;
}

bool qwen4exp_replay_accepted_prefix(
        const Qwen4ExpState & committed_state,
        const std::vector<float> & committed_logits,
        Qwen4ExpState & verified_state,
        std::vector<float> & verified_logits,
        const std::vector<Qwen4ExpReplayRow> & draft_rows,
        size_t accepted_rows,
        Qwen4ExpReplayStep step,
        void * opaque,
        Qwen4ExpReplayResult & result,
        std::string & error) {
    result = {};
    error.clear();
    if (accepted_rows > draft_rows.size()) {
        error = "Qwen4Exp MTP accepted row count exceeds the verified draft";
        return false;
    }

    // Complete acceptance is the only case where the batched verifier's state
    // is authoritative. There are no rejected recurrent intermediates to
    // discard, so keep the fast path without serial replay.
    if (accepted_rows == draft_rows.size()) {
        result.disposition = Qwen4ExpReplayDisposition::FullAcceptance;
        return true;
    }

    verified_state = committed_state;
    verified_logits = committed_logits;
    if (accepted_rows == 0) {
        result.disposition = Qwen4ExpReplayDisposition::RestoredCheckpoint;
        return true;
    }
    if (!step) {
        error = "Qwen4Exp MTP partial acceptance requires a q=1 replay step";
        return false;
    }

    for (size_t i = 0; i < accepted_rows; ++i) {
        if (!step(opaque, verified_state, draft_rows[i], verified_logits,
                  error)) {
            // Fail closed at the same complete frontier supplied by the
            // caller. A half-replayed GDN/PLE/QSA state must never escape.
            verified_state = committed_state;
            verified_logits = committed_logits;
            result = {};
            if (error.empty()) error = "Qwen4Exp MTP accepted-prefix replay failed";
            return false;
        }
        ++result.rows_replayed;
    }
    result.disposition = Qwen4ExpReplayDisposition::ReplayedAcceptedPrefix;
    return true;
}

} // namespace dflash::common
