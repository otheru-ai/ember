#include "qwen4exp_mtp.h"

#include <algorithm>

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
