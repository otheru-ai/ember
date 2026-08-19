// Engine-level policy for multi-session continuous batching.
//
// This class deliberately owns no model or KV objects. It turns resident
// session state into one serialized engine submission at a time:
//
//   - every decode-ready session advances in the same batch;
//   - at most one prefill suffix joins that batch, in a small quantum;
//   - with no active generation, prefill advances in a larger quantum;
//   - briefly coalesce decode-ready sessions to avoid issuing batch-size-one
//     work while peers are about to become ready.
//
// The DeepSeek backend will bind session ids to independent KV/cache state and
// execute ContinuousBatchPlan. Keeping policy separate makes its fairness and
// lifecycle invariants GPU-free and deterministic to test. The scheduler is
// intentionally not internally synchronized: one engine coordinator owns it,
// and admission/cancellation messages enter through that coordinator's queue.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace dflash::common {

using ContinuousBatchSessionId = std::uint64_t;
using ContinuousBatchSubmissionId = std::uint64_t;

struct ContinuousBatchConfig {
    std::size_t max_sessions = 1;
    int prefill_quantum = 2048;
    int mixed_prefill_quantum = 128;
    std::int64_t decode_coalesce_us = 2000;
};

enum class ContinuousBatchSessionState {
    Empty,
    PrefillReady,
    PrefillInFlight,
    DecodeIdle,
    DecodeReady,
    DecodeInFlight,
    Finished,
    Cancelled,
    Failed,
};

struct ContinuousBatchPlan {
    // Non-zero identity shared by every component of this engine submission.
    // Completion calls must echo it, preventing a late result from an older
    // step from completing newer work for the same resident session.
    ContinuousBatchSubmissionId submission_id = 0;
    std::vector<ContinuousBatchSessionId> decode_sessions;
    ContinuousBatchSessionId prefill_session = 0;
    int prefill_tokens = 0;
    // When no work is returned because a partial decode batch is coalescing,
    // this is the monotonic deadline at which the caller should ask again.
    // -1 means wait for a state change.
    std::int64_t wake_at_us = -1;

    bool empty() const {
        return decode_sessions.empty() && prefill_session == 0;
    }
    bool mixed() const {
        return !decode_sessions.empty() && prefill_session != 0;
    }
};

struct ContinuousBatchSessionInfo {
    ContinuousBatchSessionState state = ContinuousBatchSessionState::Empty;
    int prompt_tokens = 0;
    int prefilled_tokens = 0;
    int max_new_tokens = 0;
    int generated_tokens = 0;
    bool cancel_requested = false;
};

struct ContinuousBatchStats {
    // Instantaneous state.
    std::size_t resident = 0;
    std::size_t prefill_ready = 0;
    std::size_t decode_idle = 0;
    std::size_t decode_ready = 0;
    std::size_t in_flight = 0;
    std::size_t terminal = 0;

    // Lifetime counters.
    std::uint64_t admissions = 0;
    std::uint64_t releases = 0;
    std::uint64_t cancellations = 0;
    std::uint64_t failures = 0;
    std::uint64_t submissions = 0;
    std::uint64_t decode_batches = 0;
    std::uint64_t decode_rows_scheduled = 0;
    std::uint64_t decode_tokens_completed = 0;
    std::uint64_t prefill_batches = 0;
    std::uint64_t prefill_tokens_scheduled = 0;
    std::uint64_t prefill_tokens_completed = 0;
    std::uint64_t mixed_submissions = 0;
    std::uint64_t coalesce_waits = 0;
    std::size_t max_decode_batch = 0;
};

class ContinuousBatchScheduler {
public:
    explicit ContinuousBatchScheduler(ContinuousBatchConfig config);

    // Returns a generation-stamped id. Reusing a released slot never reuses an
    // old id, so a late completion/cancellation cannot mutate its new owner.
    std::optional<ContinuousBatchSessionId> admit(int prompt_tokens,
                                                   int max_new_tokens);

    // A prefilled session becomes decode-ready only after its owner has sampled
    // the next token. now_us must use the same monotonic clock as plan().
    bool mark_decode_ready(ContinuousBatchSessionId id, std::int64_t now_us);

    // Produce the next single engine submission. Until every component of the
    // returned plan is completed, subsequent calls return an empty plan.
    ContinuousBatchPlan plan(std::int64_t now_us);

    // consumed_tokens may be smaller than the requested quantum. Zero progress
    // on a successful operation is rejected to prevent a scheduler spin.
    bool complete_prefill(ContinuousBatchSubmissionId submission_id,
                          ContinuousBatchSessionId id, int consumed_tokens,
                          bool ok);

    // One completion per decode session in the current plan. terminal is EOS or
    // another clean stop. completed_tokens is the exact output width committed
    // by the backend (one for AR, potentially wider for speculative decode);
    // max_new_tokens is enforced independently.
    bool complete_decode(ContinuousBatchSubmissionId submission_id,
                         ContinuousBatchSessionId id, bool ok, bool terminal,
                         int completed_tokens = 1);

    bool cancel(ContinuousBatchSessionId id);
    // Coordinator-side terminal transitions for a resident backend that
    // discovers completion/failure between engine submissions.
    bool finish(ContinuousBatchSessionId id);
    bool fail(ContinuousBatchSessionId id);
    // Slots are retained after a terminal state so callers can inspect results.
    // release() makes one available for a new generation-stamped admission.
    bool release(ContinuousBatchSessionId id);

    std::optional<ContinuousBatchSessionInfo>
    session(ContinuousBatchSessionId id) const;

    std::size_t resident() const;
    std::size_t capacity() const { return slots_.size(); }
    std::size_t decode_ready() const;
    bool submission_in_flight() const;
    ContinuousBatchStats stats() const;

private:
    struct Slot {
        ContinuousBatchSessionId id = 0;
        ContinuousBatchSessionInfo info;
        std::int64_t ready_at_us = -1;
        ContinuousBatchSubmissionId in_flight_submission = 0;
    };

    ContinuousBatchConfig config_;
    std::vector<Slot> slots_;
    std::size_t last_prefill_slot_ = 0;
    std::uint64_t next_session_id_ = 1;
    std::uint64_t next_submission_id_ = 1;
    std::int64_t coalesce_deadline_us_ = -1;
    ContinuousBatchStats lifetime_stats_;

    Slot *find(ContinuousBatchSessionId id);
    const Slot *find(ContinuousBatchSessionId id) const;
    std::optional<std::size_t> choose_prefill_slot() const;
    std::size_t active_decode_sessions() const;
};

}  // namespace dflash::common
