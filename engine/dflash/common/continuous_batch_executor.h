// Execution bridge for ContinuousBatchScheduler plans.
//
// The scheduler owns lifecycle/fairness; this layer owns the synchronous
// scheduler-to-engine transaction. Backends may implement one native mixed
// prefill/decode submission, or rely on the correctness-first fallback
// (prefill first, then one decode batch). Every returned decode row is
// validated before scheduler state is committed.
#pragma once

#include "continuous_batch_scheduler.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

struct ContinuousBatchPrefillCompletion {
    bool ok = false;
    int consumed_tokens = 0;
};

struct ContinuousBatchDecodeCompletion {
    ContinuousBatchSessionId session_id = 0;
    bool ok = false;
    bool terminal = false;
};

struct ContinuousBatchMixedCompletion {
    ContinuousBatchPrefillCompletion prefill;
    std::vector<ContinuousBatchDecodeCompletion> decode;
};

class ContinuousBatchWorkBackend {
public:
    virtual ~ContinuousBatchWorkBackend() = default;

    virtual ContinuousBatchPrefillCompletion
    prefill(ContinuousBatchSessionId session_id, int requested_tokens) = 0;

    // Results may be returned in any order, but must contain every requested
    // session exactly once and no other session.
    virtual std::vector<ContinuousBatchDecodeCompletion>
    decode_batch(const std::vector<ContinuousBatchSessionId> &sessions) = 0;

    virtual bool supports_native_mixed() const { return false; }

    // Called only when supports_native_mixed() is true and the plan contains
    // both work classes.
    virtual ContinuousBatchMixedCompletion
    execute_mixed(ContinuousBatchSessionId prefill_session,
                  int requested_prefill_tokens,
                  const std::vector<ContinuousBatchSessionId> &decode_sessions);
};

enum class ContinuousBatchRunStatus {
    Idle,
    Waiting,
    Completed,
};

struct ContinuousBatchRunResult {
    ContinuousBatchRunStatus status = ContinuousBatchRunStatus::Idle;
    std::int64_t wake_at_us = -1;
    ContinuousBatchSubmissionId submission_id = 0;
};

struct ContinuousBatchExecutorStats {
    std::uint64_t plans_executed = 0;
    std::uint64_t native_mixed = 0;
    std::uint64_t fallback_mixed = 0;
    std::uint64_t prefill_calls = 0;
    std::uint64_t decode_batch_calls = 0;
    std::uint64_t backend_failures = 0;
    std::uint64_t backend_exceptions = 0;
    std::uint64_t malformed_results = 0;
    std::uint64_t aborted_components = 0;
    std::uint64_t completion_rejections = 0;
};

class ContinuousBatchExecutor {
public:
    ContinuousBatchExecutor(ContinuousBatchScheduler &scheduler,
                            ContinuousBatchWorkBackend &backend)
        : scheduler_(scheduler), backend_(backend) {}

    ContinuousBatchRunResult run_once(std::int64_t now_us);
    const ContinuousBatchExecutorStats &stats() const { return stats_; }

private:
    ContinuousBatchScheduler &scheduler_;
    ContinuousBatchWorkBackend &backend_;
    ContinuousBatchExecutorStats stats_;

    bool valid_prefill(const ContinuousBatchPlan &plan,
                       const ContinuousBatchPrefillCompletion &result) const;
    bool valid_decode(
        const ContinuousBatchPlan &plan,
        const std::vector<ContinuousBatchDecodeCompletion> &result) const;
    void fail_prefill(const ContinuousBatchPlan &plan);
    void fail_decode(const ContinuousBatchPlan &plan);
    void fail_all(const ContinuousBatchPlan &plan);
    void apply_prefill(const ContinuousBatchPlan &plan,
                       const ContinuousBatchPrefillCompletion &result);
    void apply_decode(
        const ContinuousBatchPlan &plan,
        const std::vector<ContinuousBatchDecodeCompletion> &result);
};

}  // namespace dflash::common
