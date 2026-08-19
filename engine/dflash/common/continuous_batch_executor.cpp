#include "continuous_batch_executor.h"

#include <algorithm>

namespace dflash::common {

ContinuousBatchMixedCompletion
ContinuousBatchWorkBackend::execute_mixed(
        ContinuousBatchSessionId,
        int,
        const std::vector<ContinuousBatchSessionId> &) {
    // A backend must explicitly advertise and implement native mixed work.
    return {};
}

bool ContinuousBatchExecutor::valid_prefill(
        const ContinuousBatchPlan &plan,
        const ContinuousBatchPrefillCompletion &result) const {
    if (plan.prefill_session == 0) return true;
    if (!result.ok) return true;
    return result.consumed_tokens > 0 &&
           result.consumed_tokens <= plan.prefill_tokens;
}

bool ContinuousBatchExecutor::valid_decode(
        const ContinuousBatchPlan &plan,
        const std::vector<ContinuousBatchDecodeCompletion> &result) const {
    if (result.size() != plan.decode_sessions.size()) return false;
    for (ContinuousBatchSessionId expected : plan.decode_sessions) {
        int matches = 0;
        for (const ContinuousBatchDecodeCompletion &row : result) {
            if (row.session_id == expected) ++matches;
        }
        if (matches != 1) return false;
    }
    for (const ContinuousBatchDecodeCompletion &row : result) {
        if (std::find(plan.decode_sessions.begin(), plan.decode_sessions.end(),
                      row.session_id) == plan.decode_sessions.end()) {
            return false;
        }
        if (row.ok) {
            const auto session = scheduler_.session(row.session_id);
            if (!session || row.completed_tokens <= 0 ||
                row.completed_tokens >
                    session->max_new_tokens - session->generated_tokens) {
                return false;
            }
        }
    }
    return true;
}

void ContinuousBatchExecutor::fail_prefill(
        const ContinuousBatchPlan &plan) {
    if (plan.prefill_session == 0) return;
    if (!scheduler_.complete_prefill(plan.submission_id,
                                     plan.prefill_session, 0, false)) {
        ++stats_.completion_rejections;
    }
}

void ContinuousBatchExecutor::fail_decode(
        const ContinuousBatchPlan &plan) {
    for (ContinuousBatchSessionId id : plan.decode_sessions) {
        if (!scheduler_.complete_decode(plan.submission_id,
                                        id, false, false)) {
            ++stats_.completion_rejections;
        }
    }
}

void ContinuousBatchExecutor::fail_all(const ContinuousBatchPlan &plan) {
    fail_prefill(plan);
    fail_decode(plan);
}

void ContinuousBatchExecutor::apply_prefill(
        const ContinuousBatchPlan &plan,
        const ContinuousBatchPrefillCompletion &result) {
    if (plan.prefill_session == 0) return;
    if (!result.ok) ++stats_.backend_failures;
    if (!scheduler_.complete_prefill(plan.submission_id,
                                     plan.prefill_session,
                                     result.consumed_tokens,
                                     result.ok)) {
        ++stats_.completion_rejections;
    }
}

void ContinuousBatchExecutor::apply_decode(
        const ContinuousBatchPlan &plan,
        const std::vector<ContinuousBatchDecodeCompletion> &result) {
    for (ContinuousBatchSessionId expected : plan.decode_sessions) {
        const auto it = std::find_if(
            result.begin(), result.end(),
            [expected](const ContinuousBatchDecodeCompletion &row) {
                return row.session_id == expected;
            });
        if (!it->ok) ++stats_.backend_failures;
        if (!scheduler_.complete_decode(plan.submission_id, expected,
                                        it->ok, it->terminal,
                                        it->completed_tokens)) {
            ++stats_.completion_rejections;
        }
    }
}

ContinuousBatchRunResult
ContinuousBatchExecutor::run_once(std::int64_t now_us) {
    ContinuousBatchRunResult run;
    ContinuousBatchPlan plan = scheduler_.plan(now_us);
    if (plan.empty()) {
        if (plan.wake_at_us >= 0) {
            run.status = ContinuousBatchRunStatus::Waiting;
            run.wake_at_us = plan.wake_at_us;
        }
        return run;
    }

    run.status = ContinuousBatchRunStatus::Completed;
    run.submission_id = plan.submission_id;
    ++stats_.plans_executed;

    if (plan.mixed() && backend_.supports_native_mixed()) {
        ++stats_.native_mixed;
        try {
            ContinuousBatchMixedCompletion result =
                backend_.execute_mixed(plan.prefill_session,
                                       plan.prefill_tokens,
                                       plan.decode_sessions);
            if (!valid_prefill(plan, result.prefill) ||
                !valid_decode(plan, result.decode)) {
                ++stats_.malformed_results;
                fail_all(plan);
                return run;
            }
            apply_prefill(plan, result.prefill);
            apply_decode(plan, result.decode);
        } catch (...) {
            ++stats_.backend_exceptions;
            fail_all(plan);
        }
        return run;
    }

    if (plan.mixed()) ++stats_.fallback_mixed;
    if (plan.prefill_session != 0) {
        ++stats_.prefill_calls;
        ContinuousBatchPrefillCompletion result;
        try {
            result = backend_.prefill(plan.prefill_session,
                                      plan.prefill_tokens);
        } catch (...) {
            ++stats_.backend_exceptions;
            fail_prefill(plan);
            stats_.aborted_components += plan.decode_sessions.size();
            fail_decode(plan);
            return run;
        }
        if (!valid_prefill(plan, result)) {
            ++stats_.malformed_results;
            fail_prefill(plan);
            stats_.aborted_components += plan.decode_sessions.size();
            fail_decode(plan);
            return run;
        }
        apply_prefill(plan, result);
        if (!result.ok) {
            stats_.aborted_components += plan.decode_sessions.size();
            fail_decode(plan);
            return run;
        }
    }

    if (!plan.decode_sessions.empty()) {
        ++stats_.decode_batch_calls;
        std::vector<ContinuousBatchDecodeCompletion> result;
        try {
            result = backend_.decode_batch(plan.decode_sessions);
        } catch (...) {
            ++stats_.backend_exceptions;
            fail_decode(plan);
            return run;
        }
        if (!valid_decode(plan, result)) {
            ++stats_.malformed_results;
            fail_decode(plan);
            return run;
        }
        apply_decode(plan, result);
    }
    return run;
}

}  // namespace dflash::common
