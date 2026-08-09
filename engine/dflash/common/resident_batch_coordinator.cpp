#include "resident_batch_coordinator.h"

#include <algorithm>

namespace dflash::common {

ResidentBatchCoordinator::ResidentBatchCoordinator(
        ResidentBatchBackend &backend,
        ContinuousBatchConfig config)
    : backend_(backend), scheduler_(config), executor_(scheduler_, backend_) {}

ResidentBatchCoordinator::~ResidentBatchCoordinator() {
    for (ContinuousBatchSessionId id : sessions_) {
        (void)backend_.resident_session_destroy(id);
    }
}

std::optional<ContinuousBatchSessionId>
ResidentBatchCoordinator::admit(
        const GenerateRequest &request,
        const DaemonIO &io,
        int restore_slot,
        int restored_prompt_tokens,
        std::string *error) {
    if (error) error->clear();
    if (restored_prompt_tokens < 0 ||
        restored_prompt_tokens > (int)request.prompt.size()) {
        if (error) *error = "invalid restored prompt position";
        return std::nullopt;
    }
    const int prefill_tokens =
        (int)request.prompt.size() - restored_prompt_tokens;
    auto id = scheduler_.admit(prefill_tokens, request.n_gen);
    if (!id) {
        if (error) *error = "resident session capacity exhausted";
        return std::nullopt;
    }
    std::string backend_error;
    if (!backend_.resident_session_create(*id, request, io, restore_slot,
                                          &backend_error)) {
        (void)scheduler_.cancel(*id);
        (void)scheduler_.release(*id);
        if (error) {
            *error = backend_error.empty()
                ? "resident backend rejected session" : backend_error;
        }
        return std::nullopt;
    }
    const ResidentBatchBackend::SessionStatus status =
        backend_.resident_session_status(*id);
    if (status.failed || status.prefilled_tokens != restored_prompt_tokens) {
        (void)backend_.resident_session_destroy(*id);
        (void)scheduler_.cancel(*id);
        (void)scheduler_.release(*id);
        if (error) *error = "resident backend restored an unexpected frontier";
        return std::nullopt;
    }
    sessions_.push_back(*id);
    reconcile(/*now_us=*/0);
    return id;
}

void ResidentBatchCoordinator::reconcile(std::int64_t now_us) {
    for (ContinuousBatchSessionId id : sessions_) {
        auto info = scheduler_.session(id);
        if (!info) continue;
        const ContinuousBatchSessionState state = info->state;
        if (state == ContinuousBatchSessionState::Finished ||
            state == ContinuousBatchSessionState::Cancelled ||
            state == ContinuousBatchSessionState::Failed ||
            state == ContinuousBatchSessionState::PrefillInFlight ||
            state == ContinuousBatchSessionState::DecodeInFlight) {
            continue;
        }

        const ResidentBatchBackend::SessionStatus status =
            backend_.resident_session_status(id);
        if (status.failed) {
            (void)scheduler_.fail(id);
        } else if (status.cancelled) {
            (void)scheduler_.cancel(id);
        } else if (status.terminal) {
            (void)scheduler_.finish(id);
        } else if (status.decode_ready &&
                   state == ContinuousBatchSessionState::DecodeIdle) {
            (void)scheduler_.mark_decode_ready(id, now_us);
        }
    }
}

ContinuousBatchRunResult ResidentBatchCoordinator::pump(
        std::int64_t now_us) {
    reconcile(now_us);
    ContinuousBatchRunResult run = executor_.run_once(now_us);
    reconcile(now_us);
    return run;
}

bool ResidentBatchCoordinator::cancel(ContinuousBatchSessionId id) {
    if (!scheduler_.cancel(id)) return false;
    return backend_.resident_session_cancel(id);
}

bool ResidentBatchCoordinator::terminal(
        ContinuousBatchSessionId id) const {
    auto info = scheduler_.session(id);
    return info &&
        (info->state == ContinuousBatchSessionState::Finished ||
         info->state == ContinuousBatchSessionState::Cancelled ||
         info->state == ContinuousBatchSessionState::Failed);
}

std::optional<GenerateResult>
ResidentBatchCoordinator::result(ContinuousBatchSessionId id) const {
    auto info = scheduler_.session(id);
    if (!info ||
        (info->state != ContinuousBatchSessionState::Finished &&
         info->state != ContinuousBatchSessionState::Cancelled &&
         info->state != ContinuousBatchSessionState::Failed)) {
        return std::nullopt;
    }
    GenerateResult result = backend_.resident_session_result(id);
    if (info->state == ContinuousBatchSessionState::Failed && result.ok()) {
        result.fail(GenerateErrorCode::BackendSpecific,
                    "continuous batch submission failed");
    }
    return result;
}

bool ResidentBatchCoordinator::release(ContinuousBatchSessionId id) {
    if (!terminal(id)) return false;
    if (!backend_.resident_session_destroy(id)) return false;
    if (!scheduler_.release(id)) return false;
    sessions_.erase(std::remove(sessions_.begin(), sessions_.end(), id),
                    sessions_.end());
    return true;
}

}  // namespace dflash::common
