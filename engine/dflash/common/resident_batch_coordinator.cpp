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
    // Admission is transactional from here. Each failure return below used to
    // release the slot by hand, which was correct for a returned false and
    // wrong for a throw -- and nothing on this path is noexcept. Backend
    // creation allocates and constructs a random_device, session status is
    // backend code, and the sessions_ push_back can throw on its own. An
    // exception therefore left the scheduler slot allocated with no tracked
    // lease able to release it: sessions() could not enumerate it, shutdown
    // could not reclaim it, and the capacity was gone for the life of the
    // coordinator. With the wake predicate now correctly requiring admission
    // capacity, the worker then sleeps on a capacity that can never free,
    // which turns a leak into a stall.
    //
    // The guard owns every failure path, including the returns, so the two
    // cannot drift apart again.
    struct AdmissionRollback {
        ResidentBatchBackend *backend;
        ContinuousBatchScheduler *scheduler;
        ContinuousBatchSessionId id;
        bool backend_created;
        bool armed;
        ~AdmissionRollback() {
            if (!armed) return;
            // Runs while an exception may be in flight, so it must not throw.
            if (backend_created) {
                try {
                    (void)backend->resident_session_destroy(id);
                } catch (...) {
                }
            }
            try {
                (void)scheduler->cancel(id);
                (void)scheduler->release(id);
            } catch (...) {
            }
        }
    } rollback{&backend_, &scheduler_, *id, false, true};

    std::string backend_error;
    if (!backend_.resident_session_create(*id, request, io, restore_slot,
                                          &backend_error)) {
        if (error) {
            *error = backend_error.empty()
                ? "resident backend rejected session" : backend_error;
        }
        return std::nullopt;
    }
    rollback.backend_created = true;
    const ResidentBatchBackend::SessionStatus status =
        backend_.resident_session_status(*id);
    if (status.failed || status.prefilled_tokens != restored_prompt_tokens) {
        if (error) *error = "resident backend restored an unexpected frontier";
        return std::nullopt;
    }
    sessions_.push_back(*id);
    rollback.armed = false;
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
