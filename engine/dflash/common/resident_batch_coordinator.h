// Resident model-session coordinator.
//
// This composes ContinuousBatchScheduler, ContinuousBatchExecutor and a
// model-coupled ResidentBatchBackend. It is single-thread owned; a serving
// thread queues admissions/cancellations around pump().
#pragma once

#include "model_backend.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dflash::common {

class ResidentBatchCoordinator {
public:
    ResidentBatchCoordinator(ResidentBatchBackend &backend,
                             ContinuousBatchConfig config);
    ~ResidentBatchCoordinator();

    std::optional<ContinuousBatchSessionId> admit(
        const GenerateRequest &request,
        const DaemonIO &io,
        int restore_slot,
        int restored_prompt_tokens,
        std::string *error);

    ContinuousBatchRunResult pump(std::int64_t now_us);
    bool cancel(ContinuousBatchSessionId id);
    bool terminal(ContinuousBatchSessionId id) const;
    std::optional<GenerateResult> result(
        ContinuousBatchSessionId id) const;
    bool release(ContinuousBatchSessionId id);

    const ContinuousBatchScheduler &scheduler() const { return scheduler_; }
    const ContinuousBatchExecutorStats &executor_stats() const {
        return executor_.stats();
    }
    const std::vector<ContinuousBatchSessionId> &sessions() const {
        return sessions_;
    }

private:
    ResidentBatchBackend &backend_;
    ContinuousBatchScheduler scheduler_;
    ContinuousBatchExecutor executor_;
    std::vector<ContinuousBatchSessionId> sessions_;

    void reconcile(std::int64_t now_us);
};

}  // namespace dflash::common
