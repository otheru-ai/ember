#include "continuous_batch_scheduler.h"

#include <algorithm>
#include <limits>

namespace dflash::common {
namespace {

bool is_terminal(ContinuousBatchSessionState state) {
    return state == ContinuousBatchSessionState::Finished ||
           state == ContinuousBatchSessionState::Cancelled ||
           state == ContinuousBatchSessionState::Failed;
}

bool is_decode_active(ContinuousBatchSessionState state) {
    return state == ContinuousBatchSessionState::DecodeIdle ||
           state == ContinuousBatchSessionState::DecodeReady ||
           state == ContinuousBatchSessionState::DecodeInFlight;
}

}  // namespace

ContinuousBatchScheduler::ContinuousBatchScheduler(ContinuousBatchConfig config)
    : config_(config) {
    if (config_.max_sessions == 0) config_.max_sessions = 1;
    if (config_.prefill_quantum <= 0) config_.prefill_quantum = 1;
    if (config_.mixed_prefill_quantum <= 0) config_.mixed_prefill_quantum = 1;
    if (config_.decode_coalesce_us < 0) config_.decode_coalesce_us = 0;
    slots_.resize(config_.max_sessions);
    last_prefill_slot_ = slots_.empty() ? 0 : slots_.size() - 1;
}

std::optional<ContinuousBatchSessionId>
ContinuousBatchScheduler::admit(int prompt_tokens, int max_new_tokens) {
    if (prompt_tokens < 0 || max_new_tokens < 0) return std::nullopt;
    for (Slot &slot : slots_) {
        if (slot.info.state != ContinuousBatchSessionState::Empty) continue;
        // Zero remains the "no session" sentinel in plans. Wraparound is only
        // theoretical, but skip ids still owned by resident slots so the stale
        // completion guarantee remains true even then.
        for (;;) {
            slot.id = next_session_id_++;
            if (slot.id == 0) continue;
            bool duplicate = false;
            for (const Slot &other : slots_) {
                if (&other != &slot && other.id == slot.id &&
                    other.info.state != ContinuousBatchSessionState::Empty) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) break;
        }
        slot.info = {};
        slot.info.prompt_tokens = prompt_tokens;
        slot.info.max_new_tokens = max_new_tokens;
        if (prompt_tokens > 0) {
            slot.info.state = ContinuousBatchSessionState::PrefillReady;
        } else if (max_new_tokens > 0) {
            slot.info.state = ContinuousBatchSessionState::DecodeIdle;
        } else {
            slot.info.state = ContinuousBatchSessionState::Finished;
        }
        slot.ready_at_us = -1;
        slot.in_flight_submission = 0;
        ++lifetime_stats_.admissions;
        return slot.id;
    }
    return std::nullopt;
}

ContinuousBatchScheduler::Slot *
ContinuousBatchScheduler::find(ContinuousBatchSessionId id) {
    if (id == 0) return nullptr;
    for (Slot &slot : slots_) {
        if (slot.id == id &&
            slot.info.state != ContinuousBatchSessionState::Empty) {
            return &slot;
        }
    }
    return nullptr;
}

const ContinuousBatchScheduler::Slot *
ContinuousBatchScheduler::find(ContinuousBatchSessionId id) const {
    return const_cast<ContinuousBatchScheduler *>(this)->find(id);
}

bool ContinuousBatchScheduler::mark_decode_ready(
        ContinuousBatchSessionId id, std::int64_t now_us) {
    Slot *slot = find(id);
    if (!slot || slot->info.state != ContinuousBatchSessionState::DecodeIdle ||
        slot->info.cancel_requested) {
        return false;
    }
    slot->info.state = ContinuousBatchSessionState::DecodeReady;
    slot->ready_at_us = now_us;
    return true;
}

std::optional<std::size_t>
ContinuousBatchScheduler::choose_prefill_slot() const {
    if (slots_.empty()) return std::nullopt;
    for (std::size_t n = 1; n <= slots_.size(); ++n) {
        const std::size_t index = (last_prefill_slot_ + n) % slots_.size();
        if (slots_[index].info.state ==
            ContinuousBatchSessionState::PrefillReady) {
            return index;
        }
    }
    return std::nullopt;
}

std::size_t ContinuousBatchScheduler::active_decode_sessions() const {
    std::size_t count = 0;
    for (const Slot &slot : slots_) {
        if (is_decode_active(slot.info.state) &&
            !slot.info.cancel_requested) {
            ++count;
        }
    }
    return count;
}

ContinuousBatchPlan ContinuousBatchScheduler::plan(std::int64_t now_us) {
    ContinuousBatchPlan result;
    if (submission_in_flight()) return result;

    std::int64_t earliest_ready = std::numeric_limits<std::int64_t>::max();
    for (const Slot &slot : slots_) {
        if (slot.info.state != ContinuousBatchSessionState::DecodeReady) continue;
        result.decode_sessions.push_back(slot.id);
        earliest_ready = std::min(earliest_ready, slot.ready_at_us);
    }

    const std::size_t active = active_decode_sessions();
    if (!result.decode_sessions.empty() &&
        result.decode_sessions.size() < active &&
        config_.decode_coalesce_us > 0) {
        if (coalesce_deadline_us_ < 0) {
            coalesce_deadline_us_ =
                earliest_ready > std::numeric_limits<std::int64_t>::max() -
                                     config_.decode_coalesce_us
                    ? std::numeric_limits<std::int64_t>::max()
                    : earliest_ready + config_.decode_coalesce_us;
            ++lifetime_stats_.coalesce_waits;
        }
        if (now_us < coalesce_deadline_us_) {
            result.decode_sessions.clear();
            result.wake_at_us = coalesce_deadline_us_;
            return result;
        }
    } else {
        coalesce_deadline_us_ = -1;
    }

    const std::optional<std::size_t> prefill = choose_prefill_slot();
    if (prefill) {
        Slot &slot = slots_[*prefill];
        const int remaining =
            slot.info.prompt_tokens - slot.info.prefilled_tokens;
        // Match Dwarfstar's server_prefill_quantum(): once any generation is
        // active, use the small mixed quantum even during the brief interval
        // where no decode row is ready. A full idle-prefill chunk here could
        // otherwise add head-of-line latency to the next decode step.
        const int quantum = active == 0
            ? config_.prefill_quantum : config_.mixed_prefill_quantum;
        result.prefill_session = slot.id;
        result.prefill_tokens = std::min(remaining, quantum);
    }

    if (result.empty()) return result;
    result.submission_id = next_submission_id_++;
    if (result.submission_id == 0) result.submission_id = next_submission_id_++;
    ++lifetime_stats_.submissions;
    if (!result.decode_sessions.empty()) {
        ++lifetime_stats_.decode_batches;
        lifetime_stats_.decode_rows_scheduled += result.decode_sessions.size();
        lifetime_stats_.max_decode_batch =
            std::max(lifetime_stats_.max_decode_batch,
                     result.decode_sessions.size());
    }
    if (result.prefill_session != 0) {
        ++lifetime_stats_.prefill_batches;
        lifetime_stats_.prefill_tokens_scheduled += result.prefill_tokens;
    }
    if (result.mixed()) ++lifetime_stats_.mixed_submissions;
    coalesce_deadline_us_ = -1;
    for (ContinuousBatchSessionId id : result.decode_sessions) {
        Slot *slot = find(id);
        slot->info.state = ContinuousBatchSessionState::DecodeInFlight;
        slot->in_flight_submission = result.submission_id;
    }
    if (result.prefill_session != 0) {
        Slot *slot = find(result.prefill_session);
        slot->info.state = ContinuousBatchSessionState::PrefillInFlight;
        slot->in_flight_submission = result.submission_id;
        last_prefill_slot_ =
            (std::size_t)(slot - slots_.data());
    }
    return result;
}

bool ContinuousBatchScheduler::complete_prefill(
        ContinuousBatchSubmissionId submission_id,
        ContinuousBatchSessionId id, int consumed_tokens, bool ok) {
    Slot *slot = find(id);
    if (!slot ||
        slot->info.state != ContinuousBatchSessionState::PrefillInFlight ||
        submission_id == 0 ||
        slot->in_flight_submission != submission_id) {
        return false;
    }
    if (!ok) {
        slot->in_flight_submission = 0;
        slot->info.state = slot->info.cancel_requested
            ? ContinuousBatchSessionState::Cancelled
            : ContinuousBatchSessionState::Failed;
        if (!slot->info.cancel_requested) ++lifetime_stats_.failures;
        return true;
    }
    const int remaining =
        slot->info.prompt_tokens - slot->info.prefilled_tokens;
    if (consumed_tokens <= 0 || consumed_tokens > remaining) return false;
    slot->info.prefilled_tokens += consumed_tokens;
    slot->in_flight_submission = 0;
    lifetime_stats_.prefill_tokens_completed += consumed_tokens;
    if (slot->info.cancel_requested) {
        slot->info.state = ContinuousBatchSessionState::Cancelled;
    } else if (slot->info.prefilled_tokens < slot->info.prompt_tokens) {
        slot->info.state = ContinuousBatchSessionState::PrefillReady;
    } else if (slot->info.max_new_tokens > 0) {
        slot->info.state = ContinuousBatchSessionState::DecodeIdle;
    } else {
        slot->info.state = ContinuousBatchSessionState::Finished;
    }
    return true;
}

bool ContinuousBatchScheduler::complete_decode(
        ContinuousBatchSubmissionId submission_id,
        ContinuousBatchSessionId id, bool ok, bool terminal) {
    Slot *slot = find(id);
    if (!slot ||
        slot->info.state != ContinuousBatchSessionState::DecodeInFlight ||
        submission_id == 0 ||
        slot->in_flight_submission != submission_id) {
        return false;
    }
    if (!ok) {
        slot->in_flight_submission = 0;
        slot->info.state = slot->info.cancel_requested
            ? ContinuousBatchSessionState::Cancelled
            : ContinuousBatchSessionState::Failed;
        if (!slot->info.cancel_requested) ++lifetime_stats_.failures;
        return true;
    }
    ++slot->info.generated_tokens;
    slot->in_flight_submission = 0;
    ++lifetime_stats_.decode_tokens_completed;
    if (slot->info.cancel_requested) {
        slot->info.state = ContinuousBatchSessionState::Cancelled;
    } else if (terminal ||
               slot->info.generated_tokens >= slot->info.max_new_tokens) {
        slot->info.state = ContinuousBatchSessionState::Finished;
    } else {
        slot->info.state = ContinuousBatchSessionState::DecodeIdle;
    }
    slot->ready_at_us = -1;
    return true;
}

bool ContinuousBatchScheduler::cancel(ContinuousBatchSessionId id) {
    Slot *slot = find(id);
    if (!slot || is_terminal(slot->info.state) ||
        slot->info.cancel_requested) return false;
    ++lifetime_stats_.cancellations;
    if (slot->info.state == ContinuousBatchSessionState::PrefillInFlight ||
        slot->info.state == ContinuousBatchSessionState::DecodeInFlight) {
        slot->info.cancel_requested = true;
    } else {
        slot->info.state = ContinuousBatchSessionState::Cancelled;
    }
    return true;
}

bool ContinuousBatchScheduler::finish(ContinuousBatchSessionId id) {
    Slot *slot = find(id);
    if (!slot || is_terminal(slot->info.state) ||
        slot->info.state == ContinuousBatchSessionState::PrefillInFlight ||
        slot->info.state == ContinuousBatchSessionState::DecodeInFlight) {
        return false;
    }
    slot->info.state = ContinuousBatchSessionState::Finished;
    slot->ready_at_us = -1;
    return true;
}

bool ContinuousBatchScheduler::fail(ContinuousBatchSessionId id) {
    Slot *slot = find(id);
    if (!slot || is_terminal(slot->info.state) ||
        slot->info.state == ContinuousBatchSessionState::PrefillInFlight ||
        slot->info.state == ContinuousBatchSessionState::DecodeInFlight) {
        return false;
    }
    slot->info.state = ContinuousBatchSessionState::Failed;
    slot->ready_at_us = -1;
    ++lifetime_stats_.failures;
    return true;
}

bool ContinuousBatchScheduler::release(ContinuousBatchSessionId id) {
    Slot *slot = find(id);
    if (!slot || !is_terminal(slot->info.state)) return false;
    slot->id = 0;
    slot->info = {};
    slot->ready_at_us = -1;
    slot->in_flight_submission = 0;
    ++lifetime_stats_.releases;
    return true;
}

std::optional<ContinuousBatchSessionInfo>
ContinuousBatchScheduler::session(ContinuousBatchSessionId id) const {
    const Slot *slot = find(id);
    if (!slot) return std::nullopt;
    return slot->info;
}

std::size_t ContinuousBatchScheduler::resident() const {
    std::size_t count = 0;
    for (const Slot &slot : slots_) {
        if (slot.info.state != ContinuousBatchSessionState::Empty) ++count;
    }
    return count;
}

std::size_t ContinuousBatchScheduler::decode_ready() const {
    std::size_t count = 0;
    for (const Slot &slot : slots_) {
        if (slot.info.state == ContinuousBatchSessionState::DecodeReady) ++count;
    }
    return count;
}

bool ContinuousBatchScheduler::submission_in_flight() const {
    for (const Slot &slot : slots_) {
        if (slot.info.state == ContinuousBatchSessionState::PrefillInFlight ||
            slot.info.state == ContinuousBatchSessionState::DecodeInFlight) {
            return true;
        }
    }
    return false;
}

ContinuousBatchStats ContinuousBatchScheduler::stats() const {
    ContinuousBatchStats result = lifetime_stats_;
    for (const Slot &slot : slots_) {
        switch (slot.info.state) {
        case ContinuousBatchSessionState::Empty:
            break;
        case ContinuousBatchSessionState::PrefillReady:
            ++result.resident;
            ++result.prefill_ready;
            break;
        case ContinuousBatchSessionState::PrefillInFlight:
            ++result.resident;
            ++result.in_flight;
            break;
        case ContinuousBatchSessionState::DecodeIdle:
            ++result.resident;
            ++result.decode_idle;
            break;
        case ContinuousBatchSessionState::DecodeReady:
            ++result.resident;
            ++result.decode_ready;
            break;
        case ContinuousBatchSessionState::DecodeInFlight:
            ++result.resident;
            ++result.in_flight;
            break;
        case ContinuousBatchSessionState::Finished:
        case ContinuousBatchSessionState::Cancelled:
        case ContinuousBatchSessionState::Failed:
            ++result.resident;
            ++result.terminal;
            break;
        }
    }
    return result;
}

}  // namespace dflash::common
