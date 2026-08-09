#include "continuous_batch_scheduler.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace dflash::common;

static int passed = 0;
static int failed = 0;

#define CHECK(expr) do { \
    if (expr) { ++passed; } \
    else { ++failed; std::fprintf(stderr, "FAIL %s:%d: %s\n", \
                                  __FILE__, __LINE__, #expr); } \
} while (0)

static ContinuousBatchSessionInfo info(
        const ContinuousBatchScheduler &scheduler,
        ContinuousBatchSessionId id) {
    auto value = scheduler.session(id);
    CHECK(value.has_value());
    return value.value_or(ContinuousBatchSessionInfo{});
}

static void complete_full_prefill(ContinuousBatchScheduler &scheduler,
                                  ContinuousBatchSessionId id,
                                  std::int64_t &now) {
    while (info(scheduler, id).state ==
           ContinuousBatchSessionState::PrefillReady) {
        ContinuousBatchPlan work = scheduler.plan(now++);
        CHECK(work.prefill_session == id);
        CHECK(work.prefill_tokens > 0);
        CHECK(scheduler.complete_prefill(work.submission_id, id,
                                         work.prefill_tokens, true));
    }
}

int main() {
    {
        ContinuousBatchScheduler scheduler({2, 8, 2, 10});
        auto a = scheduler.admit(10, 4);
        auto b = scheduler.admit(3, 2);
        CHECK(a.has_value());
        CHECK(b.has_value());
        CHECK(!scheduler.admit(1, 1).has_value());
        CHECK(scheduler.resident() == 2);

        auto first = scheduler.plan(0);
        CHECK(first.prefill_session == *a);
        CHECK(first.prefill_tokens == 8);
        CHECK(first.decode_sessions.empty());
        CHECK(first.submission_id != 0);
        CHECK(scheduler.submission_in_flight());
        CHECK(scheduler.plan(1).empty());
        CHECK(scheduler.complete_prefill(first.submission_id, *a, 8, true));

        auto second = scheduler.plan(2);
        CHECK(second.prefill_session == *b);  // round-robin, not a again
        CHECK(second.prefill_tokens == 3);
        CHECK(scheduler.complete_prefill(second.submission_id, *b, 3, true));

        auto third = scheduler.plan(3);
        CHECK(third.prefill_session == *a);
        CHECK(third.prefill_tokens == 2);
        CHECK(scheduler.complete_prefill(third.submission_id, *a, 2, true));
        CHECK(info(scheduler, *a).state ==
              ContinuousBatchSessionState::DecodeIdle);
        CHECK(info(scheduler, *b).state ==
              ContinuousBatchSessionState::DecodeIdle);
    }

    {
        ContinuousBatchScheduler scheduler({3, 16, 4, 100});
        auto a = scheduler.admit(0, 3).value();
        auto b = scheduler.admit(0, 3).value();
        auto c = scheduler.admit(20, 1).value();
        CHECK(scheduler.mark_decode_ready(a, 1000));

        // b is active but not ready: hold a briefly so both can share a step.
        auto wait = scheduler.plan(1050);
        CHECK(wait.empty());
        CHECK(wait.wake_at_us == 1100);
        CHECK(scheduler.plan(1075).wake_at_us == 1100);

        CHECK(scheduler.mark_decode_ready(b, 1060));
        auto mixed = scheduler.plan(1060);
        CHECK(mixed.decode_sessions.size() == 2);
        CHECK(mixed.decode_sessions[0] == a);
        CHECK(mixed.decode_sessions[1] == b);
        CHECK(mixed.prefill_session == c);
        CHECK(mixed.prefill_tokens == 4);
        CHECK(mixed.mixed());
        CHECK(mixed.submission_id != 0);

        CHECK(scheduler.complete_decode(mixed.submission_id, a, true, false));
        CHECK(scheduler.submission_in_flight());
        CHECK(scheduler.complete_prefill(mixed.submission_id, c, 4, true));
        CHECK(scheduler.submission_in_flight());
        CHECK(scheduler.complete_decode(mixed.submission_id, b, true, false));
        CHECK(!scheduler.submission_in_flight());
        CHECK(info(scheduler, a).generated_tokens == 1);
        CHECK(info(scheduler, b).generated_tokens == 1);
        CHECK(info(scheduler, c).prefilled_tokens == 4);
    }

    {
        ContinuousBatchScheduler scheduler({1, 8, 2, 0});
        auto id = scheduler.admit(0, 2).value();
        CHECK(scheduler.mark_decode_ready(id, 0));
        auto step = scheduler.plan(0);
        CHECK(step.decode_sessions.size() == 1);
        CHECK(scheduler.complete_decode(step.submission_id, id, true, false));
        CHECK(scheduler.mark_decode_ready(id, 1));
        step = scheduler.plan(1);
        CHECK(scheduler.complete_decode(step.submission_id, id, true, false));
        CHECK(info(scheduler, id).state ==
              ContinuousBatchSessionState::Finished);
        CHECK(!scheduler.mark_decode_ready(id, 2));
        CHECK(scheduler.release(id));
        CHECK(scheduler.resident() == 0);

        auto replacement = scheduler.admit(0, 1).value();
        CHECK(replacement != id);
        CHECK(!scheduler.cancel(id));  // stale generation-stamped id
        CHECK(scheduler.cancel(replacement));
        CHECK(info(scheduler, replacement).state ==
              ContinuousBatchSessionState::Cancelled);
        CHECK(scheduler.release(replacement));
    }

    {
        ContinuousBatchScheduler scheduler({2, 8, 2, 0});
        auto prefill = scheduler.admit(4, 1).value();
        auto decode = scheduler.admit(0, 3).value();
        auto p = scheduler.plan(0);
        CHECK(p.prefill_session == prefill);
        CHECK(scheduler.cancel(prefill));  // deferred until in-flight work returns
        CHECK(info(scheduler, prefill).cancel_requested);
        CHECK(scheduler.complete_prefill(p.submission_id, prefill,
                                         p.prefill_tokens, true));
        CHECK(info(scheduler, prefill).state ==
              ContinuousBatchSessionState::Cancelled);

        CHECK(scheduler.mark_decode_ready(decode, 1));
        auto d = scheduler.plan(1);
        CHECK(d.decode_sessions.size() == 1);
        CHECK(scheduler.cancel(decode));
        CHECK(scheduler.complete_decode(d.submission_id, decode, true, false));
        CHECK(info(scheduler, decode).state ==
              ContinuousBatchSessionState::Cancelled);
    }

    {
        ContinuousBatchScheduler scheduler({1, 8, 2, 0});
        auto id = scheduler.admit(5, 1).value();
        auto p = scheduler.plan(0);
        CHECK(!scheduler.complete_prefill(p.submission_id, id, 0, true));
        CHECK(scheduler.complete_prefill(p.submission_id, id,
                                         p.prefill_tokens, false));
        CHECK(info(scheduler, id).state ==
              ContinuousBatchSessionState::Failed);
        CHECK(scheduler.release(id));
        CHECK(!scheduler.admit(-1, 2).has_value());
        CHECK(!scheduler.admit(1, -2).has_value());
    }

    {
        ContinuousBatchScheduler scheduler({1, 8, 2, 0});
        auto id = scheduler.admit(17, 0).value();
        std::int64_t now = 0;
        complete_full_prefill(scheduler, id, now);
        CHECK(info(scheduler, id).state ==
              ContinuousBatchSessionState::Finished);
        CHECK(info(scheduler, id).prefilled_tokens == 17);
    }

    {
        // Any active generation forces the small mixed prefill quantum, even
        // during the interval where that generation has no decode row ready.
        ContinuousBatchScheduler scheduler({2, 8, 2, 0});
        auto generation = scheduler.admit(0, 2).value();
        auto prefill = scheduler.admit(10, 1).value();
        auto p = scheduler.plan(0);
        CHECK(p.decode_sessions.empty());
        CHECK(p.prefill_session == prefill);
        CHECK(p.prefill_tokens == 2);
        CHECK(scheduler.complete_prefill(p.submission_id, prefill, 2, true));
        CHECK(scheduler.mark_decode_ready(generation, 1));
    }

    {
        // A late/duplicate completion cannot finish a newer step for the same
        // resident session.
        ContinuousBatchScheduler scheduler({1, 8, 2, 0});
        auto id = scheduler.admit(0, 3).value();
        CHECK(scheduler.mark_decode_ready(id, 0));
        auto first = scheduler.plan(0);
        CHECK(!scheduler.complete_decode(first.submission_id + 1,
                                         id, true, false));
        CHECK(scheduler.complete_decode(first.submission_id, id, true, false));
        CHECK(scheduler.mark_decode_ready(id, 1));
        auto second = scheduler.plan(1);
        CHECK(second.submission_id != first.submission_id);
        CHECK(!scheduler.complete_decode(first.submission_id, id, true, false));
        CHECK(scheduler.complete_decode(second.submission_id, id, true, true));
    }

    {
        ContinuousBatchScheduler scheduler({3, 8, 2, 100});
        auto a = scheduler.admit(0, 2).value();
        auto b = scheduler.admit(0, 2).value();
        auto p = scheduler.admit(3, 0).value();
        CHECK(scheduler.mark_decode_ready(a, 0));
        CHECK(scheduler.plan(10).empty());
        CHECK(scheduler.plan(20).empty());
        CHECK(scheduler.mark_decode_ready(b, 20));
        auto work = scheduler.plan(20);
        CHECK(work.mixed());
        CHECK(scheduler.complete_decode(work.submission_id, a, true, false));
        CHECK(scheduler.complete_decode(work.submission_id, b, true, true));
        CHECK(scheduler.complete_prefill(work.submission_id, p, 2, true));

        ContinuousBatchStats stats = scheduler.stats();
        CHECK(stats.resident == 3);
        CHECK(stats.prefill_ready == 1);
        CHECK(stats.decode_idle == 1);
        CHECK(stats.terminal == 1);
        CHECK(stats.in_flight == 0);
        CHECK(stats.admissions == 3);
        CHECK(stats.submissions == 1);
        CHECK(stats.decode_batches == 1);
        CHECK(stats.decode_rows_scheduled == 2);
        CHECK(stats.decode_tokens_completed == 2);
        CHECK(stats.prefill_batches == 1);
        CHECK(stats.prefill_tokens_scheduled == 2);
        CHECK(stats.prefill_tokens_completed == 2);
        CHECK(stats.mixed_submissions == 1);
        CHECK(stats.coalesce_waits == 1);
        CHECK(stats.max_decode_batch == 2);
    }

    {
        // Deterministic adversarial walk: repeatedly fill/reuse slots while
        // interleaving readiness, queued/in-flight cancellation, clean stops,
        // and failures. This is intentionally single-threaded because the
        // engine coordinator owns the scheduler; the event order models the
        // messages that coordinator receives.
        ContinuousBatchScheduler scheduler({8, 17, 3, 7});
        std::vector<ContinuousBatchSessionId> ids;
        std::uint64_t rng = 0x9e3779b97f4a7c15ULL;
        auto next = [&]() {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            return rng * 0x2545f4914f6cdd1dULL;
        };
        std::int64_t now = 0;

        for (int iteration = 0; iteration < 5000; ++iteration) {
            now += (std::int64_t)(next() % 5);
            if (ids.size() < scheduler.capacity() && (next() % 3) != 0) {
                auto id = scheduler.admit((int)(next() % 50),
                                          (int)(next() % 12));
                if (id) ids.push_back(*id);
            }

            for (ContinuousBatchSessionId id : ids) {
                auto current = scheduler.session(id);
                if (!current) continue;
                if (current->state == ContinuousBatchSessionState::DecodeIdle &&
                    (next() % 4) != 0) {
                    CHECK(scheduler.mark_decode_ready(id, now));
                } else if ((current->state ==
                                ContinuousBatchSessionState::PrefillReady ||
                            current->state ==
                                ContinuousBatchSessionState::DecodeReady) &&
                           (next() % 97) == 0) {
                    CHECK(scheduler.cancel(id));
                }
            }

            ContinuousBatchPlan work = scheduler.plan(now);
            if (!work.empty()) {
                if (work.prefill_session != 0) {
                    if ((next() % 113) == 0) {
                        CHECK(scheduler.cancel(work.prefill_session));
                    }
                    const bool ok = (next() % 251) != 0;
                    CHECK(scheduler.complete_prefill(
                        work.submission_id, work.prefill_session,
                        work.prefill_tokens, ok));
                }
                for (ContinuousBatchSessionId id : work.decode_sessions) {
                    if ((next() % 127) == 0) CHECK(scheduler.cancel(id));
                    const bool ok = (next() % 257) != 0;
                    const bool terminal = (next() % 23) == 0;
                    CHECK(scheduler.complete_decode(
                        work.submission_id, id, ok, terminal));
                }
            } else if (work.wake_at_us > now) {
                now = work.wake_at_us;
            }

            for (ContinuousBatchSessionId id : ids) {
                auto current = scheduler.session(id);
                if (!current) continue;
                CHECK(current->prefilled_tokens >= 0);
                CHECK(current->prefilled_tokens <= current->prompt_tokens);
                CHECK(current->generated_tokens >= 0);
                CHECK(current->generated_tokens <= current->max_new_tokens);
                if (current->state ==
                        ContinuousBatchSessionState::PrefillReady) {
                    CHECK(current->prefilled_tokens < current->prompt_tokens);
                }
                if (current->state ==
                        ContinuousBatchSessionState::DecodeIdle ||
                    current->state ==
                        ContinuousBatchSessionState::DecodeReady) {
                    CHECK(current->prefilled_tokens == current->prompt_tokens);
                    CHECK(current->generated_tokens < current->max_new_tokens);
                }
            }

            ids.erase(std::remove_if(ids.begin(), ids.end(),
                [&](ContinuousBatchSessionId id) {
                    auto current = scheduler.session(id);
                    if (!current) return true;
                    const bool terminal =
                        current->state == ContinuousBatchSessionState::Finished ||
                        current->state == ContinuousBatchSessionState::Cancelled ||
                        current->state == ContinuousBatchSessionState::Failed;
                    return terminal && scheduler.release(id);
                }), ids.end());
            CHECK(scheduler.stats().resident == scheduler.resident());
            CHECK(scheduler.stats().decode_ready == scheduler.decode_ready());
        }
        CHECK(!scheduler.submission_in_flight());
    }

    std::printf("continuous batch scheduler tests: %d passed, %d failed\n",
                passed, failed);
    return failed == 0 ? 0 : 1;
}
