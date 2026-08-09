#include "continuous_batch_executor.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

using namespace dflash::common;

static int passed = 0;
static int failed = 0;

#define CHECK(expr) do { \
    if (expr) { ++passed; } \
    else { ++failed; std::fprintf(stderr, "FAIL %s:%d: %s\n", \
                                  __FILE__, __LINE__, #expr); } \
} while (0)

struct FakeBackend : ContinuousBatchWorkBackend {
    bool native_mixed = false;
    bool fail_prefill = false;
    bool throw_prefill = false;
    bool throw_decode = false;
    bool malformed_decode = false;
    bool malformed_prefill = false;
    bool reverse_decode = false;
    int terminal_after = 0;
    int prefill_calls = 0;
    int decode_calls = 0;
    int mixed_calls = 0;
    std::unordered_map<ContinuousBatchSessionId, int> prefilled;
    std::unordered_map<ContinuousBatchSessionId, int> decoded;

    ContinuousBatchPrefillCompletion
    prefill(ContinuousBatchSessionId id, int requested) override {
        ++prefill_calls;
        if (throw_prefill) throw std::runtime_error("prefill");
        if (fail_prefill) return {false, 0};
        if (malformed_prefill) return {true, requested + 1};
        prefilled[id] += requested;
        return {true, requested};
    }

    std::vector<ContinuousBatchDecodeCompletion>
    decode_batch(const std::vector<ContinuousBatchSessionId> &ids) override {
        ++decode_calls;
        if (throw_decode) throw std::runtime_error("decode");
        std::vector<ContinuousBatchDecodeCompletion> result;
        for (ContinuousBatchSessionId id : ids) {
            int n = ++decoded[id];
            result.push_back({id, true, terminal_after > 0 &&
                                        n >= terminal_after});
        }
        if (malformed_decode && !result.empty()) result.pop_back();
        if (reverse_decode) std::reverse(result.begin(), result.end());
        return result;
    }

    bool supports_native_mixed() const override { return native_mixed; }

    ContinuousBatchMixedCompletion execute_mixed(
            ContinuousBatchSessionId prefill_id,
            int requested,
            const std::vector<ContinuousBatchSessionId> &ids) override {
        ++mixed_calls;
        ContinuousBatchMixedCompletion result;
        // Do not count native work as calls to the fallback entry points.
        if (throw_prefill || throw_decode) throw std::runtime_error("mixed");
        if (fail_prefill) {
            result.prefill = {false, 0};
        } else if (malformed_prefill) {
            result.prefill = {true, requested + 1};
        } else {
            prefilled[prefill_id] += requested;
            result.prefill = {true, requested};
        }
        for (ContinuousBatchSessionId id : ids) {
            int n = ++decoded[id];
            result.decode.push_back(
                {id, true, terminal_after > 0 && n >= terminal_after});
        }
        if (malformed_decode && !result.decode.empty()) {
            result.decode.pop_back();
        }
        if (reverse_decode) {
            std::reverse(result.decode.begin(), result.decode.end());
        }
        return result;
    }
};

static ContinuousBatchSessionState state(
        const ContinuousBatchScheduler &scheduler,
        ContinuousBatchSessionId id) {
    auto session = scheduler.session(id);
    CHECK(session.has_value());
    return session ? session->state : ContinuousBatchSessionState::Empty;
}

static void make_ready(ContinuousBatchScheduler &scheduler,
                       ContinuousBatchSessionId id, std::int64_t now) {
    CHECK(scheduler.mark_decode_ready(id, now));
}

int main() {
    {
        ContinuousBatchScheduler scheduler({3, 8, 2, 0});
        FakeBackend backend;
        backend.reverse_decode = true;
        ContinuousBatchExecutor executor(scheduler, backend);
        auto a = scheduler.admit(0, 3).value();
        auto b = scheduler.admit(0, 3).value();
        auto p = scheduler.admit(5, 1).value();
        make_ready(scheduler, a, 0);
        make_ready(scheduler, b, 0);

        auto run = executor.run_once(0);
        CHECK(run.status == ContinuousBatchRunStatus::Completed);
        CHECK(run.submission_id != 0);
        CHECK(backend.prefill_calls == 1);
        CHECK(backend.decode_calls == 1);
        CHECK(backend.mixed_calls == 0);
        CHECK(backend.prefilled[p] == 2);
        CHECK(backend.decoded[a] == 1);
        CHECK(backend.decoded[b] == 1);
        CHECK(state(scheduler, p) ==
              ContinuousBatchSessionState::PrefillReady);
        CHECK(state(scheduler, a) ==
              ContinuousBatchSessionState::DecodeIdle);
        CHECK(state(scheduler, b) ==
              ContinuousBatchSessionState::DecodeIdle);
        CHECK(executor.stats().fallback_mixed == 1);
        CHECK(!scheduler.submission_in_flight());
    }

    {
        ContinuousBatchScheduler scheduler({3, 8, 2, 0});
        FakeBackend backend;
        backend.native_mixed = true;
        ContinuousBatchExecutor executor(scheduler, backend);
        auto a = scheduler.admit(0, 1).value();
        auto b = scheduler.admit(0, 1).value();
        auto p = scheduler.admit(4, 1).value();
        make_ready(scheduler, a, 0);
        make_ready(scheduler, b, 0);

        executor.run_once(0);
        CHECK(backend.mixed_calls == 1);
        CHECK(backend.prefill_calls == 0);
        CHECK(backend.decode_calls == 0);
        CHECK(state(scheduler, a) ==
              ContinuousBatchSessionState::Finished);
        CHECK(state(scheduler, b) ==
              ContinuousBatchSessionState::Finished);
        CHECK(state(scheduler, p) ==
              ContinuousBatchSessionState::PrefillReady);
        CHECK(executor.stats().native_mixed == 1);
    }

    {
        ContinuousBatchScheduler scheduler({2, 8, 2, 100});
        FakeBackend backend;
        ContinuousBatchExecutor executor(scheduler, backend);
        auto a = scheduler.admit(0, 2).value();
        scheduler.admit(0, 2);
        make_ready(scheduler, a, 1000);
        auto run = executor.run_once(1050);
        CHECK(run.status == ContinuousBatchRunStatus::Waiting);
        CHECK(run.wake_at_us == 1100);
        CHECK(backend.decode_calls == 0);
    }

    {
        ContinuousBatchScheduler scheduler({2, 8, 2, 0});
        FakeBackend backend;
        backend.malformed_decode = true;
        ContinuousBatchExecutor executor(scheduler, backend);
        auto a = scheduler.admit(0, 2).value();
        auto b = scheduler.admit(0, 2).value();
        make_ready(scheduler, a, 0);
        make_ready(scheduler, b, 0);
        executor.run_once(0);
        CHECK(state(scheduler, a) == ContinuousBatchSessionState::Failed);
        CHECK(state(scheduler, b) == ContinuousBatchSessionState::Failed);
        CHECK(executor.stats().malformed_results == 1);
        CHECK(!scheduler.submission_in_flight());
    }

    {
        ContinuousBatchScheduler scheduler({2, 8, 2, 0});
        FakeBackend backend;
        backend.throw_decode = true;
        ContinuousBatchExecutor executor(scheduler, backend);
        auto a = scheduler.admit(0, 2).value();
        auto b = scheduler.admit(0, 2).value();
        make_ready(scheduler, a, 0);
        make_ready(scheduler, b, 0);
        executor.run_once(0);
        CHECK(state(scheduler, a) == ContinuousBatchSessionState::Failed);
        CHECK(state(scheduler, b) == ContinuousBatchSessionState::Failed);
        CHECK(executor.stats().backend_exceptions == 1);
    }

    {
        // The sequential mixed fallback does not submit decode work after a
        // failed prefill, matching Dwarfstar's correctness-first fallback.
        ContinuousBatchScheduler scheduler({2, 8, 2, 0});
        FakeBackend backend;
        backend.fail_prefill = true;
        ContinuousBatchExecutor executor(scheduler, backend);
        auto a = scheduler.admit(0, 2).value();
        auto p = scheduler.admit(4, 1).value();
        make_ready(scheduler, a, 0);
        executor.run_once(0);
        CHECK(backend.prefill_calls == 1);
        CHECK(backend.decode_calls == 0);
        CHECK(state(scheduler, p) == ContinuousBatchSessionState::Failed);
        CHECK(state(scheduler, a) == ContinuousBatchSessionState::Failed);
        CHECK(executor.stats().backend_failures == 1);
        CHECK(executor.stats().aborted_components == 1);
    }

    {
        ContinuousBatchScheduler scheduler({2, 8, 2, 0});
        FakeBackend backend;
        backend.native_mixed = true;
        backend.malformed_prefill = true;
        ContinuousBatchExecutor executor(scheduler, backend);
        auto a = scheduler.admit(0, 2).value();
        auto p = scheduler.admit(4, 1).value();
        make_ready(scheduler, a, 0);
        executor.run_once(0);
        CHECK(state(scheduler, p) == ContinuousBatchSessionState::Failed);
        CHECK(state(scheduler, a) == ContinuousBatchSessionState::Failed);
        CHECK(executor.stats().malformed_results == 1);
    }

    {
        ContinuousBatchScheduler scheduler({1, 8, 2, 0});
        FakeBackend backend;
        ContinuousBatchExecutor executor(scheduler, backend);
        auto p = scheduler.admit(3, 0).value();
        auto run = executor.run_once(0);
        CHECK(run.status == ContinuousBatchRunStatus::Completed);
        CHECK(state(scheduler, p) == ContinuousBatchSessionState::Finished);
        run = executor.run_once(1);
        CHECK(run.status == ContinuousBatchRunStatus::Idle);
    }

    std::printf("continuous batch executor tests: %d passed, %d failed\n",
                passed, failed);
    return failed == 0 ? 0 : 1;
}
