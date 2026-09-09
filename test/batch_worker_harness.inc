// Scaffolding for test/extract_batch_worker.py. Everything between the markers
// is the PRODUCTION worker, extracted verbatim; only the surrounding stubs are
// written here. Keep this file free of any reimplementation of the loop.
#include "backend/batch_wake.h"
#include "resident_batch_coordinator.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

using namespace dflash::common;

static int failures = 0;
#define CHECK(expr, what) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s\n", what); ++failures; \
    } \
} while (0)

// @@FAKE@@

// The worker touches only these members of the real ember_backend.
struct DummyModel {
    void shutdown() {}
    // The worker consults these when a call carries a restore slot; the seam
    // never exercises restore, so they answer "nothing snapshotted".
    bool snapshot_used(int) const { return false; }
    int snapshot_cur_pos(int) const { return 0; }
};

struct InstrumentedCoordinator;
struct ember_backend;

// Counts pumps and can hold the worker inside pump() on demand, which is how
// the lost-wakeup window is made deterministic instead of raced for.
struct InstrumentedCoordinator : ResidentBatchCoordinator {
    using ResidentBatchCoordinator::ResidentBatchCoordinator;
    std::atomic<long> pumps{0};
    // Hold from a chosen pump onwards, not "from now": arming a plain flag
    // catches whichever pump happens to be in flight, including the startup
    // one, and then the worker never services the warm-up that is supposed to
    // put it there.
    std::atomic<long> hold_from{-1};
    std::atomic<bool> in_pump{false};
    std::atomic<bool> throw_next{false};

    ContinuousBatchRunResult pump(std::int64_t now_us) {
        pumps.fetch_add(1);
        if (throw_next.exchange(false)) {
            throw std::runtime_error("injected pump failure");
        }
        const long hf = hold_from.load();
        if (hf >= 0 && pumps.load() >= hf) {
            in_pump.store(true);
            while (hold_from.load() >= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            in_pump.store(false);
        }
        return ResidentBatchCoordinator::pump(now_us);
    }
};

struct ember_batch_stats { int pending; };

static std::int64_t batch_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The production worker constructs ResidentBatchCoordinator by name; this
// substitutes the instrumented subclass without editing the extracted code.
#define ResidentBatchCoordinator InstrumentedCoordinator

struct ember_backend {
    std::unique_ptr<DummyModel> be = std::make_unique<DummyModel>();
    std::unique_ptr<int> disk;
    ResidentBatchBackend *resident = nullptr;
    int batch_sessions = 2;
    int batch_prefill_quantum = 4;
    int batch_mixed_prefill_quantum = 2;
    std::int64_t batch_decode_coalesce_us = 0;
    std::unique_ptr<InstrumentedCoordinator> coordinator;
    std::mutex batch_mu;
    std::condition_variable batch_cv;
    std::thread batch_thread;
    std::deque<struct ember_batch_call *> batch_pending;
    std::deque<struct ember_batch_control *> batch_controls;
    std::unordered_map<ContinuousBatchSessionId, struct ember_batch_call *>
        batch_active;
    bool batch_stop = false;
    bool batch_running = false;
    bool batch_start_done = false;
    std::string batch_start_error;
    ember_batch_stats batch_stats_cache{};
};

static void batch_refresh_stats_locked(ember_backend *) {}

// @@WORKER@@

static void start_worker(ember_backend &b, FakeResidentBackend &backend) {
    b.resident = &backend;
    b.batch_thread = std::thread(ember_batch_thread_main, &b);
    std::unique_lock<std::mutex> lock(b.batch_mu);
    b.batch_cv.wait(lock, [&b] { return b.batch_start_done; });
}

static void stop_worker(ember_backend &b) {
    {
        std::lock_guard<std::mutex> lock(b.batch_mu);
        b.batch_stop = true;
        b.batch_cv.notify_all();
    }
    if (b.batch_thread.joinable()) b.batch_thread.join();
}

static GenerateRequest req(int prompt_tokens, int gen_tokens) {
    GenerateRequest r;
    r.prompt.resize((size_t)prompt_tokens, 7);
    r.n_gen = gen_tokens;
    return r;
}

int main() {
    std::fprintf(stderr, "-- scenario 1\n");
    // 1. The lost wakeup. Enqueue a control while the worker is inside pump()
    //    with batch_mu released, so its notify_one() lands with nobody waiting
    //    and is dropped. An unpredicated wait sleeps forever here; this is the
    //    scenario that deadlocked gate 2 on gfx1151 twice.
    {
        FakeResidentBackend backend;
        ember_backend b;
        start_worker(b, backend);
        // A correctly predicated worker SLEEPS when idle rather than pumping,
        // so it has to be woken before it can be caught inside pump(). Arm the
        // hold first, then send a warm-up control: the worker services that,
        // loops, and blocks in the next pump with batch_mu released -- which is
        // exactly the window a real submitter races.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        b.coordinator->hold_from.store(b.coordinator->pumps.load() + 1);
        CHECK(ember_batch_control_run(&b, [] {}), "warm-up control must run");
        for (int i = 0; i < 2000 && !b.coordinator->in_pump.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CHECK(b.coordinator->in_pump.load(), "worker did not reach pump()");
        std::atomic<int> state{0};
        std::thread caller([&] {
            bool ran = ember_batch_control_run(&b, [&] { state.store(1); });
            state.store(ran ? 2 : -1);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        b.coordinator->hold_from.store(-1);
        caller.join();
        CHECK(state.load() == 2, "control enqueued during pump must run");
        stop_worker(b);
    }

    std::fprintf(stderr, "-- scenario 2\n");
    // 2. A control submitted after the coordinator has failed must return
    //    rather than wait on a worker that has exited. The release path of
    //    every completed generation is a control, so this stranded callers.
    {
        FakeResidentBackend backend;
        ember_backend b;
        start_worker(b, backend);
        // Arm the failure, then give the worker a reason to pump. A bare
        // notify is not enough now that the wait is predicated: with nothing
        // queued the worker correctly ignores it and never reaches pump().
        b.coordinator->throw_next.store(true);
        CHECK(ember_batch_control_run(&b, [] {}),
              "control before the failure must still run");
        for (int i = 0; i < 400; ++i) {
            {
                std::lock_guard<std::mutex> lock(b.batch_mu);
                if (!b.batch_running) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (b.batch_thread.joinable()) b.batch_thread.join();
        CHECK(!b.batch_running, "worker must have exited after pump failure");
        CHECK(b.coordinator == nullptr, "coordinator must be gone");

        std::atomic<int> done{0};
        std::thread caller([&] {
            bool ran = ember_batch_control_run(&b, [] {});
            done.store(ran ? 2 : 1);
        });
        for (int i = 0; i < 200 && done.load() == 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        CHECK(done.load() != 0,
              "control after coordinator failure must not block forever");
        CHECK(done.load() == 1,
              "control after coordinator failure must report it never ran");
        caller.join();
    }

    std::fprintf(stderr, "-- scenario 3\n");
    // 3. The spin. Retained terminal leases hold capacity, so a queued request
    //    cannot be admitted; waking on it regardless burns the core.
    {
        FakeResidentBackend backend;
        ember_backend b;
        start_worker(b, backend);
        ember_batch_call held[2];
        for (int i = 0; i < 2; ++i) {
            held[i].request = req(0, 0);   // terminal on admission, lease kept
            std::lock_guard<std::mutex> lock(b.batch_mu);
            b.batch_pending.push_back(&held[i]);
            b.batch_cv.notify_one();
        }
        for (int i = 0; i < 200; ++i) {
            std::unique_lock<std::mutex> lock(b.batch_mu);
            if (held[0].done && held[1].done) break;
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        ember_batch_call queued;
        queued.request = req(0, 1);
        {
            std::lock_guard<std::mutex> lock(b.batch_mu);
            b.batch_pending.push_back(&queued);
            b.batch_cv.notify_one();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const long before = b.coordinator->pumps.load();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const long spun = b.coordinator->pumps.load() - before;
        if (spun > 50) {
            std::fprintf(stderr,
                         "FAIL pending at full capacity spun: %ld pumps/100ms\n",
                         spun);
            ++failures;
        }
        stop_worker(b);
    }

    std::fprintf(stderr, "-- all scenarios returned\n");
    if (failures) {
        std::fprintf(stderr, "%d scenario(s) failed\n", failures);
        return 1;
    }
    std::printf("extracted worker scenarios: ok\n");
    return 0;
}
