// backend_dflash.cc — the real backend: an extern-C shim implementing Ember's
// ember_backend.h over lucebox's libdflash_common (the tuned ROCMFP / HIP
// / DSpark DeepSeek4 forward pass + its byte-exact tokenizer). Ember's C server
// links this instead of backend_stub.o to serve real tokens; nothing above the
// ABI changes.
//
// This translation unit is C++ (it uses the lucebox C++ classes) but exports
// only the C ABI. It compiles against the lucebox headers and links the static
// lib; it must be built in the ROCm/HIP container (the C server does not).
#include "ember_backend.h"

#include <sys/stat.h>
#include <unistd.h>

#include <atomic>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/backend_factory.h"
#include "common/model_backend.h"
#include "common/resident_batch_coordinator.h"
#include "common/sampler.h"
#include "server/disk_prefix_cache.h"
#include "server/tokenizer.h"

#include <xgrammar/xgrammar.h>

using dflash::common::BackendArgs;
using dflash::common::DaemonIO;
using dflash::common::DiskCacheConfig;
using dflash::common::DiskPrefixCache;
using dflash::common::GenerateRequest;
using dflash::common::GenerateResult;
using dflash::common::ModelBackend;
using dflash::common::PrefillAttentionMode;
using dflash::common::ResidentBatchBackend;
using dflash::common::ResidentBatchCoordinator;
using dflash::common::Tokenizer;

struct ember_batch_call {
    GenerateRequest request;
    DaemonIO io;
    int restore_slot = -1;
    bool cancelled = false;
    bool done = false;
    GenerateResult result;
    dflash::common::ContinuousBatchSessionId session_id = 0;
    std::condition_variable cv;
};

struct ember_batch_control {
    std::function<void()> fn;
    bool done = false;
    std::condition_variable cv;
};

struct ember_backend {
    std::unique_ptr<ModelBackend>    be;
    std::unique_ptr<DiskPrefixCache> disk;  // cross-restart KV persistence (opt)
    Tokenizer                        tok;
    std::string                      model_name;
    int                              n_ctx = 65536;
    std::array<uint8_t, 16>          cache_identity{};
    bool                             cache_identity_valid = false;

    int batch_sessions = 1;
    int batch_prefill_quantum = 2048;
    int batch_mixed_prefill_quantum = 128;
    std::int64_t batch_decode_coalesce_us = 2000;
    ResidentBatchBackend *resident = nullptr;  // borrowed from be
    std::unique_ptr<ResidentBatchCoordinator> coordinator;
    std::mutex batch_mu;
    std::condition_variable batch_cv;
    std::thread batch_thread;
    std::deque<ember_batch_call *> batch_pending;
    std::deque<ember_batch_control *> batch_controls;
    std::unordered_map<dflash::common::ContinuousBatchSessionId,
                       ember_batch_call *> batch_active;
    bool batch_stop = false;
    bool batch_running = false;
    bool batch_start_done = false;
    std::string batch_start_error;
    ember_batch_stats batch_stats_cache{};

    // Constrained tool-call decoding, built lazily on first use so a model that
    // never sees tools pays nothing. Compiling a grammar is expensive relative
    // to a request, so results are cached by EBNF text; a request's tool set is
    // stable across a conversation, making the hit rate essentially 1.
    std::mutex                                 xg_mu;
    std::unique_ptr<xgrammar::TokenizerInfo>   xg_vocab;
    std::unique_ptr<xgrammar::GrammarCompiler> xg_compiler;
    std::unordered_map<std::string, xgrammar::CompiledGrammar> xg_cache;
};

static thread_local std::unordered_map<
    ember_backend *, dflash::common::ContinuousBatchSessionId>
    tls_batch_sessions;

static std::vector<int32_t> vec(const int32_t *p, int n) {
    if (!p || n <= 0) return {};
    return std::vector<int32_t>(p, p + n);
}

static char *c_err(const char *m) { return m ? strdup(m) : nullptr; }

static int64_t batch_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int batch_env_int(const char *name, int fallback,
                         int minimum, int maximum) {
    const char *raw = std::getenv(name);
    if (!raw || !raw[0]) return fallback;
    char *end = nullptr;
    errno = 0;
    const long value = std::strtol(raw, &end, 10);
    return errno == 0 && end != raw && *end == '\0' &&
           value >= minimum && value <= maximum
        ? (int)value : fallback;
}

static void batch_fail_call(ember_batch_call *call, const char *detail) {
    if (!call) return;
    call->result.fail(dflash::common::GenerateErrorCode::BackendSpecific,
                      detail ? detail : "continuous batch failed");
    call->done = true;
    call->cv.notify_one();
}

// batch_mu must be held. The coordinator itself is engine-thread-owned; this
// cache keeps /status non-blocking during a long GPU submission.
static void batch_refresh_stats_locked(ember_backend *b) {
    ember_batch_stats &stats = b->batch_stats_cache;
    std::memset(&stats, 0, sizeof(stats));
    stats.enabled = b->batch_sessions > 1;
    stats.capacity = b->batch_sessions;
    stats.pending = (int)b->batch_pending.size();
    if (!b->coordinator) return;
    const auto scheduler = b->coordinator->scheduler().stats();
    const auto executor = b->coordinator->executor_stats();
    stats.resident = (int)scheduler.resident;
    stats.prefill_ready = (int)scheduler.prefill_ready;
    stats.decode_ready = (int)scheduler.decode_ready;
    stats.in_flight = (int)scheduler.in_flight;
    stats.terminal = (int)scheduler.terminal;
    stats.admissions = scheduler.admissions;
    stats.releases = scheduler.releases;
    stats.submissions = scheduler.submissions;
    stats.decode_batches = scheduler.decode_batches;
    stats.decode_rows = scheduler.decode_rows_scheduled;
    stats.prefill_tokens = scheduler.prefill_tokens_completed;
    stats.mixed_submissions = scheduler.mixed_submissions;
    stats.coalesce_waits = scheduler.coalesce_waits;
    stats.backend_failures = executor.backend_failures;
    stats.backend_exceptions = executor.backend_exceptions;
    stats.max_decode_batch = (int)scheduler.max_decode_batch;
}

static void ember_batch_thread_main(ember_backend *b) {
    try {
        auto coordinator = std::make_unique<ResidentBatchCoordinator>(
            *b->resident,
            dflash::common::ContinuousBatchConfig{
                (size_t)b->batch_sessions,
                b->batch_prefill_quantum,
                b->batch_mixed_prefill_quantum,
                b->batch_decode_coalesce_us});
        std::lock_guard<std::mutex> lock(b->batch_mu);
        b->coordinator = std::move(coordinator);
        b->batch_running = true;
        b->batch_start_done = true;
        batch_refresh_stats_locked(b);
        b->batch_cv.notify_all();
    } catch (const std::exception &ex) {
        std::lock_guard<std::mutex> lock(b->batch_mu);
        b->batch_start_error = ex.what();
        b->batch_start_done = true;
        b->batch_cv.notify_all();
        return;
    } catch (...) {
        std::lock_guard<std::mutex> lock(b->batch_mu);
        b->batch_start_error =
            "unknown exception constructing continuous batch coordinator";
        b->batch_start_done = true;
        b->batch_cv.notify_all();
        return;
    }

    std::unique_lock<std::mutex> lock(b->batch_mu);
    for (;;) {
        if (!b->batch_controls.empty()) {
            ember_batch_control *control = b->batch_controls.front();
            b->batch_controls.pop_front();
            lock.unlock();
            try {
                control->fn();
            } catch (...) {
                // The control wrapper owns its result and treats an exception
                // as failure. Always wake the caller.
            }
            lock.lock();
            control->done = true;
            batch_refresh_stats_locked(b);
            control->cv.notify_one();
            continue;
        }

        bool admitted = false;
        while (!b->batch_pending.empty() &&
               b->coordinator->scheduler().resident() <
                   b->coordinator->scheduler().capacity()) {
            ember_batch_call *call = b->batch_pending.front();
            b->batch_pending.pop_front();
            int restored = 0;
            if (call->restore_slot >= 0 &&
                b->be->snapshot_used(call->restore_slot)) {
                restored = b->be->snapshot_cur_pos(call->restore_slot);
                if (restored < 0 ||
                    restored > (int)call->request.prompt.size()) {
                    restored = 0;
                    call->restore_slot = -1;
                }
            } else {
                call->restore_slot = -1;
            }
            std::string error;
            lock.unlock();
            std::optional<dflash::common::ContinuousBatchSessionId> id;
            try {
                id = b->coordinator->admit(
                    call->request, call->io, call->restore_slot,
                    restored, &error);
            } catch (const std::exception &ex) {
                error = ex.what();
            } catch (...) {
                error = "unknown resident admission exception";
            }
            lock.lock();
            if (!id) {
                batch_fail_call(call, error.c_str());
                continue;
            }
            call->session_id = *id;
            b->batch_active.emplace(*id, call);
            admitted = true;
        }

        lock.unlock();
        dflash::common::ContinuousBatchRunResult run;
        std::string pump_error;
        try {
            run = b->coordinator->pump(batch_now_us());
        } catch (const std::exception &ex) {
            pump_error = ex.what();
        } catch (...) {
            pump_error = "unknown continuous batch pump exception";
        }
        if (!pump_error.empty()) {
            lock.lock();
            for (auto &entry : b->batch_active) {
                batch_fail_call(entry.second, pump_error.c_str());
            }
            b->batch_active.clear();
            b->batch_stop = true;
            batch_refresh_stats_locked(b);
            continue;
        }
        std::vector<dflash::common::ContinuousBatchSessionId> completed;
        for (const auto &entry : b->batch_active) {
            if (!b->coordinator->terminal(entry.first)) continue;
            auto result = b->coordinator->result(entry.first);
            if (result) entry.second->result = std::move(*result);
            else entry.second->result.fail(
                dflash::common::GenerateErrorCode::BackendSpecific,
                "resident generation completed without a result");
            completed.push_back(entry.first);
        }
        lock.lock();
        for (auto id : completed) {
            auto it = b->batch_active.find(id);
            if (it == b->batch_active.end()) continue;
            it->second->done = true;
            it->second->cv.notify_one();
            b->batch_active.erase(it);
        }
        batch_refresh_stats_locked(b);

        if (b->batch_stop) {
            while (!b->batch_pending.empty()) {
                ember_batch_call *call = b->batch_pending.front();
                b->batch_pending.pop_front();
                batch_fail_call(call, "continuous batch coordinator stopping");
            }
            if (b->batch_active.empty() && b->batch_controls.empty()) {
                std::vector<dflash::common::ContinuousBatchSessionId> retained =
                    b->coordinator->sessions();
                lock.unlock();
                for (auto id : retained) {
                    if (!b->coordinator->terminal(id)) {
                        (void)b->coordinator->cancel(id);
                    }
                    (void)b->coordinator->release(id);
                }
                lock.lock();
                if (b->coordinator->scheduler().resident() == 0) break;
            }
        }

        if (admitted || !completed.empty() ||
            run.status == dflash::common::ContinuousBatchRunStatus::Completed) {
            continue;
        }
        if (run.status ==
                dflash::common::ContinuousBatchRunStatus::Waiting &&
            run.wake_at_us >= 0) {
            const auto deadline =
                std::chrono::steady_clock::time_point(
                    std::chrono::microseconds(run.wake_at_us));
            b->batch_cv.wait_until(lock, deadline);
        } else {
            b->batch_cv.wait(lock);
        }
    }

    lock.unlock();
    b->coordinator.reset();
    b->disk.reset();
    if (b->be) b->be->shutdown();
    lock.lock();
    b->batch_running = false;
    b->batch_cv.notify_all();
}

static bool ember_batch_start(ember_backend *b, int sessions,
                              std::string *error) {
    b->resident = dynamic_cast<ResidentBatchBackend *>(b->be.get());
    if (!b->resident) {
        if (error) *error = "model backend has no resident-session support";
        return false;
    }
    b->batch_sessions = sessions;
    b->batch_prefill_quantum = batch_env_int(
        "DS4_SERVER_PREFILL_QUANTUM", 2048, 1,
        std::numeric_limits<int>::max());
    b->batch_mixed_prefill_quantum = batch_env_int(
        "DS4_SERVER_MIXED_PREFILL_QUANTUM", 128, 1,
        std::numeric_limits<int>::max());
    b->batch_decode_coalesce_us = batch_env_int(
        "DS4_SERVER_DECODE_COALESCE_US", 2000, 0, 100000);
    try {
        b->batch_thread = std::thread(ember_batch_thread_main, b);
    } catch (const std::exception &ex) {
        if (error) *error = ex.what();
        return false;
    }
    std::unique_lock<std::mutex> lock(b->batch_mu);
    b->batch_cv.wait(lock, [b] { return b->batch_start_done; });
    if (!b->batch_running) {
        if (error) {
            *error = b->batch_start_error.empty()
                ? "continuous batch coordinator failed to start"
                : b->batch_start_error;
        }
        lock.unlock();
        if (b->batch_thread.joinable()) b->batch_thread.join();
        return false;
    }
    return true;
}

static void ember_batch_stop(ember_backend *b) {
    if (!b || !b->batch_thread.joinable()) return;
    {
        std::lock_guard<std::mutex> lock(b->batch_mu);
        b->batch_stop = true;
        b->batch_cv.notify_all();
    }
    b->batch_thread.join();
}

template <typename Fn>
static void ember_batch_control_run(ember_backend *b, Fn &&fn) {
    ember_batch_control control;
    control.fn = std::forward<Fn>(fn);
    std::unique_lock<std::mutex> lock(b->batch_mu);
    b->batch_controls.push_back(&control);
    b->batch_cv.notify_one();
    control.cv.wait(lock, [&control] { return control.done; });
}

extern "C" ember_backend *ember_backend_load(const ember_backend_config *cfg,
                                             char **err) {
    if (err) *err = nullptr;
    if (!cfg || !cfg->model_path) {
        if (err) *err = c_err("no model_path");
        return nullptr;
    }
    try {
    auto owned = std::make_unique<ember_backend>();
    ember_backend *b = owned.get();
    b->model_name = cfg->model_name ? cfg->model_name : "deepseek-v4-flash";
    b->n_ctx = cfg->max_ctx > 0 ? cfg->max_ctx : 65536;

    if (!b->tok.load_from_gguf(cfg->model_path)) {
        if (err) *err = c_err("tokenizer load_from_gguf failed");
        return nullptr;
    }

    BackendArgs args;
    args.model_path = cfg->model_path;
    args.device.gpu = 0;
    args.device.max_ctx = b->n_ctx;  // KV cache context; default 8192 is too small
    args.chunk = 2048;
    switch (cfg->ds4_prefill_mode) {
        case EMBER_DS4_PREFILL_EXACT:
            args.ds4_prefill_mode = PrefillAttentionMode::Exact;
            break;
        case EMBER_DS4_PREFILL_DENSE:
            args.ds4_prefill_mode = PrefillAttentionMode::Dense;
            break;
        case EMBER_DS4_PREFILL_SPARSE:
        default:
            args.ds4_prefill_mode = PrefillAttentionMode::Sparse;
            break;
    }
    args.ds4_prefill_mode_set = true;
    args.ds4_fused_decode = true;
    args.ds4_expert_top_k = cfg->expert_top_k;  // 0 = model default

    b->be = dflash::common::create_backend(args);
    if (!b->be) {
        if (err) *err = c_err("create_backend failed");
        return nullptr;
    }

    // Optional disk KV cache (cross-restart). Reuses the backend's snapshot
    // serialization + layout fingerprint via lucebox's DiskPrefixCache.
    if (cfg->kv_cache_dir && cfg->kv_cache_dir[0]) {
        DiskCacheConfig dc;
        dc.cache_dir = cfg->kv_cache_dir;
        if (cfg->kv_cache_mb > 0) {
            const size_t mb = (size_t)cfg->kv_cache_mb;
            if (mb > std::numeric_limits<size_t>::max() / (1024u * 1024u)) {
                if (err) *err = c_err("kv_cache_mb is too large");
                return nullptr;
            }
            dc.budget_bytes = mb * 1024u * 1024u;
        }
        b->disk = std::make_unique<DiskPrefixCache>(dc, *b->be);
        // Fold the path plus high-resolution file identity. Size and
        // second-resolution mtime alone collide for same-sized replacements.
        // Fail closed if metadata cannot be read rather than use path-only
        // identity and risk restoring KV produced by different weights.
        std::array<uint8_t, 16> salt{};
        uint64_t h = 1469598103934665603ULL;
        auto fnv = [&h](const void *data, size_t n) {
            const uint8_t *b = (const uint8_t *)data;
            for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
        };
        for (const char *p = cfg->model_path; *p; ++p) fnv(p, 1);
        struct stat st;
        if (stat(cfg->model_path, &st) != 0) {
            std::fprintf(stderr,
                         "[ember] disk KV cache disabled: cannot stat model %s\n",
                         cfg->model_path);
            b->disk.reset();
        } else {
            uint64_t meta[] = {
                (uint64_t)st.st_dev,
                (uint64_t)st.st_ino,
                (uint64_t)st.st_size,
                (uint64_t)st.st_mtim.tv_sec,
                (uint64_t)st.st_mtim.tv_nsec,
                (uint64_t)st.st_ctim.tv_sec,
                (uint64_t)st.st_ctim.tv_nsec,
            };
            fnv(meta, sizeof(meta));
            for (int i = 0; i < 8; i++) salt[i] = (uint8_t)(h >> (i * 8));
            uint64_t h2 = (h ^ 0x9e3779b97f4a7c15ULL) * 1099511628211ULL;
            for (int i = 0; i < 8; i++) salt[8 + i] = (uint8_t)(h2 >> (i * 8));
            b->cache_identity = salt;
            b->cache_identity_valid = true;
            b->disk->set_identity_salt(salt);
            if (!b->disk->init()) {
                std::fprintf(stderr,
                             "[ember] disk KV cache disabled: initialization failed\n");
                b->disk.reset();
                b->cache_identity_valid = false;
            }
        }
    }
    if (cfg->batch_sessions > 1) {
        std::string batch_error;
        if (!ember_batch_start(b, cfg->batch_sessions, &batch_error)) {
            if (err) *err = c_err(batch_error.c_str());
            return nullptr;
        }
        std::fprintf(stderr,
                     "[ember] continuous batching enabled: resident_sessions=%d "
                     "prefill_quantum=%d mixed_prefill_quantum=%d "
                     "decode_coalesce_us=%lld\n",
                     cfg->batch_sessions, b->batch_prefill_quantum,
                     b->batch_mixed_prefill_quantum,
                     (long long)b->batch_decode_coalesce_us);
    }
    return owned.release();
    } catch (const std::exception &ex) {
        if (err) *err = c_err(ex.what());
        return nullptr;
    } catch (...) {
        if (err) *err = c_err("unknown exception loading backend");
        return nullptr;
    }
}

extern "C" void ember_backend_free(ember_backend *b) {
    ember_batch_stop(b);
    delete b;
}

extern "C" int ember_backend_encode(ember_backend *b, const char *text,
                                    int32_t **ids_out) {
    if (!ids_out) return -1;
    *ids_out = nullptr;
    if (!b) return -1;
    try {
        std::vector<int32_t> ids = b->tok.encode(text ? text : "");
        size_t n = ids.size();
        if (n > (size_t)std::numeric_limits<int>::max() ||
            n > std::numeric_limits<size_t>::max() / sizeof(int32_t))
            return -1;
        int32_t *out =
            (int32_t *)std::malloc((n ? n : 1) * sizeof(int32_t));
        if (!out) return -1;
        if (n) std::memcpy(out, ids.data(), n * sizeof(int32_t));
        *ids_out = out;
        return (int)n;
    } catch (...) {
        return -1;
    }
}

extern "C" const char *ember_backend_token_text(ember_backend *b, int32_t id) {
    // token_text() returns the decoded UTF-8 bytes (byte-level BPE reversed).
    // Stash it per calling generation thread. Concurrent session callbacks
    // must not invalidate one another's token bytes.
    if (!b) return "";
    try {
        static thread_local std::string scratch;
        scratch = b->tok.token_text(id);
        return scratch.c_str();
    } catch (...) {
        return "";
    }
}

// A request that pins no seed used to inherit the engine's fixed default, which
// made sampling deterministic for a given prompt even at temperature 0.6. That
// turned every recoverable model pathology into a permanently fatal one: a
// client that retries a failed turn re-sends the same bytes, gets the same
// tokens, and fails identically.
//
// 2026-08-07 is the worked example. The model re-derived 512 tokens of its own
// replayed <think> content, the prompt-echo watchdog correctly stopped it, and
// ember returned a typed error saying so. agent gateway retried the byte-identical
// request twice more; both reproduced the same failure down to the trailing
// tokens (resp_bytes=114404 twice, identical gen_tail). The turn died and the
// agent went silent. Retry is the one recovery move clients actually implement,
// and a fixed seed guarantees it cannot work.
//
// So an unpinned SAMPLED request gets a fresh seed per generation. Determinism
// is preserved exactly where it is load-bearing: an explicit `seed` still wins,
// greedy decode never reaches here (do_sample is false), and the differential
// validator runs its own greedy AR path through ember_backend_validate rather
// than this function. backend_stub.c is untouched, so the GPU-free gauntlet
// stays reproducible.
static uint64_t next_auto_seed(void) {
    // splitmix64 over a monotonic counter salted once with the clock, so seeds
    // differ across requests AND across restarts. The counter alone would repeat
    // the same sequence after every restart, which is the failure this fixes.
    static std::atomic<uint64_t> counter{0};
    static const uint64_t salt = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()) ^
        static_cast<uint64_t>(::getpid()) << 32;
    uint64_t z = salt + 0x9E3779B97F4A7C15ULL *
                 (counter.fetch_add(1, std::memory_order_relaxed) + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    // 0 is avoided in case the engine treats it as "unset".
    return z ? z : 1ull;
}

static void build_generate_request(const ember_gen_request *req,
                                   GenerateRequest &greq,
                                   DaemonIO &io,
                                   bool *cancelled) {
    greq.prompt.assign(req->prompt, req->prompt + req->n_prompt);
    // main.c already resolves the budget (unset -> context room); a literal 0
    // means "generate nothing", so only a negative (shouldn't occur) falls back.
    greq.n_gen = req->max_tokens < 0 ? 2048 : req->max_tokens;
    greq.force_ar_decode = req->force_ar_decode;
    greq.force_exact_prefill = req->force_exact_prefill;
    greq.sampler.temp = req->greedy ? 0.0f : req->temperature;
    greq.sampler.top_p = req->top_p > 0.0f ? req->top_p : 1.0f;
    greq.sampler.top_k = req->top_k > 0 ? req->top_k : 0;
    if (req->seed_set) greq.sampler.seed = req->seed;
    // Penalties (SamplerCfg supports these; OpenAI clients send them).
    greq.sampler.rep_pen = req->rep_pen > 0.0f ? req->rep_pen : 1.0f;
    if (req->rep_window > 0) greq.sampler.rep_window = req->rep_window;
    greq.sampler.freq_pen = req->freq_pen;
    greq.sampler.pres_pen = req->pres_pen;
    greq.sampler.min_p = req->min_p;  // enforced by the extended SamplerCfg
    // DRY. Zero/negative means "unset" on the ABI so a caller that never touches
    // these fields inherits SamplerCfg's llama.cpp-compatible defaults rather
    // than a base of 0 (which would make the penalty collapse to a constant).
    greq.sampler.dry_multiplier = req->dry_multiplier;
    if (req->dry_base > 1.0f) greq.sampler.dry_base = req->dry_base;
    if (req->dry_allowed_length >= 0)
        greq.sampler.dry_allowed_length = req->dry_allowed_length;
    if (req->dry_window != 0) greq.sampler.dry_window = req->dry_window;
    // Match lucebox's own server: sample through the logit path when any modifier
    // is active (temp>0 / penalties / top_k / min_p), else greedy argmax (DSpark).
    greq.do_sample = greq.sampler.needs_logit_processing();
    // Must follow do_sample: only a genuinely sampled decode gets a fresh seed.
    // Greedy argmax ignores the seed anyway, and leaving it fixed there keeps
    // greedy reproducible for anyone bisecting a decode.
    if (!req->seed_set && greq.do_sample) {
        greq.sampler.seed = next_auto_seed();
        // Logged because varying the seed otherwise makes a bad generation
        // impossible to reproduce -- this line is how you replay one.
        fprintf(stderr, "[ember] auto-seed=%llu (client sent none)\n",
                (unsigned long long)greq.sampler.seed);
    }
    greq.snap_slot = req->snap_slot;   // inline snapshot for future reuse
    greq.snap_pos  = req->snap_pos;

    // Level-2 thinking force-close: when the reply budget is reached, the
    // backend injects the close sequence (</think>...) so the model writes a
    // visible answer instead of thinking until EOS. Setting this routes
    // through AR decode (spec skipped) — matches lucebox's thinking path.
    if (req->budget_close_ids && req->n_budget_close > 0 && req->reply_budget > 0) {
        greq.budget_hook.close_token_ids.assign(
            req->budget_close_ids, req->budget_close_ids + req->n_budget_close);
        if (req->budget_natural_close_ids &&
            req->n_budget_natural_close > 0) {
            greq.budget_hook.natural_close_token_ids.assign(
                req->budget_natural_close_ids,
                req->budget_natural_close_ids +
                    req->n_budget_natural_close);
        }
        greq.budget_hook.hard_limit_remaining = req->reply_budget;
    }

    // Bridge the per-token callback (backend uses req.on_token via
    // with_token_callback). Return false cancels generation.
    ember_token_cb tok_cb = req->on_token;
    void *ud = req->ud;
    // #7: the backend cancels on its own DaemonIO copy (out_io = io.with_token_
    // _callback), so the bridge's `io.cancelled` never flips. Track cancellation
    // here in the callbacks the bridge owns.
    if (tok_cb) {
        greq.on_token = [tok_cb, ud, cancelled](int32_t t) -> bool {
            if (!tok_cb(t, ud)) {
                if (cancelled) *cancelled = true;
                return false;
            }
            return true;
        };
    }

    // B6: bridge the structural-greedy predicate (consulted per token by the AR loop).
    if (req->force_greedy) {
        bool (*fg)(void *) = req->force_greedy;
        void *fud = req->fg_ud;
        greq.force_greedy_next = [fg, fud]() -> bool { return fg(fud); };
    }

    io.stream_fd = -1;  // callback-driven, no pipe
    ember_keepalive_cb ka_cb = req->on_prefill;
    if (ka_cb) {
        io.on_prefill_keepalive = [ka_cb, ud, cancelled]() -> bool {
            if (!ka_cb(ud)) {
                if (cancelled) *cancelled = true;
                return false;
            }
            return true;
        };
    }
}

static ember_gen_result map_generate_result(
        const GenerateResult &res, bool cancelled, int requested_tokens) {
    ember_gen_result r;
    memset(&r, 0, sizeof(r));
    strcpy(r.finish_reason, "stop");

    r.ok = res.ok();
    r.cancelled = cancelled;
    r.n_generated = (int)res.tokens.size();
    r.prefill_tokens = res.prefill_tokens;
    r.prefill_s = res.prefill_s;
    r.decode_s = res.decode_s;
    r.accept_rate = res.accept_rate;
    // Surface the backend's generation-quality + error signals (were dropped).
    r.budget_forced_close     = res.budget_forced_close;
    r.degenerate_decode_close = res.degenerate_decode_close;
    snprintf(r.termination_reason, sizeof(r.termination_reason), "%s",
             res.termination_reason.c_str());
    r.empty_visible_output    = res.empty_visible_output;
    r.spec_decode_ran         = res.spec_decode_ran;
    snprintf(r.prefill_mode, sizeof(r.prefill_mode), "%s",
             res.prefill_mode.c_str());
    snprintf(r.prefill_reason, sizeof(r.prefill_reason), "%s",
             res.prefill_reason.c_str());
    r.snapshot_saved          = res.snapshot_saved;  // #2: real-save signal
    if (!res.ok()) {
        std::string_view code = res.error_code();
        std::string_view detail = res.error_detail();
        snprintf(r.error_code, sizeof(r.error_code), "%.*s",
                 (int)code.size(), code.data());
        snprintf(r.error_detail, sizeof(r.error_detail), "%.*s",
                 (int)detail.size(), detail.data());
    }
    if (requested_tokens > 0 && r.n_generated >= requested_tokens) {
        strcpy(r.finish_reason, "length");
    }
    return r;
}

// DSML tool-call region markers for the engine's prompt-echo watchdog (see
// GenerateRequest::tool_region_open_ids). Encoded with the model's own
// tokenizer because these are special tokens that do not survive a
// detokenize/retokenize round trip, so the ids must come from the same
// vocabulary decode will emit. Adjacent string literals keep \x from eating the
// following 'D' (same reason as sse.c's PIPE).
#define DSML_PIPE_U8 "\xef\xbd\x9c"
static void set_tool_region_ids(ember_backend *b, GenerateRequest &greq) {
    if (!b) return;
    try {
        // Deliberately WITHOUT the closing '>'. The template renders the
        // marker followed by a newline, and the model emits ">\n" as one
        // merged token, so encoding the full marker yields a final '>' token
        // the generated stream never contains -- the sequence match then never
        // fires and every consumer is silently inert. Measured directly:
        //   expect [30,128825,72461,4941,12548,32]
        //   actual [30,128825,72461,4941,12548,1018]
        // The 5-token prefix cannot merge with what follows and still uniquely
        // identifies the block.
        greq.tool_region_open_ids =
            b->tok.encode("<" DSML_PIPE_U8 "DSML" DSML_PIPE_U8 "tool_calls");
        greq.tool_region_close_ids =
            b->tok.encode("</" DSML_PIPE_U8 "DSML" DSML_PIPE_U8 "tool_calls");
        // DRY MUST NEVER PENALISE THE PROTOCOL ITSELF.
        //
        // A tool-call marker is, by design, the identical token sequence every
        // single time: <  ｜DSML｜  tool  _c  alls. In a tool-heavy conversation
        // those recur every few hundred tokens, well inside dry_window, so DRY
        // scores re-emitting a CORRECT marker as though it were a runaway
        // repetition and the penalty grows geometrically with each call.
        //
        // Measured: an unbroken marker drove the model to spell the delimiter in
        // ASCII instead, so the tool never fired and raw markup reached the user.
        // test/test_sampler_dry.cpp reproduces it as an assertion.
        //
        // One breaker closes this for the WHOLE protocol: ｜DSML｜ is a single
        // token (128825) shared by every marker -- tool_calls, invoke,
        // parameter, and their closers (chat_template.c and tool_grammar.c both
        // build every marker from the same "<" PIPE "DSML" PIPE prefix). A
        // breaker is never itself penalised AND terminates backward matching, so
        // a match cannot cross it and no marker-containing span ever reaches
        // dry_allowed_length.
        //
        // Deliberately NOT the whole marker: '<', 'tool', '_c', 'alls' are
        // ordinary text and code, and exempting them would blunt DRY broadly.
        // The delimiter alone is surgical.
        greq.sampler.dry_breaker_ids =
            b->tok.encode(DSML_PIPE_U8 "DSML" DSML_PIPE_U8);
    } catch (...) {
        // Watchdog simply keeps its previous unscoped behaviour.
        greq.tool_region_open_ids.clear();
        greq.tool_region_close_ids.clear();
        greq.sampler.dry_breaker_ids.clear();
    }
    // Both the echo watchdog scoping and constrained decoding are silently
    // inert if these come back empty, so say so once rather than let a
    // tokenizer change disable them without a trace.
    static bool logged = false;
    if (!logged) {
        logged = true;
        std::fprintf(stderr,
                     "[ember] DSML tool-region tokens: open=%zu close=%zu "
                     "dry_breakers=%zu\n",
                     greq.tool_region_open_ids.size(),
                     greq.tool_region_close_ids.size(),
                     greq.sampler.dry_breaker_ids.size());
    }
}


// ── constrained tool-call decoding ──────────────────────────────────────
//
// Masks the sampler to a grammar derived from the request's tool schemas, but
// ONLY inside a DSML tool-call block. Prose and reasoning are untouched: the
// mask reports active() only between the tool_calls open and close markers, so
// a turn that never opens a block is decoded exactly as before.
//
// This lives here rather than in the engine because it needs the tokenizer and
// xgrammar, neither of which the engine knows about. The engine sees only the
// abstract TokenMask interface (engine/dflash/common/token_mask.h).
namespace {

bool ids_end_with(const std::vector<int32_t> &hay,
                  const std::vector<int32_t> &needle) {
    if (needle.empty() || hay.size() < needle.size()) return false;
    return std::equal(needle.begin(), needle.end(),
                      hay.end() - (std::ptrdiff_t)needle.size());
}

class DsmlGrammarMask final : public dflash::common::TokenMask {
public:
    DsmlGrammarMask(xgrammar::CompiledGrammar grammar,
                    std::vector<int32_t> open_ids,
                    std::vector<int32_t> close_ids,
                    int vocab)
        : grammar_(std::move(grammar)),
          open_(std::move(open_ids)),
          close_(std::move(close_ids)),
          vocab_(vocab),
          bitmask_((size_t)xgrammar::GetBitmaskSize(vocab)) {}

    void accept(int32_t token) override {
        if (!matcher_) {
            // Waiting for a tool block to open.
            window_.push_back(token);
            const size_t cap = open_.size();
            if (cap && window_.size() > cap) window_.erase(window_.begin());
            if (!ids_end_with(window_, open_)) return;
            window_.clear();
            // The grammar's root begins with the open marker, so replay it to
            // put the fresh matcher in the same state the stream is in.
            matcher_ = std::make_unique<xgrammar::GrammarMatcher>(
                grammar_, std::nullopt, /*terminate_without_stop_token=*/true);
            for (int32_t t : open_) {
                if (!matcher_->AcceptToken(t)) {
                    std::fprintf(stderr,
                                 "[ember] tool grammar: marker replay REJECTED "
                                 "at token %d -- mask disabled for this call\n",
                                 t);
                    matcher_.reset();
                    return;
                }
            }
            ++regions_;
            return;
        }
        // A token the grammar rejects should be impossible while we are
        // masking, but a server-injected token (thinking force-close) can
        // arrive here. Stop constraining rather than wedge the block.
        //
        // This fail-open is silent by construction, so it can disable the mask
        // mid-call and leave the rest of the block unconstrained -- producing
        // exactly the malformed shapes the grammar exists to prevent. Log it:
        // an unexplained rejection here means the mask is not covering some
        // sampling path, which is a defect, not an expected event.
        if (!matcher_->AcceptToken(token)) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr,
                             "[ember] tool grammar DISENGAGED mid-call: token "
                             "%d rejected after %zu accepted\n",
                             token, accepted_);
            }
            matcher_.reset();
            return;
        }
        ++accepted_;
        if (matcher_->IsTerminated()) matcher_.reset();
    }

    bool active() const override { return matcher_ != nullptr; }

    void apply(float *logits, int vocab) override {
        if (!matcher_ || vocab != vocab_) return;
        DLTensor t{};
        t.data = bitmask_.data();
        t.device = DLDevice{kDLCPU, 0};
        t.ndim = 1;
        t.dtype = xgrammar::GetBitmaskDLType();
        shape_[0] = (int64_t)bitmask_.size();
        t.shape = shape_;
        t.strides = nullptr;
        t.byte_offset = 0;
        matcher_->FillNextTokenBitmask(&t);

        // A bitmask that allows nothing would make sampling undefined. That
        // should not happen, but failing open beats emitting garbage.
        bool any = false;
        for (int32_t w : bitmask_) {
            if (w != 0) { any = true; break; }
        }
        if (!any) return;

        // Prove engagement once per process. Without this, "the mask is
        // attached" and "the mask is constraining" look identical from
        // outside -- which is exactly how two inert features hid all day.
        static bool announced = false;
        if (!announced) {
            announced = true;
            int allowed = 0;
            for (int32_t w : bitmask_) {
                for (int b = 0; b < 32; b++) if ((w >> b) & 1) ++allowed;
            }
            std::fprintf(stderr,
                         "[ember] tool grammar ENGAGED: %d of %d tokens allowed "
                         "(%.3f%%)\n", allowed, vocab,
                         100.0 * allowed / (vocab ? vocab : 1));
        }

        for (int i = 0; i < vocab; i++) {
            const bool allowed =
                (bitmask_[(size_t)(i >> 5)] >> (i & 31)) & 1;
            if (!allowed) logits[i] = -std::numeric_limits<float>::infinity();
        }
    }

private:
    xgrammar::CompiledGrammar             grammar_;
    std::unique_ptr<xgrammar::GrammarMatcher> matcher_;
    std::vector<int32_t>                  open_;
    std::vector<int32_t>                  close_;
    std::vector<int32_t>                  window_;
    int                                   vocab_;
    std::size_t                           regions_ = 0;
    std::size_t                           accepted_ = 0;
    std::vector<int32_t>                  bitmask_;
    int64_t                               shape_[1] = {0};
};

}  // namespace

// Attach a grammar mask when the request carries one. Failure is never fatal:
// unconstrained decoding is the previous behaviour and still correct, since
// tool_schema.c validates every call afterwards regardless.
static void set_tool_grammar(ember_backend *b, const ember_gen_request *req,
                             GenerateRequest &greq) {
    if (!b || !req || !req->tool_grammar || !req->tool_grammar[0]) return;
    if (greq.tool_region_open_ids.empty()) return;
    try {
        std::lock_guard<std::mutex> lock(b->xg_mu);
        if (!b->xg_compiler) {
            // BYTE_LEVEL, not BYTE_FALLBACK: the GGUF declares
            // tokenizer.ggml.model=gpt2 and 51,172 tokens carry the byte-level
            // space marker while zero use sentencepiece's U+2581 or <0xNN>
            // byte fallbacks. With the wrong type xgrammar decodes "Ġname" as
            // those literal characters instead of " name", so the bitmask is
            // built over mis-decoded strings: it permits tokens the grammar
            // does not accept, and AcceptToken then rejects what was just
            // sampled, silently disengaging the mask mid-call.
            b->xg_vocab = std::make_unique<xgrammar::TokenizerInfo>(
                b->tok.raw_vocab(), xgrammar::VocabType::BYTE_LEVEL,
                (int)b->tok.vocab_size());
            b->xg_compiler =
                std::make_unique<xgrammar::GrammarCompiler>(*b->xg_vocab);
        }
        const std::string key(req->tool_grammar);
        auto it = b->xg_cache.find(key);
        if (it == b->xg_cache.end()) {
            if (b->xg_cache.size() >= 8) b->xg_cache.clear();
            it = b->xg_cache.emplace(
                key, b->xg_compiler->CompileGrammar(
                         xgrammar::Grammar::FromEBNF(key))).first;
        }
        greq.token_mask = std::make_shared<DsmlGrammarMask>(
            it->second, greq.tool_region_open_ids, greq.tool_region_close_ids,
            (int)b->tok.vocab_size());
        // Once per process: constrained decoding is otherwise invisible, and
        // silence would look identical to the feature being disabled.
        static bool announced = false;
        if (!announced) {
            announced = true;
            std::fprintf(stderr,
                         "[ember] tool grammar active: %zu bytes, %zu tools\n",
                         key.size(), b->xg_cache.size());
        }
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "[ember] tool grammar unavailable: %s\n", ex.what());
    } catch (...) {
        std::fprintf(stderr, "[ember] tool grammar unavailable\n");
    }
}

static ember_gen_result backend_generate_impl(ember_backend *b,
                                              const ember_gen_request *req) {
    GenerateRequest greq;
    DaemonIO io;
    bool cancelled = false;
    build_generate_request(req, greq, io, &cancelled);
    set_tool_region_ids(b, greq);
    set_tool_grammar(b, req, greq);
    GenerateResult res =
        (req->restore_slot >= 0 &&
         b->be->snapshot_used(req->restore_slot))
        ? b->be->restore_and_generate(req->restore_slot, greq, io)
        : b->be->generate(greq, io);
    return map_generate_result(res, cancelled, req->max_tokens);
}

static ember_gen_result backend_generate_batched(
        ember_backend *b, const ember_gen_request *req) {
    // A caller cannot begin another generation until it releases the prior
    // resident lease used by snapshot/tool-continuation post-processing.
    auto previous = tls_batch_sessions.find(b);
    if (previous != tls_batch_sessions.end()) {
        dflash::common::ContinuousBatchSessionId id = previous->second;
        ember_batch_control_run(b, [b, id] {
            (void)b->coordinator->release(id);
        });
        tls_batch_sessions.erase(previous);
    }

    ember_batch_call call;
    call.restore_slot = req->restore_slot;
    build_generate_request(req, call.request, call.io, &call.cancelled);
    set_tool_region_ids(b, call.request);
    set_tool_grammar(b, req, call.request);
    {
        std::unique_lock<std::mutex> lock(b->batch_mu);
        if (b->batch_stop || !b->batch_running) {
            GenerateResult failed;
            failed.fail(dflash::common::GenerateErrorCode::BackendSpecific,
                        "continuous batch coordinator is not running");
            return map_generate_result(failed, false, req->max_tokens);
        }
        b->batch_pending.push_back(&call);
        b->batch_cv.notify_one();
        call.cv.wait(lock, [&call] { return call.done; });
    }
    if (call.session_id != 0) {
        tls_batch_sessions[b] = call.session_id;
    }
    return map_generate_result(call.result, call.cancelled, req->max_tokens);
}

extern "C" ember_gen_result ember_backend_generate(
        ember_backend *b, const ember_gen_request *req) {
    ember_gen_result failed{};
    std::strcpy(failed.finish_reason, "stop");
    if (!b || !b->be || !req || !req->prompt || req->n_prompt < 0) {
        std::strcpy(failed.error_code, "invalid_request");
        std::strcpy(failed.error_detail, "invalid backend generation request");
        return failed;
    }
    try {
        return b->batch_sessions > 1
            ? backend_generate_batched(b, req)
            : backend_generate_impl(b, req);
    } catch (const std::exception &ex) {
        std::strcpy(failed.error_code, "backend_exception");
        std::snprintf(failed.error_detail, sizeof(failed.error_detail),
                      "%s", ex.what());
        return failed;
    } catch (...) {
        std::strcpy(failed.error_code, "backend_exception");
        std::strcpy(failed.error_detail, "unknown backend exception");
        return failed;
    }
}

extern "C" void ember_backend_generation_release(ember_backend *b) {
    if (!b || b->batch_sessions <= 1) return;
    auto it = tls_batch_sessions.find(b);
    if (it == tls_batch_sessions.end()) return;
    const dflash::common::ContinuousBatchSessionId id = it->second;
    bool released = false;
    ember_batch_control_run(b, [b, id, &released] {
        released = b->coordinator->release(id);
    });
    if (!released) {
        std::fprintf(stderr,
                     "[ember] failed to release resident batch session %llu\n",
                     (unsigned long long)id);
    }
    tls_batch_sessions.erase(it);
    {
        std::lock_guard<std::mutex> lock(b->batch_mu);
        b->batch_cv.notify_one();
    }
}

extern "C" bool ember_backend_batch_enabled(const ember_backend *b) {
    return b && b->batch_sessions > 1;
}

extern "C" bool ember_backend_batch_stats_get(
        const ember_backend *b, ember_batch_stats *stats) {
    if (!b || !stats) return false;
    std::memset(stats, 0, sizeof(*stats));
    stats->enabled = b->batch_sessions > 1;
    stats->capacity = b->batch_sessions;
    if (!stats->enabled) return true;

    ember_backend *mutable_b = const_cast<ember_backend *>(b);
    std::lock_guard<std::mutex> lock(mutable_b->batch_mu);
    *stats = mutable_b->batch_stats_cache;
    stats->pending = (int)mutable_b->batch_pending.size();
    return true;
}

extern "C" bool ember_backend_snapshot_now(ember_backend *b, int slot) {
    // B3 Layer 2: snapshot the live cache_ at its current (post-generate) cur_pos.
    // DeepSeek4Backend::snapshot_save captures cache_ at cur_pos + returns true
    // on success; snapshot_used(slot) then reports it as restorable.
    if (!b || !b->be || slot < 0) return false;
    bool saved = false;
    auto lease = tls_batch_sessions.find(b);
    if (b->batch_sessions > 1) {
        const dflash::common::ContinuousBatchSessionId id =
            lease == tls_batch_sessions.end() ? 0 : lease->second;
        ember_batch_control_run(b, [b, id, slot, &saved] {
            if (id != 0) {
                saved = b->resident->resident_session_snapshot(id, slot);
            } else {
                saved = b->be->snapshot_save(slot) &&
                        b->be->snapshot_used(slot);
            }
        });
        return saved;
    }
    try {
        return b->be->snapshot_save(slot) && b->be->snapshot_used(slot);
    } catch (...) {
        return false;
    }
}

extern "C" void ember_backend_release_idle_graphs(ember_backend *b) {
    if (!b || !b->be) return;
    if (b->batch_sessions > 1) {
        ember_batch_control_run(b, [b] { b->be->release_idle_graphs(); });
        return;
    }
    try {
        b->be->release_idle_graphs();
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "[ember] idle graph release failed: %s\n", ex.what());
    } catch (...) {
        std::fprintf(stderr, "[ember] idle graph release failed\n");
    }
}

extern "C" int ember_backend_snapshot_pos(const ember_backend *b, int slot) {
    if (!b || !b->be || slot < 0) return -1;
    if (b->batch_sessions > 1) {
        int pos = -1;
        ember_backend *mutable_b = const_cast<ember_backend *>(b);
        ember_batch_control_run(mutable_b, [mutable_b, slot, &pos] {
            auto ref = mutable_b->be->snapshot_ref(slot);
            pos = ref.ctx ? ref.cur_pos : -1;
        });
        return pos;
    }
    try {
        auto ref = b->be->snapshot_ref(slot);
        return ref.ctx ? ref.cur_pos : -1;
    } catch (...) {
        return -1;
    }
}

static bool validation_tokens_equal(const std::vector<int32_t> & expected,
                                    const std::vector<int32_t> & actual,
                                    ember_validation_report * report) {
    const size_t common = std::min(expected.size(), actual.size());
    for (size_t i = 0; i < common; ++i) {
        if (expected[i] != actual[i]) {
            if (report->mismatch_index < 0) {
                report->mismatch_index = (int)i;
                report->expected_token = expected[i];
                report->actual_token = actual[i];
            }
            return false;
        }
    }
    if (expected.size() != actual.size()) {
        if (report->mismatch_index < 0) {
            report->mismatch_index = (int)common;
            report->expected_token =
                common < expected.size() ? expected[common] : -1;
            report->actual_token =
                common < actual.size() ? actual[common] : -1;
        }
        return false;
    }
    return true;
}

static bool backend_validate_impl(
        ember_backend *b, const int32_t *prompt, int n_prompt, int n_gen,
        ember_validation_report *report) {
    if (!b || !b->be || !prompt || n_prompt < 1 || n_gen < 2 || !report) {
        return false;
    }
    std::memset(report, 0, sizeof(*report));
    report->mismatch_index = -1;
    report->expected_token = -1;
    report->actual_token = -1;

    GenerateRequest ar;
    ar.prompt = vec(prompt, n_prompt);
    ar.n_gen = n_gen;
    ar.sampler.temp = 0.0f;
    ar.do_sample = false;
    ar.force_ar_decode = true;
    ar.force_exact_prefill = true;
    ar.snap_slot = 0;
    ar.snap_pos = n_prompt;

    DaemonIO io;
    io.stream_fd = -1;
    GenerateResult baseline = b->be->generate(ar, io);
    report->baseline_tokens = (int)baseline.tokens.size();
    report->snapshot_ok =
        baseline.ok() && baseline.snapshot_saved &&
        b->be->snapshot_used(0) &&
        b->be->snapshot_cur_pos(0) == n_prompt;
    if (!report->snapshot_ok) {
        std::snprintf(report->detail, sizeof(report->detail),
                      "AR baseline or inline snapshot failed: %.*s",
                      (int)baseline.error_detail().size(),
                      baseline.error_detail().data());
        return true;
    }

    GenerateRequest spec = ar;
    spec.force_ar_decode = false;
    spec.snap_slot = -1;
    spec.snap_pos = -1;
    // Exercise both DSpark entry seams. Restoring the AR snapshot isolates
    // decode verification/commit parity; a fresh run additionally proves that
    // passive feature capture during exact prefill leaves target logits intact.
    GenerateResult restored_speculative =
        b->be->restore_and_generate(0, spec, io);
    GenerateResult speculative = b->be->generate(spec, io);
    report->spec_tokens = (int)speculative.tokens.size();
    report->spec_checked =
        restored_speculative.ok() &&
        restored_speculative.spec_decode_ran &&
        speculative.ok() && speculative.spec_decode_ran;
    report->spec_accept_rate = speculative.accept_rate;
    report->spec_exact =
        restored_speculative.ok() && speculative.ok() &&
        validation_tokens_equal(
            baseline.tokens, restored_speculative.tokens, report) &&
        validation_tokens_equal(baseline.tokens, speculative.tokens, report);

    if (b->disk && n_prompt >= 512) {
        report->disk_checked = true;
        const bool saved = b->disk->save(0, ar.prompt);
        const bool loaded = saved && b->disk->lookup(ar.prompt, 1);
        GenerateResult disk_ar;
        if (loaded) {
            GenerateRequest disk_req = ar;
            disk_req.snap_slot = -1;
            disk_req.snap_pos = -1;
            disk_ar = b->be->restore_and_generate(1, disk_req, io);
        }
        report->disk_tokens = (int)disk_ar.tokens.size();
        report->disk_exact =
            loaded && disk_ar.ok() &&
            validation_tokens_equal(baseline.tokens, disk_ar.tokens, report);
    } else {
        report->disk_exact = true;
    }

    report->batch_exact = true;
    if (b->batch_sessions > 1 && b->coordinator) {
        report->batch_checked = true;
        GenerateRequest resident_request = ar;
        resident_request.snap_slot = -1;
        resident_request.snap_pos = -1;
        std::vector<dflash::common::ContinuousBatchSessionId> ids;
        std::string batch_error;
        for (int row = 0; row < 2; ++row) {
            auto id = b->coordinator->admit(
                resident_request, io, -1, 0, &batch_error);
            if (!id) {
                report->batch_exact = false;
                break;
            }
            ids.push_back(*id);
        }
        report->batch_rows = (int)ids.size();
        const int max_pumps =
            std::max(100, (n_prompt / 128 + n_gen + 8) * 4);
        const std::int64_t started = batch_now_us();
        for (int step = 0; step < max_pumps; ++step) {
            bool all_terminal = !ids.empty();
            for (auto id : ids)
                all_terminal = all_terminal && b->coordinator->terminal(id);
            if (all_terminal) break;
            (void)b->coordinator->pump(
                started + (std::int64_t)step * 1000000);
        }
        for (auto id : ids) {
            if (!b->coordinator->terminal(id)) {
                report->batch_exact = false;
                (void)b->coordinator->cancel(id);
            }
            auto result = b->coordinator->result(id);
            if (!result || !result->ok()) {
                report->batch_exact = false;
            } else {
                report->batch_tokens += (int)result->tokens.size();
                if (!validation_tokens_equal(
                        baseline.tokens, result->tokens, report)) {
                    report->batch_exact = false;
                }
            }
            if (b->coordinator->terminal(id))
                (void)b->coordinator->release(id);
        }
        if (ids.size() != 2) report->batch_exact = false;
    }

    report->ok = report->snapshot_ok && restored_speculative.ok() &&
                 speculative.ok() &&
                 report->spec_exact &&
                 (!report->disk_checked || report->disk_exact) &&
                 (!report->batch_checked || report->batch_exact);
    if (!restored_speculative.ok() || !speculative.ok()) {
        const GenerateResult & failed = restored_speculative.ok()
            ? speculative
            : restored_speculative;
        std::snprintf(report->detail, sizeof(report->detail),
                      "snapshot restore/spec path failed: %.*s",
                      (int)failed.error_detail().size(),
                      failed.error_detail().data());
    } else if (!report->spec_exact || !report->disk_exact ||
               !report->batch_exact) {
        std::snprintf(report->detail, sizeof(report->detail),
                      "token mismatch at %d: expected=%d actual=%d",
                      report->mismatch_index, report->expected_token,
                      report->actual_token);
    } else if (!report->spec_checked) {
        std::snprintf(report->detail, sizeof(report->detail),
                      "snapshot/disk exact; DSpark unavailable or did not run");
    } else {
        std::snprintf(report->detail, sizeof(report->detail),
                      "AR, restored/fresh DSpark, disk, and resident batch are token-exact");
    }
    return true;
}

extern "C" bool ember_backend_validate(
        ember_backend *b, const int32_t *prompt, int n_prompt, int n_gen,
        ember_validation_report *report) {
    if (!report) return false;
    if (b && b->batch_sessions > 1) {
        bool invoked = false;
        ember_batch_control_run(b, [=, &invoked] {
            invoked = backend_validate_impl(
                b, prompt, n_prompt, n_gen, report);
        });
        return invoked;
    }
    try {
        return backend_validate_impl(b, prompt, n_prompt, n_gen, report);
    } catch (const std::exception &ex) {
        std::memset(report, 0, sizeof(*report));
        std::snprintf(report->detail, sizeof(report->detail),
                      "backend validation exception: %s", ex.what());
        return true;
    } catch (...) {
        std::memset(report, 0, sizeof(*report));
        std::snprintf(report->detail, sizeof(report->detail),
                      "unknown backend validation exception");
        return true;
    }
}

extern "C" int ember_backend_n_ctx(const ember_backend *b) {
    return b ? b->n_ctx : 0;
}
extern "C" const char *ember_backend_model_name(const ember_backend *b) {
    return b ? b->model_name.c_str() : "";
}
extern "C" int32_t ember_backend_eos_id(const ember_backend *b) {
    return b ? b->tok.eos_id() : -1;
}

extern "C" bool ember_backend_disk_enabled(const ember_backend *b) {
    return b && b->disk != nullptr;
}
extern "C" int ember_backend_disk_prefix(ember_backend *b, const int32_t *p, int n) {
    if (!b || !b->disk || !p || n <= 0) return 0;
    if (b->batch_sessions > 1) {
        int prefix = 0;
        std::vector<int32_t> prompt = vec(p, n);
        ember_batch_control_run(b, [b, &prompt, &prefix] {
            prefix = b->disk->longest_prefix_len(prompt);
        });
        return prefix;
    }
    try {
        return b->disk->longest_prefix_len(vec(p, n));
    } catch (...) {
        return 0;
    }
}
extern "C" bool ember_backend_disk_lookup(ember_backend *b, const int32_t *p,
                                          int len, int slot) {
    if (!b || !b->disk || !p || len <= 0 || slot < 0) return false;
    if (b->batch_sessions > 1) {
        bool loaded = false;
        std::vector<int32_t> prompt = vec(p, len);
        ember_batch_control_run(b, [b, &prompt, slot, &loaded] {
            loaded = b->disk->lookup(prompt, slot);
        });
        return loaded;
    }
    try {
        return b->disk->lookup(vec(p, len), slot);
    } catch (...) {
        return false;
    }
}
extern "C" bool ember_backend_disk_save(ember_backend *b, int slot,
                                        const int32_t *p, int cut, int reason) {
    if (!b || !b->disk || !p || cut <= 0 || slot < 0) return false;
    const uint8_t r = (uint8_t)reason;
    if (b->batch_sessions > 1) {
        bool saved = false;
        std::vector<int32_t> prompt = vec(p, cut);
        ember_batch_control_run(b, [b, &prompt, slot, r, &saved] {
            saved = b->disk->save(slot, prompt, r);
        });
        return saved;
    }
    try {
        return b->disk->save(slot, vec(p, cut), r);
    } catch (...) {
        return false;
    }
}

extern "C" bool ember_backend_cache_identity(const ember_backend *b,
                                              uint8_t out[16]) {
    if (!b || !out || !b->disk || !b->cache_identity_valid) return false;
    std::memcpy(out, b->cache_identity.data(), b->cache_identity.size());
    return true;
}
