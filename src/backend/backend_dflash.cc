// backend_dflash.cc — the real backend: an extern-C shim implementing Ember's
// ember_backend.h over lucebox's libdflash_common (the tuned ROCMFP / graphs-ON
// / DSpark DeepSeek4 forward pass + its byte-exact tokenizer). Ember's C server
// links this instead of backend_stub.o to serve real tokens; nothing above the
// ABI changes.
//
// This translation unit is C++ (it uses the lucebox C++ classes) but exports
// only the C ABI. It compiles against the lucebox headers and links the static
// lib; it must be built in the ROCm/HIP container (the C server does not).
#include "ember_backend.h"

#include <sys/stat.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "common/backend_factory.h"
#include "common/model_backend.h"
#include "common/sampler.h"
#include "server/disk_prefix_cache.h"
#include "server/tokenizer.h"

using dflash::common::BackendArgs;
using dflash::common::DaemonIO;
using dflash::common::DiskCacheConfig;
using dflash::common::DiskPrefixCache;
using dflash::common::GenerateRequest;
using dflash::common::GenerateResult;
using dflash::common::ModelBackend;
using dflash::common::PrefillAttentionMode;
using dflash::common::Tokenizer;

struct ember_backend {
    std::unique_ptr<ModelBackend>    be;
    std::unique_ptr<DiskPrefixCache> disk;  // cross-restart KV persistence (opt)
    Tokenizer                        tok;
    std::string                      model_name;
    std::string                      tok_scratch;
    int                              n_ctx = 65536;
};

static std::vector<int32_t> vec(const int32_t *p, int n) {
    return std::vector<int32_t>(p, p + (n > 0 ? n : 0));
}

static char *c_err(const char *m) { return m ? strdup(m) : nullptr; }

extern "C" ember_backend *ember_backend_load(const ember_backend_config *cfg,
                                             char **err) {
    if (!cfg || !cfg->model_path) { if (err) *err = c_err("no model_path"); return nullptr; }
    auto *b = new ember_backend();
    b->model_name = cfg->model_name ? cfg->model_name : "deepseek-v4-flash";
    b->n_ctx = cfg->max_ctx > 0 ? cfg->max_ctx : 65536;

    if (!b->tok.load_from_gguf(cfg->model_path)) {
        if (err) *err = c_err("tokenizer load_from_gguf failed");
        delete b;
        return nullptr;
    }

    BackendArgs args;
    args.model_path = cfg->model_path;
    args.device.gpu = 0;
    args.device.max_ctx = b->n_ctx;  // KV cache context; default 8192 is too small
    args.chunk = 2048;
    args.ds4_prefill_mode = PrefillAttentionMode::Sparse;
    args.ds4_prefill_mode_set = true;
    args.ds4_fused_decode = true;
    args.ds4_expert_top_k = cfg->expert_top_k;  // 0 = model default

    b->be = dflash::common::create_backend(args);
    if (!b->be) {
        if (err) *err = c_err("create_backend failed");
        delete b;
        return nullptr;
    }

    // Optional disk KV cache (cross-restart). Reuses the backend's snapshot
    // serialization + layout fingerprint via lucebox's DiskPrefixCache.
    if (cfg->kv_cache_dir && cfg->kv_cache_dir[0]) {
        DiskCacheConfig dc;
        dc.cache_dir = cfg->kv_cache_dir;
        if (cfg->kv_cache_mb > 0)
            dc.budget_bytes = (size_t)cfg->kv_cache_mb * 1024 * 1024;
        b->disk = std::make_unique<DiskPrefixCache>(dc, *b->be);
        // Identity salt: fold the model path AND file size + mtime, so replacing
        // the model in-place (a different model at the same path — the disk
        // layout fingerprint only guards tensor shapes, not weights) invalidates
        // cached KV instead of silently restoring stale state (#1).
        std::array<uint8_t, 16> salt{};
        uint64_t h = 1469598103934665603ULL;
        auto fnv = [&h](const void *data, size_t n) {
            const uint8_t *b = (const uint8_t *)data;
            for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
        };
        for (const char *p = cfg->model_path; *p; ++p) fnv(p, 1);
        struct stat st;
        if (stat(cfg->model_path, &st) == 0) {
            uint64_t meta[2] = { (uint64_t)st.st_size, (uint64_t)st.st_mtime };
            fnv(meta, sizeof(meta));
        }
        for (int i = 0; i < 8; i++) salt[i] = (uint8_t)(h >> (i * 8));
        uint64_t h2 = (h ^ 0x9e3779b97f4a7c15ULL) * 1099511628211ULL;
        for (int i = 0; i < 8; i++) salt[8 + i] = (uint8_t)(h2 >> (i * 8));
        b->disk->set_identity_salt(salt);
        b->disk->init();
    }
    return b;
}

extern "C" void ember_backend_free(ember_backend *b) { delete b; }

extern "C" int ember_backend_encode(ember_backend *b, const char *text,
                                    int32_t **ids_out) {
    std::vector<int32_t> ids = b->tok.encode(text ? text : "");
    size_t n = ids.size();
    int32_t *out = (int32_t *)malloc((n ? n : 1) * sizeof(int32_t));
    if (n) memcpy(out, ids.data(), n * sizeof(int32_t));
    *ids_out = out;
    return (int)n;
}

extern "C" const char *ember_backend_token_text(ember_backend *b, int32_t id) {
    // token_text() returns the decoded UTF-8 bytes (byte-level BPE reversed).
    // Stash in a per-backend scratch for a stable pointer; safe because
    // generation is single-slot (the server serializes it) so on_token — the
    // only caller — never overlaps.
    b->tok_scratch = b->tok.token_text(id);
    return b->tok_scratch.c_str();
}

extern "C" ember_gen_result ember_backend_generate(ember_backend *b,
                                                   const ember_gen_request *req) {
    ember_gen_result r;
    memset(&r, 0, sizeof(r));
    strcpy(r.finish_reason, "stop");

    GenerateRequest greq;
    greq.prompt.assign(req->prompt, req->prompt + req->n_prompt);
    // main.c already resolves the budget (unset -> context room); a literal 0
    // means "generate nothing", so only a negative (shouldn't occur) falls back.
    greq.n_gen = req->max_tokens < 0 ? 2048 : req->max_tokens;
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
    // Match lucebox's own server: sample through the logit path when any modifier
    // is active (temp>0 / penalties / top_k / min_p), else greedy argmax (DSpark).
    greq.do_sample = greq.sampler.needs_logit_processing();
    greq.snap_slot = req->snap_slot;   // inline snapshot for future reuse
    greq.snap_pos  = req->snap_pos;

    // Level-2 thinking force-close: when the reply budget is reached, the
    // backend injects the close sequence (</think>...) so the model writes a
    // visible answer instead of thinking until EOS. Setting this routes
    // through AR decode (spec skipped) — matches lucebox's thinking path.
    if (req->budget_close_ids && req->n_budget_close > 0 && req->reply_budget > 0) {
        greq.budget_hook.close_token_ids.assign(
            req->budget_close_ids, req->budget_close_ids + req->n_budget_close);
        greq.budget_hook.hard_limit_remaining = req->reply_budget;
    }

    // Bridge the per-token callback (backend uses req.on_token via
    // with_token_callback). Return false cancels generation.
    ember_token_cb tok_cb = req->on_token;
    void *ud = req->ud;
    // #7: the backend cancels on its own DaemonIO copy (out_io = io.with_token_
    // _callback), so the bridge's `io.cancelled` never flips. Track cancellation
    // here in the callbacks the bridge owns.
    bool cancelled = false;
    if (tok_cb) {
        greq.on_token = [tok_cb, ud, &cancelled](int32_t t) -> bool {
            if (!tok_cb(t, ud)) { cancelled = true; return false; }
            return true;
        };
    }

    // B6: bridge the structural-greedy predicate (consulted per token by the AR loop).
    if (req->force_greedy) {
        bool (*fg)(void *) = req->force_greedy;
        void *fud = req->fg_ud;
        greq.force_greedy_next = [fg, fud]() -> bool { return fg(fud); };
    }

    DaemonIO io;
    io.stream_fd = -1;  // callback-driven, no pipe
    ember_keepalive_cb ka_cb = req->on_prefill;
    if (ka_cb) {
        io.on_prefill_keepalive = [ka_cb, ud, &cancelled]() -> bool {
            if (!ka_cb(ud)) { cancelled = true; return false; }
            return true;
        };
    }

    GenerateResult res = (req->restore_slot >= 0 && b->be->snapshot_used(req->restore_slot))
        ? b->be->restore_and_generate(req->restore_slot, greq, io)  // reuse prior KV
        : b->be->generate(greq, io);                                // fresh prefill
    r.ok = res.ok();
    r.cancelled = cancelled;  // #7: not io.cancelled (backend mutated its own copy)
    r.n_generated = (int)res.tokens.size();
    r.prefill_s = res.prefill_s;
    r.decode_s = res.decode_s;
    r.accept_rate = res.accept_rate;
    // Surface the backend's generation-quality + error signals (were dropped).
    r.budget_forced_close     = res.budget_forced_close;
    r.degenerate_decode_close = res.degenerate_decode_close;
    r.empty_visible_output    = res.empty_visible_output;
    r.spec_decode_ran         = res.spec_decode_ran;
    r.snapshot_saved          = res.snapshot_saved;  // #2: real-save signal
    if (!res.ok()) {
        std::string_view code = res.error_code();
        std::string_view detail = res.error_detail();
        snprintf(r.error_code, sizeof(r.error_code), "%.*s",
                 (int)code.size(), code.data());
        snprintf(r.error_detail, sizeof(r.error_detail), "%.*s",
                 (int)detail.size(), detail.data());
    }
    if (req->max_tokens > 0 && r.n_generated >= req->max_tokens) {
        strcpy(r.finish_reason, "length");
    }
    return r;
}

extern "C" bool ember_backend_snapshot_now(ember_backend *b, int slot) {
    // B3 Layer 2: snapshot the live cache_ at its current (post-generate) cur_pos.
    // DeepSeek4Backend::snapshot_save captures cache_ at cur_pos + returns true
    // on success; snapshot_used(slot) then reports it as restorable.
    if (!b || !b->be || slot < 0) return false;
    return b->be->snapshot_save(slot) && b->be->snapshot_used(slot);
}

extern "C" int ember_backend_n_ctx(const ember_backend *b) { return b->n_ctx; }
extern "C" const char *ember_backend_model_name(const ember_backend *b) {
    return b->model_name.c_str();
}
extern "C" int32_t ember_backend_eos_id(const ember_backend *b) {
    return b->tok.eos_id();
}

extern "C" bool ember_backend_disk_enabled(const ember_backend *b) {
    return b->disk != nullptr;
}
extern "C" int ember_backend_disk_prefix(ember_backend *b, const int32_t *p, int n) {
    return b->disk ? b->disk->longest_prefix_len(vec(p, n)) : 0;
}
extern "C" bool ember_backend_disk_lookup(ember_backend *b, const int32_t *p,
                                          int len, int slot) {
    return b->disk && b->disk->lookup(vec(p, len), slot);
}
extern "C" bool ember_backend_disk_save(ember_backend *b, int slot,
                                        const int32_t *p, int cut) {
    return b->disk && b->disk->save(slot, vec(p, cut));
}
