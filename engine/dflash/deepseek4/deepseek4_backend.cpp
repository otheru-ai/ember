// DeepSeek4Backend implementation — AR-only decode, chunked prefill.

#include "deepseek4_backend.h"
#include "deepseek4_internal.h"
// dspark_worker_note_target_eval: the AR loop feeds the speculative scheduler its
// genuine single-token baseline (see the header for why that matters).
#include "deepseek4_dspark_scheduler.h"
#include "common/dspark_head.h"
#include "common/sampler.h"

#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
#include "common/gpu_runtime_compat.h"
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <limits>

namespace dflash::common {

namespace {
using Clock = std::chrono::steady_clock;

int ds4_spec_emit_budget(const GenerateRequest & req);
int ds4_spec_context_budget(int committed);

static double elapsed_s(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

static uint64_t elapsed_us(Clock::time_point start, Clock::time_point end) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

static bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static long env_nonnegative_long(const char * name, long fallback) {
    const char * raw = std::getenv(name);
    if (!raw || !raw[0]) return fallback;
    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(raw, &end, 10);
    if (errno == 0 && end != raw && *end == '\0' && value >= 0) {
        return value;
    }
    std::fprintf(stderr, "[deepseek4] invalid %s='%s'; using %ld\n",
                 name, raw, fallback);
    return fallback;
}

// Exact prefill normally replays the target's q=1 graph for every prompt
// token.  The layer-range implementation also has a conservative q=2..4
// path: attention and compressor publication remain tokenwise, while HC and
// FFN work can reuse weights across the small batch.  Keep it opt-in until the
// real-model parity suite proves that its final logits match q=1 on gfx1151.
// Four is also the largest width for which production pins ROCMFP matmuls to
// MMVQ rather than changing to the wider MMQ reduction topology.
static int exact_prefill_chunk_limit() {
    static const int configured = []() {
        const long requested = env_nonnegative_long(
            "DFLASH_DS4_EXACT_PREFILL_CHUNK", 1);
        const int clamped = (int) std::max(1L, std::min(requested, 4L));
        if (requested != clamped) {
            std::fprintf(stderr,
                         "[deepseek4] clamped exact prefill chunk %ld to %d "
                         "(supported range 1..4)\n",
                         requested, clamped);
        }
        if (clamped > 1) {
            std::fprintf(stderr,
                         "[deepseek4] experimental exact prefill chunk=%d\n",
                         clamped);
        }
        return clamped;
    }();
    return configured;
}

// Keep an emergency/A-B escape hatch even though this optimization is
// mathematically exact: the intermediate LM head is stateless and its result
// is discarded. Set to 0 to restore the former every-token projection.
static bool exact_prefill_skip_intermediate_logits() {
    static const bool enabled = []() {
        const char * raw = std::getenv(
            "DFLASH_DS4_EXACT_PREFILL_SKIP_LOGITS");
        return !raw || !raw[0] || std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

static void maybe_log_prefill_fingerprint(
        const std::vector<float> & logits, int position) {
    if (!env_flag_enabled("DFLASH_DS4_LOG_PREFILL_FINGERPRINT") ||
        logits.empty()) {
        return;
    }
    // FNV-1a over the float bit patterns is intentionally sensitive to every
    // rounding bit. Matching generated text is not sufficient evidence for
    // the exact-prefill contract; matching this fingerprint is.
    uint64_t hash = UINT64_C(14695981039346656037);
    int top = -1;
    float top_value = -std::numeric_limits<float>::infinity();
    float runner_up = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < logits.size(); ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &logits[i], sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= (bits >> (8 * byte)) & 0xffu;
            hash *= UINT64_C(1099511628211);
        }
        if (logits[i] > top_value) {
            runner_up = top_value;
            top_value = logits[i];
            top = (int) i;
        } else if (logits[i] > runner_up) {
            runner_up = logits[i];
        }
    }
    std::fprintf(stderr,
                 "[deepseek4-prefill-fingerprint] pos=%d n=%zu "
                 "hash=%016" PRIx64 " top=%d value=%.9g margin=%.9g\n",
                 position, logits.size(), hash, top, top_value,
                 top_value - runner_up);
}

// Paired with the LUCE_MMVQ_MAX_NCOLS fallback in ggml-cuda.cu, which defaults
// to 3 (an sm_86 crossover) when this function does not run. Changing either
// without the other splits the effective default by code path.
static void configure_gfx1151_dspark_mmvq_default(int gpu) {
#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    if (!env_flag_enabled("DFLASH_DS4_SPEC") ||
        std::getenv("LUCE_MMVQ_MAX_NCOLS") != nullptr) {
        return;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, gpu) != cudaSuccess ||
        std::strncmp(prop.gcnArchName, "gfx1151", 7) != 0) {
        return;
    }

    // Tie the MMVQ ncols ceiling to the speculation width. Above the ceiling a
    // quantized mul_mat falls off MMVQ onto MMQ, and MMVQ is ~65% of decode
    // here, so a width the ceiling does not cover silently moves the dominant
    // kernel to the slower path. Measured on this box, same build, width 3.20:
    //     ncols=4 (width falls to MMQ) : 34.69 tok/s
    //     ncols=5 (width stays on MMVQ): 35.65 tok/s
    // 4 remains the default when no width is requested. The hard kernel limit is
    // MMVQ_MAX_BATCH_SIZE from ggml-cuda/mmvq.cuh, which is a private backend
    // header this TU cannot include, so it is mirrored here; keep the two in
    // step if ggml raises it.
    constexpr int kMmvqMaxBatchSize = 8;
    int ncols = DS4_CONSERVATIVE_VERIFY_MAX_TOKENS;
    // The wide path raises the verify width on its own, without needing an
    // explicit DFLASH_DS4_SPEC_Q, so key off the flag too. Keying only off
    // SPEC_Q would leave the ceiling at 4 while the width went to 6 -- the
    // exact mismatch that cost ~1 tok/s before it was diagnosed.
    if (env_flag_enabled("DFLASH_DS4_Q5_VERIFY")) {
        ncols = DS4_Q5_VERIFY_TOKENS;
    }
    if (const char * q = std::getenv("DFLASH_DS4_SPEC_Q")) {
        const int v = std::atoi(q);
        if (v > ncols) ncols = v;
    }
    if (ncols > kMmvqMaxBatchSize) ncols = kMmvqMaxBatchSize;
    char ncols_buf[16];
    std::snprintf(ncols_buf, sizeof(ncols_buf), "%d", ncols);
    if (::setenv("LUCE_MMVQ_MAX_NCOLS", ncols_buf, 0) == 0) {
        std::fprintf(stderr,
                     "[deepseek4] gfx1151 DSpark: defaulting "
                     "LUCE_MMVQ_MAX_NCOLS=%d\n", ncols);
    }
#else
    (void) gpu;
#endif
}

static double gib(uint64_t bytes) {
    return (double) bytes / 1024.0 / 1024.0 / 1024.0;
}

static void add_step_tel(DeepSeek4StepTelemetry & dst, const DeepSeek4StepTelemetry & src) {
    dst.total_us += src.total_us;
    dst.embed_us += src.embed_us;
    dst.hc_pre_attn_us += src.hc_pre_attn_us;
    dst.hc_pre_build_us += src.hc_pre_build_us;
    dst.hc_pre_input_us += src.hc_pre_input_us;
    dst.hc_pre_compute_us += src.hc_pre_compute_us;
    dst.attn_build_us += src.attn_build_us;
    dst.attn_compute_us += src.attn_compute_us;
    dst.attn_read_us += src.attn_read_us;
    dst.hc_post_attn_us += src.hc_post_attn_us;
    dst.hc_pre_ffn_us += src.hc_pre_ffn_us;
    dst.ffn_build_us += src.ffn_build_us;
    dst.ffn_compute_us += src.ffn_compute_us;
    dst.ffn_read_us += src.ffn_read_us;
    dst.route_build_us += src.route_build_us;
    dst.route_compute_us += src.route_compute_us;
    dst.route_read_us += src.route_read_us;
    dst.route_select_us += src.route_select_us;
    dst.ffn_eval_us += src.ffn_eval_us;
    dst.ffn_hot_us += src.ffn_hot_us;
    dst.ffn_cold_us += src.ffn_cold_us;
    dst.ffn_combine_us += src.ffn_combine_us;
    dst.ffn_partition_us += src.ffn_partition_us;
    dst.ffn_hot_graph_builds += src.ffn_hot_graph_builds;
    dst.ffn_hot_graph_hits += src.ffn_hot_graph_hits;
    dst.ffn_cold_graph_builds += src.ffn_cold_graph_builds;
    dst.ffn_cold_graph_hits += src.ffn_cold_graph_hits;
    dst.hc_post_ffn_us += src.hc_post_ffn_us;
    dst.output_us += src.output_us;
    dst.sample_us += src.sample_us;
    dst.emit_us += src.emit_us;
    dst.full_graph_build_us += src.full_graph_build_us;
    dst.full_graph_alloc_us += src.full_graph_alloc_us;
    dst.full_graph_set_us += src.full_graph_set_us;
    dst.full_graph_compute_us += src.full_graph_compute_us;
    dst.full_graph_read_us += src.full_graph_read_us;
    dst.fused_verify_compute_us += src.fused_verify_compute_us;
    dst.fused_verify_calls += src.fused_verify_calls;
    dst.fused_verify_rows += src.fused_verify_rows;
    for (int q = 0; q < 5; ++q) {
        dst.fused_verify_q_compute_us[q] += src.fused_verify_q_compute_us[q];
        dst.fused_verify_q_calls[q] += src.fused_verify_q_calls[q];
    }
    dst.hot_selected += src.hot_selected;
    dst.cold_selected += src.cold_selected;
}

static double ms(uint64_t us) {
    return (double)us / 1000.0;
}

static void log_step_tel(const char * phase,
                         int tokens,
                         int steps,
                         double wall_s,
                         const DeepSeek4StepTelemetry & t) {
    const double tok_s = wall_s > 0.0 ? (double)tokens / wall_s : 0.0;
    std::fprintf(stderr,
        "[deepseek4-timing] %s tokens=%d steps=%d wall=%.3fs %.2f tok/s "
        "step=%.1fms embed=%.1fms attn_build=%.1fms attn_compute=%.1fms attn_read=%.1fms "
        "ffn_build=%.1fms ffn_compute=%.1fms ffn_read=%.1fms "
        "route_build=%.1fms route_compute=%.1fms route_read=%.1fms route_select=%.1fms "
        "ffn=%.1fms hot=%.1fms cold=%.1fms combine=%.1fms partition=%.1fms "
        "ffn_hot_graph_build=%llu ffn_hot_graph_hit=%llu ffn_cold_graph_build=%llu ffn_cold_graph_hit=%llu "
        "hc_pre=%.1fms hc_pre_build=%.1fms hc_pre_input=%.1fms hc_pre_compute=%.1fms "
        "hc_post=%.1fms output=%.1fms sample=%.1fms emit=%.1fms "
        "graph_build=%.1fms graph_alloc=%.1fms graph_set=%.1fms "
        "graph_compute=%.1fms graph_read=%.1fms "
        "hot_sel=%d cold_sel=%d\n",
        phase, tokens, steps, wall_s, tok_s,
        ms(t.total_us), ms(t.embed_us), ms(t.attn_build_us), ms(t.attn_compute_us), ms(t.attn_read_us),
        ms(t.ffn_build_us), ms(t.ffn_compute_us), ms(t.ffn_read_us),
        ms(t.route_build_us), ms(t.route_compute_us), ms(t.route_read_us), ms(t.route_select_us),
        ms(t.ffn_eval_us), ms(t.ffn_hot_us), ms(t.ffn_cold_us), ms(t.ffn_combine_us),
        ms(t.ffn_partition_us),
        (unsigned long long)t.ffn_hot_graph_builds, (unsigned long long)t.ffn_hot_graph_hits,
        (unsigned long long)t.ffn_cold_graph_builds, (unsigned long long)t.ffn_cold_graph_hits,
        ms(t.hc_pre_attn_us + t.hc_pre_ffn_us),
        ms(t.hc_pre_build_us),
        ms(t.hc_pre_input_us),
        ms(t.hc_pre_compute_us),
        ms(t.hc_post_attn_us + t.hc_post_ffn_us),
        ms(t.output_us), ms(t.sample_us), ms(t.emit_us),
        ms(t.full_graph_build_us), ms(t.full_graph_alloc_us),
        ms(t.full_graph_set_us), ms(t.full_graph_compute_us),
        ms(t.full_graph_read_us),
        t.hot_selected, t.cold_selected);
}

static uint64_t layer_expert_bytes(const DeepSeek4Layer & layer, int n_expert) {
    if (n_expert <= 0) return 0;
    uint64_t bytes = 0;
    if (layer.ffn_gate_exps) bytes += ggml_nbytes(layer.ffn_gate_exps) / (uint64_t) n_expert;
    if (layer.ffn_up_exps) bytes += ggml_nbytes(layer.ffn_up_exps) / (uint64_t) n_expert;
    if (layer.ffn_down_exps) bytes += ggml_nbytes(layer.ffn_down_exps) / (uint64_t) n_expert;
    return bytes;
}

struct Ds4ExpertMemoryInfo {
    std::vector<uint64_t> layer_expert_bytes;
    uint64_t total_expert_bytes = 0;
    uint64_t bytes_per_uniform_round = 0;
    uint64_t hot_bytes = 0;
    uint64_t cold_bytes = 0;
    int total_hot = 0;
    int total_cold = 0;
};

struct Ds4HybridBudgetInfo {
    Ds4ExpertMemoryInfo mem;
    size_t gpu_free = 0;
    size_t gpu_total = 0;
    uint64_t core_bytes = 0;
    uint64_t kv_bytes = 0;
    uint64_t warm_bytes = 256ULL * 1024 * 1024;
    uint64_t safety_bytes = 512ULL * 1024 * 1024;
    uint64_t expert_budget = 0;
    int max_hot_per_layer = 0;
};

static bool compute_ds4_expert_memory_info(const DeepSeek4Weights & w,
                                           const MoeHybridPlacement * placement,
                                           Ds4ExpertMemoryInfo & out,
                                           std::string * err) {
    out = {};
    out.layer_expert_bytes.assign((size_t) w.n_layer, 0);
    for (int il = 0; il < w.n_layer; ++il) {
        const uint64_t bytes = layer_expert_bytes(w.layers[(size_t) il], w.n_expert);
        out.layer_expert_bytes[(size_t) il] = bytes;
        out.total_expert_bytes += bytes * (uint64_t) w.n_expert;
        out.bytes_per_uniform_round += bytes;
    }
    if (out.bytes_per_uniform_round == 0) {
        if (err) *err = "expert tensor metadata missing after partial load";
        return false;
    }
    if (!placement) return true;
    if (!placement->matches(w.n_layer, w.n_expert, w.n_expert_used)) {
        if (err) *err = "placement does not match DS4 dimensions";
        return false;
    }
    out.total_hot = placement->total_hot;
    out.total_cold = w.n_layer * w.n_expert - placement->total_hot;
    for (int il = 0; il < w.n_layer; ++il) {
        const uint64_t layer_bytes = out.layer_expert_bytes[(size_t) il];
        const uint64_t hot_count = (uint64_t) placement->hot_counts[(size_t) il];
        out.hot_bytes += layer_bytes * hot_count;
        out.cold_bytes += layer_bytes * ((uint64_t) w.n_expert - hot_count);
    }
    return true;
}

static void log_ds4_expert_memory_info(const char * tag,
                                       const Ds4ExpertMemoryInfo & info,
                                       int n_layer) {
    (void) n_layer;
    std::fprintf(stderr,
                 "[deepseek4] %s expert_memory: total=%.2f GiB uniform_round=%.2f MiB hot=%d %.2f GiB cold=%d %.2f GiB\n",
                 tag,
                 gib(info.total_expert_bytes),
                 (double) info.bytes_per_uniform_round / 1024.0 / 1024.0,
                 info.total_hot,
                 gib(info.hot_bytes),
                 info.total_cold,
                 gib(info.cold_bytes));
}

static uint64_t estimate_ds4_cache_bytes(const DeepSeek4Weights & w, int max_ctx) {
    size_t total_bytes = 0;
    const size_t head_dim = (size_t) w.head_dim;
    const size_t swa_size = (size_t) w.n_swa;

    for (int il = 0; il < w.n_layer; ++il) {
        total_bytes += swa_size * head_dim * sizeof(uint16_t);
        const uint32_t ratio = w.compress_ratios[(size_t) il];
        if (ratio == 0) continue;

        const size_t comp_cap = (size_t) (max_ctx / (int) ratio) + 16;
        total_bytes += comp_cap * head_dim * sizeof(uint16_t);

        const size_t window = (ratio == 4) ? 8 : ratio;
        total_bytes += window * head_dim * sizeof(float) * 2;

        if (ratio == 4) {
            const size_t index_comp_width = (size_t) w.n_indexer_head * (size_t) w.n_indexer_head_dim;
            total_bytes += comp_cap * index_comp_width * sizeof(uint16_t);
            total_bytes += window * index_comp_width * sizeof(float) * 2;
        }
    }

    total_bytes += (size_t) w.n_hc * (size_t) w.n_embd * sizeof(float);
    return total_bytes;
}

static void fill_prefix_hot_placement(const DeepSeek4Weights & w,
                                      int hot_per_layer,
                                      MoeHybridPlacement & out) {
    out = {};
    out.n_layer = w.n_layer;
    out.n_expert = w.n_expert;
    out.n_expert_used = w.n_expert_used;
    out.hot_counts.assign((size_t) w.n_layer, hot_per_layer);
    out.hot_expert_ids.resize((size_t) w.n_layer);
    out.total_hot = hot_per_layer * w.n_layer;
    for (int il = 0; il < w.n_layer; ++il) {
        auto & ids = out.hot_expert_ids[(size_t) il];
        ids.reserve((size_t) hot_per_layer);
        for (int ie = 0; ie < hot_per_layer; ++ie) {
            ids.push_back((int32_t) ie);
        }
    }
}

static bool compute_ds4_hybrid_budget_info(const DeepSeek4Weights & w,
                                           int gpu,
                                           int max_ctx,
                                           Ds4HybridBudgetInfo & out,
                                           std::string * err) {
    out = {};
    ggml_backend_cuda_get_device_memory(gpu, &out.gpu_free, &out.gpu_total);
    if (out.gpu_total == 0) {
        if (err) *err = "could not query GPU memory";
        return false;
    }

    if (!compute_ds4_expert_memory_info(w, nullptr, out.mem, err)) {
        return false;
    }

    if (out.gpu_free > out.gpu_total ||
        out.mem.bytes_per_uniform_round == 0) {
        if (err) *err = "invalid GPU memory or expert-size accounting";
        return false;
    }
    out.core_bytes = out.gpu_total - out.gpu_free;
    out.kv_bytes = estimate_ds4_cache_bytes(w, max_ctx);

    uint64_t reserved = out.core_bytes;
    const uint64_t additions[] = {
        out.kv_bytes, out.warm_bytes, out.safety_bytes
    };
    bool reserved_overflow = false;
    for (uint64_t amount : additions) {
        if (amount > UINT64_MAX - reserved) {
            reserved_overflow = true;
            break;
        }
        reserved += amount;
    }
    if (!reserved_overflow && out.gpu_total > reserved) {
        out.expert_budget = out.gpu_total - reserved;
    }
    if (out.expert_budget > out.mem.total_expert_bytes) {
        out.expert_budget = out.mem.total_expert_bytes;
    }
    if (std::getenv("DFLASH_EXPERT_BUDGET_MB") != nullptr) {
        const long cap_mb =
            env_nonnegative_long("DFLASH_EXPERT_BUDGET_MB", 0);
        const uint64_t cap_bytes =
            (uint64_t)cap_mb > UINT64_MAX / (1024ULL * 1024ULL)
                ? UINT64_MAX
                : (uint64_t)cap_mb * 1024ULL * 1024ULL;
        if (cap_bytes > 0 && cap_bytes < out.expert_budget) {
            out.expert_budget = cap_bytes;
        }
    }
    if (out.expert_budget == 0) {
        if (err) *err = "no VRAM budget available for DS4 experts";
        return false;
    }

    out.max_hot_per_layer = std::min(w.n_expert, (int) (out.expert_budget / out.mem.bytes_per_uniform_round));
    if (out.max_hot_per_layer <= 0) {
        if (err) *err = "expert budget is smaller than one uniform expert round";
        return false;
    }
    return true;
}

static MoeHybridConfig make_ds4_parent_worker_cfg(const DeepSeek4Weights & w) {
    MoeHybridConfig cfg;
    cfg.n_embd = w.n_embd;
    cfg.n_expert = w.n_expert;
    cfg.n_expert_used = w.n_expert_used;
    cfg.n_ff_exp = w.n_ff_exp;
    cfg.n_ff_shexp = w.n_ff_exp;
    cfg.n_layer = w.n_layer;
    cfg.first_moe_layer = 0;
    cfg.swiglu_clamp = w.swiglu_clamp_exp;
    cfg.materialize_cold_experts = false;
    return cfg;
}

}  // namespace

DeepSeek4Backend::DeepSeek4Backend(const DeepSeek4BackendConfig & cfg)
    : cfg_(cfg) {}

DeepSeek4Backend::~DeepSeek4Backend() {
    shutdown();
}

struct DeepSeek4Backend::ResidentSession {
    GenerateRequest request;
    DaemonIO io;
    DeepSeek4Cache cache;
    SamplerCfg sampler;
    std::mt19937_64 sampler_rng{std::random_device{}()};
    std::vector<float> last_logits;
    std::vector<float> spec_feat_window;
    DeepSeek4DSparkResidentProposal spec_proposal;
    std::vector<int32_t> history;
    std::vector<int32_t> generated;
    int prefilled = 0;
    int restore_slot = -1;
    int restore_pos = 0;
    int prefill_tokens = 0;
    int32_t pending_token = -1;
    bool pending_ready = false;
    bool pending_forced_close_token = false;
    bool terminal = false;
    bool cancelled = false;
    bool failed = false;
    bool spec_eligible = false;
    bool spec_ran = false;
    bool budget_forced_close = false;
    bool degenerate_decode_close = false;
    std::string termination_reason;
    ThinkingBudgetState thinking_budget;
    ProgressCycleDetector progress_cycle;
    bool inline_snapshot_saved = false;
    size_t forced_close_index = 0;
    double prefill_s = 0.0;
    double decode_s = 0.0;
    long spec_offered = 0;
    long spec_accepted = 0;
    int spec_cycles = 0;
    double spec_provider_age_s = 0.0;
    double spec_provider_block_s = 0.0;
    double spec_head_s = 0.0;
    double spec_verify_s = 0.0;
    std::string error;
};

void DeepSeek4Backend::swap_resident_state(ResidentSession &session) {
    using std::swap;
    swap(cache_, session.cache);
    swap(sampler_, session.sampler);
    swap(sampler_rng_, session.sampler_rng);
    swap(last_logits_, session.last_logits);
    swap(spec_feat_window_, session.spec_feat_window);
    swap(inline_snapshot_saved_, session.inline_snapshot_saved);
}

bool DeepSeek4Backend::rebuild_resident_spec_features(
        ResidentSession &session, std::string *error) {
    if (error) error->clear();
    if (!session.spec_eligible || !spec_drafter_ ||
        session.request.force_exact_prefill ||
        !prefill_attention_mode_is_approximate(cfg_.prefill_mode)) {
        return true;
    }

    const int exact_rows = (int)std::min<long>(
        w_.n_swa,
        env_nonnegative_long("DFLASH_DS4_SPEC_SHADOW_SUFFIX_ROWS", 4));
    if (exact_rows <= 0) return true;

    const int prompt_size = (int)session.request.prompt.size();
    const int feat_row = spec_drafter_->n_target_layers * w_.n_embd;
    if (prompt_size <= 0 || feat_row <= 0) {
        if (error) *error = "invalid resident DSpark shadow-capture shape";
        return false;
    }

    auto reset_shadow = [&]() {
        reset_deepseek4_cache(cache_);
        cache_.prefill_mode = cfg_.prefill_mode;
    };
    reset_shadow();

    int pos = 0;
    std::vector<float> rebuilt;
    if (session.restore_slot >= 0 && session.restore_pos > 0) {
        if (!snapshot_used(session.restore_slot) ||
            !deepseek4_snapshot_restore(
                snapshots_[session.restore_slot], cache_)) {
            reset_shadow();
            if (error) *error =
                "resident DSpark shadow snapshot restore failed";
            return false;
        }
        pos = session.restore_pos;
        rebuilt = snapshot_spec_features_[session.restore_slot];
        if (cache_.cur_pos != pos ||
            rebuilt.size() % (size_t)feat_row != 0 ||
            rebuilt.size() / (size_t)feat_row > (size_t)w_.n_swa) {
            reset_shadow();
            if (error) *error =
                "resident DSpark shadow snapshot metadata mismatch";
            return false;
        }
    }
    if (pos < 0 || pos > prompt_size) {
        reset_shadow();
        if (error) *error = "resident DSpark shadow restore position mismatch";
        return false;
    }

    const int exact_from = std::max(pos, prompt_size - exact_rows);
    const int shadow_chunk = std::max(
        1, std::min({w_.n_swa, DS4_MAX_LAYER_MAJOR_PREFILL_TOKENS,
                     cfg_.chunk > 0 ? cfg_.chunk : w_.n_swa}));
    std::vector<float> embed;
    std::vector<float> hc_state;
    std::vector<float> logits;

    // Reconstruct the state preceding the exact suffix in a spare cache. This
    // intentionally mirrors the old hybrid capture experiment, but never
    // touches the authoritative resident cache or its sparse-prefill logits.
    while (pos < exact_from) {
        const int count = std::min(shadow_chunk, exact_from - pos);
        embed.resize((size_t)w_.n_embd * (size_t)count);
        if (!w_.embedder.embed(session.request.prompt.data() + pos, count,
                               embed.data())) {
            reset_shadow();
            if (error) *error =
                "resident DSpark shadow prefix embedding failed";
            return false;
        }
        cache_.prefill_mode = cfg_.prefill_mode;
        logits.clear();
        if (!deepseek4_step_layer_range(
                backend_, cfg_.device.gpu, w_, cache_, hc_state,
                embed.data(), count, pos, 0, w_.n_layer, &logits,
                session.request.prompt.data() + pos, nullptr,
                /*allow_decode_graph_reuse=*/false, nullptr)) {
            reset_shadow();
            if (error) *error =
                "resident DSpark shadow prefix replay failed";
            return false;
        }
        pos += count;
    }

    cache_.prefill_mode = PrefillAttentionMode::Exact;
    for (; pos < prompt_size; ++pos) {
        embed.resize((size_t)w_.n_embd);
        if (!w_.embedder.embed(session.request.prompt.data() + pos, 1,
                               embed.data())) {
            reset_shadow();
            if (error) *error =
                "resident DSpark shadow suffix embedding failed";
            return false;
        }
        std::vector<float> captured;
        Ds4VerifyHooks hooks;
        hooks.capture_layer_ids = &spec_drafter_->capture_layer_ids;
        hooks.capture_out = &captured;
        hooks.require_fused_q1 = true;
        logits.clear();
        if (!deepseek4_step_layer_range(
                backend_, cfg_.device.gpu, w_, cache_, hc_state,
                embed.data(), 1, pos, 0, w_.n_layer, &logits,
                session.request.prompt.data() + pos, nullptr,
                /*allow_decode_graph_reuse=*/true, &hooks) ||
            captured.size() != (size_t)feat_row) {
            reset_shadow();
            if (error) *error =
                "resident DSpark exact shadow suffix failed";
            return false;
        }
        const size_t rows = rebuilt.size() / (size_t)feat_row;
        if (rows >= (size_t)w_.n_swa) {
            std::memmove(rebuilt.data(), rebuilt.data() + feat_row,
                         (rebuilt.size() - (size_t)feat_row) * sizeof(float));
            rebuilt.resize(rebuilt.size() - (size_t)feat_row);
        }
        rebuilt.insert(rebuilt.end(), captured.begin(), captured.end());
    }

    session.spec_feat_window = std::move(rebuilt);
    reset_shadow();
    return true;
}

void DeepSeek4Backend::free_resident_sessions() {
    for (auto &entry : resident_sessions_) {
        if (entry.second) free_deepseek4_cache(entry.second->cache);
    }
    resident_sessions_.clear();
    for (DeepSeek4Cache &cache : resident_cache_pool_) {
        free_deepseek4_cache(cache);
    }
    resident_cache_pool_.clear();
}

void DeepSeek4Backend::recycle_resident_cache(
        DeepSeek4Cache &cache) noexcept {
    if (!cache.ctx && !cache.buf) return;
    reset_deepseek4_cache(cache);
    try {
        resident_cache_pool_.push_back(std::move(cache));
        cache = {};
    } catch (...) {
        free_deepseek4_cache(cache);
    }
}

bool DeepSeek4Backend::resident_sample_next(ResidentSession &session) {
    if (session.terminal || session.failed || session.cancelled ||
        session.pending_ready) {
        return !session.failed;
    }
    if ((int)session.generated.size() >= session.request.n_gen) {
        session.terminal = true;
        return true;
    }

    const BudgetHook &hook = session.request.budget_hook;
    session.thinking_budget.observe_latest(
        session.generated, hook.natural_close_token_ids);
    if (!session.budget_forced_close &&
        session.thinking_budget.should_force_close(
            session.request.n_gen, session.generated.size(),
            hook.hard_limit_remaining, !hook.close_token_ids.empty())) {
        session.budget_forced_close = true;
        session.thinking_budget.mark_closed();
        session.forced_close_index = 0;
    }

    int32_t next = -1;
    session.pending_forced_close_token = false;
    if (session.budget_forced_close &&
        session.forced_close_index < hook.close_token_ids.size()) {
        next = hook.close_token_ids[session.forced_close_index++];
        session.pending_forced_close_token = true;
    } else {
        if (session.last_logits.size() != (size_t)w_.n_vocab) {
            session.failed = true;
            session.error = "resident decode has no valid logits frontier";
            return false;
        }
        const bool force_greedy =
            session.request.force_greedy_next &&
            session.request.force_greedy_next();
        if (session.sampler.needs_logit_processing() && !force_greedy) {
            next = sample_logits(session.last_logits.data(), w_.n_vocab,
                                 session.sampler, session.history,
                                 session.sampler_rng);
        } else {
            next = 0;
            float max_value = session.last_logits[0];
            for (int i = 1; i < w_.n_vocab; ++i) {
                if (session.last_logits[(size_t)i] > max_value) {
                    max_value = session.last_logits[(size_t)i];
                    next = i;
                }
            }
        }
    }
    session.pending_token = next;
    session.pending_ready = true;
    return true;
}

int DeepSeek4Backend::resident_spec_commit_cap(
        const ResidentSession &session) const {
    if (!session.spec_eligible || !spec_enabled_ || !spec_drafter_ ||
        !spec_xdna_draft_compute_ || !spec_xdna_draft_compute_->healthy() ||
        !session.pending_ready || session.pending_forced_close_token ||
        session.pending_token < 0 || session.terminal || session.failed ||
        session.cancelled || deepseek4_is_eos_tok(session.pending_token, w_)) {
        return 0;
    }
    const int request_left =
        session.request.n_gen - (int)session.generated.size();
    const int speculative_left =
        ds4_spec_emit_budget(session.request) - (int)session.generated.size();
    const int context_left = ds4_spec_context_budget(session.cache.cur_pos);
    // Match the ordinary DSpark scheduler: never let one q-wide verification
    // cross a ratio-4 compressor boundary. Crossing changes the target
    // reduction topology and can flip a near-tied first argmax even when the
    // draft candidate matches ordinary q=1 autoregressive decode.
    const int boundary_left = 4 - (session.cache.cur_pos & 3);
    const int cap = std::min(
        {4, boundary_left, request_left, speculative_left, context_left});
    if (cap < 2) return 0;
    return cap;
}

bool DeepSeek4Backend::resident_submit_spec(
        ResidentSession &session, std::string *error) {
    if (error) error->clear();
    if (session.spec_proposal.pending()) return true;
    const int cap = resident_spec_commit_cap(session);
    if (cap < 2) return false;
    if (!deepseek4_dspark_resident_prepare(
            backend_, w_, *spec_drafter_, session.spec_feat_window,
            session.cache.cur_pos, session.pending_token, cap,
            *spec_xdna_draft_compute_, session.spec_proposal, error)) {
        return false;
    }
    return true;
}

bool DeepSeek4Backend::requires_monolithic_model() const {
    // An explicit expert cap is an operator request for hybrid placement.  In
    // particular, the XDNA provider needs cold experts even on Strix Halo,
    // where the full model otherwise fits in unified memory.  Exact prefill
    // supports the layered hybrid path; load_model() will disable fused decode
    // after that placement is built.  Approximate prefill still requires every
    // expert resident and therefore retains monolithic precedence.
    const bool explicit_hybrid_budget =
        env_nonnegative_long("DFLASH_EXPERT_BUDGET_MB", 0) > 0;
    if (explicit_hybrid_budget &&
        !prefill_attention_mode_is_approximate(cfg_.prefill_mode)) {
        return false;
    }
    return cfg_.fused_decode ||
           prefill_attention_mode_is_approximate(cfg_.prefill_mode);
}

bool DeepSeek4Backend::validate_prefill_mode() const {
    if (cfg_.prefill_mode == PrefillAttentionMode::Exact) {
        return true;
    }
    if (w_.moe_hybrid || moe_hybrid_) {
        std::fprintf(stderr,
            "[deepseek4] %s prefill requires every expert to be resident; "
            "the selected placement has cold experts\n",
            prefill_attention_mode_name(cfg_.prefill_mode));
        return false;
    }
    return true;
}

bool DeepSeek4Backend::load_model() {
    // Fused decode and layer-major prefill reference every expert directly.
    // Make their residency requirement explicit instead of silently falling
    // back to tokenwise hybrid execution.
    if (requires_monolithic_model()) {
        std::fprintf(stderr,
                     "[deepseek4] monolithic execution requested "
                     "(fused_decode=%s, prefill=%s)\n",
                     cfg_.fused_decode ? "on" : "off",
                     prefill_attention_mode_name(cfg_.prefill_mode));
        if (!load_deepseek4_gguf(cfg_.model_path, backend_, w_)) {
            if (prefill_attention_mode_is_approximate(cfg_.prefill_mode)) {
                std::fprintf(stderr,
                    "[deepseek4] monolithic HIP load required for %s prefill: %s\n",
                    prefill_attention_mode_name(cfg_.prefill_mode),
                    dflash27b_last_error());
                return false;
            }
            std::fprintf(stderr,
                         "[deepseek4] monolithic HIP load failed; trying hybrid mode\n");
            if (!init_hybrid_model()) {
                std::fprintf(stderr, "[deepseek4] hybrid mode also failed: %s\n",
                             cfg_.model_path);
                return false;
            }
        }
    } else {
        std::fprintf(stderr,
                     "[deepseek4] HIP target detected; using hybrid expert load path\n");
        if (!init_hybrid_model()) {
            std::fprintf(stderr, "[deepseek4] hybrid mode failed: %s\n", cfg_.model_path);
            return false;
        }
    }

    if (cfg_.expert_top_k < 0 || cfg_.expert_top_k > w_.n_expert_used) {
        std::fprintf(stderr,
                     "[deepseek4] expert top-k must be in [0,%d], got %d\n",
                     w_.n_expert_used, cfg_.expert_top_k);
        return false;
    }
    w_.routed_expert_top_k = cfg_.expert_top_k;
    w_.fused_decode = cfg_.fused_decode && !moe_hybrid_;
    // PORTED from lucebox d03bcc4. Off by default: upstream documents that it
    // changes verifier floating-point inputs and can change generated tokens,
    // which is exactly the class of change Ember ships opt-in. Requires the
    // monolithic single-device HIP path, same precondition as upstream.
    w_.fused_verify_f16_kv =
        env_flag_enabled("DFLASH_DS4_FUSED_VERIFY_F16_KV") && !moe_hybrid_;
    if (cfg_.fused_decode && moe_hybrid_) {
        std::fprintf(stderr,
                     "[deepseek4] fused decode unavailable with hybrid expert placement; "
                     "using layered decode\n");
    }
    return true;
}

bool DeepSeek4Backend::load_spec_drafter() {
    if (spec_draft_path_.empty()) return true;
    if (moe_hybrid_) {
        std::fprintf(stderr,
                     "[deepseek4] cannot load DSpark drafter without a resident "
                     "monolithic target\n");
        return false;
    }

    auto drafter = std::make_unique<DSparkDrafter>();
    if (!load_deepseek4_dspark_drafter(spec_draft_path_, backend_, *drafter)) {
        std::fprintf(stderr, "[deepseek4] DSpark drafter load FAILED: %s\n",
                     deepseek4_dspark_last_error());
        return false;
    }

    const DSparkDrafter & d = *drafter;
    bool compatible = d.core.n_embd == w_.n_embd &&
                      d.core.n_vocab == w_.n_vocab &&
                      d.vocab_size == w_.n_vocab &&
                      d.mask_token_id >= 0 && d.mask_token_id < w_.n_vocab &&
                      (int) d.capture_layer_ids.size() == d.n_target_layers;
    for (int layer : d.capture_layer_ids) {
        compatible = compatible && layer >= 0 && layer < w_.n_layer;
    }
    if (!compatible) {
        std::fprintf(stderr,
                     "[deepseek4] DSpark drafter is incompatible with target "
                     "(target embd/vocab/layers=%d/%d/%d, draft=%d/%d)\n",
                     w_.n_embd, w_.n_vocab, w_.n_layer,
                     d.core.n_embd, d.vocab_size);
        free_deepseek4_dspark_drafter(*drafter);
        return false;
    }

    spec_drafter_ = std::move(drafter);
    const char * xdna_plugin = std::getenv("DFLASH_DSPARK_XDNA_PLUGIN");
    const bool xdna_required =
        env_flag_enabled("DFLASH_DSPARK_XDNA_REQUIRED");
    if (xdna_required && (!xdna_plugin || !*xdna_plugin)) {
        std::fprintf(stderr,
                     "[xdna-dspark] DFLASH_DSPARK_XDNA_REQUIRED needs "
                     "DFLASH_DSPARK_XDNA_PLUGIN\n");
        free_deepseek4_dspark_drafter(*spec_drafter_);
        spec_drafter_.reset();
        return false;
    }
    if (xdna_plugin && *xdna_plugin) {
        // The provider is created synchronously and the drafter outlives it,
        // so immutable tensor descriptors can point at engine-owned managed
        // UMA instead of loading an 11-GiB second copy of the draft model.
        ggml_backend_synchronize(backend_);
        spec_xdna_weight_views_.clear();
        for (ggml_tensor * tensor = ggml_get_first_tensor(d.core.ctx);
             tensor != nullptr;
             tensor = ggml_get_next_tensor(d.core.ctx, tensor)) {
            if (!tensor->data || !tensor->name[0]) continue;
            ember_xdna_dspark_tensor_view_v1 view{};
            view.abi_version = EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION;
            view.struct_size = sizeof(view);
            view.name = tensor->name;
            view.data = tensor->data;
            view.bytes = ggml_nbytes(tensor);
            view.type = static_cast<int32_t>(tensor->type);
            view.n_dims = GGML_MAX_DIMS;
            for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
                view.dims[dimension] = tensor->ne[dimension];
                view.strides[dimension] = tensor->nb[dimension];
            }
            spec_xdna_weight_views_.push_back(view);
        }
        XdnaDSparkDraftConfig config;
        config.plugin_path = xdna_plugin;
        config.draft_model_path = spec_draft_path_;
        config.n_embd = d.core.n_embd;
        config.n_target_layers = d.n_target_layers;
        config.block_size = d.block_size;
        config.n_swa = w_.n_swa;
        config.head_dim = d.core.head_dim;
        config.weights_cpu_accessible =
            ggml_backend_buffer_is_host(d.core.buf) ||
            ggml_backend_cuda_buffer_is_managed(d.core.buf);
        config.weight_views = spec_xdna_weight_views_.data();
        config.weight_view_count = static_cast<uint32_t>(
            spec_xdna_weight_views_.size());
        config.required = xdna_required;
        std::string error;
        spec_xdna_draft_compute_ =
            make_xdna_dspark_draft_compute(config, &error);
        if (!spec_xdna_draft_compute_) {
            std::fprintf(stderr, "[xdna-dspark] %s%s\n", error.c_str(),
                         config.required ? " (required)" :
                                           "; using GPU drafter");
            if (config.required) {
                free_deepseek4_dspark_drafter(*spec_drafter_);
                spec_drafter_.reset();
                return false;
            }
        } else {
            std::fprintf(stderr, "[xdna-dspark] provider ready: %s%s\n",
                         spec_xdna_draft_compute_->name(),
                         config.required ? " required" : "");
            if (env_flag_enabled("DFLASH_DSPARK_XDNA_GPU_MAIN")) {
                std::fprintf(stderr,
                             "[xdna-dspark] placement: GPU main projection, "
                             "XDNA draft-layer body\n");
            }
        }
    }
    spec_enabled_ = true;
    std::fprintf(stderr, "[deepseek4] DSpark spec-decode ENABLED (drafter=%s)\n",
                 spec_draft_path_.c_str());
    return true;
}

void DeepSeek4Backend::release_spec_drafter() {
    spec_xdna_draft_compute_.reset();
    spec_xdna_weight_views_.clear();
    if (spec_drafter_) {
        free_deepseek4_dspark_drafter(*spec_drafter_);
    }
    spec_drafter_.reset();
    spec_enabled_ = false;
    spec_feat_window_.clear();
}

bool DeepSeek4Backend::init() {
    // The shared MMVQ/MMQ crossover defaults to q=3 for NVIDIA. On gfx1151,
    // DSpark q=4 is faster through MMVQ. Keep AR and other devices unchanged,
    // and preserve LUCE_MMVQ_MAX_NCOLS as an explicit override.
    configure_gfx1151_dspark_mmvq_default(cfg_.device.gpu);

    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[deepseek4] failed to create CUDA backend (gpu=%d)\n",
                     cfg_.device.gpu);
        return false;
    }

    snap_backend_ = ggml_backend_init_by_name("cpu", nullptr);

    if (!load_model()) {
        return false;
    }
    if (!validate_prefill_mode()) {
        return false;
    }
    if (prefill_attention_mode_is_approximate(cfg_.prefill_mode)) {
        std::fprintf(stderr,
            "[deepseek4] warning: %s prefill is approximate and may change "
            "generated tokens; use --ds4-prefill exact for reference output\n",
            prefill_attention_mode_name(cfg_.prefill_mode));
    }

    const int max_ctx = cfg_.max_ctx > 0 ? cfg_.max_ctx : 8192;
    if (!create_deepseek4_cache(backend_, w_, max_ctx, cache_)) {
        std::fprintf(stderr, "[deepseek4] failed to allocate KV cache (ctx=%d)\n", max_ctx);
        return false;
    }
    cache_.prefill_mode = cfg_.prefill_mode;

    const int active_experts =
        w_.routed_expert_top_k > 0 ? w_.routed_expert_top_k : w_.n_expert_used;
    std::fprintf(stderr,
                 "[deepseek4] initialized: %d layers, ctx=%d, %d experts "
                 "(%d/%d routed), fused_decode=%s, prefill=%s%s\n",
                 w_.n_layer, max_ctx, w_.n_expert, active_experts, w_.n_expert_used,
                 w_.fused_decode ? "on" : "off",
                 prefill_attention_mode_name(cfg_.prefill_mode),
                 moe_hybrid_ ? " [hybrid]" : "");

    if (env_flag_enabled("DFLASH_DS4_SPEC")) {
        const char * dp = std::getenv("DFLASH_DS4_DRAFT");
        if (dp && *dp) {
            spec_draft_path_ = dp;
            if (moe_hybrid_) {
                std::fprintf(stderr,
                             "[deepseek4] DSpark spec-decode requires monolithic model "
                             "placement; disabled for hybrid expert placement\n");
            } else {
                const bool loaded = load_spec_drafter();
                if (!loaded &&
                    env_flag_enabled("DFLASH_DSPARK_XDNA_REQUIRED")) {
                    return false;
                }
            }
        } else {
            std::fprintf(stderr, "[deepseek4] DFLASH_DS4_SPEC set but DFLASH_DS4_DRAFT gguf missing\n");
        }
    }
    return true;
}

bool DeepSeek4Backend::compute_uniform_hybrid_placement(const DeepSeek4Weights & w,
                                                       int max_ctx,
                                                       MoeHybridPlacement & out,
                                                       std::string * err) const {
    Ds4HybridBudgetInfo budget;
    if (!compute_ds4_hybrid_budget_info(w, cfg_.device.gpu, max_ctx, budget, err)) {
        return false;
    }

    const int hot_per_layer = budget.max_hot_per_layer;
    fill_prefix_hot_placement(w, hot_per_layer, out);

    Ds4ExpertMemoryInfo placed_mem;
    if (!compute_ds4_expert_memory_info(w, &out, placed_mem, err)) {
        return false;
    }

    std::fprintf(stderr,
                 "[deepseek4] hybrid placement: gpu_total=%.2f GiB gpu_free=%.2f GiB core=%.2f GiB kv=%.2f GiB warm=%.2f GiB safety=%.2f GiB expert_budget=%.2f GiB hot/layer=%d\n",
                 gib((uint64_t) budget.gpu_total),
                 gib((uint64_t) budget.gpu_free),
                 gib(budget.core_bytes),
                 gib(budget.kv_bytes),
                 gib(budget.warm_bytes),
                 gib(budget.safety_bytes),
                 gib(budget.expert_budget),
                 hot_per_layer);
    log_ds4_expert_memory_info("placement", placed_mem, w.n_layer);
    return true;
}

bool DeepSeek4Backend::init_hybrid_model() {
    TargetLoadPlan plan;
    plan.skip_expert_tensors = true;
    if (!load_deepseek4_gguf_partial(cfg_.model_path, backend_, plan, w_)) {
        std::fprintf(stderr, "[deepseek4] failed to partially load model for hybrid mode: %s\n",
                     cfg_.model_path);
        return false;
    }

    std::string err;
    const int max_ctx = cfg_.max_ctx > 0 ? cfg_.max_ctx : 8192;
    if (!compute_uniform_hybrid_placement(w_, max_ctx, moe_placement_, &err)) {
        std::fprintf(stderr, "[deepseek4] failed to compute hybrid placement: %s\n", err.c_str());
        return false;
    }

    if (moe_placement_.total_hot >= w_.n_layer * w_.n_expert) {
        free_deepseek4_weights(w_);
        if (!load_deepseek4_gguf(cfg_.model_path, backend_, w_)) {
            std::fprintf(stderr, "[deepseek4] failed to reload full model after placement: %s\n",
                         cfg_.model_path);
            return false;
        }
        return true;
    }

    auto hybrid = std::make_shared<MoeHybridStorage>();
    const MoeHybridConfig hybrid_cfg = make_ds4_parent_worker_cfg(w_);
    if (!build_deepseek4_moe_hybrid_storage_from_file_with_mmap(
            cfg_.model_path, backend_, w_, moe_placement_, &hybrid_cfg, *hybrid, &err)) {
        std::fprintf(stderr, "[deepseek4] failed to build hybrid expert storage: %s\n", err.c_str());
        return false;
    }

    if (hybrid->has_mmap() && !hybrid->materialized_cold_experts) {
        size_t max_expert_bytes = 0;
        for (const auto & layer : hybrid->layers) {
            const size_t per_expert_bytes = layer.fused_gate_up
                ? layer.gate_up_expert_bytes + layer.down_expert_bytes
                : layer.gate_expert_bytes + layer.up_expert_bytes + layer.down_expert_bytes;
            max_expert_bytes = std::max(max_expert_bytes, per_expert_bytes);
        }
        if (max_expert_bytes == 0) {
            std::fprintf(stderr, "[deepseek4] failed to compute streaming expert size\n");
            return false;
        }
        if (!stream_engine_.init(backend_, max_expert_bytes, &err)) {
            std::fprintf(stderr, "[deepseek4] failed to init cold-expert stream engine: %s\n",
                         err.c_str());
            return false;
        }
        std::fprintf(stderr,
                     "[deepseek4] cold-expert stream engine ready: pinned=%.1f MiB scratch=%.1f MiB\n",
                     stream_engine_.pinned_bytes() / 1024.0 / 1024.0,
                     stream_engine_.scratch_bytes() / 1024.0 / 1024.0);
    }

    moe_hybrid_ = std::move(hybrid);
    w_.moe_hybrid = true;
    const int total_cold = w_.n_layer * w_.n_expert - moe_placement_.total_hot;
    const char * cold_backend =
        moe_hybrid_->cold_backend_kind == MoeHybridColdBackend::Gpu ? "gpu" : "cpu";
    std::fprintf(stderr, "[deepseek4] hybrid experts ready: hot=%d cold=%d cold_backend=%s%s\n",
                 moe_placement_.total_hot, total_cold, cold_backend, "");
    return true;
}

int DeepSeek4Backend::clamp_prefill_chunk(
        int proposed_tokens,
        int relative_offset,
        int absolute_pos,
        int snap_pos,
        int capture_from) {
    int n_tokens = proposed_tokens;
    if (n_tokens <= 0) return 0;

    // A snapshot must contain the exact compressor/HC state at its token
    // boundary, never state from later positions in the same graph.
    if (snap_pos > absolute_pos && snap_pos < absolute_pos + n_tokens) {
        n_tokens = snap_pos - absolute_pos;
    }

    // DSpark only consumes the final SWA window of prompt features. Stop the
    // preceding chunk exactly at that window so its capture hooks remain null
    // and it can use the device-resident layer-major scheduler. Without this
    // split, the hooks attach to the entire final (up to 2K-token) chunk and
    // force it through compressor-safe four-token forwards.
    if (capture_from > relative_offset &&
        capture_from < relative_offset + n_tokens) {
        n_tokens = capture_from - relative_offset;
    }
    return n_tokens;
}

// Pre-flight system-memory guard: fail a prefill gracefully instead of letting a
// doomed cudaMalloc (UMA/managed) segfault a HIP worker thread under RAM pressure.
static long ember_mem_available_mb() {
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    }
    std::fclose(f);
    return kb < 0 ? -1 : kb / 1024;
}

int DeepSeek4Backend::do_prefill(const std::vector<int32_t> & tokens,
                                  const DaemonIO & io,
                                  int kv_offset,
                                  int snap_pos,
                                  int snap_slot,
                                  bool allow_spec_capture,
                                  bool force_exact_prefill) {
    // The all-hot layer-range path supports causal chunked prefill. The
    // optimized graph snapshots the previous raw SWA window, attends over
    // that snapshot plus the current ubatch, and commits only the final SWA
    // tail. Learned compressor boundaries are emitted inside the same graph.
    //
    // Mixed hot/cold hybrid execution still has single-token HC semantics, so
    // retain the reference path there.  --chunk 1 is the explicit fallback.
    const PrefillAttentionMode base_prefill_mode = force_exact_prefill
        ? PrefillAttentionMode::Exact
        : cfg_.prefill_mode;
    const int requested_chunk = cfg_.chunk > 0 ? cfg_.chunk : w_.n_swa;
    const int n_total = (int)tokens.size();
    // Bound the layer-major graph to the topology validated by the prefill
    // kernels. Smaller tail chunks use the same scheduler or its reference
    // fallback.
    const int layer_major_cap = DS4_MAX_LAYER_MAJOR_PREFILL_TOKENS;
    const int exact_chunk =
        std::min(requested_chunk, exact_prefill_chunk_limit());
    const int batched_chunk =
        std::max(1, std::min(requested_chunk, layer_major_cap));
    // The graph scheduler consults cache_.prefill_mode. A DSpark request may
    // switch from configured batched prefill to exact q=1 at the final feature
    // window, so restore the resident cache policy on every exit path.
    struct PrefillModeGuard {
        DeepSeek4Cache &cache;
        PrefillAttentionMode saved;
        ~PrefillModeGuard() { cache.prefill_mode = saved; }
    } mode_guard{cache_, cache_.prefill_mode};
    int pos = kv_offset;
    inline_snapshot_saved_ = false;  // set true below iff we persist the snapshot
    // New sequence: clear the cache buffer so compressor state double-buffers
    // and compressed-KV rows start from zeros, exactly like a fresh server.
    // Without this, the first flush windows of a request pool over the
    // previous request's leftover state rows and outputs from the 2nd/3rd
    // request on can drift by a token or two.
    if (kv_offset == 0) {
        reset_deepseek4_cache(cache_);
    }
    // A restored request may ask for another cache boundary exactly where
    // its existing prefix ends. Save before clearing the restored logits or
    // mutating the DSpark rolling feature window for the new delta.
    if (snap_slot >= 0 && snap_pos > 0 && snap_pos == kv_offset) {
        cache_.cur_pos = kv_offset;
        if (snapshot_save(snap_slot)) {
            inline_snapshot_saved_ = true;
            std::fprintf(stderr, "[snap] boundary slot=%d cur_pos=%d\n",
                         snap_slot, kv_offset);
        }
        snap_pos = -1;
        snap_slot = -1;
    }
    last_logits_.clear();
    int spec_capture_from = n_total;
    const bool capture_spec_features =
        allow_spec_capture && spec_enabled_ && spec_drafter_;
    // The layer-major HIP graph can publish mean-over-HC capture rows while it
    // performs the configured dense/sparse prefill. Resident XDNA decode may
    // replace these with an isolated exact shadow suffix after authoritative
    // prefill completes; never change the target graph merely for capture.
    const bool layer_major_spec_capture =
        capture_spec_features && !force_exact_prefill && !moe_hybrid_ &&
        prefill_attention_mode_is_approximate(base_prefill_mode);
    if (!allow_spec_capture) {
        // A request that cannot enter speculative decode must be
        // observationally identical to the configured target-only path.
        // Keeping a stale feature window is unnecessary, and attaching capture
        // hooks below changes the prefill graph even when speculation never
        // runs. This applies to restored deltas as well as fresh prompts.
        spec_feat_window_.clear();
    }
    if (capture_spec_features) {
        const int feat_row = spec_drafter_->n_target_layers * w_.n_embd;
        if (kv_offset == 0 || n_total >= w_.n_swa || feat_row <= 0 ||
            spec_feat_window_.size() % (size_t) feat_row != 0) {
            spec_feat_window_.clear();
            spec_capture_from = std::max(0, n_total - w_.n_swa);
        } else {
            // Keep enough prior rows for the new prompt suffix, then append all
            // new rows. This bounds host capture storage at n_swa without
            // shifting a multi-megabyte feature window after every token.
            const size_t old_rows = spec_feat_window_.size() / (size_t) feat_row;
            const size_t keep_rows = (size_t) std::max(0, w_.n_swa - n_total);
            if (old_rows > keep_rows) {
                const size_t drop_floats = (old_rows - keep_rows) * (size_t) feat_row;
                const size_t keep_floats = keep_rows * (size_t) feat_row;
                std::memmove(spec_feat_window_.data(),
                             spec_feat_window_.data() + drop_floats,
                             keep_floats * sizeof(float));
                spec_feat_window_.resize(keep_floats);
            }
            spec_capture_from = 0;
        }
    }
    const bool timing = env_flag_enabled("DFLASH_DS4_TIMING");
    const auto phase_t0 = Clock::now();
    DeepSeek4StepTelemetry tel_acc;
    int steps = 0;

    // Streaming keepalive: a long prefill emits no tokens, so without this the
    // client sees silence until decode. Fire the callback at chunk boundaries,
    // rate-limited to ~4s; it writes a `: prefill` SSE comment to the client.
    auto keepalive_last = Clock::now();

    for (int i = 0; i < n_total;) {
        if (io.cancelled) return pos;

        if (io.on_prefill_keepalive) {
            auto now = Clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - keepalive_last).count() >= 4000) {
                if (!io.on_prefill_keepalive()) {  // client gone
                    io.cancelled = true;
                    return pos;
                }
                keepalive_last = now;
            }
        }

        // When the layer-major graph owns feature capture, the final SWA window
        // stays on the configured approximate prefill topology. Exact mode and
        // hybrid placement retain the q=1 capture contract.
        const PrefillAttentionMode step_mode = layer_major_spec_capture
            ? base_prefill_mode
            : select_prefill_step_mode(
                base_prefill_mode, force_exact_prefill,
                capture_spec_features, i, spec_capture_from);
        const bool capture_exact =
            capture_spec_features && i >= spec_capture_from &&
            !layer_major_spec_capture;
        const bool step_exact =
            !prefill_attention_mode_is_approximate(step_mode);
        cache_.prefill_mode = step_mode;
        const int step_chunk = moe_hybrid_
            ? 1
            : step_exact ? exact_chunk : batched_chunk;
        const int proposed = capture_exact
            ? 1
            : std::min(step_chunk, n_total - i);
        int n_tok = clamp_prefill_chunk(
            proposed, i, pos,
            snap_slot >= 0 ? snap_pos : -1,
            capture_spec_features ? spec_capture_from : -1);

        // Embed tokens
        std::vector<float> embed(w_.n_embd * n_tok);
        const auto embed_t0 = Clock::now();
        w_.embedder.embed(tokens.data() + i, n_tok, embed.data());
        DeepSeek4StepTelemetry step_tel;
        if (timing) step_tel.embed_us = elapsed_us(embed_t0, Clock::now());

        std::vector<float> logits;
        Ds4VerifyHooks spec_hooks;
        std::vector<float> spec_cap;
        Ds4VerifyHooks * hp = nullptr;
        if (capture_spec_features && i + n_tok > spec_capture_from) {
            spec_hooks.capture_layer_ids = &spec_drafter_->capture_layer_ids;
            spec_hooks.capture_out = &spec_cap;
            spec_hooks.require_fused_q1 = !layer_major_spec_capture;
            hp = &spec_hooks;
        }
        const bool need_step_logits =
            i + n_tok == n_total ||
            (snap_slot >= 0 && snap_pos > 0 && pos + n_tok == snap_pos) ||
            (hp && hp->all_logits_out);
        {
            static const long ember_min_avail_mb = []() {
                return env_nonnegative_long("EMBER_MIN_AVAIL_MB", 1024L);
            }();
            if (ember_min_avail_mb > 0) {
                const long avail = ember_mem_available_mb();
                if (avail >= 0 && avail < ember_min_avail_mb) {
                    std::fprintf(stderr,
                        "[deepseek4] prefill aborted at pos=%d: low memory "
                        "(avail=%ld MiB < %ld MiB floor) - failing request gracefully\n",
                        pos, avail, ember_min_avail_mb);
                    return -1;
                }
            }
        }
        bool ok = false;
        if (moe_hybrid_) {
            ok = deepseek4_step(backend_, cfg_.device.gpu, w_, cache_, embed.data(), n_tok, pos,
                                logits, moe_hybrid_.get(), tokens.data() + i,
                                &stream_engine_, timing ? &step_tel : nullptr,
                                routing_stats_.get(), hp);
        } else {
            std::vector<float> hc_state;
            // In exact q=1 prefill, intermediate logits have no consumer and
            // no effect on KV/compressor state.  Avoid the final HC/output
            // projection for those steps, while retaining it at the prompt
            // end and every inline snapshot boundary.
            std::vector<float> * step_logits =
                step_exact && n_tok == 1 && !need_step_logits &&
                        exact_prefill_skip_intermediate_logits()
                    ? nullptr
                    : &logits;
            ok = deepseek4_step_layer_range(
                backend_, cfg_.device.gpu, w_, cache_, hc_state, embed.data(), n_tok, pos,
                0, w_.n_layer, step_logits, tokens.data() + i,
                timing ? &step_tel : nullptr,
                step_mode != PrefillAttentionMode::Sparse, hp);
        }
        if (ok && hp && !spec_cap.empty()) {
            const int feat_row = spec_drafter_->n_target_layers * w_.n_embd;
            const int first_capture = std::max(0, spec_capture_from - i);
            for (int t = first_capture; t < n_tok; ++t) {
                spec_feat_window_.insert(
                    spec_feat_window_.end(),
                    spec_cap.begin() + (size_t) t * feat_row,
                    spec_cap.begin() + (size_t) (t + 1) * feat_row);
            }
        }
        if (!ok) {
            std::fprintf(stderr, "[deepseek4] prefill step failed at pos=%d\n", pos);
            return -1;
        }
        if (timing) {
            add_step_tel(tel_acc, step_tel);
            steps++;
        }
        if (!logits.empty()) {
            last_logits_ = std::move(logits);
        }
        pos += n_tok;
        cache_.cur_pos = pos;
        i += n_tok;

        if (snap_slot >= 0 && snap_pos > 0 && pos == snap_pos) {
            if (snapshot_save(snap_slot)) {
                inline_snapshot_saved_ = true;
                std::fprintf(stderr, "[snap] boundary slot=%d cur_pos=%d\n",
                             snap_slot, pos);
            } else {
                std::fprintf(stderr,
                             "[snap] boundary save failed slot=%d cur_pos=%d\n",
                             snap_slot, pos);
            }
            snap_pos = -1;
            snap_slot = -1;
        }
    }
    if (timing) {
        log_step_tel("prefill", n_total, steps, elapsed_s(phase_t0), tel_acc);
    }
    return pos;
}

bool DeepSeek4Backend::do_decode(int committed, int n_gen,
                                  const std::vector<int32_t> & history_prefix,
                                  std::vector<int32_t> & out_tokens,
                                  const DaemonIO & io,
                                  const BudgetHook & budget_hook,
                                  bool * forced_close_out,
                                  bool * degenerate_close_out,
                                  std::string * termination_reason_out,
                                  const std::function<bool()> & force_greedy,
                                  int resume_from,
                                  const std::vector<int32_t> & tool_open_ids,
                                  const std::vector<int32_t> & tool_close_ids,
                                  const std::shared_ptr<dflash::common::TokenMask> &
                                      token_mask) {
    if (forced_close_out) *forced_close_out = false;
    if (degenerate_close_out) *degenerate_close_out = false;
    if (termination_reason_out) termination_reason_out->clear();
    const bool timing = env_flag_enabled("DFLASH_DS4_TIMING");
    const auto phase_t0 = Clock::now();
    DeepSeek4StepTelemetry tel_acc;
    int steps = 0;
    const bool process_logits = sampler_.needs_logit_processing();
    std::vector<int32_t> history;
    if (process_logits) {
        history = history_prefix;
        if (n_gen > 0) {
            history.reserve(history.size() + (size_t)n_gen);
        }
    }

    // Feed the already-appended token out_tokens[idx] through the model so its KV
    // is established at position committed+idx; `lg` receives the logits predicting
    // the next position. Mirrors the main loop's per-token step; used by the
    // force-close continuation below.
    auto feed_idx = [&](int idx, std::vector<float> & lg) -> bool {
        int32_t tok = out_tokens[idx];
        std::vector<float> embed(w_.n_embd);
        w_.embedder.embed(&tok, 1, embed.data());
        return deepseek4_step(backend_, cfg_.device.gpu, w_, cache_, embed.data(), 1,
                              committed + idx, lg, moe_hybrid_.get(), &tok,
                              moe_hybrid_ ? &stream_engine_ : nullptr, nullptr,
                              routing_stats_.get(), nullptr);
    };
    auto sample_from = [&](std::vector<float> & lg) -> int32_t {
        if (token_mask && token_mask->active())
            token_mask->apply(lg.data(), w_.n_vocab);
        if (process_logits && !(force_greedy && force_greedy()))
            return sample_logits(lg.data(), w_.n_vocab, sampler_, history, sampler_rng_);
        int32_t best = 0; float mv = lg[0];
        for (int i = 1; i < w_.n_vocab; i++) if (lg[i] > mv) { mv = lg[i]; best = i; }
        return best;
    };

    // Resuming after spec decode: out_tokens is pre-populated, so the counter
    // starts at its size. This also means the `generated == 0` shortcut below
    // is never taken on resume, which is required — last_logits_ is stale once
    // spec has run, and the correct seed is out_tokens.back() fed at
    // committed+generated-1 (the seam invariant).
    if (resume_from > 0 && process_logits) {
        for (int i = 0; i < resume_from && i < (int) out_tokens.size(); i++)
            history.push_back(out_tokens[(size_t) i]);
    }
    ThinkingBudgetState thinking_budget;
    ProgressCycleDetector progress_cycle(
        budget_hook.natural_close_token_ids, history_prefix,
        tool_open_ids, tool_close_ids);
    // Speculative decode may have naturally closed thinking before AR takes
    // over at the reply-budget seam. Detect that once across the existing
    // prefix, then track only the newly appended suffix below.
    thinking_budget.observe_existing(
        out_tokens, budget_hook.natural_close_token_ids);
    for (int32_t token : out_tokens) {
        if (progress_cycle.observe(token)) {
            if (degenerate_close_out) *degenerate_close_out = true;
            if (termination_reason_out)
                *termination_reason_out = progress_cycle.reason_name();
            std::fprintf(stderr,
                         "[deepseek4] progress watchdog fired at AR seam: "
                         "reason=%s period=%zu span=%zu tokens\n",
                         progress_cycle.reason_name(),
                         progress_cycle.cycle_period(),
                         progress_cycle.repeated_span());
            return true;
        }
    }
    for (int generated = resume_from; generated < n_gen; generated++) {
        if (io.cancelled) break;

        // Budget hook: reply-budget reached — force the model to stop thinking and
        // ANSWER. Inject the close sequence (a "wrap up now" directive + </think>),
        // FEED it through the model so its KV holds the forced </think>, then keep
        // decoding the reply into the reserved budget until EOS or n_gen.
        //
        // Previously this injected the close tokens and broke immediately, so the
        // model never wrote a reply — every runaway-thinking turn returned only
        // "<reasoning></think>" with empty content, which upstream agents retry in
        // an unbounded loop. The reserved reply budget was reserved but unused.
        if (thinking_budget.should_force_close(
                n_gen, (size_t)generated,
                budget_hook.hard_limit_remaining,
                !budget_hook.close_token_ids.empty())) {
            thinking_budget.mark_closed();
            if (forced_close_out) *forced_close_out = true;
            std::vector<float> logits;
            // The last sampled token is not in KV yet (fed lazily next iteration);
            // feed it first so the close sequence follows it in-context.
            if (!out_tokens.empty() &&
                !feed_idx((int)out_tokens.size() - 1, logits)) return false;
            // Append, emit, and feed each close token so </think> lands in KV.
            for (int32_t close_tok : budget_hook.close_token_ids) {
                if (io.cancelled) break;
                out_tokens.push_back(close_tok);
                if (token_mask) token_mask->accept(close_tok);
                io.emit(close_tok);
                progress_cycle.observe(close_tok);
                if (process_logits) history.push_back(close_tok);
                if (!feed_idx((int)out_tokens.size() - 1, logits)) return false;
            }
            progress_cycle.begin_visible();
            // Decode the actual reply into the reserved budget.
            while (!io.cancelled && (int)out_tokens.size() < n_gen) {
                int32_t next = sample_from(logits);
                if (process_logits) history.push_back(next);
                out_tokens.push_back(next);
                if (token_mask) token_mask->accept(next);
                io.emit(next);
                if (deepseek4_is_eos_tok(next, w_)) break;
                if (progress_cycle.observe(next)) {
                    if (degenerate_close_out) *degenerate_close_out = true;
                    if (termination_reason_out)
                        *termination_reason_out = progress_cycle.reason_name();
                    std::fprintf(stderr,
                                 "[deepseek4] progress watchdog fired after "
                                 "forced close: reason=%s period=%zu span=%zu "
                                 "tokens\n",
                                 progress_cycle.reason_name(),
                                 progress_cycle.cycle_period(),
                                 progress_cycle.repeated_span());
                    break;
                }
                if (!feed_idx((int)out_tokens.size() - 1, logits)) return false;
            }
            break;
        }

        // Get last logits and sample
        std::vector<float> logits;
        if (generated == 0 && !last_logits_.empty()) {
            logits = last_logits_;
        } else {
            std::vector<float> embed(w_.n_embd);
            int32_t tok_to_eval = out_tokens.empty() ? 0 : out_tokens.back();
            const auto embed_t0 = Clock::now();
            w_.embedder.embed(&tok_to_eval, 1, embed.data());
            DeepSeek4StepTelemetry step_tel;
            if (timing) step_tel.embed_us = elapsed_us(embed_t0, Clock::now());

            const int pos = std::max(0, committed + generated - 1);
            // This is the genuine single-token target eval: exactly what a plain
            // AR token costs, with none of the verify-graph overhead. It is the
            // baseline DSpark's profitability accounting needs, and ds4 samples
            // the same thing around its ordinary target decode (ds4.c:59514).
            // Timed unconditionally (not gated on `timing`) because the scheduler
            // depends on it in production; two Clock::now() calls per token are
            // negligible against a full forward pass.
            const auto ar_step_t0 = Clock::now();
            if (!deepseek4_step(backend_, cfg_.device.gpu, w_, cache_, embed.data(), 1,
                                pos, logits,
                                moe_hybrid_.get(), &tok_to_eval,
                                moe_hybrid_ ? &stream_engine_ : nullptr,
                                timing ? &step_tel : nullptr,
                                routing_stats_.get(), nullptr)) {
                std::fprintf(stderr, "[deepseek4] decode step failed\n");
                return false;
            }
            dspark_worker_note_target_eval(
                (double) elapsed_us(ar_step_t0, Clock::now()) / 1000.0);
            if (timing) {
                add_step_tel(tel_acc, step_tel);
                steps++;
            }
        }

        int32_t next_token = 0;
        const auto sample_t0 = Clock::now();
        // B6: force greedy argmax for structural (tool-call scaffolding) tokens
        // even when the sampler would otherwise process logits, so temperature
        // noise cannot corrupt the DSML/JSON structure. Payload positions still
        // sample. The forced token is pushed to `history` below (process_logits),
        // so rep/freq penalties still see it.
        // Constrain BEFORE the branch: the greedy path (and DSpark's argmax)
        // must obey the grammar just as the sampled path does.
        if (token_mask && token_mask->active())
            token_mask->apply(logits.data(), w_.n_vocab);
        const bool greedy_this = !process_logits || (force_greedy && force_greedy());
        if (!greedy_this) {
            next_token = sample_logits(logits.data(), w_.n_vocab, sampler_,
                                       history, sampler_rng_);
        } else {
            float max_val = logits[0];
            for (int i = 1; i < w_.n_vocab; i++) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    next_token = i;
                }
            }
        }
        if (timing) tel_acc.sample_us += elapsed_us(sample_t0, Clock::now());
        if (process_logits) {
            history.push_back(next_token);
        }
        out_tokens.push_back(next_token);
        if (token_mask) token_mask->accept(next_token);
        thinking_budget.observe_latest(
            out_tokens, budget_hook.natural_close_token_ids);
        const auto emit_t0 = Clock::now();
        io.emit(next_token);
        if (timing) tel_acc.emit_us += elapsed_us(emit_t0, Clock::now());

        if (deepseek4_is_eos_tok(next_token, w_)) {
            break;
        }
        if (progress_cycle.observe(next_token)) {
            if (degenerate_close_out) *degenerate_close_out = true;
            if (termination_reason_out)
                *termination_reason_out = progress_cycle.reason_name();
            std::fprintf(stderr,
                         "[deepseek4] progress watchdog fired: reason=%s "
                         "period=%zu span=%zu tokens\n",
                         progress_cycle.reason_name(),
                         progress_cycle.cycle_period(),
                         progress_cycle.repeated_span());
            // Tool-region scoping diagnostic. entries=0 means the open marker
            // was never matched, i.e. the model's token split for it differs
            // from the server's encode(); a small since_exit instead means the
            // region was entered and the echo fired on the stale window just
            // after it closed. The id dumps make the split directly visible.
            {
                auto join = [](const std::vector<int32_t> &v, size_t from,
                               size_t n) {
                    std::string out;
                    for (size_t i = from; i < v.size() && i < from + n; i++) {
                        if (!out.empty()) out += ',';
                        out += std::to_string(v[i]);
                    }
                    return out;
                };
                const auto &all = progress_cycle.all_tokens();
                const auto &open_ids = progress_cycle.tool_open_ids();
                std::fprintf(stderr,
                    "[deepseek4] tool-region: entries=%zu in_region=%d "
                    "since_exit=%zu expect_open=[%s]\n",
                    progress_cycle.region_entries(),
                    progress_cycle.in_tool_region() ? 1 : 0,
                    progress_cycle.tokens_since_region_exit(),
                    join(open_ids, 0, 8).c_str());
                std::fprintf(stderr, "[deepseek4] gen_head=[%s]\n",
                             join(all, 0, 48).c_str());
                std::fprintf(stderr, "[deepseek4] gen_tail=[%s]\n",
                             join(all, all.size() > 24 ? all.size() - 24 : 0,
                                  24).c_str());
            }
            break;
        }
    }
    if (timing) {
        log_step_tel("decode", (int)out_tokens.size(), steps, elapsed_s(phase_t0), tel_acc);
    }
    return true;
}

// ── Spec / force-close coexistence ──────────────────────────────────────
//
// The thinking force-close used to disable speculative decode outright
// (budget_requires_ar), which meant every thinking request — i.e. essentially
// all agent traffic — ran plain AR. They can coexist: speculate for the bulk,
// stop short of the reply reserve, and let AR own the tail, since AR is the
// only path that can inject the close sequence at an exact position.
//
// ds4 does the same thing with its `tail_min` (ds4.c: default 10).
namespace {

// Tokens speculative decode may emit before AR must take over.
int ds4_spec_emit_budget(const GenerateRequest & req) {
    if (req.budget_hook.close_token_ids.empty()) return req.n_gen;  // no hook: all of it
    // A verify step commits up to q_cap (4) tokens at once, so stop a block
    // short of the trigger rather than exactly on it — landing inside the
    // reserve would hand AR a position past where the close must fire.
    const int margin = 4;
    const int budget = req.n_gen - req.budget_hook.hard_limit_remaining - margin;
    return budget > 0 ? budget : 0;
}

// ── Context ceiling (MEMORY, not throughput) ────────────────────────────
//
// Speculation stays profitable to the full context window. This ceiling exists
// because of a real 2026-07-28 incident, not a throughput cliff, and both of its
// original justifications were re-measured on 2026-08-22 before it was raised.
//
// MEMORY (the real reason it existed). The batched verify holds activations for
// q tokens across every layer at the full KV length, and that footprint is what
// OOMed, not the KV cache -- the whole compressed cache at 65k is under 1 GB
// (comp_cap = max_ctx/4 + 16 rows, ~21 MB/layer x 43). The 2026-07-28 numbers
// were taken with a ~95 GB model resident:
//   AR   @ ~65k -> GTT ~21-22 GB   (~12 GB headroom)
//   spec @ ~65k -> GTT  33.5 GB    -> avail 21 MB -> "cudaMalloc failed"
// Re-measured here with the ~85 GB published model, spec forced on, q=6,
// sampling peak mem_info_gtt_used across the generation (GTT total 124 GiB):
//   ctx 18553 tok -> peak 12.79 GiB
//   ctx 38059 tok -> peak 13.98 GiB
//   ctx 57562 tok -> peak 15.25 GiB   (~109 GiB spare)
// Less than half the 2026-07-28 consumption with an order of magnitude more
// headroom, so the OOM condition no longer applies at this model size.
//
// THROUGHPUT. The old note claimed a knee at "~14k (near parity), ~28k (54%
// slower)". That does not reproduce. Those A/Bs appear to have confounded
// context with draft ACCEPTANCE: a prompt whose continuation the drafter cannot
// predict makes speculation lose at ANY length (measured: acceptance 0.17-0.35
// loses even at 222 tokens), which looks like a context effect if length and
// workload vary together. Holding the generation task identical and varying only
// preceding context, acceptance stays ~0.98 and speculation wins throughout:
//   ctx     43 tok -> 37.62 vs AR 23.48  (+60%)
//   ctx   3925 tok -> 35.31 vs AR 22.71  (+56%)
//   ctx   8800 tok -> 33.37 vs AR 22.01  (+52%)
//   ctx  18553 tok -> 30.07 vs AR 20.93  (+44%)
//   ctx  38059 tok -> 23.97 vs AR 19.07  (+26%)
// The advantage decays with depth but never reaches parity. At the old 16384
// default, an 18.5k-token request was handed to AR at 20.93 instead of 30.07 --
// a 30% loss for no measured reason.
//
// What actually decides profitability is acceptance, not position: from the
// per-phase timings the verify is ~90% of a step and is context-INDEPENDENT
// (86.6 ms at 8764 tok vs 95.8 ms at 216 tok), so break-even sits near 2.3
// accepted tokens per step (~0.40 acceptance at width 6). DSparkProfitScheduler
// already measures that directly and stands down when it is not met, which is
// the mechanism that should own this decision; a position ceiling is a proxy for
// something the engine observes first-hand.
//
// So this is now a memory backstop at the context limit rather than a
// throughput gate, and it tracks max_ctx (main.c, 131072). A ceiling pinned
// below the context window just recreates the cliff this replaced: requests
// past it lose the +26..60% speculation gives, for no measured reason.
// Supporting measurements at the new value: the engine reports KV cache
// 877.8 MB at ctx=131072, a load there was verified (GTT 11.2 GiB, 24 GiB host
// free), and speculation ran at 77,068 prompt tokens with acceptance 1.00.
// 0 still disables speculation entirely. Lower it again only with a measured
// GTT peak at the target length, the same way it was raised.
constexpr int kSpecMaxCtxDefault = 131072;

int ds4_spec_max_ctx() {
    static const int v = [] {
        if (const char * e = std::getenv("DFLASH_DS4_SPEC_MAX_CTX")) {
            char * end = nullptr;
            errno = 0;
            const long n = std::strtol(e, &end, 10);
            if (errno == 0 && end != e && *end == '\0' &&
                n >= 0 && n <= INT_MAX) {
                return (int) n;   // 0 disables spec entirely
            }
            std::fprintf(stderr,
                         "[ds4-spec] invalid DFLASH_DS4_SPEC_MAX_CTX='%s'; "
                         "using default %d\n",
                         e, kSpecMaxCtxDefault);
        }
        return kSpecMaxCtxDefault;
    }();
    return v;
}

// Speculation may begin below the ceiling and generate across it in one turn.
// Bound the number of tokens owned by DSpark so the already-supported spec→AR
// seam takes over at the boundary instead of checking only once per request.
int ds4_spec_context_budget(int committed) {
    const int ceiling = ds4_spec_max_ctx();
    if (ceiling == 0 || committed >= ceiling) return 0;
    return ceiling - committed;
}

// committed = KV tokens already in cache (the context this request decodes on).
bool ds4_spec_should_run(const GenerateRequest & req, bool spec_enabled,
                         bool have_drafter, bool sampling_requires_ar,
                         int spec_budget, int committed,
                         bool profitability_allowed) {
    const char * debug_env = std::getenv("DFLASH_DS4_SPEC_DEBUG");
    const bool debug = debug_env && (*debug_env == '1' ||
                                    *debug_env == 'y' ||
                                    *debug_env == 'Y' ||
                                    *debug_env == 't' ||
                                    *debug_env == 'T');
    const char * reason = nullptr;
    if (!spec_enabled) reason = "disabled";
    else if (!have_drafter) reason = "no_drafter";
    const int ceiling = ds4_spec_max_ctx();
    if (!reason && (ceiling == 0 || committed >= ceiling)) {
        static thread_local int last_logged = -1;
        if (last_logged != committed / 4096) {   // throttle: once per 4k band
            last_logged = committed / 4096;
            std::fprintf(stderr,
                         "[ds4-spec] skipped: ctx=%d max=%d (large-context "
                         "verify disabled; set DFLASH_DS4_SPEC_MAX_CTX to override)\n",
                         committed, ceiling);
        }
        reason = "context";
    }
    if (!reason && req.force_ar_decode) reason = "force_ar";
    // Constrained decoding owns the sampler, and the speculative loop commits
    // tokens through its own path — those would bypass both the mask and the
    // matcher's accept(), desyncing grammar state and letting exactly the
    // malformed shapes the grammar exists to prevent slip through. Route
    // constrained requests through AR, the same way the thinking budget hook
    // already does. Masking drafts directly is the follow-up.
    else if (!reason && req.token_mask) reason = "token_mask";
    else if (!reason && sampling_requires_ar) reason = "sampling";
    else if (!reason && req.n_gen <= 0) reason = "empty_budget";
    else if (!reason && spec_budget < kDSparkMinSpecBudget) reason = "short_budget";
    if (!reason && !profitability_allowed) reason = "profitability_gate";
    const bool run = reason == nullptr;
    if (debug) {
        std::fprintf(stderr,
                     "[ds4-spec] gate: run=%d reason=%s ctx=%d max=%d "
                     "force_ar=%d sampled=%d n_gen=%d budget=%d\n",
                     run ? 1 : 0, run ? "eligible" : reason,
                     committed, ceiling, req.force_ar_decode ? 1 : 0,
                     sampling_requires_ar ? 1 : 0, req.n_gen, spec_budget);
    }
    return run;
}

static void describe_prefill(GenerateResult &result,
                             PrefillAttentionMode configured_mode,
                             bool force_exact_prefill,
                             bool prepare_spec,
                             bool force_ar_decode,
                             int token_count) {
    result.prefill_tokens = std::max(0, token_count);
    if (token_count <= 0) {
        result.prefill_mode = "none";
        result.prefill_reason = "cache_hit";
        return;
    }
    if (force_exact_prefill) {
        result.prefill_mode = "exact";
        result.prefill_reason = "forced_exact";
        return;
    }
    if (prepare_spec) {
        const bool configured_approx =
            prefill_attention_mode_is_approximate(configured_mode);
        result.prefill_mode = configured_approx
            ? prefill_attention_mode_name(configured_mode)
            : "exact";
        result.prefill_reason = "dspark_capture";
        return;
    }
    result.prefill_mode = prefill_attention_mode_name(configured_mode);
    result.prefill_reason = force_ar_decode ? "target_ar" : "configured";
}

}  // namespace

GenerateResult DeepSeek4Backend::generate_impl(const GenerateRequest & req,
                                                const DaemonIO & io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    auto t0 = Clock::now();
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    // Decide before prefill whether this request can actually reach DSpark.
    // The final committed position is known from the prompt length here; using
    // the same context- and reply-clamped budget as decode prevents capture
    // hooks from perturbing requests for which speculation will be skipped.
    const bool sampling_requires_ar = sampler_.needs_logit_processing();
    const int capture_spec_budget = std::min(
        ds4_spec_emit_budget(req),
        ds4_spec_context_budget((int) req.prompt.size()));
    const bool eligible_without_gate = dspark_request_can_prepare(
        spec_enabled_, spec_drafter_ != nullptr, req.force_ar_decode,
        sampling_requires_ar, req.n_gen, capture_spec_budget);
    const bool profitability_allowed = !eligible_without_gate ||
        dspark_worker_scheduler().allow_spec_request();
    const bool prepare_spec = eligible_without_gate && profitability_allowed;
    describe_prefill(result, cfg_.prefill_mode, req.force_exact_prefill,
                     prepare_spec, req.force_ar_decode,
                     (int)req.prompt.size());

    // Prefill
    int committed = do_prefill(req.prompt, out_io, /*kv_offset=*/0,
                               req.snap_pos, req.snap_slot,
                               prepare_spec,
                               req.force_exact_prefill);
    if (committed < 0) {
        result.fail(GenerateErrorCode::PrefillFailed);
        return result;
    }
    maybe_log_prefill_fingerprint(last_logits_, committed);
    result.prefill_s = elapsed_s(t0);
    result.snapshot_saved = inline_snapshot_saved_;  // #2: only true on a real save

    if (out_io.cancelled) {
        result.succeed();
        maybe_save_routing_stats();
        return result;
    }

    if (req.n_gen <= 0) {
        result.succeed();
        maybe_save_routing_stats();
        return result;
    }

    // Decode
    auto t1 = Clock::now();
    // The DSpark verifier is greedy-only. Route sampling and penalties through
    // AR so the request's sampler contract is not silently ignored.
    const int spec_budget = std::min(
        ds4_spec_emit_budget(req), ds4_spec_context_budget(committed));
    std::vector<int32_t> gen;
    gen.reserve((size_t) req.n_gen);
    float accept_rate = 0.0f;
    bool spec_ran = false;
    bool spec_terminal = false;   // spec finished the generation on its own
    bool spec_degenerate = false;
    std::string termination_reason;
    ProgressCycleDetector spec_progress(
        req.budget_hook.natural_close_token_ids, req.prompt,
        req.tool_region_open_ids, req.tool_region_close_ids);
    if (ds4_spec_should_run(req, spec_enabled_, spec_drafter_ != nullptr,
                            sampling_requires_ar, spec_budget, committed,
                            profitability_allowed)) {
        if (last_logits_.empty()) {
            result.fail(GenerateErrorCode::DecodeFailed, "spec: no prefill logits");
            return result;
        }
        int seed = 0;
        { float mv = last_logits_[0];
          for (int i = 1; i < w_.n_vocab; i++) if (last_logits_[i] > mv) { mv = last_logits_[i]; seed = i; } }
        gen.push_back(seed);
        out_io.emit(seed);
        spec_degenerate = spec_progress.observe(seed);
        if (!out_io.cancelled && !deepseek4_is_eos_tok(seed, w_) && spec_budget > 1) {
            const int feat_row = spec_drafter_->n_target_layers * w_.n_embd;
            const int win_len = feat_row > 0 ? (int) (spec_feat_window_.size() / feat_row) : 0;
            std::vector<int32_t> spec_toks;
            spec_ran = true;
            if (!run_deepseek4_dspark_spec_decode(
                    backend_, cfg_.device.gpu, w_, cache_, *spec_drafter_, committed, seed,
                    spec_budget - 1,
                    win_len > 0 ? spec_feat_window_.data() : nullptr, win_len,
                    spec_toks, &accept_rate, spec_xdna_draft_compute_.get(),
                    [&out_io, &spec_progress, &spec_degenerate,
                     &termination_reason](int32_t tok) {
                        if (out_io.cancelled) return false;
                        out_io.emit(tok);
                        if (spec_progress.observe(tok)) {
                            spec_degenerate = true;
                            termination_reason = spec_progress.reason_name();
                            std::fprintf(stderr,
                                         "[deepseek4] speculative progress "
                                         "watchdog fired: reason=%s period=%zu "
                                         "span=%zu tokens\n",
                                         spec_progress.reason_name(),
                                         spec_progress.cycle_period(),
                                         spec_progress.repeated_span());
                            return false;
                        }
                        return !out_io.cancelled;
                    })) {
                result.fail(GenerateErrorCode::DecodeFailed,
                            "DSpark speculative decode failed");
                return result;
            }
            gen.insert(gen.end(), spec_toks.begin(), spec_toks.end());
        }
        // Spec owns the whole generation only when it ran out the clock, hit
        // EOS, or was cancelled. Otherwise it stopped at the spec budget and AR
        // continues below from the seam.
        spec_terminal = spec_degenerate || out_io.cancelled ||
                        (int) gen.size() >= req.n_gen ||
                        (!gen.empty() && deepseek4_is_eos_tok(gen.back(), w_));
    }

    bool forced_close = false;
    bool degenerate_close = spec_degenerate;
    if (!spec_terminal &&
        !do_decode(committed, req.n_gen, req.prompt, gen, out_io,
                   req.budget_hook, &forced_close, &degenerate_close,
                   &termination_reason,
                   req.force_greedy_next,
                   (int) gen.size(),
                   req.tool_region_open_ids, req.tool_region_close_ids,
                   req.token_mask)) {
        result.fail(GenerateErrorCode::DecodeFailed);
        return result;
    }

    result.succeed();
    result.tokens = std::move(gen);
    result.decode_s = elapsed_s(t1);
    result.budget_forced_close = forced_close;
    result.degenerate_decode_close = degenerate_close;
    result.termination_reason = termination_reason;
    result.accept_rate = accept_rate;
    result.spec_decode_ran = spec_ran;
    if (spec_ran) {
        dspark_worker_scheduler().note_request_result(accept_rate);
        std::fprintf(stderr,
                     "[deepseek4] DSpark decode: %zu tok in %.3fs (%.1f tok/s) "
                     "accept_rate=%.2f spec_budget=%d tail_ar=%d\n",
                     result.tokens.size(), result.decode_s,
                     result.decode_s > 0 ? result.tokens.size() / result.decode_s : 0.0,
                     accept_rate, spec_budget,
                     spec_terminal ? 0 : 1);
    }
    maybe_save_routing_stats();
    return result;
}

// ── Snapshots ───────────────────────────────────────────────────────────

void DeepSeek4Backend::release_idle_graphs() {
    // Must run on the generation worker thread: the layer-major graph caches,
    // the shared gallocr and the hybrid runtime are thread_local, so a call
    // from any other thread would free nothing and null the wrong handles.
    // deepseek4_release_runtime_graphs() is owner-scoped (keyed on w_.ctx), so
    // it is safe on a live model - weights and KV cache are untouched and the
    // graphs rebuild lazily on the next request.
    deepseek4_release_runtime_graphs(w_);
    reset_dspark_head_runtime_cache();
}

bool DeepSeek4Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    if (!snap_backend_ || cache_.cur_pos <= 0) return false;
    if (!deepseek4_snapshot_save(cache_, snap_backend_, snapshots_[slot],
                                 &last_logits_, &spec_feat_window_)) {
        snapshot_logits_[slot].clear();
        snapshot_spec_features_[slot].clear();
        return false;
    }
    // The tensor snapshot alone is insufficient for resuming generation:
    // exact-prefix hits need the last prompt logits, and DSpark needs its
    // host-side rolling target-feature window.
    snapshot_logits_[slot] = last_logits_;
    snapshot_spec_features_[slot] = spec_feat_window_;
    if (snapshot_logits_[slot].empty()) {
        snapshot_free(slot);
        return false;
    }
    return true;
}

void DeepSeek4Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    free_deepseek4_snapshot(snapshots_[slot]);
    std::vector<float>().swap(snapshot_logits_[slot]);
    std::vector<float>().swap(snapshot_spec_features_[slot]);
}

bool DeepSeek4Backend::snapshot_used(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    return snapshots_[slot].ctx != nullptr;
}

int DeepSeek4Backend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return 0;
    return snapshots_[slot].cur_pos;
}

ModelBackend::SnapshotRef DeepSeek4Backend::snapshot_ref(int slot) const {
    SnapshotRef ref;
    if (slot < 0 || slot >= PREFIX_SLOTS) return ref;
    const auto & snap = snapshots_[slot];
    if (!snap.ctx) return ref;
    ref.ctx      = snap.ctx;
    ref.buf      = snap.buf;
    ref.cur_pos  = snap.cur_pos;
    ref.last_tok = -1;  // deepseek4 reseeds decode from the restored logits
    return ref;
}

bool DeepSeek4Backend::snapshot_adopt(int slot, ggml_context * ctx,
                                      ggml_backend_buffer_t buf, int cur_pos,
                                      int32_t last_tok) {
    (void)last_tok;  // decode reseeds from ds4_snap_logits, not the last token
    if (slot < 0 || slot >= PREFIX_SLOTS || !ctx) return false;
    if (cache_.n_layer <= 0 || cache_.layers.size() != (size_t)cache_.n_layer) {
        return false;
    }
    snapshot_free(slot);

    DeepSeek4Snapshot snap;
    snap.ctx = ctx;
    snap.buf = buf;
    snap.cur_pos = cur_pos;
    snap.layers.resize((size_t)cache_.n_layer);

    // Rebind by name (names are assigned in deepseek4_snapshot_save).
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t;
         t = ggml_get_next_tensor(ctx, t)) {
        if (!t->name[0]) continue;
        int il = -1;
        if (std::strcmp(t->name, "ds4_hc_state_snap") == 0) {
            snap.hc_state_snap = t;
        } else if (std::strcmp(t->name, "ds4_snap_meta") == 0) {
            snap.meta_snap = t;
        } else if (std::strcmp(t->name, "ds4_snap_logits") == 0) {
            snap.logits_snap = t;
        } else if (std::strcmp(t->name, "ds4_snap_spec_feat") == 0) {
            snap.spec_feat_snap = t;
        } else if (std::sscanf(t->name, "ds4_raw_kv_%d", &il) == 1) {
            if (il >= 0 && il < cache_.n_layer) snap.layers[(size_t)il].raw_kv = t;
        } else if (std::sscanf(t->name, "ds4_comp_kv_%d", &il) == 1) {
            if (il >= 0 && il < cache_.n_layer) snap.layers[(size_t)il].comp_kv = t;
        } else if (std::sscanf(t->name, "ds4_index_comp_kv_%d", &il) == 1) {
            if (il >= 0 && il < cache_.n_layer) snap.layers[(size_t)il].index_comp_kv = t;
        } else if (std::sscanf(t->name, "ds4_attn_ckv_%d", &il) == 1) {
            if (il >= 0 && il < cache_.n_layer)
                snap.layers[(size_t)il].attn_compressor.state_kv = t;
        } else if (std::sscanf(t->name, "ds4_attn_cscore_%d", &il) == 1) {
            if (il >= 0 && il < cache_.n_layer)
                snap.layers[(size_t)il].attn_compressor.state_score = t;
        } else if (std::sscanf(t->name, "ds4_idx_ckv_%d", &il) == 1) {
            if (il >= 0 && il < cache_.n_layer)
                snap.layers[(size_t)il].indexer_compressor.state_kv = t;
        } else if (std::sscanf(t->name, "ds4_idx_cscore_%d", &il) == 1) {
            if (il >= 0 && il < cache_.n_layer)
                snap.layers[(size_t)il].indexer_compressor.state_score = t;
        }
    }

    // Validate against the live cache. Anything missing, mismatched in shape,
    // or lacking logits means this file does not describe this model — reject
    // rather than attach KV that decode would silently misread. `raw_kv` is
    // mandatory per layer; the compressed/indexer tensors are optional in the
    // same pattern snapshot_save uses (null when the layer has none).
    auto shape_ok = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (!a && !b) return true;
        if (!a || !b) return false;
        if (a->type != b->type || ggml_n_dims(a) != ggml_n_dims(b)) return false;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) if (a->ne[i] != b->ne[i]) return false;
        return true;
    };
    auto prefix_shape_ok = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (!a && !b) return true;
        if (!a || !b) return false;
        if (a->type != b->type || ggml_n_dims(a) != ggml_n_dims(b)) return false;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (i == 1) {
                if (a->ne[i] < 1 || a->ne[i] > b->ne[i]) return false;
            } else if (a->ne[i] != b->ne[i]) {
                return false;
            }
        }
        return true;
    };
    bool ok = snap.hc_state_snap && snap.meta_snap && snap.logits_snap &&
              shape_ok(snap.hc_state_snap, cache_.hc_state) &&
              ggml_nelements(snap.meta_snap) == (int64_t)cache_.n_layer * 2;
    for (int il = 0; ok && il < cache_.n_layer; ++il) {
        const auto & live = cache_.layers[(size_t)il];
        const auto & got  = snap.layers[(size_t)il];
        ok = got.raw_kv && shape_ok(got.raw_kv, live.raw_kv) &&
             prefix_shape_ok(got.comp_kv, live.comp_kv) &&
             prefix_shape_ok(got.index_comp_kv, live.index_comp_kv) &&
             shape_ok(got.attn_compressor.state_kv, live.attn_compressor.state_kv) &&
             shape_ok(got.attn_compressor.state_score, live.attn_compressor.state_score) &&
             shape_ok(got.indexer_compressor.state_kv, live.indexer_compressor.state_kv) &&
             shape_ok(got.indexer_compressor.state_score, live.indexer_compressor.state_score);
    }
    if (!ok) {
        // Caller owns ctx/buf on failure — detach before returning.
        snap.ctx = nullptr;
        snap.buf = nullptr;
        return false;
    }

    // Recover the per-layer compressor counters.
    {
        std::vector<int32_t> meta((size_t)cache_.n_layer * 2);
        ggml_backend_tensor_get(snap.meta_snap, meta.data(), 0,
                                meta.size() * sizeof(int32_t));
        for (int il = 0; il < cache_.n_layer; ++il) {
            snap.layers[(size_t)il].n_comp       = meta[(size_t)il * 2 + 0];
            snap.layers[(size_t)il].n_index_comp = meta[(size_t)il * 2 + 1];
            const auto & got = snap.layers[(size_t)il];
            const auto & live = cache_.layers[(size_t)il];
            const int comp_rows = got.comp_kv ? (int)got.comp_kv->ne[1] : 0;
            const int index_rows =
                got.index_comp_kv ? (int)got.index_comp_kv->ne[1] : 0;
            const int comp_cap = live.comp_kv ? (int)live.comp_kv->ne[1] : 0;
            const int index_cap =
                live.index_comp_kv ? (int)live.index_comp_kv->ne[1] : 0;
            if (got.n_comp < 0 || got.n_comp > comp_rows ||
                got.n_comp > comp_cap ||
                got.n_index_comp < 0 ||
                got.n_index_comp > index_rows ||
                got.n_index_comp > index_cap) {
                snap.ctx = nullptr;
                snap.buf = nullptr;
                return false;
            }
        }
    }

    // Recover host-side state. Without logits, restore_and_generate_impl
    // rejects the slot outright.
    snapshot_logits_[slot].resize((size_t)ggml_nelements(snap.logits_snap));
    ggml_backend_tensor_get(snap.logits_snap, snapshot_logits_[slot].data(), 0,
                            snapshot_logits_[slot].size() * sizeof(float));
    if (snap.spec_feat_snap) {
        snapshot_spec_features_[slot].resize(
            (size_t)ggml_nelements(snap.spec_feat_snap));
        ggml_backend_tensor_get(snap.spec_feat_snap,
                                snapshot_spec_features_[slot].data(), 0,
                                snapshot_spec_features_[slot].size() * sizeof(float));
    } else {
        std::vector<float>().swap(snapshot_spec_features_[slot]);
    }

    snapshots_[slot] = std::move(snap);
    return true;
}

GenerateResult DeepSeek4Backend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    if (slot < 0 || slot >= PREFIX_SLOTS || !snapshot_used(slot) ||
        snapshot_logits_[slot].empty()) {
        result.fail(GenerateErrorCode::InvalidSnapshotSlot);
        return result;
    }

    const int snap_pos = snapshots_[slot].cur_pos;
    const int prompt_len = (int) req.prompt.size();
    if (prompt_len < snap_pos) {
        // Agent harnesses may edit or summarize history. A longer cached
        // prefix is not applicable; use the normal fresh-prefill path.
        std::fprintf(stderr,
                     "[pc] snapshot longer than prompt (snap=%d > prompt=%d) — "
                     "fresh prefill fallback\n",
                     snap_pos, prompt_len);
        return generate_impl(req, io);
    }

    auto t0 = Clock::now();
    if (!deepseek4_snapshot_restore(snapshots_[slot], cache_)) {
        result.fail(GenerateErrorCode::BackendSpecific,
                    "DeepSeek V4 snapshot restore failed");
        return result;
    }
    last_logits_ = snapshot_logits_[slot];
    spec_feat_window_ = snapshot_spec_features_[slot];

    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    const bool sampling_requires_ar = sampler_.needs_logit_processing();
    const int capture_spec_budget = std::min(
        ds4_spec_emit_budget(req), ds4_spec_context_budget(prompt_len));
    const bool eligible_without_gate = dspark_request_can_prepare(
        spec_enabled_, spec_drafter_ != nullptr, req.force_ar_decode,
        sampling_requires_ar, req.n_gen, capture_spec_budget);
    const bool profitability_allowed = !eligible_without_gate ||
        dspark_worker_scheduler().allow_spec_request();
    const bool prepare_spec = eligible_without_gate && profitability_allowed;
    if (!prepare_spec) {
        // An exact snapshot hit does not call do_prefill(), so clear restored
        // features here as well when this request cannot enter DSpark.
        spec_feat_window_.clear();
    }

    int committed = snap_pos;
    inline_snapshot_saved_ = false;  // do_prefill / the clone branch set it on save
    const int delta_tokens = std::max(0, prompt_len - snap_pos);
    describe_prefill(result, cfg_.prefill_mode, req.force_exact_prefill,
                     prepare_spec, req.force_ar_decode,
                     delta_tokens);
    if (prompt_len > snap_pos) {
        std::vector<int32_t> delta(
            req.prompt.begin() + snap_pos, req.prompt.end());
        committed = do_prefill(delta, out_io, snap_pos,
                               req.snap_pos, req.snap_slot,
                               prepare_spec,
                               req.force_exact_prefill);
        if (committed < 0) {
            result.fail(GenerateErrorCode::PrefillFailed);
            return result;
        }
    } else if (req.snap_slot >= 0 && req.snap_pos == snap_pos &&
               req.snap_slot != slot) {
        // Exact hit plus a request for another slot: clone the restored state.
        if (snapshot_save(req.snap_slot)) {
            inline_snapshot_saved_ = true;
            std::fprintf(stderr, "[snap] cloned slot=%d cur_pos=%d\n",
                         req.snap_slot, snap_pos);
        }
    }
    maybe_log_prefill_fingerprint(last_logits_, committed);
    result.prefill_s = elapsed_s(t0);
    result.snapshot_saved = inline_snapshot_saved_;  // #2: only true on a real save

    if (out_io.cancelled) {
        result.succeed();
        maybe_save_routing_stats();
        return result;
    }
    if (req.n_gen <= 0) {
        result.succeed();
        maybe_save_routing_stats();
        return result;
    }

    auto t1 = Clock::now();
    const int spec_budget = std::min(
        ds4_spec_emit_budget(req), ds4_spec_context_budget(committed));
    std::vector<int32_t> generated;
    generated.reserve((size_t) req.n_gen);
    float accept_rate = 0.0f;
    bool spec_ran = false;
    bool spec_terminal = false;
    bool spec_degenerate = false;
    std::string termination_reason;
    ProgressCycleDetector spec_progress(
        req.budget_hook.natural_close_token_ids, req.prompt,
        req.tool_region_open_ids, req.tool_region_close_ids);
    if (ds4_spec_should_run(req, spec_enabled_, spec_drafter_ != nullptr,
                            sampling_requires_ar, spec_budget, committed,
                            profitability_allowed)) {
        int seed = 0;
        float max_value = last_logits_[0];
        for (int i = 1; i < w_.n_vocab; ++i) {
            if (last_logits_[i] > max_value) {
                max_value = last_logits_[i];
                seed = i;
            }
        }

        generated.push_back(seed);
        out_io.emit(seed);
        spec_degenerate = spec_progress.observe(seed);
        if (!out_io.cancelled && !deepseek4_is_eos_tok(seed, w_) &&
            spec_budget > 1) {
            const int feat_row =
                spec_drafter_->n_target_layers * w_.n_embd;
            const int win_len = feat_row > 0
                ? (int) (spec_feat_window_.size() / feat_row)
                : 0;
            std::vector<int32_t> spec_tokens;
            spec_ran = true;
            if (!run_deepseek4_dspark_spec_decode(
                    backend_, cfg_.device.gpu, w_, cache_, *spec_drafter_,
                    committed, seed, spec_budget - 1,
                    win_len > 0 ? spec_feat_window_.data() : nullptr,
                    win_len, spec_tokens, &accept_rate,
                    spec_xdna_draft_compute_.get(),
                    [&out_io, &spec_progress, &spec_degenerate,
                     &termination_reason](int32_t token) {
                        if (out_io.cancelled) return false;
                        out_io.emit(token);
                        if (spec_progress.observe(token)) {
                            spec_degenerate = true;
                            termination_reason = spec_progress.reason_name();
                            std::fprintf(stderr,
                                         "[deepseek4] restore speculative "
                                         "progress watchdog fired: reason=%s "
                                         "period=%zu span=%zu tokens\n",
                                         spec_progress.reason_name(),
                                         spec_progress.cycle_period(),
                                         spec_progress.repeated_span());
                            return false;
                        }
                        return !out_io.cancelled;
                    })) {
                result.fail(GenerateErrorCode::DecodeFailed,
                            "DSpark speculative decode failed after restore");
                return result;
            }
            generated.insert(generated.end(), spec_tokens.begin(), spec_tokens.end());
        }
        spec_terminal = spec_degenerate || out_io.cancelled ||
                        (int) generated.size() >= req.n_gen ||
                        (!generated.empty() &&
                         deepseek4_is_eos_tok(generated.back(), w_));
    }

    bool forced_close = false;
    bool degenerate_close = spec_degenerate;
    if (!spec_terminal &&
        !do_decode(committed, req.n_gen, req.prompt, generated, out_io,
                   req.budget_hook, &forced_close, &degenerate_close,
                   &termination_reason,
                   req.force_greedy_next,
                   (int) generated.size(),
                   req.tool_region_open_ids, req.tool_region_close_ids,
                   req.token_mask)) {
        result.fail(GenerateErrorCode::DecodeFailed);
        return result;
    }
    result.succeed();
    result.tokens = std::move(generated);
    result.decode_s = elapsed_s(t1);
    result.budget_forced_close = forced_close;
    result.degenerate_decode_close = degenerate_close;
    result.termination_reason = termination_reason;
    result.accept_rate = accept_rate;
    result.spec_decode_ran = spec_ran;
    if (spec_ran) {
        dspark_worker_scheduler().note_request_result(accept_rate);
        std::fprintf(stderr,
                     "[deepseek4] DSpark restore decode: %zu tok in %.3fs "
                     "(%.1f tok/s) accept_rate=%.2f spec_budget=%d tail_ar=%d\n",
                     result.tokens.size(), result.decode_s,
                     result.decode_s > 0
                         ? result.tokens.size() / result.decode_s
                         : 0.0,
                     accept_rate, spec_budget, spec_terminal ? 0 : 1);
    }
    maybe_save_routing_stats();
    return result;
}

bool DeepSeek4Backend::resident_session_create(
        ContinuousBatchSessionId id,
        const GenerateRequest &request,
        const DaemonIO &io,
        int restore_slot,
        std::string *error) {
    if (error) error->clear();
    if (id == 0 || resident_sessions_.count(id) != 0) {
        if (error) *error = "invalid or duplicate resident session";
        return false;
    }
    if (request.n_gen < 0 ||
        request.prompt.size() > (size_t)cache_.max_ctx) {
        if (error) *error = "resident session request exceeds context";
        return false;
    }

    auto session = std::make_unique<ResidentSession>();
    session->request = request;
    session->restore_slot = restore_slot;
    // ModelBackend::generate_impl normally composes GenerateRequest::on_token
    // into DaemonIO before decode. Resident sessions bypass that wrapper, so
    // compose it here or streaming/cancellation silently disappears.
    session->io = io.with_token_callback(request.on_token);
    session->sampler = request.sampler;
    if (request.do_sample && session->sampler.seed != 0) {
        session->sampler_rng.seed(session->sampler.seed);
    }
    session->history = request.prompt;
    session->generated.reserve((size_t)request.n_gen);
    session->progress_cycle = ProgressCycleDetector(
        request.budget_hook.natural_close_token_ids, request.prompt,
        request.tool_region_open_ids, request.tool_region_close_ids);
    const bool sampling_requires_ar = session->sampler.needs_logit_processing();
    const int capture_spec_budget = std::min(
        ds4_spec_emit_budget(request),
        ds4_spec_context_budget((int)request.prompt.size()));
    session->spec_eligible =
        spec_xdna_draft_compute_ && spec_xdna_draft_compute_->healthy() &&
        dspark_request_can_prepare(
            spec_enabled_, spec_drafter_ != nullptr, request.force_ar_decode,
            sampling_requires_ar, request.n_gen, capture_spec_budget);
    if (!resident_cache_pool_.empty()) {
        session->cache = std::move(resident_cache_pool_.back());
        resident_cache_pool_.pop_back();
    } else {
        const int max_ctx = cfg_.max_ctx > 0 ? cfg_.max_ctx : 8192;
        if (!create_deepseek4_cache(backend_, w_, max_ctx, session->cache)) {
            if (error) *error = "failed to allocate resident KV cache";
            return false;
        }
    }
    session->cache.prefill_mode = cfg_.prefill_mode;

    if (restore_slot >= 0 && snapshot_used(restore_slot)) {
        if (!deepseek4_snapshot_restore(snapshots_[restore_slot],
                                        session->cache)) {
            recycle_resident_cache(session->cache);
            if (error) *error = "failed to restore resident KV snapshot";
            return false;
        }
        session->last_logits = snapshot_logits_[restore_slot];
        session->spec_feat_window = snapshot_spec_features_[restore_slot];
        session->prefilled = session->cache.cur_pos;
        session->restore_pos = session->prefilled;
        if (session->prefilled < 0 ||
            session->prefilled > (int)request.prompt.size() ||
            session->last_logits.empty()) {
            recycle_resident_cache(session->cache);
            if (error) *error = "resident snapshot does not match request";
            return false;
        }
    }

    if (session->prefilled == (int)request.prompt.size() &&
        request.n_gen > 0 &&
        !resident_sample_next(*session)) {
        if (error) *error = session->error;
        recycle_resident_cache(session->cache);
        return false;
    }
    if (session->pending_ready && session->spec_eligible) {
        std::string spec_error;
        if (!resident_submit_spec(*session, &spec_error) &&
            !spec_error.empty()) {
            std::fprintf(stderr,
                         "[ds4-resident-spec] initial submit failed: %s; "
                         "using target AR\n",
                         spec_error.c_str());
            session->spec_eligible = false;
        }
    }
    resident_sessions_.emplace(id, std::move(session));
    return true;
}

bool DeepSeek4Backend::resident_session_destroy(
        ContinuousBatchSessionId id) {
    auto it = resident_sessions_.find(id);
    if (it == resident_sessions_.end()) return false;
    recycle_resident_cache(it->second->cache);
    resident_sessions_.erase(it);
    return true;
}

bool DeepSeek4Backend::resident_session_cancel(
        ContinuousBatchSessionId id) {
    auto it = resident_sessions_.find(id);
    if (it == resident_sessions_.end() || it->second->terminal ||
        it->second->cancelled || it->second->failed)
        return false;
    it->second->spec_proposal.cancel();
    it->second->cancelled = true;
    it->second->terminal = true;
    it->second->pending_ready = false;
    return true;
}

bool DeepSeek4Backend::resident_session_decode_ready(
        ContinuousBatchSessionId id) const {
    auto it = resident_sessions_.find(id);
    return it != resident_sessions_.end() &&
           it->second->pending_ready &&
           !it->second->terminal &&
           !it->second->failed &&
           !it->second->cancelled;
}

ResidentBatchBackend::SessionStatus
DeepSeek4Backend::resident_session_status(
        ContinuousBatchSessionId id) const {
    ResidentBatchBackend::SessionStatus status;
    auto it = resident_sessions_.find(id);
    if (it == resident_sessions_.end()) {
        status.failed = true;
        return status;
    }
    const ResidentSession &session = *it->second;
    status.prefilled_tokens = session.prefilled;
    status.decode_ready = session.pending_ready &&
                          !session.terminal &&
                          !session.cancelled &&
                          !session.failed;
    status.terminal = session.terminal;
    status.cancelled = session.cancelled;
    status.failed = session.failed;
    return status;
}

GenerateResult DeepSeek4Backend::resident_session_result(
        ContinuousBatchSessionId id) const {
    GenerateResult result;
    auto it = resident_sessions_.find(id);
    if (it == resident_sessions_.end()) {
        result.fail(GenerateErrorCode::BackendSpecific,
                    "resident session not found");
        return result;
    }
    const ResidentSession &session = *it->second;
    if (session.failed) {
        result.fail(GenerateErrorCode::DecodeFailed, session.error);
    } else {
        result.succeed();
    }
    result.tokens = session.generated;
    result.prefill_s = session.prefill_s;
    result.decode_s = session.decode_s;
    describe_prefill(result, cfg_.prefill_mode,
                     session.request.force_exact_prefill,
                     session.spec_eligible || session.spec_ran,
                     session.request.force_ar_decode,
                     session.prefill_tokens);
    result.budget_forced_close = session.budget_forced_close;
    result.degenerate_decode_close = session.degenerate_decode_close;
    result.termination_reason = session.termination_reason;
    result.snapshot_saved = session.inline_snapshot_saved;
    result.spec_decode_ran = session.spec_ran;
    result.accept_rate = session.spec_offered > 0
        ? (float)session.spec_accepted / (float)session.spec_offered
        : 0.0f;
    result.spec_cycles = session.spec_cycles;
    result.spec_provider_age_s = session.spec_provider_age_s;
    result.spec_provider_block_s = session.spec_provider_block_s;
    result.spec_head_s = session.spec_head_s;
    result.spec_verify_s = session.spec_verify_s;
    return result;
}

bool DeepSeek4Backend::resident_session_snapshot(
        ContinuousBatchSessionId id, int slot) {
    auto it = resident_sessions_.find(id);
    if (it == resident_sessions_.end() || slot < 0) return false;
    ResidentSession &session = *it->second;
    swap_resident_state(session);
    bool saved = false;
    try {
        saved = snapshot_save(slot);
    } catch (...) {
        swap_resident_state(session);
        return false;
    }
    swap_resident_state(session);
    session.inline_snapshot_saved =
        session.inline_snapshot_saved || saved;
    return saved;
}

ContinuousBatchPrefillCompletion DeepSeek4Backend::prefill(
        ContinuousBatchSessionId id, int requested_tokens) {
    auto it = resident_sessions_.find(id);
    if (it == resident_sessions_.end() || requested_tokens <= 0) return {};
    ResidentSession &session = *it->second;
    if (session.failed || session.cancelled || session.terminal ||
        session.prefilled >= (int)session.request.prompt.size()) {
        return {};
    }

    const int old_pos = session.prefilled;
    const int count = std::min(
        requested_tokens,
        (int)session.request.prompt.size() - old_pos);
    std::vector<int32_t> suffix(
        session.request.prompt.begin() + old_pos,
        session.request.prompt.begin() + old_pos + count);

    swap_resident_state(session);
    const auto t0 = Clock::now();
    int committed = -1;
    const bool shadow_spec_capture =
        session.spec_eligible && !session.request.force_exact_prefill &&
        prefill_attention_mode_is_approximate(cfg_.prefill_mode) &&
        env_nonnegative_long(
            "DFLASH_DS4_SPEC_SHADOW_SUFFIX_ROWS", 4) > 0;
    try {
        committed = do_prefill(suffix, session.io, old_pos,
                               session.request.snap_pos,
                               session.request.snap_slot,
                               session.spec_eligible && !shadow_spec_capture,
                               session.request.force_exact_prefill);
    } catch (...) {
        swap_resident_state(session);
        session.failed = true;
        session.error = "resident prefill threw an exception";
        return {};
    }
    session.prefill_s += elapsed_s(t0);
    swap_resident_state(session);

    if (committed < old_pos) {
        session.failed = true;
        session.error = "resident prefill failed";
        return {};
    }
    const int consumed = committed - old_pos;
    session.prefill_tokens += consumed;
    session.prefilled = committed;
    if (session.spec_eligible && shadow_spec_capture &&
        session.prefilled == (int)session.request.prompt.size()) {
        const auto shadow_t0 = Clock::now();
        std::string shadow_error;
        if (!rebuild_resident_spec_features(session, &shadow_error)) {
            std::fprintf(stderr,
                         "[ds4-resident-spec] shadow capture failed: %s; "
                         "using target AR\n",
                         shadow_error.c_str());
            session.spec_eligible = false;
            session.spec_feat_window.clear();
        }
        session.prefill_s += elapsed_s(shadow_t0);
    }
    if (session.io.cancelled) {
        session.cancelled = true;
        session.terminal = true;
    } else if (session.prefilled == (int)session.request.prompt.size() &&
               session.request.n_gen > 0 &&
               !resident_sample_next(session)) {
        return {};
    }
    if (session.pending_ready && session.spec_eligible &&
        !session.spec_proposal.pending()) {
        std::string spec_error;
        if (!resident_submit_spec(session, &spec_error) &&
            !spec_error.empty()) {
            std::fprintf(stderr,
                         "[ds4-resident-spec] prefill submit failed: %s; "
                         "using target AR\n",
                         spec_error.c_str());
            session.spec_eligible = false;
        }
    }
    return {consumed > 0, consumed};
}

std::vector<ContinuousBatchDecodeCompletion>
DeepSeek4Backend::decode_batch(
        const std::vector<ContinuousBatchSessionId> &sessions) {
    std::vector<ContinuousBatchDecodeCompletion> result;
    result.reserve(sessions.size());

    auto fail_all = [&](const char *detail) {
        result.clear();
        for (ContinuousBatchSessionId member : sessions) {
            auto found = resident_sessions_.find(member);
            if (found != resident_sessions_.end() &&
                !found->second->cancelled) {
                found->second->failed = true;
                found->second->terminal = true;
                found->second->pending_ready = false;
                if (found->second->error.empty())
                    found->second->error = detail;
            }
            result.push_back({member, false, false});
        }
    };

    for (ContinuousBatchSessionId id : sessions) {
        auto it = resident_sessions_.find(id);
        if (it == resident_sessions_.end() ||
            !it->second->pending_ready || it->second->failed ||
            it->second->cancelled || it->second->terminal) {
            fail_all("invalid resident decode batch member");
            return result;
        }
    }

    // Fill the provider queue before collecting any job.  In steady state most
    // sessions already own a proposal submitted after their previous verify;
    // this pass covers newly-ready and target-only handoff sessions.  The XDNA
    // worker serializes NPU execution while the loop below performs GPU target
    // work for an earlier session.
    for (ContinuousBatchSessionId id : sessions) {
        ResidentSession &session = *resident_sessions_.find(id)->second;
        if (!session.spec_eligible || session.spec_proposal.pending()) continue;
        std::string spec_error;
        if (!resident_submit_spec(session, &spec_error) &&
            !spec_error.empty()) {
            std::fprintf(stderr,
                         "[ds4-resident-spec] submit failed session=%" PRIu64
                         ": %s; using target AR\n",
                         (uint64_t)id, spec_error.c_str());
            session.spec_eligible = false;
        }
    }

    for (ContinuousBatchSessionId id : sessions) {
        auto it = resident_sessions_.find(id);
        ResidentSession &session = *it->second;

        if (session.spec_proposal.pending()) {
            std::vector<int32_t> committed_tokens;
            std::vector<float> frontier_logits;
            int32_t next_token = -1;
            int offered = 0;
            int accepted = 0;
            DeepSeek4DSparkResidentTiming spec_timing;
            std::string spec_error;
            const auto finish_t0 = Clock::now();
            if (!deepseek4_dspark_resident_finish(
                    backend_, cfg_.device.gpu, w_, session.cache,
                    *spec_drafter_, session.spec_feat_window,
                    session.spec_proposal, committed_tokens, next_token,
                    frontier_logits, offered, accepted, spec_timing,
                    &spec_error)) {
                session.failed = true;
                session.error = spec_error.empty()
                    ? "resident DSpark cycle failed" : spec_error;
                fail_all("resident DSpark cycle failed");
                return result;
            }
            session.decode_s += elapsed_s(finish_t0);
            session.spec_ran = true;
            session.spec_offered += offered;
            session.spec_accepted += accepted;
            ++session.spec_cycles;
            session.spec_provider_age_s += spec_timing.provider_age_s;
            session.spec_provider_block_s += spec_timing.provider_block_s;
            session.spec_head_s += spec_timing.head_s;
            session.spec_verify_s += spec_timing.verify_s;
            // A partial resident block is both a numerical-risk signal and a
            // measured loss on this topology. resident_finish rolls it back
            // completely; remain on the ordinary target graph for the request.
            if (offered > 0 && accepted < offered) {
                session.spec_eligible = false;
            }

            // Confidence admission can decline before touching target state.
            // Disable speculation for this request and execute the already-
            // pending token through the ordinary resident AR path below.
            if (committed_tokens.empty()) {
                session.spec_eligible = false;
            } else {
                session.pending_ready = false;
                session.pending_forced_close_token = false;
                int completed_tokens = 0;
                for (int32_t token : committed_tokens) {
                    session.generated.push_back(token);
                    session.history.push_back(token);
                    session.io.emit(token);
                    ++completed_tokens;
                    const bool cycle_detected =
                        session.progress_cycle.observe(token);
                    if (session.io.cancelled) {
                        session.cancelled = true;
                        session.terminal = true;
                        break;
                    }
                    if (cycle_detected) {
                        session.degenerate_decode_close = true;
                        session.termination_reason =
                            session.progress_cycle.reason_name();
                        session.terminal = true;
                        std::fprintf(
                            stderr,
                            "[deepseek4] resident speculative progress "
                            "watchdog fired: reason=%s period=%zu span=%zu "
                            "tokens\n",
                            session.progress_cycle.reason_name(),
                            session.progress_cycle.cycle_period(),
                            session.progress_cycle.repeated_span());
                        break;
                    }
                    if (deepseek4_is_eos_tok(token, w_) ||
                        (int)session.generated.size() >=
                            session.request.n_gen) {
                        session.terminal = true;
                        break;
                    }
                }
                if (session.terminal) {
                    result.push_back({id, true, true, completed_tokens});
                    continue;
                }

                session.last_logits = std::move(frontier_logits);
                if (!resident_sample_next(session)) {
                    fail_all("resident speculative sampling failed");
                    return result;
                }
                if (!session.pending_forced_close_token &&
                    session.pending_token != next_token) {
                    session.failed = true;
                    session.error =
                        "resident DSpark bonus disagrees with greedy frontier";
                    fail_all("resident DSpark frontier mismatch");
                    return result;
                }

                // Refill this session's proposal behind jobs already queued
                // for other sessions. Its NPU work can run while their GPU
                // verifiers execute, and remains owned by this session.
                std::string submit_error;
                if (!resident_submit_spec(session, &submit_error) &&
                    !submit_error.empty()) {
                    std::fprintf(
                        stderr,
                        "[ds4-resident-spec] refill failed session=%" PRIu64
                        ": %s; using target AR\n",
                        (uint64_t)id, submit_error.c_str());
                    session.spec_eligible = false;
                }
                result.push_back({id, true, false, completed_tokens});
                continue;
            }
        }

        const int32_t token = session.pending_token;
        const bool completed_forced_close =
            session.pending_forced_close_token &&
            session.forced_close_index ==
                session.request.budget_hook.close_token_ids.size() &&
            session.forced_close_index > 0;
        session.pending_ready = false;
        session.pending_forced_close_token = false;
        session.generated.push_back(token);
        session.history.push_back(token);
        session.io.emit(token);
        const bool cycle_detected = session.progress_cycle.observe(token);
        if (completed_forced_close) session.progress_cycle.begin_visible();
        if (session.io.cancelled) {
            session.cancelled = true;
            session.terminal = true;
            result.push_back({id, true, true});
            continue;
        }
        if (cycle_detected) {
            session.degenerate_decode_close = true;
            session.termination_reason = session.progress_cycle.reason_name();
            session.terminal = true;
            std::fprintf(stderr,
                         "[deepseek4] resident progress watchdog fired: "
                         "reason=%s period=%zu span=%zu tokens\n",
                         session.progress_cycle.reason_name(),
                         session.progress_cycle.cycle_period(),
                         session.progress_cycle.repeated_span());
            result.push_back({id, true, true});
            continue;
        }

        const bool terminal =
            deepseek4_is_eos_tok(token, w_) ||
            (int)session.generated.size() >= session.request.n_gen;
        if (terminal) {
            session.terminal = true;
            result.push_back({id, true, true});
            continue;
        }

        std::vector<float> embed((size_t)w_.n_embd);
        w_.embedder.embed(&token, 1, embed.data());
        std::vector<float> logits;
        const auto t0 = Clock::now();
        const int pos = session.cache.cur_pos;
        bool ok = deepseek4_step(
            backend_, cfg_.device.gpu, w_, session.cache,
            embed.data(), 1, pos, logits,
            moe_hybrid_.get(), &token,
            moe_hybrid_ ? &stream_engine_ : nullptr,
            nullptr, routing_stats_.get(), nullptr);
        session.decode_s += elapsed_s(t0);
        if (!ok) {
            session.failed = true;
            session.error = "resident decode step failed";
            fail_all("resident decode batch failed");
            return result;
        }
        session.last_logits = std::move(logits);
        if (!resident_sample_next(session)) {
            fail_all("resident decode sampling failed");
            return result;
        }
        if (session.spec_eligible) {
            std::string spec_error;
            if (!resident_submit_spec(session, &spec_error) &&
                !spec_error.empty()) {
                std::fprintf(stderr,
                             "[ds4-resident-spec] AR handoff submit failed "
                             "session=%" PRIu64 ": %s; disabling resident spec\n",
                             (uint64_t)id, spec_error.c_str());
                session.spec_eligible = false;
            }
        }
        result.push_back({id, true, false});
    }
    return result;
}

ContinuousBatchMixedCompletion DeepSeek4Backend::execute_mixed(
        ContinuousBatchSessionId prefill_session,
        int requested_tokens,
        const std::vector<ContinuousBatchSessionId> &decode_sessions) {
    ContinuousBatchMixedCompletion mixed;
    mixed.prefill = prefill(prefill_session, requested_tokens);
    if (!mixed.prefill.ok) {
        mixed.decode.reserve(decode_sessions.size());
        for (ContinuousBatchSessionId id : decode_sessions)
            mixed.decode.push_back({id, false, false});
        return mixed;
    }
    mixed.decode = decode_batch(decode_sessions);
    const bool decode_ok = std::all_of(
        mixed.decode.begin(), mixed.decode.end(),
        [](const ContinuousBatchDecodeCompletion &row) { return row.ok; });
    if (!decode_ok) {
        auto it = resident_sessions_.find(prefill_session);
        if (it != resident_sessions_.end()) {
            it->second->failed = true;
            it->second->terminal = true;
            it->second->pending_ready = false;
            it->second->error =
                "mixed prefill/decode submission failed after prefill";
        }
        mixed.prefill = {};
    }
    return mixed;
}

void DeepSeek4Backend::maybe_save_routing_stats() {
    if (!routing_stats_ || routing_stats_out_path_.empty()) return;
    std::string err;
    if (!routing_stats_->save_csv(routing_stats_out_path_, &err)) {
        std::fprintf(stderr, "[deepseek4] failed to save routing stats %s: %s\n",
                     routing_stats_out_path_.c_str(), err.c_str());
    }
}

void DeepSeek4Backend::shutdown() {
    maybe_save_routing_stats();
    release_spec_drafter();
    free_resident_sessions();
    for (int i = 0; i < PREFIX_SLOTS; i++) {
        free_deepseek4_snapshot(snapshots_[i]);
    }
    free_deepseek4_cache(cache_);
    stream_engine_.destroy();
    moe_hybrid_.reset();
    routing_stats_.reset();
    routing_stats_out_path_.clear();
    moe_placement_ = {};
    free_deepseek4_weights(w_);
    if (snap_backend_) { ggml_backend_free(snap_backend_); snap_backend_ = nullptr; }
    if (backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
}

}  // namespace dflash::common
