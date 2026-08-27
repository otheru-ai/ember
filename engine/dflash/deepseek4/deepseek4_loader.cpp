// Loads DeepSeek V4 Flash from a GGUF file.
//
// Tensor naming follows the ds4 GGUF conversion:
//   token_embd.weight, output_norm.weight, output.weight,
//   output_hc_base.weight, output_hc_fn.weight, output_hc_scale.weight
//   blk.<i>.attn_norm.weight, blk.<i>.attn_q_a.weight, attn_q_a_norm,
//   attn_q_b, attn_kv, attn_kv_a_norm, attn_sinks, attn_output_a, attn_output_b,
//   attn_compressor_{ape,kv,gate,norm}, indexer.{attn_q_b, proj},
//   indexer_compressor_{ape,kv,gate,norm},
//   hc_attn_fn, hc_attn_scale, hc_attn_base,
//   ffn_norm, ffn_gate_inp, exp_probs_b (bias), ffn_gate_tid2eid,
//   ffn_gate_exps, ffn_up_exps, ffn_down_exps,
//   ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp,
//   hc_ffn_fn, hc_ffn_scale, hc_ffn_base

#include "deepseek4_internal.h"
#include "common/errors.h"
#include "common/gguf_bounds.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_hybrid_types.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
extern "C" bool ggml_backend_cuda_buffer_is_managed(ggml_backend_buffer_t buffer);

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dflash::common {

namespace {

struct DS4Mmap {
    void *  addr = nullptr;
    size_t  len  = 0;
    int     fd   = -1;

    bool open_ro(const std::string & path, std::string & err) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open: " + path + " " + strerror(errno); return false; }
        struct stat st;
        if (fstat(fd, &st) < 0) { err = "fstat"; ::close(fd); fd = -1; return false; }
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) { err = "mmap"; addr = nullptr; ::close(fd); fd = -1; return false; }
        return true;
    }
    void close_map() {
        if (addr) { ::munmap(addr, len); addr = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }
};

uint32_t get_u32_or(gguf_context * g, const char * key, uint32_t def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    const enum gguf_type type = gguf_get_kv_type(g, id);
    if (type == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return def;
        const void * data = gguf_get_arr_data(g, id);
        if (!data) return def;
        const enum gguf_type arr_type = gguf_get_arr_type(g, id);
        if (arr_type == GGUF_TYPE_UINT32)
            return static_cast<const uint32_t *>(data)[0];
        if (arr_type == GGUF_TYPE_INT32) {
            const int32_t value = static_cast<const int32_t *>(data)[0];
            return value >= 0 ? static_cast<uint32_t>(value) : def;
        }
        return def;
    }
    if (type == GGUF_TYPE_UINT32) return gguf_get_val_u32(g, id);
    if (type == GGUF_TYPE_INT32) {
        const int32_t value = gguf_get_val_i32(g, id);
        return value >= 0 ? static_cast<uint32_t>(value) : def;
    }
    if (type == GGUF_TYPE_UINT64) {
        const uint64_t value = gguf_get_val_u64(g, id);
        return value <= UINT32_MAX ? static_cast<uint32_t>(value) : def;
    }
    if (type == GGUF_TYPE_INT64) {
        const int64_t value = gguf_get_val_i64(g, id);
        return value >= 0 && value <= UINT32_MAX
            ? static_cast<uint32_t>(value) : def;
    }
    return def;
}

uint64_t get_u64_or(gguf_context * g, const char * key, uint64_t def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    const enum gguf_type type = gguf_get_kv_type(g, id);
    if (type == GGUF_TYPE_UINT32) return gguf_get_val_u32(g, id);
    if (type == GGUF_TYPE_UINT64) return gguf_get_val_u64(g, id);
    if (type == GGUF_TYPE_INT32) {
        const int32_t value = gguf_get_val_i32(g, id);
        return value >= 0 ? static_cast<uint64_t>(value) : def;
    }
    if (type == GGUF_TYPE_INT64) {
        const int64_t value = gguf_get_val_i64(g, id);
        return value >= 0 ? static_cast<uint64_t>(value) : def;
    }
    return def;
}

float get_f32_or(gguf_context * g, const char * key, float def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    const enum gguf_type type = gguf_get_kv_type(g, id);
    if (type == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return def;
        const void * data = gguf_get_arr_data(g, id);
        if (!data) return def;
        const enum gguf_type arr_type = gguf_get_arr_type(g, id);
        if (arr_type == GGUF_TYPE_FLOAT32)
            return static_cast<const float *>(data)[0];
        if (arr_type == GGUF_TYPE_FLOAT64)
            return static_cast<float>(static_cast<const double *>(data)[0]);
        return def;
    }
    if (type == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(g, id);
    if (type == GGUF_TYPE_FLOAT64)
        return static_cast<float>(gguf_get_val_f64(g, id));
    return def;
}

bool get_u32_arr(gguf_context * g, const char * key, std::vector<uint32_t> & out,
                 std::string * err = nullptr) {
    out.clear();
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return true;
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) {
        if (err) *err = std::string(key) + " must be an array";
        return false;
    }
    const enum gguf_type arr_type = gguf_get_arr_type(g, id);
    if (arr_type != GGUF_TYPE_INT32 && arr_type != GGUF_TYPE_UINT32) {
        if (err) {
            *err = std::string(key) + " array element type must be i32 or u32";
        }
        return false;
    }

    const size_t n = gguf_get_arr_n(g, id);
    const void * raw = gguf_get_arr_data(g, id);
    out.resize(n);
    if (arr_type == GGUF_TYPE_INT32) {
        const int32_t * vals = static_cast<const int32_t *>(raw);
        for (size_t i = 0; i < n; ++i) {
            if (vals[i] < 0) {
                if (err) *err = std::string(key) + " array values must be non-negative";
                out.clear();
                return false;
            }
            out[i] = (uint32_t)vals[i];
        }
    } else {
        const uint32_t * vals = static_cast<const uint32_t *>(raw);
        out.assign(vals, vals + n);
    }
    return true;
}

ggml_tensor * find_tensor(ggml_context * ctx, const char * name) {
    return ggml_get_tensor(ctx, name);
}

static size_t align_up_size(size_t x, size_t a) {
    if (a == 0) return x;
    const size_t r = x % a;
    if (r == 0) return x;
    const size_t add = a - r;
    return x > SIZE_MAX - add ? SIZE_MAX : x + add;
}

static bool parse_block_tensor_name(const char * name, int & layer_id) {
    const char prefix[] = "blk.";
    const size_t prefix_len = sizeof(prefix) - 1;
    if (std::strncmp(name, prefix, prefix_len) != 0) return false;
    const char * p = name + prefix_len;
    if (*p < '0' || *p > '9') return false;
    char * end = nullptr;
    const long v = std::strtol(p, &end, 10);
    if (!end || *end != '.' || v < 0 || v > INT32_MAX) return false;
    layer_id = (int)v;
    return true;
}

static bool is_expert_tensor(const char * name) {
    return std::strstr(name, "ffn_gate_exps") != nullptr ||
           std::strstr(name, "ffn_up_exps") != nullptr ||
           std::strstr(name, "ffn_down_exps") != nullptr;
}

static bool should_keep_ds4_tensor(const char * name,
                                   const TargetLoadPlan & plan) {
    int layer_id = -1;
    if (plan.expert_metadata_only) {
        return parse_block_tensor_name(name, layer_id) &&
               layer_id >= plan.layer_begin &&
               layer_id < plan.layer_end &&
               is_expert_tensor(name);
    }

    // Global tensors
    if (std::strcmp(name, "token_embd.weight") == 0 ||
        std::strcmp(name, "output_norm.weight") == 0 ||
        std::strcmp(name, "output.weight") == 0 ||
        std::strcmp(name, "output_hc_base.weight") == 0 ||
        std::strcmp(name, "output_hc_fn.weight") == 0 ||
        std::strcmp(name, "output_hc_scale.weight") == 0) {
        return plan.load_output;
    }

    if (!parse_block_tensor_name(name, layer_id)) return false;
    return layer_id >= plan.layer_begin && layer_id < plan.layer_end;
}

static bool should_upload_ds4_tensor(const char * name,
                                     const TargetLoadPlan & plan) {
    if (!should_keep_ds4_tensor(name, plan)) return false;
    if (plan.expert_metadata_only) return false;
    // token_embd stays on CPU for embedding lookup
    if (std::strcmp(name, "token_embd.weight") == 0) return false;
    return !(plan.skip_expert_tensors && is_expert_tensor(name));
}

struct DS4TensorAlloc {
    ggml_tensor * tensor = nullptr;
    size_t tensor_offset = 0;
    size_t file_offset = 0;
    size_t file_size = 0;
    size_t buffer_offset = 0;
    bool upload_to_backend = true;
};

}  // namespace

// ─── Compute per-layer compression ratios (matches ds4.c logic) ─────────
static std::vector<uint32_t> compute_compress_ratios(int n_layer) {
    std::vector<uint32_t> ratios(n_layer, 0);
    for (int il = 0; il < n_layer; il++) {
        if (il < 2) {
            ratios[il] = 0;  // First 2 layers: no compression
        } else if ((il & 1) == 0) {
            ratios[il] = 4;  // Even layers ≥2: ratio 4
        } else {
            ratios[il] = 128;  // Odd layers ≥2: ratio 128
        }
    }
    return ratios;
}

bool load_deepseek4_gguf(const std::string & path,
                          ggml_backend_t backend,
                          DeepSeek4Weights & out) {
    TargetLoadPlan plan;
    return load_deepseek4_gguf_partial(path, backend, plan, out);
}

bool load_deepseek4_gguf_partial(const std::string & path,
                                  ggml_backend_t backend,
                                  const TargetLoadPlan & plan_in,
                                  DeepSeek4Weights & out) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) { set_last_error("gguf_init failed: " + path); return false; }

    // Validate arch
    {
        int64_t aid = gguf_find_key(gctx, "general.architecture");
        if (aid < 0) {
            set_last_error("missing general.architecture");
            gguf_free(gctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
        if (gguf_get_kv_type(gctx, aid) != GGUF_TYPE_STRING) {
            set_last_error("general.architecture must be a string");
            gguf_free(gctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, aid);
        if (!arch || std::strcmp(arch, "deepseek4") != 0) {
            set_last_error(std::string("unexpected arch: ") +
                           (arch ? arch : "(null)") +
                           " (expected deepseek4)");
            gguf_free(gctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
    }

    static const char * kRequiredU32Keys[] = {
        "deepseek4.block_count",
        "deepseek4.embedding_length",
        "deepseek4.vocab_size",
        "deepseek4.attention.head_count",
        "deepseek4.attention.head_count_kv",
        "deepseek4.attention.key_length",
        "deepseek4.attention.value_length",
        "deepseek4.rope.dimension_count",
        "deepseek4.attention.q_lora_rank",
        "deepseek4.attention.output_lora_rank",
        "deepseek4.attention.output_group_count",
        "deepseek4.expert_count",
        "deepseek4.expert_used_count",
        "deepseek4.expert_shared_count",
        "deepseek4.expert_feed_forward_length",
        "deepseek4.hash_layer_count",
        "deepseek4.attention.sliding_window",
        "deepseek4.attention.indexer.head_count",
        "deepseek4.attention.indexer.key_length",
        "deepseek4.attention.indexer.top_k",
        "deepseek4.hyper_connection.count",
        "deepseek4.hyper_connection.sinkhorn_iterations",
    };
    for (const char * key : kRequiredU32Keys) {
        const int64_t id = gguf_find_key(gctx, key);
        if (id < 0 || gguf_get_kv_type(gctx, id) != GGUF_TYPE_UINT32) {
            set_last_error(std::string("missing or non-u32 required key: ") + key);
            gguf_free(gctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
    }

    static const char * kRequiredF32Keys[] = {
        "deepseek4.rope.freq_base",
        "deepseek4.attention.compress_rope_freq_base",
        "deepseek4.attention.layer_norm_rms_epsilon",
        "deepseek4.hyper_connection.epsilon",
        "deepseek4.expert_weights_scale",
    };
    for (const char * key : kRequiredF32Keys) {
        const int64_t id = gguf_find_key(gctx, key);
        if (id < 0 || gguf_get_kv_type(gctx, id) != GGUF_TYPE_FLOAT32) {
            set_last_error(std::string("missing or non-f32 required key: ") + key);
            gguf_free(gctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
    }
    const int64_t rope_ctx_id =
        gguf_find_key(gctx, "deepseek4.rope.scaling.original_context_length");
    // The production affine converter omits the YaRN scaling keys. Dwarfstar
    // and Ember historically use the architecture defaults in that case, so
    // validate a present value's type without requiring optional metadata.
    if (rope_ctx_id >= 0 &&
        (gguf_get_kv_type(gctx, rope_ctx_id) != GGUF_TYPE_UINT32 &&
         gguf_get_kv_type(gctx, rope_ctx_id) != GGUF_TYPE_UINT64)) {
        set_last_error(
            "invalid deepseek4.rope.scaling.original_context_length");
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    // ── Read hyperparameters ────────────────────────────────────────────
    const uint32_t n_layer        = get_u32_or(gctx, "deepseek4.block_count", 43);
    const uint32_t n_embd         = get_u32_or(gctx, "deepseek4.embedding_length", 4096);
    const uint32_t n_vocab        = get_u32_or(gctx, "deepseek4.vocab_size", 129280);
    const uint32_t n_head         = get_u32_or(gctx, "deepseek4.attention.head_count", 64);
    const uint32_t n_head_kv      = get_u32_or(gctx, "deepseek4.attention.head_count_kv", 1);
    const uint32_t head_dim       = get_u32_or(gctx, "deepseek4.attention.key_length", 512);
    const uint32_t n_rot          = get_u32_or(gctx, "deepseek4.rope.dimension_count", 64);
    const uint32_t n_lora_q       = get_u32_or(gctx, "deepseek4.attention.q_lora_rank", 1024);
    const uint32_t n_lora_o       = get_u32_or(gctx, "deepseek4.attention.output_lora_rank", 1024);
    const uint32_t n_out_group    = get_u32_or(gctx, "deepseek4.attention.output_group_count", 8);
    const uint32_t n_expert       = get_u32_or(gctx, "deepseek4.expert_count", 256);
    const uint32_t n_expert_used  = get_u32_or(gctx, "deepseek4.expert_used_count", 6);
    const uint32_t n_expert_shared = get_u32_or(gctx, "deepseek4.expert_shared_count", 1);
    const uint32_t n_ff_exp       = get_u32_or(gctx, "deepseek4.expert_feed_forward_length", 2048);
    const uint32_t n_hash_layer   = get_u32_or(gctx, "deepseek4.hash_layer_count", 3);
    const uint32_t n_swa          = get_u32_or(gctx, "deepseek4.attention.sliding_window", 128);
    const uint32_t n_indexer_head = get_u32_or(gctx, "deepseek4.attention.indexer.head_count", 64);
    const uint32_t n_indexer_head_dim = get_u32_or(gctx, "deepseek4.attention.indexer.key_length", 128);
    const uint32_t n_indexer_top_k = get_u32_or(gctx, "deepseek4.attention.indexer.top_k", 512);
    const uint32_t n_hc           = get_u32_or(gctx, "deepseek4.hyper_connection.count", 4);
    const uint32_t n_hc_sinkhorn  = get_u32_or(gctx, "deepseek4.hyper_connection.sinkhorn_iterations", 20);
    const uint32_t n_value_dim    = get_u32_or(
        gctx, "deepseek4.attention.value_length", head_dim);

    // RoPE parameters
    const float rope_freq_base    = get_f32_or(gctx, "deepseek4.rope.freq_base", 10000.0f);
    const float rope_scale_factor = get_f32_or(gctx, "deepseek4.rope.scaling.factor", 16.0f);
    const float rope_yarn_beta_fast = get_f32_or(gctx, "deepseek4.rope.scaling.yarn_beta_fast", 32.0f);
    const float rope_yarn_beta_slow = get_f32_or(gctx, "deepseek4.rope.scaling.yarn_beta_slow", 1.0f);
    const float compress_rope_freq_base = get_f32_or(gctx, "deepseek4.attention.compress_rope_freq_base", 160000.0f);
    const uint64_t rope_orig_ctx  = get_u64_or(gctx, "deepseek4.rope.scaling.original_context_length", 65536);

    // Other parameters
    const float rms_eps           = get_f32_or(gctx, "deepseek4.attention.layer_norm_rms_epsilon", 1e-6f);
    const float hc_eps            = get_f32_or(gctx, "deepseek4.hyper_connection.epsilon", 1e-6f);
    const float expert_weight_scale = get_f32_or(gctx, "deepseek4.expert_weights_scale", 1.5f);
    const float swiglu_clamp      = get_f32_or(gctx, "deepseek4.swiglu_clamp_exp", 10.0f);

    const int64_t expert_norm_id =
        gguf_find_key(gctx, "deepseek4.expert_weights_norm");
    if (expert_norm_id < 0 ||
        gguf_get_kv_type(gctx, expert_norm_id) != GGUF_TYPE_BOOL ||
        !gguf_get_val_bool(gctx, expert_norm_id)) {
        set_last_error("deepseek4.expert_weights_norm must be true");
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    const int64_t clamp_id =
        gguf_find_key(gctx, "deepseek4.swiglu_clamp_exp");
    if (clamp_id < 0 ||
        gguf_get_kv_type(gctx, clamp_id) != GGUF_TYPE_ARRAY ||
        (gguf_get_arr_type(gctx, clamp_id) != GGUF_TYPE_FLOAT32 &&
         gguf_get_arr_type(gctx, clamp_id) != GGUF_TYPE_FLOAT64) ||
        gguf_get_arr_n(gctx, clamp_id) < n_layer) {
        set_last_error(
            "deepseek4.swiglu_clamp_exp must be a full per-layer float array");
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    const void * clamp_data = gguf_get_arr_data(gctx, clamp_id);
    if (!clamp_data) {
        set_last_error("deepseek4.swiglu_clamp_exp data is missing");
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    for (uint32_t il = 0; il < n_layer; ++il) {
        const float value =
            gguf_get_arr_type(gctx, clamp_id) == GGUF_TYPE_FLOAT32
                ? static_cast<const float *>(clamp_data)[il]
                : static_cast<float>(
                    static_cast<const double *>(clamp_data)[il]);
        if (std::isfinite(value) && std::fabs(value - 10.0f) <= 1.0e-5f)
            continue;
        set_last_error("unsupported swiglu clamp at layer " +
                       std::to_string(il));
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    // This backend implements the DeepSeek V4 Flash graph.  Its tensor
    // reshapes and fused kernels encode this architecture's fixed semantic
    // dimensions; accepting a different or overflowed shape can produce
    // plausible-looking but incorrect logits or invalid tensor views.  Match
    // Dwarfstar's load-time validation and reject it before allocation.
    struct ExpectedU32 {
        const char * name;
        uint32_t got;
        uint32_t expected;
    };
    const ExpectedU32 expected_shape[] = {
        {"block_count", n_layer, 43},
        {"embedding_length", n_embd, 4096},
        {"vocab_size", n_vocab, 129280},
        {"attention.head_count", n_head, 64},
        {"attention.head_count_kv", n_head_kv, 1},
        {"attention.key_length", head_dim, 512},
        {"attention.value_length", n_value_dim, 512},
        {"rope.dimension_count", n_rot, 64},
        {"attention.q_lora_rank", n_lora_q, 1024},
        {"attention.output_lora_rank", n_lora_o, 1024},
        {"attention.output_group_count", n_out_group, 8},
        {"expert_count", n_expert, 256},
        {"expert_used_count", n_expert_used, 6},
        {"expert_shared_count", n_expert_shared, 1},
        {"expert_feed_forward_length", n_ff_exp, 2048},
        {"hash_layer_count", n_hash_layer, 3},
        {"attention.sliding_window", n_swa, 128},
        {"attention.indexer.head_count", n_indexer_head, 64},
        {"attention.indexer.key_length", n_indexer_head_dim, 128},
        {"attention.indexer.top_k", n_indexer_top_k, 512},
        {"hyper_connection.count", n_hc, 4},
        {"hyper_connection.sinkhorn_iterations", n_hc_sinkhorn, 20},
    };
    for (const auto & item : expected_shape) {
        if (item.got == item.expected) continue;
        set_last_error(std::string("unsupported DeepSeek V4 Flash shape: ") +
                       item.name + "=" + std::to_string(item.got) +
                       " (expected " + std::to_string(item.expected) + ")");
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    struct ExpectedF32 {
        const char * name;
        float got;
        float expected;
    };
    const ExpectedF32 expected_scalars[] = {
        {"rope.freq_base", rope_freq_base, 10000.0f},
        {"rope.scaling.factor", rope_scale_factor, 16.0f},
        {"rope.scaling.yarn_beta_fast", rope_yarn_beta_fast, 32.0f},
        {"rope.scaling.yarn_beta_slow", rope_yarn_beta_slow, 1.0f},
        {"attention.compress_rope_freq_base", compress_rope_freq_base, 160000.0f},
        {"attention.layer_norm_rms_epsilon", rms_eps, 1.0e-6f},
        {"hyper_connection.epsilon", hc_eps, 1.0e-6f},
        {"expert_weights_scale", expert_weight_scale, 1.5f},
    };
    for (const auto & item : expected_scalars) {
        const float scale = std::max(1.0f, std::fabs(item.expected));
        if (std::isfinite(item.got) &&
            std::fabs(item.got - item.expected) <= scale * 1.0e-6f) {
            continue;
        }
        set_last_error(std::string("unsupported DeepSeek V4 Flash scalar: ") +
                       item.name + "=" + std::to_string(item.got));
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    if (rope_orig_ctx != 65536) {
        set_last_error("unsupported DeepSeek V4 Flash original context length: " +
                       std::to_string(rope_orig_ctx));
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    // Compression ratios from metadata (or compute default)
    std::vector<uint32_t> compress_ratios_meta;
    std::string compress_ratios_err;
    if (!get_u32_arr(gctx, "deepseek4.attention.compress_ratios",
                     compress_ratios_meta, &compress_ratios_err)) {
        set_last_error(compress_ratios_err);
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    std::vector<uint32_t> compress_ratios;
    const std::vector<uint32_t> expected_ratios =
        compute_compress_ratios((int)n_layer);
    if (compress_ratios_meta.size() != n_layer ||
        compress_ratios_meta != expected_ratios) {
        set_last_error(
            "deepseek4.attention.compress_ratios is missing or incompatible");
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    compress_ratios = compress_ratios_meta;

    const uint32_t kMissingSpecial = 0xFFFFFFFFu;
    const uint32_t raw_eos = get_u32_or(gctx, "tokenizer.ggml.eos_token_id", kMissingSpecial);
    const uint32_t raw_eot = get_u32_or(gctx, "tokenizer.ggml.eot_token_id", kMissingSpecial);

    std::fprintf(stderr, "[deepseek4] model: layers=%u embd=%u heads=%u head_dim=%u "
                 "lora_q=%u lora_o=%u out_groups=%u\n",
                 n_layer, n_embd, n_head, head_dim, n_lora_q, n_lora_o, n_out_group);
    std::fprintf(stderr, "[deepseek4] moe: experts=%u used=%u shared=%u ff=%u hash_layers=%u\n",
                 n_expert, n_expert_used, n_expert_shared, n_ff_exp, n_hash_layer);
    std::fprintf(stderr, "[deepseek4] attention: swa=%u rot=%u indexer_heads=%u top_k=%u hc=%u\n",
                 n_swa, n_rot, n_indexer_head, n_indexer_top_k, n_hc);

    // Fill output metadata
    out.n_layer         = (int)n_layer;
    out.n_embd          = (int)n_embd;
    out.n_vocab         = (int)n_vocab;
    out.n_head          = (int)n_head;
    out.n_head_kv       = (int)n_head_kv;
    out.head_dim        = (int)head_dim;
    out.n_rot           = (int)n_rot;
    out.n_out_group     = (int)n_out_group;
    out.n_lora_q        = (int)n_lora_q;
    out.n_lora_o        = (int)n_lora_o;
    out.n_expert        = (int)n_expert;
    out.n_expert_used   = (int)n_expert_used;
    out.n_expert_shared = (int)n_expert_shared;
    out.n_ff_exp        = (int)n_ff_exp;
    out.n_hash_layer    = (int)n_hash_layer;
    out.n_swa           = (int)n_swa;
    out.n_indexer_head  = (int)n_indexer_head;
    out.n_indexer_head_dim = (int)n_indexer_head_dim;
    out.n_indexer_top_k = (int)n_indexer_top_k;
    out.n_hc            = (int)n_hc;
    out.n_hc_sinkhorn_iter = (int)n_hc_sinkhorn;
    out.compress_ratios = compress_ratios;
    out.expert_weight_scale = expert_weight_scale;
    out.rope_freq_base  = rope_freq_base;
    out.rope_scale_factor = rope_scale_factor;
    out.rope_yarn_beta_fast = rope_yarn_beta_fast;
    out.rope_yarn_beta_slow = rope_yarn_beta_slow;
    out.compress_rope_freq_base = compress_rope_freq_base;
    out.rope_orig_ctx   = rope_orig_ctx;
    out.rms_eps         = rms_eps;
    out.hc_eps          = hc_eps;
    out.swiglu_clamp_exp = swiglu_clamp;
    out.eos_id          = (raw_eos == kMissingSpecial) ? -1 : (int32_t)raw_eos;
    out.eos_chat_id     = (raw_eot == kMissingSpecial) ? -1 : (int32_t)raw_eot;

    out.layers.resize(n_layer);
    out.backend = backend;

    // ── Build load plan ─────────────────────────────────────────────────
    TargetLoadPlan plan = plan_in;
    if (plan.layer_end < 0) plan.layer_end = (int)n_layer;
    if (plan.expert_metadata_only) {
        plan.load_output = false;
        plan.skip_expert_tensors = true;
    }
    if (plan.layer_begin < 0 || plan.layer_end < plan.layer_begin ||
        plan.layer_end > (int)n_layer) {
        set_last_error("invalid DeepSeek4 layer load range [" +
                       std::to_string(plan.layer_begin) + "," +
                       std::to_string(plan.layer_end) + ")");
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    // ── Collect tensors for allocation ──────────────────────────────────
    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    if (n_tensors < 0 ||
        (uint64_t)n_tensors > (uint64_t)std::numeric_limits<size_t>::max()) {
        set_last_error("GGUF tensor count is out of range");
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    const size_t data_offset = gguf_get_data_offset(gctx);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);

    std::vector<DS4TensorAlloc> allocs;
    allocs.reserve((size_t)n_tensors);
    size_t total_buf_size = 0;
    size_t tok_embd_alloc_idx = SIZE_MAX;

    for (int64_t ti = 0; ti < n_tensors; ti++) {
        const char * tname = gguf_get_tensor_name(gctx, ti);
        if (!should_keep_ds4_tensor(tname, plan)) continue;

        ggml_tensor * t = find_tensor(meta_ctx, tname);
        if (!t) continue;

        const size_t tensor_offset = gguf_get_tensor_offset(gctx, ti);
        const bool upload_to_backend = should_upload_ds4_tensor(tname, plan);

        DS4TensorAlloc a;
        a.tensor = t;
        a.tensor_offset = tensor_offset;
        a.file_size = gguf_get_tensor_size(gctx, ti);
        a.upload_to_backend = upload_to_backend;
        if (upload_to_backend) {
            total_buf_size = align_up_size(total_buf_size, alignment);
            const size_t alloc_size =
                ggml_backend_buft_get_alloc_size(buft, t);
            if (total_buf_size == SIZE_MAX ||
                alloc_size > SIZE_MAX - total_buf_size) {
                set_last_error("DeepSeek4 weight buffer size overflow");
                gguf_free(gctx);
                if (meta_ctx) ggml_free(meta_ctx);
                return false;
            }
            a.buffer_offset = total_buf_size;
            total_buf_size += alloc_size;
        }
        allocs.push_back(a);
        if (std::strcmp(tname, "token_embd.weight") == 0) {
            tok_embd_alloc_idx = allocs.size() - 1;
        }
    }

    // ── Allocate GPU buffer ─────────────────────────────────────────────
    ggml_backend_buffer_t buf = nullptr;
    if (total_buf_size > 0) {
        buf = ggml_backend_alloc_buffer(backend, total_buf_size);
        if (!buf) {
            set_last_error("failed to allocate GPU buffer (" + std::to_string(total_buf_size) + " bytes)");
            gguf_free(gctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // ── Assign tensors from meta_ctx to the backend buffer ──────────────
    // Use ggml_backend_tensor_alloc to properly set the buffer association.
    char * buf_base = buf ? (char *)ggml_backend_buffer_get_base(buf) : nullptr;
    for (auto & a : allocs) {
        if (!a.upload_to_backend || !buf) continue;
        if (ggml_backend_tensor_alloc(buf, a.tensor, buf_base + a.buffer_offset) != GGML_STATUS_SUCCESS) {
            set_last_error("ggml_backend_tensor_alloc failed");
            if (buf) ggml_backend_buffer_free(buf);
            gguf_free(gctx);
            ggml_free(meta_ctx);
            return false;
        }
    }

    // ── Memory-map the file and copy tensor data ────────────────────────
    DS4Mmap mmap;
    std::string mmap_err;
    if (!mmap.open_ro(path, mmap_err)) {
        set_last_error("mmap: " + mmap_err);
        if (buf) ggml_backend_buffer_free(buf);
        gguf_free(gctx);
        ggml_free(meta_ctx);
        return false;
    }
    for (auto & a : allocs) {
        if (!gguf_tensor_in_file(data_offset, a.tensor_offset, a.file_size, mmap.len)) {
            set_last_error(gguf_bounds_error("deepseek4 GGUF",
                                             ggml_get_name(a.tensor),
                                             ggml_type_name(a.tensor->type),
                                             data_offset,
                                             a.tensor_offset,
                                             a.file_size,
                                             mmap.len));
            mmap.close_map();
            if (buf) ggml_backend_buffer_free(buf);
            gguf_free(gctx);
            ggml_free(meta_ctx);
            return false;
        }
        a.file_offset = data_offset + a.tensor_offset;
    }

    bool fast_managed = (buf != nullptr) && ggml_backend_cuda_buffer_is_managed(buf) && (getenv("DFLASH_NO_PREAD") == nullptr);
    if (fast_managed) {
        // Unified/managed buffer: read weights straight off disk into it in parallel at
        // disk bandwidth, instead of mmap page-faults (~5x slower). Drop the cached file
        // pages afterward so the page cache does not double a near-RAM-size model.
        unsigned nth = std::thread::hardware_concurrency();
        if (nth == 0) nth = 4;
        if (nth > 8)  nth = 8;
        std::atomic<size_t> next{0};
        std::atomic<bool> read_ok{true};
        auto worker = [&]() {
            size_t i;
            while ((i = next.fetch_add(1)) < allocs.size()) {
                auto & a = allocs[i];
                if (!a.upload_to_backend) continue;
                char * dst = (char *) a.tensor->data;
                size_t done = 0;
                while (done < a.file_size) {
                    ssize_t r = pread(mmap.fd, dst + done, a.file_size - done,
                                      (off_t) (a.file_offset + done));
                    if (r < 0 && errno == EINTR) continue;
                    if (r <= 0) { read_ok = false; return; }
                    done += (size_t) r;
                }
            }
        };
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < nth; t++) pool.emplace_back(worker);
        for (auto & th : pool) th.join();
        posix_fadvise(mmap.fd, 0, (off_t) mmap.len, POSIX_FADV_DONTNEED);
        ggml_backend_synchronize(backend);  // make CPU-written managed pages visible to GPU
        if (!read_ok) {
            set_last_error("parallel weight read failed");
            mmap.close_map();
            if (buf) ggml_backend_buffer_free(buf);
            gguf_free(gctx);
            ggml_free(meta_ctx);
            return false;
        }
    } else {
        for (auto & a : allocs) {
            if (!a.upload_to_backend) continue;
            const void * src_data = (const char *)mmap.addr + a.file_offset;
            ggml_backend_tensor_set(a.tensor, src_data, 0, a.file_size);
        }
    }
    mmap.close_map();

    // ── Set up CPU embedder ─────────────────────────────────────────────
    // The embedder is set up using the mmap data directly (like gemma4).
    // For now, we use an owned copy of the token embedding table bytes.
    if (tok_embd_alloc_idx != SIZE_MAX) {
        const auto & a = allocs[tok_embd_alloc_idx];
        out.embedder.tok_embd_owned.resize(a.file_size);
        // Re-read from mmap (already closed). Use the GPU tensor instead:
        // Actually, we need the raw bytes for dequantization. Reopen mmap briefly.
        DS4Mmap emb_mmap;
        std::string emb_err;
        if (emb_mmap.open_ro(path, emb_err)) {
            std::memcpy(out.embedder.tok_embd_owned.data(),
                        (const char *)emb_mmap.addr + a.file_offset, a.file_size);
            emb_mmap.close_map();
        } else {
            set_last_error("embedder mmap: " + emb_err);
            if (buf) ggml_backend_buffer_free(buf);
            gguf_free(gctx);
            ggml_free(meta_ctx);
            return false;
        }
        out.embedder.tok_embd_bytes = out.embedder.tok_embd_owned.data();
        out.embedder.tok_embd_type  = a.tensor->type;
        out.embedder.n_embd         = n_embd;
        out.embedder.n_vocab        = (int64_t)n_vocab;
        out.embedder.row_bytes      = a.file_size / (size_t)n_vocab;
    }

    // ── Bind tensors to weight struct fields ────────────────────────────
    for (auto & a : allocs) {
        const char * name = ggml_get_name(a.tensor);

        // Global tensors
        if (std::strcmp(name, "token_embd.weight") == 0) { out.tok_embd = a.tensor; continue; }
        if (std::strcmp(name, "output_norm.weight") == 0) { out.out_norm = a.tensor; continue; }
        if (std::strcmp(name, "output.weight") == 0) { out.output = a.tensor; continue; }
        if (std::strcmp(name, "output_hc_base.weight") == 0) { out.output_hc_base = a.tensor; continue; }
        if (std::strcmp(name, "output_hc_fn.weight") == 0) { out.output_hc_fn = a.tensor; continue; }
        if (std::strcmp(name, "output_hc_scale.weight") == 0) { out.output_hc_scale = a.tensor; continue; }

        // Per-layer tensors
        int il = -1;
        if (!parse_block_tensor_name(name, il) || il < 0 || il >= (int)n_layer) continue;
        DeepSeek4Layer & L = out.layers[il];

        // Find the suffix after "blk.<il>."
        const char * p = name;
        while (*p && *p != '.') p++;  // skip "blk"
        if (*p == '.') p++;           // skip first '.'
        while (*p && *p != '.') p++;  // skip layer number
        if (*p == '.') p++;           // skip second '.'
        const std::string suffix(p);

        // Attention
        if (suffix == "attn_norm.weight")          { L.attn_norm = a.tensor; continue; }
        if (suffix == "attn_q_a.weight")           { L.attn_q_a = a.tensor; continue; }
        if (suffix == "attn_q_a_norm.weight")      { L.attn_q_a_norm = a.tensor; continue; }
        if (suffix == "attn_q_b.weight")           { L.attn_q_b = a.tensor; continue; }
        if (suffix == "attn_kv.weight")            { L.attn_kv = a.tensor; continue; }
        if (suffix == "attn_kv_a_norm.weight")     { L.attn_kv_a_norm = a.tensor; continue; }
        if (suffix == "attn_sinks.weight")         { L.attn_sinks = a.tensor; continue; }
        if (suffix == "attn_output_a.weight")      { L.attn_output_a = a.tensor; continue; }
        if (suffix == "attn_output_b.weight")      { L.attn_output_b = a.tensor; continue; }

        // Compressor
        if (suffix == "attn_compressor_ape.weight")  { L.attn_compressor_ape = a.tensor; continue; }
        if (suffix == "attn_compressor_kv.weight")   { L.attn_compressor_kv = a.tensor; continue; }
        if (suffix == "attn_compressor_gate.weight") { L.attn_compressor_gate = a.tensor; continue; }
        if (suffix == "attn_compressor_norm.weight") { L.attn_compressor_norm = a.tensor; continue; }

        // Indexer
        if (suffix == "indexer.attn_q_b.weight")     { L.indexer_attn_q_b = a.tensor; continue; }
        if (suffix == "indexer.proj.weight")          { L.indexer_proj = a.tensor; continue; }
        if (suffix == "indexer_compressor_ape.weight")  { L.indexer_compressor_ape = a.tensor; continue; }
        if (suffix == "indexer_compressor_kv.weight")   { L.indexer_compressor_kv = a.tensor; continue; }
        if (suffix == "indexer_compressor_gate.weight") { L.indexer_compressor_gate = a.tensor; continue; }
        if (suffix == "indexer_compressor_norm.weight") { L.indexer_compressor_norm = a.tensor; continue; }

        // HC attention
        if (suffix == "hc_attn_fn.weight")         { L.hc_attn_fn = a.tensor; continue; }
        if (suffix == "hc_attn_scale.weight")      { L.hc_attn_scale = a.tensor; continue; }
        if (suffix == "hc_attn_base.weight")       { L.hc_attn_base = a.tensor; continue; }

        // FFN
        if (suffix == "ffn_norm.weight")           { L.ffn_norm = a.tensor; continue; }
        if (suffix == "ffn_gate_inp.weight")       { L.ffn_gate_inp = a.tensor; continue; }
        if (suffix == "exp_probs_b.bias")          { L.ffn_exp_probs_b = a.tensor; continue; }
        if (suffix == "ffn_gate_tid2eid.weight")   { L.ffn_gate_tid2eid = a.tensor; continue; }
        if (suffix == "ffn_gate_exps.weight")      { L.ffn_gate_exps = a.tensor; continue; }
        if (suffix == "ffn_up_exps.weight")        { L.ffn_up_exps = a.tensor; continue; }
        if (suffix == "ffn_down_exps.weight")      { L.ffn_down_exps = a.tensor; continue; }
        if (suffix == "ffn_gate_shexp.weight")     { L.ffn_gate_shexp = a.tensor; continue; }
        if (suffix == "ffn_up_shexp.weight")       { L.ffn_up_shexp = a.tensor; continue; }
        if (suffix == "ffn_down_shexp.weight")     { L.ffn_down_shexp = a.tensor; continue; }

        // HC FFN
        if (suffix == "hc_ffn_fn.weight")          { L.hc_ffn_fn = a.tensor; continue; }
        if (suffix == "hc_ffn_scale.weight")       { L.hc_ffn_scale = a.tensor; continue; }
        if (suffix == "hc_ffn_base.weight")        { L.hc_ffn_base = a.tensor; continue; }
    }

    // A GGUF can be structurally parseable while omitting a tensor the graph
    // unconditionally dereferences. Reject incomplete load plans now instead
    // of crashing during the first prefill after a multi-minute model load.
    std::string missing;
    auto require_tensor = [&](ggml_tensor * tensor, const std::string & name) {
        if (!tensor && missing.empty()) missing = name;
    };
    if (plan.load_output && !plan.expert_metadata_only) {
        require_tensor(out.tok_embd, "token_embd.weight");
        require_tensor(out.out_norm, "output_norm.weight");
        require_tensor(out.output, "output.weight");
        require_tensor(out.output_hc_base, "output_hc_base.weight");
        require_tensor(out.output_hc_fn, "output_hc_fn.weight");
        require_tensor(out.output_hc_scale, "output_hc_scale.weight");
    }
    for (int il = plan.layer_begin; il < plan.layer_end && missing.empty(); ++il) {
        const DeepSeek4Layer & layer = out.layers[(size_t)il];
        const std::string prefix = "blk." + std::to_string(il) + ".";
        if (plan.expert_metadata_only) {
            require_tensor(layer.ffn_gate_exps, prefix + "ffn_gate_exps.weight");
            require_tensor(layer.ffn_up_exps, prefix + "ffn_up_exps.weight");
            require_tensor(layer.ffn_down_exps, prefix + "ffn_down_exps.weight");
            continue;
        }
        require_tensor(layer.attn_norm, prefix + "attn_norm.weight");
        require_tensor(layer.attn_q_a, prefix + "attn_q_a.weight");
        require_tensor(layer.attn_q_a_norm, prefix + "attn_q_a_norm.weight");
        require_tensor(layer.attn_q_b, prefix + "attn_q_b.weight");
        require_tensor(layer.attn_kv, prefix + "attn_kv.weight");
        require_tensor(layer.attn_kv_a_norm, prefix + "attn_kv_a_norm.weight");
        require_tensor(layer.attn_sinks, prefix + "attn_sinks.weight");
        require_tensor(layer.attn_output_a, prefix + "attn_output_a.weight");
        require_tensor(layer.attn_output_b, prefix + "attn_output_b.weight");
        require_tensor(layer.hc_attn_fn, prefix + "hc_attn_fn.weight");
        require_tensor(layer.hc_attn_scale, prefix + "hc_attn_scale.weight");
        require_tensor(layer.hc_attn_base, prefix + "hc_attn_base.weight");
        require_tensor(layer.ffn_norm, prefix + "ffn_norm.weight");
        require_tensor(layer.ffn_gate_inp, prefix + "ffn_gate_inp.weight");
        require_tensor(layer.ffn_gate_exps, prefix + "ffn_gate_exps.weight");
        require_tensor(layer.ffn_up_exps, prefix + "ffn_up_exps.weight");
        require_tensor(layer.ffn_down_exps, prefix + "ffn_down_exps.weight");
        require_tensor(layer.ffn_gate_shexp, prefix + "ffn_gate_shexp.weight");
        require_tensor(layer.ffn_up_shexp, prefix + "ffn_up_shexp.weight");
        require_tensor(layer.ffn_down_shexp, prefix + "ffn_down_shexp.weight");
        require_tensor(layer.hc_ffn_fn, prefix + "hc_ffn_fn.weight");
        require_tensor(layer.hc_ffn_scale, prefix + "hc_ffn_scale.weight");
        require_tensor(layer.hc_ffn_base, prefix + "hc_ffn_base.weight");
        if (il < (int)n_hash_layer) {
            require_tensor(layer.ffn_gate_tid2eid,
                           prefix + "ffn_gate_tid2eid.weight");
        }
        const uint32_t ratio = compress_ratios[(size_t)il];
        if (ratio > 0) {
            require_tensor(layer.attn_compressor_ape,
                           prefix + "attn_compressor_ape.weight");
            require_tensor(layer.attn_compressor_kv,
                           prefix + "attn_compressor_kv.weight");
            require_tensor(layer.attn_compressor_gate,
                           prefix + "attn_compressor_gate.weight");
            require_tensor(layer.attn_compressor_norm,
                           prefix + "attn_compressor_norm.weight");
        }
        if (ratio == 4) {
            require_tensor(layer.indexer_attn_q_b,
                           prefix + "indexer.attn_q_b.weight");
            require_tensor(layer.indexer_proj,
                           prefix + "indexer.proj.weight");
            require_tensor(layer.indexer_compressor_ape,
                           prefix + "indexer_compressor_ape.weight");
            require_tensor(layer.indexer_compressor_kv,
                           prefix + "indexer_compressor_kv.weight");
            require_tensor(layer.indexer_compressor_gate,
                           prefix + "indexer_compressor_gate.weight");
            require_tensor(layer.indexer_compressor_norm,
                           prefix + "indexer_compressor_norm.weight");
        }
    }
    if (!missing.empty()) {
        set_last_error("missing required DeepSeek4 tensor: " + missing);
        if (buf) ggml_backend_buffer_free(buf);
        gguf_free(gctx);
        ggml_free(meta_ctx);
        return false;
    }

    out.ctx = meta_ctx;
    out.buf = buf;

    gguf_free(gctx);
    // Note: meta_ctx is now owned by out.ctx — do NOT free it here.

    std::fprintf(stderr, "[deepseek4] loaded %zu tensors, %.1f MB GPU buffer%s\n",
                 allocs.size(), (double)total_buf_size / (1024.0 * 1024.0),
                 plan.expert_metadata_only ? " [expert-metadata-only]" : "");
    return true;
}

namespace {

static MoeHybridColdBackend ds4_cold_backend_from_env() {
    const char * value = std::getenv("DFLASH_MOE_COLD_BACKEND");
    if (!value || !value[0]) return MoeHybridColdBackend::Cpu;
    if (std::strcmp(value, "gpu") == 0 || std::strcmp(value, "hip") == 0 ||
        std::strcmp(value, "rocm") == 0) {
        return MoeHybridColdBackend::Gpu;
    }
    return MoeHybridColdBackend::Cpu;
}

static MoeHybridConfig make_ds4_moe_hybrid_config(const DeepSeek4Weights & w) {
    MoeHybridConfig cfg;
    cfg.n_embd = w.n_embd;
    cfg.n_expert = w.n_expert;
    cfg.n_expert_used = w.n_expert_used;
    cfg.n_ff_exp = w.n_ff_exp;
    cfg.n_layer = w.n_layer;
    cfg.cold_expert_backend = ds4_cold_backend_from_env();
    return cfg;
}

static MoeLayerDesc make_ds4_moe_layer_desc(const DeepSeek4Layer & L) {
    MoeLayerDesc desc;
    desc.ffn_gate_exps = L.ffn_gate_exps;
    desc.ffn_up_exps = L.ffn_up_exps;
    desc.ffn_down_exps = L.ffn_down_exps;
    desc.ffn_gate_up_exps = nullptr;
    desc.ffn_gate_shexp = L.ffn_gate_shexp;
    desc.ffn_up_shexp = L.ffn_up_shexp;
    desc.ffn_down_shexp = L.ffn_down_shexp;
    desc.ffn_gate_inp_shexp = nullptr;
    return desc;
}

}  // namespace

bool build_deepseek4_moe_hybrid_storage_from_file_with_mmap(
        const std::string & path,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        const MoeHybridPlacement & placement,
        const MoeHybridConfig * cfg_override,
        MoeHybridStorage & out,
        std::string * err) {
    ggml_context * expert_meta = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx = &expert_meta;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        if (err) *err = "failed to re-open GGUF for expert loading";
        return false;
    }

    DS4Mmap mmap;
    std::string mmap_err;
    if (!mmap.open_ro(path, mmap_err)) {
        gguf_free(gctx);
        if (expert_meta) ggml_free(expert_meta);
        if (err) *err = mmap_err;
        return false;
    }
    if (mmap.fd >= 0) {
        ::close(mmap.fd);
        mmap.fd = -1;
    }

    const size_t data_start = gguf_get_data_offset(gctx);
    const auto * file_bytes = static_cast<const uint8_t *>(mmap.addr);
    std::vector<LayerExpertFileData> layer_file_data((size_t)w.n_layer);
    bool bad_bounds = false;
    std::string bounds_err;

    for (int il = 0; il < w.n_layer; ++il) {
        char name[128];
        auto find_tensor_data = [&](const char * suffix) -> ExpertTensorFileData {
            std::snprintf(name, sizeof(name), "blk.%d.%s.weight", il, suffix);
            int64_t tid = gguf_find_tensor(gctx, name);
            if (tid < 0) return {};
            const size_t tensor_off = gguf_get_tensor_offset(gctx, tid);
            const size_t sz = gguf_get_tensor_size(gctx, tid);
            if (!gguf_tensor_in_file(data_start, tensor_off, sz, mmap.len)) {
                bad_bounds = true;
                bounds_err = gguf_bounds_error("deepseek4 expert GGUF", name,
                                               ggml_type_name(gguf_get_tensor_type(gctx, tid)),
                                               data_start, tensor_off, sz, mmap.len);
                return {};
            }
            const size_t off = data_start + tensor_off;
            return { file_bytes + off, sz };
        };

        layer_file_data[(size_t)il].gate_exps = find_tensor_data("ffn_gate_exps");
        layer_file_data[(size_t)il].up_exps = find_tensor_data("ffn_up_exps");
        layer_file_data[(size_t)il].down_exps = find_tensor_data("ffn_down_exps");
        if (bad_bounds) {
            mmap.close_map();
            gguf_free(gctx);
            if (expert_meta) ggml_free(expert_meta);
            if (err) *err = bounds_err;
            return false;
        }
    }

    std::vector<MoeLayerDesc> layer_descs((size_t)w.n_layer);
    for (int il = 0; il < w.n_layer; ++il) {
        layer_descs[(size_t)il] = make_ds4_moe_layer_desc(w.layers[(size_t)il]);
    }

    const MoeHybridConfig cfg = cfg_override ? *cfg_override : make_ds4_moe_hybrid_config(w);
    const bool ok = build_moe_hybrid_storage_from_file_with_mmap(
        cfg, backend, placement, layer_descs, layer_file_data,
        mmap.addr, mmap.len, out, err);

    if (!ok) {
        mmap.close_map();
    } else {
        mmap.addr = nullptr;
        mmap.len = 0;
    }
    gguf_free(gctx);
    if (expert_meta) ggml_free(expert_meta);
    return ok;
}

void free_deepseek4_weights(DeepSeek4Weights & w) {
    deepseek4_release_runtime_graphs(w);
    if (w.ctx) { ggml_free(w.ctx); w.ctx = nullptr; }
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    w.layers.clear();
    w.embedder.tok_embd_owned.clear();
    w.embedder.tok_embd_bytes = nullptr;
    w.moe_hybrid = false;
}

}  // namespace dflash::common
