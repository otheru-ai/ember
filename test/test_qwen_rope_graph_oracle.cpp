// GPU-free oracle for the QSA graph RoPE path.
//
// Tranche 1 replaces the host's scalar `ember_qwen_yarn_apply` with an
// in-graph `ggml_rope_multi`. That swap has five parameters which are silent
// when wrong: they neither crash nor assert, and the failing path is GPU-side,
// so the ordinary host gauntlet cannot see them. One of them -- passing our
// inverse-frequency table directly as the `c` argument, which is a *divisor*
// rather than a frequency table -- was proposed in review and would have
// collapsed every angle to `pos`.
//
// This runs `ggml_rope_multi` on the CPU backend and compares it against
// `ember_qwen_yarn_apply`, the scalar reference. CPU and HIP share the op, so
// agreement here is necessary but not sufficient: it proves the parameter
// mapping, not that the HIP kernel matches. Hardware differential remains the
// acceptance gate.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "qwen_yarn.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (condition) {                                                    \
            ++g_pass;                                                       \
        } else {                                                            \
            ++g_fail;                                                       \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__,     \
                         (message));                                        \
        }                                                                   \
    } while (0)

namespace {

constexpr int kHeadDim = EMBER_QWEN_ROPE_DIM * 4;  // 256, partial rotary 0.25
constexpr int kHeads   = 2;
constexpr int kTokens  = 3;

// Deterministic, non-degenerate input. Values straddle zero so a dropped sign
// or a swapped rotation half is visible.
std::vector<float> patterned_head(int seed) {
    std::vector<float> v(static_cast<size_t>(kHeadDim));
    for (int i = 0; i < kHeadDim; ++i) {
        const float x = static_cast<float>((seed + 1) * (i + 3));
        v[static_cast<size_t>(i)] =
            0.31f * std::sin(x * 0.017f) + 0.11f * std::cos(x * 0.041f);
    }
    return v;
}

// c[k] = theta_scale^k / inv_freq[k]. ggml divides its own angle by c, so this
// ratio is what reproduces our table. Identity (all 1.0) when YaRN is off.
std::vector<float> freq_factor_ratio(const ember_qwen_yarn_config & cfg) {
    std::vector<float> c(EMBER_QWEN_ROPE_FREQ_COUNT);
    const double theta_scale =
        std::pow(static_cast<double>(cfg.theta),
                 -2.0 / static_cast<double>(EMBER_QWEN_ROPE_DIM));
    for (int k = 0; k < EMBER_QWEN_ROPE_FREQ_COUNT; ++k) {
        const double base = std::pow(theta_scale, k);
        c[static_cast<size_t>(k)] =
            static_cast<float>(base / static_cast<double>(cfg.inv_freq[k]));
    }
    return c;
}

// The two candidate mappings from our static YaRN policy onto ggml's rope
// parameters. They should agree; the oracle is how we find out rather than
// arguing about it.
//
//   FreqFactors -- hand ggml the already-corrected table as `c` and turn its
//     own YaRN interpolation off. `c` is a *divisor*, so the entry is
//     theta_scale^k / inv_freq[k], not inv_freq[k].
//   ExtFactor   -- hand ggml the raw recipe (freq_scale = 1/4, ext_factor = 1)
//     and let it derive the ramp itself. `n_ctx_orig` is the *native* 262144,
//     never --max-ctx: the correction dims are computed from it.
enum class Path { FreqFactors, ExtFactor };

// Run ggml_rope_multi on the CPU backend over [head_dim, heads, tokens].
// Returns false if the graph could not be built or executed.
bool graph_rope(const ember_qwen_yarn_config & cfg,
                const std::vector<float> & input,
                const std::vector<int32_t> & positions_axis_major,
                Path path,
                std::vector<float> & out) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) return false;

    ggml_init_params params{};
    params.mem_size = 4U * 1024U * 1024U;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * a =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kHeads, kTokens);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * kTokens);
    ggml_tensor * c = path == Path::FreqFactors
        ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, EMBER_QWEN_ROPE_FREQ_COUNT)
        : nullptr;

    int sections[GGML_MROPE_SECTIONS] = {cfg.mrope_sections[0],
                                         cfg.mrope_sections[1],
                                         cfg.mrope_sections[2], 0};

    // ExtFactor lets ggml derive mscale = 1 + 0.1*ln(1/freq_scale) itself, so
    // attn_factor stays 1 there; folding our attention_factor in as well would
    // apply the scale twice.
    const bool derive = path == Path::ExtFactor;
    ggml_tensor * rotated = ggml_rope_multi(
        ctx, a, b, c, EMBER_QWEN_ROPE_DIM, sections, GGML_ROPE_TYPE_IMROPE,
        cfg.original_context, cfg.theta,
        /*freq_scale=*/derive ? 1.0f / cfg.factor : 1.0f,
        /*ext_factor=*/derive ? 1.0f : 0.0f,
        /*attn_factor=*/derive ? 1.0f : cfg.attention_factor,
        cfg.beta_fast, cfg.beta_slow);
    if (!rotated) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(a, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(b, positions_axis_major.data(), 0,
                            positions_axis_major.size() * sizeof(int32_t));
    if (c) {
        const std::vector<float> ratio = freq_factor_ratio(cfg);
        ggml_backend_tensor_set(c, ratio.data(), 0,
                                ratio.size() * sizeof(float));
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, rotated);
    const bool ok =
        ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (ok) {
        out.resize(input.size());
        ggml_backend_tensor_get(rotated, out.data(), 0,
                                out.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

// The scalar reference, applied per head exactly as qwen4exp_runtime.cpp does.
std::vector<float> reference_rope(const ember_qwen_yarn_config & cfg,
                                  const std::vector<float> & input,
                                  const std::vector<int32_t> & pos_t,
                                  const std::vector<int32_t> & pos_h,
                                  const std::vector<int32_t> & pos_w) {
    std::vector<float> out = input;
    for (int token = 0; token < kTokens; ++token) {
        const int32_t position[3] = {pos_t[static_cast<size_t>(token)],
                                     pos_h[static_cast<size_t>(token)],
                                     pos_w[static_cast<size_t>(token)]};
        for (int head = 0; head < kHeads; ++head) {
            float * slot =
                out.data() +
                (static_cast<size_t>(token) * kHeads + head) * kHeadDim;
            ember_qwen_yarn_apply(slot, kHeadDim, &cfg, position);
        }
    }
    return out;
}

float max_abs_delta(const std::vector<float> & a,
                    const std::vector<float> & b) {
    float worst = 0.0f;
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; ++i) {
        worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

void run_case(bool enable_yarn, int32_t max_context, Path path,
              const char * label) {
    ember_qwen_yarn_config cfg;
    char err[192];
    if (!ember_qwen_yarn_configure(enable_yarn, max_context, &cfg, err,
                                   sizeof(err))) {
        CHECK(false, "yarn configure failed");
        return;
    }

    std::vector<float> input;
    input.reserve(static_cast<size_t>(kHeadDim) * kHeads * kTokens);
    for (int token = 0; token < kTokens; ++token) {
        for (int head = 0; head < kHeads; ++head) {
            const std::vector<float> h = patterned_head(token * kHeads + head);
            input.insert(input.end(), h.begin(), h.end());
        }
    }

    // Text M-RoPE: the three axes carry the same scalar; lane 3 is zero, as
    // llama.cpp writes for text tokens.
    const std::vector<int32_t> pos_t{5, 6, 7};
    std::vector<int32_t> pos(4U * kTokens, 0);
    for (int axis = 0; axis < 3; ++axis) {
        for (int token = 0; token < kTokens; ++token) {
            pos[static_cast<size_t>(axis * kTokens + token)] =
                pos_t[static_cast<size_t>(token)];
        }
    }

    std::vector<float> graph_out;
    const bool ok = graph_rope(cfg, input, pos, path, graph_out);
    CHECK(ok, "rope_multi graph executes on the CPU backend");
    if (!ok) return;

    const std::vector<float> expected =
        reference_rope(cfg, input, pos_t, pos_t, pos_t);

    const float delta = max_abs_delta(graph_out, expected);
    std::fprintf(stderr, "[rope-oracle] case=%s max_abs_delta=%.9g\n", label,
                 static_cast<double>(delta));
    CHECK(delta < 1.0e-5f,
          "graph rope_multi matches ember_qwen_yarn_apply");

    // Elements [64, 256) must be untouched: partial rotary width is 64. This
    // is the n_dims trap -- passing head_dim instead of 64 rotates dimensions
    // that must be preserved, and nothing else in the suite would notice.
    float tail_delta = 0.0f;
    for (int token = 0; token < kTokens; ++token) {
        for (int head = 0; head < kHeads; ++head) {
            const size_t base =
                (static_cast<size_t>(token) * kHeads + head) * kHeadDim;
            for (int i = EMBER_QWEN_ROPE_DIM; i < kHeadDim; ++i) {
                tail_delta = std::fmax(
                    tail_delta,
                    std::fabs(graph_out[base + static_cast<size_t>(i)] -
                              input[base + static_cast<size_t>(i)]));
            }
        }
    }
    CHECK(tail_delta == 0.0f,
          "elements outside the 64-wide rotary window are untouched");
}

void test_freq_factor_ratio_is_identity_without_yarn() {
    ember_qwen_yarn_config cfg;
    char err[192];
    if (!ember_qwen_yarn_configure(false, 262144, &cfg, err, sizeof(err))) {
        CHECK(false, "yarn-off configure failed");
        return;
    }
    const std::vector<float> ratio = freq_factor_ratio(cfg);
    float worst = 0.0f;
    for (float v : ratio) worst = std::fmax(worst, std::fabs(v - 1.0f));
    CHECK(worst < 1.0e-5f,
          "freq-factor ratio is identity when YaRN is off");
}

// The correction range is derived from the *native* context, not --max-ctx.
// Passing 1,000,000 here moves the ramp and is silent.
void test_correction_dims_agree_with_ggml() {
    ember_qwen_yarn_config cfg;
    char err[192];
    if (!ember_qwen_yarn_configure(true, 1000000, &cfg, err, sizeof(err))) {
        CHECK(false, "yarn-on configure failed");
        return;
    }
    CHECK(cfg.original_context == EMBER_QWEN_NATIVE_CONTEXT,
          "yarn keeps the native context as the correction basis");

    float dims[2] = {0.0f, 0.0f};
    ggml_rope_yarn_corr_dims(EMBER_QWEN_ROPE_DIM, cfg.original_context,
                             cfg.theta, cfg.beta_fast, cfg.beta_slow, dims);
    CHECK(static_cast<int32_t>(dims[0]) == cfg.correction_low,
          "ggml low correction dim matches the host policy");
    CHECK(static_cast<int32_t>(dims[1]) == cfg.correction_high,
          "ggml high correction dim matches the host policy");
}

}  // namespace

int main() {
    test_freq_factor_ratio_is_identity_without_yarn();
    test_correction_dims_agree_with_ggml();
    // Native: YaRN off, so the freq-factor table is the identity and the
    // ext-factor path does not apply (ggml skips the ramp when ext_factor is
    // 0, which is what freq_scale = 1 with factor 1 already gives).
    run_case(false, 262144, Path::FreqFactors, "native/freq-factors");
    run_case(true, 1000000, Path::FreqFactors, "yarn4/freq-factors");
    run_case(true, 1000000, Path::ExtFactor, "yarn4/ext-factor");

    std::fprintf(stderr, "%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
