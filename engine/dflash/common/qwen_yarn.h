// Qwen3.8-Flash-Next static YaRN policy and scalar reference.
//
// The released BF16 checkpoint is native at 262,144 tokens and carries
// `rope_type = default`; YaRN is an operator-selected inference override, not
// checkpoint metadata.  The official pinned model README prescribes static
// YaRN with factor 4, original context 262,144, theta 1e7, partial rotary
// width 64, and interleaved M-RoPE sections [11, 11, 10] for a 1,000,000-token
// server context.  Keeping policy and arithmetic in this C-compatible module
// gives the C server, the C++ model runtime, and GPU-free parity tests one
// source of truth.
//
// Provenance:
//   Qwen/Qwen3.8-Flash-Next README revision
//   f5d08274bafd880402bd16f5e3e6c514136ec06c, "Processing Ultra-Long
//   Texts".  The recipe is static and is explicitly warned to affect shorter
//   inputs, so Ember never enables it from --max-ctx alone.
//   Hugging Face transformers modeling_rope_utils.py revision
//   36deb0b53ed0863f4b4dfdea23dcaec7f3df3701, `_compute_yarn_parameters`.
//
// This file is Ember-owned vendored-engine divergence.  It deliberately does
// not add a new HIP kernel: ggml_rope_multi already accepts the same YaRN and
// interleaved-MRoPE parameters on CPU and HIP.

#ifndef EMBER_DFLASH_QWEN_YARN_H
#define EMBER_DFLASH_QWEN_YARN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    EMBER_QWEN_NATIVE_CONTEXT = 262144,
    EMBER_QWEN_YARN_MAX_CONTEXT = 1000000,
    EMBER_QWEN_ROPE_DIM = 64,
    EMBER_QWEN_ROPE_FREQ_COUNT = 32,
};

typedef struct {
    bool enabled;
    int32_t original_context;
    int32_t max_context;
    float factor;
    float theta;
    float beta_fast;
    float beta_slow;
    float attention_factor;
    int32_t correction_low;
    int32_t correction_high;
    int32_t mrope_sections[3];
} ember_qwen_yarn_config;

// Resolve the release policy.  Native mode accepts contexts through 262,144.
// YaRN mode is explicit, accepts exactly 1,000,000, and resolves only the
// official factor-4 recipe. This deliberately does not silently apply factor
// 4 to shorter extensions (the official README recommends factor 2 for a
// typical 524,288-token context). `error` may be NULL; otherwise it is always
// NUL-terminated when error_size is nonzero.
bool ember_qwen_yarn_configure(bool enable_yarn, int32_t max_context,
                               ember_qwen_yarn_config *config,
                               char *error, size_t error_size);

// Build the 32 inverse frequencies used by the first 64 dimensions of every
// 256-wide QSA/indexer head.  Native mode returns ordinary RoPE.  YaRN mode
// follows transformers' truncated correction range and default attention
// scaling (beta_fast=32, beta_slow=1, 1 + 0.1*ln(factor)).
void ember_qwen_yarn_inv_freq(const ember_qwen_yarn_config *config,
                              float out[EMBER_QWEN_ROPE_FREQ_COUNT]);

// Compute the Qwen interleaved M-RoPE cosine/sine vector.  Positions are
// temporal/height/width.  Text passes the same scalar in all three lanes;
// multimodal callers preserve the three axes.  Output layout duplicates the
// 32 frequencies into NeoX halves exactly as Qwen4ExpTextRotaryEmbedding.
void ember_qwen_yarn_cos_sin(const ember_qwen_yarn_config *config,
                             const int32_t positions[3],
                             float cos_out[EMBER_QWEN_ROPE_DIM],
                             float sin_out[EMBER_QWEN_ROPE_DIM]);

// Scalar reference application to the first 64 elements of a 256-wide QSA
// head.  Elements [64, head_dim) are untouched.  This is for runtime fallback
// and differential tests; graph code should use ggml_rope_multi/IMROPE.
bool ember_qwen_yarn_apply(float *head, size_t head_dim,
                           const ember_qwen_yarn_config *config,
                           const int32_t positions[3]);

#ifdef __cplusplus
}
#endif

#endif
