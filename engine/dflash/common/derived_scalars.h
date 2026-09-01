// Pure helper: verify that tensor-shape-derived scalars match GGUF-declared
// metadata.  No IO; safe to call from any loader after weights are loaded.
//
// Returns true when derived == declared for all three dimensions.
// On mismatch fills `err` with a diagnostic and returns false.
//
// Callers must compute the *expected* values from their declared scalars:
//   draft loader   : expected_q_dim = n_head * head_dim
//                    expected_kv_dim = n_head_kv * head_dim
//   qwen35 target  : expected_q_dim = n_head * n_embd_head_k * 2  (Q+gate packed)
//                    expected_kv_dim = n_head_kv * n_embd_head_k
// Both loaders: expected_n_embd = n_embd (wq->ne[0] = input projection dim).
//
// Equivalent pattern for gemma4 lives inline in gemma4_backend.cpp (~line 1072)
// as a silent override rather than an assertion; kept separate intentionally.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace dflash::common {

// verify_derived_scalars
//   wq_ne1         : weight_q->ne[1]  (output dim of Q projection)
//   wk_ne1         : weight_k->ne[1]  (output dim of K projection)
//   wq_ne0         : weight_q->ne[0]  (input dim of Q projection == n_embd)
//   expected_q_dim : n_head * head_dim [* 2 for packed Q+gate]
//   expected_kv_dim: n_head_kv * head_dim
//   expected_n_embd: n_embd
//   layer_tag      : short string for the error message (e.g. "blk.0" or "blk.3")
//   err            : filled on mismatch
inline bool verify_derived_scalars(
        int64_t wq_ne1, int64_t wk_ne1, int64_t wq_ne0,
        int64_t expected_q_dim, int64_t expected_kv_dim, int64_t expected_n_embd,
        const char * layer_tag,
        std::string & err)
{
    if (wq_ne1 != expected_q_dim) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "GGUF shape mismatch: %s attn_q.weight->ne[1]=%lld != expected_q_dim=%lld",
            layer_tag, (long long)wq_ne1, (long long)expected_q_dim);
        err = buf;
        return false;
    }
    if (wk_ne1 != expected_kv_dim) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "GGUF shape mismatch: %s attn_k.weight->ne[1]=%lld != expected_kv_dim=%lld",
            layer_tag, (long long)wk_ne1, (long long)expected_kv_dim);
        err = buf;
        return false;
    }
    if (wq_ne0 != expected_n_embd) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "GGUF shape mismatch: %s attn_q.weight->ne[0]=%lld != n_embd=%lld",
            layer_tag, (long long)wq_ne0, (long long)expected_n_embd);
        err = buf;
        return false;
    }
    return true;
}

} // namespace dflash::common
