#ifndef GGML_CUDA_GATED_DELTA_NET_LAYOUT_H
#define GGML_CUDA_GATED_DELTA_NET_LAYOUT_H

// Host-testable contract for the CUDA GDN gate indexing. The kernel derives
// both gate and beta offsets from beta's outer strides. Scalar gates therefore
// need identical strides; KDA gates have an S_v-wide leading dimension and the
// kernel deliberately rescales the shared offset by S_v.

#include "ggml.h"

static inline bool ggml_cuda_gated_delta_net_gate_layout_supported(
        const ggml_tensor * g,
        const ggml_tensor * beta,
        int64_t             s_v) {
    const bool kda = g->ne[0] == s_v;
    return (g->ne[0] == 1 || kda) && beta->ne[0] == 1 &&
           g->ne[1] == beta->ne[1] &&
           g->ne[2] == beta->ne[2] &&
           g->ne[3] == beta->ne[3] &&
           (kda || ggml_are_same_stride(g, beta));
}

#endif
