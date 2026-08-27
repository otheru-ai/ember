// CPU token-embedding lookup for the DeepSeek4 target.
//
// The target embedding tensor stays in host memory because the HIP get_rows
// path does not support every quant format used by released checkpoints.

#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dflash::common {

struct CpuEmbedder {
    const uint8_t * tok_embd_bytes = nullptr;
    ggml_type       tok_embd_type  = GGML_TYPE_COUNT;
    int64_t         n_embd         = 0;
    int64_t         n_vocab        = 0;
    size_t          row_bytes      = 0;
    std::vector<uint8_t> tok_embd_owned;

    bool embed(const int32_t * ids, int n, float * out_f32) const;
};

} // namespace dflash::common
