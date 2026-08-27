#include "cpu_embedder.h"

namespace dflash::common {

bool CpuEmbedder::embed(const int32_t * ids, int n, float * out_f32) const {
    if (n < 0 || (n > 0 && (!ids || !out_f32)) || !tok_embd_bytes ||
        tok_embd_type == GGML_TYPE_COUNT || n_embd <= 0 || n_vocab <= 0 ||
        row_bytes == 0) return false;
    const ggml_type_traits * traits = ggml_get_type_traits(tok_embd_type);
    if (!traits || !traits->to_float) return false;

    for (int i = 0; i < n; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || id >= n_vocab) return false;
        const uint8_t * row = tok_embd_bytes + static_cast<size_t>(id) * row_bytes;
        traits->to_float(row, out_f32 + static_cast<size_t>(i) * static_cast<size_t>(n_embd),
                         n_embd);
    }
    return true;
}

} // namespace dflash::common
