#include "qwen4exp_vision_provider.h"

#include <cstdlib>
#include <cstring>

using namespace dflash::common;

namespace {
void * create(const char *, const char *, int, char *, size_t) {
    return reinterpret_cast<void *>(1);
}
void destroy(void *) {}
bool encode(void *, const uint8_t * bytes, size_t size,
            qwen4exp_vision_provider_output_v1 * out, char *, size_t) {
    if (!bytes || size == 0 || !out) return false;
    *out = {};
    out->grid_t = 1; out->grid_h = 2; out->grid_w = 2;
    out->embedding_width = std::getenv("EMBER_TEST_BAD_VISION_PROVIDER") ? 1 : 2560;
    out->row_count = 1;
    out->rows = static_cast<float *>(
        std::malloc(static_cast<size_t>(out->embedding_width) * sizeof(float)));
    if (!out->rows) return false;
    for (uint32_t i = 0; i < out->embedding_width; ++i)
        out->rows[i] = static_cast<float>(bytes[0]) + static_cast<float>(i);
    return true;
}
void free_output(void *, qwen4exp_vision_provider_output_v1 * out) {
    std::free(out->rows); *out = {};
}
const qwen4exp_vision_provider_v1 api = {
    kQwen4ExpVisionProviderAbi, create, destroy, encode, free_output};
} // namespace

extern "C" const qwen4exp_vision_provider_v1 *
qwen4exp_vision_provider_get_v1() { return &api; }
