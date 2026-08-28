#include "qwen4exp_vision_provider.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace dflash::common;

namespace {
std::atomic<int> g_create_calls{0};
std::atomic<int> g_active_encodes{0};
std::atomic<int> g_max_active_encodes{0};

void * create(const char *, const char *, int, char *, size_t) {
    g_create_calls.fetch_add(1);
    if (std::getenv("EMBER_TEST_SLOW_VISION_PROVIDER"))
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    return reinterpret_cast<void *>(1);
}
void destroy(void *) {}
bool encode(void *, const uint8_t * bytes, size_t size,
            qwen4exp_vision_provider_output_v1 * out, char *, size_t) {
    if (!bytes || size == 0 || !out) return false;
    const int active = g_active_encodes.fetch_add(1) + 1;
    int previous = g_max_active_encodes.load();
    while (previous < active &&
           !g_max_active_encodes.compare_exchange_weak(previous, active)) {}
    if (std::getenv("EMBER_TEST_SLOW_VISION_PROVIDER"))
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    *out = {};
    out->grid_t = 1; out->grid_h = 2; out->grid_w = 2;
    out->embedding_width = std::getenv("EMBER_TEST_BAD_VISION_PROVIDER")
        ? 1 : kQwen4ExpVisionEmbeddingWidth;
    out->row_count = 1;
    out->rows = static_cast<float *>(
        std::malloc(static_cast<size_t>(out->embedding_width) * sizeof(float)));
    if (!out->rows) {
        g_active_encodes.fetch_sub(1);
        return false;
    }
    for (uint32_t i = 0; i < out->embedding_width; ++i)
        out->rows[i] = static_cast<float>(bytes[0]) + static_cast<float>(i);
    g_active_encodes.fetch_sub(1);
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

extern "C" void ember_test_vision_provider_reset_stats() {
    g_create_calls.store(0);
    g_active_encodes.store(0);
    g_max_active_encodes.store(0);
}

extern "C" int ember_test_vision_provider_create_calls() {
    return g_create_calls.load();
}

extern "C" int ember_test_vision_provider_max_active_encodes() {
    return g_max_active_encodes.load();
}
