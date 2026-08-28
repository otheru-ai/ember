#include "qwen4exp_vision_provider.h"
#include "common/model_backend.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <thread>
#include <vector>

using namespace dflash::common;
static int g_pass, g_fail;
#define CHECK(c, m) do { if (c) ++g_pass; else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", m); } } while (0)

template <typename T>
static T load_symbol(void * library, const char * name) {
    void * raw = dlsym(library, name);
    T symbol = nullptr;
    static_assert(sizeof(symbol) == sizeof(raw));
    std::memcpy(&symbol, &raw, sizeof(symbol));
    return symbol;
}

int main(int argc, char ** argv) {
    CHECK(argc == 2, "mock provider path supplied");
    if (argc != 2) return 1;
    unsetenv("DFLASH_QWEN_VISION_PROVIDER");
    unsetenv("DFLASH_QWEN_VISION_MMPROJ");
    const uint8_t bytes[] = {7, 8, 9};
    EncodedVisionImage out;
    std::string error;
    Qwen4ExpLazyVisionProvider absent("/unused/text.gguf", 0);
    CHECK(!absent.encode(bytes, sizeof(bytes), out, error) &&
          error.find("DFLASH_QWEN_VISION_PROVIDER") != std::string::npos,
          "provider is lazy and absent configuration fails descriptively");

    void * mock_library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    CHECK(mock_library != nullptr, "mock provider opens for test instrumentation");
    using reset_stats_fn = void (*)();
    using read_stat_fn = int (*)();
    reset_stats_fn reset_stats = mock_library
        ? load_symbol<reset_stats_fn>(mock_library,
              "ember_test_vision_provider_reset_stats") : nullptr;
    read_stat_fn create_calls = mock_library
        ? load_symbol<read_stat_fn>(mock_library,
              "ember_test_vision_provider_create_calls") : nullptr;
    read_stat_fn max_active = mock_library
        ? load_symbol<read_stat_fn>(mock_library,
              "ember_test_vision_provider_max_active_encodes") : nullptr;
    CHECK(reset_stats && create_calls && max_active,
          "mock provider exposes concurrency instrumentation");

    setenv("DFLASH_QWEN_VISION_PROVIDER", argv[1], 1);
    setenv("DFLASH_QWEN_VISION_MMPROJ", "/unused/mock-mmproj.gguf", 1);
    Qwen4ExpLazyVisionProvider provider("/unused/text.gguf", 0);
    if (reset_stats) reset_stats();
    setenv("EMBER_TEST_SLOW_VISION_PROVIDER", "1", 1);
    constexpr int n_threads = 8;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::vector<int> encoded(static_cast<size_t>(n_threads), 0);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        threads.emplace_back([&, i] {
            ready.fetch_add(1);
            while (!start.load()) std::this_thread::yield();
            EncodedVisionImage thread_out;
            std::string thread_error;
            const uint8_t thread_bytes[] = {
                static_cast<uint8_t>(7 + i), 8, 9};
            encoded[static_cast<size_t>(i)] = provider.encode(
                thread_bytes, sizeof(thread_bytes), thread_out, thread_error) &&
                thread_out.embeddings.size() == kQwen4ExpVisionEmbeddingWidth &&
                thread_out.embeddings[0] == static_cast<float>(7 + i);
        });
    }
    while (ready.load() != n_threads) std::this_thread::yield();
    start.store(true);
    for (std::thread & thread : threads) thread.join();
    unsetenv("EMBER_TEST_SLOW_VISION_PROVIDER");
    bool all_encoded = true;
    for (int value : encoded) all_encoded = all_encoded && value != 0;
    CHECK(all_encoded, "concurrent provider calls all encode exactly");
    CHECK(create_calls && create_calls() == 1,
          "concurrent first use creates one provider context");
    CHECK(max_active && max_active() == 1,
          "provider serializes encode access to its mtmd context");

    CHECK(provider.encode(bytes, sizeof(bytes), out, error),
          "provider remains usable after concurrent first use");
    CHECK(out.grid_t == 1 && out.grid_h == 2 && out.grid_w == 2 &&
          out.embeddings.size() == kQwen4ExpVisionEmbeddingWidth &&
          out.embeddings[0] == 7.0f,
          "provider result is copied exactly");
    setenv("EMBER_TEST_BAD_VISION_PROVIDER", "1", 1);
    CHECK(!provider.encode(bytes, sizeof(bytes), out, error) &&
          out.embeddings.empty() &&
          error.find("invalid image embedding contract") != std::string::npos,
          "provider width mismatch fails closed before exposing embeddings");
    unsetenv("EMBER_TEST_BAD_VISION_PROVIDER");
    if (mock_library) dlclose(mock_library);
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
