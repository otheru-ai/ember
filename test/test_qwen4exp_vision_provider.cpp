#include "qwen4exp_vision_provider.h"
#include "common/model_backend.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace dflash::common;
static int g_pass, g_fail;
#define CHECK(c, m) do { if (c) ++g_pass; else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", m); } } while (0)

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

    setenv("DFLASH_QWEN_VISION_PROVIDER", argv[1], 1);
    setenv("DFLASH_QWEN_VISION_MMPROJ", "/unused/mock-mmproj.gguf", 1);
    Qwen4ExpLazyVisionProvider provider("/unused/text.gguf", 0);
    CHECK(provider.encode(bytes, sizeof(bytes), out, error),
          "configured provider encodes");
    CHECK(out.grid_t == 1 && out.grid_h == 2 && out.grid_w == 2 &&
          out.embeddings.size() == 2560 && out.embeddings[0] == 7.0f,
          "provider result is copied exactly");
    setenv("EMBER_TEST_BAD_VISION_PROVIDER", "1", 1);
    CHECK(!provider.encode(bytes, sizeof(bytes), out, error) &&
          error.find("invalid image embedding contract") != std::string::npos,
          "invalid provider shape fails closed");
    unsetenv("EMBER_TEST_BAD_VISION_PROVIDER");
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
