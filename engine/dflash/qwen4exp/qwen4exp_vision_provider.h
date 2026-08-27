// Lazy external projector ABI for Qwen3.8-Flash-Next.
//
// llama.cpp converts the stock Qwen3-VL tower to a separate BF16 `--mmproj`
// GGUF. Its public mtmd API cannot encode an image without a llama_model/vocab,
// while Ember already owns a different text runtime and must not load a second
// ~90 GiB model merely to tokenize media markers. A tiny adapter built beside
// llama.cpp's internal clip/mtmd code can export this C table instead. Keeping
// that dependency dynamic also means ordinary text requests never map the
// roughly 2 GiB tower on a 128-GiB UMA device.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace dflash::common {

struct EncodedVisionImage;

constexpr uint32_t kQwen4ExpVisionProviderAbi = 1;

extern "C" {

struct qwen4exp_vision_provider_output_v1 {
    uint32_t grid_t;
    uint32_t grid_h;
    uint32_t grid_w;
    uint32_t embedding_width;
    size_t row_count;
    float *rows;
};

struct qwen4exp_vision_provider_v1 {
    uint32_t abi_version;
    void * (*create)(const char * mmproj_path, const char * text_model_path,
                     int gpu,
                     char * error, size_t error_cap);
    void (*destroy)(void * context);
    bool (*encode)(void * context, const uint8_t * encoded, size_t encoded_size,
                   qwen4exp_vision_provider_output_v1 * output,
                   char * error, size_t error_cap);
    void (*free_output)(void * context,
                        qwen4exp_vision_provider_output_v1 * output);
};

using qwen4exp_vision_provider_entry_v1 =
    const qwen4exp_vision_provider_v1 * (*)();

} // extern "C"

class Qwen4ExpLazyVisionProvider {
public:
    Qwen4ExpLazyVisionProvider(std::string text_model_path, int gpu);
    ~Qwen4ExpLazyVisionProvider();
    Qwen4ExpLazyVisionProvider(const Qwen4ExpLazyVisionProvider &) = delete;
    Qwen4ExpLazyVisionProvider & operator=(
        const Qwen4ExpLazyVisionProvider &) = delete;

    bool encode(const uint8_t * encoded, size_t encoded_size,
                EncodedVisionImage & out, std::string & error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dflash::common
