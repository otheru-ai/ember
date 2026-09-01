// Strict metadata loader for the native DeepSeek-V4 Vision-Exp mmproj.
//
// This stage intentionally loads no tensor payload and creates no backend.
// It proves that a regular GGUF file has the exact typed configuration and
// 299-tensor inventory the future lazy tower graph will bind. Payload offsets
// are bounds-checked and retained so the graph loader cannot reinterpret a
// merely marker-compatible or foreign projector as this architecture.

#ifndef DFLASH_DEEPSEEK4_VISION_MMPROJ_H
#define DFLASH_DEEPSEEK4_VISION_MMPROJ_H

#include "deepseek4_vision_native_contract.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct gguf_context;

namespace dflash {

struct Deepseek4VisionTensorSlice {
    std::string name;
    std::vector<int64_t> shape;
    Deepseek4VisionStorage storage = Deepseek4VisionStorage::F32;
    size_t file_offset = 0;
    size_t byte_count = 0;
};

struct Deepseek4VisionMmprojMetadata {
    Deepseek4VisionNativeConfig config;
    size_t file_size = 0;
    size_t data_offset = 0;
    std::vector<Deepseek4VisionTensorSlice> tensors;
};

// Exposed for metadata-only mutation tests. file_size is the complete GGUF
// size; all tensor slices must fit after gguf_get_data_offset().
bool deepseek4_validate_vision_mmproj_metadata(
    const gguf_context * gguf, ggml_context * meta,
    int32_t model_n_embd, size_t file_size,
    Deepseek4VisionMmprojMetadata & out, std::string & error);

bool deepseek4_load_vision_mmproj_metadata(
    const std::string & path, int32_t model_n_embd,
    Deepseek4VisionMmprojMetadata & out, std::string & error);

}  // namespace dflash

#endif  // DFLASH_DEEPSEEK4_VISION_MMPROJ_H
