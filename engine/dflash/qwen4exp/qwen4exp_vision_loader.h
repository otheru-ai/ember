// Qwen3.8-Flash-Next vision tensor inventory and encoder-provider boundary.
//
// PR #27742 converts the released Qwen4Exp tower through the Qwen3-VL mmproj
// path.  The converter splits the temporal-2 Conv3D kernel into two GGUF
// tensors and uses the stock v.blk.*, v.post_ln, and mm.{0,2} names.  Checking
// that exact inventory prevents a partially converted tower from appearing
// usable.  The runtime provider is built from the pinned rotated-KV follow-up
// PR #27774 revision abdc7a0bf815d3b83e26dd523c6960e4dd597e82 and remains
// lazy: an unconfigured or invalid provider fails closed.

#pragma once

#include "qwen4exp_vision.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen4ExpVisionTensorSpec {
    std::string name;
    std::array<int64_t, 4> ne{}; // GGML logical order
    int n_dims = 0;
};

struct Qwen4ExpImageSize {
    uint32_t height = 0;
    uint32_t width = 0;
};

// Exact size oracle for the pinned preprocessor_config.json and
// Qwen2VLImageProcessorFast.smart_resize used by Qwen3VLProcessor. It does not
// decode or resize pixels; it only returns the required resized shape and
// corresponding patch grid.
bool qwen4exp_image_smart_resize(
    const Qwen4ExpImageSize & input, Qwen4ExpImageSize & resized,
    Qwen4ExpVisionGrid & grid, std::string & error);

// Exact patchify step after RGB conversion, bicubic resizing, rescaling, and
// normalization have already happened. `normalized_rgb` is planar [3,H,W].
// Output is the official block-major [H/16*W/16,1536] layout; temporal-2
// duplicates the still image. This deliberately does not invent an image
// decoder or an approximate resize implementation.
bool qwen4exp_patchify_normalized_rgb(
    const std::vector<float> & normalized_rgb,
    const Qwen4ExpImageSize & resized,
    std::vector<float> & flattened_patches, Qwen4ExpVisionGrid & grid,
    std::string & error);

// Exact 334-tensor mmproj inventory emitted by llama.cpp PR #27742 for the
// pinned checkpoint. These are logical GGML dimensions, not PyTorch order.
// The generated implementation is checked against qwen4exp_vision_inventory.json,
// which is also consumed by candidate construction and release packaging.
std::vector<Qwen4ExpVisionTensorSpec> qwen4exp_vision_tensor_contract();

// Input may be in any order but must contain every contract tensor exactly
// once, with no unknown v.* or mm.* entry.
bool qwen4exp_validate_vision_tensor_inventory(
    const std::vector<Qwen4ExpVisionTensorSpec> & tensors,
    std::string & error);

using Qwen4ExpVisionEncodeFn = bool (*)(
    void * context, const float * flattened_patches, size_t value_count,
    const Qwen4ExpVisionGrid & grid, std::vector<float> & merged_embeddings,
    std::string & error);

struct Qwen4ExpVisionEncoderProvider {
    void * context = nullptr;
    Qwen4ExpVisionEncodeFn encode = nullptr;
};

// Executes an installed encoder over processor-produced flattened patches.
// Expected input is [T*H*W, 3*2*16*16], and output is
// [T*(H/2)*(W/2),2560].  A null provider returns a descriptive error; it never
// fabricates or ignores image rows.
bool qwen4exp_encode_vision_patches(
    const Qwen4ExpVisionEncoderProvider * provider,
    const std::vector<float> & flattened_patches,
    const Qwen4ExpVisionGrid & grid, Qwen4ExpEncodedImage & out,
    std::string & error);

} // namespace dflash::common
