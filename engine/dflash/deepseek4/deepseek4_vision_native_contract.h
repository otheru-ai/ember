// GPU-free contract for the native DeepSeek-V4-Flash-Vision-Exp tower.
//
// This is deliberately separate from deepseek4_vision_contract.h: that file
// owns the language-side learned-token layout, while this file owns the
// mmproj inventory and the preprocessing values consumed by the ViT. Keeping
// both pure makes converter/runtime drift executable without a HIP toolchain.
//
// Source of record (DeepSeek's published minimal inference at
// e46e16bf6035c6f317eb2ac7458eb0362926d402):
//   inference/config.json
//   inference/image_processor.py:25-155
//   inference/vision.py:8-118

#ifndef DFLASH_DEEPSEEK4_VISION_NATIVE_CONTRACT_H
#define DFLASH_DEEPSEEK4_VISION_NATIVE_CONTRACT_H

#include <cstdint>
#include <string>
#include <vector>

namespace dflash {

enum class Deepseek4VisionStorage {
    F32,
    F16,
};

struct Deepseek4VisionTensorSpec {
    std::string name;
    // GGML ne[] order, not source-framework shape order.
    std::vector<int64_t> shape;
    Deepseek4VisionStorage storage = Deepseek4VisionStorage::F32;
};

struct Deepseek4VisionNativeConfig {
    int block_count = 32;
    int embedding_length = 1024;
    int head_count = 16;
    int feed_forward_length = 2816;
    int patch_size = 14;
    int scale_factor = 3;
    int image_min_pixels = 147456;
    int max_n_token = 384;
    float max_wh_ratio = 8.0f;
    float norm_epsilon = 1.0e-6f;
    float rope_theta = 10000.0f;
    int nominal_image_size = 798;
    int output_embedding_length = 4096;
};

struct Deepseek4VisionResizePlan {
    int source_height = 0;
    int source_width = 0;
    int effective_height = 0;
    int effective_width = 0;
    int resized_height = 0;
    int resized_width = 0;
    int n_vit_h = 0;
    int n_vit_w = 0;
    int n_llm_h = 0;
    int n_llm_w = 0;
    int image_rows = 0;
    int block_tokens = 0;
    // image_processor.py directly resizes panoramas at or above the configured
    // ratio; all other images use ImageOps.pad's aspect-preserving path.
    bool panoramic_direct_resize = false;
};

const Deepseek4VisionNativeConfig & deepseek4_vision_native_config();

// Exact 299-name contract emitted by the Vision-Exp converter. Matrix shapes
// are expressed in GGML ne[] order and the expected F16/F32 partition is part
// of the contract rather than inferred from rank.
std::vector<Deepseek4VisionTensorSpec> deepseek4_vision_tensor_specs();

// Reproduce image_processor.py's dynamic size policy. start_pos affects the
// leading four-token alignment padding in the final block, not the resize
// budget (which reserves the worst-case three leading pads).
bool deepseek4_vision_resize_plan(
    int source_height, int source_width, int start_pos,
    Deepseek4VisionResizePlan & out, std::string * error = nullptr);

// Pack an already resized RGB8 image exactly as image_processor.py does:
// patch-row major, then patch-column, channel, local y, local x. Values are
// normalized through float32 and rounded to bfloat16. Image decoding/resizing
// remains outside this helper so no decoder can silently choose a different
// resampling policy.
bool deepseek4_vision_pack_rgb8_patches(
    const uint8_t * rgb, int height, int width,
    std::vector<uint16_t> & bf16_patches,
    std::string * error = nullptr);

// Per-patch 2D RoPE angles before cos/sin: the first half is the row position
// and the second half the column position, each scaled by the official inverse
// frequencies. The returned row width is head_dim/2 (32 for this model).
bool deepseek4_vision_rope_angles(
    int n_vit_h, int n_vit_w, std::vector<float> & angles,
    std::string * error = nullptr);

// Indices for the aligner's F.unfold(kernel=3,stride=3) input. The output is
// [n_llm_h*n_llm_w, 9] in patch-position order; the graph transposes the
// gathered [channel,9,row] tensor so 9 is contiguous inside each channel,
// matching PyTorch's channel-major unfold layout.
bool deepseek4_vision_pixel_shuffle_indices(
    int n_vit_h, int n_vit_w,
    int & padded_h, int & padded_w,
    std::vector<int32_t> & indices,
    std::string * error = nullptr);

}  // namespace dflash

#endif  // DFLASH_DEEPSEEK4_VISION_NATIVE_CONTRACT_H
