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
#include <cstddef>
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
    // Request-facing decode ceilings. The pixel cap is the actual work/memory
    // bound. The larger per-axis cap rejects degenerate strips while still
    // admitting ordinary panoramas and tall scans within that pixel budget.
    size_t max_encoded_bytes = 64u * 1024u * 1024u;
    int max_decode_dimension = 16384;
    uint64_t max_decode_pixels = UINT64_C(4096) * UINT64_C(4096);
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

struct Deepseek4VisionPngInfo {
    int width = 0;
    int height = 0;
    int channels = 0;
    // Exact bytes expected after zlib expansion, including one filter byte per
    // row. A decoder must require this size rather than accept partial output.
    size_t filtered_bytes = 0;
};

const Deepseek4VisionNativeConfig & deepseek4_vision_native_config();

// Fail-closed request-byte preflight for the initial native PNG-only path.
// This walks the complete chunk stream, checks every CRC, requires a terminal
// IEND with no trailing bytes, rejects APNG animation chunks, and applies the
// allocation ceilings before a decoder sees the payload. The initial decoder
// intentionally accepts only non-interlaced RGB/RGBA8; JPEG/WebP/GIF remain
// explicit unsupported formats rather than inheriting permissive defaults.
bool deepseek4_vision_validate_still_png(
    const uint8_t * encoded, size_t encoded_size,
    int max_dimension, uint64_t max_pixels,
    Deepseek4VisionPngInfo & out, std::string * error = nullptr);

// Decode the validated PNG subset to tightly packed RGB8. zlib must consume
// every IDAT byte, produce exactly `filtered_bytes`, and reach STREAM_END;
// truncated, overlong, or partially decodable streams therefore fail rather
// than yielding plausible pixels. RGBA is converted like Pillow's RGB mode by
// dropping alpha, without compositing against an implicit background.
bool deepseek4_vision_decode_still_png_rgb8(
    const uint8_t * encoded, size_t encoded_size,
    int max_dimension, uint64_t max_pixels,
    std::vector<uint8_t> & rgb, Deepseek4VisionPngInfo & info,
    std::string * error = nullptr);

// Apply the exact Pillow RGB/BICUBIC pipeline selected by
// image_processor.py. Non-panoramic inputs use ImageOps.pad with a 127 fill;
// panoramas use a direct resize. The implementation follows Pillow 12.3's
// 22-bit coefficient quantization and horizontal-then-vertical uint8 rounding,
// so the result is an executable compatibility contract rather than a generic
// cubic approximation.
bool deepseek4_vision_resize_rgb8(
    const uint8_t * source_rgb,
    const Deepseek4VisionResizePlan & plan,
    std::vector<uint8_t> & resized_rgb,
    std::string * error = nullptr);

// Complete bounded request-byte preprocessing. The decode ceilings above are
// fixed by the model contract rather than caller-supplied. source_digest is a
// stable nonzero FNV-1a binding of the normalized encoded bytes.
bool deepseek4_vision_preprocess_still_png(
    const uint8_t * encoded, size_t encoded_size, int prompt_offset,
    Deepseek4VisionResizePlan & plan,
    std::vector<uint16_t> & bf16_patches,
    uint64_t & source_digest,
    std::string * error = nullptr);

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
