#include "deepseek4_vision_native_contract.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace dflash {
namespace {

constexpr int kCompressPadTo = 4;

void set_error(std::string * error, const char * text) {
    if (error) *error = text;
}

bool checked_grid_tokens(int best_height, int best_width,
                         const Deepseek4VisionNativeConfig & config,
                         int & n_llm_h, int & n_llm_w, int & tokens) {
    if (best_height <= 0 || best_width <= 0 ||
        best_height % config.patch_size != 0 ||
        best_width % config.patch_size != 0) {
        return false;
    }
    n_llm_h = (best_height / config.patch_size +
               config.scale_factor - 1) / config.scale_factor;
    n_llm_w = (best_width / config.patch_size +
               config.scale_factor - 1) / config.scale_factor;
    const int64_t row_len = static_cast<int64_t>(n_llm_w) + 1;
    int64_t count = static_cast<int64_t>(n_llm_h) * row_len + 2;
    if (n_llm_h % 2 == 1) count += row_len;
    count += ((static_cast<int64_t>(n_llm_h) + 1) / 2 * row_len) % 2 * 2;
    if (n_llm_h <= 0 || n_llm_w <= 0 || count > INT32_MAX) return false;
    tokens = static_cast<int>(count);
    return true;
}

bool solve_resize_ratio(int height, int width, int budget,
                        const Deepseek4VisionNativeConfig & config,
                        int & best_height, int & best_width) {
    if (height <= 0 || width <= 0 || budget <= 2) return false;
    const double ratio = static_cast<double>(height) / width;
    const double max_w_float =
        std::sqrt((static_cast<double>(budget) - 2.0) / ratio + 0.25) - 0.5;
    const double max_h_float = max_w_float * ratio;
    if (max_w_float < 1.0) {
        const int max_w = 1;
        int max_h = (budget - 2) / (max_w + 1);
        if (max_h % 2 == 1) --max_h;
        if (max_h <= 0) return false;
        best_width = max_w * config.patch_size * config.scale_factor;
        best_height = max_h * config.patch_size * config.scale_factor;
    } else if (max_h_float < 2.0) {
        const int max_h = 2;
        const int max_w = (budget - 2) / max_h - 1;
        if (max_w <= 1) return false;
        best_width = max_w * config.patch_size * config.scale_factor;
        best_height = max_h * config.patch_size * config.scale_factor;
    } else {
        const int max_w = static_cast<int>(std::floor(max_w_float));
        int max_h = static_cast<int>(std::floor(max_h_float));
        if (max_h % 2 == 1) --max_h;
        if (max_w <= 0 || max_h <= 0) return false;
        const double beta = std::min(
            static_cast<double>(max_w * config.patch_size * config.scale_factor) /
                width,
            static_cast<double>(max_h * config.patch_size * config.scale_factor) /
                height);
        best_width = static_cast<int>(
            std::floor(width * beta / config.patch_size)) * config.patch_size;
        best_height = static_cast<int>(
            std::floor(height * beta / config.patch_size)) * config.patch_size;
    }
    return best_height > 0 && best_width > 0;
}

uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

void add_spec(std::vector<Deepseek4VisionTensorSpec> & specs,
              std::string name, std::initializer_list<int64_t> shape,
              Deepseek4VisionStorage storage) {
    specs.push_back({std::move(name), shape, storage});
}

}  // namespace

const Deepseek4VisionNativeConfig & deepseek4_vision_native_config() {
    static const Deepseek4VisionNativeConfig config;
    return config;
}

std::vector<Deepseek4VisionTensorSpec> deepseek4_vision_tensor_specs() {
    using Storage = Deepseek4VisionStorage;
    std::vector<Deepseek4VisionTensorSpec> specs;
    specs.reserve(299);
    add_spec(specs, "v.patch_embd.weight", {588, 1024}, Storage::F16);
    add_spec(specs, "v.patch_embd.bias", {1024}, Storage::F32);
    for (int layer = 0; layer < 32; ++layer) {
        const std::string prefix = "v.blk." + std::to_string(layer) + ".";
        add_spec(specs, prefix + "ln1.weight", {1024}, Storage::F32);
        add_spec(specs, prefix + "attn_qkv.weight", {1024, 3072}, Storage::F16);
        add_spec(specs, prefix + "attn_qkv.bias", {3072}, Storage::F32);
        add_spec(specs, prefix + "attn_out.weight", {1024, 1024}, Storage::F16);
        add_spec(specs, prefix + "attn_out.bias", {1024}, Storage::F32);
        add_spec(specs, prefix + "ln2.weight", {1024}, Storage::F32);
        add_spec(specs, prefix + "ffn_gate.weight", {1024, 2816}, Storage::F16);
        add_spec(specs, prefix + "ffn_up.weight", {1024, 2816}, Storage::F16);
        add_spec(specs, prefix + "ffn_down.weight", {2816, 1024}, Storage::F16);
    }
    add_spec(specs, "v.post_ln.weight", {1024}, Storage::F32);
    add_spec(specs, "mm.1.weight", {9216, 4096}, Storage::F16);
    add_spec(specs, "mm.1.bias", {4096}, Storage::F32);
    add_spec(specs, "mm.2.weight", {4096, 4096}, Storage::F16);
    add_spec(specs, "mm.2.bias", {4096}, Storage::F32);
    add_spec(specs, "v.image_newline.weight", {4096, 1}, Storage::F32);
    add_spec(specs, "mm.image_begin.weight", {4096, 1}, Storage::F32);
    add_spec(specs, "mm.image_end.weight", {4096, 1}, Storage::F32);
    add_spec(specs, "mm.image_pad.weight", {4096, 1}, Storage::F32);
    return specs;
}

bool deepseek4_vision_resize_plan(
        int source_height, int source_width, int start_pos,
        Deepseek4VisionResizePlan & out, std::string * error) {
    out = {};
    const auto & config = deepseek4_vision_native_config();
    if (source_height <= 0 || source_width <= 0 || start_pos < 0) {
        set_error(error, "invalid DeepSeek4 image dimensions or start position");
        return false;
    }
    int height = source_height;
    int width = source_width;
    if (static_cast<double>(width) >
        static_cast<double>(height) * config.max_wh_ratio) {
        width = static_cast<int>(static_cast<double>(height) *
                                 static_cast<double>(config.max_wh_ratio));
    }
    const int64_t pixels = static_cast<int64_t>(width) * height;
    if (pixels > 0 && pixels < config.image_min_pixels) {
        const double ratio = std::sqrt(
            static_cast<double>(config.image_min_pixels) /
            static_cast<double>(pixels));
        width = static_cast<int>(width * ratio);
        height = static_cast<int>(height * ratio);
    }
    if (height <= 0 || width <= 0 ||
        height > INT32_MAX - config.patch_size ||
        width > INT32_MAX - config.patch_size) {
        set_error(error, "DeepSeek4 image dimensions overflow resize planning");
        return false;
    }
    int best_height = ((height + config.patch_size - 1) /
                       config.patch_size) * config.patch_size;
    int best_width = ((width + config.patch_size - 1) /
                      config.patch_size) * config.patch_size;
    const int max_tokens = config.max_n_token - (kCompressPadTo - 1);
    int n_llm_h = 0;
    int n_llm_w = 0;
    int tokens = 0;
    if (!checked_grid_tokens(best_height, best_width, config,
                             n_llm_h, n_llm_w, tokens)) {
        set_error(error, "invalid DeepSeek4 image grid");
        return false;
    }
    int budget = max_tokens;
    while (tokens > max_tokens) {
        if (!solve_resize_ratio(height, width, budget, config,
                                best_height, best_width) ||
            !checked_grid_tokens(best_height, best_width, config,
                                 n_llm_h, n_llm_w, tokens)) {
            set_error(error, "DeepSeek4 image cannot fit the token budget");
            return false;
        }
        --budget;
        if (budget <= 2) {
            set_error(error, "DeepSeek4 image resize budget was exhausted");
            return false;
        }
    }
    const int64_t image_rows = static_cast<int64_t>(n_llm_h) * n_llm_w;
    const int leading_pad = kCompressPadTo - 1 - start_pos % kCompressPadTo;
    const int64_t block_tokens = static_cast<int64_t>(tokens) + leading_pad;
    if (image_rows > INT32_MAX || block_tokens > config.max_n_token) {
        set_error(error, "DeepSeek4 image block exceeds its token budget");
        return false;
    }
    out.source_height = source_height;
    out.source_width = source_width;
    out.effective_height = height;
    out.effective_width = width;
    out.resized_height = best_height;
    out.resized_width = best_width;
    out.n_vit_h = best_height / config.patch_size;
    out.n_vit_w = best_width / config.patch_size;
    out.n_llm_h = n_llm_h;
    out.n_llm_w = n_llm_w;
    out.image_rows = static_cast<int>(image_rows);
    out.block_tokens = static_cast<int>(block_tokens);
    out.panoramic_direct_resize =
        static_cast<double>(source_width) >=
        static_cast<double>(config.max_wh_ratio) *
            static_cast<double>(source_height);
    return true;
}

bool deepseek4_vision_pack_rgb8_patches(
        const uint8_t * rgb, int height, int width,
        std::vector<uint16_t> & bf16_patches, std::string * error) {
    bf16_patches.clear();
    const int patch = deepseek4_vision_native_config().patch_size;
    if (!rgb || height <= 0 || width <= 0 ||
        height % patch != 0 || width % patch != 0) {
        set_error(error, "resized DeepSeek4 RGB image is not patch aligned");
        return false;
    }
    const int64_t pixels = static_cast<int64_t>(height) * width;
    if (pixels > static_cast<int64_t>(
            std::numeric_limits<size_t>::max() / (3 * sizeof(uint16_t)))) {
        set_error(error, "DeepSeek4 RGB patch matrix is too large");
        return false;
    }
    bf16_patches.resize(static_cast<size_t>(pixels) * 3);
    size_t destination = 0;
    for (int patch_y = 0; patch_y < height; patch_y += patch) {
        for (int patch_x = 0; patch_x < width; patch_x += patch) {
            for (int channel = 0; channel < 3; ++channel) {
                for (int y = 0; y < patch; ++y) {
                    for (int x = 0; x < patch; ++x) {
                        const size_t source =
                            (static_cast<size_t>(patch_y + y) *
                                 static_cast<size_t>(width) +
                             static_cast<size_t>(patch_x + x)) * 3u +
                            static_cast<size_t>(channel);
                        float value = static_cast<float>(rgb[source]) / 255.0f;
                        value = (value - 0.5f) / 0.5f;
                        bf16_patches[destination++] = float_to_bf16(value);
                    }
                }
            }
        }
    }
    return true;
}

bool deepseek4_vision_rope_angles(
        int n_vit_h, int n_vit_w, std::vector<float> & angles,
        std::string * error) {
    angles.clear();
    const auto & config = deepseek4_vision_native_config();
    const int head_dim = config.embedding_length / config.head_count;
    const int rope_dim = head_dim / 2;
    const int frequencies = rope_dim / 2;
    if (n_vit_h <= 0 || n_vit_w <= 0 || head_dim % 4 != 0) {
        set_error(error, "invalid DeepSeek4 vision RoPE grid");
        return false;
    }
    const int64_t rows = static_cast<int64_t>(n_vit_h) * n_vit_w;
    if (rows > static_cast<int64_t>(
            std::numeric_limits<size_t>::max() /
            (static_cast<size_t>(rope_dim) * sizeof(float)))) {
        set_error(error, "DeepSeek4 vision RoPE table is too large");
        return false;
    }
    std::vector<float> inverse(static_cast<size_t>(frequencies));
    for (int i = 0; i < frequencies; ++i) {
        const float exponent = static_cast<float>(2 * i) /
                               static_cast<float>(rope_dim);
        inverse[static_cast<size_t>(i)] =
            1.0f / std::pow(config.rope_theta, exponent);
    }
    angles.resize(static_cast<size_t>(rows) *
                  static_cast<size_t>(rope_dim));
    for (int h = 0; h < n_vit_h; ++h) {
        for (int w = 0; w < n_vit_w; ++w) {
            float * row = angles.data() +
                (static_cast<size_t>(h) * static_cast<size_t>(n_vit_w) +
                 static_cast<size_t>(w)) * static_cast<size_t>(rope_dim);
            for (int i = 0; i < frequencies; ++i) {
                row[i] = static_cast<float>(h) * inverse[static_cast<size_t>(i)];
                row[frequencies + i] =
                    static_cast<float>(w) * inverse[static_cast<size_t>(i)];
            }
        }
    }
    return true;
}

}  // namespace dflash
