#include "qwen4exp_vision_loader.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace dflash::common {
namespace {

Qwen4ExpVisionTensorSpec tensor(std::string name,
                                std::initializer_list<int64_t> dimensions) {
    Qwen4ExpVisionTensorSpec result;
    result.name = std::move(name);
    result.n_dims = static_cast<int>(dimensions.size());
    size_t i = 0;
    for (int64_t dimension : dimensions) result.ne[i++] = dimension;
    return result;
}

bool checked_mul(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    out = a * b;
    return true;
}

constexpr uint64_t kResizeFactor =
    Qwen4ExpVisionContract::patch_size *
    Qwen4ExpVisionContract::spatial_merge_size;
constexpr uint64_t kMinimumPixels = 65536;
constexpr uint64_t kMaximumPixels = 16777216;

// Python round() is ties-to-even, unlike std::round. The first smart_resize
// step operates on an integer divided by the integer factor, so implement it
// without depending on the process floating-point rounding mode.
uint64_t round_to_factor_even(uint32_t value) {
    uint64_t quotient = static_cast<uint64_t>(value) / kResizeFactor;
    const uint64_t remainder = static_cast<uint64_t>(value) % kResizeFactor;
    const uint64_t halfway = kResizeFactor / 2;
    if (remainder > halfway ||
        (remainder == halfway && quotient % 2 != 0)) {
        ++quotient;
    }
    return quotient * kResizeFactor;
}

bool scaled_dimension(double value, bool round_up, uint32_t & out) {
    const double units = value / static_cast<double>(kResizeFactor);
    const double rounded = round_up ? std::ceil(units) : std::floor(units);
    const double pixels = std::max(1.0, rounded) *
                          static_cast<double>(kResizeFactor);
    if (!std::isfinite(pixels) ||
        pixels > static_cast<double>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    out = static_cast<uint32_t>(pixels);
    return true;
}

} // namespace

bool qwen4exp_image_smart_resize(
    const Qwen4ExpImageSize & input, Qwen4ExpImageSize & resized,
    Qwen4ExpVisionGrid & grid, std::string & error) {
    error.clear();
    resized = {};
    grid = {};
    if (input.height == 0 || input.width == 0) {
        error = "Qwen4Exp image dimensions must be nonzero";
        return false;
    }
    const uint64_t short_side = std::min(input.height, input.width);
    const uint64_t long_side = std::max(input.height, input.width);
    if (long_side > short_side * 200) {
        error = "Qwen4Exp image aspect ratio must not exceed 200";
        return false;
    }

    const uint64_t rounded_h = round_to_factor_even(input.height);
    const uint64_t rounded_w = round_to_factor_even(input.width);
    if (rounded_h != 0 && rounded_w > UINT64_MAX / rounded_h) {
        error = "Qwen4Exp rounded image area overflows";
        return false;
    }
    const uint64_t rounded_pixels = rounded_h * rounded_w;
    if (rounded_pixels > kMaximumPixels) {
        const double beta = std::sqrt(
            (static_cast<double>(input.height) * input.width) /
            static_cast<double>(kMaximumPixels));
        if (!scaled_dimension(input.height / beta, false, resized.height) ||
            !scaled_dimension(input.width / beta, false, resized.width)) {
            error = "Qwen4Exp maximum-pixel resize is out of range";
            return false;
        }
    } else if (rounded_pixels < kMinimumPixels) {
        const double beta = std::sqrt(
            static_cast<double>(kMinimumPixels) /
            (static_cast<double>(input.height) * input.width));
        if (!scaled_dimension(input.height * beta, true, resized.height) ||
            !scaled_dimension(input.width * beta, true, resized.width)) {
            error = "Qwen4Exp minimum-pixel resize is out of range";
            return false;
        }
    } else {
        if (rounded_h > std::numeric_limits<uint32_t>::max() ||
            rounded_w > std::numeric_limits<uint32_t>::max()) {
            error = "Qwen4Exp rounded image dimensions are out of range";
            return false;
        }
        resized.height = static_cast<uint32_t>(rounded_h);
        resized.width = static_cast<uint32_t>(rounded_w);
    }

    grid = {1,
            resized.height / Qwen4ExpVisionContract::patch_size,
            resized.width / Qwen4ExpVisionContract::patch_size};
    if (qwen4exp_vision_merged_tokens(grid) == 0) {
        error = "Qwen4Exp smart resize produced an invalid patch grid";
        resized = {};
        grid = {};
        return false;
    }
    return true;
}

bool qwen4exp_patchify_normalized_rgb(
    const std::vector<float> & normalized_rgb,
    const Qwen4ExpImageSize & resized,
    std::vector<float> & flattened_patches, Qwen4ExpVisionGrid & grid,
    std::string & error) {
    error.clear();
    flattened_patches.clear();
    grid = {};
    constexpr uint32_t patch = Qwen4ExpVisionContract::patch_size;
    constexpr uint32_t merge = Qwen4ExpVisionContract::spatial_merge_size;
    constexpr uint32_t temporal =
        Qwen4ExpVisionContract::temporal_patch_size;
    constexpr uint32_t channels = Qwen4ExpVisionContract::input_channels;
    const uint32_t factor = patch * merge;
    if (resized.height == 0 || resized.width == 0 ||
        resized.height % factor != 0 || resized.width % factor != 0) {
        error = "Qwen4Exp resized image dimensions must be divisible by 32";
        return false;
    }
    size_t pixels = 0;
    size_t input_values = 0;
    if (!checked_mul(static_cast<size_t>(resized.height), resized.width,
                     pixels) ||
        !checked_mul(pixels, channels, input_values) ||
        normalized_rgb.size() != input_values) {
        error = "Qwen4Exp normalized RGB input must be planar [3,H,W]";
        return false;
    }
    grid = {1, resized.height / patch, resized.width / patch};
    size_t output_values = 0;
    constexpr size_t values_per_patch =
        static_cast<size_t>(channels) * temporal * patch * patch;
    if (!checked_mul(static_cast<size_t>(grid.h), grid.w, output_values) ||
        !checked_mul(output_values, values_per_patch, output_values)) {
        error = "Qwen4Exp flattened patch matrix overflows";
        grid = {};
        return false;
    }
    flattened_patches.reserve(output_values);
    const size_t plane = pixels;
    for (uint32_t block_h = 0; block_h < grid.h / merge; ++block_h) {
        for (uint32_t block_w = 0; block_w < grid.w / merge; ++block_w) {
            for (uint32_t inner_h = 0; inner_h < merge; ++inner_h) {
                for (uint32_t inner_w = 0; inner_w < merge; ++inner_w) {
                    for (uint32_t channel = 0; channel < channels; ++channel) {
                        for (uint32_t frame = 0; frame < temporal; ++frame) {
                            (void)frame; // still images repeat along temporal-2
                            for (uint32_t patch_h = 0; patch_h < patch; ++patch_h) {
                                const size_t row = static_cast<size_t>(
                                    (block_h * merge + inner_h) * patch + patch_h);
                                for (uint32_t patch_w = 0; patch_w < patch; ++patch_w) {
                                    const size_t column = static_cast<size_t>(
                                        (block_w * merge + inner_w) * patch + patch_w);
                                    flattened_patches.push_back(
                                        normalized_rgb[static_cast<size_t>(channel) * plane +
                                                       row * resized.width + column]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (flattened_patches.size() != output_values) {
        error = "internal Qwen4Exp patchify size mismatch";
        flattened_patches.clear();
        grid = {};
        return false;
    }
    return true;
}

std::vector<Qwen4ExpVisionTensorSpec> qwen4exp_vision_tensor_contract() {
    std::vector<Qwen4ExpVisionTensorSpec> tensors;
#include "qwen4exp_vision_inventory.inc"
    return tensors;
}

bool qwen4exp_validate_vision_tensor_inventory(
    const std::vector<Qwen4ExpVisionTensorSpec> & tensors,
    std::string & error) {
    error.clear();
    const auto expected = qwen4exp_vision_tensor_contract();
    if (tensors.size() != expected.size()) {
        error = "Qwen4Exp vision tensor count mismatch: expected " +
                std::to_string(expected.size()) + ", got " +
                std::to_string(tensors.size());
        return false;
    }
    std::unordered_map<std::string, const Qwen4ExpVisionTensorSpec *> by_name;
    by_name.reserve(tensors.size());
    for (const Qwen4ExpVisionTensorSpec & actual : tensors) {
        if (!by_name.emplace(actual.name, &actual).second) {
            error = "duplicate Qwen4Exp vision tensor: " + actual.name;
            return false;
        }
    }
    for (const Qwen4ExpVisionTensorSpec & wanted : expected) {
        const auto found = by_name.find(wanted.name);
        if (found == by_name.end()) {
            error = "missing Qwen4Exp vision tensor: " + wanted.name;
            return false;
        }
        const Qwen4ExpVisionTensorSpec & actual = *found->second;
        if (actual.n_dims != wanted.n_dims || actual.ne != wanted.ne) {
            error = "Qwen4Exp vision tensor shape mismatch: " + wanted.name;
            return false;
        }
    }
    return true;
}

bool qwen4exp_encode_vision_patches(
    const Qwen4ExpVisionEncoderProvider * provider,
    const std::vector<float> & flattened_patches,
    const Qwen4ExpVisionGrid & grid, Qwen4ExpEncodedImage & out,
    std::string & error) {
    error.clear();
    out = {};
    if (grid.t != 1) {
        error = "Qwen4Exp image grid T must be 1; video encoder is not implemented";
        return false;
    }
    const size_t merged_rows = qwen4exp_vision_merged_tokens(grid);
    if (merged_rows == 0) {
        error = "Qwen4Exp vision grid must be nonzero and divisible by the 2x2 merger";
        return false;
    }
    size_t patch_rows = static_cast<size_t>(grid.t);
    size_t input_values = 0;
    constexpr size_t values_per_patch =
        Qwen4ExpVisionContract::input_channels *
        Qwen4ExpVisionContract::temporal_patch_size *
        Qwen4ExpVisionContract::patch_size *
        Qwen4ExpVisionContract::patch_size;
    if (!checked_mul(patch_rows, static_cast<size_t>(grid.h), patch_rows) ||
        !checked_mul(patch_rows, static_cast<size_t>(grid.w), patch_rows) ||
        !checked_mul(patch_rows, values_per_patch, input_values) ||
        flattened_patches.size() != input_values) {
        error = "Qwen4Exp flattened patches must be [T*H*W,1536]";
        return false;
    }
    if (!provider || !provider->encode) {
        error = "Qwen4Exp vision encoder is not installed";
        return false;
    }

    std::vector<float> encoded;
    if (!provider->encode(provider->context, flattened_patches.data(),
                          flattened_patches.size(), grid, encoded, error)) {
        if (error.empty()) error = "Qwen4Exp vision encoder failed";
        return false;
    }
    size_t expected_values = 0;
    if (!checked_mul(merged_rows,
                     Qwen4ExpVisionContract::output_hidden_size,
                     expected_values) || encoded.size() != expected_values) {
        error = "Qwen4Exp vision encoder returned a non-[merged_tokens,2560] matrix";
        return false;
    }
    out.grid = grid;
    out.embeddings = std::move(encoded);
    return true;
}

} // namespace dflash::common
