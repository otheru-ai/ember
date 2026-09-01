#include "deepseek4_vision_native_contract.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <zlib.h>

namespace dflash {
namespace {

constexpr int kCompressPadTo = 4;
constexpr uint8_t kPngSignature[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
};

void set_error(std::string * error, const char * text) {
    if (error) *error = text;
}

uint32_t read_be32(const uint8_t * value) {
    return static_cast<uint32_t>(value[0]) << 24 |
           static_cast<uint32_t>(value[1]) << 16 |
           static_cast<uint32_t>(value[2]) << 8 |
           static_cast<uint32_t>(value[3]);
}

uint32_t png_crc32(const uint8_t * data, size_t size) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return crc ^ 0xffffffffu;
}

bool chunk_is(const uint8_t * type, const char (&name)[5]) {
    return std::memcmp(type, name, 4) == 0;
}

uint8_t paeth_predictor(uint8_t left, uint8_t up, uint8_t upper_left) {
    const int p = static_cast<int>(left) + static_cast<int>(up) -
                  static_cast<int>(upper_left);
    const int pa = std::abs(p - static_cast<int>(left));
    const int pb = std::abs(p - static_cast<int>(up));
    const int pc = std::abs(p - static_cast<int>(upper_left));
    return pa <= pb && pa <= pc ? left : pb <= pc ? up : upper_left;
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

bool deepseek4_vision_validate_still_png(
        const uint8_t * encoded, size_t encoded_size,
        int max_dimension, uint64_t max_pixels,
        Deepseek4VisionPngInfo & out, std::string * error) {
    out = {};
    if (!encoded || encoded_size < sizeof(kPngSignature) + 25u ||
        max_dimension <= 0 || max_pixels == 0) {
        set_error(error, "invalid or truncated DeepSeek4 PNG input");
        return false;
    }
    if (std::memcmp(encoded, kPngSignature, sizeof(kPngSignature)) != 0) {
        set_error(error, "DeepSeek4 native vision accepts PNG only");
        return false;
    }

    size_t cursor = sizeof(kPngSignature);
    bool seen_ihdr = false;
    bool seen_idat = false;
    bool left_idat = false;
    uint64_t idat_bytes = 0;
    Deepseek4VisionPngInfo candidate;
    while (cursor < encoded_size) {
        if (encoded_size - cursor < 12u) {
            set_error(error, "truncated DeepSeek4 PNG chunk");
            return false;
        }
        const uint32_t length = read_be32(encoded + cursor);
        const size_t available = encoded_size - cursor - 12u;
        if (static_cast<uint64_t>(length) >
            static_cast<uint64_t>(available)) {
            set_error(error, "DeepSeek4 PNG chunk length exceeds its payload");
            return false;
        }
        const uint8_t * type = encoded + cursor + 4u;
        const uint8_t * data = type + 4u;
        for (int i = 0; i < 4; ++i) {
            if (!((type[i] >= 'A' && type[i] <= 'Z') ||
                  (type[i] >= 'a' && type[i] <= 'z'))) {
                set_error(error, "DeepSeek4 PNG chunk name is invalid");
                return false;
            }
        }
        if (type[2] < 'A' || type[2] > 'Z') {
            set_error(error, "DeepSeek4 PNG reserved chunk-name bit is invalid");
            return false;
        }
        const uint32_t expected_crc = read_be32(data + length);
        const size_t crc_input = static_cast<size_t>(length) + 4u;
        if (png_crc32(type, crc_input) != expected_crc) {
            set_error(error, "DeepSeek4 PNG chunk CRC mismatch");
            return false;
        }

        if (!seen_ihdr) {
            if (!chunk_is(type, "IHDR") || length != 13u) {
                set_error(error, "DeepSeek4 PNG does not begin with IHDR");
                return false;
            }
            const uint32_t width = read_be32(data);
            const uint32_t height = read_be32(data + 4u);
            const uint8_t bit_depth = data[8];
            const uint8_t color_type = data[9];
            if (width == 0 || height == 0 ||
                width > static_cast<uint32_t>(max_dimension) ||
                height > static_cast<uint32_t>(max_dimension) ||
                static_cast<uint64_t>(width) * height > max_pixels) {
                set_error(error, "DeepSeek4 PNG dimensions exceed the decode ceiling");
                return false;
            }
            if (bit_depth != 8u || (color_type != 2u && color_type != 6u) ||
                data[10] != 0u || data[11] != 0u || data[12] != 0u) {
                set_error(error, "DeepSeek4 PNG must be non-interlaced RGB/RGBA8");
                return false;
            }
            const int channels = color_type == 2u ? 3 : 4;
            const uint64_t row = 1u + static_cast<uint64_t>(width) *
                                      static_cast<uint64_t>(channels);
            const uint64_t filtered = row * static_cast<uint64_t>(height);
            if (filtered > static_cast<uint64_t>(SIZE_MAX)) {
                set_error(error, "DeepSeek4 PNG decoded size is not addressable");
                return false;
            }
            candidate.width = static_cast<int>(width);
            candidate.height = static_cast<int>(height);
            candidate.channels = channels;
            candidate.filtered_bytes = static_cast<size_t>(filtered);
            seen_ihdr = true;
        } else if (chunk_is(type, "IHDR")) {
            set_error(error, "DeepSeek4 PNG contains duplicate IHDR");
            return false;
        } else if (chunk_is(type, "acTL") || chunk_is(type, "fcTL") ||
                   chunk_is(type, "fdAT")) {
            set_error(error, "animated PNG is not a supported still image");
            return false;
        } else if (chunk_is(type, "IDAT")) {
            if (left_idat || length == 0u ||
                idat_bytes > UINT64_MAX - static_cast<uint64_t>(length)) {
                set_error(error, "DeepSeek4 PNG has an invalid IDAT sequence");
                return false;
            }
            seen_idat = true;
            idat_bytes += length;
        } else if (chunk_is(type, "IEND")) {
            const size_t next = cursor + 12u + static_cast<size_t>(length);
            if (length != 0u || !seen_idat || next != encoded_size) {
                set_error(error, "DeepSeek4 PNG has an invalid terminal IEND");
                return false;
            }
            out = candidate;
            return true;
        } else {
            if (seen_idat) left_idat = true;
            // PNG names with an uppercase first byte are critical. The
            // RGB/RGBA8 subset needs no critical chunk beyond IHDR/IDAT/IEND.
            if ((type[0] & 0x20u) == 0u) {
                set_error(error, "DeepSeek4 PNG contains an unsupported critical chunk");
                return false;
            }
        }
        cursor += 12u + static_cast<size_t>(length);
    }
    set_error(error, "DeepSeek4 PNG is missing IEND");
    out = {};
    return false;
}

bool deepseek4_vision_decode_still_png_rgb8(
        const uint8_t * encoded, size_t encoded_size,
        int max_dimension, uint64_t max_pixels,
        std::vector<uint8_t> & rgb, Deepseek4VisionPngInfo & info,
        std::string * error) {
    rgb.clear();
    info = {};
    Deepseek4VisionPngInfo parsed;
    if (!deepseek4_vision_validate_still_png(
            encoded, encoded_size, max_dimension, max_pixels,
            parsed, error)) {
        return false;
    }

    std::vector<uint8_t> compressed;
    size_t cursor = sizeof(kPngSignature);
    while (cursor < encoded_size) {
        const uint32_t length = read_be32(encoded + cursor);
        const uint8_t * type = encoded + cursor + 4u;
        if (chunk_is(type, "IDAT")) {
            if (static_cast<uint64_t>(compressed.size()) + length >
                static_cast<uint64_t>(UINT_MAX)) {
                set_error(error, "DeepSeek4 PNG compressed stream is too large");
                return false;
            }
            const uint8_t * data = type + 4u;
            compressed.insert(compressed.end(), data, data + length);
        }
        cursor += 12u + static_cast<size_t>(length);
    }
    if (compressed.empty() || parsed.filtered_bytes > UINT_MAX) {
        set_error(error, "DeepSeek4 PNG stream is not addressable by zlib");
        return false;
    }

    std::vector<uint8_t> filtered(parsed.filtered_bytes);
    z_stream stream{};
    stream.next_in = compressed.data();
    stream.avail_in = static_cast<uInt>(compressed.size());
    stream.next_out = filtered.data();
    stream.avail_out = static_cast<uInt>(filtered.size());
    if (inflateInit(&stream) != Z_OK) {
        set_error(error, "cannot initialize DeepSeek4 PNG zlib stream");
        return false;
    }
    const int inflate_status = inflate(&stream, Z_FINISH);
    const bool exact_stream = inflate_status == Z_STREAM_END &&
        stream.avail_in == 0u && stream.avail_out == 0u &&
        stream.total_in == compressed.size() &&
        stream.total_out == filtered.size();
    const int end_status = inflateEnd(&stream);
    if (!exact_stream || end_status != Z_OK) {
        set_error(error, "DeepSeek4 PNG zlib stream is partial or overlong");
        return false;
    }

    const size_t source_row = 1u + static_cast<size_t>(parsed.width) *
                                   static_cast<size_t>(parsed.channels);
    const size_t raw_row = source_row - 1u;
    std::vector<uint8_t> raw(
        raw_row * static_cast<size_t>(parsed.height));
    for (int y = 0; y < parsed.height; ++y) {
        const uint8_t * source = filtered.data() +
            static_cast<size_t>(y) * source_row;
        uint8_t * destination = raw.data() +
            static_cast<size_t>(y) * raw_row;
        const uint8_t * previous = y == 0 ? nullptr : destination - raw_row;
        const uint8_t filter = source[0];
        if (filter > 4u) {
            set_error(error, "DeepSeek4 PNG uses an invalid row filter");
            return false;
        }
        for (size_t x = 0; x < raw_row; ++x) {
            const uint8_t left = x < static_cast<size_t>(parsed.channels)
                ? 0u : destination[x - static_cast<size_t>(parsed.channels)];
            const uint8_t up = previous ? previous[x] : 0u;
            const uint8_t upper_left = previous &&
                x >= static_cast<size_t>(parsed.channels)
                ? previous[x - static_cast<size_t>(parsed.channels)] : 0u;
            uint8_t predictor = 0;
            switch (filter) {
                case 0: predictor = 0; break;
                case 1: predictor = left; break;
                case 2: predictor = up; break;
                case 3: predictor = static_cast<uint8_t>(
                    (static_cast<unsigned>(left) + up) / 2u); break;
                case 4: predictor = paeth_predictor(
                    left, up, upper_left); break;
                default: break;
            }
            destination[x] = static_cast<uint8_t>(
                static_cast<unsigned>(source[x + 1u]) + predictor);
        }
    }

    const size_t pixels = static_cast<size_t>(parsed.width) *
                          static_cast<size_t>(parsed.height);
    // `filtered_bytes = height + pixels*channels` was checked against
    // SIZE_MAX before decode, and channels is at least three, so this smaller
    // RGB allocation cannot overflow independently. Keep that earlier check:
    // it is the allocation bound for both the filtered and published buffers.
    std::vector<uint8_t> decoded(pixels * 3u);
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
        decoded[pixel * 3u] = raw[pixel *
            static_cast<size_t>(parsed.channels)];
        decoded[pixel * 3u + 1u] = raw[pixel *
            static_cast<size_t>(parsed.channels) + 1u];
        decoded[pixel * 3u + 2u] = raw[pixel *
            static_cast<size_t>(parsed.channels) + 2u];
    }
    rgb = std::move(decoded);
    info = parsed;
    return true;
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

bool deepseek4_vision_pixel_shuffle_indices(
        int n_vit_h, int n_vit_w,
        int & padded_h, int & padded_w,
        std::vector<int32_t> & indices, std::string * error) {
    padded_h = 0;
    padded_w = 0;
    indices.clear();
    const int scale = deepseek4_vision_native_config().scale_factor;
    if (n_vit_h <= 0 || n_vit_w <= 0 ||
        n_vit_h > INT32_MAX - (scale - 1) ||
        n_vit_w > INT32_MAX - (scale - 1)) {
        set_error(error, "invalid DeepSeek4 pixel-shuffle grid");
        return false;
    }
    const int n_llm_h = (n_vit_h + scale - 1) / scale;
    const int n_llm_w = (n_vit_w + scale - 1) / scale;
    if (n_llm_h > INT32_MAX / scale || n_llm_w > INT32_MAX / scale) {
        set_error(error, "DeepSeek4 pixel-shuffle padded grid overflows");
        return false;
    }
    padded_h = n_llm_h * scale;
    padded_w = n_llm_w * scale;
    const int64_t output_rows = static_cast<int64_t>(n_llm_h) * n_llm_w;
    const int64_t count = output_rows * scale * scale;
    if (count <= 0 || count > INT32_MAX ||
        static_cast<uint64_t>(count) >
            static_cast<uint64_t>(SIZE_MAX / sizeof(int32_t))) {
        padded_h = 0;
        padded_w = 0;
        set_error(error, "DeepSeek4 pixel-shuffle index count overflows");
        return false;
    }
    indices.resize(static_cast<size_t>(count));
    const size_t group = static_cast<size_t>(scale) *
                         static_cast<size_t>(scale);
    for (int out_row = 0; out_row < n_llm_h; ++out_row) {
        for (int out_column = 0; out_column < n_llm_w; ++out_column) {
            const int output_index = out_row * n_llm_w + out_column;
            for (int local_row = 0; local_row < scale; ++local_row) {
                for (int local_column = 0; local_column < scale;
                     ++local_column) {
                    const size_t local =
                        static_cast<size_t>(local_row) *
                        static_cast<size_t>(scale) +
                        static_cast<size_t>(local_column);
                    const int source_row = out_row * scale + local_row;
                    const int source_column = out_column * scale + local_column;
                    indices[static_cast<size_t>(output_index) * group + local] =
                        source_row * padded_w + source_column;
                }
            }
        }
    }
    return true;
}

}  // namespace dflash
