#include "dflash/deepseek4/deepseek4_vision_native_contract.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include <zlib.h>

using namespace dflash;

static int g_pass;
static int g_fail;

#define CHECK(cond, msg) do {                                              \
    if (cond) { ++g_pass; }                                                \
    else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", msg); }           \
} while (0)

static const Deepseek4VisionTensorSpec * find_spec(
        const std::vector<Deepseek4VisionTensorSpec> & specs,
        const std::string & name) {
    for (const auto & spec : specs) {
        if (spec.name == name) return &spec;
    }
    return nullptr;
}

static uint32_t test_png_crc32(const uint8_t * data, size_t size) {
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

static void write_be32(uint8_t * value, uint32_t number) {
    value[0] = static_cast<uint8_t>(number >> 24);
    value[1] = static_cast<uint8_t>(number >> 16);
    value[2] = static_cast<uint8_t>(number >> 8);
    value[3] = static_cast<uint8_t>(number);
}

static void repair_ihdr_crc(std::vector<uint8_t> & png) {
    write_be32(png.data() + 29, test_png_crc32(png.data() + 12, 17));
}

static std::vector<uint8_t> valid_rgb_png() {
    return {
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
        0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x7b, 0x40, 0xe8, 0xdd,
        0x00, 0x00, 0x00, 0x0f, 0x49, 0x44, 0x41, 0x54,
        0x78, 0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xc0, 0xf0, 0x9f,
        0x01, 0x00, 0x07, 0xff, 0x01, 0xff, 0x01, 0x7f, 0x89, 0xa7,
        0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44,
        0xae, 0x42, 0x60, 0x82,
    };
}

static std::vector<uint8_t> png_with_scanline(
        const std::vector<uint8_t> & scanline) {
    const auto base = valid_rgb_png();
    uLongf compressed_size = compressBound(scanline.size());
    std::vector<uint8_t> compressed(static_cast<size_t>(compressed_size));
    if (compress2(compressed.data(), &compressed_size,
                  scanline.data(), scanline.size(), Z_BEST_COMPRESSION) != Z_OK) {
        std::abort();
    }
    compressed.resize(static_cast<size_t>(compressed_size));
    std::vector<uint8_t> png(base.begin(), base.begin() + 33);
    const size_t chunk = png.size();
    png.resize(chunk + 12u + compressed.size());
    write_be32(png.data() + chunk, static_cast<uint32_t>(compressed.size()));
    std::memcpy(png.data() + chunk + 4u, "IDAT", 4);
    std::memcpy(png.data() + chunk + 8u,
                compressed.data(), compressed.size());
    write_be32(png.data() + chunk + 8u + compressed.size(),
               test_png_crc32(png.data() + chunk + 4u,
                              compressed.size() + 4u));
    png.insert(png.end(), base.end() - 12, base.end());
    return png;
}

static void test_png_preflight() {
    const auto valid = valid_rgb_png();
    Deepseek4VisionPngInfo info;
    std::string error;
    CHECK(deepseek4_vision_validate_still_png(
              valid.data(), valid.size(), 8192, 64u * 1024u * 1024u,
              info, &error) && info.width == 2 && info.height == 1 &&
              info.channels == 3 && info.filtered_bytes == 7,
          "complete RGB8 PNG passes bounded still-image preflight");

    auto malformed = valid;
    malformed[0] = 0;
    CHECK(!deepseek4_vision_validate_still_png(
              malformed.data(), malformed.size(), 8192,
              64u * 1024u * 1024u, info, &error) &&
              error.find("PNG only") != std::string::npos && info.width == 0,
          "wrong PNG magic fails closed");
    error.clear();
    CHECK(!deepseek4_vision_validate_still_png(
              valid.data(), valid.size() - 1, 8192,
              64u * 1024u * 1024u, info, &error) &&
              error.find("truncated") != std::string::npos,
          "truncated PNG fails closed instead of partially decoding");

    malformed = valid;
    std::memset(malformed.data() + 16, 0, 4);
    repair_ihdr_crc(malformed);
    CHECK(!deepseek4_vision_validate_still_png(
              malformed.data(), malformed.size(), 8192,
              64u * 1024u * 1024u, info, &error) &&
              error.find("dimensions") != std::string::npos,
          "zero-dimension PNG fails before allocation");
    malformed = valid;
    write_be32(malformed.data() + 16, 10000);
    write_be32(malformed.data() + 20, 10000);
    repair_ihdr_crc(malformed);
    CHECK(!deepseek4_vision_validate_still_png(
              malformed.data(), malformed.size(), 20000,
              64u * 1024u * 1024u, info, &error) &&
              error.find("dimensions") != std::string::npos,
          "dimension-lying PNG exceeds the pixel ceiling before allocation");

    malformed = valid;
    malformed[29] ^= 1u;
    CHECK(!deepseek4_vision_validate_still_png(
              malformed.data(), malformed.size(), 8192,
              64u * 1024u * 1024u, info, &error) &&
              error.find("CRC") != std::string::npos,
          "PNG with a bad chunk CRC fails closed");

    malformed = valid;
    const uint8_t actl[] = {
        0x00, 0x00, 0x00, 0x08, 0x61, 0x63, 0x54, 0x4c,
        0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00,
        0, 0, 0, 0,
    };
    std::vector<uint8_t> animation(
        malformed.begin(), malformed.begin() + 33);
    animation.insert(animation.end(), std::begin(actl), std::end(actl));
    write_be32(animation.data() + 49,
               test_png_crc32(animation.data() + 37, 12));
    animation.insert(animation.end(), malformed.begin() + 33, malformed.end());
    CHECK(!deepseek4_vision_validate_still_png(
              animation.data(), animation.size(), 8192,
              64u * 1024u * 1024u, info, &error) &&
              error.find("animated") != std::string::npos,
          "animated PNG is rejected instead of inheriting a first-frame default");
}

static void test_png_decode() {
    const auto valid = valid_rgb_png();
    Deepseek4VisionPngInfo info;
    std::vector<uint8_t> rgb;
    std::string error;
    CHECK(deepseek4_vision_decode_still_png_rgb8(
              valid.data(), valid.size(), 8192, 64u * 1024u * 1024u,
              rgb, info, &error) &&
              rgb == std::vector<uint8_t>({255, 0, 0, 0, 255, 0}),
          "validated PNG decodes to its exact RGB8 pixels");

    const std::vector<std::vector<uint8_t>> filtered_rows = {
        {1, 255, 0, 0, 1, 255, 0},
        {2, 255, 0, 0, 0, 255, 0},
        {3, 255, 0, 0, 129, 255, 0},
        {4, 255, 0, 0, 1, 255, 0},
    };
    bool every_filter = true;
    for (const auto & row : filtered_rows) {
        const auto filtered_png = png_with_scanline(row);
        every_filter = every_filter && deepseek4_vision_decode_still_png_rgb8(
            filtered_png.data(), filtered_png.size(), 8192,
            64u * 1024u * 1024u, rgb, info, &error) &&
            rgb == std::vector<uint8_t>({255, 0, 0, 0, 255, 0});
    }
    CHECK(every_filter,
          "Sub, Up, Average, and Paeth filters reconstruct the same RGB row");

    auto malformed = valid;
    malformed[45] ^= 0x80u;
    write_be32(malformed.data() + 56,
               test_png_crc32(malformed.data() + 37, 19));
    CHECK(!deepseek4_vision_decode_still_png_rgb8(
              malformed.data(), malformed.size(), 8192,
              64u * 1024u * 1024u, rgb, info, &error) && rgb.empty() &&
              info.width == 0 && error.find("zlib") != std::string::npos,
          "CRC-valid corrupt compression stream cannot partially decode");

    malformed = png_with_scanline({5, 255, 0, 0, 0, 255, 0});
    CHECK(!deepseek4_vision_decode_still_png_rgb8(
              malformed.data(), malformed.size(), 8192,
              64u * 1024u * 1024u, rgb, info, &error) && rgb.empty() &&
              error.find("filter") != std::string::npos,
          "invalid PNG filter fails after exact decompression");

    malformed = png_with_scanline({0, 255, 0});
    CHECK(!deepseek4_vision_decode_still_png_rgb8(
              malformed.data(), malformed.size(), 8192,
              64u * 1024u * 1024u, rgb, info, &error) && rgb.empty() &&
              error.find("partial") != std::string::npos,
          "short decoded payload cannot be accepted as a partial image");
}

static void test_exact_tensor_inventory() {
    const auto specs = deepseek4_vision_tensor_specs();
    std::set<std::string> names;
    int f16 = 0;
    int f32 = 0;
    for (const auto & spec : specs) {
        names.insert(spec.name);
        if (spec.storage == Deepseek4VisionStorage::F16) ++f16;
        else ++f32;
    }
    CHECK(specs.size() == 299, "native mmproj inventory has 299 tensors");
    CHECK(names.size() == specs.size(), "native mmproj names are unique");
    CHECK(f16 == 163 && f32 == 136,
          "native mmproj pins its exact F16/F32 partition");

    const auto * qkv = find_spec(specs, "v.blk.31.attn_qkv.weight");
    CHECK(qkv && qkv->shape == std::vector<int64_t>({1024, 3072}) &&
              qkv->storage == Deepseek4VisionStorage::F16,
          "fused QKV uses exact GGML ne order");
    const auto * gate = find_spec(specs, "v.blk.0.ffn_gate.weight");
    CHECK(gate && gate->shape == std::vector<int64_t>({1024, 2816}),
          "split gate matrix uses exact GGML ne order");
    const auto * down = find_spec(specs, "v.blk.0.ffn_down.weight");
    CHECK(down && down->shape == std::vector<int64_t>({2816, 1024}),
          "FFN down matrix does not inherit the gate orientation");
    const auto * aligner = find_spec(specs, "mm.1.weight");
    CHECK(aligner && aligner->shape == std::vector<int64_t>({9216, 4096}),
          "pixel-shuffle aligner input is 1024 times three squared");
    const auto * marker = find_spec(specs, "mm.image_begin.weight");
    CHECK(marker && marker->shape == std::vector<int64_t>({4096, 1}) &&
              marker->storage == Deepseek4VisionStorage::F32,
          "learned marker remains one F32 language-width row");
}

static void check_plan(int height, int width, int start,
                       int effective_h, int effective_w,
                       int resized_h, int resized_w,
                       int vit_h, int vit_w, int llm_h, int llm_w,
                       int rows, int tokens, const char * message) {
    Deepseek4VisionResizePlan plan;
    std::string error;
    const bool ok = deepseek4_vision_resize_plan(
        height, width, start, plan, &error);
    CHECK(ok && plan.effective_height == effective_h &&
              plan.effective_width == effective_w &&
              plan.resized_height == resized_h &&
              plan.resized_width == resized_w &&
              plan.n_vit_h == vit_h && plan.n_vit_w == vit_w &&
              plan.n_llm_h == llm_h && plan.n_llm_w == llm_w &&
              plan.image_rows == rows && plan.block_tokens == tokens,
          message);
}

static void test_resize_policy() {
    // Values are emitted by the pinned image_processor.py source, not derived
    // from this implementation. Together they enter the minimum-pixel,
    // ordinary-pad, panorama-clamp and token-budget resize branches.
    check_plan(1, 1, 0, 384, 384, 392, 392,
               28, 28, 10, 10, 100, 117,
               "one-pixel input follows minimum-pixel and patch-ceil policy");
    check_plan(100, 200, 0, 271, 543, 280, 546,
               20, 39, 7, 13, 91, 117,
               "landscape minimum-pixel scaling preserves truncation order");
    check_plan(3000, 4000, 3, 3000, 4000, 658, 882,
               47, 63, 16, 21, 336, 354,
               "large image enters the iterative token-budget solver");
    check_plan(100, 1000, 2, 135, 1086, 140, 1092,
               10, 78, 4, 26, 104, 111,
               "exact 8-to-1 aspect remains unclamped before minimum scaling");
    check_plan(200, 1601, 0, 200, 1600, 210, 1610,
               15, 115, 5, 39, 195, 245,
               "over-wide panorama clamps planning width before patch ceil");

    Deepseek4VisionResizePlan panoramic;
    std::string panoramic_error;
    CHECK(deepseek4_vision_resize_plan(
              200, 1600, 0, panoramic, &panoramic_error) &&
              panoramic.panoramic_direct_resize,
          "exact max-aspect boundary selects direct panorama resize");
    CHECK(deepseek4_vision_resize_plan(
              200, 1599, 0, panoramic, &panoramic_error) &&
              !panoramic.panoramic_direct_resize,
          "sub-boundary image retains aspect-preserving pad mode");

    Deepseek4VisionResizePlan plan;
    std::string error;
    CHECK(!deepseek4_vision_resize_plan(0, 1, 0, plan, &error),
          "zero-height image is rejected");
    CHECK(!deepseek4_vision_resize_plan(1, 1, -1, plan, &error),
          "negative prompt position is rejected");
}

static void test_patch_packing() {
    constexpr int height = 14;
    constexpr int width = 28;
    std::vector<uint8_t> rgb(static_cast<size_t>(height) * width * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const uint8_t left[] = {0, 127, 255};
            const uint8_t right[] = {255, 0, 127};
            const uint8_t * value = x < 14 ? left : right;
            for (int c = 0; c < 3; ++c) {
                rgb[(static_cast<size_t>(y) *
                         static_cast<size_t>(width) +
                     static_cast<size_t>(x)) * 3u +
                    static_cast<size_t>(c)] = value[c];
            }
        }
    }
    std::vector<uint16_t> patches;
    std::string error;
    CHECK(deepseek4_vision_pack_rgb8_patches(
              rgb.data(), height, width, patches, &error),
          "patch-aligned RGB8 input packs");
    CHECK(patches.size() == 2 * 3 * 14 * 14,
          "two horizontal patches retain every channel value");
    CHECK(patches[0] == 0xbf80 && patches[196] == 0xbb81 &&
              patches[392] == 0x3f80,
          "first patch is channel-major and bfloat16-normalized");
    CHECK(patches[588] == 0x3f80 && patches[588 + 196] == 0xbf80 &&
              patches[588 + 392] == 0xbb81,
          "second patch follows the first instead of interleaving pixels");
    CHECK(!deepseek4_vision_pack_rgb8_patches(
              rgb.data(), height, width - 1, patches, &error),
          "non-patch-aligned resized input is rejected");
}

static void test_rope_positions() {
    std::vector<float> angles;
    std::string error;
    CHECK(deepseek4_vision_rope_angles(2, 3, angles, &error),
          "2D RoPE table builds for a patch grid");
    CHECK(angles.size() == 2 * 3 * 32,
          "2D RoPE retains head-dimension-half width");
    bool first_zero = true;
    for (int i = 0; i < 32; ++i)
        first_zero &= angles[static_cast<size_t>(i)] == 0.0f;
    CHECK(first_zero, "top-left patch has zero row and column phase");
    CHECK(angles[32] == 0.0f && angles[32 + 16] == 1.0f,
          "top-middle patch advances only the column phase");
    CHECK(angles[3 * 32] == 1.0f && angles[3 * 32 + 16] == 0.0f,
          "second-row first patch advances only the row phase");
    CHECK(std::fabs(angles[32 + 17] - std::pow(10000.0f, -2.0f / 32.0f)) <
              1.0e-7f,
          "RoPE inverse-frequency exponent uses dim 32, not head dim 64");
    CHECK(!deepseek4_vision_rope_angles(0, 3, angles, &error),
          "empty RoPE grid is rejected");
}

static void test_pixel_shuffle_indices() {
    int padded_h = 0;
    int padded_w = 0;
    std::vector<int32_t> indices;
    std::string error;
    CHECK(deepseek4_vision_pixel_shuffle_indices(
              2, 4, padded_h, padded_w, indices, &error),
          "non-multiple patch grid builds pixel-shuffle indices");
    CHECK(padded_h == 3 && padded_w == 6 && indices ==
              std::vector<int32_t>({
                  0, 1, 2, 6, 7, 8, 12, 13, 14,
                  3, 4, 5, 9, 10, 11, 15, 16, 17,
              }),
          "pixel-shuffle indices preserve output-row then kernel-row order");
    CHECK(!deepseek4_vision_pixel_shuffle_indices(
              0, 4, padded_h, padded_w, indices, &error) && indices.empty(),
          "empty pixel-shuffle grid is rejected without partial indices");
    CHECK(!deepseek4_vision_pixel_shuffle_indices(
              50000, 50000, padded_h, padded_w, indices, &error) &&
              indices.empty(),
          "pixel-shuffle grid exceeding int32 indices is rejected");
}

int main() {
    test_exact_tensor_inventory();
    test_png_preflight();
    test_png_decode();
    test_resize_policy();
    test_patch_packing();
    test_rope_positions();
    test_pixel_shuffle_indices();
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
