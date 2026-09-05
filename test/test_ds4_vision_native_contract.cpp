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
              error.find("10000x10000") != std::string::npos &&
              error.find("67108864") != std::string::npos,
          "dimension-lying PNG exceeds the pixel ceiling before allocation");

    malformed = valid;
    write_be32(malformed.data() + 16, 16000);
    repair_ihdr_crc(malformed);
    CHECK(deepseek4_vision_validate_still_png(
              malformed.data(), malformed.size(), 16384,
              UINT64_C(4096) * UINT64_C(4096), info, &error) &&
              info.width == 16000 && info.height == 1,
          "wide scan inside the pixel budget passes structural preflight");
    malformed = valid;
    write_be32(malformed.data() + 16, 16385);
    repair_ihdr_crc(malformed);
    CHECK(!deepseek4_vision_validate_still_png(
              malformed.data(), malformed.size(), 16384,
              UINT64_C(4096) * UINT64_C(4096), info, &error) &&
              error.find("16385x1") != std::string::npos &&
              error.find("16384x16384") != std::string::npos,
          "degenerate strip rejection reports actual and accepted dimensions");

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

static void test_bounded_png_preprocess() {
    const uint8_t uniform[] = {10, 20, 30, 10, 20, 30};
    Deepseek4VisionResizePlan plan;
    plan.source_height = 1;
    plan.source_width = 2;
    plan.resized_height = 4;
    plan.resized_width = 4;
    std::vector<uint8_t> resized;
    std::string error;
    CHECK(deepseek4_vision_resize_rgb8(uniform, plan, resized, &error) &&
              resized.size() == 4u * 4u * 3u &&
              resized[0] == 127u && resized[3u * 4u * 3u] == 127u &&
              resized[1u * 4u * 3u] == 10u &&
              resized[1u * 4u * 3u + 1u] == 20u &&
              resized[1u * 4u * 3u + 2u] == 30u,
          "aspect-preserving resize uses Pillow pad geometry and fill");

    plan.panoramic_direct_resize = true;
    CHECK(deepseek4_vision_resize_rgb8(uniform, plan, resized, &error) &&
              resized.size() == 4u * 4u * 3u && resized[0] == 10u &&
              resized[1] == 20u && resized[2] == 30u &&
              resized.back() == 30u,
          "panorama boundary uses direct bicubic resize without padding");

    const auto png = valid_rgb_png();
    std::vector<uint16_t> patches;
    uint64_t source_digest = 0;
    CHECK(deepseek4_vision_preprocess_still_png(
              png.data(), png.size(), 0, plan, patches,
              source_digest, &error) && source_digest != 0 &&
              plan.source_height == 1 && plan.source_width == 2 &&
              plan.n_vit_h > 0 && plan.n_vit_w > 0 &&
              patches.size() ==
                  static_cast<size_t>(plan.n_vit_h) *
                  static_cast<size_t>(plan.n_vit_w) * 588u,
          "complete PNG preprocessing binds bytes and emits whole BF16 patches");

    const auto & config = deepseek4_vision_native_config();
    CHECK(config.max_decode_dimension == 16384 &&
              config.max_decode_pixels == UINT64_C(4096) * UINT64_C(4096) &&
              !deepseek4_vision_preprocess_still_png(
                  png.data(), config.max_encoded_bytes + 1u, 0,
                  plan, patches, source_digest, &error) &&
              plan.source_height == 0 && patches.empty() &&
              source_digest == 0,
          "complete preprocessor owns fixed encoded and decoded ceilings");
}

static bool read_fixture_file(
        const std::string & path, std::vector<uint8_t> & bytes) {
    bytes.clear();
    std::FILE * file = std::fopen(path.c_str(), "rb");
    if (!file) return false;
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        return false;
    }
    const long length = std::ftell(file);
    if (length < 0 || std::fseek(file, 0, SEEK_SET) != 0) {
        std::fclose(file);
        return false;
    }
    bytes.resize(static_cast<size_t>(length));
    const bool read = bytes.empty() ||
        std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size();
    const bool closed = std::fclose(file) == 0;
    if (!read || !closed) bytes.clear();
    return read && closed;
}

static void test_real_preprocess_fixture(const char * directory) {
    struct FixtureCase {
        const char * name;
        int source_height;
        int source_width;
        int n_vit_h;
        int n_vit_w;
        int n_llm_h;
        int n_llm_w;
    };
    static const FixtureCase cases[] = {
        {"odd-pad", 301, 509, 22, 37, 8, 13},
        {"small-odd-pad", 17, 23, 24, 32, 8, 11},
        {"panorama-direct", 17, 136, 10, 78, 4, 26},
    };
    for (const FixtureCase & fixture : cases) {
        const std::string prefix =
            std::string(directory) + "/" + fixture.name;
        std::vector<uint8_t> png;
        std::vector<uint8_t> expected;
        std::vector<uint16_t> patches;
        Deepseek4VisionResizePlan plan;
        uint64_t source_digest = 0;
        std::string error;
        bool exact = read_fixture_file(prefix + ".png", png) &&
            read_fixture_file(prefix + ".patches.bf16", expected) &&
            deepseek4_vision_preprocess_still_png(
                png.data(), png.size(), 0, plan, patches,
                source_digest, &error) &&
            plan.source_height == fixture.source_height &&
            plan.source_width == fixture.source_width &&
            plan.n_vit_h == fixture.n_vit_h &&
            plan.n_vit_w == fixture.n_vit_w &&
            plan.n_llm_h == fixture.n_llm_h &&
            plan.n_llm_w == fixture.n_llm_w &&
            expected.size() == patches.size() * 2u;
        for (size_t i = 0; exact && i < patches.size(); ++i) {
            exact = expected[i * 2u] ==
                        static_cast<uint8_t>(patches[i]) &&
                    expected[i * 2u + 1u] ==
                        static_cast<uint8_t>(patches[i] >> 8);
        }
        CHECK(exact, fixture.name);
    }
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


static void test_jpeg_decode() {
    const std::string directory = EMBER_JPEG_FIXTURE_DIR;
    const char * names[] = {"rgb444", "rgb422", "rgb420", "progressive",
                           "gray", "gray-progressive"};
    std::string error;
    for (const char * name : names) {
        std::vector<uint8_t> jpeg, png, expected, rgb;
        const std::string stem = directory + "/" + name;
        CHECK(read_fixture_file(stem + ".jpg", jpeg) &&
              read_fixture_file(stem + ".png", png) &&
              read_fixture_file(stem + ".rgb", expected), "JPEG fixtures load");
        if (jpeg.empty() || png.empty() || expected.empty()) continue;
        Deepseek4VisionJpegInfo info;
        CHECK(deepseek4_vision_decode_still_jpeg_rgb8(
                  jpeg.data(), jpeg.size(), 16384, 16777216, rgb, info, &error) &&
              info.width == 17 && info.height == 13 && rgb == expected,
              "JPEG RGB8 matches retained Pillow decoding");
        CHECK(info.progressive == (std::strstr(name, "progressive") != nullptr),
              "JPEG frame mode retained");
        for (int offset = 0; offset < 4; ++offset) {
            Deepseek4VisionResizePlan jp, pp;
            std::vector<uint16_t> jpatch, ppatch;
            uint64_t jd = 0, pd = 0;
            CHECK(deepseek4_vision_preprocess_still_image(
                      jpeg.data(), jpeg.size(), offset, jp, jpatch, jd, &error) &&
                  deepseek4_vision_preprocess_still_image(
                      png.data(), png.size(), offset, pp, ppatch, pd, &error) &&
                  jpatch == ppatch && jp.block_tokens == pp.block_tokens &&
                  jp.source_width == pp.source_width &&
                  jp.source_height == pp.source_height && jd != 0 && pd != 0 && jd != pd,
                  "JPEG shares exact PNG RGB patch path at every prompt offset");
        }
        // Every incomplete prefix must reject, not return partially decoded pixels.
        for (size_t length = 0; length < jpeg.size(); ++length) {
            rgb = {42}; info.width = 42;
            CHECK(!deepseek4_vision_decode_still_jpeg_rgb8(
                      jpeg.data(), length, 16384, 16777216, rgb, info, &error) &&
                  rgb.empty() && info.width == 0 && !error.empty(),
                  "all JPEG truncations fail with cleared outputs");
        }
    }
    std::vector<uint8_t> original;
    CHECK(read_fixture_file(directory + "/rgb420.jpg", original), "JPEG mutation source");
    if (original.empty()) return;
    const auto rejected = [&](const std::vector<uint8_t> & bytes, const char * label) {
        std::vector<uint8_t> rgb = {42};
        Deepseek4VisionJpegInfo info; info.width = 42;
        CHECK(!deepseek4_vision_decode_still_jpeg_rgb8(
                  bytes.data(), bytes.size(), 16384, 16777216, rgb, info, &error) &&
              rgb.empty() && info.width == 0 && !error.empty(), label);
    };
    auto changed = original; changed[0] = 0;
    rejected(changed, "JPEG wrong magic rejected");
    changed = original; changed.push_back(0);
    rejected(changed, "JPEG trailing byte rejected");
    changed = original; changed.insert(changed.end(), original.begin(), original.end());
    rejected(changed, "concatenated JPEG images rejected");
    std::vector<uint8_t> rgb;
    Deepseek4VisionJpegInfo info;
    CHECK(!deepseek4_vision_decode_still_jpeg_rgb8(
              original.data(), original.size(), 16, 16777216, rgb, info, &error),
          "JPEG axis limit enforced before decode");
    CHECK(!deepseek4_vision_decode_still_jpeg_rgb8(
              original.data(), original.size(), 16384, 220, rgb, info, &error),
          "JPEG pixel limit enforced before decode");
    CHECK(!deepseek4_vision_validate_still_jpeg(
              original.data(), 64u * 1024u * 1024u + 1u, 16384, 16777216, info, &error),
          "JPEG encoded limit checked before touching oversized payload");
    size_t sof = 0, sos = 0, dqt = 0;
    for (size_t pos = 2; pos + 3 < original.size();) {
        const unsigned marker = original[pos + 1];
        if (marker == 0xc0) sof = pos;
        if (marker == 0xdb) dqt = pos;
        if (marker == 0xda) { sos = pos; break; }
        pos += 2u + (static_cast<size_t>(original[pos + 2]) << 8) + original[pos + 3];
    }
    CHECK(sof && sos && dqt, "fixture marker locations found");
    if (!sof || !sos || !dqt) return;
    changed = original; changed[sof + 5] = 0; changed[sof + 6] = 0;
    rejected(changed, "JPEG zero height rejected");
    changed = original; changed[sof + 7] = 0xff; changed[sof + 8] = 0xff;
    rejected(changed, "JPEG huge width rejected");
    changed = original; changed[sof + 4] = 12;
    rejected(changed, "JPEG non-8-bit precision rejected");
    changed = original; changed[sof + 1] = 0xc9;
    rejected(changed, "JPEG arithmetic mode rejected");
    changed = original; changed[sof + 2] = 0; changed[sof + 3] = 1;
    rejected(changed, "JPEG invalid segment length rejected");
    changed = original; changed[sof + 11] = 0;
    rejected(changed, "JPEG invalid sampling factors rejected");
    changed = original; changed[sof + 13] = changed[sof + 10];
    rejected(changed, "JPEG duplicate frame components rejected");
    changed = original; changed[dqt + 4] = 0xf0;
    CHECK(deepseek4_vision_validate_still_jpeg(
              changed.data(), changed.size(), 16384, 16777216, info, &error),
          "bad quantization reaches codec beyond marker preflight");
    rejected(changed, "JPEG corrupt quantization rejected by codec");
    changed = original;
    const size_t scan_header = 2u + (static_cast<size_t>(original[sos + 2]) << 8) + original[sos + 3];
    changed.erase(changed.begin() + static_cast<std::ptrdiff_t>(sos + scan_header),
                  changed.end() - 2);
    CHECK(deepseek4_vision_validate_still_jpeg(
              changed.data(), changed.size(), 16384, 16777216, info, &error),
          "missing entropy reaches codec with complete marker stream");
    rejected(changed, "JPEG partial-decode warning is fatal");
    changed.assign(original.begin(), original.begin() + static_cast<std::ptrdiff_t>(sos));
    for (int i = 0; i < 65; ++i)
        changed.insert(changed.end(), original.begin() + static_cast<std::ptrdiff_t>(sos),
                       original.begin() + static_cast<std::ptrdiff_t>(sos + scan_header));
    changed.insert(changed.end(), {0xff, 0xd9});
    CHECK(!deepseek4_vision_validate_still_jpeg(
              changed.data(), changed.size(), 16384, 16777216, info, &error) &&
          error.find("64-scan") != std::string::npos, "JPEG scan bomb rejected by scan bound");
    std::vector<uint8_t> cmyk;
    CHECK(read_fixture_file(directory + "/cmyk-rejected.jpg", cmyk), "CMYK fixture loads");
    rejected(cmyk, "JPEG CMYK explicitly unsupported");
    // Mutations may remain valid JPEGs; success must still honor all output bounds.
    for (size_t i = 0; i < original.size(); ++i) {
        changed = original; changed[i] ^= 0x80;
        const bool ok = deepseek4_vision_decode_still_jpeg_rgb8(
            changed.data(), changed.size(), 64, 4096, rgb, info, &error);
        CHECK(ok ? (info.width > 0 && info.width <= 64 && info.height > 0 &&
                    info.height <= 64 && rgb.size() == static_cast<size_t>(info.width) *
                    static_cast<size_t>(info.height) * 3u)
                 : (rgb.empty() && info.width == 0 && !error.empty()),
              "JPEG byte mutations preserve success/failure output contract");
    }
}

int main(int argc, char ** argv) {
    test_jpeg_decode();
    test_exact_tensor_inventory();
    test_png_preflight();
    test_png_decode();
    test_bounded_png_preprocess();
    test_resize_policy();
    test_patch_packing();
    test_rope_positions();
    test_pixel_shuffle_indices();
    if (argc == 2) test_real_preprocess_fixture(argv[1]);
    if (argc > 2) {
        std::fprintf(stderr,
                     "usage: %s [real-image-processor-fixture-dir]\n",
                     argv[0]);
        return 2;
    }
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
