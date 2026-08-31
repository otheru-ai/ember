#include "dflash/deepseek4/deepseek4_vision_native_contract.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

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
                rgb[(static_cast<size_t>(y) * width + x) * 3 + c] = value[c];
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
    for (int i = 0; i < 32; ++i) first_zero &= angles[i] == 0.0f;
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

int main() {
    test_exact_tensor_inventory();
    test_resize_policy();
    test_patch_packing();
    test_rope_positions();
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
