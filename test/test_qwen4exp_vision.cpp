#include "qwen4exp_vision.h"
#include "qwen4exp_vision_loader.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace dflash::common;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) ++g_pass; \
    else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", msg); } \
} while (0)

static void test_config_contract(void) {
    Qwen4ExpVisionConfig config;
    std::string error;
    CHECK(qwen4exp_validate_vision_config(config, error),
          "released vision config matches exact contract");
    config.output_hidden_size = 3584;
    CHECK(!qwen4exp_validate_vision_config(config, error) &&
              error.find("out_hidden_size") != std::string::npos,
          "generic default output width cannot masquerade as released width");
    config.output_hidden_size = Qwen4ExpVisionContract::output_hidden_size;
    config.deepstack_layer_count = 1;
    CHECK(!qwen4exp_validate_vision_config(config, error) &&
              error.find("deepstack") != std::string::npos,
          "released checkpoint rejects an unexpected deepstack merger");
}

static void test_tensor_inventory(void) {
    auto tensors = qwen4exp_vision_tensor_contract();
    std::string error;
    CHECK(tensors.size() == 334, "PR #27742 vision inventory has 334 tensors");
    std::reverse(tensors.begin(), tensors.end());
    CHECK(qwen4exp_validate_vision_tensor_inventory(tensors, error),
          "vision inventory validation is independent of GGUF tensor order");

    tensors[0].ne[0] += 1;
    CHECK(!qwen4exp_validate_vision_tensor_inventory(tensors, error) &&
              error.find("shape mismatch") != std::string::npos,
          "vision inventory rejects a mismatched tensor shape");
    tensors = qwen4exp_vision_tensor_contract();
    tensors.back().name = tensors.front().name;
    CHECK(!qwen4exp_validate_vision_tensor_inventory(tensors, error) &&
              error.find("duplicate") != std::string::npos,
          "vision inventory rejects duplicate names");
    tensors = qwen4exp_vision_tensor_contract();
    tensors.pop_back();
    CHECK(!qwen4exp_validate_vision_tensor_inventory(tensors, error) &&
              error.find("count mismatch") != std::string::npos,
          "vision inventory rejects an incomplete conversion");
}

static void test_vit_patch_positions(void) {
    std::vector<Qwen4ExpVisionPatchPosition> positions;
    std::vector<int32_t> cu;
    std::string error;
    CHECK(qwen4exp_vision_patch_positions({{1, 4, 6}}, positions, cu, error),
          "ViT position builder accepts a merge-aligned grid");
    CHECK(positions.size() == 24 && cu == std::vector<int32_t>({0, 24}),
          "ViT positions retain every pre-merger patch and frame segment");
    const std::vector<Qwen4ExpVisionPatchPosition> first_block = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1},
        {0, 2}, {0, 3}, {1, 2}, {1, 3},
    };
    bool exact = positions.size() >= first_block.size();
    for (size_t i = 0; exact && i < first_block.size(); ++i) {
        exact = positions[i].h == first_block[i].h &&
                positions[i].w == first_block[i].w;
    }
    CHECK(exact, "ViT coordinates use spatial-block then within-block order");

    CHECK(qwen4exp_vision_patch_positions({{2, 2, 2}}, positions, cu, error) &&
              cu == std::vector<int32_t>({0, 4, 8}),
          "ViT attention cu-seqlens isolate each temporal frame");
    CHECK(!qwen4exp_vision_patch_positions({{1, 3, 4}}, positions, cu, error),
          "ViT position builder rejects a non-merge-aligned grid");
}

static std::vector<float> rows(size_t count, float base) {
    std::vector<float> result(
        count * Qwen4ExpVisionContract::output_hidden_size);
    for (size_t row = 0; row < count; ++row) {
        std::fill_n(result.begin() + static_cast<std::ptrdiff_t>(
                        row * Qwen4ExpVisionContract::output_hidden_size),
                    Qwen4ExpVisionContract::output_hidden_size,
                    base + static_cast<float>(row));
    }
    return result;
}

static void test_mrope_and_ordered_splice(void) {
    const int32_t image =
        static_cast<int32_t>(Qwen4ExpVisionContract::image_token_id);
    const std::vector<int32_t> ids = {11, 12, image, image, image, image,
                                      image, image, 13, 14};
    const std::vector<Qwen4ExpModality> modalities = {
        Qwen4ExpModality::TEXT, Qwen4ExpModality::TEXT,
        Qwen4ExpModality::IMAGE, Qwen4ExpModality::IMAGE,
        Qwen4ExpModality::IMAGE, Qwen4ExpModality::IMAGE,
        Qwen4ExpModality::IMAGE, Qwen4ExpModality::IMAGE,
        Qwen4ExpModality::TEXT, Qwen4ExpModality::TEXT,
    };
    const auto original = rows(ids.size(), 10.0f);
    Qwen4ExpEncodedImage encoded{{1, 4, 6}, rows(6, 100.0f)};
    Qwen4ExpPreparedVisionInput out;
    std::string error;
    CHECK(qwen4exp_prepare_vision_input(ids, modalities, original, {encoded},
                                        out, error),
          "precomputed image rows splice into a mixed prompt");
    CHECK(out.ple_input_ids == ids,
          "PLE receives original image placeholder ids after embedding splice");
    CHECK(out.embeddings[0] == 10.0f &&
              out.embeddings[2 * Qwen4ExpVisionContract::output_hidden_size] ==
                  100.0f &&
              out.embeddings[7 * Qwen4ExpVisionContract::output_hidden_size] ==
                  105.0f &&
              out.embeddings[8 * Qwen4ExpVisionContract::output_hidden_size] ==
                  18.0f,
          "only image rows are replaced and their row order is preserved");
    CHECK(out.position_ids[0] ==
              std::vector<int32_t>({0, 1, 2, 2, 2, 2, 2, 2, 5, 6}),
          "language-model temporal positions match pinned get_rope_index");
    CHECK(out.position_ids[1] ==
              std::vector<int32_t>({0, 1, 2, 2, 2, 3, 3, 3, 5, 6}),
          "language-model height positions match merged image rows");
    CHECK(out.position_ids[2] ==
              std::vector<int32_t>({0, 1, 2, 3, 4, 2, 3, 4, 5, 6}),
          "language-model width positions match merged image rows");
    CHECK(out.text_position_ids ==
              std::vector<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
          "QSA text positions remain physical token offsets");
    CHECK(out.rope_delta == -3,
          "M-RoPE delta advances image span by max merged spatial axis");

    std::array<std::vector<int32_t>, 3> incremental;
    CHECK(qwen4exp_vision_incremental_positions(10, out.rope_delta, 2,
                                                incremental, error) &&
              incremental[0] == std::vector<int32_t>({7, 8}) &&
              incremental[0] == incremental[1] &&
              incremental[1] == incremental[2],
          "incremental decode adds cached rope delta on every axis");
}

static void test_mrope_overflow_guard(void) {
    std::array<std::vector<int32_t>, 3> positions;
    int64_t rope_delta = 0;
    std::string error;
    const size_t max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
    const std::vector<Qwen4ExpMropeRun> runs = {
        {max + 1, false, {}}, {1, true, {1, 2, 2}},
    };
    CHECK(!qwen4exp_assign_mrope_positions(runs, max + 2, positions,
                                           rope_delta, error) &&
              error.find("int32") != std::string::npos,
          "shared runtime M-RoPE walk rejects image-axis int32 overflow");
}

static void test_processor_shape_and_patch_layout(void) {
    Qwen4ExpImageSize resized;
    Qwen4ExpVisionGrid grid;
    std::string error;
    CHECK(qwen4exp_image_smart_resize({100, 200}, resized, grid, error) &&
              resized.height == 192 && resized.width == 384 &&
              grid.t == 1 && grid.h == 12 && grid.w == 24,
          "pinned minimum-pixel smart resize matches the official processor");
    CHECK(qwen4exp_image_smart_resize({272, 304}, resized, grid, error) &&
              resized.height == 256 && resized.width == 320,
          "smart resize uses Python ties-to-even rounding");
    CHECK(qwen4exp_image_smart_resize({5000, 5000}, resized, grid, error) &&
              resized.height == 4096 && resized.width == 4096,
          "pinned maximum-pixel smart resize matches the official processor");
    CHECK(!qwen4exp_image_smart_resize({1, 201}, resized, grid, error),
          "smart resize rejects an aspect ratio above 200");

    const Qwen4ExpImageSize image_size{32, 32};
    std::vector<float> planar(3 * 32 * 32);
    for (size_t channel = 0; channel < 3; ++channel) {
        for (size_t h = 0; h < 32; ++h) {
            for (size_t w = 0; w < 32; ++w) {
                planar[channel * 32 * 32 + h * 32 + w] =
                    static_cast<float>(channel * 1000000 + h * 1000 + w);
            }
        }
    }
    std::vector<float> patches;
    CHECK(qwen4exp_patchify_normalized_rgb(
              planar, image_size, patches, grid, error) &&
              grid.t == 1 && grid.h == 2 && grid.w == 2 &&
              patches.size() == 4 * 1536,
          "normalized RGB patchify emits the official [THW,1536] matrix");
    CHECK(patches[0] == 0.0f && patches[255] == 15015.0f &&
              patches[256] == 0.0f && patches[512] == 1000000.0f,
          "patch rows order channel, duplicated temporal plane, H, W");
    CHECK(patches[1536] == 16.0f &&
              patches[2 * 1536] == 16000.0f &&
              patches[3 * 1536] == 16016.0f,
          "patch rows follow spatial block-major inner-patch order");
    CHECK(!qwen4exp_patchify_normalized_rgb(
              planar, {31, 32}, patches, grid, error),
          "patchify rejects a non-merge-aligned resized image");
}

static void test_multiple_images_and_fail_closed(void) {
    const int32_t image =
        static_cast<int32_t>(Qwen4ExpVisionContract::image_token_id);
    const std::vector<int32_t> ids = {1, image, 2, image, image, 3};
    const std::vector<Qwen4ExpModality> modalities = {
        Qwen4ExpModality::TEXT, Qwen4ExpModality::IMAGE,
        Qwen4ExpModality::TEXT, Qwen4ExpModality::IMAGE,
        Qwen4ExpModality::IMAGE, Qwen4ExpModality::TEXT,
    };
    Qwen4ExpPreparedVisionInput out;
    std::string error;
    CHECK(qwen4exp_prepare_vision_input(
              ids, modalities, rows(ids.size(), 1.0f),
              {{{1, 2, 2}, rows(1, 50.0f)},
               {{1, 2, 4}, rows(2, 70.0f)}}, out, error) &&
              out.embeddings[1 * Qwen4ExpVisionContract::output_hidden_size] ==
                  50.0f &&
              out.embeddings[3 * Qwen4ExpVisionContract::output_hidden_size] ==
                  70.0f &&
              out.embeddings[4 * Qwen4ExpVisionContract::output_hidden_size] ==
                  71.0f,
          "multiple encoded images are consumed in placeholder-run order");

    auto bad_modalities = modalities;
    bad_modalities[1] = Qwen4ExpModality::VIDEO;
    auto video_ids = ids;
    video_ids[1] = static_cast<int32_t>(
        Qwen4ExpVisionContract::video_token_id);
    CHECK(!qwen4exp_prepare_vision_input(video_ids, bad_modalities,
                                         rows(ids.size(), 1.0f), {}, out,
                                         error) &&
              error.find("video") != std::string::npos,
          "video requests fail closed until timestamp-aware support exists");
    CHECK(!qwen4exp_prepare_vision_input(
              {1, image}, {Qwen4ExpModality::TEXT, Qwen4ExpModality::IMAGE},
              rows(2, 1.0f), {{{1, 2, 4}, rows(2, 1.0f)}}, out, error),
          "placeholder count mismatch cannot silently drop vision rows");
    CHECK(!qwen4exp_prepare_vision_input(
              {image}, {Qwen4ExpModality::TEXT}, rows(1, 1.0f), {}, out,
              error) && error.find("masks do not match") != std::string::npos,
          "untyped image placeholder cannot survive as an ordinary text row");
    CHECK(!qwen4exp_prepare_vision_input(
              {image, image},
              {Qwen4ExpModality::IMAGE, Qwen4ExpModality::IMAGE},
              rows(2, 1.0f), {{{2, 2, 2}, rows(2, 1.0f)}}, out, error) &&
              error.find("T must be 1") != std::string::npos,
          "image seam rejects temporal grids owned by unsupported video path");
}

static bool fake_encoder(void *, const float *, size_t,
                         const Qwen4ExpVisionGrid & grid,
                         std::vector<float> & output, std::string &) {
    output = rows(qwen4exp_vision_merged_tokens(grid), 42.0f);
    return true;
}

static bool bad_encoder(void *, const float *, size_t,
                        const Qwen4ExpVisionGrid &,
                        std::vector<float> & output, std::string &) {
    output = {1.0f};
    return true;
}

static void test_encoder_boundary(void) {
    const Qwen4ExpVisionGrid grid{1, 2, 2};
    constexpr size_t width = 3 * 2 * 16 * 16;
    const std::vector<float> patches(4 * width, 0.25f);
    Qwen4ExpEncodedImage out;
    std::string error;
    CHECK(!qwen4exp_encode_vision_patches(nullptr, patches, grid, out, error) &&
              error.find("not installed") != std::string::npos,
          "missing vision encoder provider fails closed");
    Qwen4ExpVisionEncoderProvider provider{nullptr, fake_encoder};
    CHECK(qwen4exp_encode_vision_patches(&provider, patches, grid, out, error) &&
              out.embeddings.size() ==
                  Qwen4ExpVisionContract::output_hidden_size &&
              out.embeddings.front() == 42.0f,
          "encoder provider seam validates and returns merged rows");
    provider.encode = bad_encoder;
    CHECK(!qwen4exp_encode_vision_patches(&provider, patches, grid, out, error),
          "encoder provider cannot return a wrong-shaped feature matrix");
    provider.encode = fake_encoder;
    CHECK(!qwen4exp_encode_vision_patches(
              &provider, std::vector<float>(patches.size() - 1), grid, out,
              error),
          "encoder provider rejects wrong-shaped flattened patches");
    CHECK(!qwen4exp_encode_vision_patches(
              &provider, std::vector<float>(8 * width), {2, 2, 2}, out,
              error) && error.find("T must be 1") != std::string::npos,
          "image encoder boundary rejects temporal grids");
}

int main(void) {
    test_config_contract();
    test_tensor_inventory();
    test_vit_patch_positions();
    test_mrope_and_ordered_splice();
    test_mrope_overflow_guard();
    test_processor_shape_and_patch_layout();
    test_multiple_images_and_fail_closed();
    test_encoder_boundary();
    std::printf("qwen4exp vision: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
