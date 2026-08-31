#include "dflash/deepseek4/deepseek4_vision_contract.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace dflash;

static int g_pass;
static int g_fail;

#define CHECK(cond, msg) do {                                              \
    if (cond) { ++g_pass; }                                                \
    else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", msg); }           \
} while (0)

static void test_two_row_n_layout() {
    constexpr int32_t vocab = 1000;
    Deepseek4ImageBlock block;
    std::string error;
    CHECK(deepseek4_build_image_block(vocab, 2, 3, 0, block, &error),
          "2x3 block builds");
    const std::vector<int32_t> expected_types = {
        1, 1, 1, 0,
        2, 2, 2, 2, 2, 2, 3, 3,
        4,
    };
    std::vector<int32_t> expected_ids;
    for (int32_t type : expected_types) expected_ids.push_back(vocab + type);
    CHECK(block.token_ids == expected_ids,
          "two rows interleave by column with learned boundaries");
    CHECK(block.image_perm == std::vector<int32_t>({0, 3, 1, 4, 2, 5}),
          "IMAGE slots permute row-major aligner rows into N-layout");
}

static std::vector<int32_t> reference_types(int h, int w, int start_pos) {
    const int rows = h + h % 2;
    const int row_len = w + 1;
    std::vector<int32_t> row_major;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < row_len; ++col) {
            row_major.push_back(row >= h ? DEEPSEEK4_IMAGE_PAD
                                : col == w ? DEEPSEEK4_IMAGE_NEWLINE
                                           : DEEPSEEK4_IMAGE);
        }
    }
    std::vector<int32_t> out(
        static_cast<size_t>(3 - start_pos % 4), DEEPSEEK4_IMAGE_PAD);
    out.push_back(DEEPSEEK4_IMAGE_START);
    for (int pair = 0; pair < rows / 2; ++pair)
        for (int col = 0; col < row_len; ++col)
            for (int side = 0; side < 2; ++side)
                out.push_back(row_major[static_cast<size_t>(
                    (pair * 2 + side) * row_len + col)]);
    const int trailing = (rows / 2 * row_len) % 2 * 2;
    out.insert(out.end(), static_cast<size_t>(trailing),
               DEEPSEEK4_IMAGE_PAD);
    out.push_back(DEEPSEEK4_IMAGE_END);
    return out;
}

static void test_grid_sweep_against_scalar_reference() {
    constexpr int32_t vocab = 1500;
    bool all_match = true;
    for (int h = 1; h <= 7; ++h) {
        for (int w = 1; w <= 7; ++w) {
            for (int start = 0; start < 8; ++start) {
                Deepseek4ImageBlock block;
                if (!deepseek4_build_image_block(
                        vocab, h, w, start, block)) {
                    all_match = false;
                    continue;
                }
                std::vector<int32_t> expected =
                    reference_types(h, w, start);
                for (int32_t & type : expected) type += vocab;
                if (block.token_ids != expected) all_match = false;
            }
        }
    }
    CHECK(all_match,
          "grid/start sweep matches independent row-major transpose reference");
}

static void test_padding_and_mutations() {
    constexpr int32_t vocab = 2000;
    Deepseek4ImageBlock block;
    CHECK(deepseek4_build_image_block(vocab, 3, 2, 2, block),
          "odd-height block builds");
    CHECK(block.token_ids.front() == vocab + DEEPSEEK4_IMAGE_PAD &&
          block.token_ids[1] == vocab + DEEPSEEK4_IMAGE_START,
          "start-position compression pad is retained");
    CHECK(block.image_perm == std::vector<int32_t>({0, 2, 1, 3, 4, 5}),
          "odd final row omits synthetic pad-row indices");
    CHECK(deepseek4_validate_image_block(
              block.token_ids, vocab, 3, 2, 2),
          "unmodified learned layout validates");

    std::vector<int32_t> missing_newline = block.token_ids;
    for (int32_t & id : missing_newline) {
        if (id == vocab + DEEPSEEK4_IMAGE_NEWLINE) {
            id = vocab + DEEPSEEK4_IMAGE;
            break;
        }
    }
    CHECK(!deepseek4_validate_image_block(
              missing_newline, vocab, 3, 2, 2),
          "missing newline mutation is rejected");

    std::vector<int32_t> missing_pad = block.token_ids;
    missing_pad.erase(missing_pad.begin());
    CHECK(!deepseek4_validate_image_block(
              missing_pad, vocab, 3, 2, 2),
          "missing alignment pad mutation is rejected");
}

static void test_embedding_assembly() {
    constexpr int32_t vocab = 2500;
    constexpr int n_embd = 2;
    std::vector<float> image_rows;
    for (int row = 0; row < 6; ++row) {
        image_rows.push_back(static_cast<float>(row));
        image_rows.push_back(static_cast<float>(100 + row));
    }
    Deepseek4ImageMarkers markers;
    markers.start = {-10.0f, -11.0f};
    markers.pad = {-20.0f, -21.0f};
    markers.newline = {-30.0f, -31.0f};
    markers.end = {-40.0f, -41.0f};
    Deepseek4PreparedImage prepared;
    std::string error;
    CHECK(deepseek4_prepare_image(vocab, 2, 3, 0, n_embd,
                                  image_rows, markers, prepared, &error),
          "row-major aligner output assembles with learned markers");
    CHECK(prepared.block.image_perm ==
              std::vector<int32_t>({0, 3, 1, 4, 2, 5}),
          "prepared image retains the official N-layout permutation");
    bool rows_match = prepared.embeddings.size() ==
                      prepared.block.token_ids.size() * n_embd;
    size_t image_slot = 0;
    for (size_t row = 0; rows_match && row < prepared.block.token_ids.size(); ++row) {
        const int32_t type = prepared.block.token_ids[row] - vocab;
        const float * got = prepared.embeddings.data() + row * n_embd;
        if (type == DEEPSEEK4_IMAGE) {
            const int source = prepared.block.image_perm[image_slot++];
            rows_match = got[0] == static_cast<float>(source) &&
                         got[1] == static_cast<float>(100 + source);
        } else {
            const std::vector<float> * want = nullptr;
            if (type == DEEPSEEK4_IMAGE_START) want = &markers.start;
            if (type == DEEPSEEK4_IMAGE_PAD) want = &markers.pad;
            if (type == DEEPSEEK4_IMAGE_NEWLINE) want = &markers.newline;
            if (type == DEEPSEEK4_IMAGE_END) want = &markers.end;
            rows_match = want && got[0] == (*want)[0] && got[1] == (*want)[1];
        }
    }
    CHECK(rows_match && image_slot == 6,
          "every prepared row comes from its exact marker or permuted IMAGE source");

    std::vector<float> short_rows = image_rows;
    short_rows.pop_back();
    CHECK(!deepseek4_prepare_image(vocab, 2, 3, 0, n_embd,
                                   short_rows, markers, prepared, &error),
          "aligner row-count mismatch fails closed");
    Deepseek4ImageMarkers missing_marker = markers;
    missing_marker.newline.pop_back();
    CHECK(!deepseek4_prepare_image(vocab, 2, 3, 0, n_embd,
                                   image_rows, missing_marker, prepared, &error),
          "learned marker width mismatch fails closed");
}

static void test_trailing_alignment_pad() {
    constexpr int32_t vocab = 3000;
    Deepseek4ImageBlock block;
    CHECK(deepseek4_build_image_block(vocab, 2, 2, 3, block),
          "2x2 block builds");
    CHECK(block.token_ids.size() >= 3 &&
          block.token_ids[block.token_ids.size() - 2] ==
              vocab + DEEPSEEK4_IMAGE_PAD &&
          block.token_ids[block.token_ids.size() - 3] ==
              vocab + DEEPSEEK4_IMAGE_PAD,
          "odd paired-row width gets two trailing alignment pads");
}

static void test_visibility_and_prefill_boundaries() {
    constexpr int32_t vocab = 4000;
    Deepseek4ImageBlock block;
    CHECK(deepseek4_build_image_block(vocab, 2, 2, 0, block),
          "visibility fixture builds");
    std::vector<int32_t> prompt = {10, 11};
    prompt.insert(prompt.end(), block.token_ids.begin(), block.token_ids.end());
    prompt.push_back(12);
    std::vector<int32_t> left;
    std::vector<int32_t> right;
    deepseek4_image_visible(prompt, vocab, 64, left, right);
    size_t start = 0;
    size_t end = 0;
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (prompt[i] == vocab + DEEPSEEK4_IMAGE_START) start = i;
        if (prompt[i] == vocab + DEEPSEEK4_IMAGE_END) end = i;
    }
    CHECK(left[start] == 0 && right[start] == static_cast<int32_t>(end - start),
          "image start sees through the complete block to its right");
    CHECK(left[end] == static_cast<int32_t>(end - start) && right[end] == 0,
          "image end sees the complete block to its left");
    CHECK(left[start - 1] == 0 && right[start - 1] == 0 &&
          left[end + 1] == 0 && right[end + 1] == 0,
          "tokens outside learned boundaries keep ordinary visibility");
    CHECK(!deepseek4_prefill_cut_safe(prompt, vocab,
                                      static_cast<int>(start + 1)),
          "prefill cut cannot split an image block");
    CHECK(deepseek4_prefill_cut_safe(prompt, vocab, static_cast<int>(start)),
          "prefill cut immediately before image is safe");
    CHECK(deepseek4_chunk_accepts_image_tokens(0, prompt, vocab),
          "initial prefill chunk accepts learned image ids");
    CHECK(!deepseek4_chunk_accepts_image_tokens(1, block.token_ids, vocab),
          "later prefill chunk rejects every learned image span");
    CHECK(deepseek4_chunk_accepts_image_tokens(1, {1, 2, 3}, vocab),
          "later text-only chunk remains valid");
}

static void test_image_aware_prefill_chunks() {
    constexpr int32_t vocab = 4500;
    Deepseek4ImageBlock block;
    CHECK(deepseek4_build_image_block(vocab, 2, 3, 2, block),
          "chunking fixture builds at a nonzero prompt alignment");
    std::vector<int32_t> prompt = {10, 11};
    prompt.insert(prompt.end(), block.token_ids.begin(), block.token_ids.end());
    prompt.insert(prompt.end(), {12, 13, 14});
    std::string error;
    CHECK(deepseek4_image_aware_prefill_chunk(
              prompt, vocab, 0, 5, 64, &error) == 2,
          "a chunk that would enter an image run stops before its leading pad");
    CHECK(deepseek4_image_aware_prefill_chunk(
              prompt, vocab, 2, 1, 64, &error) ==
              static_cast<int>(block.token_ids.size()),
          "a chunk at the image-run boundary expands across the complete block");
    CHECK(deepseek4_image_aware_prefill_chunk(
              prompt, vocab, 2, 1,
              static_cast<int>(block.token_ids.size()) - 1, &error) == -1,
          "an image block larger than the graph cap fails closed");
    CHECK(deepseek4_image_aware_prefill_chunk(
              prompt, vocab, 3, 4, 64, &error) == -1,
          "a restored prefill cannot begin inside a learned image run");
    CHECK(deepseek4_image_aware_prefill_chunk(
              {1, 2, 3, 4, 5}, vocab, 1, 3, 64, &error) == 3,
          "text-only chunk sizing remains inert");
    CHECK(deepseek4_image_aware_prefill_chunk(
              {1, 2, 3}, INT32_MAX, 0, 1, 1, &error) == -1,
          "a vocabulary whose virtual-id range would overflow fails closed");

    std::vector<int32_t> missing_end = prompt;
    const auto end = std::find(
        missing_end.begin(), missing_end.end(),
        vocab + DEEPSEEK4_IMAGE_END);
    if (end != missing_end.end()) missing_end.erase(end);
    CHECK(deepseek4_image_aware_prefill_chunk(
              missing_end, vocab, 2, 1, 64, &error) == -1,
          "a malformed image run missing its end fails closed");

    const int full = static_cast<int>(prompt.size());
    CHECK(deepseek4_image_aware_prefill_chunk(
              prompt, vocab, 0, full, full, &error) == full,
          "a chunk already containing the full image run is unchanged");

    std::vector<int32_t> adjacent = block.token_ids;
    adjacent.insert(adjacent.end(), block.token_ids.begin(),
                    block.token_ids.end());
    const int one_block = static_cast<int>(block.token_ids.size());
    CHECK(deepseek4_image_aware_prefill_chunk(
              adjacent, vocab, 0, one_block + 1,
              static_cast<int>(adjacent.size()), &error) == one_block,
          "a chunk cannot consume a prefix of an adjacent second image block");
    CHECK(deepseek4_image_aware_prefill_chunk(
              adjacent, vocab, one_block, 1, one_block, &error) == one_block,
          "a complete adjacent image block can start immediately after END");
}

static void test_routing_modes() {
    constexpr int32_t vocab = 5000;
    CHECK(deepseek4_is_vision_router_bias_suffix("exp_probs_b_vl") &&
              !deepseek4_is_vision_router_bias_suffix(
                  "exp_probs_b_vl.bias") &&
              !deepseek4_is_vision_router_bias_suffix("exp_probs_b.bias"),
          "only the converter's exact vision-router suffix is recognized");
    CHECK(deepseek4_optional_vision_bias_set_valid(0, 43),
          "an absent vision-bias set keeps text-only checkpoints valid");
    CHECK(deepseek4_optional_vision_bias_set_valid(43, 43) &&
              !deepseek4_optional_vision_bias_set_valid(42, 43) &&
              !deepseek4_optional_vision_bias_set_valid(1, 43),
          "vision router bias is an all-or-none per-layer contract");
    CHECK(deepseek4_route_mode(true, true, vocab + DEEPSEEK4_IMAGE,
                              vocab) == Deepseek4RouteMode::SCORE_VISION,
          "image sentinel bypasses tid2eid in hash layers");
    CHECK(deepseek4_route_mode(true, true, 42, vocab) ==
              Deepseek4RouteMode::HASH_TEXT,
          "ordinary token retains hash routing");
    CHECK(deepseek4_route_mode(false, true, vocab + DEEPSEEK4_IMAGE_PAD,
                              vocab) == Deepseek4RouteMode::SCORE_VISION,
          "image padding selects bias_vl in score layers");
    CHECK(deepseek4_route_mode(false, true, 42, vocab) ==
              Deepseek4RouteMode::SCORE_TEXT,
          "ordinary score-routed token selects text bias");
    CHECK(deepseek4_route_mode(true, false, 42, vocab) ==
              Deepseek4RouteMode::HASH_TEXT,
          "text-only model remains on its original routing path");
    CHECK(deepseek4_route_mode(true, false,
                              vocab + DEEPSEEK4_IMAGE_START, vocab) ==
              Deepseek4RouteMode::INVALID_IMAGE,
          "image sentinel cannot index text hash table without vision weights");
}

int main() {
    test_two_row_n_layout();
    test_grid_sweep_against_scalar_reference();
    test_padding_and_mutations();
    test_embedding_assembly();
    test_trailing_alignment_pad();
    test_visibility_and_prefill_boundaries();
    test_image_aware_prefill_chunks();
    test_routing_modes();
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
