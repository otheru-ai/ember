// Qwen3.8-Flash-Next vision contract and GPU-free multimodal splice reference.
//
// The vision tower is Qwen3-VL-shaped, but image features are not ordinary
// prompt tokens: its 2x2 patch merger produces one 2560-wide row per image
// placeholder and the text graph must keep the original placeholder ids for
// PLE hashing.  M-RoPE positions also advance by the largest merged spatial
// axis, not by the number of image rows.  Keeping those rules in a small,
// dependency-free seam makes them testable before the HIP vision graph exists.
//
// Authoritative implementation/config:
//   Qwen/Qwen3.8-Flash-Next revision
//   f5d08274bafd880402bd16f5e3e6c514136ec06c
//   transformers revision
//   36deb0b53ed0863f4b4dfdea23dcaec7f3df3701
// Experimental conversion/graph naming reference (not a support claim):
//   llama.cpp PR #27742 head
//   035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
//
// This interface accepts only already-encoded image rows.  It deliberately
// does not decode files/URLs, preprocess pixels, or execute the ViT; the lazy
// provider owns those operations.  Callers fail closed when that provider is
// absent or when its rows do not satisfy this contract.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen4ExpVisionContract {
    static constexpr uint32_t depth = 27;
    static constexpr uint32_t hidden_size = 1152;
    static constexpr uint32_t intermediate_size = 4304;
    static constexpr uint32_t head_count = 16;
    static constexpr uint32_t head_dim = 72;
    static constexpr uint32_t input_channels = 3;
    static constexpr uint32_t patch_size = 16;
    static constexpr uint32_t temporal_patch_size = 2;
    static constexpr uint32_t spatial_merge_size = 2;
    static constexpr uint32_t output_hidden_size = 2560;
    static constexpr uint32_t position_embeddings = 2304;
    static constexpr uint32_t position_grid_side = 48;
    static constexpr uint32_t image_token_id = 248056;
    static constexpr uint32_t video_token_id = 248057;
    static constexpr uint32_t vision_start_token_id = 248053;
    static constexpr uint32_t vision_end_token_id = 248054;
};

struct Qwen4ExpVisionConfig {
    uint32_t depth = Qwen4ExpVisionContract::depth;
    uint32_t hidden_size = Qwen4ExpVisionContract::hidden_size;
    uint32_t intermediate_size = Qwen4ExpVisionContract::intermediate_size;
    uint32_t head_count = Qwen4ExpVisionContract::head_count;
    uint32_t input_channels = Qwen4ExpVisionContract::input_channels;
    uint32_t patch_size = Qwen4ExpVisionContract::patch_size;
    uint32_t temporal_patch_size =
        Qwen4ExpVisionContract::temporal_patch_size;
    uint32_t spatial_merge_size =
        Qwen4ExpVisionContract::spatial_merge_size;
    uint32_t output_hidden_size =
        Qwen4ExpVisionContract::output_hidden_size;
    uint32_t position_embeddings =
        Qwen4ExpVisionContract::position_embeddings;
    uint32_t deepstack_layer_count = 0;
};

bool qwen4exp_validate_vision_config(const Qwen4ExpVisionConfig & config,
                                     std::string & error);

struct Qwen4ExpVisionGrid {
    uint32_t t = 0;
    uint32_t h = 0;
    uint32_t w = 0;
};

struct Qwen4ExpMropeRun {
    size_t token_count = 0;
    bool image = false;
    Qwen4ExpVisionGrid grid;
};

bool qwen4exp_assign_mrope_positions(
    const std::vector<Qwen4ExpMropeRun> & runs, size_t token_count,
    std::array<std::vector<int32_t>, 3> & positions, int64_t & rope_delta,
    std::string & error);

// Number of rows after the official spatial merger.  Zero means the grid is
// malformed or its product overflows size_t.
size_t qwen4exp_vision_merged_tokens(const Qwen4ExpVisionGrid & grid);

struct Qwen4ExpVisionPatchPosition {
    int32_t h = 0;
    int32_t w = 0;
};

// Qwen4ExpVisionModel.forward/get_vision_position_ids: returns one position
// for every pre-merger patch in temporal, spatial-block, within-block order.
bool qwen4exp_vision_patch_positions(
    const std::vector<Qwen4ExpVisionGrid> & grids,
    std::vector<Qwen4ExpVisionPatchPosition> & positions,
    std::vector<int32_t> & attention_cu_seqlens, std::string & error);

enum class Qwen4ExpModality : uint8_t {
    TEXT = 0,
    IMAGE = 1,
    VIDEO = 2,
};

struct Qwen4ExpEncodedImage {
    Qwen4ExpVisionGrid grid;
    // Row-major [merged_tokens, 2560].
    std::vector<float> embeddings;
};

struct Qwen4ExpPreparedVisionInput {
    // Row-major [token_count, 2560], with image placeholders replaced in
    // prompt order.  Text rows remain byte-for-byte equal to the input rows.
    std::vector<float> embeddings;

    // PLE hashes the pre-splice ids.  In particular, every image row remains
    // token 248056 even though its language-model embedding was replaced.
    std::vector<int32_t> ple_input_ids;

    // Axis-major [T,H,W], one entry per prompt token.
    std::array<std::vector<int32_t>, 3> position_ids;

    // Qwen4ExpTextModel receives a fourth, ordinary scalar position lane for
    // the QSA indexer before passing the remaining T/H/W lanes to M-RoPE.
    // Unlike the vision axes above, this is always the physical token offset.
    std::vector<int32_t> text_position_ids;
    int64_t rope_delta = 0;
};

// Exact single-sequence counterpart of Qwen4ExpModel.get_rope_index plus
// masked_scatter.  Every contiguous IMAGE modality run consumes exactly one
// encoded image, in order.  VIDEO is rejected until its timestamp-aware split
// and encoder path are implemented.
bool qwen4exp_prepare_vision_input(
    const std::vector<int32_t> & input_ids,
    const std::vector<Qwen4ExpModality> & modalities,
    const std::vector<float> & text_embeddings,
    const std::vector<Qwen4ExpEncodedImage> & images,
    Qwen4ExpPreparedVisionInput & out, std::string & error);

// Incremental decode positions after a multimodal prefill.  Transformers adds
// the cached rope_delta to the ordinary past-token scalar on all three axes.
bool qwen4exp_vision_incremental_positions(
    int64_t past_seen_tokens, int64_t rope_delta, size_t token_count,
    std::array<std::vector<int32_t>, 3> & positions, std::string & error);

} // namespace dflash::common
