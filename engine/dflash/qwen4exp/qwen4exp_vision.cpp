#include "qwen4exp_vision.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace dflash::common {
namespace {

constexpr size_t kTextWidth = Qwen4ExpVisionContract::output_hidden_size;

bool checked_mul(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    out = a * b;
    return true;
}

bool append_position(std::array<std::vector<int32_t>, 3> & positions,
                     int64_t t, int64_t h, int64_t w, std::string & error) {
    constexpr int64_t kMax = std::numeric_limits<int32_t>::max();
    if (t < 0 || h < 0 || w < 0 || t > kMax || h > kMax || w > kMax) {
        error = "Qwen4Exp M-RoPE position exceeds int32 range";
        return false;
    }
    positions[0].push_back(static_cast<int32_t>(t));
    positions[1].push_back(static_cast<int32_t>(h));
    positions[2].push_back(static_cast<int32_t>(w));
    return true;
}

} // namespace

bool qwen4exp_validate_vision_config(const Qwen4ExpVisionConfig & config,
                                     std::string & error) {
    error.clear();
    const struct Field {
        const char * name;
        uint32_t actual;
        uint32_t expected;
    } fields[] = {
        {"depth", config.depth, Qwen4ExpVisionContract::depth},
        {"hidden_size", config.hidden_size,
         Qwen4ExpVisionContract::hidden_size},
        {"intermediate_size", config.intermediate_size,
         Qwen4ExpVisionContract::intermediate_size},
        {"num_heads", config.head_count,
         Qwen4ExpVisionContract::head_count},
        {"in_channels", config.input_channels,
         Qwen4ExpVisionContract::input_channels},
        {"patch_size", config.patch_size,
         Qwen4ExpVisionContract::patch_size},
        {"temporal_patch_size", config.temporal_patch_size,
         Qwen4ExpVisionContract::temporal_patch_size},
        {"spatial_merge_size", config.spatial_merge_size,
         Qwen4ExpVisionContract::spatial_merge_size},
        {"out_hidden_size", config.output_hidden_size,
         Qwen4ExpVisionContract::output_hidden_size},
        {"num_position_embeddings", config.position_embeddings,
         Qwen4ExpVisionContract::position_embeddings},
        {"deepstack_visual_indexes length", config.deepstack_layer_count, 0},
    };
    for (const Field & field : fields) {
        if (field.actual != field.expected) {
            error = std::string("Qwen4Exp vision ") + field.name +
                    " mismatch: expected " + std::to_string(field.expected) +
                    ", got " + std::to_string(field.actual);
            return false;
        }
    }
    return true;
}

size_t qwen4exp_vision_merged_tokens(const Qwen4ExpVisionGrid & grid) {
    constexpr size_t merge = Qwen4ExpVisionContract::spatial_merge_size;
    if (grid.t == 0 || grid.h == 0 || grid.w == 0 ||
        grid.h % merge != 0 || grid.w % merge != 0) {
        return 0;
    }
    size_t rows = static_cast<size_t>(grid.t);
    if (!checked_mul(rows, static_cast<size_t>(grid.h / merge), rows) ||
        !checked_mul(rows, static_cast<size_t>(grid.w / merge), rows)) {
        return 0;
    }
    return rows;
}

bool qwen4exp_vision_patch_positions(
    const std::vector<Qwen4ExpVisionGrid> & grids,
    std::vector<Qwen4ExpVisionPatchPosition> & positions,
    std::vector<int32_t> & attention_cu_seqlens, std::string & error) {
    error.clear();
    positions.clear();
    attention_cu_seqlens.clear();
    attention_cu_seqlens.push_back(0);

    int64_t cumulative = 0;
    constexpr uint32_t merge = Qwen4ExpVisionContract::spatial_merge_size;
    for (const Qwen4ExpVisionGrid & grid : grids) {
        if (qwen4exp_vision_merged_tokens(grid) == 0) {
            error = "Qwen4Exp vision grid must be nonzero and divisible by the 2x2 merger";
            positions.clear();
            attention_cu_seqlens.clear();
            return false;
        }
        for (uint32_t frame = 0; frame < grid.t; ++frame) {
            (void)frame; // 2-D ViT RoPE repeats the same H/W grid per frame.
            for (uint32_t block_h = 0; block_h < grid.h / merge; ++block_h) {
                for (uint32_t block_w = 0; block_w < grid.w / merge; ++block_w) {
                    for (uint32_t inner_h = 0; inner_h < merge; ++inner_h) {
                        for (uint32_t inner_w = 0; inner_w < merge; ++inner_w) {
                            positions.push_back({
                                static_cast<int32_t>(block_h * merge + inner_h),
                                static_cast<int32_t>(block_w * merge + inner_w),
                            });
                        }
                    }
                }
            }
            cumulative += static_cast<int64_t>(grid.h) * grid.w;
            if (cumulative > std::numeric_limits<int32_t>::max()) {
                error = "Qwen4Exp vision attention length exceeds int32 range";
                positions.clear();
                attention_cu_seqlens.clear();
                return false;
            }
            // Reference get_vision_cu_seqlens creates one segment per frame.
            attention_cu_seqlens.push_back(static_cast<int32_t>(cumulative));
        }
    }
    return true;
}

bool qwen4exp_prepare_vision_input(
    const std::vector<int32_t> & input_ids,
    const std::vector<Qwen4ExpModality> & modalities,
    const std::vector<float> & text_embeddings,
    const std::vector<Qwen4ExpEncodedImage> & images,
    Qwen4ExpPreparedVisionInput & out, std::string & error) {
    error.clear();
    out = {};
    const size_t tokens = input_ids.size();
    if (modalities.size() != tokens) {
        error = "Qwen4Exp modality count does not match prompt token count";
        return false;
    }
    size_t embedding_count = 0;
    if (!checked_mul(tokens, kTextWidth, embedding_count) ||
        text_embeddings.size() != embedding_count) {
        error = "Qwen4Exp text embedding matrix must be [token_count,2560]";
        return false;
    }
    for (size_t i = 0; i < tokens; ++i) {
        const bool placeholder = input_ids[i] ==
            static_cast<int32_t>(Qwen4ExpVisionContract::image_token_id);
        const bool image_modality = modalities[i] == Qwen4ExpModality::IMAGE;
        if (placeholder != image_modality) {
            error = "Qwen4Exp image placeholder and modality masks do not match";
            return false;
        }
    }

    out.embeddings = text_embeddings;
    out.ple_input_ids = input_ids;
    for (auto & axis : out.position_ids) axis.reserve(tokens);
    out.text_position_ids.reserve(tokens);
    for (size_t i = 0; i < tokens; ++i) {
        if (i > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            error = "Qwen4Exp text position exceeds int32 range";
            out = {};
            return false;
        }
        out.text_position_ids.push_back(static_cast<int32_t>(i));
    }

    size_t image_index = 0;
    size_t cursor = 0;
    int64_t current_position = 0;
    int64_t maximum_position = -1;

    while (cursor < tokens) {
        const Qwen4ExpModality modality = modalities[cursor];
        size_t end = cursor + 1;
        while (end < tokens && modalities[end] == modality) ++end;
        const size_t run = end - cursor;

        if (modality == Qwen4ExpModality::TEXT) {
            for (size_t i = 0; i < run; ++i) {
                const int64_t p = current_position + static_cast<int64_t>(i);
                if (!append_position(out.position_ids, p, p, p, error)) {
                    out = {};
                    return false;
                }
                maximum_position = std::max(maximum_position, p);
            }
            current_position += static_cast<int64_t>(run);
        } else if (modality == Qwen4ExpModality::IMAGE) {
            if (image_index >= images.size()) {
                error = "Qwen4Exp image modality run has no encoded image";
                out = {};
                return false;
            }
            const Qwen4ExpEncodedImage & image = images[image_index++];
            // The image processor emits T=1.  T>1 belongs to the separate
            // timestamp-aware video path; accepting it here could make the
            // reference's spatial-only current_position increment overlap a
            // later text span when T is the largest axis.
            if (image.grid.t != 1) {
                error = "Qwen4Exp image grid T must be 1; video is not implemented";
                out = {};
                return false;
            }
            const size_t expected_rows = qwen4exp_vision_merged_tokens(image.grid);
            if (expected_rows == 0 || run != expected_rows) {
                error = "Qwen4Exp image placeholder count does not match merged vision rows";
                out = {};
                return false;
            }
            size_t expected_values = 0;
            if (!checked_mul(expected_rows, kTextWidth, expected_values) ||
                image.embeddings.size() != expected_values) {
                error = "Qwen4Exp encoded image matrix must be [merged_tokens,2560]";
                out = {};
                return false;
            }
            for (size_t i = 0; i < run; ++i) {
                if (input_ids[cursor + i] !=
                    static_cast<int32_t>(Qwen4ExpVisionContract::image_token_id)) {
                    error = "Qwen4Exp image modality requires image placeholder token 248056";
                    out = {};
                    return false;
                }
                const float * src = image.embeddings.data() + i * kTextWidth;
                float * dst = out.embeddings.data() + (cursor + i) * kTextWidth;
                std::memcpy(dst, src, kTextWidth * sizeof(float));
            }

            const uint32_t merged_h =
                image.grid.h / Qwen4ExpVisionContract::spatial_merge_size;
            const uint32_t merged_w =
                image.grid.w / Qwen4ExpVisionContract::spatial_merge_size;
            for (uint32_t t = 0; t < image.grid.t; ++t) {
                for (uint32_t h = 0; h < merged_h; ++h) {
                    for (uint32_t w = 0; w < merged_w; ++w) {
                        const int64_t tp = current_position + t;
                        const int64_t hp = current_position + h;
                        const int64_t wp = current_position + w;
                        if (!append_position(out.position_ids, tp, hp, wp,
                                             error)) {
                            out = {};
                            return false;
                        }
                        maximum_position = std::max(
                            maximum_position, std::max(tp, std::max(hp, wp)));
                    }
                }
            }
            // This intentionally excludes T, matching get_rope_index.
            current_position += static_cast<int64_t>(
                std::max(image.grid.h, image.grid.w) /
                Qwen4ExpVisionContract::spatial_merge_size);
        } else {
            error = "Qwen4Exp video vision path is not implemented";
            out = {};
            return false;
        }
        cursor = end;
    }

    if (image_index != images.size()) {
        error = "Qwen4Exp encoded image has no matching placeholder run";
        out = {};
        return false;
    }
    for (const auto & axis : out.position_ids) {
        if (axis.size() != tokens) {
            error = "internal Qwen4Exp M-RoPE position count mismatch";
            out = {};
            return false;
        }
    }
    if (out.text_position_ids.size() != tokens) {
        error = "internal Qwen4Exp text position count mismatch";
        out = {};
        return false;
    }
    // Empty prompts follow the mathematical max+1-length identity at zero.
    out.rope_delta = tokens == 0
        ? 0
        : maximum_position + 1 - static_cast<int64_t>(tokens);
    return true;
}

bool qwen4exp_vision_incremental_positions(
    int64_t past_seen_tokens, int64_t rope_delta, size_t token_count,
    std::array<std::vector<int32_t>, 3> & positions, std::string & error) {
    error.clear();
    for (auto & axis : positions) {
        axis.clear();
        axis.reserve(token_count);
    }
    if (past_seen_tokens < 0) {
        error = "Qwen4Exp past token count cannot be negative";
        return false;
    }
    for (size_t i = 0; i < token_count; ++i) {
        if (i > static_cast<size_t>(std::numeric_limits<int64_t>::max() -
                                    past_seen_tokens)) {
            error = "Qwen4Exp incremental position overflow";
            for (auto & axis : positions) axis.clear();
            return false;
        }
        const int64_t base = past_seen_tokens + static_cast<int64_t>(i);
        if ((rope_delta > 0 &&
             base > std::numeric_limits<int64_t>::max() - rope_delta) ||
            (rope_delta < 0 &&
             base < std::numeric_limits<int64_t>::min() - rope_delta)) {
            error = "Qwen4Exp incremental position overflow";
            for (auto & axis : positions) axis.clear();
            return false;
        }
        const int64_t p = base + rope_delta;
        if (!append_position(positions, p, p, p, error)) {
            for (auto & axis : positions) axis.clear();
            return false;
        }
    }
    return true;
}

} // namespace dflash::common
