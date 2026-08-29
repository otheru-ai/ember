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

bool checked_advance(int64_t current, int64_t amount, int64_t & next,
                     std::string & error) {
    if (amount < 0 || current > std::numeric_limits<int64_t>::max() - amount) {
        error = "Qwen4Exp M-RoPE position exceeds int64 range";
        return false;
    }
    next = current + amount;
    return true;
}

} // namespace

bool qwen4exp_assign_mrope_positions(
    const std::vector<Qwen4ExpMropeRun> & runs, size_t token_count,
    std::array<std::vector<int32_t>, 3> & positions, int64_t & rope_delta,
    std::string & error) {
    error.clear();
    int64_t current = 0;
    int64_t maximum = -1;
    size_t counted = 0;
    for (const Qwen4ExpMropeRun & run : runs) {
        if (run.token_count > token_count - counted) {
            error = "Qwen4Exp M-RoPE run count exceeds token count";
            return false;
        }
        counted += run.token_count;
        if (!run.image) {
            if (run.token_count > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
                error = "Qwen4Exp M-RoPE text position exceeds int64 range";
                return false;
            }
            if (run.token_count != 0) {
                int64_t last = 0;
                if (!checked_advance(current,
                                     static_cast<int64_t>(run.token_count - 1),
                                     last, error) ||
                    last > std::numeric_limits<int32_t>::max()) {
                    error = "Qwen4Exp M-RoPE position exceeds int32 range";
                    return false;
                }
                maximum = std::max(maximum, last);
            }
            if (!checked_advance(current, static_cast<int64_t>(run.token_count),
                                 current, error)) return false;
            continue;
        }
        const Qwen4ExpVisionGrid & grid = run.grid;
        const size_t merged = qwen4exp_vision_merged_tokens(grid);
        if (grid.t != 1 || merged == 0 || merged != run.token_count) {
            error = "invalid Qwen4Exp vision position run contract";
            return false;
        }
        const int64_t max_axis = static_cast<int64_t>(
            std::max(grid.h, grid.w) / Qwen4ExpVisionContract::spatial_merge_size);
        const int64_t max_offset = static_cast<int64_t>(
            std::max(grid.h, grid.w) / Qwen4ExpVisionContract::spatial_merge_size - 1);
        int64_t last = 0;
        if (!checked_advance(current, max_offset, last, error) ||
            last > std::numeric_limits<int32_t>::max()) {
            error = "Qwen4Exp M-RoPE position exceeds int32 range";
            return false;
        }
        maximum = std::max(maximum, last);
        if (!checked_advance(current, max_axis, current, error)) return false;
    }
    if (counted != token_count) {
        error = "Qwen4Exp M-RoPE run count does not match token count";
        return false;
    }

    for (auto & axis : positions) {
        axis.clear();
        axis.reserve(token_count);
    }
    current = 0;
    for (const Qwen4ExpMropeRun & run : runs) {
        if (!run.image) {
            for (size_t i = 0; i < run.token_count; ++i) {
                const int64_t p = current + static_cast<int64_t>(i);
                if (!append_position(positions, p, p, p, error)) return false;
            }
            current += static_cast<int64_t>(run.token_count);
            continue;
        }
        const uint32_t merged_h = run.grid.h /
            Qwen4ExpVisionContract::spatial_merge_size;
        const uint32_t merged_w = run.grid.w /
            Qwen4ExpVisionContract::spatial_merge_size;
        for (uint32_t h = 0; h < merged_h; ++h) {
            for (uint32_t w = 0; w < merged_w; ++w) {
                if (!append_position(positions, current,
                                     current + static_cast<int64_t>(h),
                                     current + static_cast<int64_t>(w), error)) {
                    return false;
                }
            }
        }
        current += static_cast<int64_t>(std::max(run.grid.h, run.grid.w) /
            Qwen4ExpVisionContract::spatial_merge_size);
    }
    rope_delta = token_count == 0 ? 0 : maximum + 1 -
        static_cast<int64_t>(token_count);
    return true;
}

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
    std::vector<Qwen4ExpMropeRun> mrope_runs;

    while (cursor < tokens) {
        const Qwen4ExpModality modality = modalities[cursor];
        size_t end = cursor + 1;
        while (end < tokens && modalities[end] == modality) ++end;
        const size_t run = end - cursor;

        if (modality == Qwen4ExpModality::TEXT) {
            mrope_runs.push_back({run, false, {}});
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

            mrope_runs.push_back({run, true, image.grid});
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
    if (!qwen4exp_assign_mrope_positions(mrope_runs, tokens,
                                         out.position_ids, out.rope_delta,
                                         error)) {
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
