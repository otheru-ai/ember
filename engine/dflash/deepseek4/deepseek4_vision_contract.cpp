#include "deepseek4_vision_contract.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace dflash {
namespace {

void set_error(std::string * error, const char * text) {
    if (error) *error = text;
}

bool is_type(int32_t token, int32_t vocab_size, Deepseek4ImageType type) {
    return token == vocab_size + static_cast<int32_t>(type);
}

}  // namespace

bool deepseek4_build_image_block(int32_t vocab_size, int n_llm_h, int n_llm_w,
                                 int start_pos, Deepseek4ImageBlock & out,
                                 std::string * error) {
    out = {};
    if (vocab_size <= 0 || vocab_size > INT32_MAX - DEEPSEEK4_IMAGE_END ||
        n_llm_h <= 0 || n_llm_w <= 0 || start_pos < 0) {
        set_error(error, "invalid image block dimensions or vocabulary");
        return false;
    }

    const int64_t rows = static_cast<int64_t>(n_llm_h) + n_llm_h % 2;
    const int64_t row_len = static_cast<int64_t>(n_llm_w) + 1;
    const int64_t ordered = rows * row_len;
    const int compress_pad = 3 - start_pos % 4;
    const int pad_last = static_cast<int>(((rows / 2 * row_len) % 2) * 2);
    const int64_t total = compress_pad + 1 + ordered + pad_last + 1;
    const int64_t image_count =
        static_cast<int64_t>(n_llm_h) * n_llm_w;
    if (ordered <= 0 || total > INT_MAX || image_count > INT_MAX ||
        static_cast<uint64_t>(total) >
            std::numeric_limits<size_t>::max() / sizeof(int32_t)) {
        set_error(error, "image block is too large");
        return false;
    }

    out.token_ids.reserve(static_cast<size_t>(total));
    out.image_perm.reserve(static_cast<size_t>(image_count));
    const auto push_type = [&](Deepseek4ImageType type) {
        out.token_ids.push_back(vocab_size + static_cast<int32_t>(type));
    };
    for (int i = 0; i < compress_pad; ++i)
        push_type(DEEPSEEK4_IMAGE_PAD);
    push_type(DEEPSEEK4_IMAGE_START);

    // image_processor.py:142-147: reshape rows into pairs, transpose the
    // [2,row_len] axes, then flatten. This is column-major within each row
    // pair, while image_perm retains the aligner's original row-major index.
    for (int row_pair = 0; row_pair < static_cast<int>(rows / 2); ++row_pair) {
        for (int col = 0; col < static_cast<int>(row_len); ++col) {
            for (int side = 0; side < 2; ++side) {
                const int row = row_pair * 2 + side;
                if (row < n_llm_h && col < n_llm_w) {
                    push_type(DEEPSEEK4_IMAGE);
                    out.image_perm.push_back(row * n_llm_w + col);
                } else if (row < n_llm_h) {
                    push_type(DEEPSEEK4_IMAGE_NEWLINE);
                } else {
                    push_type(DEEPSEEK4_IMAGE_PAD);
                }
            }
        }
    }
    for (int i = 0; i < pad_last; ++i)
        push_type(DEEPSEEK4_IMAGE_PAD);
    push_type(DEEPSEEK4_IMAGE_END);
    return true;
}

bool deepseek4_validate_image_block(const std::vector<int32_t> & token_ids,
                                    int32_t vocab_size, int n_llm_h,
                                    int n_llm_w, int start_pos,
                                    std::string * error) {
    Deepseek4ImageBlock expected;
    if (!deepseek4_build_image_block(vocab_size, n_llm_h, n_llm_w,
                                     start_pos, expected, error))
        return false;
    if (token_ids != expected.token_ids) {
        set_error(error, "image block does not match the learned N-layout");
        return false;
    }
    return true;
}

bool deepseek4_prepare_image(int32_t vocab_size, int n_llm_h, int n_llm_w,
                             int start_pos, int n_embd,
                             const std::vector<float> & image_embeddings,
                             const Deepseek4ImageMarkers & markers,
                             Deepseek4PreparedImage & out,
                             std::string * error) {
    out = {};
    if (n_embd <= 0) {
        set_error(error, "invalid image embedding width");
        return false;
    }
    Deepseek4ImageBlock block;
    if (!deepseek4_build_image_block(vocab_size, n_llm_h, n_llm_w,
                                     start_pos, block, error)) {
        return false;
    }
    const size_t width = static_cast<size_t>(n_embd);
    const size_t image_rows = block.image_perm.size();
    if (image_rows > std::numeric_limits<size_t>::max() / width ||
        image_embeddings.size() != image_rows * width) {
        set_error(error, "aligner image rows do not match the required grid");
        return false;
    }
    if (markers.start.size() != width || markers.pad.size() != width ||
        markers.newline.size() != width || markers.end.size() != width) {
        set_error(error, "learned image marker width mismatch");
        return false;
    }
    if (block.token_ids.size() >
        std::numeric_limits<size_t>::max() / width) {
        set_error(error, "prepared image embedding matrix is too large");
        return false;
    }

    out.embeddings.resize(block.token_ids.size() * width);
    size_t image_slot = 0;
    for (size_t row = 0; row < block.token_ids.size(); ++row) {
        const int32_t type = block.token_ids[row] - vocab_size;
        const float * source = nullptr;
        switch (type) {
            case DEEPSEEK4_IMAGE_START:
                source = markers.start.data();
                break;
            case DEEPSEEK4_IMAGE_PAD:
                source = markers.pad.data();
                break;
            case DEEPSEEK4_IMAGE: {
                if (image_slot >= block.image_perm.size()) {
                    set_error(error, "image permutation underflow");
                    out = {};
                    return false;
                }
                const int32_t source_row = block.image_perm[image_slot++];
                if (source_row < 0 ||
                    static_cast<size_t>(source_row) >= image_rows) {
                    set_error(error, "image permutation is out of range");
                    out = {};
                    return false;
                }
                source = image_embeddings.data() +
                         static_cast<size_t>(source_row) * width;
                break;
            }
            case DEEPSEEK4_IMAGE_NEWLINE:
                source = markers.newline.data();
                break;
            case DEEPSEEK4_IMAGE_END:
                source = markers.end.data();
                break;
            default:
                set_error(error, "unknown learned image token type");
                out = {};
                return false;
        }
        std::memcpy(out.embeddings.data() + row * width, source,
                    width * sizeof(float));
    }
    if (image_slot != image_rows) {
        set_error(error, "image permutation did not consume every aligner row");
        out = {};
        return false;
    }
    out.block = std::move(block);
    return true;
}

void deepseek4_image_visible(const std::vector<int32_t> & input_ids,
                             int32_t vocab_size, int max_image_tokens,
                             std::vector<int32_t> & left,
                             std::vector<int32_t> & right) {
    const size_t n = input_ids.size();
    left.assign(n, 0);
    right.assign(n, 0);
    if (n == 0 || max_image_tokens <= 0) return;

    std::vector<int32_t> last_start(n, 0);
    std::vector<unsigned char> valid(n, 0);
    int32_t starts = 0;
    int32_t ends = 0;
    int32_t latest = 0;
    for (size_t i = 0; i < n; ++i) {
        const bool start = is_type(
            input_ids[i], vocab_size, DEEPSEEK4_IMAGE_START);
        const bool end = is_type(
            input_ids[i], vocab_size, DEEPSEEK4_IMAGE_END);
        if (start) {
            ++starts;
            latest = static_cast<int32_t>(i);
        }
        if (end) ++ends;
        last_start[i] = latest;
        valid[i] = static_cast<unsigned char>(starts > ends || end);
    }

    int32_t nearest_end = static_cast<int32_t>(n);
    for (size_t rev = n; rev-- > 0;) {
        if (is_type(input_ids[rev], vocab_size, DEEPSEEK4_IMAGE_END))
            nearest_end = static_cast<int32_t>(rev);
        if (!valid[rev]) continue;
        const int32_t pos = static_cast<int32_t>(rev);
        left[rev] = std::min(pos - last_start[rev], max_image_tokens - 1);
        right[rev] = std::min(nearest_end - pos, max_image_tokens);
    }
}

bool deepseek4_prefill_cut_safe(const std::vector<int32_t> & input_ids,
                                int32_t vocab_size, int cut) {
    if (cut < 0 || static_cast<size_t>(cut) > input_ids.size()) return false;
    int depth = 0;
    for (int i = 0; i < cut; ++i) {
        if (is_type(input_ids[static_cast<size_t>(i)], vocab_size,
                    DEEPSEEK4_IMAGE_START))
            ++depth;
        if (is_type(input_ids[static_cast<size_t>(i)], vocab_size,
                    DEEPSEEK4_IMAGE_END) && depth > 0)
            --depth;
    }
    return depth == 0;
}

bool deepseek4_chunk_accepts_image_tokens(int start_pos,
                                          const std::vector<int32_t> & ids,
                                          int32_t vocab_size) {
    if (start_pos == 0) return true;
    return std::all_of(ids.begin(), ids.end(), [vocab_size](int32_t id) {
        return id < vocab_size;
    });
}

Deepseek4RouteMode deepseek4_route_mode(bool hash_layer,
                                        bool vision_weights_loaded,
                                        int32_t token_id,
                                        int32_t vocab_size) {
    if (token_id >= vocab_size) {
        return vision_weights_loaded ? Deepseek4RouteMode::SCORE_VISION
                                     : Deepseek4RouteMode::INVALID_IMAGE;
    }
    if (hash_layer) return Deepseek4RouteMode::HASH_TEXT;
    return Deepseek4RouteMode::SCORE_TEXT;
}

}  // namespace dflash
