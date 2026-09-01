#include "deepseek4_vision_contract.h"

#include <algorithm>
#include <climits>
#include <cmath>
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

bool deepseek4_rebase_vision_runs(
        const std::vector<Deepseek4VisionRunView> & request_runs,
        int prefill_offset, int prefill_tokens,
        std::vector<Deepseek4VisionRunView> & local_runs,
        std::string * error) {
    local_runs.clear();
    if (prefill_offset < 0 || prefill_tokens < 0 ||
        prefill_offset > INT_MAX - prefill_tokens) {
        set_error(error, "invalid DeepSeek4 vision prefill span");
        return false;
    }
    const int prefill_end = prefill_offset + prefill_tokens;
    int previous_end = 0;
    for (const Deepseek4VisionRunView & run : request_runs) {
        if (run.prompt_offset < 0 || run.n_tokens <= 0 ||
            run.prompt_offset > INT_MAX - run.n_tokens ||
            run.prompt_offset < previous_end || run.embedding_width <= 0 ||
            !run.token_ids || !run.embeddings ||
            static_cast<size_t>(run.n_tokens) >
                std::numeric_limits<size_t>::max() /
                    static_cast<size_t>(run.embedding_width) ||
            run.embedding_values != static_cast<size_t>(run.n_tokens) *
                static_cast<size_t>(run.embedding_width)) {
            set_error(error, "invalid or unordered request-owned vision runs");
            local_runs.clear();
            return false;
        }
        const int run_end = run.prompt_offset + run.n_tokens;
        previous_end = run_end;
        if (run_end <= prefill_offset || run.prompt_offset >= prefill_end)
            continue;
        if (run.prompt_offset < prefill_offset || run_end > prefill_end) {
            set_error(error, "prefill span bisects a learned image run");
            local_runs.clear();
            return false;
        }
        Deepseek4VisionRunView local = run;
        local.prompt_offset -= prefill_offset;
        local_runs.push_back(local);
    }
    return true;
}

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

bool deepseek4_expand_image_placeholders(
        const std::vector<int32_t> & input_ids,
        const std::vector<int32_t> & placeholder_ids,
        int32_t vocab_size, int n_embd,
        const std::vector<Deepseek4ImageRows> & images,
        const Deepseek4ImageMarkers & markers,
        std::vector<int32_t> & output_ids,
        std::vector<Deepseek4PreparedRun> & runs,
        std::string * error) {
    output_ids.clear();
    runs.clear();
    if (placeholder_ids.empty() || input_ids.size() >
            static_cast<size_t>(INT_MAX)) {
        set_error(error, "invalid DeepSeek4 image placeholder contract");
        return false;
    }
    size_t cursor = 0;
    for (const Deepseek4ImageRows & source : images) {
        const auto found = std::search(
            input_ids.begin() + static_cast<std::ptrdiff_t>(cursor),
            input_ids.end(), placeholder_ids.begin(), placeholder_ids.end());
        if (found == input_ids.end()) {
            set_error(error, "rendered prompt has fewer image placeholders than images");
            output_ids.clear();
            runs.clear();
            return false;
        }
        const size_t placeholder = static_cast<size_t>(
            found - input_ids.begin());
        output_ids.insert(output_ids.end(),
                          input_ids.begin() + static_cast<std::ptrdiff_t>(cursor),
                          found);
        if (output_ids.size() > static_cast<size_t>(INT_MAX)) {
            set_error(error, "expanded DeepSeek4 vision prompt is too large");
            output_ids.clear();
            runs.clear();
            return false;
        }
        Deepseek4PreparedRun run;
        run.prompt_offset = static_cast<int>(output_ids.size());
        if (!deepseek4_prepare_image(
                vocab_size, source.n_llm_h, source.n_llm_w,
                run.prompt_offset, n_embd, source.embeddings, markers,
                run.image, error)) {
            output_ids.clear();
            runs.clear();
            return false;
        }
        if (run.image.block.token_ids.size() > static_cast<size_t>(INT_MAX) -
                output_ids.size()) {
            set_error(error, "expanded DeepSeek4 vision prompt is too large");
            output_ids.clear();
            runs.clear();
            return false;
        }
        output_ids.insert(output_ids.end(), run.image.block.token_ids.begin(),
                          run.image.block.token_ids.end());
        runs.push_back(std::move(run));
        cursor = placeholder + placeholder_ids.size();
    }
    const auto extra = std::search(
        input_ids.begin() + static_cast<std::ptrdiff_t>(cursor),
        input_ids.end(), placeholder_ids.begin(), placeholder_ids.end());
    if (extra != input_ids.end()) {
        set_error(error, "rendered prompt has more image placeholders than images");
        output_ids.clear();
        runs.clear();
        return false;
    }
    output_ids.insert(output_ids.end(),
                      input_ids.begin() + static_cast<std::ptrdiff_t>(cursor),
                      input_ids.end());
    if (output_ids.size() > static_cast<size_t>(INT_MAX)) {
        set_error(error, "expanded DeepSeek4 vision prompt is too large");
        output_ids.clear();
        runs.clear();
        return false;
    }
    return true;
}

bool deepseek4_prepare_vision_prefill(
        const std::vector<int32_t> & input_ids, int32_t vocab_size,
        int n_embd, const std::vector<Deepseek4VisionRunView> & runs,
        Deepseek4EmbedOnlyTokenIds & embed_token_ids, std::string * error) {
    embed_token_ids.values.clear();
    if (vocab_size <= 0 ||
        vocab_size > INT32_MAX - DEEPSEEK4_IMAGE_END || n_embd <= 0 ||
        input_ids.size() > static_cast<size_t>(INT_MAX)) {
        set_error(error, "invalid DeepSeek4 vision prefill shape");
        return false;
    }

    embed_token_ids.values = input_ids;
    size_t covered_until = 0;
    for (const Deepseek4VisionRunView & run : runs) {
        if (run.prompt_offset < 0 || run.n_tokens <= 0 ||
            run.embedding_width != n_embd || !run.token_ids ||
            !run.embeddings) {
            set_error(error, "invalid request-owned DeepSeek4 vision run");
            embed_token_ids.values.clear();
            return false;
        }
        const size_t start = static_cast<size_t>(run.prompt_offset);
        const size_t count = static_cast<size_t>(run.n_tokens);
        if (start < covered_until || start > input_ids.size() ||
            count > input_ids.size() - start ||
            count > std::numeric_limits<size_t>::max() /
                        static_cast<size_t>(n_embd) ||
            run.embedding_values != count * static_cast<size_t>(n_embd)) {
            set_error(error, "overlapping or out-of-range DeepSeek4 vision run");
            embed_token_ids.values.clear();
            return false;
        }
        for (size_t i = covered_until; i < start; ++i) {
            if (input_ids[i] < 0 || input_ids[i] >= vocab_size) {
                set_error(error, "learned image token is not covered by a vision run");
                embed_token_ids.values.clear();
                return false;
            }
        }

        bool saw_start = false;
        bool saw_image = false;
        bool saw_end = false;
        for (size_t row = 0; row < count; ++row) {
            const int32_t token = run.token_ids[row];
            if (input_ids[start + row] != token || token < vocab_size ||
                token > vocab_size + DEEPSEEK4_IMAGE_END) {
                set_error(error, "vision run token ids do not match the expanded prompt");
                embed_token_ids.values.clear();
                return false;
            }
            const int32_t type = token - vocab_size;
            if (type == DEEPSEEK4_IMAGE_START) {
                if (saw_start || saw_end) {
                    set_error(error, "vision run has duplicate or misplaced image start");
                    embed_token_ids.values.clear();
                    return false;
                }
                saw_start = true;
            } else if (type == DEEPSEEK4_IMAGE_END) {
                if (!saw_start || saw_end || row + 1 != count) {
                    set_error(error, "vision run has misplaced image end");
                    embed_token_ids.values.clear();
                    return false;
                }
                saw_end = true;
            } else if (!saw_start && type != DEEPSEEK4_IMAGE_PAD) {
                set_error(error, "vision run has content before image start");
                embed_token_ids.values.clear();
                return false;
            }
            if (type == DEEPSEEK4_IMAGE) saw_image = true;
            embed_token_ids.values[start + row] = 0;
        }
        if (!saw_start || !saw_image || !saw_end) {
            set_error(error, "vision run is missing a learned boundary or image row");
            embed_token_ids.values.clear();
            return false;
        }
        for (size_t value = 0; value < run.embedding_values; ++value) {
            if (!std::isfinite(run.embeddings[value])) {
                set_error(error, "vision run contains a non-finite embedding");
                embed_token_ids.values.clear();
                return false;
            }
        }
        covered_until = start + count;
    }
    for (size_t i = covered_until; i < input_ids.size(); ++i) {
        if (input_ids[i] < 0 || input_ids[i] >= vocab_size) {
            set_error(error, "learned image token is not covered by a vision run");
            embed_token_ids.values.clear();
            return false;
        }
    }
    return true;
}

bool deepseek4_splice_vision_chunk(
        const std::vector<Deepseek4VisionRunView> & runs, int n_embd,
        int chunk_offset, int chunk_tokens, float * embeddings,
        std::string * error) {
    if (n_embd <= 0 || chunk_offset < 0 || chunk_tokens <= 0 || !embeddings ||
        chunk_offset > INT_MAX - chunk_tokens) {
        set_error(error, "invalid DeepSeek4 vision splice request");
        return false;
    }
    const int chunk_end = chunk_offset + chunk_tokens;
    for (const Deepseek4VisionRunView & run : runs) {
        if (run.prompt_offset < 0 || run.n_tokens <= 0 ||
            run.prompt_offset > INT_MAX - run.n_tokens ||
            run.embedding_width != n_embd || !run.embeddings) {
            set_error(error, "invalid request-owned DeepSeek4 vision run");
            return false;
        }
        const int run_end = run.prompt_offset + run.n_tokens;
        const size_t values = static_cast<size_t>(run.n_tokens) *
                              static_cast<size_t>(n_embd);
        if (run.embedding_values != values) {
            set_error(error, "vision run embedding length changed after validation");
            return false;
        }
        if (run_end <= chunk_offset || run.prompt_offset >= chunk_end) continue;
        if (run.prompt_offset < chunk_offset || run_end > chunk_end) {
            set_error(error, "prefill chunk splits a learned image run");
            return false;
        }
        const size_t destination =
            static_cast<size_t>(run.prompt_offset - chunk_offset) *
            static_cast<size_t>(n_embd);
        std::memcpy(embeddings + destination, run.embeddings,
                    values * sizeof(float));
    }
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

bool deepseek4_raw_attention_visible(int query_pos, int key_pos,
                                     int window_size, int visible_left,
                                     int visible_right) {
    if (query_pos < 0 || key_pos < 0 || window_size <= 0 ||
        visible_left < 0 || visible_right < 0) {
        return false;
    }
    const int64_t ordinary_start =
        static_cast<int64_t>(query_pos) - window_size + 1;
    const int64_t image_start =
        static_cast<int64_t>(query_pos) - visible_left;
    const int64_t start = std::min(ordinary_start, image_start);
    const int64_t end =
        static_cast<int64_t>(query_pos) + visible_right;
    return static_cast<int64_t>(key_pos) >= std::max<int64_t>(0, start) &&
           static_cast<int64_t>(key_pos) <= end;
}

bool deepseek4_validate_vision_chunk_ids(
        const std::vector<int32_t> & input_ids, int32_t vocab_size,
        std::string * error) {
    if (vocab_size <= 0 ||
        vocab_size > INT32_MAX - DEEPSEEK4_IMAGE_END) {
        set_error(error, "invalid vision-chunk vocabulary");
        return false;
    }
    bool inside = false;
    bool pending_pad = false;
    bool saw_image = false;
    bool saw_block = false;
    for (int32_t token : input_ids) {
        if (token >= 0 && token < vocab_size) {
            if (inside || pending_pad) {
                set_error(error, "text token interrupts a learned image block");
                return false;
            }
            continue;
        }
        if (token < vocab_size || token > vocab_size + DEEPSEEK4_IMAGE_END) {
            set_error(error, "unknown token id in vision prefill chunk");
            return false;
        }
        const int32_t type = token - vocab_size;
        if (!inside) {
            if (type == DEEPSEEK4_IMAGE_PAD) {
                pending_pad = true;
                continue;
            }
            if (type != DEEPSEEK4_IMAGE_START) {
                set_error(error, "learned image content appears outside a block");
                return false;
            }
            pending_pad = false;
            inside = true;
            saw_image = false;
            continue;
        }
        if (type == DEEPSEEK4_IMAGE_START) {
            set_error(error, "nested learned image block");
            return false;
        }
        if (type == DEEPSEEK4_IMAGE_END) {
            if (!saw_image) {
                set_error(error, "learned image block has no image row");
                return false;
            }
            inside = false;
            saw_block = true;
            continue;
        }
        if (type == DEEPSEEK4_IMAGE) saw_image = true;
    }
    if (inside || pending_pad) {
        set_error(error, "prefill chunk ends inside a learned image block");
        return false;
    }
    if (!saw_block) {
        set_error(error, "vision prefill chunk contains no complete image block");
        return false;
    }
    return true;
}

bool deepseek4_vision_graph_cache_safe(
        const std::vector<int32_t> & input_ids, int32_t vocab_size) {
    return vocab_size > 0 &&
           std::all_of(input_ids.begin(), input_ids.end(),
                       [vocab_size](int32_t token) {
                           return token >= 0 && token < vocab_size;
                       });
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

int deepseek4_image_aware_prefill_chunk(
        const std::vector<int32_t> & input_ids, int32_t vocab_size,
        int offset, int proposed, int max_chunk, std::string * error) {
    if (vocab_size <= 0 || vocab_size > INT32_MAX - DEEPSEEK4_IMAGE_END ||
        input_ids.size() > static_cast<size_t>(INT_MAX) || offset < 0 ||
        proposed <= 0 || max_chunk <= 0 || proposed > max_chunk ||
        static_cast<size_t>(offset) >= input_ids.size()) {
        set_error(error, "invalid image-aware prefill chunk request");
        return -1;
    }
    const int remaining = static_cast<int>(input_ids.size()) - offset;
    const int ordinary = std::min(proposed, remaining);
    const auto sentinel = [vocab_size](int32_t id) {
        return id >= vocab_size &&
               id <= vocab_size + DEEPSEEK4_IMAGE_END;
    };

    const size_t chunk_begin = static_cast<size_t>(offset);
    size_t chunk_end = chunk_begin + static_cast<size_t>(ordinary);
    size_t cursor = chunk_begin;
    if (cursor > 0 && sentinel(input_ids[cursor - 1]) &&
        input_ids[cursor - 1] != vocab_size + DEEPSEEK4_IMAGE_END) {
        set_error(error, "prefill starts inside a learned image block");
        return -1;
    }
    while (cursor < chunk_end) {
        while (cursor < chunk_end && !sentinel(input_ids[cursor]))
            ++cursor;
        if (cursor == chunk_end) break;
        const size_t run_begin = cursor;
        bool saw_start = false;
        bool saw_end = false;
        while (cursor < input_ids.size()) {
            if (!sentinel(input_ids[cursor])) break;
            const int32_t type = input_ids[cursor] - vocab_size;
            if (type == DEEPSEEK4_IMAGE_START) {
                if (saw_start) {
                    set_error(error,
                              "learned image run contains duplicate starts");
                    return -1;
                }
                saw_start = true;
            } else if (!saw_start && type != DEEPSEEK4_IMAGE_PAD) {
                set_error(error,
                          "learned image run has content before its start");
                return -1;
            }
            ++cursor;
            if (type == DEEPSEEK4_IMAGE_END) {
                saw_end = true;
                break;
            }
        }
        if (!saw_start || !saw_end) {
            set_error(error, "learned image run is missing a boundary");
            return -1;
        }
        const size_t run_end = cursor;

        if (chunk_begin == run_begin && chunk_end < run_end) {
            const size_t run_size = run_end - run_begin;
            if (run_size > static_cast<size_t>(max_chunk)) {
                set_error(error,
                          "learned image block exceeds the prefill graph cap");
                return -1;
            }
            chunk_end = run_end;
        }
        if (chunk_begin < run_begin && chunk_end > run_begin &&
            chunk_end < run_end) {
            return static_cast<int>(run_begin - chunk_begin);
        }
    }
    return static_cast<int>(chunk_end - chunk_begin);
}

bool deepseek4_reference_chunk_accepts_image_tokens(
        int start_pos, const std::vector<int32_t> & ids,
        int32_t vocab_size) {
    if (start_pos == 0) return true;
    return std::all_of(ids.begin(), ids.end(), [vocab_size](int32_t id) {
        return id < vocab_size;
    });
}

bool deepseek4_is_vision_router_bias_suffix(const std::string & suffix) {
    return suffix == DEEPSEEK4_VISION_ROUTER_BIAS_SUFFIX;
}

bool deepseek4_optional_vision_bias_set_valid(int loaded, int layer_count) {
    return layer_count >= 0 &&
           (loaded == 0 || loaded == layer_count);
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
