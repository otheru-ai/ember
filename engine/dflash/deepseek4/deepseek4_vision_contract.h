// Pure language-side contract for DeepSeek-V4-Flash-Vision-Exp.
//
// The vision tower produces only the IMAGE embedding rows. The surrounding
// learned markers, their N-layout permutation, attention visibility and MoE
// routing mode are language-model semantics and must stay testable without a
// GPU or model weights. This file deliberately depends only on the C++ STL.
//
// Source of record (DeepSeek's published minimal inference at
// e46e16bf6035c6f317eb2ac7458eb0362926d402):
//   inference/image_processor.py:11-12,25-33,135-155,158-181
//   inference/model.py:283-305,589-639,975-999
//   inference/generate.py:68-82

#ifndef DFLASH_DEEPSEEK4_VISION_CONTRACT_H
#define DFLASH_DEEPSEEK4_VISION_CONTRACT_H

#include <cstdint>
#include <string>
#include <vector>

namespace dflash {

enum Deepseek4ImageType : int32_t {
    DEEPSEEK4_IMAGE_START = 0,
    DEEPSEEK4_IMAGE_PAD = 1,
    DEEPSEEK4_IMAGE = 2,
    DEEPSEEK4_IMAGE_NEWLINE = 3,
    DEEPSEEK4_IMAGE_END = 4,
};

struct Deepseek4ImageBlock {
    std::vector<int32_t> token_ids;
    // Row-major aligner indices in the order consumed by IMAGE token slots.
    std::vector<int32_t> image_perm;
};

// The tower/aligner produces IMAGE rows only. The four other image token
// types have learned vectors in the vision projector artifact. Keeping them
// explicit here makes the offline sidecar and the eventual native tower feed
// exactly the same language-side assembly path.
struct Deepseek4ImageMarkers {
    std::vector<float> start;
    std::vector<float> pad;
    std::vector<float> newline;
    std::vector<float> end;
};

struct Deepseek4PreparedImage {
    Deepseek4ImageBlock block;
    // block.token_ids.size() rows, n_embd floats per row.
    std::vector<float> embeddings;
};

bool deepseek4_build_image_block(int32_t vocab_size, int n_llm_h, int n_llm_w,
                                 int start_pos, Deepseek4ImageBlock & out,
                                 std::string * error = nullptr);

bool deepseek4_validate_image_block(const std::vector<int32_t> & token_ids,
                                    int32_t vocab_size, int n_llm_h,
                                    int n_llm_w, int start_pos,
                                    std::string * error = nullptr);

// Assemble row-major aligner IMAGE rows and learned marker rows into the
// official N-layout at the actual prompt position. image_embeddings contains
// n_llm_h*n_llm_w rows and never carries learned marker vectors itself.
bool deepseek4_prepare_image(int32_t vocab_size, int n_llm_h, int n_llm_w,
                             int start_pos, int n_embd,
                             const std::vector<float> & image_embeddings,
                             const Deepseek4ImageMarkers & markers,
                             Deepseek4PreparedImage & out,
                             std::string * error = nullptr);

void deepseek4_image_visible(const std::vector<int32_t> & input_ids,
                             int32_t vocab_size, int max_image_tokens,
                             std::vector<int32_t> & left,
                             std::vector<int32_t> & right);

// A cut is a prefix length. It is unsafe only when it would divide a learned
// [IMAGE_START, IMAGE_END] block.
bool deepseek4_prefill_cut_safe(const std::vector<int32_t> & input_ids,
                                int32_t vocab_size, int cut);

// Adjust one proposed prefill chunk so each learned image block (leading PAD
// through END) stays in one graph invocation. Returns a positive row count, or
// -1 when the caller starts inside a block or the complete block exceeds max_chunk.
// Text-only inputs return the ordinary min(proposed, remaining) count.
int deepseek4_image_aware_prefill_chunk(
    const std::vector<int32_t> & input_ids, int32_t vocab_size,
    int offset, int proposed, int max_chunk, std::string * error = nullptr);

// The published reference merges image embeddings only at start_pos == 0 and
// asserts that every later chunk contains ordinary vocabulary IDs.
bool deepseek4_chunk_accepts_image_tokens(int start_pos,
                                          const std::vector<int32_t> & ids,
                                          int32_t vocab_size);

// The converter's exact per-layer tensor suffix. Vision bias weights are an
// optional all-or-none set: zero keeps text-only checkpoints valid, while a
// partial set would silently route some image rows with the text contract.
bool deepseek4_is_vision_router_bias_suffix(const std::string & suffix);
bool deepseek4_optional_vision_bias_set_valid(int loaded, int layer_count);

enum class Deepseek4RouteMode {
    HASH_TEXT,
    SCORE_TEXT,
    SCORE_VISION,
    INVALID_IMAGE,
};

Deepseek4RouteMode deepseek4_route_mode(bool hash_layer,
                                        bool vision_weights_loaded,
                                        int32_t token_id,
                                        int32_t vocab_size);

}  // namespace dflash

#endif  // DFLASH_DEEPSEEK4_VISION_CONTRACT_H
