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

bool deepseek4_build_image_block(int32_t vocab_size, int n_llm_h, int n_llm_w,
                                 int start_pos, Deepseek4ImageBlock & out,
                                 std::string * error = nullptr);

bool deepseek4_validate_image_block(const std::vector<int32_t> & token_ids,
                                    int32_t vocab_size, int n_llm_h,
                                    int n_llm_w, int start_pos,
                                    std::string * error = nullptr);

void deepseek4_image_visible(const std::vector<int32_t> & input_ids,
                             int32_t vocab_size, int max_image_tokens,
                             std::vector<int32_t> & left,
                             std::vector<int32_t> & right);

// A cut is a prefix length. It is unsafe only when it would divide a learned
// [IMAGE_START, IMAGE_END] block.
bool deepseek4_prefill_cut_safe(const std::vector<int32_t> & input_ids,
                                int32_t vocab_size, int cut);

// The published reference merges image embeddings only at start_pos == 0 and
// asserts that every later chunk contains ordinary vocabulary IDs.
bool deepseek4_chunk_accepts_image_tokens(int start_pos,
                                          const std::vector<int32_t> & ids,
                                          int32_t vocab_size);

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
