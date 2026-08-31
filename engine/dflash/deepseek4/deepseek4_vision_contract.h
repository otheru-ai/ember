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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash {

inline constexpr char DEEPSEEK4_IMAGE_PLACEHOLDER_UTF8[] =
    "<\xEF\xBD\x9C" "deepseek_image" "\xEF\xBD\x9C>";

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

struct Deepseek4ImageRows {
    int n_llm_h = 0;
    int n_llm_w = 0;
    std::vector<float> embeddings;
};

struct Deepseek4PreparedRun {
    int prompt_offset = 0;
    Deepseek4PreparedImage image;
};

// Non-owning request view used at the language prefill seam. The owning
// GenerateRequest remains architecture-neutral; DeepSeek validates these
// fields once before any CPU embedding lookup or graph construction.
struct Deepseek4VisionRunView {
    int prompt_offset = 0;
    int n_tokens = 0;
    int embedding_width = 0;
    const int32_t * token_ids = nullptr;
    const float * embeddings = nullptr;
    size_t embedding_values = 0;
};

// Deliberately not implicitly convertible to std::vector<int32_t>. These IDs
// are valid only for CpuEmbedder::embed(); routing must receive the original
// sentinel-bearing prompt or every image row would silently classify as text.
struct Deepseek4EmbedOnlyTokenIds {
    std::vector<int32_t> values;
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

// Replace each exact tokenizer placeholder sequence in prompt order. The
// actual expanded offset feeds the learned block's alignment padding, so a
// second image cannot reuse the first image's pre-expansion position.
bool deepseek4_expand_image_placeholders(
    const std::vector<int32_t> & input_ids,
    const std::vector<int32_t> & placeholder_ids,
    int32_t vocab_size, int n_embd,
    const std::vector<Deepseek4ImageRows> & images,
    const Deepseek4ImageMarkers & markers,
    std::vector<int32_t> & output_ids,
    std::vector<Deepseek4PreparedRun> & runs,
    std::string * error = nullptr);

// Validate that every virtual image token is covered exactly once by an
// ordered request-owned run, and that every run exactly matches the expanded
// prompt. `embed_token_ids` substitutes token 0 for learned sentinels so the
// ordinary vocabulary embedder never indexes out of bounds; callers must then
// overwrite those rows with the validated embeddings before graph compute.
bool deepseek4_prepare_vision_prefill(
    const std::vector<int32_t> & input_ids, int32_t vocab_size, int n_embd,
    const std::vector<Deepseek4VisionRunView> & runs,
    Deepseek4EmbedOnlyTokenIds & embed_token_ids,
    std::string * error = nullptr);

// Install complete request-owned image runs into one prefill chunk. A chunk
// that intersects only part of a run is rejected: right-visible K/V and every
// replacement row must enter the same graph invocation.
bool deepseek4_splice_vision_chunk(
    const std::vector<Deepseek4VisionRunView> & runs, int n_embd,
    int chunk_offset, int chunk_tokens, float * embeddings,
    std::string * error = nullptr);

void deepseek4_image_visible(const std::vector<int32_t> & input_ids,
                             int32_t vocab_size, int max_image_tokens,
                             std::vector<int32_t> & left,
                             std::vector<int32_t> & right);

// Raw-window visibility for one query/key pair. The ordinary causal SWA
// window is widened only by the query's learned image-span counts. This is
// the scalar form of inference/model.py:get_window_topk_idxs_visible and is
// shared by GPU-free contract tests and the ggml attention-mask builder.
bool deepseek4_raw_attention_visible(int query_pos, int key_pos,
                                     int window_size, int visible_left,
                                     int visible_right);

// Validate the sentinel grammar visible to one graph invocation. Complete
// blocks may be surrounded by ordinary text, but no learned block may be
// partial, nested, empty, or interrupted by a vocabulary token.
bool deepseek4_validate_vision_chunk_ids(
    const std::vector<int32_t> & input_ids, int32_t vocab_size,
    std::string * error = nullptr);

// Layer-major graph-cache keys do not carry per-request row partitions or
// visibility counts. Only an all-vocabulary-token chunk may reuse a graph.
bool deepseek4_vision_graph_cache_safe(
    const std::vector<int32_t> & input_ids, int32_t vocab_size);

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

// Published-reference scheduler probe, not Ember's runtime predicate. The
// reference sends the whole prompt at start_pos==0, then asserts later calls
// contain no image IDs with the reason "image spans must be prefilled in a
// single chunk" (model.py:991-999). Ember deliberately generalizes that
// scheduler: a restored pre-image prefix may place a complete image block at a
// nonzero KV position, while the entire validated block still shares one graph
// invocation and receives explicit chunk-local visibility.
bool deepseek4_reference_chunk_accepts_image_tokens(
    int start_pos, const std::vector<int32_t> & ids, int32_t vocab_size);

// The converter's exact per-layer tensor suffix. Vision bias weights are an
// optional all-or-none set: zero keeps text-only checkpoints valid, while a
// partial set would silently route some image rows with the text contract.
inline constexpr char DEEPSEEK4_VISION_ROUTER_BIAS_SUFFIX[] =
    "exp_probs_b_vl.bias";
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
