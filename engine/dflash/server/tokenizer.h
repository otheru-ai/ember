// BPE tokenizer for dflash::common native server.
//
// Loads vocabulary (token strings) and merge rules from GGUF metadata,
// then provides encode (text → token IDs) and decode (token IDs → text).
//
// Pre-tokenization is selected from tokenizer.ggml.pre. DeepSeek-V4-Flash's
// joyai-llm splitter is model-coupled: substituting another splitter changes
// BPE boundaries and silently conditions the model on the wrong prompt token
// stream.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "pre_tokenizer.h"

namespace dflash::common {

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    Tokenizer(const Tokenizer &) = delete;
    Tokenizer & operator=(const Tokenizer &) = delete;
    Tokenizer(Tokenizer &&) noexcept = default;
    Tokenizer & operator=(Tokenizer &&) noexcept = default;

    // Load vocabulary and merges from a GGUF file's metadata.
    // Returns false on failure (prints diagnostic to stderr).
    bool load_from_gguf(const char * model_path);

    // ─── Encode ──────────────────────────────────────────────────────
    // Tokenize a UTF-8 string into token IDs.
    std::vector<int32_t> encode(const std::string & text) const;

    // ─── Decode ──────────────────────────────────────────────────────
    // Convert a single token ID to its text representation.
    // Returns empty string for out-of-range IDs.
    std::string token_text(int32_t id) const;

    // Return the raw BPE-encoded token string (as stored in the GGUF vocab).
    // Useful for checking special tokens without GPT-2 decode overhead.
    const std::string & raw_token(int32_t id) const;

    // Convert a sequence of token IDs to text.
    std::string decode(const std::vector<int32_t> & ids) const;

    // ─── Special tokens ──────────────────────────────────────────────
    int32_t eos_id() const { return eos_id_; }
    int32_t eos_chat_id() const { return eos_chat_id_; }
    int32_t bos_id() const { return bos_id_; }
    int32_t vocab_size() const { return (int32_t)id_to_token_.size(); }

    // Raw GGUF token strings, unprocessed. token_text() decodes byte-fallback
    // and special forms for display; constrained decoding needs the stored
    // form instead, because that is what a BYTE_FALLBACK vocabulary describes.
    const std::vector<std::string> & raw_vocab() const { return id_to_token_; }

    // Look up a token by its exact string. Returns -1 if not found.
    int32_t token_to_id(const std::string & token) const;

private:
    bool load_from_gguf_in_place(const char * model_path);

    // Pre-tokenize text into model-selected pieces.
    std::vector<std::string> pre_tokenize(const std::string & text) const;

    // Apply BPE merges to a single pre-tokenized piece.
    std::vector<int32_t> bpe_encode_piece(const std::string & piece) const;

    // Vocabulary: id → token string
    std::vector<std::string> id_to_token_;

    // Reverse map: token string → id
    std::unordered_map<std::string, int32_t> token_to_id_;

    // BPE merge ranks: "A B" → rank (lower = higher priority)
    std::unordered_map<std::string, int> merge_rank_;

    // Added special tokens (e.g. <think>, </think>) — sorted longest-first
    // for greedy matching during encode.
    std::vector<std::pair<std::string, int32_t>> added_tokens_;

    // First-byte index into added_tokens_, preserving that longest-first order.
    // encode() previously probed every added token (1283 on the production
    // vocab) with a text.find() from the current offset for every segment,
    // making the special-token split O(markers * added_tokens * text_len) — a
    // 250 KB history with 100 markers cost ~1.9 s on the worker thread, and it
    // is NOT amortized by KV prefix reuse. Bucketing by first byte turns the
    // scan linear: a token can only match at `pos` if its first byte equals
    // text[pos]. Order within a bucket is inherited from the longest-first sort,
    // so greedy matching stays byte-identical.
    std::array<std::vector<uint32_t>, 256> added_by_first_byte_;
    // Match the longest added token at `pos`; returns its index in
    // added_tokens_, or -1 when none matches.
    int match_added_token_at(const std::string & text, size_t pos) const;

    // Special token IDs
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t eos_chat_id_ = -1;  // <|im_end|> for ChatML models

    // Selected from tokenizer.ggml.pre at load time. Keeping this as runtime
    // state prevents another architecture from silently taking DeepSeek's
    // JoyAI splitter.
    PreTokenizer pre_type_ = PreTokenizer::JOYAI_LLM;

    // Decode mode: SentencePiece tokens use UTF-8 with ▁ for space;
    // GPT-2/BPE tokens use byte-level Unicode encoding.
    bool is_sentencepiece_ = false;
};

}  // namespace dflash::common
