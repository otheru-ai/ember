// Model-selected byte pre-tokenization for the dflash BPE tokenizer.
//
// A BPE merge never crosses a pre-tokenized piece boundary.  Selecting the
// wrong splitter therefore changes token IDs even when encode->decode still
// reproduces the same text bytes.  DeepSeek-V4-Flash declares
// tokenizer.ggml.pre="joyai-llm"; treating that value as Qwen silently turns
// common source-code numbers such as 0038 and 379 into different prompt token
// streams and changes the model's logits.

#pragma once

#include <string>
#include <vector>

namespace dflash::common {

enum class PreTokenizer {
    QWEN2,
    QWEN35,
    JOYAI_LLM,
};

// Exact GGUF metadata mapping.  Unknown non-empty names are rejected instead
// of silently taking a different tokenizer; byte-exact failure is safer than a
// plausible-looking but incorrectly conditioned generation.
bool pre_tokenizer_from_name(const char * name, PreTokenizer & out);
const char * pre_tokenizer_name(PreTokenizer type);

std::vector<std::string> pre_tokenize_text(const std::string & text,
                                           PreTokenizer type);

}  // namespace dflash::common
