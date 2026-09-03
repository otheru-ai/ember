// Model-selected byte pre-tokenization for dflash BPE tokenizers.
//
// A BPE merge never crosses a pre-tokenized piece boundary.  Selecting the
// wrong splitter therefore changes token IDs even when encode->decode still
// reproduces the same text bytes. DeepSeek-V4-Flash uses `joyai-llm`.

#pragma once

#include <string>
#include <vector>

namespace dflash::common {

enum class PreTokenizer {
    JOYAI_LLM,
};

// Exact GGUF metadata mapping. Unknown names are rejected rather than silently
// changing prompt token IDs.
bool pre_tokenizer_from_name(const char * name, PreTokenizer & out);
const char * pre_tokenizer_name(PreTokenizer type);
bool pre_tokenizer_supported(const char * name);

std::vector<std::string> pre_tokenize_text(const std::string & text,
                                           PreTokenizer type);

// Compatibility overload retained for DeepSeek-owned call sites/tests.
inline std::vector<std::string> pre_tokenize_text(const std::string & text) {
    return pre_tokenize_text(text, PreTokenizer::JOYAI_LLM);
}

}  // namespace dflash::common
