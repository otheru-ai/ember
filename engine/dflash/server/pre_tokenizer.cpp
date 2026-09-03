// Model-selected BPE pre-tokenization. This translation unit has no GPU/ggml
// dependency so model-specific split rules stay in the host test gauntlet.

#include "pre_tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace dflash::common {
namespace {

uint32_t utf8_decode(const char * text, size_t remaining, int * length) {
    if (remaining == 0) {
        *length = 0;
        return 0xFFFD;
    }
    const uint8_t c = static_cast<uint8_t>(text[0]);
    if (c < 0x80) {
        *length = 1;
        return c;
    }
    if ((c & 0xE0) == 0xC0 && remaining >= 2) {
        *length = 2;
        return (static_cast<uint32_t>(c & 0x1F) << 6) |
               (static_cast<uint32_t>(static_cast<uint8_t>(text[1])) & 0x3F);
    }
    if ((c & 0xF0) == 0xE0 && remaining >= 3) {
        *length = 3;
        return (static_cast<uint32_t>(c & 0x0F) << 12) |
               ((static_cast<uint32_t>(static_cast<uint8_t>(text[1])) & 0x3F) << 6) |
               (static_cast<uint32_t>(static_cast<uint8_t>(text[2])) & 0x3F);
    }
    if ((c & 0xF8) == 0xF0 && remaining >= 4) {
        *length = 4;
        return (static_cast<uint32_t>(c & 0x07) << 18) |
               ((static_cast<uint32_t>(static_cast<uint8_t>(text[1])) & 0x3F) << 12) |
               ((static_cast<uint32_t>(static_cast<uint8_t>(text[2])) & 0x3F) << 6) |
               (static_cast<uint32_t>(static_cast<uint8_t>(text[3])) & 0x3F);
    }
    *length = 1;
    return 0xFFFD;
}

size_t next_utf8(const std::string & text, size_t position) {
    int length = 0;
    (void)utf8_decode(text.data() + position, text.size() - position, &length);
    if (length <= 0 || position + static_cast<size_t>(length) > text.size())
        length = 1;
    return position + static_cast<size_t>(length);
}
bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool ascii_digit(uint8_t c) { return c >= '0' && c <= '9'; }

bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

bool ascii_newline(uint8_t c) { return c == '\n' || c == '\r'; }

bool ascii_punctuation(uint8_t c) {
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}

bool cjk(uint32_t cp) {
    // Exact DS4/JoyAI ranges (ds4.c:35671-35730).
    return (cp >= 0x4E00 && cp <= 0x9FA5) ||
           (cp >= 0x3040 && cp <= 0x309F) ||
           (cp >= 0x30A0 && cp <= 0x30FF);
}

bool cjk_at(const std::string & text, size_t position) {
    if (static_cast<uint8_t>(text[position]) < 0x80) return false;
    int length = 0;
    const uint32_t cp = utf8_decode(
        text.data() + position, text.size() - position, &length);
    return length > 0 && cjk(cp);
}

bool letter_like_at(const std::string & text, size_t position) {
    const uint8_t c = static_cast<uint8_t>(text[position]);
    // DS4 treats ordinary non-ASCII codepoints as letter-like after the
    // CJK/kana rule has first refusal.
    return c < 0x80 ? ascii_alpha(c) : true;
}

size_t consume_letters(const std::string & text, size_t position) {
    while (position < text.size() && letter_like_at(text, position))
        position = next_utf8(text, position);
    return position;
}

// Port of ds4.c:35971-36057. Piece boundaries are part of the model contract
// because BPE merges do not cross them.
std::vector<std::string> pre_tokenize_joyai(const std::string & text) {
    std::vector<std::string> pieces;
    size_t position = 0;
    while (position < text.size()) {
        const size_t start = position;
        const uint8_t c = static_cast<uint8_t>(text[position]);

        if (ascii_digit(c)) {
            int digits = 0;
            while (position < text.size() &&
                   ascii_digit(static_cast<uint8_t>(text[position])) &&
                   digits < 3) {
                ++position;
                ++digits;
            }
        } else if (cjk_at(text, position)) {
            do {
                position = next_utf8(text, position);
            } while (position < text.size() && cjk_at(text, position));
        } else if (ascii_punctuation(c) && position + 1 < text.size() &&
                   ascii_alpha(static_cast<uint8_t>(text[position + 1]))) {
            ++position;
            while (position < text.size() &&
                   ascii_alpha(static_cast<uint8_t>(text[position])))
                ++position;
        } else if (letter_like_at(text, position)) {
            position = consume_letters(text, position);
        } else if (!ascii_newline(c) && !ascii_punctuation(c) &&
                   position + 1 < text.size() &&
                   letter_like_at(text, position + 1)) {
            ++position;
            position = consume_letters(text, position);
        } else if (c == ' ' && position + 1 < text.size() &&
                   ascii_punctuation(static_cast<uint8_t>(text[position + 1]))) {
            ++position;
            while (position < text.size() &&
                   ascii_punctuation(static_cast<uint8_t>(text[position])))
                ++position;
            while (position < text.size() &&
                   ascii_newline(static_cast<uint8_t>(text[position])))
                ++position;
        } else if (ascii_punctuation(c)) {
            while (position < text.size() &&
                   ascii_punctuation(static_cast<uint8_t>(text[position])))
                ++position;
            while (position < text.size() &&
                   ascii_newline(static_cast<uint8_t>(text[position])))
                ++position;
        } else if (ascii_space(c)) {
            size_t scan = position;
            size_t last_newline_end = 0;
            while (scan < text.size() &&
                   ascii_space(static_cast<uint8_t>(text[scan]))) {
                const uint8_t space = static_cast<uint8_t>(text[scan++]);
                if (ascii_newline(space)) last_newline_end = scan;
            }
            if (last_newline_end != 0) {
                position = last_newline_end;
            } else if (scan < text.size() && scan > position + 1 &&
                       (letter_like_at(text, scan) ||
                        ascii_punctuation(static_cast<uint8_t>(text[scan])))) {
                position = scan - 1;
            } else {
                position = scan;
            }
        } else {
            position = next_utf8(text, position);
        }

        if (position == start) position = next_utf8(text, position);
        pieces.push_back(text.substr(start, position - start));
    }
    return pieces;
}

} // namespace

bool pre_tokenizer_from_name(const char * name, PreTokenizer & out) {
    if (!name) return false;
    if (std::strcmp(name, "joyai-llm") == 0) {
        out = PreTokenizer::JOYAI_LLM;
        return true;
    }
    return false;
}

const char * pre_tokenizer_name(PreTokenizer type) {
    switch (type) {
        case PreTokenizer::JOYAI_LLM: return "joyai-llm";
    }
    return "unknown";
}

bool pre_tokenizer_supported(const char * name) {
    PreTokenizer ignored = PreTokenizer::JOYAI_LLM;
    return pre_tokenizer_from_name(name, ignored);
}

std::vector<std::string> pre_tokenize_text(const std::string & text,
                                           PreTokenizer type) {
    switch (type) {
        case PreTokenizer::JOYAI_LLM:
            return pre_tokenize_joyai(text);
    }
    return {};
}

} // namespace dflash::common
