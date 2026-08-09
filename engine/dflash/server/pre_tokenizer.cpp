// Model-selected BPE pre-tokenizers.  This file deliberately has no ggml/GPU
// dependency so every split rule can be covered by the host test gauntlet.

#include "pre_tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace dflash::common {

namespace {

static uint32_t utf8_decode(const char * text, size_t remaining, int * length) {
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

static size_t next_utf8(const std::string & text, size_t position) {
    int length = 0;
    (void)utf8_decode(text.data() + position, text.size() - position, &length);
    if (length <= 0 || position + static_cast<size_t>(length) > text.size())
        length = 1;
    return position + static_cast<size_t>(length);
}

static bool is_letter(uint32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp >= 0xC0 && cp <= 0xFF && cp != 0xD7 && cp != 0xF7) return true;
    if (cp >= 0x100 && cp <= 0x24F) return true;
    if (cp >= 0x370 && cp <= 0x3FF) return true;
    if (cp >= 0x400 && cp <= 0x4FF) return true;
    if (cp >= 0x600 && cp <= 0x6FF) return true;
    if (cp >= 0x900 && cp <= 0x97F) return true;
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    if (cp >= 0x3040 && cp <= 0x309F) return true;
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    if (cp >= 0xFF21 && cp <= 0xFF3A) return true;
    if (cp >= 0xFF41 && cp <= 0xFF5A) return true;
    if (cp >= 0x0E01 && cp <= 0x0E3A) return true;
    if (cp >= 0x05D0 && cp <= 0x05EA) return true;
    if (cp >= 0x1100 && cp <= 0x11FF) return true;
    if (cp >= 0x2E80 && cp <= 0x2EFF) return true;
    return false;
}

static bool is_digit(uint32_t cp) {
    return (cp >= '0' && cp <= '9') ||
           (cp >= 0xFF10 && cp <= 0xFF19) ||
           (cp >= 0x0660 && cp <= 0x0669) ||
           (cp >= 0x06F0 && cp <= 0x06F9) ||
           (cp >= 0x0966 && cp <= 0x096F) ||
           (cp >= 0x09E6 && cp <= 0x09EF) ||
           (cp >= 0x0E50 && cp <= 0x0E59);
}

static bool is_mark(uint32_t cp) {
    return (cp >= 0x0300 && cp <= 0x036F) ||
           (cp >= 0x0591 && cp <= 0x05BD) ||
           (cp >= 0x0610 && cp <= 0x061A) ||
           (cp >= 0x064B && cp <= 0x065F) ||
           (cp >= 0x0900 && cp <= 0x0903) ||
           (cp >= 0x093A && cp <= 0x094F) || cp == 0x0E31 ||
           (cp >= 0x0E34 && cp <= 0x0E3A) ||
           (cp >= 0xFE20 && cp <= 0xFE2F) ||
           (cp >= 0x20D0 && cp <= 0x20FF);
}

static bool is_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == '\v' || cp == 0x00A0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

static bool is_newline(uint32_t cp) {
    return cp == '\n' || cp == '\r';
}

// Qwen2 and Qwen3.5 both use the single-number pattern for the model families
// currently supported by this vendored engine.  Qwen3.5 additionally relies on
// the mark handling below.  Keeping the selected type explicit prevents the
// metadata value from becoming dead state again.
static std::vector<std::string> pre_tokenize_qwen(const std::string & text) {
    std::vector<std::string> pieces;
    const size_t length = text.size();
    size_t position = 0;

    auto peek = [&](size_t at, int * cp_length) -> uint32_t {
        if (at >= length) {
            *cp_length = 0;
            return 0;
        }
        return utf8_decode(text.data() + at, length - at, cp_length);
    };

    while (position < length) {
        const size_t start = position;
        int cp_length = 0;
        const uint32_t cp = peek(position, &cp_length);

        if (cp == '\'') {
            const size_t saved = position;
            ++position;
            bool matched = false;
            if (position < length) {
                const char c = static_cast<char>(text[position] | 0x20);
                if (c == 's' || c == 't' || c == 'm' || c == 'd') {
                    ++position;
                    matched = true;
                } else if ((c == 'r' || c == 'v' || c == 'l') &&
                           position + 1 < length) {
                    const char next =
                        static_cast<char>(text[position + 1] | 0x20);
                    if ((c == 'r' && next == 'e') ||
                        (c == 'v' && next == 'e') ||
                        (c == 'l' && next == 'l')) {
                        position += 2;
                        matched = true;
                    }
                }
            }
            if (matched) {
                pieces.push_back(text.substr(start, position - start));
                continue;
            }
            position = saved;
        }

        {
            size_t scan = position;
            int scan_length = 0;
            uint32_t scan_cp = peek(scan, &scan_length);
            if (scan_length > 0 && !is_newline(scan_cp) &&
                !is_letter(scan_cp) && !is_digit(scan_cp)) {
                scan += static_cast<size_t>(scan_length);
                scan_cp = peek(scan, &scan_length);
            }
            if (scan_length > 0 && (is_letter(scan_cp) || is_mark(scan_cp))) {
                while (scan_length > 0 &&
                       (is_letter(scan_cp) || is_mark(scan_cp))) {
                    scan += static_cast<size_t>(scan_length);
                    scan_cp = peek(scan, &scan_length);
                }
                pieces.push_back(text.substr(position, scan - position));
                position = scan;
                continue;
            }
        }

        if (is_digit(cp)) {
            position += static_cast<size_t>(cp_length);
            pieces.push_back(text.substr(start, position - start));
            continue;
        }

        {
            size_t scan = position;
            int scan_length = 0;
            uint32_t scan_cp = peek(scan, &scan_length);
            if (scan_cp == ' ') {
                scan += static_cast<size_t>(scan_length);
                scan_cp = peek(scan, &scan_length);
            }
            const size_t punctuation_start = scan;
            while (scan_length > 0 && !is_whitespace(scan_cp) &&
                   !is_letter(scan_cp) && !is_mark(scan_cp) &&
                   !is_digit(scan_cp)) {
                scan += static_cast<size_t>(scan_length);
                scan_cp = peek(scan, &scan_length);
            }
            if (scan > punctuation_start) {
                while (scan_length > 0 && is_newline(scan_cp)) {
                    scan += static_cast<size_t>(scan_length);
                    scan_cp = peek(scan, &scan_length);
                }
                pieces.push_back(text.substr(position, scan - position));
                position = scan;
                continue;
            }
        }

        if (is_whitespace(cp)) {
            size_t scan = position;
            int scan_length = 0;
            uint32_t scan_cp = peek(scan, &scan_length);
            while (scan_length > 0 && is_whitespace(scan_cp) &&
                   !is_newline(scan_cp)) {
                scan += static_cast<size_t>(scan_length);
                scan_cp = peek(scan, &scan_length);
            }
            if (scan_length > 0 && is_newline(scan_cp)) {
                while (scan_length > 0 && is_newline(scan_cp)) {
                    scan += static_cast<size_t>(scan_length);
                    scan_cp = peek(scan, &scan_length);
                }
                pieces.push_back(text.substr(position, scan - position));
                position = scan;
                continue;
            }

            scan = position;
            scan_cp = peek(scan, &scan_length);
            size_t before_last = position;
            while (scan_length > 0 && is_whitespace(scan_cp)) {
                before_last = scan;
                scan += static_cast<size_t>(scan_length);
                scan_cp = peek(scan, &scan_length);
            }
            const bool followed_by_non_ws =
                scan < length && !is_whitespace(scan_cp);
            if (!followed_by_non_ws) {
                pieces.push_back(text.substr(position, scan - position));
                position = scan;
            } else if (before_last > position) {
                pieces.push_back(text.substr(position, before_last - position));
                position = before_last;
            } else {
                pieces.push_back(text.substr(position, scan - position));
                position = scan;
            }
            continue;
        }

        position += cp_length > 0 ? static_cast<size_t>(cp_length) : 1;
        pieces.push_back(text.substr(start, position - start));
    }
    return pieces;
}

static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

static bool joyai_ascii_punctuation(uint8_t c) {
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}

static bool joyai_cjk(uint32_t cp) {
    // Exact DS4/JoyAI ranges (ds4.c:35671-35730).  In particular the CJK
    // endpoint is 0x9fa5, not the broader Qwen Unicode letter range.
    return (cp >= 0x4E00 && cp <= 0x9FA5) ||
           (cp >= 0x3040 && cp <= 0x309F) ||
           (cp >= 0x30A0 && cp <= 0x30FF);
}

static bool joyai_cjk_at(const std::string & text, size_t position) {
    if (static_cast<uint8_t>(text[position]) < 0x80) return false;
    int length = 0;
    const uint32_t cp = utf8_decode(
        text.data() + position, text.size() - position, &length);
    return length > 0 && joyai_cjk(cp);
}

static bool joyai_letter_like_at(const std::string & text, size_t position) {
    const uint8_t c = static_cast<uint8_t>(text[position]);
    if (c < 0x80) return ascii_alpha(c);
    // DS4's JoyAI implementation treats ordinary non-ASCII UTF-8 codepoints as
    // letter-like after the CJK/kana rule has had first refusal.  Preserve that
    // measured reference behavior rather than mixing Qwen Unicode classes in.
    return true;
}

static size_t joyai_consume_letters(const std::string & text,
                                    size_t position) {
    while (position < text.size() &&
           joyai_letter_like_at(text, position)) {
        position = next_utf8(text, position);
    }
    return position;
}

// Port of ds4.c:35971-36057.  Piece boundaries, including the newline attached
// to a punctuation run, are part of the model contract because BPE merges do
// not cross them.
static std::vector<std::string> pre_tokenize_joyai(const std::string & text) {
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
        } else if (joyai_cjk_at(text, position)) {
            do {
                position = next_utf8(text, position);
            } while (position < text.size() && joyai_cjk_at(text, position));
        } else if (joyai_ascii_punctuation(c) &&
                   position + 1 < text.size() &&
                   ascii_alpha(static_cast<uint8_t>(text[position + 1]))) {
            ++position;
            while (position < text.size() &&
                   ascii_alpha(static_cast<uint8_t>(text[position])))
                ++position;
        } else if (joyai_letter_like_at(text, position)) {
            position = joyai_consume_letters(text, position);
        } else if (!ascii_newline(c) && !joyai_ascii_punctuation(c) &&
                   position + 1 < text.size() &&
                   joyai_letter_like_at(text, position + 1)) {
            ++position;
            position = joyai_consume_letters(text, position);
        } else if (c == ' ' && position + 1 < text.size() &&
                   joyai_ascii_punctuation(
                       static_cast<uint8_t>(text[position + 1]))) {
            ++position;
            while (position < text.size() &&
                   joyai_ascii_punctuation(
                       static_cast<uint8_t>(text[position])))
                ++position;
            while (position < text.size() &&
                   ascii_newline(static_cast<uint8_t>(text[position])))
                ++position;
        } else if (joyai_ascii_punctuation(c)) {
            while (position < text.size() &&
                   joyai_ascii_punctuation(
                       static_cast<uint8_t>(text[position])))
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
                       (joyai_letter_like_at(text, scan) ||
                        joyai_ascii_punctuation(
                            static_cast<uint8_t>(text[scan])))) {
                // JoyAI leaves one leading space for the following word/run:
                // "    int" -> "   ", " int".
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

}  // namespace

bool pre_tokenizer_from_name(const char * name, PreTokenizer & out) {
    if (!name) return false;
    if (std::strcmp(name, "qwen2") == 0) {
        out = PreTokenizer::QWEN2;
        return true;
    }
    if (std::strcmp(name, "qwen35") == 0) {
        out = PreTokenizer::QWEN35;
        return true;
    }
    if (std::strcmp(name, "joyai-llm") == 0) {
        out = PreTokenizer::JOYAI_LLM;
        return true;
    }
    return false;
}

const char * pre_tokenizer_name(PreTokenizer type) {
    switch (type) {
        case PreTokenizer::QWEN2: return "qwen2";
        case PreTokenizer::QWEN35: return "qwen35";
        case PreTokenizer::JOYAI_LLM: return "joyai-llm";
    }
    return "unknown";
}

std::vector<std::string> pre_tokenize_text(const std::string & text,
                                           PreTokenizer type) {
    switch (type) {
        case PreTokenizer::JOYAI_LLM:
            return pre_tokenize_joyai(text);
        case PreTokenizer::QWEN2:
        case PreTokenizer::QWEN35:
            return pre_tokenize_qwen(text);
    }
    return {};
}

}  // namespace dflash::common
