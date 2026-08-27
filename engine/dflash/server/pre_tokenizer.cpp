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

struct QwenRange { uint32_t first; uint32_t last; };
struct QwenCombining { uint32_t cp; uint8_t value; };
struct QwenDecomposition { uint32_t cp; uint16_t offset; uint8_t length; };
struct QwenComposition { uint64_t key; uint32_t cp; };

#include "qwen_unicode_tables.inc"

template <size_t N>
bool qwen_in_ranges(uint32_t cp, const QwenRange (&ranges)[N]) {
    const auto * it = std::lower_bound(
        ranges, ranges + N, cp,
        [](const QwenRange & range, uint32_t value) {
            return range.last < value;
        });
    return it != ranges + N && it->first <= cp;
}

bool qwen_letter(uint32_t cp) {
    return qwen_in_ranges(cp, kQwenLetterRanges);
}

bool qwen_number(uint32_t cp) {
    return qwen_in_ranges(cp, kQwenNumberRanges);
}

bool qwen_mark(uint32_t cp) {
    return qwen_in_ranges(cp, kQwenMarkRanges);
}

uint8_t qwen_combining_class(uint32_t cp) {
    const auto * first = std::begin(kQwenCombining);
    const auto * last = std::end(kQwenCombining);
    const auto * it = std::lower_bound(
        first, last, cp,
        [](const QwenCombining & item, uint32_t value) {
            return item.cp < value;
        });
    return it != last && it->cp == cp ? it->value : 0;
}

const QwenDecomposition * qwen_decomposition(uint32_t cp) {
    const auto * first = std::begin(kQwenDecompositions);
    const auto * last = std::end(kQwenDecompositions);
    const auto * it = std::lower_bound(
        first, last, cp,
        [](const QwenDecomposition & item, uint32_t value) {
            return item.cp < value;
        });
    return it != last && it->cp == cp ? it : nullptr;
}

uint32_t qwen_composition(uint32_t first_cp, uint32_t second_cp) {
    constexpr uint32_t s_base = 0xAC00;
    constexpr uint32_t l_base = 0x1100;
    constexpr uint32_t v_base = 0x1161;
    constexpr uint32_t t_base = 0x11A7;
    constexpr uint32_t l_count = 19;
    constexpr uint32_t v_count = 21;
    constexpr uint32_t t_count = 28;
    constexpr uint32_t n_count = v_count * t_count;
    constexpr uint32_t s_count = l_count * n_count;
    if (first_cp >= l_base && first_cp < l_base + l_count &&
        second_cp >= v_base && second_cp < v_base + v_count) {
        return s_base + (first_cp - l_base) * n_count +
               (second_cp - v_base) * t_count;
    }
    if (first_cp >= s_base && first_cp < s_base + s_count &&
        (first_cp - s_base) % t_count == 0 &&
        second_cp > t_base && second_cp < t_base + t_count) {
        return first_cp + second_cp - t_base;
    }

    const uint64_t key = (uint64_t(first_cp) << 21) | second_cp;
    const auto * first = std::begin(kQwenCompositions);
    const auto * last = std::end(kQwenCompositions);
    const auto * it = std::lower_bound(
        first, last, key,
        [](const QwenComposition & item, uint64_t value) {
            return item.key < value;
        });
    return it != last && it->key == key ? it->cp : 0;
}

void qwen_decompose(uint32_t cp, std::vector<uint32_t> & out) {
    constexpr uint32_t s_base = 0xAC00;
    constexpr uint32_t l_base = 0x1100;
    constexpr uint32_t v_base = 0x1161;
    constexpr uint32_t t_base = 0x11A7;
    constexpr uint32_t v_count = 21;
    constexpr uint32_t t_count = 28;
    constexpr uint32_t n_count = v_count * t_count;
    constexpr uint32_t s_count = 19 * n_count;
    if (cp >= s_base && cp < s_base + s_count) {
        const uint32_t index = cp - s_base;
        out.push_back(l_base + index / n_count);
        out.push_back(v_base + (index % n_count) / t_count);
        const uint32_t trailing = index % t_count;
        if (trailing != 0) out.push_back(t_base + trailing);
        return;
    }
    const QwenDecomposition * decomposition = qwen_decomposition(cp);
    if (!decomposition) {
        out.push_back(cp);
        return;
    }
    for (uint8_t i = 0; i < decomposition->length; ++i) {
        qwen_decompose(kQwenDecompositionValues[decomposition->offset + i], out);
    }
}

void qwen_append_utf8(std::string & out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string qwen_normalize_nfc(const std::string & text) {
    std::vector<uint32_t> decomposed;
    for (size_t position = 0; position < text.size();) {
        int length = 0;
        const uint32_t cp = utf8_decode(text.data() + position,
                                        text.size() - position, &length);
        qwen_decompose(cp, decomposed);
        position += length > 0 ? static_cast<size_t>(length) : 1;
    }

    // Canonical ordering is stable within each starter segment.
    for (size_t i = 1; i < decomposed.size(); ++i) {
        const uint8_t current_class = qwen_combining_class(decomposed[i]);
        if (current_class == 0) continue;
        size_t j = i;
        while (j > 0) {
            const uint8_t previous_class = qwen_combining_class(decomposed[j - 1]);
            if (previous_class == 0 || previous_class <= current_class) break;
            std::swap(decomposed[j], decomposed[j - 1]);
            --j;
        }
    }

    std::vector<uint32_t> composed;
    composed.reserve(decomposed.size());
    size_t starter = 0;
    uint8_t last_class = 0;
    for (uint32_t cp : decomposed) {
        const uint8_t current_class = qwen_combining_class(cp);
        const uint32_t merged = composed.empty()
            ? 0
            : qwen_composition(composed[starter], cp);
        if (merged != 0 && (last_class < current_class || last_class == 0)) {
            composed[starter] = merged;
            continue;
        }
        if (current_class == 0) starter = composed.size();
        composed.push_back(cp);
        last_class = current_class;
    }

    std::string normalized;
    normalized.reserve(text.size());
    for (uint32_t cp : composed) qwen_append_utf8(normalized, cp);
    return normalized;
}

bool qwen_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == '\v' || cp == 0x00A0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

bool qwen_newline(uint32_t cp) { return cp == '\n' || cp == '\r'; }

// Direct scanner for the official Qwen2 pre-tokenizer regex:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|
// \p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// The checked-in scanner avoids a locale/regex-engine dependency and, in
// particular, preserves Qwen's one-codepoint number pieces.
std::vector<std::string> pre_tokenize_qwen(const std::string & text) {
    std::vector<std::string> pieces;
    size_t position = 0;
    auto peek = [&](size_t at, int * cp_length) -> uint32_t {
        if (at >= text.size()) {
            *cp_length = 0;
            return 0;
        }
        return utf8_decode(text.data() + at, text.size() - at, cp_length);
    };

    while (position < text.size()) {
        const size_t start = position;
        int cp_length = 0;
        const uint32_t cp = peek(position, &cp_length);

        if (cp == '\'') {
            size_t scan = position + 1;
            bool matched = false;
            if (scan < text.size()) {
                const char c = static_cast<char>(text[scan] | 0x20);
                if (c == 's' || c == 't' || c == 'm' || c == 'd') {
                    ++scan;
                    matched = true;
                } else if (scan + 1 < text.size()) {
                    const char next = static_cast<char>(text[scan + 1] | 0x20);
                    if ((c == 'r' && next == 'e') ||
                        (c == 'v' && next == 'e') ||
                        (c == 'l' && next == 'l')) {
                        scan += 2;
                        matched = true;
                    }
                }
            }
            if (matched) {
                pieces.push_back(text.substr(position, scan - position));
                position = scan;
                continue;
            }
        }

        size_t scan = position;
        int scan_length = 0;
        uint32_t scan_cp = peek(scan, &scan_length);
        if (scan_length > 0 && !qwen_newline(scan_cp) &&
            !qwen_letter(scan_cp) && !qwen_number(scan_cp)) {
            scan += static_cast<size_t>(scan_length);
            scan_cp = peek(scan, &scan_length);
        }
        if (scan_length > 0 && (qwen_letter(scan_cp) || qwen_mark(scan_cp))) {
            while (scan_length > 0 &&
                   (qwen_letter(scan_cp) || qwen_mark(scan_cp))) {
                scan += static_cast<size_t>(scan_length);
                scan_cp = peek(scan, &scan_length);
            }
            pieces.push_back(text.substr(position, scan - position));
            position = scan;
            continue;
        }

        if (qwen_number(cp)) {
            position += static_cast<size_t>(cp_length);
            pieces.push_back(text.substr(start, position - start));
            continue;
        }

        scan = position;
        scan_cp = peek(scan, &scan_length);
        if (scan_cp == ' ') {
            scan += static_cast<size_t>(scan_length);
            scan_cp = peek(scan, &scan_length);
        }
        const size_t punctuation_start = scan;
        while (scan_length > 0 && !qwen_space(scan_cp) &&
               !qwen_letter(scan_cp) && !qwen_mark(scan_cp) &&
               !qwen_number(scan_cp)) {
            scan += static_cast<size_t>(scan_length);
            scan_cp = peek(scan, &scan_length);
        }
        if (scan > punctuation_start) {
            while (scan_length > 0 && qwen_newline(scan_cp)) {
                scan += static_cast<size_t>(scan_length);
                scan_cp = peek(scan, &scan_length);
            }
            pieces.push_back(text.substr(position, scan - position));
            position = scan;
            continue;
        }

        if (qwen_space(cp)) {
            scan = position;
            scan_cp = peek(scan, &scan_length);
            while (scan_length > 0 && qwen_space(scan_cp) &&
                   !qwen_newline(scan_cp)) {
                scan += static_cast<size_t>(scan_length);
                scan_cp = peek(scan, &scan_length);
            }
            if (scan_length > 0 && qwen_newline(scan_cp)) {
                while (scan_length > 0 && qwen_newline(scan_cp)) {
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
            while (scan_length > 0 && qwen_space(scan_cp)) {
                before_last = scan;
                scan += static_cast<size_t>(scan_length);
                scan_cp = peek(scan, &scan_length);
            }
            const bool followed_by_non_space =
                scan < text.size() && !qwen_space(scan_cp);
            if (!followed_by_non_space) {
                position = scan;
            } else if (before_last > position) {
                position = before_last;
            } else {
                position = scan;
            }
            pieces.push_back(text.substr(start, position - start));
            continue;
        }

        position += cp_length > 0 ? static_cast<size_t>(cp_length) : 1;
        pieces.push_back(text.substr(start, position - start));
    }
    return pieces;
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

bool pre_tokenizer_supported(const char * name) {
    PreTokenizer ignored = PreTokenizer::JOYAI_LLM;
    return pre_tokenizer_from_name(name, ignored);
}

std::vector<std::string> pre_tokenize_text(const std::string & text,
                                           PreTokenizer type) {
    switch (type) {
        case PreTokenizer::QWEN2:
        case PreTokenizer::QWEN35:
            return pre_tokenize_qwen(qwen_normalize_nfc(text));
        case PreTokenizer::JOYAI_LLM:
            return pre_tokenize_joyai(text);
    }
    return {};
}

} // namespace dflash::common
