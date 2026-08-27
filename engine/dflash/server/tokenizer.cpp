// BPE tokenizer implementation.
//
// GGUF loading uses ggml's gguf_init_from_file API (already vendored).
// Pre-tokenization is delegated to the model-selected splitter in
// pre_tokenizer.cpp; BPE and detokenization stay here.

#include "tokenizer.h"

#include "gguf.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

namespace dflash::common {

// ─── Unicode helpers ────────────────────────────────────────────────────

static uint32_t utf8_decode(const char * s, size_t remaining, int * len) {
    if (remaining == 0) { *len = 0; return 0xFFFD; }
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) { *len = 1; return c; }
    if ((c & 0xE0) == 0xC0 && remaining >= 2) {
        *len = 2;
        return ((uint32_t)(c & 0x1F) << 6) |
               ((uint32_t)((uint8_t)s[1]) & 0x3F);
    }
    if ((c & 0xF0) == 0xE0 && remaining >= 3) {
        *len = 3;
        return ((uint32_t)(c & 0x0F) << 12) |
               (((uint32_t)((uint8_t)s[1]) & 0x3F) << 6) |
               ((uint32_t)((uint8_t)s[2]) & 0x3F);
    }
    if ((c & 0xF8) == 0xF0 && remaining >= 4) {
        *len = 4;
        return ((uint32_t)(c & 0x07) << 18) |
               (((uint32_t)((uint8_t)s[1]) & 0x3F) << 12) |
               (((uint32_t)((uint8_t)s[2]) & 0x3F) << 6) |
               ((uint32_t)((uint8_t)s[3]) & 0x3F);
    }
    *len = 1;
    return 0xFFFD;
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string & text) const {
    return pre_tokenize_text(text, pre_type_);
}

// ─── BPE encoding ──────────────────────────────────────────────────────

// Forward GPT-2 byte encoding: raw byte → Unicode codepoint (UTF-8 string).
// This is the inverse of gpt2_unicode_to_byte (defined later, near decode).
// Bytes in {33-126, 161-172, 174-255} map to themselves as a codepoint;
// all others (0-32, 127-160, 173) map to U+0100..U+0143.
static std::string byte_to_gpt2_unicode(uint8_t b) {
    // Build forward table once (thread-safe via C++11 static init).
    static const auto fwd = []() {
        std::array<uint32_t, 256> t{};
        int n = 0;
        for (int i = 0; i < 256; i++) {
            if ((i >= 33  && i <= 126) ||
                (i >= 161 && i <= 172) ||
                (i >= 174 && i <= 255)) {
                t[i] = (uint32_t)i;
            } else {
                t[i] = 256 + n;
                n++;
            }
        }
        return t;
    }();
    uint32_t cp = fwd[b];
    // Encode codepoint as UTF-8.
    char buf[4];
    int len;
    if (cp < 0x80) {
        buf[0] = (char)cp; len = 1;
    } else if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        len = 2;
    } else {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        len = 3;
    }
    return std::string(buf, len);
}

// Convert a raw UTF-8 text piece to GPT-2 byte-encoded form for BPE lookup.
static std::string encode_gpt2_bpe(const std::string & text) {
    std::string out;
    out.reserve(text.size() * 2);  // GPT-2 encoding may expand
    for (uint8_t b : text) {
        out += byte_to_gpt2_unicode(b);
    }
    return out;
}

// Encode a single pre-tokenized piece using BPE merges.
std::vector<int32_t> Tokenizer::bpe_encode_piece(const std::string & piece) const {
    if (piece.empty()) return {};

    std::vector<std::string> symbols;

    if (is_sentencepiece_) {
        // SentencePiece: replace leading/embedded spaces with U+2581.
        std::string sp_piece;
        sp_piece.reserve(piece.size() + 2);
        size_t start = 0;
        if (!piece.empty() && piece[0] == ' ') {
            sp_piece += "\xe2\x96\x81";
            start = 1;
        }
        sp_piece += piece.substr(start);
        std::string encoded;
        encoded.reserve(sp_piece.size());
        for (char c : sp_piece) {
            if (c == ' ') encoded += "\xe2\x96\x81";
            else encoded += c;
        }

        auto it = token_to_id_.find(encoded);
        if (it != token_to_id_.end()) return {it->second};

        const char * p = encoded.c_str();
        const char * end = p + encoded.size();
        while (p < end) {
            int cplen = 0;
            (void)utf8_decode(p, static_cast<size_t>(end - p), &cplen);
            if (cplen <= 0) cplen = 1;
            std::string sym(p, static_cast<size_t>(cplen));
            auto sit = token_to_id_.find(sym);
            if (sit != token_to_id_.end()) {
                symbols.push_back(sym);
            } else {
                for (int j = 0; j < cplen; ++j) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "<0x%02X>",
                                  static_cast<unsigned>(
                                      static_cast<uint8_t>(p[j])));
                    symbols.push_back(buf);
                }
            }
            p += cplen;
        }
    } else {
        // Qwen and JoyAI use GPT-2 byte-level BPE.
        std::string encoded = encode_gpt2_bpe(piece);
        auto it = token_to_id_.find(encoded);
        if (it != token_to_id_.end()) return {it->second};
        for (size_t i = 0; i < piece.size(); ++i) {
            std::string sym = byte_to_gpt2_unicode(
                static_cast<uint8_t>(piece[i]));
            auto sit = token_to_id_.find(sym);
            if (sit != token_to_id_.end()) {
                symbols.push_back(sym);
            } else {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>",
                              static_cast<unsigned>(
                                  static_cast<uint8_t>(piece[i])));
                symbols.push_back(buf);
            }
        }
    }

    if (symbols.size() <= 1) {
        if (symbols.empty()) return {};
        auto sit = token_to_id_.find(symbols[0]);
        if (sit != token_to_id_.end()) return { sit->second };
        return {};  // unknown token
    }

    // Iteratively merge the highest-priority pair until no more merges apply.
    while (symbols.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_pos = SIZE_MAX;

        for (size_t i = 0; i + 1 < symbols.size(); i++) {
            std::string pair = symbols[i] + " " + symbols[i + 1];
            auto mit = merge_rank_.find(pair);
            if (mit != merge_rank_.end() && mit->second < best_rank) {
                best_rank = mit->second;
                best_pos = i;
            }
        }

        if (best_pos == SIZE_MAX) break;  // no more merges

        // Merge the best pair.
        symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
        symbols.erase(symbols.begin() + best_pos + 1);
    }

    // Convert merged symbols to token IDs.
    std::vector<int32_t> ids;
    ids.reserve(symbols.size());
    for (const auto & sym : symbols) {
        auto sit = token_to_id_.find(sym);
        if (sit != token_to_id_.end()) {
            ids.push_back(sit->second);
        } else {
            // Unknown symbol — emit byte-fallback tokens if available.
            // Symbols are in GPT-2 byte encoding; decode each Unicode codepoint
            // back to the original byte before emitting <0xNN>.
            static const auto gpt2_rev = []() {
                // Build reverse table: codepoint → original byte.
                std::array<uint8_t, 324> t{};  // covers U+0000..U+0143
                int n = 0;
                for (int b = 0; b < 256; b++) {
                    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
                        t[b] = (uint8_t)b;
                    } else {
                        t[256 + n] = (uint8_t)b;
                        n++;
                    }
                }
                return t;
            }();
            const char * p = sym.c_str();
            const char * end = p + sym.size();
            while (p < end) {
                int cplen;
                uint32_t cp = utf8_decode(p, (size_t)(end - p), &cplen);
                uint8_t orig_byte;
                if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
                    orig_byte = (uint8_t)cp;
                } else if (cp >= 256 && cp < 256 + 68) {
                    orig_byte = gpt2_rev[cp];
                } else {
                    orig_byte = '?';
                }
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned)orig_byte);
                auto bit = token_to_id_.find(buf);
                if (bit != token_to_id_.end()) {
                    ids.push_back(bit->second);
                }
                p += cplen;
            }
        }
    }

    return ids;
}

// ─── Public API ─────────────────────────────────────────────────────────

bool Tokenizer::load_from_gguf(const char * model_path) {
    Tokenizer next;
    if (!next.load_from_gguf_in_place(model_path)) return false;
    *this = std::move(next);
    return true;
}

bool Tokenizer::load_from_gguf_in_place(const char * model_path) {
    if (!model_path || model_path[0] == '\0') {
        std::fprintf(stderr, "[tokenizer] model path is empty\n");
        return false;
    }

    struct gguf_init_params params = { /*.no_alloc=*/ true, /*.ctx=*/ nullptr };
    struct gguf_context * gctx = gguf_init_from_file(model_path, params);
    if (!gctx) {
        std::fprintf(stderr, "[tokenizer] failed to open GGUF: %s\n", model_path);
        return false;
    }

    // Load token strings.
    int tokens_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tokens_key < 0) {
        std::fprintf(stderr, "[tokenizer] missing tokenizer.ggml.tokens in %s\n",
                     model_path);
        gguf_free(gctx);
        return false;
    }

    if (gguf_get_arr_type(gctx, tokens_key) != GGUF_TYPE_STRING) {
        std::fprintf(stderr, "[tokenizer] tokenizer.ggml.tokens has the wrong type\n");
        gguf_free(gctx);
        return false;
    }
    const size_t n_vocab_raw = gguf_get_arr_n(gctx, tokens_key);
    if (n_vocab_raw == 0 ||
        n_vocab_raw > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        std::fprintf(stderr, "[tokenizer] invalid vocabulary size: %zu\n", n_vocab_raw);
        gguf_free(gctx);
        return false;
    }
    const int n_vocab = static_cast<int>(n_vocab_raw);

    id_to_token_.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const char * tok = gguf_get_arr_str(gctx, tokens_key, i);
        id_to_token_[i] = tok ? tok : "";
        token_to_id_[id_to_token_[i]] = i;
    }

    // Load merge table.
    int merges_key = gguf_find_key(gctx, "tokenizer.ggml.merges");
    if (merges_key >= 0) {
        if (gguf_get_arr_type(gctx, merges_key) != GGUF_TYPE_STRING) {
            std::fprintf(stderr, "[tokenizer] tokenizer.ggml.merges has the wrong type\n");
            gguf_free(gctx);
            return false;
        }
        const size_t n_merges = gguf_get_arr_n(gctx, merges_key);
        if (n_merges > static_cast<size_t>(std::numeric_limits<int>::max())) {
            std::fprintf(stderr, "[tokenizer] merge table is too large: %zu\n", n_merges);
            gguf_free(gctx);
            return false;
        }
        for (size_t i = 0; i < n_merges; i++) {
            const char * merge = gguf_get_arr_str(gctx, merges_key, i);
            if (merge) {
                merge_rank_[merge] = static_cast<int>(i);
            }
        }
    }

    // Load token types and build added-tokens list.
    // GGUF token_type: 1=normal, 3=control, 4=user-defined, 5=unused
    // Types 3 and 4 are "special" tokens matched as whole strings before BPE.
    int type_key = gguf_find_key(gctx, "tokenizer.ggml.token_type");
    if (type_key >= 0) {
        const enum gguf_type type = gguf_get_arr_type(gctx, type_key);
        if (type != GGUF_TYPE_UINT32 && type != GGUF_TYPE_INT32) {
            std::fprintf(stderr,
                         "[tokenizer] tokenizer.ggml.token_type has the wrong type\n");
            gguf_free(gctx);
            return false;
        }
        const size_t n_types = std::min(gguf_get_arr_n(gctx, type_key), n_vocab_raw);
        const void * type_data = gguf_get_arr_data(gctx, type_key);
        if (n_types > 0 && !type_data) {
            std::fprintf(stderr, "[tokenizer] tokenizer token types are missing\n");
            gguf_free(gctx);
            return false;
        }
        for (size_t i = 0; i < n_types; i++) {
            const uint32_t ttype = type == GGUF_TYPE_UINT32
                ? static_cast<const uint32_t *>(type_data)[i]
                : static_cast<uint32_t>(static_cast<const int32_t *>(type_data)[i]);
            if (ttype == 3 || ttype == 4) {
                const std::string & tok = id_to_token_[i];
                if (!tok.empty()) {
                    added_tokens_.push_back({tok, (int32_t)i});
                }
            }
        }
        // Sort longest-first for greedy matching.
        std::sort(added_tokens_.begin(), added_tokens_.end(),
                  [](const auto & a, const auto & b) {
                      return a.first.size() > b.first.size();
                  });
        // Index by first byte, inheriting the longest-first order so encode()
        // need only probe the tokens that can possibly match at a given offset.
        for (auto & bucket : added_by_first_byte_) bucket.clear();
        for (uint32_t i = 0; i < added_tokens_.size(); ++i) {
            const std::string & tok = added_tokens_[i].first;
            if (tok.empty()) continue;
            added_by_first_byte_[(unsigned char) tok[0]].push_back(i);
        }
        std::fprintf(stderr, "[tokenizer] added_tokens: %zu special tokens\n",
                     added_tokens_.size());
    }

    // Detect tokenizer model type (SentencePiece vs byte-level BPE).
    const int model_key = gguf_find_key(gctx, "tokenizer.ggml.model");
    if (model_key >= 0) {
        const char * model = gguf_get_val_str(gctx, model_key);
        if (model && (std::strcmp(model, "llama") == 0 ||
                      std::strncmp(model, "gemma", 5) == 0)) {
            is_sentencepiece_ = true;
        }
    }

    // Detect pre-tokenizer type.
    int pre_key = gguf_find_key(gctx, "tokenizer.ggml.pre");
    if (pre_key < 0) {
        std::fprintf(stderr,
                     "[tokenizer] tokenizer.ggml.pre metadata is missing\n");
        gguf_free(gctx);
        return false;
    }
    if (gguf_get_kv_type(gctx, pre_key) != GGUF_TYPE_STRING) {
        std::fprintf(stderr,
                     "[tokenizer] tokenizer.ggml.pre has the wrong type\n");
        gguf_free(gctx);
        return false;
    }
    const char * pre = gguf_get_val_str(gctx, pre_key);
    if (!pre_tokenizer_from_name(pre, pre_type_)) {
        std::fprintf(stderr,
                     "[tokenizer] unsupported tokenizer.ggml.pre: %s\n",
                     pre ? pre : "(null)");
        gguf_free(gctx);
        return false;
    }

    // Load special token IDs.
    auto get_i32 = [&](const char * key) -> int32_t {
        int k = gguf_find_key(gctx, key);
        if (k < 0) return -1;
        int64_t value = -1;
        switch (gguf_get_kv_type(gctx, k)) {
            case GGUF_TYPE_UINT32: value = gguf_get_val_u32(gctx, k); break;
            case GGUF_TYPE_INT32:  value = gguf_get_val_i32(gctx, k); break;
            case GGUF_TYPE_UINT64: {
                const uint64_t raw = gguf_get_val_u64(gctx, k);
                if (raw > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
                    return -1;
                }
                value = static_cast<int64_t>(raw);
                break;
            }
            case GGUF_TYPE_INT64: value = gguf_get_val_i64(gctx, k); break;
            default: return -1;
        }
        if (value < 0 || value >= n_vocab) return -1;
        return static_cast<int32_t>(value);
    };

    bos_id_ = get_i32("tokenizer.ggml.bos_token_id");
    eos_id_ = get_i32("tokenizer.ggml.eos_token_id");
    eos_chat_id_ = get_i32("tokenizer.ggml.eot_token_id");
    if (eos_chat_id_ < 0) {
        const auto eot = token_to_id_.find("<|im_end|>");
        if (eot != token_to_id_.end()) eos_chat_id_ = eot->second;
    }
    if (eos_chat_id_ < 0) {
        const auto eot = token_to_id_.find("<turn|>");
        if (eot != token_to_id_.end()) eos_chat_id_ = eot->second;
    }

    gguf_free(gctx);

    std::fprintf(stderr,
                 "[tokenizer] loaded vocab=%d merges=%zu bos=%d eos=%d "
                 "eot=%d pre=%s sp=%s\n",
                 n_vocab, merge_rank_.size(), bos_id_, eos_id_, eos_chat_id_,
                 pre_tokenizer_name(pre_type_),
                 is_sentencepiece_ ? "yes" : "no");
    return true;
}

// Longest added token matching at `pos`, or -1. added_by_first_byte_ inherits
// the longest-first order of added_tokens_, and a token can only match here if
// its first byte equals text[pos], so probing that one bucket in order returns
// exactly what a full longest-first scan would.
int Tokenizer::match_added_token_at(const std::string & text,
                                    size_t pos) const {
    if (pos >= text.size()) return -1;
    const std::vector<uint32_t> & bucket =
        added_by_first_byte_[(unsigned char) text[pos]];
    for (uint32_t idx : bucket) {
        const std::string & tok = added_tokens_[idx].first;
        if (pos + tok.size() <= text.size() &&
            text.compare(pos, tok.size(), tok) == 0) {
            return (int) idx;
        }
    }
    return -1;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    // If no added tokens, fast path: pre-tokenize → BPE entire text.
    if (added_tokens_.empty()) {
        std::vector<std::string> pieces = pre_tokenize(text);
        std::vector<int32_t> ids;
        for (const auto & piece : pieces) {
            auto piece_ids = bpe_encode_piece(piece);
            ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
        }
        return ids;
    }

    // Split text into segments: alternating normal text and special tokens.
    // Special tokens are matched greedily (longest first).
    std::vector<int32_t> ids;
    size_t pos = 0;
    while (pos < text.size()) {
        // Try to match any added token at current position.
        const int hit = match_added_token_at(text, pos);
        if (hit >= 0) {
            ids.push_back(added_tokens_[(size_t) hit].second);
            pos += added_tokens_[(size_t) hit].first.size();
            continue;
        }

        // Find the next special token (or end of string). Scanning forward and
        // testing only the first-byte bucket is equivalent to taking the
        // earliest text.find() over every added token, but linear in the
        // remaining text instead of (added_tokens * text_len) per segment.
        size_t next_special = pos;
        while (next_special < text.size() &&
               match_added_token_at(text, next_special) < 0) {
            ++next_special;
        }

        // Pre-tokenize + BPE the normal segment.
        std::string segment = text.substr(pos, next_special - pos);
        std::vector<std::string> pieces = pre_tokenize(segment);
        for (const auto & piece : pieces) {
            auto piece_ids = bpe_encode_piece(piece);
            ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
        }
        pos = next_special;
    }
    return ids;
}

// GPT-2 byte-level BPE uses a Unicode mapping where each byte 0-255 is
// represented by a specific Unicode codepoint.  Bytes that already have a
// printable representation (33-126, 161-172, 174-255) map to themselves;
// all other bytes (0-32, 127-160, 173) are offset into U+0100..U+0143.
// Token strings in the GGUF vocabulary are stored in this encoding, so we
// must reverse-map each codepoint back to its original byte.

static uint8_t gpt2_unicode_to_byte(uint32_t cp) {
    // Direct-mapped ranges: the codepoint IS the byte.
    if ((cp >= 33  && cp <= 126) ||
        (cp >= 161 && cp <= 172) ||
        (cp >= 174 && cp <= 255)) {
        return (uint8_t)cp;
    }
    // Offset-mapped range: U+0100..U+0143 → non-printable bytes.
    // Build the reverse table once (thread-safe via C++11 static init).
    static const auto table = []() {
        std::array<uint8_t, 68> t{};
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if ((b >= 33  && b <= 126) ||
                (b >= 161 && b <= 172) ||
                (b >= 174 && b <= 255)) continue;
            t[n] = (uint8_t)b;
            n++;
        }
        return t;
    }();
    if (cp >= 256 && cp < 256 + 68) {
        return table[cp - 256];
    }
    // Shouldn't happen for valid BPE tokens — return replacement.
    return '?';
}

static std::string decode_gpt2_bpe(const std::string & tok) {
    std::string out;
    out.reserve(tok.size());
    const char * p = tok.c_str();
    const char * end = p + tok.size();
    while (p < end) {
        int cplen;
        uint32_t cp = utf8_decode(p, (size_t)(end - p), &cplen);
        out.push_back((char)gpt2_unicode_to_byte(cp));
        p += cplen;
    }
    return out;
}

std::string Tokenizer::token_text(int32_t id) const {
    if (id < 0 || id >= (int32_t)id_to_token_.size()) return "";
    const std::string & tok = id_to_token_[id];

    // Handle byte-fallback tokens like <0xNN>.
    if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' &&
        tok[2] == 'x' && tok[5] == '>') {
        unsigned val = 0;
        if (std::sscanf(tok.c_str(), "<0x%02X>", &val) == 1) {
            return std::string(1, (char)(uint8_t)val);
        }
    }

    // Special tokens are returned as-is.
    if (!tok.empty() && tok[0] == '<' && tok.back() == '>') {
        return tok;
    }

    if (is_sentencepiece_) {
        std::string out;
        out.reserve(tok.size());
        const char * p = tok.c_str();
        const char * end = p + tok.size();
        while (p < end) {
            if (end - p >= 3 &&
                static_cast<uint8_t>(p[0]) == 0xE2 &&
                static_cast<uint8_t>(p[1]) == 0x96 &&
                static_cast<uint8_t>(p[2]) == 0x81) {
                out.push_back(' ');
                p += 3;
            } else {
                out.push_back(*p++);
            }
        }
        return out;
    }

    // Decode GPT-2 byte-level BPE encoding → raw bytes.
    return decode_gpt2_bpe(tok);
}

std::string Tokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string result;
    for (int32_t id : ids) {
        result += token_text(id);
    }
    return result;
}

const std::string & Tokenizer::raw_token(int32_t id) const {
    static const std::string empty;
    if (id < 0 || id >= (int32_t)id_to_token_.size()) return empty;
    return id_to_token_[id];
}

int32_t Tokenizer::token_to_id(const std::string & token) const {
    auto it = token_to_id_.find(token);
    return it != token_to_id_.end() ? it->second : -1;
}

}  // namespace dflash::common
