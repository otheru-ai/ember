// UTF-8 stream safety. The DeepSeek tokenizer is byte-level (gpt2) BPE, so a
// single emoji / CJK codepoint is emitted across several tokens, each a partial
// UTF-8 fragment. A streaming emitter must never cut mid-codepoint: it holds
// back a trailing incomplete sequence until the continuation bytes arrive.
//
// Ported from ds4_server.c's utf8_stream_safe_len (the correct version):
// unlike a naive walk-back-over-continuations, this ALSO excludes a lone
// trailing lead byte that begins an as-yet-incomplete sequence. Lucebox shipped
// the naive version and corrupted emoji ("Hey there ────") until patched.
#ifndef EMBER_UTF8_H
#define EMBER_UTF8_H

#include <stddef.h>
#include <stdint.h>

// Bytes a UTF-8 sequence needs given its lead byte (0 = not a lead byte).
static inline int ember_utf8_expected_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 0;  // continuation byte or invalid
}

// Largest length <= `limit` of s[start..] that ends on a COMPLETE codepoint
// boundary. When `final` is true, no holdback (end of stream — emit whatever's
// left). `start` is the already-emitted watermark so we never trim below it.
static inline size_t ember_utf8_stream_safe_len(const char *s, size_t start,
                                                size_t limit, int final) {
    if (final || !s || limit <= start) return limit;

    size_t p = limit, cont = 0;
    while (p > start && ((uint8_t)s[p - 1] & 0xC0) == 0x80 && cont < 4) {
        p--;
        cont++;
    }
    if (p == limit) {
        // No continuation bytes at the tail: the last byte is either ASCII
        // (complete) or a lead byte whose continuations haven't arrived.
        return ember_utf8_expected_len((uint8_t)s[limit - 1]) > 1 ? limit - 1
                                                                  : limit;
    }
    if (p == start) return start;  // all continuations (malformed) — hold all
    // p-1 is the lead byte of the trailing sequence; is it complete within limit?
    int expected = ember_utf8_expected_len((uint8_t)s[p - 1]);
    if (expected == 0) return limit;             // malformed lead; leave as-is
    if ((p - 1) + (size_t)expected > limit) return p - 1;  // incomplete: hold
    return limit;
}

#endif  // EMBER_UTF8_H
