// Minimal JSON string escaping for protocol output. Hand-rolled (like ds4) to
// avoid pulling a JSON library into the hot streaming path. Streaming normally
// cuts on codepoint boundaries, but buffered generation can stop between the
// byte tokens of one UTF-8 codepoint. The HTTP boundary therefore validates
// non-ASCII sequences and substitutes U+FFFD for malformed/incomplete input;
// emitting invalid JSON text is never an acceptable representation of a token
// cutoff.
#ifndef EMBER_JSON_UTIL_H
#define EMBER_JSON_UTIL_H

#include "buf.h"

static inline int ember_json_utf8_sequence_len(const unsigned char *s,
                                                size_t n) {
    if (n == 0) return 0;
    const unsigned char c = s[0];
    if (c < 0x80) return 1;
    if (c >= 0xc2 && c <= 0xdf) {
        return n >= 2 && (s[1] & 0xc0) == 0x80 ? 2 : 0;
    }
    if (c >= 0xe0 && c <= 0xef) {
        if (n < 3 || (s[2] & 0xc0) != 0x80) return 0;
        if (c == 0xe0) return s[1] >= 0xa0 && s[1] <= 0xbf ? 3 : 0;
        if (c == 0xed) return s[1] >= 0x80 && s[1] <= 0x9f ? 3 : 0;
        return (s[1] & 0xc0) == 0x80 ? 3 : 0;
    }
    if (c >= 0xf0 && c <= 0xf4) {
        if (n < 4 || (s[2] & 0xc0) != 0x80 ||
            (s[3] & 0xc0) != 0x80) return 0;
        if (c == 0xf0) return s[1] >= 0x90 && s[1] <= 0xbf ? 4 : 0;
        if (c == 0xf4) return s[1] >= 0x80 && s[1] <= 0x8f ? 4 : 0;
        return (s[1] & 0xc0) == 0x80 ? 4 : 0;
    }
    return 0;
}

// Number of bytes belonging to one malformed sequence. Consuming its trailing
// continuation bytes emits one replacement character per bad sequence instead
// of one per byte, while never swallowing the start of the next codepoint.
static inline size_t ember_json_invalid_utf8_span(const unsigned char *s,
                                                   size_t n) {
    if (n == 0) return 0;
    size_t limit = 1;
    const unsigned char c = s[0];
    if ((c & 0xe0) == 0xc0) limit = 2;
    else if ((c & 0xf0) == 0xe0) limit = 3;
    else if ((c & 0xf8) == 0xf0) limit = 4;
    else if ((c & 0xc0) == 0x80) limit = 4;
    size_t span = 1;
    while (span < n && span < limit && (s[span] & 0xc0) == 0x80)
        span++;
    return span;
}

// Append escaped JSON string contents without surrounding quotes.
static inline void ember_json_escape_content_n(ember_buf *b, const char *s,
                                                size_t n) {
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 0x80) {
            int seq = ember_json_utf8_sequence_len(
                (const unsigned char *)s + i, n - i);
            if (seq > 0) {
                ember_buf_append(b, s + i, (size_t)seq);
                i += (size_t)seq;
            } else {
                ember_buf_append(b, "\xef\xbf\xbd", 3);
                i += ember_json_invalid_utf8_span(
                    (const unsigned char *)s + i, n - i);
            }
            continue;
        }
        switch (c) {
            case '"':  ember_buf_puts(b, "\\\""); break;
            case '\\': ember_buf_puts(b, "\\\\"); break;
            case '\n': ember_buf_puts(b, "\\n");  break;
            case '\r': ember_buf_puts(b, "\\r");  break;
            case '\t': ember_buf_puts(b, "\\t");  break;
            case '\b': ember_buf_puts(b, "\\b");  break;
            case '\f': ember_buf_puts(b, "\\f");  break;
            default:
                if (c < 0x20) {
                    ember_buf_printf(b, "\\u%04x", (unsigned)c);
                } else {
                    ember_buf_putc(b, (char)c);
                }
        }
        i++;
    }
}

static inline void ember_json_escape_content(ember_buf *b, const char *s) {
    ember_json_escape_content_n(b, s, s ? __builtin_strlen(s) : 0);
}

// Append `s` (len bytes) to `b` as a quoted, escaped JSON string literal.
static inline void ember_json_escape_n(ember_buf *b, const char *s, size_t n) {
    ember_buf_putc(b, '"');
    ember_json_escape_content_n(b, s, n);
    ember_buf_putc(b, '"');
}

static inline void ember_json_escape(ember_buf *b, const char *s) {
    ember_json_escape_n(b, s, s ? __builtin_strlen(s) : 0);
}

#endif  // EMBER_JSON_UTIL_H
