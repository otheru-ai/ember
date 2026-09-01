// Minimal JSON string escaping for SSE delta output. Hand-rolled (like ds4) to
// avoid pulling a JSON library into the hot streaming path. Input is assumed to
// already be valid UTF-8 (the streaming layer cuts on codepoint boundaries);
// bytes >= 0x20 pass through verbatim so multi-byte UTF-8 is preserved intact.
#ifndef EMBER_JSON_UTIL_H
#define EMBER_JSON_UTIL_H

#include "buf.h"

// Append `s` (len bytes) to `b` as a quoted, escaped JSON string literal.
static inline void ember_json_escape_n(ember_buf *b, const char *s, size_t n) {
    ember_buf_putc(b, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
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
                    ember_buf_putc(b, (char)c);  // >=0x20: verbatim (UTF-8 safe)
                }
        }
    }
    ember_buf_putc(b, '"');
}

static inline void ember_json_escape(ember_buf *b, const char *s) {
    ember_json_escape_n(b, s, s ? __builtin_strlen(s) : 0);
}

#endif  // EMBER_JSON_UTIL_H
