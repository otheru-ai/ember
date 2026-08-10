// Growable byte buffer. Minimal, allocation-doubling, always NUL-terminated
// so .ptr is a valid C string for text payloads. Modeled on ds4's `buf`.
#ifndef EMBER_BUF_H
#define EMBER_BUF_H

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char  *ptr;
    size_t len;
    size_t cap;
} ember_buf;

// Buffers are used to construct wire-format JSON and prompts throughout the
// server. Continuing after an allocation failure would either dereference NULL
// or emit a truncated, syntactically valid-looking payload. Match Dwarfstar's
// fail-fast allocation policy instead: OOM/size overflow terminates cleanly at
// the allocation site rather than corrupting protocol state.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noreturn))
#endif
static inline void ember_buf_fatal(const char *why) {
    fprintf(stderr, "ember: %s\n", why);
    abort();
}

static inline void ember_buf_free(ember_buf *b) {
    free(b->ptr);
    b->ptr = NULL;
    b->len = b->cap = 0;
}

static inline void ember_buf_reserve(ember_buf *b, size_t extra) {
    if (extra > SIZE_MAX - b->len - 1) ember_buf_fatal("buffer size overflow");
    size_t need = b->len + extra + 1;
    if (need <= b->cap) {
        if (!b->ptr) ember_buf_fatal("buffer capacity without storage");
        return;
    }
    size_t ncap = b->cap ? b->cap : 64;
    while (ncap < need) {
        if (ncap > SIZE_MAX / 2) {
            ncap = need;
            break;
        }
        ncap *= 2;
    }
    char *grown = (char *)realloc(b->ptr, ncap);
    if (!grown) ember_buf_fatal("out of memory growing buffer");
    b->ptr = grown;
    b->cap = ncap;
}

static inline void ember_buf_append(ember_buf *b, const void *data, size_t n) {
    if (n == 0) return;
    ember_buf_reserve(b, n);
    memcpy(b->ptr + b->len, data, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static inline void ember_buf_puts(ember_buf *b, const char *s) {
    if (!s) ember_buf_fatal("NULL string appended to buffer");
    ember_buf_append(b, s, strlen(s));
}

static inline void ember_buf_putc(ember_buf *b, char c) {
    ember_buf_reserve(b, 1);
    b->ptr[b->len++] = c;
    b->ptr[b->len] = '\0';
}

static inline void ember_buf_printf(ember_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) ember_buf_fatal("vsnprintf failed");
    if (need > 0) {
        ember_buf_reserve(b, (size_t)need);
        va_start(ap, fmt);
        int wrote = vsnprintf(b->ptr + b->len, (size_t)need + 1, fmt, ap);
        va_end(ap);
        if (wrote != need) ember_buf_fatal("vsnprintf length changed");
        b->len += (size_t)need;
    }
}

// Take ownership of the buffer's storage; caller must free(). Resets `b`.
static inline char *ember_buf_take(ember_buf *b) {
    char *p = b->ptr;
    b->ptr = NULL;
    b->len = b->cap = 0;
    return p;
}

#endif  // EMBER_BUF_H
