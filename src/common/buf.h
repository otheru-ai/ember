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

static inline void ember_buf_free(ember_buf *b) {
    free(b->ptr);
    b->ptr = NULL;
    b->len = b->cap = 0;
}

static inline void ember_buf_reserve(ember_buf *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return;
    size_t ncap = b->cap ? b->cap : 64;
    while (ncap < b->len + extra + 1) ncap *= 2;
    b->ptr = (char *)realloc(b->ptr, ncap);
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
    va_list ap2;
    va_copy(ap2, ap);
    int need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need > 0) {
        ember_buf_reserve(b, (size_t)need);
        vsnprintf(b->ptr + b->len, (size_t)need + 1, fmt, ap2);
        b->len += (size_t)need;
    }
    va_end(ap2);
}

// Take ownership of the buffer's storage; caller must free(). Resets `b`.
static inline char *ember_buf_take(ember_buf *b) {
    char *p = b->ptr;
    b->ptr = NULL;
    b->len = b->cap = 0;
    return p;
}

#endif  // EMBER_BUF_H
