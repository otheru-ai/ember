#ifndef EMBER_COMMON_BYTES_H
#define EMBER_COMMON_BYTES_H

#include <stddef.h>
#include <stdint.h>

// Little-endian pack/unpack and FNV-1a, shared by the on-disk record formats.
// continuation.c and tool_memory.c each carried a byte-identical private copy;
// two copies of a serialisation primitive is how the two formats drift apart.
//
// static inline keeps this header-only: no linkage change, and a translation
// unit that uses only some of these does not trip -Wunused-function.
//
// chat_api.c's image digest folds one byte at a time under a different
// contract and is deliberately left alone; rewriting it would be a wider diff
// through digest correctness for no behavioural gain.

static inline uint32_t ember_get_u32le(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static inline uint64_t ember_get_u64le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

static inline void ember_put_u32le(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (i * 8));
}

static inline void ember_put_u64le(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (i * 8));
}

static inline uint64_t ember_fnv1a_update(uint64_t h, const void *data,
                                          size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

#endif  // EMBER_COMMON_BYTES_H
