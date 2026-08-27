// GGUF metadata reader: parse the header, KV metadata, and tensor directory of
// a .gguf file WITHOUT loading tensor data. Enough to drive the tokenizer
// (tokens/merges arrays), model hyperparameters, and to locate tensors for the
// backend to mmap. GGUF v3 little-endian.
#ifndef EMBER_GGUF_H
#define EMBER_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64
} gguf_type;

typedef struct {
    char       *key;         // owned
    gguf_type   type;
    // scalars
    int64_t     i;           // integer-ish value (u8..i64, bool)
    double      f;           // f32/f64
    char       *str;         // GGUF_STRING (owned)
    // arrays
    gguf_type   arr_type;    // element type when type==GGUF_ARRAY
    uint64_t    arr_len;
    char      **arr_str;     // materialized when arr_type==GGUF_STRING (owned)
    int32_t    *arr_i32;     // materialized when arr_type is a 32-bit int
} gguf_kv;

typedef struct {
    char     *name;          // owned
    uint32_t  n_dims;
    uint64_t  dims[4];
    uint32_t  type;          // ggml tensor type id
    uint64_t  offset;        // from data section start
} gguf_tensor;

typedef struct {
    uint32_t     version;
    uint64_t     n_tensors;
    uint64_t     n_kv;
    gguf_kv     *kv;
    gguf_tensor *tensors;
    uint64_t     data_offset;   // aligned start of tensor data in the file
} gguf_file;

// Read metadata from `path`. Returns NULL on failure. Free with ember_gguf_free.
gguf_file *ember_gguf_open(const char *path);
void       ember_gguf_free(gguf_file *g);

// KV lookups (NULL / defaults if absent or wrong type).
const gguf_kv *ember_gguf_get(const gguf_file *g, const char *key);
int64_t     ember_gguf_get_int(const gguf_file *g, const char *key, int64_t dflt);
double      ember_gguf_get_float(const gguf_file *g, const char *key, double dflt);
const char *ember_gguf_get_str(const gguf_file *g, const char *key, const char *dflt);

#endif  // EMBER_GGUF_H
