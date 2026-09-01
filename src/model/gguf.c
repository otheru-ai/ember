#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── little-endian readers over a FILE* ──────────────────────────────────
static bool rd(FILE *f, void *dst, size_t n) { return fread(dst, 1, n, f) == n; }
static bool rd_u32(FILE *f, uint32_t *v) { return rd(f, v, 4); }
static bool rd_u64(FILE *f, uint64_t *v) { return rd(f, v, 8); }

// GGUF string: u64 length + bytes (not NUL-terminated on disk).
static char *rd_str(FILE *f) {
    uint64_t n;
    if (!rd_u64(f, &n)) return NULL;
    if (n > (1u << 28)) return NULL;  // sanity
    char *s = (char *)malloc(n + 1);
    if (!rd(f, s, n)) { free(s); return NULL; }
    s[n] = '\0';
    return s;
}

static size_t type_size(gguf_type t) {
    switch (t) {
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
        case GGUF_U16: case GGUF_I16: return 2;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
        default: return 0;  // string/array handled separately
    }
}

// Read a scalar of type t into the kv's i/f fields.
static bool rd_scalar(FILE *f, gguf_type t, int64_t *i_out, double *f_out) {
    unsigned char buf[8];
    size_t sz = type_size(t);
    if (sz == 0 || !rd(f, buf, sz)) return false;
    switch (t) {
        case GGUF_U8:  *i_out = buf[0]; break;
        case GGUF_I8:  *i_out = (int8_t)buf[0]; break;
        case GGUF_BOOL:*i_out = buf[0] ? 1 : 0; break;
        case GGUF_U16: *i_out = (uint16_t)(buf[0] | buf[1] << 8); break;
        case GGUF_I16: *i_out = (int16_t)(buf[0] | buf[1] << 8); break;
        case GGUF_U32: case GGUF_I32: {
            uint32_t v; memcpy(&v, buf, 4);
            *i_out = (t == GGUF_I32) ? (int32_t)v : (int64_t)v;
            break;
        }
        case GGUF_F32: { float v; memcpy(&v, buf, 4); *f_out = v; break; }
        case GGUF_U64: case GGUF_I64: { uint64_t v; memcpy(&v, buf, 8); *i_out = (int64_t)v; break; }
        case GGUF_F64: { double v; memcpy(&v, buf, 8); *f_out = v; break; }
        default: return false;
    }
    return true;
}

static bool rd_kv(FILE *f, gguf_kv *kv) {
    memset(kv, 0, sizeof(*kv));
    kv->key = rd_str(f);
    if (!kv->key) return false;
    uint32_t t;
    if (!rd_u32(f, &t)) return false;
    kv->type = (gguf_type)t;

    if (kv->type == GGUF_STRING) {
        kv->str = rd_str(f);
        return kv->str != NULL;
    }
    if (kv->type == GGUF_ARRAY) {
        uint32_t et;
        uint64_t n;
        if (!rd_u32(f, &et) || !rd_u64(f, &n)) return false;
        kv->arr_type = (gguf_type)et;
        kv->arr_len = n;
        if (kv->arr_type == GGUF_STRING) {
            if (n > (1u << 24)) return false;
            kv->arr_str = (char **)calloc(n, sizeof(char *));
            for (uint64_t k = 0; k < n; k++) {
                kv->arr_str[k] = rd_str(f);
                if (!kv->arr_str[k]) return false;
            }
        } else if (kv->arr_type == GGUF_I32 || kv->arr_type == GGUF_U32) {
            if (n > (1u << 26)) return false;
            kv->arr_i32 = (int32_t *)malloc(n * 4);
            if (!rd(f, kv->arr_i32, n * 4)) return false;
        } else {
            // skip other element types (not needed by ember)
            size_t sz = type_size(kv->arr_type);
            if (sz == 0) return false;
            if (fseek(f, (long)(n * sz), SEEK_CUR) != 0) return false;
        }
        return true;
    }
    return rd_scalar(f, kv->type, &kv->i, &kv->f);
}

gguf_file *gguf_open(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[4];
    if (!rd(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0) { fclose(f); return NULL; }

    gguf_file *g = (gguf_file *)calloc(1, sizeof(gguf_file));
    if (!rd_u32(f, &g->version) || !rd_u64(f, &g->n_tensors) || !rd_u64(f, &g->n_kv)) {
        fclose(f); gguf_free(g); return NULL;
    }
    g->kv = (gguf_kv *)calloc(g->n_kv ? g->n_kv : 1, sizeof(gguf_kv));
    for (uint64_t i = 0; i < g->n_kv; i++) {
        if (!rd_kv(f, &g->kv[i])) { fclose(f); gguf_free(g); return NULL; }
    }

    g->tensors = (gguf_tensor *)calloc(g->n_tensors ? g->n_tensors : 1,
                                       sizeof(gguf_tensor));
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        t->name = rd_str(f);
        if (!t->name || !rd_u32(f, &t->n_dims) || t->n_dims > 4) {
            fclose(f); gguf_free(g); return NULL;
        }
        for (uint32_t d = 0; d < t->n_dims; d++)
            if (!rd_u64(f, &t->dims[d])) { fclose(f); gguf_free(g); return NULL; }
        if (!rd_u32(f, &t->type) || !rd_u64(f, &t->offset)) {
            fclose(f); gguf_free(g); return NULL;
        }
    }

    // data section starts at the next `alignment` boundary (default 32).
    int64_t align = gguf_get_int(g, "general.alignment", 32);
    if (align <= 0) align = 32;
    long pos = ftell(f);
    g->data_offset = (uint64_t)(((pos + align - 1) / align) * align);
    fclose(f);
    return g;
}

void gguf_free(gguf_file *g) {
    if (!g) return;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        gguf_kv *kv = &g->kv[i];
        free(kv->key);
        free(kv->str);
        if (kv->arr_str) {
            for (uint64_t k = 0; k < kv->arr_len; k++) free(kv->arr_str[k]);
            free(kv->arr_str);
        }
        free(kv->arr_i32);
    }
    free(g->kv);
    for (uint64_t i = 0; i < g->n_tensors; i++) free(g->tensors[i].name);
    free(g->tensors);
    free(g);
}

const gguf_kv *gguf_get(const gguf_file *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (strcmp(g->kv[i].key, key) == 0) return &g->kv[i];
    return NULL;
}
int64_t gguf_get_int(const gguf_file *g, const char *key, int64_t dflt) {
    const gguf_kv *kv = gguf_get(g, key);
    return kv && kv->type != GGUF_STRING && kv->type != GGUF_ARRAY &&
                   kv->type != GGUF_F32 && kv->type != GGUF_F64
               ? kv->i : dflt;
}
double gguf_get_float(const gguf_file *g, const char *key, double dflt) {
    const gguf_kv *kv = gguf_get(g, key);
    return kv && (kv->type == GGUF_F32 || kv->type == GGUF_F64) ? kv->f : dflt;
}
const char *gguf_get_str(const gguf_file *g, const char *key, const char *dflt) {
    const gguf_kv *kv = gguf_get(g, key);
    return kv && kv->type == GGUF_STRING ? kv->str : dflt;
}
