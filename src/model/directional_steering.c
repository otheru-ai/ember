#include "directional_steering.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 &&
               FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128, "direction file requires IEEE F32");

static bool fail(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message);
    return false;
}

bool ember_directional_steering_load(const char *path, float attn, float ffn,
                                    ember_directional_steering **out,
                                    char *error, size_t error_cap) {
    if (error && error_cap) error[0] = '\0';
    if (!out) return fail(error, error_cap, "missing steering output");
    *out = NULL;
    if (!isfinite(attn) || !isfinite(ffn) || fabsf(attn) > 100.0f ||
        fabsf(ffn) > 100.0f)
        return fail(error, error_cap, "steering scales must be finite in [-100,100]");
    if (!path || !path[0]) {
        if (attn == 0.0f && ffn == 0.0f) return true;
        return fail(error, error_cap, "directional steering needs --dir-steering-file");
    }
    FILE *file = fopen(path, "rb");
    if (!file) return fail(error, error_cap, "cannot open direction file");
    ember_directional_steering *s = calloc(1, sizeof(*s));
    if (!s) {
        fclose(file);
        return fail(error, error_cap, "cannot allocate steering directions");
    }
    // Read the whole fixed-size payload once: graph tensors and cache identity
    // derive from this same allocation, never a second open of a mutable path.
    const size_t got = fread(s->rows, 1, sizeof(s->rows), file);
    const int extra = fgetc(file);
    const bool valid_size = got == sizeof(s->rows) && extra == EOF && !ferror(file);
    fclose(file);
    if (!valid_size) {
        free(s);
        return fail(error, error_cap, "direction file must contain exactly 704512 bytes");
    }
    bool any = false;
    for (int layer = 0; layer < EMBER_STEERING_LAYERS; ++layer) {
        for (int i = 0; i < EMBER_STEERING_WIDTH; ++i) {
            const unsigned char *b = (const unsigned char *)&s->rows[layer][i];
            const uint32_t bits = (uint32_t)b[0] | (uint32_t)b[1] << 8 |
                                  (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
            float value;
            memcpy(&value, &bits, sizeof(value));
            if (!isfinite(value)) {
                free(s);
                return fail(error, error_cap, "direction file contains nonfinite values");
            }
            s->rows[layer][i] = value == 0.0f ? 0.0f : value;
            if (value != 0.0f) s->nonzero[layer] = any = true;
        }
    }
    if (!any || (attn == 0.0f && ffn == 0.0f)) {
        free(s);
        return true;
    }
    s->attn_scale = attn == 0.0f ? 0.0f : attn;
    s->ffn_scale = ffn == 0.0f ? 0.0f : ffn;
    *out = s;
    return true;
}

uint64_t ember_directional_steering_identity(
    const ember_directional_steering *s, uint64_t seed) {
    if (!s) return seed;
    const unsigned char tag[] = "ember-directional-steering-v1";
    for (size_t i = 0; i < sizeof(tag); ++i) {
        seed ^= tag[i];
        seed *= UINT64_C(1099511628211);
    }
    // Canonical little-endian float bits, no padding or pointer/path identity.
    for (size_t i = 0; i < 2 + EMBER_STEERING_LAYERS * EMBER_STEERING_WIDTH; ++i) {
        float value;
        if (i == 0) value = s->attn_scale;
        else if (i == 1) value = s->ffn_scale;
        else value = s->rows[(i - 2) / EMBER_STEERING_WIDTH]
                            [(i - 2) % EMBER_STEERING_WIDTH];
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        for (unsigned shift = 0; shift < 32; shift += 8) {
            seed ^= (bits >> shift) & 255u;
            seed *= UINT64_C(1099511628211);
        }
    }
    return seed;
}
