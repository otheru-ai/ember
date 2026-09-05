#ifndef EMBER_DIRECTIONAL_STEERING_H
#define EMBER_DIRECTIONAL_STEERING_H

// Immutable startup policy shared by the ABI implementations. The operator
// supplies 43 x 4096 little-endian IEEE F32 values, layer-major, without a
// header. Rows are used verbatim (no normalization); zero rows exclude layers.
// Matches ds4.c cpu_directional_steering_project_rows: y -= scale*v*dot(v,y),
// at attention output and combined shared+routed FFN output, before HC post.
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define EMBER_STEERING_LAYERS 43
#define EMBER_STEERING_WIDTH 4096

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ember_directional_steering {
    float attn_scale;
    float ffn_scale;
    bool nonzero[EMBER_STEERING_LAYERS];
    float rows[EMBER_STEERING_LAYERS][EMBER_STEERING_WIDTH];
} ember_directional_steering;

// Success with *out=NULL means disabled. Every supplied file is validated,
// even at zero scales. Caller owns *out (free). Error storage is caller-owned.
bool ember_directional_steering_load(const char *path, float attn, float ffn,
                                    ember_directional_steering **out,
                                    char *error, size_t error_cap);

// Fold effective content and scales into the existing disk-cache identity.
// Disabled configurations preserve the original unsteered namespace.
uint64_t ember_directional_steering_identity(
    const ember_directional_steering *steering, uint64_t seed);

// Host aggregate FFN seam for hybrid expert placement. The GPU graph uses
// the same formula but its own fixed-width reduction; not bitwise CPU parity.
static inline void ember_directional_steering_project(
    const ember_directional_steering *steering, int layer, bool ffn,
    float *values, size_t n_rows) {
    if (!steering || layer < 0 || layer >= EMBER_STEERING_LAYERS ||
        !steering->nonzero[layer]) return;
    const float scale = ffn ? steering->ffn_scale : steering->attn_scale;
    if (scale == 0.0f) return;
    const float *direction = steering->rows[layer];
    for (size_t row = 0; row < n_rows; ++row) {
        float *value = values + row * EMBER_STEERING_WIDTH;
        float dot = 0.0f;
        for (int i = 0; i < EMBER_STEERING_WIDTH; ++i)
            dot += value[i] * direction[i];
        const float coeff = scale * dot;
        for (int i = 0; i < EMBER_STEERING_WIDTH; ++i)
            value[i] -= coeff * direction[i];
    }
}

#ifdef __cplusplus
}
#endif
#endif
