#include "qwen_yarn.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static void set_error(char *error, size_t error_size, const char *message) {
    if (!error || error_size == 0) return;
    (void)snprintf(error, error_size, "%s", message ? message : "");
}

static float correction_dim(float rotations, float theta) {
    const float two_pi = 6.28318530717958647692f;
    return (float)EMBER_QWEN_ROPE_DIM *
           logf((float)EMBER_QWEN_NATIVE_CONTEXT /
                (rotations * two_pi)) /
           (2.0f * logf(theta));
}

bool ember_qwen_yarn_configure(bool enable_yarn, int32_t max_context,
                               ember_qwen_yarn_config *config,
                               char *error, size_t error_size) {
    if (!config) {
        set_error(error, error_size, "Qwen YaRN configuration output is null");
        return false;
    }
    memset(config, 0, sizeof(*config));
    set_error(error, error_size, "");
    if (max_context < 1) {
        set_error(error, error_size, "Qwen context must be positive");
        return false;
    }
    if (!enable_yarn && max_context > EMBER_QWEN_NATIVE_CONTEXT) {
        set_error(error, error_size,
                  "Qwen context above 262144 requires explicit --qwen-yarn");
        return false;
    }
    if (enable_yarn && max_context != EMBER_QWEN_YARN_MAX_CONTEXT) {
        set_error(error, error_size,
                  "--qwen-yarn requires the official --max-ctx 1000000 recipe");
        return false;
    }

    config->enabled = enable_yarn;
    config->original_context = EMBER_QWEN_NATIVE_CONTEXT;
    config->max_context = max_context;
    config->factor = enable_yarn ? 4.0f : 1.0f;
    config->theta = 10000000.0f;
    config->beta_fast = 32.0f;
    config->beta_slow = 1.0f;
    config->attention_factor = enable_yarn
        ? 1.0f + 0.1f * logf(config->factor)
        : 1.0f;
    config->correction_low = enable_yarn
        ? (int32_t)floorf(correction_dim(config->beta_fast, config->theta))
        : 0;
    config->correction_high = enable_yarn
        ? (int32_t)ceilf(correction_dim(config->beta_slow, config->theta))
        : 0;
    if (config->correction_low < 0) config->correction_low = 0;
    if (config->correction_high >= EMBER_QWEN_ROPE_DIM)
        config->correction_high = EMBER_QWEN_ROPE_DIM - 1;
    config->mrope_sections[0] = 11;
    config->mrope_sections[1] = 11;
    config->mrope_sections[2] = 10;
    return true;
}

void ember_qwen_yarn_inv_freq(const ember_qwen_yarn_config *config,
                              float out[EMBER_QWEN_ROPE_FREQ_COUNT]) {
    if (!config || !out) return;
    for (int32_t index = 0; index < EMBER_QWEN_ROPE_FREQ_COUNT; ++index) {
        const float exponent = (float)(2 * index) /
                               (float)EMBER_QWEN_ROPE_DIM;
        const float base_frequency = 1.0f / powf(config->theta, exponent);
        if (!config->enabled) {
            out[index] = base_frequency;
            continue;
        }
        const float denominator =
            (float)(config->correction_high - config->correction_low);
        float ramp = denominator == 0.0f
            ? 1.0f
            : ((float)index - (float)config->correction_low) / denominator;
        if (ramp < 0.0f) ramp = 0.0f;
        if (ramp > 1.0f) ramp = 1.0f;
        const float extrapolation = 1.0f - ramp;
        const float interpolation_frequency = base_frequency / config->factor;
        out[index] = interpolation_frequency * (1.0f - extrapolation) +
                     base_frequency * extrapolation;
    }
}

static int32_t mrope_axis_for_frequency(
        const ember_qwen_yarn_config *config, int32_t index) {
    if (index % 3 == 1 && index < 3 * config->mrope_sections[1]) return 1;
    if (index % 3 == 2 && index < 3 * config->mrope_sections[2]) return 2;
    return 0;
}

void ember_qwen_yarn_cos_sin(const ember_qwen_yarn_config *config,
                             const int32_t positions[3],
                             float cos_out[EMBER_QWEN_ROPE_DIM],
                             float sin_out[EMBER_QWEN_ROPE_DIM]) {
    if (!config || !positions || !cos_out || !sin_out) return;
    float inverse[EMBER_QWEN_ROPE_FREQ_COUNT];
    ember_qwen_yarn_inv_freq(config, inverse);
    for (int32_t index = 0; index < EMBER_QWEN_ROPE_FREQ_COUNT; ++index) {
        const int32_t axis = mrope_axis_for_frequency(config, index);
        const float angle = (float)positions[axis] * inverse[index];
        const float cosine = cosf(angle) * config->attention_factor;
        const float sine = sinf(angle) * config->attention_factor;
        cos_out[index] = cosine;
        cos_out[index + EMBER_QWEN_ROPE_FREQ_COUNT] = cosine;
        sin_out[index] = sine;
        sin_out[index + EMBER_QWEN_ROPE_FREQ_COUNT] = sine;
    }
}

bool ember_qwen_yarn_apply(float *head, size_t head_dim,
                           const ember_qwen_yarn_config *config,
                           const int32_t positions[3]) {
    if (!head || head_dim < (size_t)EMBER_QWEN_ROPE_DIM ||
        !config || !positions)
        return false;
    float cosines[EMBER_QWEN_ROPE_DIM];
    float sines[EMBER_QWEN_ROPE_DIM];
    ember_qwen_yarn_cos_sin(config, positions, cosines, sines);
    float original[EMBER_QWEN_ROPE_DIM];
    memcpy(original, head, sizeof(original));
    for (size_t index = 0; index < EMBER_QWEN_ROPE_FREQ_COUNT; ++index) {
        head[index] = original[index] * cosines[index] -
                      original[index + EMBER_QWEN_ROPE_FREQ_COUNT] *
                          sines[index];
        head[index + EMBER_QWEN_ROPE_FREQ_COUNT] =
            original[index + EMBER_QWEN_ROPE_FREQ_COUNT] *
                cosines[index] +
            original[index] * sines[index];
    }
    return true;
}
