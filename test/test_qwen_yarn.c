#include "qwen_yarn.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_pass;
static int g_fail;

#define CHECK(cond, msg) do {                                                \
    if (cond) { ++g_pass; }                                                  \
    else { ++g_fail; fprintf(stderr, "FAIL: %s\n", (msg)); }                \
} while (0)

static int nearly(float actual, float expected, float tolerance) {
    return isfinite(actual) && fabsf(actual - expected) <= tolerance;
}

static void test_policy(void) {
    ember_qwen_yarn_config cfg;
    char error[160];
    CHECK(ember_qwen_yarn_configure(
              false, EMBER_QWEN_NATIVE_CONTEXT, &cfg, error, sizeof(error)),
          "native context accepted without YaRN");
    CHECK(!cfg.enabled && cfg.factor == 1.0f &&
              cfg.max_context == EMBER_QWEN_NATIVE_CONTEXT,
          "native policy resolves ordinary RoPE");
    CHECK(!ember_qwen_yarn_configure(
              false, EMBER_QWEN_NATIVE_CONTEXT + 1, &cfg, error,
              sizeof(error)) && strstr(error, "--qwen-yarn") != NULL,
          "extension fails closed without opt-in");
    CHECK(!ember_qwen_yarn_configure(
              true, EMBER_QWEN_NATIVE_CONTEXT, &cfg, error, sizeof(error)),
          "static YaRN is rejected when no extension is requested");
    CHECK(!ember_qwen_yarn_configure(
              true, 524288, &cfg, error, sizeof(error)),
          "factor-four YaRN is not silently used for a factor-two context");
    CHECK(!ember_qwen_yarn_configure(
              true, EMBER_QWEN_YARN_MAX_CONTEXT + 1, &cfg, error,
              sizeof(error)),
          "context beyond official 1M recipe rejected");
    CHECK(ember_qwen_yarn_configure(
              true, EMBER_QWEN_YARN_MAX_CONTEXT, &cfg, error, sizeof(error)),
          "official 1M YaRN recipe accepted");
    CHECK(cfg.enabled && cfg.factor == 4.0f &&
              cfg.original_context == EMBER_QWEN_NATIVE_CONTEXT &&
              cfg.correction_low == 14 && cfg.correction_high == 22,
          "official static YaRN scalar policy exact");
    CHECK(nearly(cfg.attention_factor, 1.1386294361f, 1.0e-6f),
          "YaRN default attention scaling exact");
    CHECK(cfg.mrope_sections[0] == 11 && cfg.mrope_sections[1] == 11 &&
              cfg.mrope_sections[2] == 10,
          "Qwen interleaved M-RoPE sections exact");
}

static void test_frequency_boundaries(void) {
    ember_qwen_yarn_config cfg;
    float inv[EMBER_QWEN_ROPE_FREQ_COUNT];
    CHECK(ember_qwen_yarn_configure(
              true, EMBER_QWEN_YARN_MAX_CONTEXT, &cfg, NULL, 0),
          "build frequency test config");
    ember_qwen_yarn_inv_freq(&cfg, inv);
    const struct {
        int index;
        float expected;
    } pinned[] = {
        {0, 1.0f},
        {13, 0.0014330126577988267f},
        {14, 0.0008659643353894353f},
        {15, 0.00047423981595784426f},
        {16, 0.00025693507632240653f},
        {21, 0.000008759770935285792f},
        {22, 0.000003849816039291909f},
        {23, 0.000002326430149190128f},
        {31, 0.00000004137042708340921f},
    };
    int pinned_match = 1;
    for (size_t i = 0; i < sizeof(pinned) / sizeof(pinned[0]); ++i) {
        pinned_match = pinned_match &&
            nearly(inv[pinned[i].index], pinned[i].expected, 2.0e-11f);
    }
    CHECK(pinned_match,
          "inverse frequencies match pinned Transformers float32 vector");
    const float native14 = 1.0f / powf(cfg.theta, 28.0f / 64.0f);
    const float native22 = 1.0f / powf(cfg.theta, 44.0f / 64.0f);
    CHECK(nearly(inv[14], native14, 2.0e-7f * native14),
          "correction low endpoint remains unscaled");
    CHECK(nearly(inv[22], native22 / 4.0f, 2.0e-7f * native22),
          "correction high endpoint is fully interpolated");
    CHECK(inv[15] < 1.0f / powf(cfg.theta, 30.0f / 64.0f) &&
              inv[15] > 1.0f / (4.0f * powf(cfg.theta, 30.0f / 64.0f)),
          "correction band blends extrapolation and interpolation");
}

static void test_mrope_and_apply(void) {
    ember_qwen_yarn_config cfg;
    float cosines[EMBER_QWEN_ROPE_DIM];
    float sines[EMBER_QWEN_ROPE_DIM];
    const int32_t positions[3] = {2, 3, 5};
    CHECK(ember_qwen_yarn_configure(
              true, EMBER_QWEN_YARN_MAX_CONTEXT, &cfg, NULL, 0),
          "build M-RoPE test config");
    ember_qwen_yarn_cos_sin(&cfg, positions, cosines, sines);
    CHECK(nearly(cosines[0], cosf(2.0f) * cfg.attention_factor, 1.0e-6f) &&
              nearly(cosines[1],
                     cosf(3.0f * powf(cfg.theta, -2.0f / 64.0f)) *
                         cfg.attention_factor,
                     1.0e-6f) &&
              nearly(cosines[2],
                     cosf(5.0f * powf(cfg.theta, -4.0f / 64.0f)) *
                         cfg.attention_factor,
                     1.0e-6f),
          "interleaved temporal/height/width axes selected");
    CHECK(cosines[0] == cosines[32] && sines[0] == sines[32] &&
              cosines[31] == cosines[63] && sines[31] == sines[63],
          "frequency vector duplicated into NeoX halves");

    float head[256];
    for (int i = 0; i < 256; ++i) head[i] = (float)i / 17.0f;
    float tail[192];
    memcpy(tail, head + 64, sizeof(tail));
    CHECK(ember_qwen_yarn_apply(head, 256, &cfg, positions),
          "scalar QSA head application accepted");
    CHECK(memcmp(tail, head + 64, sizeof(tail)) == 0,
          "partial RoPE leaves QSA dimensions 64..255 untouched");
    CHECK(!ember_qwen_yarn_apply(head, 63, &cfg, positions),
          "undersized head rejected");
}

int main(void) {
    test_policy();
    test_frequency_boundaries();
    test_mrope_and_apply();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
