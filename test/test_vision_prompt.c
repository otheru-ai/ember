#include "model/vision_prompt.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass;
static int g_fail;

#define CHECK(condition, message) do {                                      \
    if (condition) ++g_pass;                                                \
    else { ++g_fail; fprintf(stderr, "FAIL: %s\n", message); }             \
} while (0)

static bool ids_equal(const int32_t *actual, const int32_t *expected, int n) {
    return n == 0 || memcmp(actual, expected,
                            (size_t)n * sizeof(*actual)) == 0;
}

static void test_multi_token_multiple_images(void) {
    const int32_t input[] = {10, 700, 701, 11, 700, 9, 700, 701, 12};
    const int32_t placeholder[] = {700, 701};
    const int32_t first[] = {1000, 1001, 1002};
    const int32_t second[] = {2000, 2001, 2002, 2003};
    const ember_vision_prompt_image images[] = {
        {first, 3}, {second, 4},
    };
    int32_t *output = NULL;
    int output_count = 0;
    int offsets[2] = {-1, -1};
    char error[160] = {0};
    const int32_t expected[] = {
        10, 1000, 1001, 1002, 11, 700, 9,
        2000, 2001, 2002, 2003, 12,
    };
    CHECK(ember_vision_prompt_expand(
              input, 9, placeholder, 2, images, 2,
              &output, &output_count, offsets, error, sizeof(error)),
          "two exact multi-token placeholders expand");
    CHECK(output_count == 12 && ids_equal(output, expected, 12),
          "only complete exact placeholder sequences are replaced");
    CHECK(offsets[0] == 1 && offsets[1] == 7,
          "second image offset uses the first image's expanded length");
    free(output);
}

static void test_legacy_single_token_fallback(void) {
    const int32_t input[] = {1, 42, 2};
    const int32_t placeholder[] = {42};
    const ember_vision_prompt_image image = {NULL, 3};
    int32_t *output = NULL;
    int count = 0;
    int offset = -1;
    const int32_t expected[] = {1, 42, 42, 42, 2};
    CHECK(ember_vision_prompt_expand(
              input, 3, placeholder, 1, &image, 1,
              &output, &count, &offset, NULL, 0) &&
              count == 5 && offset == 1 && ids_equal(output, expected, 5),
          "legacy Qwen run repeats its one image_pad token");
    free(output);
}

static void test_fail_closed_contracts(void) {
    const int32_t placeholder[] = {7, 8};
    const int32_t one[] = {1, 7, 8, 2};
    const int32_t replacement[] = {10, 11};
    ember_vision_prompt_image images[] = {
        {replacement, 2}, {replacement, 2},
    };
    int32_t *output = (int32_t *)(uintptr_t)1;
    int count = 99;
    int offsets[2] = {-1, -1};
    char error[160] = {0};
    CHECK(!ember_vision_prompt_expand(
              one, 4, placeholder, 2, images, 2,
              &output, &count, offsets, error, sizeof(error)) &&
              output == NULL && count == 0,
          "fewer placeholders than images clears partial output");

    const int32_t two[] = {7, 8, 3, 7, 8};
    CHECK(!ember_vision_prompt_expand(
              two, 5, placeholder, 2, images, 1,
              &output, &count, offsets, error, sizeof(error)) &&
              output == NULL && count == 0,
          "extra placeholders cannot be silently dropped");

    const ember_vision_prompt_image no_ids = {NULL, 2};
    CHECK(!ember_vision_prompt_expand(
              one, 4, placeholder, 2, &no_ids, 1,
              &output, &count, offsets, error, sizeof(error)),
          "multi-token placeholder cannot use a one-id fallback");

    const ember_vision_prompt_image overflow = {replacement, INT_MAX};
    CHECK(!ember_vision_prompt_expand(
              one, 4, placeholder, 2, &overflow, 1,
              &output, &count, offsets, error, sizeof(error)),
          "expanded prompt arithmetic fails before allocation overflow");

    const int32_t short_prompt[] = {7};
    memset(error, 0, sizeof(error));
    CHECK(!ember_vision_prompt_expand(
              short_prompt, 1, placeholder, 2, images, 1,
              &output, &count, offsets, error, sizeof(error)) &&
              strstr(error, "too short") != NULL,
          "short prompt reports a missing-placeholder diagnostic");
}

static void test_text_only_inert(void) {
    const int32_t input[] = {1, 2, 3};
    const int32_t placeholder[] = {7, 8};
    int32_t *output = NULL;
    int count = 0;
    CHECK(ember_vision_prompt_expand(
              input, 3, placeholder, 2, NULL, 0,
              &output, &count, NULL, NULL, 0) &&
              count == 3 && ids_equal(output, input, 3),
          "text-only prompt is copied byte-for-byte");
    free(output);
}

int main(void) {
    test_multi_token_multiple_images();
    test_legacy_single_token_fallback();
    test_fail_closed_contracts();
    test_text_only_inert();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
