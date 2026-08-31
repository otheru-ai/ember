#include "vision_prompt.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char *error, size_t error_cap, const char *message) {
    if (error && error_cap) snprintf(error, error_cap, "%s", message);
}

static int find_sequence(const int32_t *input, int input_count, int start,
                         const int32_t *pattern, int pattern_count) {
    if (start < 0 || pattern_count <= 0 || start > input_count ||
        pattern_count > input_count - start) {
        return -1;
    }
    const int last = input_count - pattern_count;
    for (int i = start; i <= last; ++i) {
        if (memcmp(input + i, pattern,
                   (size_t)pattern_count * sizeof(*pattern)) == 0) {
            return i;
        }
    }
    return -1;
}

bool ember_vision_prompt_expand(
        const int32_t *input_ids, int input_count,
        const int32_t *placeholder_ids, int placeholder_count,
        const ember_vision_prompt_image *images, int image_count,
        int32_t **output_ids, int *output_count, int *image_offsets,
        char *error, size_t error_cap) {
    if (output_ids) *output_ids = NULL;
    if (output_count) *output_count = 0;
    if (!output_ids || !output_count || input_count < 0 ||
        placeholder_count <= 0 || image_count < 0 ||
        (input_count > 0 && !input_ids) || !placeholder_ids ||
        (image_count > 0 && (!images || !image_offsets))) {
        set_error(error, error_cap, "invalid vision prompt expansion contract");
        return false;
    }

    int expanded_count = input_count;
    for (int i = 0; i < image_count; ++i) {
        if (images[i].n_tokens <= 0 ||
            (!images[i].token_ids && placeholder_count != 1)) {
            set_error(error, error_cap,
                      "vision replacement token contract is invalid");
            return false;
        }
        if (expanded_count < placeholder_count ||
            images[i].n_tokens >
                INT_MAX - (expanded_count - placeholder_count)) {
            set_error(error, error_cap, "expanded vision prompt is too large");
            return false;
        }
        expanded_count += images[i].n_tokens - placeholder_count;
    }
    if ((size_t)expanded_count > SIZE_MAX / sizeof(int32_t)) {
        set_error(error, error_cap, "expanded vision prompt is too large");
        return false;
    }

    int32_t *expanded = (int32_t *)malloc(
        (size_t)(expanded_count > 0 ? expanded_count : 1) * sizeof(*expanded));
    if (!expanded) abort();
    int cursor = 0;
    int destination = 0;
    for (int i = 0; i < image_count; ++i) {
        const int found = find_sequence(
            input_ids, input_count, cursor,
            placeholder_ids, placeholder_count);
        if (found < 0) {
            free(expanded);
            set_error(error, error_cap,
                      "rendered prompt has fewer image placeholders than images");
            return false;
        }
        const int prefix = found - cursor;
        if (prefix > 0) {
            memcpy(expanded + destination, input_ids + cursor,
                   (size_t)prefix * sizeof(*expanded));
            destination += prefix;
        }
        image_offsets[i] = destination;
        if (images[i].token_ids) {
            memcpy(expanded + destination, images[i].token_ids,
                   (size_t)images[i].n_tokens * sizeof(*expanded));
        } else {
            for (int row = 0; row < images[i].n_tokens; ++row)
                expanded[destination + row] = placeholder_ids[0];
        }
        destination += images[i].n_tokens;
        cursor = found + placeholder_count;
    }

    if (find_sequence(input_ids, input_count, cursor,
                      placeholder_ids, placeholder_count) >= 0) {
        free(expanded);
        set_error(error, error_cap,
                  "rendered prompt has more image placeholders than images");
        return false;
    }
    const int tail = input_count - cursor;
    if (tail > 0) {
        memcpy(expanded + destination, input_ids + cursor,
               (size_t)tail * sizeof(*expanded));
        destination += tail;
    }
    if (destination != expanded_count) {
        free(expanded);
        set_error(error, error_cap, "vision prompt expansion length mismatch");
        return false;
    }
    *output_ids = expanded;
    *output_count = expanded_count;
    return true;
}
