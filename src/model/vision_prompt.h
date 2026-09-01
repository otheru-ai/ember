// Exact token-sequence replacement for request-owned vision rows.
//
// Chat templates render an architecture-specific private placeholder and the
// model tokenizer encodes the complete prompt once. DeepSeek's placeholder can
// span multiple tokens, so locating a guessed single ID would shift every
// learned row while still producing fluent output. This helper replaces exact
// tokenizer sequences in image order and reports post-expansion offsets.

#ifndef EMBER_VISION_PROMPT_H
#define EMBER_VISION_PROMPT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    // NULL means repeat placeholder_ids[0], supported only when the
    // architecture's placeholder is exactly one token (legacy Qwen).
    const int32_t *token_ids;
    int            n_tokens;
} ember_vision_prompt_image;

// Locate every complete placeholder in source-prompt order. Offsets refer to
// the unexpanded prompt; callers add the cumulative replacement-length delta
// before passing a position-dependent image to its encoder.
bool ember_vision_prompt_find_all(
    const int32_t *input_ids, int input_count,
    const int32_t *placeholder_ids, int placeholder_count,
    int expected_count, int *offsets, char *error, size_t error_cap);

// Find exactly one complete tokenizer placeholder sequence. Operator-only
// prepared-vision gates need its real token offset before assembling learned
// marker padding; guessing a single ID is not valid for DeepSeek's placeholder.
bool ember_vision_prompt_find_unique(
    const int32_t *input_ids, int input_count,
    const int32_t *placeholder_ids, int placeholder_count,
    int *offset, char *error, size_t error_cap);

// A recognition-only check can pass when image rows are inert. This three-arm
// predicate requires mutually exclusive A/B answers and a marker-free no-image
// control, so output must depend on which artifact was supplied.
bool ember_vision_outputs_discriminate(
    const char *output_a, const char *output_b, const char *output_control,
    const char *marker_a, const char *marker_b);

// Allocates *output_ids on success. `image_offsets` has image_count entries
// and receives each replacement's offset in the final expanded prompt.
bool ember_vision_prompt_expand(
    const int32_t *input_ids, int input_count,
    const int32_t *placeholder_ids, int placeholder_count,
    const ember_vision_prompt_image *images, int image_count,
    int32_t **output_ids, int *output_count, int *image_offsets,
    char *error, size_t error_cap);

#endif  // EMBER_VISION_PROMPT_H
