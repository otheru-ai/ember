// Bounded decoding for request-owned inline images.
//
// Remote HTTP fetching is deliberately not performed by the unauthenticated
// inference server: it would create an SSRF surface and ambiguous timeout/
// redirect policy. Qwen vision accepts base64 data URLs here; other transports
// fail with a distinct error so images can never be silently discarded.

#ifndef EMBER_IMAGE_INPUT_H
#define EMBER_IMAGE_INPUT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *data;
    size_t   size;
} ember_image_bytes;

bool ember_image_data_url_decode(const char *url, ember_image_bytes *out,
                                 char *error, size_t error_cap);
void ember_image_bytes_free(ember_image_bytes *image);

#endif
