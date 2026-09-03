// Bounded decoding for request-owned inline images.
//
// Remote HTTP fetching is deliberately not performed by the unauthenticated
// inference server: it would create an SSRF surface and ambiguous timeout/
// redirect policy. Vision accepts base64 data URLs here; other transports
// fail with a distinct error so images can never be silently discarded.

#ifndef EMBER_IMAGE_INPUT_H
#define EMBER_IMAGE_INPUT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *data;
    size_t   size;
    // Normalized media identity. The decoder accepts only these explicit
    // image data-URL prefixes; callers never have to retain the source URL.
    enum {
        EMBER_IMAGE_PNG = 1,
        EMBER_IMAGE_JPEG,
        EMBER_IMAGE_WEBP,
        EMBER_IMAGE_GIF,
    } format;
} ember_image_bytes;

bool ember_image_data_url_decode(const char *url, ember_image_bytes *out,
                                 char *error, size_t error_cap);
void ember_image_bytes_free(ember_image_bytes *image);

#endif
