#include "image_input.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EMBER_IMAGE_MAX_BYTES (32u * 1024u * 1024u)

static int base64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static bool supported_prefix(const char *url, const char **payload) {
    static const char *prefixes[] = {
        "data:image/png;base64,", "data:image/jpeg;base64,",
        "data:image/webp;base64,", "data:image/gif;base64,",
    };
    for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); ++i) {
        const size_t n = strlen(prefixes[i]);
        if (strncmp(url, prefixes[i], n) == 0) {
            *payload = url + n;
            return true;
        }
    }
    return false;
}

bool ember_image_data_url_decode(const char *url, ember_image_bytes *out,
                                 char *error, size_t error_cap) {
    if (out) *out = (ember_image_bytes){0};
    if (!url || !out) {
        if (error && error_cap) snprintf(error, error_cap, "invalid image URL");
        return false;
    }
    const char *payload = NULL;
    if (!supported_prefix(url, &payload)) {
        if (error && error_cap)
            snprintf(error, error_cap,
                     "only base64 PNG/JPEG/WebP/GIF data URLs are supported; remote image fetching is disabled");
        return false;
    }
    const size_t n = strlen(payload);
    if (n == 0 || n % 4 != 0 || n / 4 > EMBER_IMAGE_MAX_BYTES / 3 + 1) {
        if (error && error_cap) snprintf(error, error_cap, "invalid or oversized image base64 payload");
        return false;
    }
    size_t padding = 0;
    if (n && payload[n - 1] == '=') ++padding;
    if (n > 1 && payload[n - 2] == '=') ++padding;
    const size_t decoded = n / 4 * 3 - padding;
    if (decoded == 0 || decoded > EMBER_IMAGE_MAX_BYTES) {
        if (error && error_cap) snprintf(error, error_cap, "decoded image exceeds 32 MiB limit");
        return false;
    }
    uint8_t *bytes = (uint8_t *)malloc(decoded);
    if (!bytes) abort();
    size_t o = 0;
    for (size_t i = 0; i < n; i += 4) {
        int v[4];
        for (int j = 0; j < 4; ++j) {
            const unsigned char c = (unsigned char)payload[i + (size_t)j];
            v[j] = c == '=' ? 0 : base64_value(c);
            if (v[j] < 0 || (c == '=' && i + 4 != n) ||
                (c == '=' && j < 2)) {
                free(bytes);
                if (error && error_cap) snprintf(error, error_cap, "invalid image base64 payload");
                return false;
            }
        }
        if (payload[i + 2] == '=' && payload[i + 3] != '=') {
            free(bytes);
            if (error && error_cap) snprintf(error, error_cap, "invalid image base64 padding");
            return false;
        }
        const uint32_t word = (uint32_t)v[0] << 18 |
                              (uint32_t)v[1] << 12 |
                              (uint32_t)v[2] << 6 | (uint32_t)v[3];
        if (o < decoded) bytes[o++] = (uint8_t)(word >> 16);
        if (o < decoded) bytes[o++] = (uint8_t)(word >> 8);
        if (o < decoded) bytes[o++] = (uint8_t)word;
    }
    if (o != decoded) {
        free(bytes);
        if (error && error_cap) snprintf(error, error_cap, "invalid image base64 length");
        return false;
    }
    out->data = bytes;
    out->size = decoded;
    return true;
}

void ember_image_bytes_free(ember_image_bytes *image) {
    if (!image) return;
    free(image->data);
    *image = (ember_image_bytes){0};
}
