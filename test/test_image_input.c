#include "model/image_input.h"

#include <stdio.h>
#include <string.h>

static int g_pass, g_fail;
#define CHECK(c, m) do { if (c) ++g_pass; else { ++g_fail; fprintf(stderr, "FAIL: %s\n", m); } } while (0)

int main(void) {
    ember_image_bytes image = {0};
    char error[160] = {0};
    CHECK(ember_image_data_url_decode("data:image/png;base64,iVBORw==",
                                      &image, error, sizeof(error)),
          "valid PNG data URL decodes");
    CHECK(image.size == 4 && memcmp(image.data, "\x89PNG", 4) == 0,
          "decoded bytes are exact");
    ember_image_bytes_free(&image);
    CHECK(!ember_image_data_url_decode("https://example.test/a.png", &image,
                                       error, sizeof(error)) &&
          strstr(error, "remote image fetching is disabled") != NULL,
          "remote URL fails closed with transport reason");
    CHECK(!ember_image_data_url_decode("data:image/png;base64,A===", &image,
                                       error, sizeof(error)),
          "malformed padding is rejected");
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
