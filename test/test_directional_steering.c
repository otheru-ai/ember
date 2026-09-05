#include "model/directional_steering.h"
#include "backend/ember_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_pass, g_fail;
#define CHECK(c, m) do { if (c) ++g_pass; else { ++g_fail; fprintf(stderr, "FAIL: %s\n", m); } } while (0)

static void write_file(const char *path, const void *data, size_t bytes) {
    FILE *file = fopen(path, "wb");
    if (!file || fwrite(data, 1, bytes, file) != bytes || fclose(file) != 0) abort();
}

int main(void) {
    char path[] = "/tmp/ember-steering-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 1;
    close(fd);
    const size_t bytes = EMBER_STEERING_LAYERS * EMBER_STEERING_WIDTH * 4u;
    unsigned char *payload = calloc(1, bytes + 1);
    float *values = calloc(2 * EMBER_STEERING_WIDTH, sizeof(float));
    if (!payload || !values) abort();
    ember_directional_steering *s = NULL;
    char error[160];
    CHECK(ember_directional_steering_load(NULL, 0, 0, &s, error, sizeof(error)) && !s,
          "default disabled");
    CHECK(!ember_directional_steering_load(NULL, 0, 1, &s, error, sizeof(error)),
          "active scale requires file");
    CHECK(!ember_directional_steering_load(path, NAN, 0, &s, error, sizeof(error)),
          "nonfinite scale rejected");
    CHECK(!ember_directional_steering_load(path, 0, INFINITY, &s, NULL, 0),
          "nonfinite FFN scale rejected without error storage");
    CHECK(!ember_directional_steering_load(path, 101, 0, &s, error, sizeof(error)),
          "out of range scale rejected");
    CHECK(!ember_directional_steering_load(path, 0, -101, &s, error, sizeof(error)),
          "negative out of range scale rejected");
    CHECK(!ember_directional_steering_load(path, 0, 0, NULL, error, sizeof(error)),
          "missing output rejected");
    CHECK(!ember_directional_steering_load(path, 0, 0, &s, error, sizeof(error)),
          "empty file rejected even when disabled");
    write_file(path, payload, bytes - 1);
    CHECK(!ember_directional_steering_load(path, 0, 1, &s, error, sizeof(error)),
          "truncated file rejected");
    write_file(path, payload, bytes + 1);
    CHECK(!ember_directional_steering_load(path, 0, 1, &s, error, sizeof(error)),
          "trailing byte rejected");
    write_file(path, payload, bytes);
    CHECK(ember_directional_steering_load(path, 1, 1, &s, error, sizeof(error)) && !s,
          "all-zero directions have no effect");
    // Little-endian +Inf; validates byte order independently of host floats.
    payload[2] = 0x80; payload[3] = 0x7f;
    write_file(path, payload, bytes);
    CHECK(!ember_directional_steering_load(path, 0, 0, &s, error, sizeof(error)),
          "nonfinite directions rejected even when disabled");
    // Layer 39, channel 0 = 1; all remaining channels/layers excluded.
    payload[2] = payload[3] = 0;
    const size_t offset = 39u * EMBER_STEERING_WIDTH * 4u;
    payload[offset + 2] = 0x80; payload[offset + 3] = 0x3f;
    write_file(path, payload, bytes);
    CHECK(ember_directional_steering_load(path, 0, 0, &s, error, sizeof(error)) && !s,
          "zero scales discard validated payload");
    CHECK(ember_directional_steering_load(path, 0, 3.5f, &s, error, sizeof(error)) && s,
          "finite direction loaded");
    if (!s) abort();
    CHECK(s->rows[39][0] == 1 && s->nonzero[39] && !s->nonzero[38],
          "layer-major layout and excluded bands");
    values[0] = 2; values[1] = 5; values[EMBER_STEERING_WIDTH] = 4;
    ember_directional_steering_project(s, 39, false, values, 2);
    ember_directional_steering_project(s, 38, true, values, 2);
    ember_directional_steering_project(NULL, 39, true, values, 2);
    ember_directional_steering_project(s, -1, true, values, 2);
    ember_directional_steering_project(s, 43, true, values, 2);
    CHECK(values[0] == 2 && values[EMBER_STEERING_WIDTH] == 4,
          "zero scale, absent policy and excluded rows preserve inputs");
    ember_directional_steering_project(s, 39, true, values, 2);
    CHECK(values[0] == -5 && values[1] == 5 && values[EMBER_STEERING_WIDTH] == -10,
          "scale 3.5 changes each row with correct sign and leaves orthogonal channel");
    const uint64_t seed = UINT64_C(1469598103934665603);
    const uint64_t identity = ember_directional_steering_identity(s, seed);
    CHECK(ember_directional_steering_identity(NULL, seed) == seed && identity != seed,
          "active and disabled cache namespaces differ");
    ember_directional_steering *same = NULL;
    CHECK(ember_directional_steering_load(path, 0, 3.5f, &same, error, sizeof(error)),
          "reload same content");
    CHECK(ember_directional_steering_identity(same, seed) == identity,
          "same content and policy retain identity");
    free(same);
    CHECK(ember_directional_steering_load(path, 1, 3.5f, &same, error, sizeof(error)),
          "load different scales");
    CHECK(ember_directional_steering_identity(same, seed) != identity,
          "scale change isolates cache");
    free(same);
    payload[offset + 3] = 0xbf; // -1; projection equal but content distinct.
    write_file(path, payload, bytes);
    CHECK(ember_directional_steering_load(path, 0, 3.5f, &same, error, sizeof(error)),
          "load replacement at same pathname and size");
    CHECK(ember_directional_steering_identity(same, seed) != identity,
          "content change isolates cache even at same path and size");
    CHECK(s->rows[39][0] == 1 && ember_directional_steering_identity(s, seed) == identity,
          "loaded policy immutable when source file is replaced");
    free(same);
    free(s);
    // Direct ABI clients receive the same validation as CLI clients.
    ember_backend_config cfg = {0};
    cfg.dir_steering_file = path;
    cfg.dir_steering_ffn = 3.5f;
    char *abi_error = NULL;
    ember_backend *backend = ember_backend_load(&cfg, &abi_error);
    CHECK(backend != NULL && !abi_error, "stub accepts valid policy");
    ember_backend_free(backend);
    unlink(path);
    CHECK(!ember_directional_steering_load(path, 0, 1, &s, error, sizeof(error)),
          "missing file rejected");
    backend = ember_backend_load(&cfg, &abi_error);
    CHECK(!backend && abi_error, "stub rejects missing direction file");
    free(abi_error);
    free(payload);
    free(values);
    printf("directional steering: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
