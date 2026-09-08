#pragma once

#include <stdbool.h>
#include <string.h>

// HIP may append target features (e.g. :xnack-) to the architecture name.
static inline bool ggml_hip_is_gfx1151(const char *arch) {
    return arch && strncmp(arch, "gfx1151", 7) == 0 &&
           (arch[7] == '\0' || arch[7] == ':');
}
