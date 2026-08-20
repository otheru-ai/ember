// Snapshot backend selection utility.
//
// Prefix cache snapshots can be large (full KV cache copies). If a future
// gfx1151 allocation is not host-accessible, keeping it in device-only memory
// wastes the GPU aperture and can cause performance degradation.
//
// On Strix Halo's host-accessible HIP allocations the GPU buffer is backed by
// unified memory, so there is no benefit from a separate CPU backend.
//
// This header provides helpers to select the appropriate backend for
// snapshot storage based on the compute backend's memory properties.

#pragma once

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>

namespace dflash::common {

// Select or create a backend for prefix cache snapshot storage.
//
// If the compute backend's default buffer type is host-accessible, returns
// compute_backend itself since gfx1151 and the CPU share physical memory.
//
// Otherwise, allocates a CPU backend so snapshots reside in system RAM. This
// fallback preserves a clear failure mode if the HIP allocation policy changes.
//
// Returns nullptr on allocation failure (caller should treat as fatal).
inline ggml_backend_t create_snapshot_backend(ggml_backend_t compute_backend) {
    auto buft = ggml_backend_get_default_buffer_type(compute_backend);
    if (ggml_backend_buft_is_host(buft)) {
        // Unified memory — snapshots can stay on compute backend.
        return compute_backend;
    }
    // Discrete VRAM — allocate a CPU backend for snapshot storage.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "[snapshot] CPU backend init failed\n");
    }
    return cpu;
}

// Free the snapshot backend if it was separately allocated.
// Safe to call with nullptr. Does NOT free the compute backend.
inline void free_snapshot_backend(ggml_backend_t snap_backend,
                                  ggml_backend_t compute_backend) {
    if (snap_backend && snap_backend != compute_backend) {
        ggml_backend_free(snap_backend);
    }
}

}  // namespace dflash::common
