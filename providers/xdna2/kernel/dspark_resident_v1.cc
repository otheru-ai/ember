// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define NOCPP

#include <aie_api/aie.hpp>

// The resident overlay shares one 2,560-byte output packet between expert and
// projection modes. Projection GEMM produces 1,280 bytes of f32 values; copy
// them into the first half of the common DMA object. The host ignores the
// expert-only tail in projection mode.
extern "C" void store_projection_resident_f32x5(
        const float * __restrict input,
        bfloat16 * __restrict packet) {
    float * output = reinterpret_cast<float *>(packet);
    for (unsigned lane = 0; lane < 320; lane += 16) {
        aie::store_v(output + lane, aie::load_v<16>(input + lane));
    }
}
