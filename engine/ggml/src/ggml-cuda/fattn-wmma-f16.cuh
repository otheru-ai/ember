#pragma once

#include "common.cuh"

#if defined(GGML_HIP_ROCWMMA_FATTN)
#if defined(RDNA3)
#define GGML_USE_WMMA_FATTN
#endif // defined(RDNA3)
#endif // defined(GGML_HIP_ROCWMMA_FATTN)

// WMMA flash attention requires FP16 matrix instructions to be available for ggml code.
static bool ggml_cuda_should_use_wmma_fattn(const int cc) {
#if defined(GGML_USE_HIP) && !defined(GGML_HIP_ROCWMMA_FATTN)
    return false;
#else
    // gfx1151 is RDNA3_5; the NVIDIA/Volta and CDNA arms cannot be reached here.
    return GGML_CUDA_CC_IS_GFX1151(cc);
#endif // defined(GGML_USE_HIP) && !defined(GGML_HIP_ROCWMMA_FATTN)
}

void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
