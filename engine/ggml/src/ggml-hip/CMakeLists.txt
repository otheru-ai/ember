if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

if (NOT DEFINED CMAKE_HIP_FLAGS_DEBUG)
    set(CMAKE_HIP_FLAGS_DEBUG "-g -O2")
endif()

string(REGEX MATCH "hipcc$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")

if (CXX_IS_HIPCC)
    if (LINUX)
        if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
            message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
        endif()

        message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
                " Prefer setting the HIP compiler directly. See README for details.")
    endif()
else()
    # Forward (AMD)GPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
endif()

find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

if (${hip_VERSION} VERSION_LESS 6.1)
    message(FATAL_ERROR "At least ROCM/HIP V6.1 is required")
endif()

message(STATUS "HIP and hipBLAS found")

file(GLOB   GGML_HEADERS_ROCM "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_ROCM "../../include/ggml-cuda.h")

file(GLOB   GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})

if (GGML_CUDA_FA_ALL_QUANTS)
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
    add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
else()
    list(APPEND GGML_SOURCES_ROCM
        ../ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu)
endif()

ggml_add_backend_library(ggml-hip
                         ${GGML_HEADERS_ROCM}
                         ${GGML_SOURCES_ROCM}
                        )

# The HIP substrate intentionally carries generic quant/template helpers that
# gfx1151 does not instantiate. Suppress only those translation-unit-local
# warnings; keep warnings for live variables and operations actionable.
target_compile_options(ggml-hip PRIVATE
    -Wno-sign-compare
    -Wno-unused-function
    -Wno-unneeded-internal-declaration)

# EMBER FORK DIVERGENCE (engine/VENDOR.md): per-translation-unit CU mode.
#
# RDNA3.5 runs workgroups in WGP mode by default: two CUs share one 128 KB LDS.
# -mcumode switches to one CU per workgroup with 64 KB LDS. Which is faster is
# not a property of the architecture, it is a property of the kernel -- so this
# is applied per TU, from measurement, not globally.
#
# MEASURED on gfx1151 (prefill probe, production quiesced, whole-build -mcumode
# vs default, kernel times in ms):
#
#   rope_norm                          749.9 -> 377.9   -49.6%   <- CU mode
#   ds4_flash_attn_d512 (compact)     2457.2 -> 2334.3   -5.0%   <- CU mode
#   ds4_flash_attn_d512 (grouped)     5495.3 -> 5433.0   -1.1%   <- CU mode
#   mul_mat_q<Q2_0_ROCMFP2,64>        5361.3 -> 5561.7   +3.7%   <- keep WGP
#   mul_mat_q<Q4_0_ROCMFP4_FAST,64>   2462.5 -> 2604.5   +5.8%   <- keep WGP
#
# The split is what the LDS budget predicts: the MMQ tiles want the full 128 KB
# a WGP provides and lose occupancy when halved, while rope and the attention
# kernels use little LDS and gain from not sharing a CU pair. A whole-build
# -mcumode was net -1.3% because those two effects nearly cancel; selecting per
# TU keeps only the wins.
#
# Do NOT widen this list without re-measuring: mmq*.cu regressed, and the same
# is likely for any kernel whose workgroup wants more than 64 KB of LDS.
# Net effect per TU, summed over every kernel that TU owns (prefill probe):
#   ds4-indexer.cu   561.8 ->  472.3 ms  -15.9%  (4 kernels)
#   ds4-hc.cu       1028.4 ->  998.4 ms   -2.9%  (5 kernels)
#   cpy.cu           423.2 ->  416.5 ms   -1.6%  (4 kernels)  -- left out, near noise
#   norm.cu         1052.9 -> 1055.0 ms   +0.2%  (4 kernels)  -- regresses, excluded
# The TU is the unit of decision, not the kernel: -mcumode cannot be applied to
# one kernel inside a file, so a TU only qualifies if its kernels net out ahead.
set(EMBER_CUMODE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cuda/rope.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cuda/fattn.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cuda/ds4-indexer.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cuda/ds4-hc.cu")
foreach(src IN LISTS EMBER_CUMODE_SOURCES)
    if (NOT EXISTS "${src}")
        message(FATAL_ERROR "EMBER_CUMODE_SOURCES names a file that does not exist: ${src}")
    endif()
endforeach()
set_source_files_properties(${EMBER_CUMODE_SOURCES}
    PROPERTIES COMPILE_OPTIONS "-mcumode")

# TODO: upstream's HIP substrate still uses the CUDA-compatible public ABI.
target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)

add_compile_definitions(GGML_USE_HIP)

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

# Ember is a single-APU engine. Keep the shared HIP kernel source's peer-copy
# paths compiled out rather than exposing upstream's multi-GPU switch.
add_compile_definitions(GGML_CUDA_NO_PEER_COPY)

if (GGML_HIP_GRAPHS)
    add_compile_definitions(GGML_HIP_GRAPHS)
endif()

if (GGML_HIP_NO_VMM)
    add_compile_definitions(GGML_HIP_NO_VMM)
endif()

if (GGML_HIP_ROCMI4_W4A4)
    add_compile_definitions(GGML_ROCMI4_W4A4=1)
    message(WARNING
        "GGML_HIP_ROCMI4_W4A4 is a lossy, experimental ROCmI4 prefill path "
        "verified only on gfx1151. Exact MMQ is used on other architectures.")
endif()

if (GGML_HIP_ROCMI4_W4A8_IU4)
    if (GGML_HIP_ROCMI4_W4A4)
        message(FATAL_ERROR
            "GGML_HIP_ROCMI4_W4A8_IU4 preserves q8 activations and cannot be "
            "combined with the lossy GGML_HIP_ROCMI4_W4A4 quantizer")
    endif()
    add_compile_definitions(GGML_ROCMI4_W4A8_IU4=1)
    message(STATUS
        "Compiling exact gfx1151 ROCmI4 W4A8 IU4 MMQ experiment; runtime "
        "selection additionally requires DFLASH_ROCMI4_W4A8_IU4=1")
endif()

if (GGML_HIP_ROCWMMA_FATTN)
    add_compile_definitions(GGML_HIP_ROCWMMA_FATTN)
endif()

if (NOT GGML_HIP_MMQ_MFMA)
    add_compile_definitions(GGML_HIP_NO_MMQ_MFMA)
endif()

if (GGML_HIP_EXPORT_METRICS)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage --save-temps")
endif()

if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()

if (CXX_IS_HIPCC)
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
    target_link_libraries(ggml-hip PRIVATE hip::device)
else()
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
endif()

if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
endif()

target_link_libraries(ggml-hip PRIVATE ggml-base hip::host roc::rocblas roc::hipblas)

# Keep -ffast-math and -funsafe-math-optimizations off for HIP. ROCmFPX
# 8e6277f measured that reassociated reductions can flip greedy argmax on
# RDNA3.5/gfx1151 and make speculative decode diverge; finite-math assumptions
# also changed an infinity into NaN. Correctness is worth the measured ~2%.
