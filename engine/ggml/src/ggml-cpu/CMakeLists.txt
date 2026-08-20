# Ember vendors only the Linux x86-64 CPU companion backend used beside the
# gfx1151 HIP backend. The upstream multi-architecture and optional library
# branches were intentionally removed; see engine/VENDOR.md.

function(ggml_add_cpu_backend_variant_impl tag_name)
    if (tag_name)
        message(FATAL_ERROR "Ember does not build CPU backend variants")
    endif()

    set(GGML_CPU_NAME ggml-cpu)
    ggml_add_backend_library(${GGML_CPU_NAME})

    set(GGML_CPU_SOURCES
        ggml-cpu/ggml-cpu.c
        ggml-cpu/ggml-cpu.cpp
        ggml-cpu/repack.cpp
        ggml-cpu/repack.h
        ggml-cpu/hbm.cpp
        ggml-cpu/hbm.h
        ggml-cpu/quants.c
        ggml-cpu/quants.h
        ggml-cpu/traits.cpp
        ggml-cpu/traits.h
        ggml-cpu/ggml-cpu-impl.h
        ggml-cpu/common.h
        ggml-cpu/binary-ops.h
        ggml-cpu/binary-ops.cpp
        ggml-cpu/unary-ops.h
        ggml-cpu/unary-ops.cpp
        ggml-cpu/simd-mappings.h
        ggml-cpu/vec.h
        ggml-cpu/vec.cpp
        ggml-cpu/ops.h
        ggml-cpu/ops.cpp
        ggml-cpu/arch/x86/quants.c
        ggml-cpu/arch/x86/repack.cpp)

    target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)
    target_include_directories(${GGML_CPU_NAME} PRIVATE . ggml-cpu)

    if (GGML_OPENMP)
        find_package(OpenMP)
        if (OpenMP_FOUND)
            set(GGML_OPENMP_ENABLED "ON" CACHE INTERNAL "")
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_OPENMP)
            target_link_libraries(${GGML_CPU_NAME} PRIVATE
                OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
        else()
            set(GGML_OPENMP_ENABLED "OFF" CACHE INTERNAL "")
            message(WARNING "OpenMP not found")
        endif()
    endif()

    if (GGML_NATIVE)
        list(APPEND ARCH_FLAGS -march=native)
    else()
        if (GGML_SSE42)
            list(APPEND ARCH_FLAGS -msse4.2)
            list(APPEND ARCH_DEFINITIONS GGML_SSE42)
        endif()
        if (GGML_F16C)
            list(APPEND ARCH_FLAGS -mf16c)
            list(APPEND ARCH_DEFINITIONS GGML_F16C)
        endif()
        if (GGML_FMA)
            list(APPEND ARCH_FLAGS -mfma)
            list(APPEND ARCH_DEFINITIONS GGML_FMA)
        endif()
        if (GGML_BMI2)
            list(APPEND ARCH_FLAGS -mbmi2)
            list(APPEND ARCH_DEFINITIONS GGML_BMI2)
        endif()
        if (GGML_AVX)
            list(APPEND ARCH_FLAGS -mavx)
            list(APPEND ARCH_DEFINITIONS GGML_AVX)
        endif()
        if (GGML_AVX2)
            list(APPEND ARCH_FLAGS -mavx2)
            list(APPEND ARCH_DEFINITIONS GGML_AVX2)
        endif()
        if (GGML_AVX_VNNI)
            list(APPEND ARCH_FLAGS -mavxvnni)
            list(APPEND ARCH_DEFINITIONS GGML_AVX_VNNI)
        endif()
        if (GGML_AVX512)
            list(APPEND ARCH_FLAGS -mavx512f -mavx512cd -mavx512vl
                                   -mavx512dq -mavx512bw)
            list(APPEND ARCH_DEFINITIONS GGML_AVX512)
        endif()
        if (GGML_AVX512_VBMI)
            list(APPEND ARCH_FLAGS -mavx512vbmi)
            list(APPEND ARCH_DEFINITIONS GGML_AVX512_VBMI)
        endif()
        if (GGML_AVX512_VNNI)
            list(APPEND ARCH_FLAGS -mavx512vnni)
            list(APPEND ARCH_DEFINITIONS GGML_AVX512_VNNI)
        endif()
        if (GGML_AVX512_BF16)
            list(APPEND ARCH_FLAGS -mavx512bf16)
            list(APPEND ARCH_DEFINITIONS GGML_AVX512_BF16)
        endif()
    endif()

    if (GGML_CPU_REPACK)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_REPACK)
    endif()

    message(STATUS "Adding gfx1151 companion CPU backend: ${ARCH_FLAGS} ${ARCH_DEFINITIONS}")
    target_sources(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_SOURCES})
    target_compile_options(${GGML_CPU_NAME} PRIVATE ${ARCH_FLAGS})
    target_compile_definitions(${GGML_CPU_NAME} PRIVATE ${ARCH_DEFINITIONS})
endfunction()
