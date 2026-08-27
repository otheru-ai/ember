cmake_minimum_required(VERSION 3.21)
project("ggml" C CXX ASM)

### GGML Version
set(GGML_VERSION_MAJOR 0)
set(GGML_VERSION_MINOR 9)
set(GGML_VERSION_PATCH 11)
set(GGML_VERSION_BASE "${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

find_program(GIT_EXE NAMES git git.exe NO_CMAKE_FIND_ROOT_PATH)
if(GIT_EXE)
    # Get current git commit hash
    execute_process(COMMAND ${GIT_EXE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GGML_BUILD_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Check if the working directory is dirty (i.e., has uncommitted changes)
    execute_process(COMMAND ${GIT_EXE} diff-index --quiet HEAD -- .
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GGML_GIT_DIRTY
        ERROR_QUIET
    )
endif()

set(GGML_VERSION "${GGML_VERSION_BASE}")

if(NOT GGML_BUILD_COMMIT)
    set(GGML_BUILD_COMMIT "unknown")
endif()

# Build the commit string with optional dirty flag
if(DEFINED GGML_GIT_DIRTY AND GGML_GIT_DIRTY EQUAL 1)
    set(GGML_BUILD_COMMIT "${GGML_BUILD_COMMIT}-dirty")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Ember's pruned ggml is built only through the parent project")
endif()

option(BUILD_SHARED_LIBS "ggml: build shared libraries" ON)

#
# option list
#

# This vendored fork is the compute substrate of a Linux/x86-64/gfx1151
# appliance. Keeping upstream backend and cross-architecture switches here
# made unsupported configurations look viable and retained megabytes of dead
# source. Port upstream changes against VENDOR.md; do not restore portability
# scaffolding unless Ember's deployment contract changes.
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
    NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR "Ember's vendored ggml supports only Linux x86-64")
endif()

if (CMAKE_CROSSCOMPILING OR DEFINED ENV{SOURCE_DATE_EPOCH})
    message(STATUS "Setting GGML_NATIVE_DEFAULT to OFF")
    set(GGML_NATIVE_DEFAULT OFF)
else()
    set(GGML_NATIVE_DEFAULT ON)
endif()

# general
option(GGML_STATIC "ggml: static link libraries"                     OFF)
option(GGML_NATIVE "ggml: optimize the build for the current system" ${GGML_NATIVE_DEFAULT})
option(GGML_LTO    "ggml: enable link time optimization"             OFF)
option(GGML_CCACHE "ggml: use ccache if available"                   ON)

# debug
option(GGML_ALL_WARNINGS           "ggml: enable all compiler warnings"                   ON)
option(GGML_GPROF                  "ggml: enable gprof"                                   OFF)

# build
option(GGML_FATAL_WARNINGS    "ggml: enable -Werror flag"    OFF)

# sanitizers
option(GGML_SANITIZE_THREAD    "ggml: enable thread sanitizer"    OFF)
option(GGML_SANITIZE_ADDRESS   "ggml: enable address sanitizer"   OFF)
option(GGML_SANITIZE_UNDEFINED "ggml: enable undefined sanitizer" OFF)

# instruction set specific
if (GGML_NATIVE OR NOT GGML_NATIVE_DEFAULT)
    set(INS_ENB OFF)
else()
    set(INS_ENB ON)
endif()

message(DEBUG "GGML_NATIVE         : ${GGML_NATIVE}")
message(DEBUG "GGML_NATIVE_DEFAULT : ${GGML_NATIVE_DEFAULT}")
message(DEBUG "INS_ENB             : ${INS_ENB}")

option(GGML_CPU_REPACK       "ggml: use runtime weight conversion of Q4_0 to Q4_X_X" ON)
option(GGML_SSE42            "ggml: enable SSE 4.2"          ${INS_ENB})
option(GGML_AVX              "ggml: enable AVX"              ${INS_ENB})
option(GGML_AVX_VNNI         "ggml: enable AVX-VNNI"         OFF)
option(GGML_AVX2             "ggml: enable AVX2"             ${INS_ENB})
option(GGML_BMI2             "ggml: enable BMI2"             ${INS_ENB})
option(GGML_AVX512           "ggml: enable AVX512F"          OFF)
option(GGML_AVX512_VBMI      "ggml: enable AVX512-VBMI"      OFF)
option(GGML_AVX512_VNNI      "ggml: enable AVX512-VNNI"      OFF)
option(GGML_AVX512_BF16      "ggml: enable AVX512-BF16"      OFF)
option(GGML_FMA              "ggml: enable FMA"              ${INS_ENB})
option(GGML_F16C             "ggml: enable F16C"             ${INS_ENB})
# ggml core
set(GGML_SCHED_MAX_COPIES  "4" CACHE STRING "ggml: max input copies for pipeline parallelism")
option(GGML_CPU                             "ggml: enable CPU backend"                        ON)
option(GGML_SCHED_NO_REALLOC                "ggml: disallow reallocations in ggml-alloc (for debugging)" OFF)

# HIP reuses upstream's ggml-cuda source files and therefore retains several
# historical CUDA-prefixed tuning names. They configure AMD kernels here; no
# NVIDIA backend is built.
option(GGML_CUDA_FORCE_MMQ                  "ggml: use mmq kernels instead of cuBLAS"         OFF)
option(GGML_CUDA_FORCE_CUBLAS               "ggml: always use cuBLAS instead of mmq kernels"  OFF)
option(GGML_CUDA_FA                         "ggml: compile ggml FlashAttention CUDA kernels"  ON)
option(GGML_CUDA_FA_ALL_QUANTS              "ggml: compile all quants for FlashAttention"     OFF)

option(GGML_HIP                             "ggml: use HIP"                                   OFF)
option(GGML_HIP_GRAPHS                      "ggml: use HIP graph, experimental, slow"         OFF)
option(GGML_HIP_NO_VMM                      "ggml: do not try to use HIP VMM"                 ON)
option(GGML_HIP_ROCMI4_W4A4                 "ggml: enable lossy gfx1151 W4A4 MMQ for ROCmI4" OFF)
option(GGML_HIP_ROCWMMA_FATTN               "ggml: enable rocWMMA for FlashAttention"         OFF)
option(GGML_HIP_MMQ_MFMA                    "ggml: enable MFMA MMA for CDNA in MMQ"           ON)
option(GGML_HIP_EXPORT_METRICS              "ggml: enable kernel perf metrics output"         OFF)
option(GGML_OPENMP                          "ggml: use OpenMP"                                ON)

# Dependencies and retained targets.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)

add_subdirectory(src)

set(GGML_INSTALL_VERSION ${GGML_VERSION})
target_compile_definitions(ggml-base PRIVATE
    GGML_VERSION="${GGML_INSTALL_VERSION}"
    GGML_COMMIT="${GGML_BUILD_COMMIT}"
)
message(STATUS "ggml version: ${GGML_INSTALL_VERSION}")
message(STATUS "ggml commit:  ${GGML_BUILD_COMMIT}")
