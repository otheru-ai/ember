cmake_minimum_required(VERSION 3.21)  # 3.21 = first-class HIP language (EMBER_ENGINE)
project(ember C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# ── server/model layer (pure C, no GPU) ──────────────────────────────────
add_library(ember_core STATIC
  src/server/sse.c
  src/model/tool_parser.c
  src/model/tool_memory.c
  src/model/dsml_decode.c
  src/common/json.c
  src/server/chat_api.c
  src/server/http.c
  src/model/chat_template.c
  src/model/gguf.c
  src/backend/backend_stub.c
  src/model/model_card.c
  src/model/kv_cache.c
)
target_include_directories(ember_core PUBLIC src)
find_package(Threads REQUIRED)
target_link_libraries(ember_core PUBLIC Threads::Threads)

# ── tests ────────────────────────────────────────────────────────────────
enable_testing()
add_executable(test_sse test/test_sse.c)
target_link_libraries(test_sse ember_core)
add_test(NAME sse COMMAND test_sse)

add_executable(test_tool_parser test/test_tool_parser.c)
target_link_libraries(test_tool_parser ember_core)
add_test(NAME tool_parser COMMAND test_tool_parser)

add_executable(test_dsml_decode test/test_dsml_decode.c)
target_link_libraries(test_dsml_decode ember_core)
add_test(NAME dsml_decode COMMAND test_dsml_decode)

add_executable(test_json test/test_json.c)
target_link_libraries(test_json ember_core m)
add_test(NAME json COMMAND test_json)

add_executable(test_chat_api test/test_chat_api.c)
target_link_libraries(test_chat_api ember_core m)
add_test(NAME chat_api COMMAND test_chat_api)

add_executable(test_http test/test_http.c)
target_link_libraries(test_http ember_core m)
add_test(NAME http COMMAND test_http)

add_executable(test_chat_template test/test_chat_template.c)
target_link_libraries(test_chat_template ember_core m)
add_test(NAME chat_template COMMAND test_chat_template)

add_executable(test_gguf test/test_gguf.c)
target_link_libraries(test_gguf ember_core m)
add_test(NAME gguf COMMAND test_gguf)

add_executable(test_model_card test/test_model_card.c)
target_link_libraries(test_model_card ember_core m)
add_test(NAME model_card COMMAND test_model_card)

add_executable(test_kv_cache test/test_kv_cache.c)
target_link_libraries(test_kv_cache ember_core m)
add_test(NAME kv_cache COMMAND test_kv_cache)

add_executable(test_qa test/test_qa.c)
target_link_libraries(test_qa ember_core m)
add_test(NAME qa COMMAND test_qa)

# ── ember-server binary ──
add_executable(ember-server src/server/main.c)
target_link_libraries(ember-server ember_core m)

# ── standalone model-compute engine (vendored ggml fork + DeepSeek4) ──────
# OFF by default so the GPU-free stub build + tests configure and run anywhere
# (no ROCm/HIP toolchain needed). Turn ON in the ROCm container to build the
# vendored engine under engine/ (see engine/VENDOR.md):
#
#   cmake -S . -B build-rocm -DEMBER_ENGINE=ON -DEMBER_ENGINE_BACKEND=hip
#   cmake --build build-rocm --target ggml-hip -j     # Phase 1: prove ggml builds
#
# Phases: P1 = ggml fork builds; P2 = ember_engine.a (DeepSeek4 + common subset)
# + link backend_dflash.cc; P3 = KV snapshot save/restore.
option(EMBER_ENGINE "Build the vendored GPU model-compute engine (ggml fork + DeepSeek4)" OFF)
set(EMBER_ENGINE_BACKEND "hip" CACHE STRING "EMBER_ENGINE GPU backend: hip or cuda")
if(EMBER_ENGINE)
  string(TOLOWER "${EMBER_ENGINE_BACKEND}" EMBER_ENGINE_BACKEND)
  if(EMBER_ENGINE_BACKEND STREQUAL "hip")
    # Default to Strix Halo (gfx1151); override with -DAMDGPU_TARGETS=gfxXXXX.
    # Must be set before enable_language(HIP) so HIP targets inherit it.
    if(NOT CMAKE_HIP_ARCHITECTURES)
      if(AMDGPU_TARGETS)
        set(CMAKE_HIP_ARCHITECTURES "${AMDGPU_TARGETS}" CACHE STRING "" FORCE)
      else()
        set(CMAKE_HIP_ARCHITECTURES "gfx1151" CACHE STRING "" FORCE)
      endif()
    endif()
    # ggml-hip + hip-config also consult (AMD)GPU_TARGETS. Pin them so device
    # code targets the right ISA even when the build container has no visible
    # GPU (amdgpu-arch detection fails → would silently default to gfx906/gfx942).
    set(GPU_TARGETS    "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "" FORCE)
    set(AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "" FORCE)
    enable_language(HIP)
  elseif(EMBER_ENGINE_BACKEND STREQUAL "cuda")
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "EMBER_ENGINE_BACKEND must be 'hip' or 'cuda', got '${EMBER_ENGINE_BACKEND}'")
  endif()
  add_subdirectory(engine)

  # ember-dflash: the real server — Ember's C pipeline + the dflash bridge,
  # linked against the vendored ember_engine (no lucebox checkout needed). Uses
  # an explicit source list rather than ember_core, because ember_core links
  # backend_stub.c which would collide with backend_dflash.cc's ABI symbols.
  add_executable(ember-dflash
    src/common/json.c
    src/server/sse.c
    src/server/http.c
    src/server/chat_api.c
    src/model/tool_parser.c
    src/model/tool_memory.c
    src/model/dsml_decode.c
    src/model/chat_template.c
    src/model/model_card.c
    src/model/kv_cache.c
    src/server/main.c
    src/backend/backend_dflash.cc)
  target_include_directories(ember-dflash PRIVATE src src/backend)
  target_compile_definitions(ember-dflash PRIVATE _GNU_SOURCE)
  # The bridge TU pulls in lucebox headers that switch on the GPU backend.
  if(EMBER_ENGINE_BACKEND STREQUAL "hip")
    set_source_files_properties(src/backend/backend_dflash.cc PROPERTIES
      COMPILE_DEFINITIONS "DFLASH27B_BACKEND_HIP=1;GGML_USE_HIP;GGML_USE_CPU")
  elseif(EMBER_ENGINE_BACKEND STREQUAL "cuda")
    set_source_files_properties(src/backend/backend_dflash.cc PROPERTIES
      COMPILE_DEFINITIONS "DFLASH27B_BACKEND_CUDA=1;GGML_USE_CUDA;GGML_USE_CPU")
  endif()
  target_link_libraries(ember-dflash PRIVATE ember_engine Threads::Threads m)
endif()
