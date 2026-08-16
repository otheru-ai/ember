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

# Strict mode, used by CI (.forgejo/workflows/ci.yml) and available locally as
# `cmake -DEMBER_STRICT=ON`. Kept opt-in so an in-progress local edit is not
# blocked by a warning, while master stays clean.
#
# src/ currently compiles with ZERO warnings under this set, so it is a real
# gate rather than an aspiration.
#
# Applied PER-TARGET, never via add_compile_options: several GPU-free tests
# (test_sampler, test_continuous_batch_*, test_resident_batch_coordinator)
# compile vendored engine/ sources directly, and a directory-scoped flag would
# hold upstream code to Ember's standard — turning every fork sync into a
# warning-fixing exercise. EMBER_STRICT_TARGETS below is the ember-owned set.
option(EMBER_STRICT "Treat warnings as errors and enable the strict warning set" OFF)
option(EMBER_XDNA_XRT_PROVIDER
  "Build the optional XRT/IRON ROCMFP2 MoE provider" OFF)
set(EMBER_XRT_ROOT "/opt/xilinx/xrt" CACHE PATH
  "XRT userspace prefix used by EMBER_XDNA_XRT_PROVIDER")
set(EMBER_STRICT_FLAGS
  -Werror
  -Wshadow            # a shadowed KV slot / token index is a real bug class
  -Wconversion        # every narrowing cast in src/ is already explicit
  -Wsign-conversion   # token counts are int, buffer lengths size_t
  -Wformat=2
  -Wnull-dereference)

function(ember_strict target)
  if(EMBER_STRICT)
    target_compile_options(${target} PRIVATE ${EMBER_STRICT_FLAGS})
  endif()
endfunction()

# Hardware-only validator for the packaged provider.  It links only the public
# provider ABI and the CPU ROCMFP2 reference, then dlopen()s the XRT module at
# runtime, so normal GPU-free builds can still compile it without XRT.
add_executable(ember-xdna-validate
  providers/xdna2/validate_xrt.cpp
  providers/xdna2/rocmfp2_pack.cpp)
target_include_directories(ember-xdna-validate PRIVATE
  providers/xdna2
  engine/dflash/common
  engine/ggml/include
  engine/ggml/src)
target_link_libraries(ember-xdna-validate PRIVATE ${CMAKE_DL_LIBS})

if(EMBER_XDNA_XRT_PROVIDER)
  find_path(EMBER_XRT_INCLUDE_DIR xrt/xrt_bo.h
    HINTS "${EMBER_XRT_ROOT}/include" REQUIRED)
  find_library(EMBER_XRT_COREUTIL xrt_coreutil
    HINTS "${EMBER_XRT_ROOT}/lib" "${EMBER_XRT_ROOT}/lib64" REQUIRED)
  find_path(EMBER_ROCM_INCLUDE_DIR hip/hip_runtime_api.h
    HINTS "/opt/rocm/include" REQUIRED)
  find_library(EMBER_HIP_RUNTIME amdhip64
    HINTS "/opt/rocm/lib" "/opt/rocm/lib64" REQUIRED)
  find_library(EMBER_HSA_RUNTIME hsa-runtime64
    HINTS "/opt/rocm/lib" "/opt/rocm/lib64" REQUIRED)
  add_library(ember_xdna_moe_xrt MODULE
    providers/xdna2/provider_xrt.cpp
    providers/xdna2/rocmfp2_pack.cpp)
  target_include_directories(ember_xdna_moe_xrt PRIVATE
    providers/xdna2
    engine/dflash/common
    engine/ggml/include
    engine/ggml/src
    "${EMBER_XRT_INCLUDE_DIR}")
  target_link_libraries(ember_xdna_moe_xrt PRIVATE "${EMBER_XRT_COREUTIL}")
  set_target_properties(ember_xdna_moe_xrt PROPERTIES
    PREFIX ""
    OUTPUT_NAME "libember_xdna_moe")
  add_executable(ember-xdna-gpu-interop
    providers/xdna2/validate_gpu_interop.cpp
    providers/xdna2/rocmfp2_pack.cpp)
  target_include_directories(ember-xdna-gpu-interop PRIVATE
    providers/xdna2
    engine/ggml/include
    engine/ggml/src
    "${EMBER_XRT_INCLUDE_DIR}"
    "${EMBER_ROCM_INCLUDE_DIR}")
  target_compile_definitions(ember-xdna-gpu-interop PRIVATE
    __HIP_PLATFORM_AMD__=1)
  target_link_libraries(ember-xdna-gpu-interop PRIVATE
    "${EMBER_XRT_COREUTIL}"
    "${EMBER_HIP_RUNTIME}"
    "${EMBER_HSA_RUNTIME}")
  add_executable(ember-xdna-rocmfp4-validate
    providers/xdna2/validate_rocmfp4_xrt.cpp
    providers/xdna2/rocmfp4_pack.cpp
    providers/xdna2/rocmfp2_pack.cpp)
  target_include_directories(ember-xdna-rocmfp4-validate PRIVATE
    providers/xdna2
    "${EMBER_XRT_INCLUDE_DIR}")
  target_link_libraries(ember-xdna-rocmfp4-validate PRIVATE
    "${EMBER_XRT_COREUTIL}")
endif()

# ── vendored constrained-decoding core ───────────────────────────────────
# Built in both configurations: the grammar and its token masks are GPU-free,
# so the whole compile/accept/reject path stays testable on the host. See
# vendor/xgrammar/VENDOR.md.
add_subdirectory(vendor/xgrammar)

# ── server/model layer (pure C, no GPU) ──────────────────────────────────
add_library(ember_core STATIC
  src/server/sse.c
  src/model/tool_parser.c
  src/model/tool_schema.c
  src/model/tool_grammar.c
  src/model/tool_memory.c
  src/model/continuation.c
  src/model/dsml_decode.c
  src/common/json.c
  src/server/chat_api.c
  src/server/api_adapters.c
  src/server/compaction.c
  src/server/http.c
  src/server/background_gate.c
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

add_executable(test_tool_schema test/test_tool_schema.c)
target_link_libraries(test_tool_schema ember_core m)
add_test(NAME tool_schema COMMAND test_tool_schema)

add_executable(test_tool_grammar test/test_tool_grammar.c)
target_link_libraries(test_tool_grammar ember_core m)
add_test(NAME tool_grammar COMMAND test_tool_grammar)

add_executable(test_tool_grammar_match test/test_tool_grammar_match.cpp)
target_link_libraries(test_tool_grammar_match ember_core xgrammar m)
add_test(NAME tool_grammar_match COMMAND test_tool_grammar_match)

add_executable(test_tool_memory test/test_tool_memory.c)
target_link_libraries(test_tool_memory ember_core)
add_test(NAME tool_memory COMMAND test_tool_memory)

add_executable(test_continuation test/test_continuation.c)
target_link_libraries(test_continuation ember_core)
add_test(NAME continuation COMMAND test_continuation)

add_executable(test_dsml_decode test/test_dsml_decode.c)
target_link_libraries(test_dsml_decode ember_core)
add_test(NAME dsml_decode COMMAND test_dsml_decode)

add_executable(test_json test/test_json.c)
target_link_libraries(test_json ember_core m)
add_test(NAME json COMMAND test_json)

add_executable(test_chat_api test/test_chat_api.c)
target_link_libraries(test_chat_api ember_core m)
add_test(NAME chat_api COMMAND test_chat_api)

add_executable(test_prefill_policy test/test_prefill_policy.cpp)
target_include_directories(test_prefill_policy PRIVATE
  engine/dflash
  engine/ggml/include
  engine/ggml/src)
add_test(NAME prefill_policy COMMAND test_prefill_policy)

add_executable(test_api_adapters test/test_api_adapters.c)
target_link_libraries(test_api_adapters ember_core m)
add_test(NAME api_adapters COMMAND test_api_adapters)

add_executable(test_http test/test_http.c)
target_link_libraries(test_http ember_core m)
add_test(NAME http COMMAND test_http)

add_executable(test_background_gate test/test_background_gate.c)
target_link_libraries(test_background_gate ember_core m)
add_test(NAME background_gate COMMAND test_background_gate)

add_executable(test_chat_template test/test_chat_template.c)
target_link_libraries(test_chat_template ember_core m)
add_test(NAME chat_template COMMAND test_chat_template)

add_executable(test_compaction test/test_compaction.c)
target_link_libraries(test_compaction ember_core m)
add_test(NAME compaction COMMAND test_compaction)

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

add_executable(test_dspark_scheduler test/test_dspark_scheduler.cpp)
target_include_directories(test_dspark_scheduler PRIVATE
  engine/dflash/deepseek4)
add_test(NAME dspark_scheduler COMMAND test_dspark_scheduler)

add_executable(test_thinking_budget test/test_thinking_budget.cpp)
target_include_directories(test_thinking_budget PRIVATE
  engine/dflash/common)
add_test(NAME thinking_budget COMMAND test_thinking_budget)

add_executable(test_progress_cycle_detector
  test/test_progress_cycle_detector.cpp)
target_include_directories(test_progress_cycle_detector PRIVATE
  engine/dflash/common)
add_test(NAME progress_cycle_detector COMMAND test_progress_cycle_detector)

add_executable(test_sampler
  test/test_sampler.cpp
  engine/dflash/common/sampler.cpp)
target_include_directories(test_sampler PRIVATE engine/dflash/common)
add_test(NAME sampler COMMAND test_sampler)

add_executable(test_pre_tokenizer
  test/test_pre_tokenizer.cpp
  engine/dflash/server/pre_tokenizer.cpp)
target_include_directories(test_pre_tokenizer PRIVATE engine/dflash/server)
add_test(NAME pre_tokenizer COMMAND test_pre_tokenizer)

add_executable(test_continuous_batch_scheduler
  test/test_continuous_batch_scheduler.cpp
  engine/dflash/common/continuous_batch_scheduler.cpp)
target_include_directories(test_continuous_batch_scheduler PRIVATE
  engine/dflash/common)
add_test(NAME continuous_batch_scheduler COMMAND test_continuous_batch_scheduler)

add_executable(test_continuous_batch_executor
  test/test_continuous_batch_executor.cpp
  engine/dflash/common/continuous_batch_scheduler.cpp
  engine/dflash/common/continuous_batch_executor.cpp)
target_include_directories(test_continuous_batch_executor PRIVATE
  engine/dflash/common)
add_test(NAME continuous_batch_executor COMMAND test_continuous_batch_executor)

add_executable(test_resident_batch_coordinator
  test/test_resident_batch_coordinator.cpp
  engine/dflash/common/continuous_batch_scheduler.cpp
  engine/dflash/common/continuous_batch_executor.cpp
  engine/dflash/common/resident_batch_coordinator.cpp)
target_include_directories(test_resident_batch_coordinator PRIVATE
  engine/dflash
  engine/dflash/common
  engine/dflash/placement
  engine/ggml/include)
add_test(NAME resident_batch_coordinator COMMAND test_resident_batch_coordinator)

add_library(mock_xdna_moe_provider MODULE
  test/mock_xdna_moe_provider.cpp
  test/mock_xdna_dspark_provider.cpp)
target_include_directories(mock_xdna_moe_provider PRIVATE
  engine/dflash/common engine/ggml/include engine/ggml/src)
set_target_properties(mock_xdna_moe_provider PROPERTIES PREFIX "")

add_executable(test_xdna_moe_provider
  test/test_xdna_moe_provider.cpp
  engine/dflash/common/moe_expert_compute_xdna.cpp)
target_include_directories(test_xdna_moe_provider PRIVATE
  engine/dflash/common engine/ggml/include engine/ggml/src)
target_link_libraries(test_xdna_moe_provider PRIVATE ${CMAKE_DL_LIBS})
add_test(NAME xdna_moe_provider COMMAND test_xdna_moe_provider
  $<TARGET_FILE:mock_xdna_moe_provider>)

add_executable(test_xdna_dspark_provider
  test/test_xdna_dspark_provider.cpp
  engine/dflash/common/dspark_draft_compute_xdna.cpp)
target_include_directories(test_xdna_dspark_provider PRIVATE
  engine/dflash/common)
target_link_libraries(test_xdna_dspark_provider PRIVATE ${CMAKE_DL_LIBS})
add_test(NAME xdna_dspark_provider COMMAND test_xdna_dspark_provider
  $<TARGET_FILE:mock_xdna_moe_provider>)

add_executable(test_xdna_rocmfp2_pack
  test/test_xdna_rocmfp2_pack.cpp
  providers/xdna2/rocmfp2_pack.cpp)
target_include_directories(test_xdna_rocmfp2_pack PRIVATE providers/xdna2)
add_test(NAME xdna_rocmfp2_pack COMMAND test_xdna_rocmfp2_pack)
ember_strict(test_xdna_rocmfp2_pack)

add_executable(test_xdna_rocmfp4_pack
  test/test_xdna_rocmfp4_pack.cpp
  providers/xdna2/rocmfp4_pack.cpp
  providers/xdna2/rocmfp2_pack.cpp)
target_include_directories(test_xdna_rocmfp4_pack PRIVATE providers/xdna2)
add_test(NAME xdna_rocmfp4_pack COMMAND test_xdna_rocmfp4_pack)
ember_strict(test_xdna_rocmfp4_pack)

# Every C/C++ test gets a wall-clock ceiling. Without one a hung test blocks
# ctest forever: `test_api_adapters` builds a socketpair and pthread_join()s a
# reader thread, so a reader that never sees EOF deadlocks the whole run rather
# than failing. That has already been observed in the wild (a stale ASan build
# sat in futex_wait for ~2 hours).
#
# 60s is deliberately far above need — the entire suite finishes in ~1.5s, and
# ~1.8s under sanitizers — so this only ever fires on a genuine hang, never on a
# slow or loaded machine. The Python server tests set their own tighter 20s
# below, since they additionally wait on a spawned ember-server.
#
# ci/check_invariants.py fails the build if a test is registered without a
# timeout, so a new add_test() cannot silently opt out of this.
set(EMBER_C_TESTS
  sse tool_parser tool_schema tool_grammar tool_grammar_match tool_memory
  continuation dsml_decode json
  chat_api prefill_policy api_adapters http background_gate chat_template
  compaction gguf model_card kv_cache qa dspark_scheduler thinking_budget
  progress_cycle_detector sampler pre_tokenizer continuous_batch_scheduler
  continuous_batch_executor resident_batch_coordinator xdna_moe_provider
  xdna_dspark_provider
  xdna_rocmfp2_pack xdna_rocmfp4_pack)
set_tests_properties(${EMBER_C_TESTS} PROPERTIES TIMEOUT 60)

# ── ember-server binary ──
add_executable(ember-server src/server/main.c)
target_link_libraries(ember-server ember_core m)

# Ember-owned targets, held to EMBER_STRICT. Excludes every target that
# compiles or includes vendored engine/ code (test_prefill_policy,
# test_dspark_scheduler, test_thinking_budget, test_progress_cycle_detector,
# test_sampler, test_continuous_batch_*, test_resident_batch_coordinator):
# upstream keeps its own warning standard. ci/check_invariants.py verifies this
# list stays complete as tests are added.
set(EMBER_STRICT_TARGETS
  ember_core ember-server
  test_sse test_tool_parser test_tool_schema test_tool_memory
  test_tool_grammar test_tool_grammar_match
  test_continuation test_dsml_decode test_json test_chat_api
  test_api_adapters test_http test_background_gate test_chat_template
  test_compaction test_gguf test_model_card test_kv_cache test_qa)
foreach(t IN LISTS EMBER_STRICT_TARGETS)
  ember_strict(${t})
endforeach()

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME quant_quality_report
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_quant_quality_report.py)
  add_test(
    NAME quant_behavior_eval
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_quant_behavior_eval.py)
  add_test(
    NAME gguf_tensor_error
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_gguf_tensor_error.py)
  add_test(
    NAME quant_manifest_corpus
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_quant_manifest_corpus.py)
  # Offline analysis scripts — no server, no network. Same 60s ceiling as the
  # C tests; server-spawning tests below keep their tighter 20s.
  set_tests_properties(
    quant_quality_report quant_behavior_eval gguf_tensor_error
    quant_manifest_corpus
    PROPERTIES TIMEOUT 60)
  add_test(
    NAME continuous_batch_server
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_continuous_batch_server.py
            $<TARGET_FILE:ember-server>)
  set_tests_properties(continuous_batch_server PROPERTIES TIMEOUT 20)
  add_test(
    NAME tool_safety_server
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_tool_safety_server.py
            $<TARGET_FILE:ember-server>)
  set_tests_properties(tool_safety_server PROPERTIES TIMEOUT 20)
  add_test(
    NAME request_budgets_server
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_request_budgets_server.py
            $<TARGET_FILE:ember-server>)
  set_tests_properties(request_budgets_server PROPERTIES TIMEOUT 20)
  add_test(
    NAME client_compatibility_server
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_client_compatibility_server.py
            $<TARGET_FILE:ember-server>)
  set_tests_properties(client_compatibility_server PROPERTIES TIMEOUT 20)
  add_test(
    NAME release_scripts
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_release_scripts.py)
  set_tests_properties(release_scripts PROPERTIES TIMEOUT 20)
  add_test(
    NAME promote_from_dev
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_promote_from_dev.py)
  set_tests_properties(promote_from_dev PROPERTIES TIMEOUT 20)
endif()

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

  # Metadata-only tokenizer probe.  Keep it independent of ember_engine so a
  # production GGUF can be inspected without loading weights or touching the
  # GPU; this is the direct parity surface for ds4's tokenizer utility.
  add_executable(ember-token-dump
    tools/token_dump.cpp
    engine/dflash/server/pre_tokenizer.cpp
    engine/dflash/server/tokenizer.cpp)
  target_include_directories(ember-token-dump PRIVATE engine/dflash/server)
  target_link_libraries(ember-token-dump PRIVATE ggml-base)
  set_target_properties(ember-token-dump PROPERTIES
    # The ROCm build runs inside a container mounted at /ember. A build-tree
    # absolute RPATH therefore points at a path that does not exist when the
    # probe is invoked on the host. Keep it relocatable beside build-rocm/.
    BUILD_RPATH "$ORIGIN/engine/ggml/src"
    BUILD_RPATH_USE_ORIGIN TRUE)

  # ember-dflash: the real server — Ember's C pipeline + the dflash bridge,
  # linked against the vendored ember_engine (no lucebox checkout needed). Uses
  # an explicit source list rather than ember_core, because ember_core links
  # backend_stub.c which would collide with backend_dflash.cc's ABI symbols.
  add_executable(ember-dflash
    src/common/json.c
    src/server/sse.c
    src/server/http.c
    src/server/chat_api.c
    src/server/api_adapters.c
    src/server/compaction.c
    src/model/tool_parser.c
    src/model/tool_schema.c
    src/model/tool_grammar.c
    src/server/background_gate.c
    src/model/tool_memory.c
    src/model/continuation.c
    src/model/dsml_decode.c
    src/model/chat_template.c
    src/model/model_card.c
    src/model/kv_cache.c
    src/server/main.c
    src/backend/backend_dflash.cc)

  # segvtrace resolves ds4_crash_breadcrumb via dlsym(RTLD_DEFAULT); an
  # executable's symbols are only visible to that if it is linked with
  # --export-dynamic, which is what ENABLE_EXPORTS sets.
  set_target_properties(ember-dflash PROPERTIES ENABLE_EXPORTS ON)
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
  # xgrammar: backend_dflash.cc compiles the tool grammar and masks logits.
  target_link_libraries(ember-dflash PRIVATE ember_engine xgrammar Threads::Threads m)
  if(EMBER_ENGINE_BACKEND STREQUAL "hip")
    # --validate-gemm-batch calls hipBLAS directly from backend_dflash.cc.
    # ggml-hip links roc::hipblas PRIVATE, so it does not propagate; ask for it
    # here rather than widening ggml's link interface.
    find_package(hip QUIET)
    find_package(hipblas QUIET)
    if(TARGET roc::hipblas AND TARGET hip::host)
      target_link_libraries(ember-dflash PRIVATE hip::host roc::hipblas)
    endif()
  endif()
endif()
