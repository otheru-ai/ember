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
  "Build the optional resident DSpark XRT/IRON provider" OFF)
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

find_package(Threads REQUIRED)

if(EMBER_XDNA_XRT_PROVIDER)
  find_path(EMBER_XRT_INCLUDE_DIR xrt/xrt_bo.h
    HINTS "${EMBER_XRT_ROOT}/include" REQUIRED)
  find_library(EMBER_XRT_COREUTIL xrt_coreutil
    HINTS "${EMBER_XRT_ROOT}/lib" "${EMBER_XRT_ROOT}/lib64" REQUIRED)
  add_library(ember_xdna_dspark_xrt MODULE
    providers/xdna2/provider_dspark_xrt.cpp
    providers/xdna2/dspark_cpu_ops.cpp
    providers/xdna2/q8_0_pack.cpp
    providers/xdna2/rocmfp4_pack.cpp)
  target_include_directories(ember_xdna_dspark_xrt PRIVATE
    providers/xdna2
    engine/dflash/common
    engine/ggml/include
    engine/ggml/src
    "${EMBER_XRT_INCLUDE_DIR}")
  target_link_libraries(ember_xdna_dspark_xrt PRIVATE
    "${EMBER_XRT_COREUTIL}" Threads::Threads m)
  set_target_properties(ember_xdna_dspark_xrt PROPERTIES
    PREFIX ""
    OUTPUT_NAME "libember_xdna_dspark")
  add_executable(ember-xdna-dspark-resident-validate
    providers/xdna2/validate_dspark_resident_xrt.cpp
    providers/xdna2/q8_model_weights.cpp
    providers/xdna2/q8_0_pack.cpp
    src/model/gguf.c)
  target_include_directories(ember-xdna-dspark-resident-validate PRIVATE
    providers/xdna2 src/model "${EMBER_XRT_INCLUDE_DIR}")
  target_link_libraries(ember-xdna-dspark-resident-validate PRIVATE
    "${EMBER_XRT_COREUTIL}")
  add_executable(ember-xdna-dspark-overlay-validate
    providers/xdna2/validate_dspark_overlay_xrt.cpp
    providers/xdna2/q8_model_weights.cpp
    providers/xdna2/q8_0_pack.cpp
    src/model/gguf.c)
  target_include_directories(ember-xdna-dspark-overlay-validate PRIVATE
    providers/xdna2 src/model "${EMBER_XRT_INCLUDE_DIR}")
  target_link_libraries(ember-xdna-dspark-overlay-validate PRIVATE
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

add_library(mock_xdna_dspark_provider MODULE
  test/mock_xdna_dspark_provider.cpp)
target_include_directories(mock_xdna_dspark_provider PRIVATE
  engine/dflash/common engine/ggml/include engine/ggml/src)
set_target_properties(mock_xdna_dspark_provider PROPERTIES PREFIX "")

add_executable(test_xdna_dspark_provider
  test/test_xdna_dspark_provider.cpp
  engine/dflash/common/dspark_draft_compute_xdna.cpp)
target_include_directories(test_xdna_dspark_provider PRIVATE
  engine/dflash/common)
target_link_libraries(test_xdna_dspark_provider PRIVATE ${CMAKE_DL_LIBS})
add_test(NAME xdna_dspark_provider COMMAND test_xdna_dspark_provider
  $<TARGET_FILE:mock_xdna_dspark_provider>)

add_executable(test_xdna_dspark_cpu_ops
  test/test_xdna_dspark_cpu_ops.cpp
  providers/xdna2/dspark_cpu_ops.cpp)
target_include_directories(test_xdna_dspark_cpu_ops PRIVATE providers/xdna2)
target_link_libraries(test_xdna_dspark_cpu_ops PRIVATE m)
add_test(NAME xdna_dspark_cpu_ops COMMAND test_xdna_dspark_cpu_ops)
ember_strict(test_xdna_dspark_cpu_ops)

add_executable(ember-dspark-cpu-bench
  providers/xdna2/benchmark_dspark_cpu_body.cpp
  providers/xdna2/dspark_cpu_ops.cpp)
target_include_directories(ember-dspark-cpu-bench PRIVATE providers/xdna2)
target_link_libraries(ember-dspark-cpu-bench PRIVATE m)
ember_strict(ember-dspark-cpu-bench)

add_executable(ember-dspark-rocmfp4-cpu-bench
  providers/xdna2/benchmark_rocmfp4_cpu.cpp
  providers/xdna2/rocmfp4_model_weights.cpp
  providers/xdna2/rocmfp4_pack.cpp
  src/model/gguf.c)
target_include_directories(ember-dspark-rocmfp4-cpu-bench PRIVATE
  providers/xdna2 src/model)
target_link_libraries(ember-dspark-rocmfp4-cpu-bench PRIVATE
  Threads::Threads m)
ember_strict(ember-dspark-rocmfp4-cpu-bench)

add_executable(test_xdna_rocmfp4_pack
  test/test_xdna_rocmfp4_pack.cpp
  providers/xdna2/rocmfp4_pack.cpp)
target_include_directories(test_xdna_rocmfp4_pack PRIVATE providers/xdna2)
add_test(NAME xdna_rocmfp4_pack COMMAND test_xdna_rocmfp4_pack)
ember_strict(test_xdna_rocmfp4_pack)

add_executable(test_xdna_rocmfp4_model_weights
  test/test_xdna_rocmfp4_model_weights.cpp
  providers/xdna2/rocmfp4_model_weights.cpp
  providers/xdna2/rocmfp4_pack.cpp
  src/model/gguf.c)
target_include_directories(test_xdna_rocmfp4_model_weights PRIVATE
  providers/xdna2 src/model)
add_test(NAME xdna_rocmfp4_model_weights
  COMMAND test_xdna_rocmfp4_model_weights)
ember_strict(test_xdna_rocmfp4_model_weights)

add_executable(test_xdna_q8_pack
  test/test_xdna_q8_pack.cpp
  providers/xdna2/q8_0_pack.cpp)
target_include_directories(test_xdna_q8_pack PRIVATE providers/xdna2)
add_test(NAME xdna_q8_pack COMMAND test_xdna_q8_pack)
ember_strict(test_xdna_q8_pack)

add_executable(test_xdna_q8_model_weights
  test/test_xdna_q8_model_weights.cpp
  providers/xdna2/q8_model_weights.cpp
  providers/xdna2/q8_0_pack.cpp
  src/model/gguf.c)
target_include_directories(test_xdna_q8_model_weights PRIVATE
  providers/xdna2 src/model)
add_test(NAME xdna_q8_model_weights COMMAND test_xdna_q8_model_weights)
ember_strict(test_xdna_q8_model_weights)

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
  continuous_batch_executor resident_batch_coordinator
  xdna_dspark_provider xdna_dspark_cpu_ops
  xdna_rocmfp4_pack xdna_rocmfp4_model_weights xdna_q8_pack
  xdna_q8_model_weights)
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
  add_test(
    NAME resident_benchmark
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_resident_benchmark.py)
  # Offline analysis scripts — no server, no network. Same 60s ceiling as the
  # C tests; server-spawning tests below keep their tighter 20s.
  set_tests_properties(
    quant_quality_report quant_behavior_eval gguf_tensor_error
    quant_manifest_corpus resident_benchmark
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
    NAME profile_scripts
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_profile_scripts.py)
  set_tests_properties(profile_scripts PROPERTIES TIMEOUT 20)
  add_test(
    NAME release_changelog
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_release_changelog.py)
  set_tests_properties(release_changelog PROPERTIES TIMEOUT 20)
  add_test(
    NAME mirror_gh_issues
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_mirror_gh_issues.py)
  set_tests_properties(mirror_gh_issues PROPERTIES TIMEOUT 20)
endif()

# ── standalone model-compute engine (vendored ggml fork + DeepSeek4) ──────
# OFF by default so the GPU-free stub build + tests configure and run anywhere
# (no ROCm/HIP toolchain needed). Turn ON in the ROCm container to build the
# vendored engine under engine/ (see engine/VENDOR.md):
#
#   cmake -S . -B build-rocm -DEMBER_ENGINE=ON
#   cmake --build build-rocm --target ggml-hip -j     # Phase 1: prove ggml builds
#
# Phases: P1 = ggml fork builds; P2 = ember_engine.a (DeepSeek4 + common subset)
# + link backend_dflash.cc; P3 = KV snapshot save/restore.
option(EMBER_ENGINE "Build the vendored GPU model-compute engine (ggml fork + DeepSeek4)" OFF)
if(EMBER_ENGINE)
  # Ember is an AMD Strix Halo appliance, not a portable ggml frontend. Pin
  # gfx1151 before enabling HIP so an image built without a visible GPU cannot
  # silently acquire a generic or host-dependent device-code target.
  if(AMDGPU_TARGETS AND NOT AMDGPU_TARGETS STREQUAL "gfx1151")
    message(FATAL_ERROR "Ember supports only AMD gfx1151, got AMDGPU_TARGETS=${AMDGPU_TARGETS}")
  endif()
  set(CMAKE_HIP_ARCHITECTURES "gfx1151" CACHE STRING "" FORCE)
  set(GPU_TARGETS             "gfx1151" CACHE STRING "" FORCE)
  set(AMDGPU_TARGETS          "gfx1151" CACHE STRING "" FORCE)
  enable_language(HIP)
  add_subdirectory(engine)

  # Narrow placement diagnostic: loads only the trained DSpark main and three
  # context-KV Q8_0 tensors, then measures the GPU pre-stage at proposal and
  # context widths. It avoids a second resident target-model allocation.
  add_executable(ember-dspark-main-bench
    providers/xdna2/benchmark_dspark_main_gpu.cpp
    providers/xdna2/q8_model_weights.cpp
    providers/xdna2/q8_0_pack.cpp
    src/model/gguf.c)
  target_include_directories(ember-dspark-main-bench PRIVATE
    providers/xdna2
    engine/ggml/include
    engine/ggml/src)
  target_link_libraries(ember-dspark-main-bench PRIVATE
    ggml ggml-base ggml-hip ggml-cpu m)

  # Complete-provider hidden-boundary differential. It loads only the draft
  # model, dlopen()s the packaged XRT provider through the stable ABI, and
  # compares CPU/NPU output with the ordinary HIP drafter.
  add_executable(ember-dspark-xdna-diff
    providers/xdna2/validate_dspark_provider_gpu.cpp)
  target_link_libraries(ember-dspark-xdna-diff PRIVATE
    ember_engine Threads::Threads m)

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
  set_source_files_properties(src/backend/backend_dflash.cc PROPERTIES
    COMPILE_DEFINITIONS "DFLASH27B_BACKEND_HIP=1;GGML_USE_HIP;GGML_USE_CPU")
  # xgrammar: backend_dflash.cc compiles the tool grammar and masks logits.
  target_link_libraries(ember-dflash PRIVATE ember_engine xgrammar Threads::Threads m)
  # --validate-gemm-batch calls hipBLAS directly from backend_dflash.cc.
  # ggml-hip links roc::hipblas PRIVATE, so it does not propagate; ask for it
  # here rather than widening ggml's link interface.
  find_package(hip QUIET)
  find_package(hipblas QUIET)
  if(TARGET roc::hipblas AND TARGET hip::host)
    target_link_libraries(ember-dflash PRIVATE hip::host roc::hipblas)
  endif()
endif()
