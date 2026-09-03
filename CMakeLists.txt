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
find_package(ZLIB REQUIRED)

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
  src/model/image_input.c
  src/model/vision_prompt.c
  src/model/gguf.c
  src/model/model_profile.c
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

add_executable(test_ds4_vision_contract test/test_ds4_vision_contract.cpp
  engine/dflash/deepseek4/deepseek4_vision_contract.cpp)
target_include_directories(test_ds4_vision_contract PRIVATE engine)
add_test(NAME ds4_vision_contract COMMAND test_ds4_vision_contract)

add_executable(test_ds4_vision_native_contract
  test/test_ds4_vision_native_contract.cpp
  engine/dflash/deepseek4/deepseek4_vision_native_contract.cpp)
target_include_directories(test_ds4_vision_native_contract PRIVATE engine)
target_link_libraries(test_ds4_vision_native_contract PRIVATE ZLIB::ZLIB)
ember_strict(test_ds4_vision_native_contract)
add_test(NAME ds4_vision_native_contract COMMAND test_ds4_vision_native_contract)

add_executable(test_ds4_vision_oracle
  test/test_ds4_vision_oracle.cpp
  engine/dflash/deepseek4/deepseek4_vision_oracle.cpp)
target_include_directories(test_ds4_vision_oracle PRIVATE engine)
target_include_directories(test_ds4_vision_oracle PRIVATE engine/third_party)
add_test(NAME ds4_vision_oracle COMMAND test_ds4_vision_oracle)

add_executable(test_ds4_vision_artifact test/test_ds4_vision_artifact.cpp
  engine/dflash/deepseek4/deepseek4_vision_artifact.cpp
  engine/dflash/deepseek4/deepseek4_vision_contract.cpp)
target_include_directories(test_ds4_vision_artifact PRIVATE engine)
add_test(NAME ds4_vision_artifact COMMAND test_ds4_vision_artifact)

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

add_executable(test_model_profile test/test_model_profile.c)
target_link_libraries(test_model_profile ember_core m)
add_test(NAME model_profile COMMAND test_model_profile)

add_executable(test_image_input test/test_image_input.c)
target_link_libraries(test_image_input ember_core)
add_test(NAME image_input COMMAND test_image_input)

add_executable(test_vision_prompt test/test_vision_prompt.c)
target_link_libraries(test_vision_prompt ember_core)
add_test(NAME vision_prompt COMMAND test_vision_prompt)

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

# Pure host arithmetic (no GPU, no ember_core): the ROCMFP UE4M3 scale decode
# is exhaustively checked against the pre-rewrite reference for all 256 inputs.
add_executable(test_rocmfp_scale test/test_rocmfp_scale.c)
add_test(NAME rocmfp_scale COMMAND test_rocmfp_scale)
add_executable(test_rocmfp2_unpack test/test_rocmfp2_unpack.c)
target_include_directories(test_rocmfp2_unpack PRIVATE engine/ggml/include)
add_test(NAME rocmfp2_unpack COMMAND test_rocmfp2_unpack)
add_executable(test_ds4_imatrix test/test_ds4_imatrix.cpp
               engine/dflash/deepseek4/deepseek4_imatrix.cpp)
target_include_directories(test_ds4_imatrix PRIVATE engine)
add_test(NAME ds4_imatrix COMMAND test_ds4_imatrix)

add_executable(test_rocmfp2_offset test/test_rocmfp2_offset.c)
target_link_libraries(test_rocmfp2_offset PRIVATE m)
add_test(NAME rocmfp2_offset COMMAND test_rocmfp2_offset)

add_executable(test_rocmi4 test/test_rocmi4.c engine/ggml/rocmfpx/rocmfpx.c)
target_include_directories(test_rocmi4 PRIVATE engine/ggml/include engine/ggml/rocmfpx)
target_link_libraries(test_rocmi4 PRIVATE m)
add_test(NAME rocmi4 COMMAND test_rocmi4)

# Pure host oracle for the exact mixed-signedness I4 identity proposed for the
# gfx1151 ROCMI4 path. It does not compile or enable a GPU kernel.
add_executable(test_rocmi4_q8_i4_exact test/test_rocmi4_q8_i4_exact.cpp)
target_include_directories(test_rocmi4_q8_i4_exact PRIVATE engine/ggml/rocmfpx)
add_test(NAME rocmi4_q8_i4_exact COMMAND test_rocmi4_q8_i4_exact)

add_executable(test_dspark_scheduler test/test_dspark_scheduler.cpp)
target_include_directories(test_dspark_scheduler PRIVATE
  engine/dflash/deepseek4)
add_test(NAME dspark_scheduler COMMAND test_dspark_scheduler)

add_executable(test_thinking_budget test/test_thinking_budget.c)
target_include_directories(test_thinking_budget PRIVATE
  engine/dflash/common)
ember_strict(test_thinking_budget)
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







add_executable(test_activation_dump
  test/test_activation_dump.cpp
  engine/dflash/common/activation_dump.cpp)
target_include_directories(test_activation_dump PRIVATE
  engine/dflash/common)
ember_strict(test_activation_dump)
add_test(NAME activation_dump COMMAND test_activation_dump)


add_executable(test_gguf_quantize_common
  test/test_gguf_quantize_common.cpp
  tools/gguf_quantize_common.cpp)
target_include_directories(test_gguf_quantize_common PRIVATE tools)
ember_strict(test_gguf_quantize_common)
add_test(NAME gguf_quantize_common COMMAND test_gguf_quantize_common)

add_executable(test_logits_probe_format
  test/test_logits_probe_format.cpp
  tools/logits_probe_format.cpp)
target_include_directories(test_logits_probe_format PRIVATE
  tools engine/dflash/deepseek4)
ember_strict(test_logits_probe_format)
add_test(NAME logits_probe_format COMMAND test_logits_probe_format)

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
  chat_api prefill_policy ds4_vision_contract ds4_vision_native_contract ds4_vision_oracle ds4_vision_artifact api_adapters http background_gate chat_template
  compaction gguf model_card model_profile image_input vision_prompt kv_cache qa rocmfp_scale rocmfp2_unpack rocmfp2_offset ds4_imatrix rocmi4 rocmi4_q8_i4_exact dspark_scheduler thinking_budget
  progress_cycle_detector sampler pre_tokenizer
 activation_dump
  gguf_quantize_common
  logits_probe_format
  continuous_batch_scheduler
  continuous_batch_executor resident_batch_coordinator
  xdna_dspark_provider xdna_dspark_cpu_ops
  xdna_rocmfp4_pack xdna_rocmfp4_model_weights xdna_q8_pack
  xdna_q8_model_weights)
set_tests_properties(${EMBER_C_TESTS} PROPERTIES TIMEOUT 60)

# ── ember-server binary ──
add_executable(ember-server src/server/main.c)
target_link_libraries(ember-server ember_core m)

# Unit tests for main.c's internal helpers. They are static, so the test
# translation unit includes main.c with its entry point renamed rather than
# widening the server's linkage for the benefit of a test. See the file header.
add_executable(test_server_helpers test/test_server_helpers.c)
target_link_libraries(test_server_helpers ember_core m)
add_test(NAME server_helpers COMMAND test_server_helpers)
set_tests_properties(server_helpers PROPERTIES TIMEOUT 60)

# Ember-owned targets, held to EMBER_STRICT. Excludes every target that
# compiles or includes vendored engine/ code (test_prefill_policy,
# test_dspark_scheduler, test_progress_cycle_detector,
# test_sampler, test_continuous_batch_*, test_resident_batch_coordinator):
# upstream keeps its own warning standard. ci/check_invariants.py verifies this
# list stays complete as tests are added.
set(EMBER_STRICT_TARGETS
  ember_core ember-server
  test_sse test_tool_parser test_tool_schema test_tool_memory
  test_tool_grammar test_tool_grammar_match
  test_continuation test_dsml_decode test_json test_chat_api
  test_api_adapters test_http test_background_gate test_chat_template
  test_compaction test_gguf test_model_card test_model_profile test_image_input
  test_vision_prompt test_kv_cache test_qa test_server_helpers)
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
    NAME ds4_vision_behavior_gate
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_ds4_vision_behavior_gate.py)
  add_test(
    NAME build_revision_guard
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_build_revision_guard.py
            ${CMAKE_COMMAND}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_git_head.cmake)
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
    quant_quality_report quant_behavior_eval ds4_vision_behavior_gate
    build_revision_guard gguf_tensor_error
    quant_manifest_corpus resident_benchmark
    PROPERTIES TIMEOUT 60)
  add_test(
    NAME continuous_batch_server
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_continuous_batch_server.py
            $<TARGET_FILE:ember-server>)
  set_tests_properties(continuous_batch_server PROPERTIES TIMEOUT 20)
  add_test(
    NAME tool_continuation_server
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_tool_continuation_server.py
            $<TARGET_FILE:ember-server>)
  set_tests_properties(tool_continuation_server PROPERTIES TIMEOUT 60)
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
    NAME rocmi4_w4a8_isa_gate
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_rocmi4_w4a8_isa_gate.py)
  set_tests_properties(rocmi4_w4a8_isa_gate PROPERTIES TIMEOUT 20)
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


  add_test(
    NAME benchmark_gate
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_benchmark_gate.py)
  set_tests_properties(benchmark_gate PROPERTIES TIMEOUT 20)
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

  # The imatrix collector against a real ggml graph. Uses the CPU backend and
  # never touches a device, so it runs in the GPU-less ROCm build container --
  # but it needs the vendored ggml, hence the engine guard.
  add_executable(test_ds4_imatrix_graph test/test_ds4_imatrix_graph.cpp
                 engine/dflash/deepseek4/deepseek4_imatrix.cpp
                 engine/dflash/deepseek4/deepseek4_imatrix_graph.cpp)
  target_include_directories(test_ds4_imatrix_graph PRIVATE engine engine/ggml/include)
  target_link_libraries(test_ds4_imatrix_graph PRIVATE ggml ggml-base ggml-cpu)
  add_test(NAME ds4_imatrix_graph COMMAND test_ds4_imatrix_graph)
  # Builds a real ggml graph on the CPU backend; a hang here would otherwise
  # block ctest indefinitely.
  set_tests_properties(ds4_imatrix_graph PROPERTIES TIMEOUT 120)

  # Metadata-only split-set coverage. It links the real loader's gguf parser
  # but never creates a HIP backend or touches a device, so it remains runnable
  # in the GPU-less ROCm build container.
  add_executable(test_ds4_vision_markers test/test_ds4_vision_markers.cpp)
  target_include_directories(test_ds4_vision_markers PRIVATE
    engine/dflash/deepseek4 engine/ggml/include)
  target_link_libraries(test_ds4_vision_markers PRIVATE ember_engine)
  ember_strict(test_ds4_vision_markers)
  add_test(NAME ds4_vision_markers COMMAND test_ds4_vision_markers)
  set_tests_properties(ds4_vision_markers PROPERTIES TIMEOUT 60)

  # Strict metadata-only gate for the complete 299-tensor native Vision-Exp
  # projector. Like the marker test, it parses GGUF without creating a backend
  # or opening a GPU device.
  add_executable(test_ds4_vision_mmproj
    test/test_ds4_vision_mmproj.cpp
    engine/dflash/deepseek4/deepseek4_vision_mmproj.cpp
    engine/dflash/deepseek4/deepseek4_vision_native_contract.cpp)
  target_include_directories(test_ds4_vision_mmproj PRIVATE
    engine/dflash engine/dflash/deepseek4 engine/ggml/include)
  target_link_libraries(test_ds4_vision_mmproj PRIVATE ggml-base)
  target_link_libraries(test_ds4_vision_mmproj PRIVATE ZLIB::ZLIB)
  ember_strict(test_ds4_vision_mmproj)
  add_test(NAME ds4_vision_mmproj COMMAND test_ds4_vision_mmproj)
  set_tests_properties(ds4_vision_mmproj PROPERTIES TIMEOUT 60)

  # Hardware oracle driver for the native tower. It is intentionally not a
  # ctest: meaningful execution requires the real mmproj, paired BF16/F32
  # authority files, their bound raw patch payload, and a target GPU.
  add_executable(ds4_vision_tower_oracle
    tools/ds4_vision_tower_oracle.cpp)
  target_include_directories(ds4_vision_tower_oracle PRIVATE
    engine/dflash engine/dflash/deepseek4 engine/ggml/include)
  target_link_libraries(ds4_vision_tower_oracle PRIVATE ember_engine ggml)
  ember_strict(ds4_vision_tower_oracle)

  # Metadata-only inventory gate for all 43 language-model vision routing
  # biases. An optional real path validates the produced full GGUF without
  # reading tensor payloads or creating a GPU backend.
  add_executable(test_ds4_vision_biases test/test_ds4_vision_biases.cpp)
  target_include_directories(test_ds4_vision_biases PRIVATE
    engine/dflash/deepseek4 engine/ggml/include)
  target_link_libraries(test_ds4_vision_biases PRIVATE ember_engine)
  ember_strict(test_ds4_vision_biases)
  add_test(NAME ds4_vision_biases COMMAND test_ds4_vision_biases)
  set_tests_properties(ds4_vision_biases PROPERTIES TIMEOUT 60)

  # CPU execution proof for the mixed text/vision row partition used by the
  # layer-major MoE graph. It catches wrong ggml gather/scatter shapes and
  # row-index input lifetime mistakes without a GPU or model weights.
  add_executable(test_ds4_vision_graph_rows
    test/test_ds4_vision_graph_rows.cpp)
  target_include_directories(test_ds4_vision_graph_rows PRIVATE
    engine/ggml/include)
  target_link_libraries(test_ds4_vision_graph_rows PRIVATE ember_engine)
  add_test(NAME ds4_vision_graph_rows COMMAND test_ds4_vision_graph_rows)
  set_tests_properties(ds4_vision_graph_rows PROPERTIES TIMEOUT 60)

  # Uses the CPU backend from the ROCm toolchain build: no GPU device or model
  # weights are required, but it exercises the same persistent ggml graph that
  # production dispatches through gfx1151 MMVQ/MMQ.
  add_executable(test_rocmi4_operator_oracle
    test/test_rocmi4_operator_oracle.hip)
  target_include_directories(test_rocmi4_operator_oracle PRIVATE
    engine/ggml/include
    engine/ggml/rocmfp4
    engine/ggml/rocmfpx
    engine/ggml/src
    engine/ggml/src/ggml-cuda)
  target_link_libraries(test_rocmi4_operator_oracle PRIVATE
    ember_engine hip::host)
  # Deliberately not ember_strict: convert.cuh includes vendored CUDA internals
  # with translation-unit-local unused helpers, as do the frontier oracles.
  add_test(NAME rocmi4_operator_oracle COMMAND test_rocmi4_operator_oracle)
  set_tests_properties(rocmi4_operator_oracle PROPERTIES TIMEOUT 60)

  # GPU-free oracle for the graph RoPE path. Tranche 1 moves QSA rotary
  # embedding off a host scalar and into an in-graph
  # `ggml_rope_multi`; that swap has five parameters (n_dims, the interleaved
  # sections, attn_factor, the axis-major position layout, and the `c`
  # freq-factor table, which ggml *divides* by) that are silent when wrong.
  # Running the op on the CPU backend against the scalar reference catches all
  # of them without hardware. CPU and HIP share the op, so agreement here is
  # necessary but not sufficient — the hardware differential remains the gate.
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

  # Architecture-neutral, bounded-memory GGUF transformer. It uses ggml-base's
  # CPU type traits and ROCMI4 reference quantizer only; no HIP backend is
  # linked or initialized even though it ships from the full engine build.
  set(EMBER_CONFIGURED_GIT_HEAD "" CACHE STRING
    "40-hex Ember revision embedded in release tools")
  find_package(Git QUIET)
  if(NOT EMBER_CONFIGURED_GIT_HEAD)
    if(Git_FOUND)
      execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse HEAD
        OUTPUT_VARIABLE EMBER_CONFIGURED_GIT_HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE EMBER_GIT_COMMAND_RESULT)
      if(NOT EMBER_GIT_COMMAND_RESULT EQUAL 0)
        set(EMBER_CONFIGURED_GIT_HEAD "")
      endif()
    endif()
    if(NOT EMBER_CONFIGURED_GIT_HEAD AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git/HEAD")
      file(READ "${CMAKE_CURRENT_SOURCE_DIR}/.git/HEAD" EMBER_GIT_HEAD_TEXT)
      string(STRIP "${EMBER_GIT_HEAD_TEXT}" EMBER_GIT_HEAD_TEXT)
      if(EMBER_GIT_HEAD_TEXT MATCHES "^ref: (.+)$")
        set(EMBER_GIT_HEAD_REF "${CMAKE_MATCH_1}")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git/${EMBER_GIT_HEAD_REF}")
          file(READ "${CMAKE_CURRENT_SOURCE_DIR}/.git/${EMBER_GIT_HEAD_REF}"
            EMBER_CONFIGURED_GIT_HEAD)
          string(STRIP "${EMBER_CONFIGURED_GIT_HEAD}"
            EMBER_CONFIGURED_GIT_HEAD)
        endif()
      else()
        set(EMBER_CONFIGURED_GIT_HEAD "${EMBER_GIT_HEAD_TEXT}")
      endif()
    endif()
  endif()
  string(LENGTH "${EMBER_CONFIGURED_GIT_HEAD}" EMBER_GIT_HEAD_LENGTH)
  if(NOT EMBER_GIT_HEAD_LENGTH EQUAL 40 OR
     NOT EMBER_CONFIGURED_GIT_HEAD MATCHES "^[0-9a-f]+$")
    # Docker intentionally excludes .git. An unrelated server image must still
    # configure when no explicit EMBER_CONFIGURED_GIT_HEAD was supplied, but a
    # provenance-free quantizer must never be emitted. Release CI passes the
    # revision explicitly; normal source checkouts discover it above.
    message(STATUS
      "ember-gguf-quantize disabled: configure with a 40-hex "
      "EMBER_CONFIGURED_GIT_HEAD to build release tools")
  else()
    # This is intentionally a phony build target, not a configure-time check.
    # A reused CMake directory must not keep emitting tools branded with the
    # revision that happened to be checked out when it was first configured.
    # Source archives have no .git entry and retain the explicit revision
    # supplied by release tooling; live checkouts are checked on every build.
    add_custom_target(ember_release_tool_revision_check
      COMMAND ${CMAKE_COMMAND}
        -DEMBER_EXPECTED_GIT_HEAD=${EMBER_CONFIGURED_GIT_HEAD}
        -DEMBER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DEMBER_GIT_EXECUTABLE=${GIT_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_git_head.cmake
      VERBATIM)

    add_executable(ember-gguf-quantize
      tools/gguf_quantize.cpp
      tools/gguf_quantize_common.cpp)
    add_dependencies(ember-gguf-quantize ember_release_tool_revision_check)
    target_include_directories(ember-gguf-quantize PRIVATE
      tools
      engine/third_party)
    target_compile_definitions(ember-gguf-quantize PRIVATE
      EMBER_CONFIGURED_GIT_HEAD="${EMBER_CONFIGURED_GIT_HEAD}")
    target_link_libraries(ember-gguf-quantize PRIVATE ggml-base Threads::Threads)
    ember_strict(ember-gguf-quantize)
    set_target_properties(ember-gguf-quantize PROPERTIES
      BUILD_RPATH "$ORIGIN/engine/ggml/src"
      BUILD_RPATH_USE_ORIGIN TRUE)

    # Default-off full-vocabulary control probe. It is intentionally a
    # separate executable: no server request or environment option can make
    # ordinary serving retain or emit this diagnostic payload. Like the
    # quantizer above, it is omitted rather than built without provenance.
    add_executable(ds4_logits_probe
      tools/ds4_logits_probe.cpp
      tools/logits_probe_format.cpp)
    add_dependencies(ds4_logits_probe ember_release_tool_revision_check)
    target_include_directories(ds4_logits_probe PRIVATE
      tools engine/dflash engine/dflash/deepseek4 engine/ggml/include)
    target_compile_definitions(ds4_logits_probe PRIVATE
      EMBER_CONFIGURED_GIT_HEAD="${EMBER_CONFIGURED_GIT_HEAD}")
    target_link_libraries(ds4_logits_probe PRIVATE ember_engine ggml)
    ember_strict(ds4_logits_probe)

    if(Python3_Interpreter_FOUND)
      add_test(
        NAME gguf_quantize_tool
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/test_gguf_quantize_tool.py
                $<TARGET_FILE:ember-gguf-quantize>)
      set_tests_properties(gguf_quantize_tool PROPERTIES TIMEOUT 60)
    endif()
  endif()

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
    src/model/image_input.c
    src/model/vision_prompt.c
    src/model/gguf.c
    src/model/model_profile.c
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
    COMPILE_DEFINITIONS "GGML_USE_HIP;GGML_USE_CPU")
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
