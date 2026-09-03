# Ember standalone model-compute engine.
#
# Phase 1 (this file): build the vendored ggml fork (with the ROCMFP4 / ROCMFPX
# kernels) as ggml-base + ggml-hip. Phase 2 adds ember_engine.a
# (the DeepSeek4 backend + the common/ subset it needs) linking that backend.
#
# The parent CMakeLists has already enable_language()'d HIP and pinned gfx1151.
# See engine/VENDOR.md for provenance / porting.

# ── trim ggml to just the compute backend we use ────────────────────────
# Mirrors lucebox/server/CMakeLists.txt: no metal/vulkan/blas/opencl, no
# runtime backend DL, no tests/examples. Ember is gfx1151/HIP-only.
set(GGML_HIP        ON  CACHE BOOL "" FORCE)
set(GGML_CCACHE     OFF CACHE BOOL "" FORCE)
  # FORCE this OFF: an ordinary option default does not overwrite an existing
  # build-tree cache entry, so incremental builds made after 4beb51e could keep
  # GGML_HIP_GRAPHS=ON even after f026c35 removed the enabling assignment.
  set(GGML_HIP_GRAPHS OFF CACHE BOOL "" FORCE)
  # HIP graph replay stays OFF — MEASURED, not assumed. USE_CUDA_GRAPH comes from
  # GGML_CUDA_USE_GRAPHS || GGML_HIP_GRAPHS (ggml-cuda/common.cuh).
  # Turning it ON looks obviously right (a fused 61-layer decode graph otherwise
  # launches every kernel individually per token) but A/B on gfx1151 with the
  # production model, same binary via GGML_CUDA_DISABLE_GRAPHS, 3 steady-state
  # requests each, was a REGRESSION:
  #     graphs ON : prefill 22.9 tok/s, decode 21.39 tok/s
  #     graphs OFF: prefill 24.6 tok/s, decode 22.29 tok/s   (+7.4% / +4.2%)
  # Cause is visible in GGML_CUDA_GRAPH_STATS: continuous "warmup complete" /
  # "warmup reset" churn plus [graph-mismatch] at node 0. The graph key never
  # stabilizes, so capture+validation cost is paid repeatedly and steady replay
  # is never reached.
  #
  # CORRECTION (2026-08-22, re-measured): an earlier version of this comment
  # blamed the per-layer padded comp rows (DS4_COMP_PAD_STRIDE=16) and advised
  # widening that stride. That diagnosis was wrong, and the suggested fix does
  # not work. Tested directly: DS4_COMP_PAD_STRIDE=256 -- which reduces the
  # comp-row key to 2 distinct values across a 256-token decode instead of ~16
  # -- with GGML_HIP_GRAPHS=ON still produced 64 graph-mismatch events and 224
  # warmup-churn events, and prefill still regressed at every context length:
  #     graphs OFF: 213.0 / 327.1 / 397.4 / 338.6 / 311.4 tok/s
  #     graphs ON : 206.8 / 317.7 / 391.0 / 332.4 / 308.0 tok/s
  #                 (154 / 538 / 2074 / 8218 / 16410 evaluated prompt tokens)
  #
  # The actual mismatch is:
  #     [graph-mismatch] node=0 op=VIEW name=ds4_raw_kv_1 (view)
  # ds4_raw_kv_N itself is a fixed-size tensor (head_dim x n_swa), so it is not
  # the tensor that changes -- it is the VIEW into it. The raw MLA KV cache is a
  # sliding-window ring buffer, so the write row rotates every token and the
  # view offset is baked into the captured topology. That invalidates the graph
  # once per token, which is far worse than the ~64-token bucket crossing the
  # old comment described, and no pad stride can affect it.
  #
  # RETESTED 2026-08-22 and still a loss, on a NEW hypothesis that turned out to
  # be wrong. The theory was that q_step_cap = 4 - (pos & 3) cycled the spec
  # verify width 4,3,2,1, so graph topology changed almost every decode step and
  # replay could never hit. The q5 fixed-width path makes q_cap constant at 5
  # (telemetry: q_cap=5, mean_accept=3.15/3.20 in both arms, so width really was
  # identical). Graphs still lost:
  #     graphs OFF: decode 34.69 tok/s, prefill-128 212.7
  #     graphs ON : decode 33.51 tok/s, prefill-128 206.6   (-3.4% / -2.9%)
  # So constant verify width is NOT sufficient. Whatever invalidates the graph
  # survives a fixed topology, and the cost is not capture churn from width.
  # DO NOT re-enable this until the ring-buffer write offset stops being part of
  # the graph topology -- e.g. express it as a graph input / indirect index so
  # the captured nodes are position-independent, or capture one graph per ring
  # position and let the cache serve them round-robin. Graph replay cannot pay
  # off while every token invalidates the capture.
set(EMBER_GGML_BACKEND ggml-hip)
set(EMBER_GGML_BACKEND "${EMBER_GGML_BACKEND}" PARENT_SCOPE)  # Phase 2 links this

# Asymmetric KV-quant fattn pairs (lucebox default DFLASH27B_FA_ALL_QUANTS=ON).
set(GGML_CUDA_FA_ALL_QUANTS ON CACHE BOOL "" FORCE)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
  message(FATAL_ERROR
    "Vendored ggml sources missing at engine/ggml (see engine/VENDOR.md).")
endif()

# Use only the ggml subtree (libllama is not built). EXCLUDE_FROM_ALL: the
# targets build when something depends on them (Phase 2's ember_engine) or when
# built explicitly (`--target ggml-hip`).
add_subdirectory(ggml EXCLUDE_FROM_ALL)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The vendored ggml HIP shim still uses a few CUDA spellings this snapshot
# does not map, plus cuda_fp16/bf16 headers — supply both (same as lucebox).
target_compile_definitions(ggml-hip PRIVATE
  cublasSgemmStridedBatched=hipblasSgemmStridedBatched
  cudaStreamCaptureStatus=hipStreamCaptureStatus
  cudaStreamCaptureStatusNone=hipStreamCaptureStatusNone
  cudaStreamIsCapturing=hipStreamIsCapturing)
target_include_directories(ggml-hip BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
find_package(hip REQUIRED)

message(STATUS
  "[ember] engine: backend=hip "
  "hip_archs=${CMAKE_HIP_ARCHITECTURES} "
  "ggml_target=${EMBER_GGML_BACKEND}")

# ── ember_engine: the DeepSeek4 forward pass (vendored dflash subset) ─────
# The transitive #include+source closure of the monolithic DeepSeek4 backend
# (see engine/VENDOR.md). Multi-GPU layer-split / shard-IPC / other-arch files
# are intentionally excluded; ember_create_backend.cpp replaces the multi-arch
# backend_factory.cpp. Include dirs mirror lucebox's DFLASH27B_SRC_INCLUDE_DIRS.
set(DFLASH ${CMAKE_CURRENT_SOURCE_DIR}/dflash)
set(EMBER_ENGINE_INC
  ${DFLASH} ${DFLASH}/common ${DFLASH}/deepseek4
  ${DFLASH}/server ${DFLASH}/placement
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_library(ember_engine STATIC
  ${DFLASH}/deepseek4/deepseek4_loader.cpp
  ${DFLASH}/deepseek4/deepseek4_graph.cpp
  ${DFLASH}/deepseek4/deepseek4_imatrix.cpp
  ${DFLASH}/deepseek4/deepseek4_imatrix_graph.cpp
  ${DFLASH}/deepseek4/deepseek4_backend.cpp
  ${DFLASH}/deepseek4/deepseek4_dspark.cpp
  ${DFLASH}/deepseek4/deepseek4_vision_biases.cpp
  ${DFLASH}/deepseek4/deepseek4_vision_artifact.cpp
  ${DFLASH}/deepseek4/deepseek4_vision_contract.cpp
  ${DFLASH}/deepseek4/deepseek4_vision_native_contract.cpp
  ${DFLASH}/deepseek4/deepseek4_vision_mmproj.cpp
  ${DFLASH}/deepseek4/deepseek4_vision_oracle.cpp
  ${DFLASH}/deepseek4/deepseek4_vision_tower.cpp
  ${DFLASH}/deepseek4/deepseek4_vision_markers.cpp
  ${DFLASH}/deepseek4/deepseek4_dspark_spec.cpp
  ${DFLASH}/common/activation_dump.cpp
  ${DFLASH}/common/model_architecture.cpp
  ${DFLASH}/common/crash_breadcrumb.c
  ${DFLASH}/common/continuous_batch_executor.cpp
  ${DFLASH}/common/continuous_batch_scheduler.cpp
  ${DFLASH}/common/resident_batch_coordinator.cpp
  ${DFLASH}/common/sampler.cpp
  ${DFLASH}/common/dspark_head.cpp
  ${DFLASH}/common/dspark_draft_compute_xdna.cpp
  ${DFLASH}/common/moe_hybrid_ffn_eval.cpp
  ${DFLASH}/common/moe_hybrid_placement.cpp
  ${DFLASH}/common/moe_hybrid_storage.cpp
  ${DFLASH}/common/moe_hybrid_stream.cpp
  ${DFLASH}/server/pre_tokenizer.cpp
  ${DFLASH}/server/tokenizer.cpp
  ${DFLASH}/server/disk_prefix_cache.cpp
  ${DFLASH}/common/cpu_embedder.cpp
  ${DFLASH}/errors.c
  ${DFLASH}/ember_create_backend.cpp)

target_include_directories(ember_engine PUBLIC ${EMBER_ENGINE_INC})
target_compile_features(ember_engine PUBLIC cxx_std_17)
target_link_libraries(ember_engine PUBLIC ggml ggml-base ${EMBER_GGML_BACKEND} ggml-cpu ZLIB::ZLIB ${CMAKE_DL_LIBS})
if(TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(ember_engine PUBLIC OpenMP::OpenMP_CXX)
endif()

# The HC kernel is compiled through HIP. Sampling uses the audited CPU path;
# the upstream NVIDIA-only geometric sampler is intentionally not vendored.
target_sources(ember_engine PRIVATE ${DFLASH}/deepseek4/deepseek4_hc_cuda.cu)
set_source_files_properties(${DFLASH}/deepseek4/deepseek4_hc_cuda.cu
  PROPERTIES LANGUAGE HIP)
set_target_properties(ember_engine PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
target_compile_definitions(ember_engine PRIVATE GGML_USE_HIP GGML_USE_CPU)
# The dflash sources use CUDA-compatible spellings supplied by HIP shims.
target_include_directories(ember_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat_engine)
find_package(hip QUIET)
if(hip_FOUND)
  target_link_libraries(ember_engine PUBLIC hip::host)
endif()

# Export for the top-level ember-dflash target (C server + backend_dflash.cc).
set(EMBER_ENGINE_READY ON PARENT_SCOPE)
