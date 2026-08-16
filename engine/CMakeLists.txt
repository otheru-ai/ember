# Ember standalone model-compute engine.
#
# Phase 1 (this file): build the vendored ggml fork (with the ROCMFP4 / ROCMFPX
# kernels) as ggml-base + ggml-hip (or ggml-cuda). Phase 2 adds ember_engine.a
# (the DeepSeek4 backend + the common/ subset it needs) linking that backend.
#
# The parent CMakeLists has already enable_language()'d HIP/CUDA and resolved
# CMAKE_HIP_ARCHITECTURES. See engine/VENDOR.md for provenance / porting.

# ── trim ggml to just the compute backend we use ────────────────────────
# Mirrors lucebox/server/CMakeLists.txt: no metal/vulkan/blas/opencl, no
# runtime backend DL, no tests/examples. Backend selection is driven by the
# parent's EMBER_ENGINE_BACKEND.
if(EMBER_ENGINE_BACKEND STREQUAL "hip")
  set(GGML_HIP        ON  CACHE BOOL "" FORCE)
  set(GGML_CUDA       OFF CACHE BOOL "" FORCE)
  set(GGML_HIP_RCCL   OFF CACHE BOOL "" FORCE)
  # FORCE this OFF: an ordinary option default does not overwrite an existing
  # build-tree cache entry, so incremental builds made after 4beb51e could keep
  # GGML_HIP_GRAPHS=ON even after f026c35 removed the enabling assignment.
  set(GGML_HIP_GRAPHS OFF CACHE BOOL "" FORCE)
  # HIP graph replay stays OFF — MEASURED, not assumed. USE_CUDA_GRAPH comes from
  # GGML_CUDA_USE_GRAPHS || GGML_HIP_GRAPHS || GGML_MUSA_GRAPHS
  # (ggml-cuda/common.cuh), so the cuda branch below gets replay and hip does not.
  # Turning it ON looks obviously right (a fused 61-layer decode graph otherwise
  # launches every kernel individually per token) but A/B on gfx1151 with the
  # production model, same binary via GGML_CUDA_DISABLE_GRAPHS, 3 steady-state
  # requests each, was a REGRESSION:
  #     graphs ON : prefill 22.9 tok/s, decode 21.39 tok/s
  #     graphs OFF: prefill 24.6 tok/s, decode 22.29 tok/s   (+7.4% / +4.2%)
  # Cause is visible in GGML_CUDA_GRAPH_STATS: continuous "warmup complete" /
  # "warmup reset" churn plus [graph-mismatch] node=0 op=REPEAT. The graph key
  # never stabilizes, so capture+validation cost is paid repeatedly and steady
  # replay is never reached. The fused decode slot key includes per-layer padded
  # comp rows, which grow monotonically (DS4_COMP_PAD_STRIDE=16), so the key
  # changes roughly every 64 tokens by construction.
  # DO NOT re-enable this until the graph key is made stable (e.g. a much larger
  # DS4_COMP_PAD_STRIDE, or keying on the max padded value across layers);
  # graph replay cannot pay off while every bucket crossing invalidates it.
  set(EMBER_GGML_BACKEND ggml-hip)
elseif(EMBER_ENGINE_BACKEND STREQUAL "cuda")
  set(GGML_CUDA        ON  CACHE BOOL "" FORCE)
  set(GGML_HIP         OFF CACHE BOOL "" FORCE)
  set(GGML_CUDA_GRAPHS ON  CACHE BOOL "" FORCE)  # AR-decode replay (lucebox)
  set(EMBER_GGML_BACKEND ggml-cuda)
endif()
set(EMBER_GGML_BACKEND "${EMBER_GGML_BACKEND}" PARENT_SCOPE)  # Phase 2 links this

set(GGML_BACKEND_DL     OFF CACHE BOOL "" FORCE)
set(GGML_METAL          OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN         OFF CACHE BOOL "" FORCE)
set(GGML_BLAS           OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL         OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
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

if(EMBER_ENGINE_BACKEND STREQUAL "hip")
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
elseif(EMBER_ENGINE_BACKEND STREQUAL "cuda")
  # CUDA-only ggml sources include <cuda_runtime.h> directly.
  find_package(CUDAToolkit REQUIRED)
endif()

message(STATUS
  "[ember] engine: backend=${EMBER_ENGINE_BACKEND} "
  "hip_archs=${CMAKE_HIP_ARCHITECTURES} cuda_archs=${CMAKE_CUDA_ARCHITECTURES} "
  "ggml_target=${EMBER_GGML_BACKEND}")

# ── ember_engine: the DeepSeek4 forward pass (vendored dflash subset) ─────
# The transitive #include+source closure of the monolithic DeepSeek4 backend
# (see engine/VENDOR.md). Multi-GPU layer-split / shard-IPC / other-arch files
# are intentionally excluded; ember_create_backend.cpp replaces the multi-arch
# backend_factory.cpp. Include dirs mirror lucebox's DFLASH27B_SRC_INCLUDE_DIRS.
set(DFLASH ${CMAKE_CURRENT_SOURCE_DIR}/dflash)
set(EMBER_ENGINE_INC
  ${DFLASH} ${DFLASH}/common ${DFLASH}/deepseek4
  ${DFLASH}/server ${DFLASH}/placement ${DFLASH}/include ${DFLASH}/qwen35
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_library(ember_engine STATIC
  ${DFLASH}/deepseek4/deepseek4_loader.cpp
  ${DFLASH}/deepseek4/deepseek4_graph.cpp
  ${DFLASH}/deepseek4/deepseek4_backend.cpp
  ${DFLASH}/deepseek4/deepseek4_dspark.cpp
  ${DFLASH}/deepseek4/deepseek4_dspark_spec.cpp
  ${DFLASH}/common/crash_breadcrumb.cpp
  ${DFLASH}/common/continuous_batch_executor.cpp
  ${DFLASH}/common/continuous_batch_scheduler.cpp
  ${DFLASH}/common/resident_batch_coordinator.cpp
  ${DFLASH}/common/sampler.cpp
  ${DFLASH}/common/ddtree.cpp
  ${DFLASH}/common/dspark_head.cpp
  ${DFLASH}/common/layer_split_utils.cpp
  ${DFLASH}/common/peer_access.cpp
  ${DFLASH}/common/moe_expert_compute.cpp
  ${DFLASH}/common/moe_expert_compute_xdna.cpp
  ${DFLASH}/common/dspark_draft_compute_xdna.cpp
  ${DFLASH}/common/moe_hybrid_ffn_eval.cpp
  ${DFLASH}/common/moe_hybrid_placement.cpp
  ${DFLASH}/common/moe_hybrid_routing_stats.cpp
  ${DFLASH}/common/moe_hybrid_storage.cpp
  ${DFLASH}/common/moe_hybrid_stream.cpp
  ${DFLASH}/placement/placement_config.cpp
  ${DFLASH}/server/pre_tokenizer.cpp
  ${DFLASH}/server/tokenizer.cpp
  ${DFLASH}/server/prefix_cache.cpp
  ${DFLASH}/server/disk_prefix_cache.cpp
  # Link-time infrastructure (base classes / helpers defined in "other" files):
  ${DFLASH}/errors.cpp                     # set_last_error
  ${DFLASH}/common/daemon_loop.cpp         # DaemonIO + ModelBackend base methods
  ${DFLASH}/qwen35/gguf_target_loader.cpp  # CpuEmbedder (generic despite the dir)
  ${DFLASH}/ember_create_backend.cpp)

target_include_directories(ember_engine PUBLIC ${EMBER_ENGINE_INC})
target_compile_features(ember_engine PUBLIC cxx_std_17)
target_link_libraries(ember_engine PUBLIC ggml ggml-base ${EMBER_GGML_BACKEND} ggml-cpu ${CMAKE_DL_LIBS})

if(EMBER_ENGINE_BACKEND STREQUAL "hip")
  # Only the HC kernel is HIP. geometric_sampler_cuda.cu is CUDA-only in lucebox
  # (it uses 32-bit warp-shuffle masks that ROCm 7.x wave64 rejects); on HIP,
  # sampler.cpp takes the CPU path because DFLASH27B_HAVE_GPU_SAMPLER is left
  # undefined. The .cu is vendored (for the cuda backend) but not compiled here.
  target_sources(ember_engine PRIVATE ${DFLASH}/deepseek4/deepseek4_hc_cuda.cu)
  set_source_files_properties(${DFLASH}/deepseek4/deepseek4_hc_cuda.cu
    PROPERTIES LANGUAGE HIP)
  set_target_properties(ember_engine PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
  target_compile_definitions(ember_engine PRIVATE DFLASH27B_BACKEND_HIP=1 GGML_USE_HIP GGML_USE_CPU)
  # The dflash sources use CUDA spellings (<cuda_runtime.h>, cuda_bf16/fp16, mma).
  # hip_compat_engine is lucebox's server/hip_compat (has cuda_runtime.h); the
  # other engine/hip_compat (server/src/hip_compat) is ggml-hip's and lacks it.
  target_include_directories(ember_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat_engine)
  find_package(hip QUIET)
  if(hip_FOUND)
    target_link_libraries(ember_engine PUBLIC hip::host)
  endif()
elseif(EMBER_ENGINE_BACKEND STREQUAL "cuda")
  target_sources(ember_engine PRIVATE
    ${DFLASH}/deepseek4/deepseek4_hc_cuda.cu
    ${DFLASH}/common/geometric_sampler_cuda.cu)
  set_target_properties(ember_engine PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(ember_engine PRIVATE DFLASH27B_BACKEND_CUDA=1 GGML_USE_CUDA GGML_USE_CPU)
endif()

# Export for the top-level ember-dflash target (C server + backend_dflash.cc).
set(EMBER_ENGINE_READY ON PARENT_SCOPE)
