# syntax=docker/dockerfile:1
#
# `dev` is the full ROCm source/debug image. `release` contains only the server
# and its gfx1151 runtime closure. The optional `dev-xdna`/`release-xdna`
# stages add the current resident DSpark XDNA2 path; historical kernel
# generations and validators are deliberately not shipped.

ARG ROCM_DEV_IMAGE=rocm/dev-ubuntu-24.04:10.0.0-full@sha256:a90cf047f615abe70fbef83c64def0a2d549ef37a39c8ea545430aba4981b374
ARG RUNTIME_IMAGE=ubuntu:24.04@sha256:019e8eb29a85e74d64925745884f2ec79aa27e3feab36353d24656f4d6b89467
ARG EMBER_VERSION=dev
ARG EMBER_VCS_REF=unknown
ARG XDNA_DRIVER_REF=cdabc45a546d3e597976e5d70f0445bc5e4dc07d
ARG HYPERLOOM_REF=434b26b958761cb5f6f323be7add4c795fc70d00
ARG MLIR_AIE_VERSION=1.3.4
ARG LLVM_AIE_VERSION=22.0.0.2026081201+046b11f9

FROM ${ROCM_DEV_IMAGE} AS toolchain

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates ccache cmake curl \
 && rm -rf /var/lib/apt/lists/*

RUN echo /opt/rocm/lib > /etc/ld.so.conf.d/10-rocm.conf \
 && ldconfig

FROM toolchain AS dev

# Native JPEG decoder dependency; keep unrelated toolchain layers reusable.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libturbojpeg0-dev \
 && rm -rf /var/lib/apt/lists/*

ARG EMBER_VERSION
ARG EMBER_VCS_REF
ARG EMBER_ROCMI4_W4A8_IU4=OFF
ARG EMBER_ROCMI4_W4A8_IU4_PREPACK=OFF
ARG EMBER_HIP_EXPORT_METRICS=OFF
LABEL org.opencontainers.image.title="Ember development toolchain" \
      org.opencontainers.image.description="ROCm development and profiling image for Ember on AMD Strix Halo" \
      org.opencontainers.image.version="${EMBER_VERSION}" \
      org.opencontainers.image.revision="${EMBER_VCS_REF}" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="OtherU" \
      org.opencontainers.image.source="https://github.com/otheru-ai/ember"

# Kernel profiling is a first-class dev-image capability (scripts/profile_gpu.sh,
# and the rocprofv3 traces cited in quantize.cu and tools/bench_decode_attn.hip).
# Both profilers arrive with the pinned ROCm full image rather than from apt.
# ROCm Compute Profiler 3.8.0 is load-bearing for its corrected gfx1150/1151/
# 1152 roofline model. Keep these assertions before COPY so a mutable-tag or
# package-content regression fails fast and stays cached.
RUN command -v rocprofv3 >/dev/null 2>&1 \
 && rocprofv3 --version 2>&1 | grep -F 'rocm_version: 10.0.0' >/dev/null \
 && command -v rocprof-compute >/dev/null 2>&1 \
 && rocprof-compute --version 2>&1 | grep -F 'rocprofiler-compute version: 3.8.0' >/dev/null \
 && install -d /opt/rocm/.info \
 && printf '10.0.0\n' > /opt/rocm/.info/version \
 || { echo 'ROCm 10.0 profiling toolchain missing or version-mismatched' >&2; exit 1; }

# Hyperloom (AMD-AGI): agentic inference-optimization toolkit -- TraceLens trace
# analysis, the GEAK kernel optimizer, and the Arbor search engine. Dev image
# only; collect-runtime.sh keeps profilers and tooling out of the release
# closure and that stays true.
#
# Scope note, so nobody assumes more than it does: upstream targets MI300X/
# MI325X/MI355X (CDNA) and drives vLLM/SGLang, none of which apply here. What
# transfers to gfx1151 is the kernel-analysis half -- comparing HIP kernel
# variants and mapping profiler names back to source, which is exactly the loop
# scripts/profile_gpu.sh feeds. Treat the serving-side automation as untested on
# this hardware.
#
# Installed into its own venv rather than the system interpreter: Ubuntu 24.04
# marks its Python externally-managed (PEP 668), and an isolated prefix keeps a
# large pure-Python dependency tree away from the ROCm toolchain's own Python.
#
# Deliberately NOT the [runtime] extra. That pulls openai-codex, whose
# codex_cli_bin is a 301 MB vendored binary for OpenAI-only deployments; this
# is a Claude deployment, so the Claude SDK alone is installed instead --
# 417 MB rather than 738 MB, both entry points verified working. `packaging` is
# listed explicitly because upstream imports it (orchestrator/knowledge/
# recipe_kb_t0.py) without declaring it, and only got away with that because
# openai-codex happened to pull it in.
ARG HYPERLOOM_REF
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3-venv git time \
 && python3 -m venv /opt/hyperloom \
 && /opt/hyperloom/bin/pip install --no-cache-dir \
      "hyperloom-inference_optimizer[web] @ git+https://github.com/AMD-AGI/Hyperloom@${HYPERLOOM_REF}" \
      "claude-agent-sdk>=0.2.110" "PyYAML>=6.0" "httpx>=0.27" "packaging>=23.0" \
 && ln -s /opt/hyperloom/bin/inference_optimizer /usr/local/bin/inference_optimizer \
 && ln -s /opt/hyperloom/bin/hyperloom /usr/local/bin/hyperloom \
 && /opt/hyperloom/bin/inference_optimizer --help >/dev/null \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /ember
COPY . /ember
ENV CCACHE_DIR=/root/.cache/ccache \
    CCACHE_COMPRESS=true \
    CCACHE_MAXSIZE=20G

# Keep the complete current debug toolset in ember-dev. Only ember-dflash is
# copied into the release runtime closure below.
RUN --mount=type=cache,id=ember-gfx1151-ccache,target=/root/.cache/ccache,sharing=locked \
    cmake -S /ember -B /ember/build-rocm \
      -DCMAKE_BUILD_TYPE=Release \
      -DEMBER_ENGINE=ON \
      -DEMBER_CONFIGURED_GIT_HEAD="${EMBER_VCS_REF}" \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_HIP_ROCMI4_W4A8_IU4="${EMBER_ROCMI4_W4A8_IU4}" \
      -DGGML_HIP_ROCMI4_W4A8_IU4_PREPACK="${EMBER_ROCMI4_W4A8_IU4_PREPACK}" \
      -DGGML_HIP_EXPORT_METRICS="${EMBER_HIP_EXPORT_METRICS}" \
 && cmake --build /ember/build-rocm \
      --target ember-dflash ember-dspark-main-bench ember-dspark-xdna-diff \
      ember-dspark-cpu-bench ember-dspark-rocmfp4-cpu-bench \
      ember-gguf-quantize ember-token-dump -j"$(nproc)" \
 && cmake --build /ember/build-rocm --target check-engine-cpu -j"$(nproc)" \
 && ccache --show-stats

RUN gcc -shared -fPIC -O2 -g \
      -o /usr/local/lib/libsegvtrace.so tools/segvtrace.c -ldl \
 && install -Dm0755 -s /ember/build-rocm/ember-dflash \
      /usr/local/bin/ember-dflash \
 && install -Dm0755 -s /ember/build-rocm/ember-dspark-main-bench \
      /usr/local/bin/ember-dspark-main-bench \
 && install -Dm0755 -s /ember/build-rocm/ember-dspark-xdna-diff \
      /usr/local/bin/ember-dspark-xdna-diff \
 && install -Dm0755 -s /ember/build-rocm/ember-dspark-cpu-bench \
      /usr/local/bin/ember-dspark-cpu-bench \
 && install -Dm0755 -s /ember/build-rocm/ember-dspark-rocmfp4-cpu-bench \
      /usr/local/bin/ember-dspark-rocmfp4-cpu-bench \
 && strip --strip-unneeded /usr/local/lib/libsegvtrace.so \
 && bash docker/collect-runtime.sh /ember-runtime \
      /usr/local/bin/ember-dflash \
      /usr/local/lib/libsegvtrace.so \
      /opt/rocm/bin/rocminfo \
 && install -Dm0644 /opt/rocm/.info/version /ember-runtime/opt/rocm/.info/version \
 && rocblas_kpack="$(find /opt/rocm -type f \
      -path '*/.kpack/blas_lib_gfx1151.kpack' -print -quit)" \
 && test -n "$rocblas_kpack" \
 && runtime_kpack_dir="/ember-runtime$(dirname "$rocblas_kpack")" \
 && mkdir -p "$runtime_kpack_dir" \
 && cp -a "$rocblas_kpack" "$runtime_kpack_dir/" \
 && test -f "$runtime_kpack_dir/blas_lib_gfx1151.kpack" \
 && test -f /opt/rocm/lib/rocblas/library/gfx1151/TensileLibrary_lazy_gfx1151.dat \
 && runtime_rocblas=/ember-runtime/opt/rocm/lib/rocblas/library \
 && mkdir -p "$runtime_rocblas" \
 && cp -a /opt/rocm/lib/rocblas/library/gfx1151 "$runtime_rocblas/" \
 && test -f "$runtime_rocblas/gfx1151/TensileLibrary_lazy_gfx1151.dat" \
 && if [ -d /opt/rocm/share/rocblas ]; then \
      mkdir -p /ember-runtime/opt/rocm/share; \
      cp -a /opt/rocm/share/rocblas /ember-runtime/opt/rocm/share/; \
    fi \
 && du -sh /ember-runtime

RUN install -Dm0644 /usr/share/doc/libturbojpeg0/copyright /ember-licenses/libjpeg-turbo/copyright \
 && install -Dm0644 LICENSE /ember-licenses/ember/LICENSE \
 && install -Dm0644 engine/ggml/LICENSE /ember-licenses/ggml/LICENSE \
 && install -Dm0644 engine/dflash/LICENSE /ember-licenses/dflash/LICENSE \
 && install -Dm0644 engine/third_party/nlohmann/LICENSE.MIT \
      /ember-licenses/nlohmann/LICENSE.MIT \
 && install -Dm0644 vendor/xgrammar/LICENSE \
      /ember-licenses/xgrammar/LICENSE \
 && install -Dm0644 vendor/xgrammar/NOTICE \
      /ember-licenses/xgrammar/NOTICE

CMD ["/bin/bash"]

FROM ${RUNTIME_IMAGE} AS release

ARG EMBER_VERSION
ARG EMBER_VCS_REF
LABEL org.opencontainers.image.title="Ember" \
      org.opencontainers.image.description="DeepSeek-V4-Flash inference server for AMD Strix Halo" \
      org.opencontainers.image.version="${EMBER_VERSION}" \
      org.opencontainers.image.revision="${EMBER_VCS_REF}" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="OtherU" \
      org.opencontainers.image.source="https://github.com/otheru-ai/ember"

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl python3 \
 && rm -rf /var/lib/apt/lists/* \
 && (getent group video >/dev/null || echo 'video:x:44:' >> /etc/group) \
 && (getent group render >/dev/null || echo 'render:x:109:' >> /etc/group) \
 && mkdir -p /models /cache

COPY --from=dev /ember-runtime/ /
COPY --from=dev /ember-licenses/ /usr/share/licenses/ember/
# The vision behavioural gate travels with the image it certifies. The certify
# job deliberately checks out nothing onto the machine that holds production, so
# a gate that lived only in the repo could not run there -- and a gate run from
# a different revision than the image would not be evidence about that image.
# The corpus images stay outside; the manifest binds each one by SHA-256.
COPY scripts/ds4_vision_behavior_gate.py /opt/ember/vision-gate/gate.py
COPY share/vision_gate/policy-v4.json /opt/ember/vision-gate/policy-v4.json
COPY docker/entrypoint.sh /usr/local/bin/ember-entrypoint
RUN chmod 0755 /usr/local/bin/ember-entrypoint \
 && printf '%s\n' /opt/rocm/lib \
      > /etc/ld.so.conf.d/10-ember-runtime.conf \
 && ldconfig \
 && ! ldd /usr/local/bin/ember-dflash | grep -q 'not found'

ENV EMBER_MODEL_DIR=/models \
    EMBER_KV_CACHE_DIR=/cache

ENTRYPOINT ["/usr/local/bin/ember-entrypoint"]

# XRT userspace is pinned to the same amdxdna tree used for host validation.
# The host owns amdxdna.ko, firmware, IOMMU policy, and /dev/accel/accel0.
FROM ${RUNTIME_IMAGE} AS xdna-userspace-build

ARG XDNA_DRIVER_REF

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates git jq \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/amd/xdna-driver.git /xdna-driver \
 && git -C /xdna-driver checkout "${XDNA_DRIVER_REF}" \
 && git -C /xdna-driver submodule update --init --recursive

RUN /xdna-driver/xrt/src/runtime_src/tools/scripts/xrtdeps.sh -docker

RUN cd /xdna-driver/xrt/build \
 && ./build.sh -npu -opt -noctest -noinit -j "$(nproc)" \
 && cp -a /xdna-driver/xrt/build/Release/opt/xilinx /opt/ \
 && echo /opt/xilinx/xrt/lib > /etc/ld.so.conf.d/20-xrt.conf \
 && ldconfig

RUN cmake -S /xdna-driver -B /xdna-driver/build/EmberRelease \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/xilinx/xrt \
      -DSKIP_KMOD=ON \
      -DBUILD_VXDNA=OFF \
 && cmake --build /xdna-driver/build/EmberRelease -j"$(nproc)" \
 && DESTDIR=/xdna-plugin cmake --install /xdna-driver/build/EmberRelease \
 && mkdir -p /xdna-runtime/opt/xilinx \
 && cp -a /xdna-driver/xrt/build/Release/opt/xilinx/xrt \
      /xdna-runtime/opt/xilinx/ \
 && cp -a /xdna-plugin/opt/xilinx/xrt/. \
      /xdna-runtime/opt/xilinx/xrt/ \
 && install -Dm0644 /xdna-driver/LICENSE.amdnpu \
      /xdna-licenses/amd-xdna-driver/LICENSE.amdnpu \
 && install -Dm0644 /xdna-driver/xrt/LICENSE \
      /xdna-licenses/xrt/LICENSE

FROM ${RUNTIME_IMAGE} AS xdna-kernel-build

ARG MLIR_AIE_VERSION
ARG LLVM_AIE_VERSION

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates libboost-filesystem1.83.0 \
      libboost-program-options1.83.0 make python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /iron

RUN /iron/bin/pip install \
      "mlir_aie==${MLIR_AIE_VERSION}" \
      "llvm-aie==${LLVM_AIE_VERSION}" \
      -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4 \
      -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

COPY --from=xdna-userspace-build /xdna-runtime/ /
COPY providers/xdna2/kernel/ /kernel/

ENV PATH=/iron/bin:/iron/lib/python3.12/site-packages/mlir_aie/bin:/opt/xilinx/xrt/bin:$PATH \
    PYTHONPATH=/iron/lib/python3.12/site-packages/mlir_aie/python \
    LD_LIBRARY_PATH=/iron/lib/python3.12/site-packages/mlir_aie/lib:/opt/xilinx/xrt/lib \
    MLIR_AIE_INSTALL_DIR=/iron/lib/python3.12/site-packages/mlir_aie \
    PEANO_INSTALL_DIR=/iron/lib/python3.12/site-packages/llvm-aie

RUN make -C /kernel K=4096 N=2048 RESIDENT_MODE=projection \
      BUILD_DIR=/kernel/build/resident-projection \
 && make -C /kernel K=1024 N=32768 RESIDENT_MODE=projection \
      BUILD_DIR=/kernel/build/resident-qb \
 && make -C /kernel K=4096 N=16384 RESIDENT_MODE=projection \
      BUILD_DIR=/kernel/build/resident-oa-grouped \
 && make -C /kernel K=8192 N=4096 RESIDENT_MODE=projection \
      BUILD_DIR=/kernel/build/resident-ob \
 && make -C /kernel K=4096 N=2048 RESIDENT_MODE=shared \
      BUILD_DIR=/kernel/build/resident-shared \
 && cmp /kernel/build/resident-projection/dspark_resident_v1.mlir.prj/main.pdi \
      /kernel/build/resident-qb/dspark_resident_v1.mlir.prj/main.pdi \
 && cmp /kernel/build/resident-projection/dspark_resident_v1.mlir.prj/main.pdi \
      /kernel/build/resident-oa-grouped/dspark_resident_v1.mlir.prj/main.pdi \
 && cmp /kernel/build/resident-projection/dspark_resident_v1.mlir.prj/main.pdi \
      /kernel/build/resident-ob/dspark_resident_v1.mlir.prj/main.pdi \
 && cmp /kernel/build/resident-projection/dspark_resident_v1.mlir.prj/main.pdi \
      /kernel/build/resident-shared/dspark_resident_v1.mlir.prj/main.pdi \
 && install -Dm0644 /kernel/build/resident-projection/dspark_resident_v1_projection_4096x2048_b5.xclbin \
      /xdna-artifacts/dspark_resident_q8_v1_b5.xclbin \
 && install -Dm0644 /kernel/build/resident-projection/dspark_resident_v1_projection_4096x2048_b5.insts \
      /xdna-artifacts/dspark_resident_q8_v1_projection_4096x2048_b5.insts \
 && install -Dm0644 /kernel/build/resident-qb/dspark_resident_v1_projection_1024x32768_b5.insts \
      /xdna-artifacts/dspark_resident_q8_v1_projection_1024x32768_b5.insts \
 && install -Dm0644 /kernel/build/resident-oa-grouped/dspark_resident_v1_projection_4096x16384_b5.insts \
      /xdna-artifacts/dspark_resident_q8_v1_projection_4096x16384_b5.insts \
 && install -Dm0644 /kernel/build/resident-ob/dspark_resident_v1_projection_8192x4096_b5.insts \
      /xdna-artifacts/dspark_resident_q8_v1_projection_8192x4096_b5.insts \
 && install -Dm0644 /kernel/build/resident-shared/dspark_resident_v1_shared_4096x2048_b5.insts \
      /xdna-artifacts/dspark_resident_q8_v1_shared_4096x2048x4096_b5.insts

FROM dev AS dev-xdna

COPY --from=xdna-userspace-build /xdna-runtime/ /
COPY --from=xdna-kernel-build /xdna-artifacts/ \
  /usr/local/share/ember/xdna2/
ENV EMBER_XDNA_ARTIFACT_DIR=/usr/local/share/ember/xdna2
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      uuid-dev \
 && rm -rf /var/lib/apt/lists/* \
 && echo /opt/xilinx/xrt/lib > /etc/ld.so.conf.d/20-xrt.conf \
 && ldconfig \
 && cmake -S /ember -B /ember/build-xdna-provider \
      -DCMAKE_BUILD_TYPE=Release \
      -DEMBER_XDNA_XRT_PROVIDER=ON \
      -DEMBER_XRT_ROOT=/opt/xilinx/xrt \
 && cmake --build /ember/build-xdna-provider \
      --target ember_xdna_dspark_xrt ember-xdna-dspark-resident-validate \
      ember-xdna-dspark-overlay-validate -j"$(nproc)" \
 && install -Dm0755 /ember/build-xdna-provider/libember_xdna_dspark.so \
      /xdna-provider/libember_xdna_dspark.so \
 && install -Dm0755 /ember/build-xdna-provider/ember-xdna-dspark-resident-validate \
      /usr/local/bin/ember-xdna-dspark-resident-validate \
 && install -Dm0755 /ember/build-xdna-provider/ember-xdna-dspark-overlay-validate \
      /usr/local/bin/ember-xdna-dspark-overlay-validate

FROM release AS release-xdna

LABEL org.opencontainers.image.description="Ember inference server with experimental CPU/GPU/XDNA2 DSpark provider"

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libboost-filesystem1.83.0 libboost-program-options1.83.0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=xdna-userspace-build /xdna-runtime/ /
COPY --from=xdna-userspace-build /xdna-licenses/ /usr/share/licenses/ember/
COPY --from=dev-xdna /xdna-provider/libember_xdna_dspark.so \
  /usr/local/lib/ember/libember_xdna_dspark.so
COPY --from=xdna-kernel-build /xdna-artifacts/ \
  /usr/local/share/ember/xdna2/
COPY providers/xdna2/VENDOR.md /usr/share/doc/ember/xdna2/VENDOR.md
COPY providers/xdna2/LLVM-EXCEPTION.txt \
  /usr/share/licenses/ember/tilefuse/LLVM-EXCEPTION.txt
COPY vendor/xgrammar/LICENSE \
  /usr/share/licenses/ember/tilefuse/LICENSE.Apache-2.0

RUN echo /opt/xilinx/xrt/lib > /etc/ld.so.conf.d/20-xrt.conf \
 && ldconfig \
 && ! ldd /usr/local/lib/ember/libember_xdna_dspark.so | grep -q 'not found' \
 && ! ldd /opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2 | grep -q 'not found' \
 && /opt/xilinx/xrt/bin/xrt-smi --version

ENV EMBER_XDNA_ARTIFACT_DIR=/usr/local/share/ember/xdna2
