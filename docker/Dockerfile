# syntax=docker/dockerfile:1
#
# Ember has two intentionally separate image targets:
#
#   docker build --target dev     -t ember:dev     -f docker/Dockerfile .
#   docker build --target release -t ember:release -f docker/Dockerfile .
#
# `dev` contains AMD's complete ROCm compiler/SDK, the source tree, build tools,
# symbols, and build tree. `release` starts again from Ubuntu and receives only
# the stripped server, crash shim, recursive ELF dependency closure, rocBLAS
# runtime kernel data, and first-run utilities. Model weights are never copied.

ARG ROCM_DEV_IMAGE=rocm/dev-ubuntu-24.04:7.14.0-full@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7
ARG RUNTIME_IMAGE=ubuntu:24.04@sha256:019e8eb29a85e74d64925745884f2ec79aa27e3feab36353d24656f4d6b89467
ARG EMBER_VERSION=dev
ARG EMBER_VCS_REF=unknown
ARG XDNA_DRIVER_REF=455fc6be78e9cdf8b41a1547eff0e30351f21fec
ARG MLIR_AIE_VERSION=1.3.4
ARG LLVM_AIE_VERSION=22.0.0.2026081201+046b11f9

FROM ${ROCM_DEV_IMAGE} AS toolchain

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
 && rm -rf /var/lib/apt/lists/*

# ROCm 7.14's base omits the loader entry shipped by earlier complete images.
RUN echo /opt/rocm/lib > /etc/ld.so.conf.d/10-rocm.conf \
 && ldconfig

FROM toolchain AS dev

WORKDIR /ember
COPY . /ember

# Static ggml/engine linkage prevents project-built .so files and their build
# tree RPATHs from leaking into the runtime image. ROCm remains dynamically
# linked so the release stage can carry its exact, auditable runtime closure.
RUN cmake -S /ember -B /ember/build-rocm \
      -DCMAKE_BUILD_TYPE=Release \
      -DEMBER_ENGINE=ON \
      -DEMBER_ENGINE_BACKEND=hip \
      -DBUILD_SHARED_LIBS=OFF \
 && cmake --build /ember/build-rocm --target ember-dflash -j"$(nproc)"

RUN gcc -shared -fPIC -O2 -g \
      -o /usr/local/lib/libsegvtrace.so tools/segvtrace.c -ldl \
 && install -Dm0755 -s /ember/build-rocm/ember-dflash \
      /usr/local/bin/ember-dflash \
 && strip --strip-unneeded /usr/local/lib/libsegvtrace.so \
 && bash docker/collect-runtime.sh /ember-runtime \
      /usr/local/bin/ember-dflash \
      /usr/local/lib/libsegvtrace.so \
      /opt/rocm/bin/rocminfo \
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

# License texts are runtime content, not development documentation. Preserve
# the project license and every vendored notice alongside the binaries that
# incorporate those components.
RUN install -Dm0644 LICENSE /ember-licenses/ember/LICENSE \
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
      ca-certificates \
      curl \
 && rm -rf /var/lib/apt/lists/* \
 && (getent group video >/dev/null || echo 'video:x:44:' >> /etc/group) \
 && (getent group render >/dev/null || echo 'render:x:109:' >> /etc/group) \
 && mkdir -p /models /cache

COPY --from=dev /ember-runtime/ /
COPY --from=dev /ember-licenses/ /usr/share/licenses/ember/
COPY docker/entrypoint.sh /usr/local/bin/ember-entrypoint
RUN chmod 0755 /usr/local/bin/ember-entrypoint \
 && echo /opt/rocm/lib > /etc/ld.so.conf.d/10-rocm.conf \
 && ldconfig

ENV EMBER_MODEL_DIR=/models \
    EMBER_KV_CACHE_DIR=/cache

ENTRYPOINT ["/usr/local/bin/ember-entrypoint"]

# ── Optional XDNA2 userspace ──────────────────────────────────────────────
#
# The container carries userspace only. amdxdna.ko and NPU firmware must be
# installed on the host, whose /dev/accel/accel0 is passed at runtime. Building
# the shim with SKIP_KMOD avoids the common container mistake of packaging a
# DKMS module that cannot control the host kernel.
FROM ${RUNTIME_IMAGE} AS xdna-userspace-build

ARG XDNA_DRIVER_REF

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates git jq \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/amd/xdna-driver.git /xdna-driver \
 && git -C /xdna-driver checkout "${XDNA_DRIVER_REF}" \
 && git -C /xdna-driver submodule update --init --recursive

# XRT's dependency helper knows the distro-specific package set. -docker
# deliberately omits host-driver-only dependencies.
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

# Compile both DeepSeek expert projection shapes with the open Peano flow.
# xclbinutil comes from the exact XRT build above; the source remains in the
# repository so every binary in the image is reproducible and auditable.
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

RUN make -C /kernel K=4096 N=2048 BUILD_DIR=/kernel/build/4096x2048 \
 && make -C /kernel K=4096 N=2048 BUILD_DIR=/kernel/build/4096x2048 expert-v5 \
 && make -C /kernel K=4096 N=2048 BUILD_DIR=/kernel/build/4096x2048 expert-v6 \
 && make -C /kernel K=4096 N=2048 BUILD_DIR=/kernel/build/4096x2048 expert-v7 \
 && make -C /kernel K=4096 N=2048 BUILD_DIR=/kernel/build/4096x2048 expert-v8 \
 && make -C /kernel K=4096 N=2048 BUILD_DIR=/kernel/build/4096x2048 shared-q8-v2 \
 && make -C /kernel K=4096 N=1024 BUILD_DIR=/kernel/build/q8-4096x1024 q8-projection-v3 \
 && make -C /kernel K=1024 N=32768 BUILD_DIR=/kernel/build/q8-1024x32768 q8-projection-v3 \
 && make -C /kernel K=4096 N=512 BUILD_DIR=/kernel/build/q8-4096x512 q8-projection-v3 \
 && make -C /kernel K=4096 N=8192 BUILD_DIR=/kernel/build/q8-4096x8192 q8-projection-v3 \
 && make -C /kernel K=8192 N=4096 BUILD_DIR=/kernel/build/q8-8192x4096 q8-projection-v3 \
 && make -C /kernel K=2048 N=4096 BUILD_DIR=/kernel/build/2048x4096 \
 && install -Dm0644 /kernel/build/4096x2048/gemv_4096x2048.xclbin \
      /xdna-artifacts/gemv_4096x2048.xclbin \
 && install -Dm0644 /kernel/build/4096x2048/gemv_4096x2048.insts \
      /xdna-artifacts/gemv_4096x2048.insts \
 && install -Dm0644 /kernel/build/4096x2048/gemv_v2_4096x2048.xclbin \
      /xdna-artifacts/gemv_v2_4096x2048.xclbin \
 && install -Dm0644 /kernel/build/4096x2048/gemv_v2_4096x2048.insts \
      /xdna-artifacts/gemv_v2_4096x2048.insts \
 && install -Dm0644 /kernel/build/4096x2048/gemv_v3_4096x2048.xclbin \
      /xdna-artifacts/gemv_v3_4096x2048.xclbin \
 && install -Dm0644 /kernel/build/4096x2048/gemv_v3_4096x2048.insts \
      /xdna-artifacts/gemv_v3_4096x2048.insts \
 && install -Dm0644 /kernel/build/4096x2048/gemv_v4_4096x2048.xclbin \
      /xdna-artifacts/gemv_v4_4096x2048.xclbin \
 && install -Dm0644 /kernel/build/4096x2048/gemv_v4_4096x2048.insts \
      /xdna-artifacts/gemv_v4_4096x2048.insts \
 && install -Dm0644 /kernel/build/4096x2048/expert_v5_4096x2048x4096.xclbin \
      /xdna-artifacts/expert_v5_4096x2048x4096.xclbin \
 && install -Dm0644 /kernel/build/4096x2048/expert_v5_4096x2048x4096.insts \
      /xdna-artifacts/expert_v5_4096x2048x4096.insts \
 && install -Dm0644 /kernel/build/4096x2048/expert_v6_4096x2048x4096_b4.xclbin \
      /xdna-artifacts/expert_v6_4096x2048x4096_b4.xclbin \
 && install -Dm0644 /kernel/build/4096x2048/expert_v6_4096x2048x4096_b4.insts \
      /xdna-artifacts/expert_v6_4096x2048x4096_b4.insts \
 && install -Dm0644 /kernel/build/4096x2048/expert_v7_4096x2048x4096_b5.xclbin \
      /xdna-artifacts/expert_v7_4096x2048x4096_b5.xclbin \
 && install -Dm0644 /kernel/build/4096x2048/expert_v7_4096x2048x4096_b5.insts \
      /xdna-artifacts/expert_v7_4096x2048x4096_b5.insts \
 && install -Dm0644 /kernel/build/4096x2048/expert_v8_4096x2048x4096_b5.xclbin \
      /xdna-artifacts/expert_v8_4096x2048x4096_b5.xclbin \
 && install -Dm0644 /kernel/build/4096x2048/expert_v8_4096x2048x4096_b5.insts \
      /xdna-artifacts/expert_v8_4096x2048x4096_b5.insts \
 && install -Dm0644 /kernel/build/4096x2048/shared_q8_v2_4096x2048x4096_b5.xclbin \
      /xdna-artifacts/shared_q8_v2_4096x2048x4096_b5.xclbin \
 && install -Dm0644 /kernel/build/4096x2048/shared_q8_v2_4096x2048x4096_b5.insts \
      /xdna-artifacts/shared_q8_v2_4096x2048x4096_b5.insts \
 && install -Dm0644 /kernel/build/q8-4096x1024/q8_projection_v3_4096x1024_b5.xclbin \
      /xdna-artifacts/q8_projection_v3_4096x1024_b5.xclbin \
 && install -Dm0644 /kernel/build/q8-4096x1024/q8_projection_v3_4096x1024_b5.insts \
      /xdna-artifacts/q8_projection_v3_4096x1024_b5.insts \
 && install -Dm0644 /kernel/build/q8-1024x32768/q8_projection_v3_1024x32768_b5.xclbin \
      /xdna-artifacts/q8_projection_v3_1024x32768_b5.xclbin \
 && install -Dm0644 /kernel/build/q8-1024x32768/q8_projection_v3_1024x32768_b5.insts \
      /xdna-artifacts/q8_projection_v3_1024x32768_b5.insts \
 && install -Dm0644 /kernel/build/q8-4096x512/q8_projection_v3_4096x512_b5.xclbin \
      /xdna-artifacts/q8_projection_v3_4096x512_b5.xclbin \
 && install -Dm0644 /kernel/build/q8-4096x512/q8_projection_v3_4096x512_b5.insts \
      /xdna-artifacts/q8_projection_v3_4096x512_b5.insts \
 && install -Dm0644 /kernel/build/q8-4096x8192/q8_projection_v3_4096x8192_b5.xclbin \
      /xdna-artifacts/q8_projection_v3_4096x8192_b5.xclbin \
 && install -Dm0644 /kernel/build/q8-4096x8192/q8_projection_v3_4096x8192_b5.insts \
      /xdna-artifacts/q8_projection_v3_4096x8192_b5.insts \
 && install -Dm0644 /kernel/build/q8-8192x4096/q8_projection_v3_8192x4096_b5.xclbin \
      /xdna-artifacts/q8_projection_v3_8192x4096_b5.xclbin \
 && install -Dm0644 /kernel/build/q8-8192x4096/q8_projection_v3_8192x4096_b5.insts \
      /xdna-artifacts/q8_projection_v3_8192x4096_b5.insts \
 && install -Dm0644 /kernel/build/2048x4096/gemv_2048x4096.xclbin \
      /xdna-artifacts/gemv_2048x4096.xclbin \
 && install -Dm0644 /kernel/build/2048x4096/gemv_2048x4096.insts \
      /xdna-artifacts/gemv_2048x4096.insts \
 && install -Dm0644 /kernel/build/2048x4096/gemv_v2_2048x4096.xclbin \
      /xdna-artifacts/gemv_v2_2048x4096.xclbin \
 && install -Dm0644 /kernel/build/2048x4096/gemv_v2_2048x4096.insts \
      /xdna-artifacts/gemv_v2_2048x4096.insts \
 && install -Dm0644 /kernel/build/2048x4096/gemv_v3_2048x4096.xclbin \
      /xdna-artifacts/gemv_v3_2048x4096.xclbin \
 && install -Dm0644 /kernel/build/2048x4096/gemv_v3_2048x4096.insts \
      /xdna-artifacts/gemv_v3_2048x4096.insts \
 && install -Dm0644 /kernel/build/2048x4096/gemv_v4_2048x4096.xclbin \
      /xdna-artifacts/gemv_v4_2048x4096.xclbin \
 && install -Dm0644 /kernel/build/2048x4096/gemv_v4_2048x4096.insts \
      /xdna-artifacts/gemv_v4_2048x4096.insts

FROM dev AS dev-xdna

COPY --from=xdna-userspace-build /xdna-runtime/ /
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
      --target ember_xdna_moe_xrt ember-xdna-validate \
      ember-xdna-gpu-interop ember-xdna-rocmfp4-validate -j"$(nproc)" \
 && cmake --build /ember/build-xdna-provider \
      --target ember-xdna-q8-validate \
      ember-xdna-q8-projection-validate -j"$(nproc)" \
 && install -Dm0755 /ember/build-xdna-provider/libember_xdna_moe.so \
      /xdna-provider/libember_xdna_moe.so \
 && install -Dm0755 /ember/build-xdna-provider/ember-xdna-validate \
      /xdna-provider/ember-xdna-validate \
 && install -Dm0755 /ember/build-xdna-provider/ember-xdna-gpu-interop \
      /xdna-provider/ember-xdna-gpu-interop \
 && install -Dm0755 /ember/build-xdna-provider/ember-xdna-rocmfp4-validate \
      /xdna-provider/ember-xdna-rocmfp4-validate \
 && install -Dm0755 /ember/build-xdna-provider/ember-xdna-q8-validate \
      /xdna-provider/ember-xdna-q8-validate \
 && install -Dm0755 /ember/build-xdna-provider/ember-xdna-q8-projection-validate \
      /xdna-provider/ember-xdna-q8-projection-validate

FROM release AS release-xdna

LABEL org.opencontainers.image.description="DeepSeek-V4-Flash inference server with experimental AMD XDNA2 MoE provider"

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libboost-filesystem1.83.0 libboost-program-options1.83.0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=xdna-userspace-build /xdna-runtime/ /
COPY --from=xdna-userspace-build /xdna-licenses/ /usr/share/licenses/ember/
COPY --from=dev-xdna /xdna-provider/libember_xdna_moe.so \
  /usr/local/lib/ember/libember_xdna_moe.so
COPY --from=dev-xdna /xdna-provider/ember-xdna-validate \
  /usr/local/bin/ember-xdna-validate
COPY --from=dev-xdna /xdna-provider/ember-xdna-gpu-interop \
  /usr/local/bin/ember-xdna-gpu-interop
COPY --from=dev-xdna /xdna-provider/ember-xdna-rocmfp4-validate \
  /usr/local/bin/ember-xdna-rocmfp4-validate
COPY --from=dev-xdna /xdna-provider/ember-xdna-q8-validate \
  /usr/local/bin/ember-xdna-q8-validate
COPY --from=dev-xdna /xdna-provider/ember-xdna-q8-projection-validate \
  /usr/local/bin/ember-xdna-q8-projection-validate
COPY --from=xdna-kernel-build /xdna-artifacts/ \
  /usr/local/share/ember/xdna2/
COPY providers/xdna2/VENDOR.md \
  /usr/share/doc/ember/xdna2/VENDOR.md
COPY providers/xdna2/LLVM-EXCEPTION.txt \
  /usr/share/licenses/ember/tilefuse/LLVM-EXCEPTION.txt
COPY vendor/xgrammar/LICENSE \
  /usr/share/licenses/ember/tilefuse/LICENSE.Apache-2.0

RUN echo /opt/xilinx/xrt/lib > /etc/ld.so.conf.d/20-xrt.conf \
 && ldconfig \
 && ! ldd /usr/local/lib/ember/libember_xdna_moe.so | grep -q 'not found' \
 && ! ldd /usr/local/bin/ember-xdna-validate | grep -q 'not found' \
 && ! ldd /usr/local/bin/ember-xdna-gpu-interop | grep -q 'not found' \
 && ! ldd /usr/local/bin/ember-xdna-rocmfp4-validate | grep -q 'not found' \
 && ! ldd /usr/local/bin/ember-xdna-q8-validate | grep -q 'not found' \
 && ! ldd /usr/local/bin/ember-xdna-q8-projection-validate | grep -q 'not found' \
 && ! ldd /opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2 | grep -q 'not found' \
 && /opt/xilinx/xrt/bin/xrt-smi --version

ENV EMBER_XDNA_ARTIFACT_DIR=/usr/local/share/ember/xdna2
