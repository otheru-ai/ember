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
 && test -d /opt/rocm/lib/rocblas/library \
 && runtime_rocblas=/ember-runtime/opt/rocm/lib/rocblas/library \
 && mkdir -p "$runtime_rocblas" \
 && find /opt/rocm/lib/rocblas/library -maxdepth 1 -type f \
      -name '*gfx1151*' -exec cp -a -t "$runtime_rocblas" {} + \
 && test -f "$runtime_rocblas/TensileLibrary_lazy_gfx1151.dat" \
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
      org.opencontainers.image.vendor="OtherU"

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
    EMBER_KV_CACHE_DIR=/cache \
    EMBER_MODEL_REPO=otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF \
    EMBER_MODEL_FILE=DeepSeek-V4-Flash-0731-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf \
    EMBER_MODEL_REVISION=f6e507774f7133568f6fec0635057cb20134f307 \
    EMBER_MODEL_SHA256=18aec8c0be4087007e557aa6020b28f12cd4c5d1f9c67b2a815c152aea97b3ed \
    EMBER_MODEL_SIZE_BYTES=91547243104

ENTRYPOINT ["/usr/local/bin/ember-entrypoint"]
