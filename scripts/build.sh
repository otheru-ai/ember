#!/usr/bin/env bash
# Build ember-dflash — the fully standalone binary (vendored engine, no lucebox).
#
# Replaces the old scripts/build_dflash.sh, which linked lucebox's prebuilt
# libdflash_common.a and therefore required an external lucebox checkout. Ember
# now vendors the whole model-compute stack under engine/ (ggml fork +
# DeepSeek4), so the build needs nothing outside this repo.
#
# Must run in the ROCm/HIP container — the HIP toolchain is not on the host.
#
#   docker build --target dev -f docker/Dockerfile \
#     -t ember-rocm:10.0-dev .                                  # once
#   scripts/build.sh                                             # from repo root
#
# Produces ./build-rocm/ember-dflash plus its ggml .so tree in
# ./build-rocm/engine/ggml/src/.
set -euo pipefail

IMAGE="${EMBER_IMAGE:-ember-rocm:10.0-dev}"
JOBS="${JOBS:-$(nproc)}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[build] repo=$REPO image=$IMAGE jobs=$JOBS"

docker run --rm \
  -v "$REPO":/ember -w /ember \
  "$IMAGE" \
  bash -lc "
    set -euo pipefail
    cmake -S /ember -B /ember/build-rocm \
      -DCMAKE_BUILD_TYPE=Release \
      -DEMBER_ENGINE=ON
    cmake --build /ember/build-rocm -j ${JOBS}
  "

echo "[build] done -> $REPO/build-rocm/ember-dflash"
ls -la "$REPO/build-rocm/ember-dflash" 2>/dev/null || true
