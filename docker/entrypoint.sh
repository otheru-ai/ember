#!/usr/bin/env bash
set -euo pipefail

model_dir="${EMBER_MODEL_DIR:-/models}"
server_bin="${EMBER_SERVER_BIN:-/usr/local/bin/ember-dflash}"
repo="otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF"
revision="9fe32d8d4a1abed16c84e2636b26950232869929"
file="DeepSeek-V4-Flash-0731-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf"
expected_sha256="a936e0a514385c8ae964c0f42263a4314a34fbc6efea9d9aced5320f320a3d54"
expected_size="91547243200"
draft_file="DeepSeek-V4-Flash-0731-Abliterated-DSpark-draft-4.25bpw.gguf"
draft_expected_sha256="1a01c80eceae302bcc1d70836759ee97974d7983c5084ef43f6ef772a8970ae6"
draft_expected_size="10897111840"
model="$model_dir/$file"
draft="$model_dir/$draft_file"
model_verified=0
draft_verified=0

die() {
  echo "ember: $*" >&2
  exit 78
}

if [[ "${EMBER_SKIP_DEVICE_CHECK:-0}" != 1 ]]; then
  [[ -r /dev/kfd ]] || die "/dev/kfd is unavailable; pass the AMD KFD device to the container"
  [[ -d /dev/dri ]] || die "/dev/dri is unavailable; pass the DRM devices to the container"

  if [[ -x /opt/rocm/bin/rocminfo ]]; then
    rocminfo_output="$(/opt/rocm/bin/rocminfo 2>&1)" || {
      echo "$rocminfo_output" >&2
      die "ROCm cannot enumerate the GPU"
    }
    grep -q 'gfx1151' <<<"$rocminfo_output" ||
      die "no gfx1151 (AMD Strix Halo) agent was found; this image is target-specific"
  fi
fi

if [[ "${EMBER_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "ember: device preflight passed"
  exit 0
fi

verify_sha256() {
  local path="$1"
  local digest="$2"
  local label="$3"
  echo "ember: verifying SHA-256 for $(basename "$path")"
  printf '%s  %s\n' "$digest" "$path" | sha256sum --check --status ||
    die "$label SHA-256 mismatch: $path (remove the file and download it again)"
}

check_download_space() {
  local partial="$1"
  local size="$2"
  local have=0
  [[ -f "$partial" ]] && have="$(stat -c %s "$partial")"
  local remaining=$((size > have ? size - have : 0))
  local reserve=$((2 * 1024 * 1024 * 1024))
  local available
  available="$(df -PB1 "$model_dir" | awk 'NR == 2 {print $4}')"
  if [[ "$available" =~ ^[0-9]+$ ]] && (( available < remaining + reserve )); then
    die "insufficient free space in $model_dir: need $((remaining + reserve)) bytes, have $available"
  fi
}

download_artifact() {
  local artifact_file="$1"
  local artifact_size="$2"
  local artifact_sha256="$3"
  local label="$4"
  local destination="$model_dir/$artifact_file"
  local partial="$destination.part"
  mkdir -p "$model_dir"
  check_download_space "$partial" "$artifact_size"
  echo "ember: downloading $repo@$revision/$artifact_file (resume is enabled)"
  local partial_size=0
  [[ -f "$partial" ]] && partial_size="$(stat -c %s "$partial")"
  if [[ "$partial_size" != "$artifact_size" ]]; then
    curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
      --output "$partial" \
      "https://huggingface.co/$repo/resolve/$revision/$artifact_file"
  fi
  verify_sha256 "$partial" "$artifact_sha256" "$label"
  mv "$partial" "$destination"
  downloaded_artifact="$destination"
}

if [[ ! -r "$model" && "$model" == "$model_dir/$file" && \
      "${EMBER_AUTO_DOWNLOAD:-1}" == 1 ]]; then
  download_artifact "$file" "$expected_size" "$expected_sha256" model
  model_verified=1
  model="$downloaded_artifact"
fi

if [[ ! -r "$model" ]]; then
  echo "ember: model is not readable: $model" >&2
  exit 66
fi

if [[ "$model_verified" != 1 ]]; then
  verify_sha256 "$model" "$expected_sha256" model
fi

if [[ ! -r "$draft" && "$draft" == "$model_dir/$draft_file" && \
      "${EMBER_AUTO_DOWNLOAD:-1}" == 1 ]]; then
  download_artifact "$draft_file" "$draft_expected_size" \
    "$draft_expected_sha256" "draft model"
  draft_verified=1
  draft="$downloaded_artifact"
fi

if [[ ! -r "$draft" ]]; then
  echo "ember: draft model is not readable: $draft" >&2
  exit 66
fi
if [[ "$draft_verified" != 1 ]]; then
  verify_sha256 "$draft" "$draft_expected_sha256" "draft model"
fi
export DFLASH_DS4_SPEC=1
export DFLASH_DS4_DRAFT="$draft"

segvtrace="${EMBER_SEGVTRACE:-/usr/local/lib/libsegvtrace.so}"
if [[ -n "$segvtrace" && -r "$segvtrace" ]]; then
  export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$segvtrace"
fi

server_args=()
if [[ -n "${EMBER_TOOL_LOOP_REPORT:-}" ]]; then
  server_args+=(--tool-loop-report "$EMBER_TOOL_LOOP_REPORT")
fi
if [[ -n "${EMBER_NO_PROGRESS_REPORT:-}" ]]; then
  server_args+=(--no-progress-report "$EMBER_NO_PROGRESS_REPORT")
fi
if [[ -n "${EMBER_AUTO_ANSWER_AFTER_LOOP:-}" ]]; then
  server_args+=(--auto-answer-after-loop "$EMBER_AUTO_ANSWER_AFTER_LOOP")
fi

exec "$server_bin" \
  -m "$model" \
  --kv-cache-dir "${EMBER_KV_CACHE_DIR:-/cache}" \
  --port "${EMBER_PORT:-8080}" \
  "${server_args[@]}" \
  "$@"
