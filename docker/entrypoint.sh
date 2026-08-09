#!/usr/bin/env bash
set -euo pipefail

model_dir="${EMBER_MODEL_DIR:-/models}"
model="${EMBER_MODEL:-}"
draft="${EMBER_DRAFT_MODEL:-}"
server_bin="${EMBER_SERVER_BIN:-/usr/local/bin/ember-dflash}"
revision="${EMBER_MODEL_REVISION:-f6e507774f7133568f6fec0635057cb20134f307}"
expected_sha256="${EMBER_MODEL_SHA256-18aec8c0be4087007e557aa6020b28f12cd4c5d1f9c67b2a815c152aea97b3ed}"
expected_size="${EMBER_MODEL_SIZE_BYTES:-91547243104}"
model_verified=0

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
  [[ -z "$expected_sha256" ]] && return 0
  [[ "$expected_sha256" =~ ^[0-9a-fA-F]{64}$ ]] ||
    die "EMBER_MODEL_SHA256 must be empty or a 64-character hexadecimal digest"
  echo "ember: verifying SHA-256 for $(basename "$path")"
  printf '%s  %s\n' "$expected_sha256" "$path" | sha256sum --check --status ||
    die "model SHA-256 mismatch: $path (remove the file and download it again)"
}

check_download_space() {
  local partial="$1"
  [[ "$expected_size" =~ ^[0-9]+$ ]] ||
    die "EMBER_MODEL_SIZE_BYTES must contain only decimal digits"
  local have=0
  [[ -f "$partial" ]] && have="$(stat -c %s "$partial")"
  local remaining=$((expected_size > have ? expected_size - have : 0))
  local reserve=$((2 * 1024 * 1024 * 1024))
  local available
  available="$(df -PB1 "$model_dir" | awk 'NR == 2 {print $4}')"
  if [[ "$available" =~ ^[0-9]+$ ]] && (( available < remaining + reserve )); then
    die "insufficient free space in $model_dir: need $((remaining + reserve)) bytes, have $available"
  fi
}

if [[ -z "$model" ]]; then
  mapfile -t candidates < <(find "$model_dir" -maxdepth 1 -type f \
    -iname '*.gguf' ! -iname '*draft*' -print | sort)
  if [[ ${#candidates[@]} -eq 0 && "${EMBER_AUTO_DOWNLOAD:-1}" == 1 ]]; then
    repo="${EMBER_MODEL_REPO:-otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF}"
    file="${EMBER_MODEL_FILE:-DeepSeek-V4-Flash-0731-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf}"
    destination="$model_dir/$file"
    partial="$destination.part"
    mkdir -p "$model_dir"
    check_download_space "$partial"
    echo "ember: downloading $repo@$revision/$file (resume is enabled; this is about 85 GiB)"
    partial_size=0
    [[ -f "$partial" ]] && partial_size="$(stat -c %s "$partial")"
    if [[ "$partial_size" != "$expected_size" ]]; then
      curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
        --output "$partial" \
        "https://huggingface.co/$repo/resolve/$revision/$file"
    fi
    verify_sha256 "$partial"
    model_verified=1
    mv "$partial" "$destination"
    candidates=("$destination")
  fi
  if [[ ${#candidates[@]} -ne 1 ]]; then
    echo "ember: expected exactly one non-draft GGUF in $model_dir; found ${#candidates[@]}" >&2
    echo "ember: set EMBER_MODEL=/models/your-model.gguf when the directory contains multiple models" >&2
    exit 64
  fi
  model="${candidates[0]}"
fi

if [[ -z "$draft" ]]; then
  mapfile -t drafts < <(find "$model_dir" -maxdepth 1 -type f \
    -iname '*draft*.gguf' -print | sort)
  if [[ ${#drafts[@]} -eq 1 ]]; then
    draft="${drafts[0]}"
  fi
fi

if [[ ! -r "$model" ]]; then
  echo "ember: model is not readable: $model" >&2
  exit 66
fi

if [[ "$model_verified" != 1 ]]; then
  verify_sha256 "$model"
fi

if [[ -n "$draft" ]]; then
  if [[ ! -r "$draft" ]]; then
    echo "ember: draft model is not readable: $draft" >&2
    exit 66
  fi
  export DFLASH_DS4_SPEC=1
  export DFLASH_DS4_DRAFT="$draft"
else
  export DFLASH_DS4_SPEC=0
fi

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
