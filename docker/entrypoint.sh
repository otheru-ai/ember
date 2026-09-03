#!/usr/bin/env bash
set -euo pipefail

model_dir="${EMBER_MODEL_DIR:-/models}"
server_bin="${EMBER_SERVER_BIN:-/usr/local/bin/ember-dflash}"
deployment_mode="${EMBER_DEPLOYMENT_MODE:-deepseek-v4-flash}"
# The default DeepSeek deployment is the VISION model as of this release. The
# previous text-only artifact is not deleted and stays servable by pointing
# EMBER_MODEL_DIR at a copy, or by setting the three EMBER_MODEL_* overrides
# below; it is simply no longer what a fresh container downloads.
#
# The drafter is the 0731 one on purpose. A drafter built from the Vision
# checkpoint's own mtp.* tensors exists and accepts 0.868 on structured
# workloads against 0.975 for this one, measured on the same target in the same
# window -- and swapping the target reproduces the deficit, so it is the head,
# not the pairing. See releases/vision-exp/PROVENANCE.md in the pipeline repo.
repo="${EMBER_MODEL_REPO:-otheru/DeepSeek-V4-Flash-Vision-Strix-Halo-GGUF}"
revision="${EMBER_MODEL_REVISION:-ba9824c9bc69ce3cdeea91d6529ff764720ec737}"
file="DeepSeek-V4-Flash-Vision-Exp-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf"
expected_sha256="2ff6ff0c4bd20d8438113404d9c7c3d4495bbc4b43b5622f37a0f68aebfebbc2"
expected_size="91547293152"
draft_file="DeepSeek-V4-Flash-0731-Abliterated-DSpark-draft-4.25bpw.gguf"
draft_expected_sha256="1a01c80eceae302bcc1d70836759ee97974d7983c5084ef43f6ef772a8970ae6"
draft_expected_size="10897111840"
# The vision tower. Required for image input and downloaded with the model; the
# server refuses image requests without it rather than answering from text.
mmproj_file="mmproj-DeepSeek-V4-Flash-Vision-Exp-F16.gguf"
mmproj_expected_sha256="9225c5562c05bd910245ab24c9274ca777eba2a801990f47ebe0c6344f144002"
mmproj_expected_size="933251200"
model="$model_dir/$file"
draft="$model_dir/$draft_file"
mmproj="$model_dir/$mmproj_file"
# An unpinned revision would fetch whatever the branch tip happens to be, which
# is how a container silently changes model between restarts. Refuse, unless
# the operator has supplied their own artifacts and turned auto-download off.
if [[ "$revision" == UNPINNED && "${EMBER_AUTO_DOWNLOAD:-1}" == 1 ]]; then
  echo "ember: model revision is not pinned; set EMBER_MODEL_REVISION to an" \
       "immutable HuggingFace commit, or set EMBER_AUTO_DOWNLOAD=0 and supply" \
       "the artifacts in EMBER_MODEL_DIR" >&2
  exit 78
fi
model_verified=0
draft_verified=0
mmproj_verified=0
verify_existing_sha256="${EMBER_VERIFY_EXISTING_SHA256:-1}"
integrity_cache_dir="${EMBER_INTEGRITY_CACHE_DIR:-\
${EMBER_KV_CACHE_DIR:-/cache}/artifact-integrity-v1}"
integrity_cache_state=0

die() {
  echo "ember: $*" >&2
  exit 78
}

case "$verify_existing_sha256" in
  0|1) ;;
  *) die "EMBER_VERIFY_EXISTING_SHA256 must be 0 or 1" ;;
esac

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

prepare_integrity_cache() {
  if [[ "$integrity_cache_state" == 1 ]]; then
    return 0
  fi
  if [[ "$integrity_cache_state" == -1 ]]; then
    return 1
  fi
  if [[ "$integrity_cache_dir" != /* || -L "$integrity_cache_dir" ]] ||
      ! mkdir -p -- "$integrity_cache_dir" ||
      [[ ! -d "$integrity_cache_dir" || ! -w "$integrity_cache_dir" ]]; then
    echo "ember: WARNING: integrity cache is unavailable; artifacts will be hashed" >&2
    integrity_cache_state=-1
    return 1
  fi
  integrity_cache_state=1
}

artifact_identity() {
  local path="$1"
  # GNU stat's %y/%z include nanoseconds.  ctime makes an in-place rewrite
  # invalidate the record even if an operator restores the size and mtime.
  stat -L --printf '%d\n%i\n%s\n%y\n%z' -- "$path"
}

persist_integrity_record() {
  local path="$1"
  local digest="$2"
  local identity="$3"
  local record record_path temporary

  if ! prepare_integrity_cache; then
    return 0
  fi
  record=$'ember.artifact-integrity.v1\n'"$digest"$'\n'"$identity"
  record_path="$integrity_cache_dir/$digest.identity"
  temporary="$(mktemp "$integrity_cache_dir/.artifact-identity.XXXXXX")" || {
    echo "ember: WARNING: cannot create integrity cache record for $path" >&2
    return 0
  }
  if printf '%s\n' "$record" >"$temporary" && mv -f -- "$temporary" "$record_path"; then
    :
  else
    rm -f -- "$temporary"
    echo "ember: WARNING: cannot persist integrity cache record for $path" >&2
  fi
}

verify_cached_sha256() {
  local path="$1"
  local digest="$2"
  local label="$3"
  local before after record record_path base

  before="$(artifact_identity "$path")" || die "cannot stat $label: $path"
  record=$'ember.artifact-integrity.v1\n'"$digest"$'\n'"$before"
  base="$(basename "$path")"
  if prepare_integrity_cache; then
    record_path="$integrity_cache_dir/$digest.identity"
    if [[ -f "$record_path" && ! -L "$record_path" ]] &&
        [[ "$(<"$record_path")" == "$record" ]]; then
      echo "ember: $base: OK (integrity cache hit)"
      return 0
    fi
  else
    record_path=
  fi

  verify_sha256 "$path" "$digest" "$label"
  after="$(artifact_identity "$path")" || die "cannot re-stat $label: $path"
  [[ "$after" == "$before" ]] || die "$label changed while it was hashed: $path"

  persist_integrity_record "$path" "$digest" "$after"
}

verify_existing_artifact() {
  local path="$1"
  local digest="$2"
  local label="$3"
  if [[ "$verify_existing_sha256" == 1 ]]; then
    verify_cached_sha256 "$path" "$digest" "$label"
  else
    echo "ember: WARNING: skipping SHA-256 verification for pre-existing $label: $path" >&2
  fi
}

require_direct_model_artifact() {
  local path="$1"
  local label="$2"
  [[ "$path" = /* ]] || die "$label path must be absolute: $path"
  [[ "$(dirname "$path")" == "$model_dir" ]] ||
    die "$label must be a direct child of EMBER_MODEL_DIR: $path"
  [[ -f "$path" && -r "$path" && ! -L "$path" ]] ||
    die "$label is not a readable regular file: $path"
}

prepare_qwen() {
  local checksum_path="${EMBER_QWEN_SHA256SUMS:-}"
  local checksum_sha256="${EMBER_QWEN_SHA256SUMS_SHA256:-}"
  local mtp="${DFLASH_QWEN_MTP:-}"
  local mtp_depth="${DFLASH_QWEN_MTP_DEPTH:-}"
  local mmproj="${DFLASH_QWEN_VISION_MMPROJ:-}"
  local vision_text_model="${DFLASH_QWEN_VISION_TEXT_MODEL:-}"
  local provider="${DFLASH_QWEN_VISION_PROVIDER:-}"

  model="${EMBER_QWEN_MODEL:-}"
  [[ -n "$model" ]] || die "EMBER_QWEN_MODEL is required in qwen3.8-flash-next mode"
  [[ -n "$mtp" ]] || die "DFLASH_QWEN_MTP is required in qwen3.8-flash-next mode"
  [[ "$mtp_depth" =~ ^[1-4]$ ]] ||
    die "DFLASH_QWEN_MTP_DEPTH must be an integer from 1 to 4 in qwen3.8-flash-next mode"
  [[ -n "$mmproj" ]] ||
    die "DFLASH_QWEN_VISION_MMPROJ is required in qwen3.8-flash-next mode"
  [[ -n "$vision_text_model" ]] ||
    die "DFLASH_QWEN_VISION_TEXT_MODEL is required in qwen3.8-flash-next mode"
  [[ -n "$provider" ]] ||
    die "DFLASH_QWEN_VISION_PROVIDER is required in qwen3.8-flash-next mode"
  [[ -f "$provider" && -r "$provider" && ! -L "$provider" ]] ||
    die "Qwen vision provider is not a readable regular file: $provider"
  [[ -n "$checksum_path" ]] ||
    die "EMBER_QWEN_SHA256SUMS is required in qwen3.8-flash-next mode"
  [[ "$checksum_sha256" =~ ^[0-9a-f]{64}$ ]] ||
    die "EMBER_QWEN_SHA256SUMS_SHA256 must be a lowercase SHA-256 in qwen3.8-flash-next mode"

  require_direct_model_artifact "$model" "Qwen main shard 1"
  require_direct_model_artifact "$mtp" "Qwen MTP companion"
  require_direct_model_artifact "$mmproj" "Qwen BF16 mmproj"
  require_direct_model_artifact "$vision_text_model" "Qwen vision vocab companion"
  require_direct_model_artifact "$checksum_path" "Qwen SHA256SUMS"

  local model_name mtp_name mmproj_name vision_text_model_name checksum_name shard_prefix shard_count_text
  model_name="$(basename "$model")"
  mtp_name="$(basename "$mtp")"
  mmproj_name="$(basename "$mmproj")"
  vision_text_model_name="$(basename "$vision_text_model")"
  checksum_name="$(basename "$checksum_path")"
  if [[ "$model_name" =~ ^(Qwen3\.8-Flash-Next-.+)-00001-of-([0-9]{5})\.gguf$ ]]; then
    shard_prefix="${BASH_REMATCH[1]}"
    shard_count_text="${BASH_REMATCH[2]}"
  else
    die "EMBER_QWEN_MODEL must name shard 00001 of an ordered Qwen3.8-Flash-Next GGUF set"
  fi
  [[ "$mtp_name" == Qwen3.8-Flash-Next-MTP-*.gguf ]] ||
    die "DFLASH_QWEN_MTP must use the Qwen3.8-Flash-Next MTP artifact naming contract"
  [[ "$mmproj_name" == "Qwen3.8-Flash-Next-BF16-mmproj.gguf" ]] ||
    die "DFLASH_QWEN_VISION_MMPROJ must be Qwen3.8-Flash-Next-BF16-mmproj.gguf"
  [[ "$vision_text_model_name" == "Qwen3.8-Flash-Next-vocab-only.gguf" ]] ||
    die "DFLASH_QWEN_VISION_TEXT_MODEL must be Qwen3.8-Flash-Next-vocab-only.gguf"

  verify_sha256 "$checksum_path" "$checksum_sha256" "Qwen checksum manifest"

  # Release packaging writes a strict, basename-only GNU SHA256SUMS file. Keep
  # the entrypoint parser deliberately narrower than sha256sum itself: a list
  # containing traversal, absolute paths, duplicate names, or binary markers
  # must not expand the set of files startup is allowed to inspect.
  local line digest artifact_name
  declare -A sealed_names=()
  declare -a sealed_order=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^([0-9a-f]{64})\ \ ([A-Za-z0-9][A-Za-z0-9._+-]*)$ ]]; then
      digest="${BASH_REMATCH[1]}"
      artifact_name="${BASH_REMATCH[2]}"
    else
      die "Qwen SHA256SUMS has a malformed or unsafe entry"
    fi
    [[ -z "${sealed_names[$artifact_name]+x}" ]] ||
      die "Qwen SHA256SUMS repeats $artifact_name"
    sealed_names["$artifact_name"]="$digest"
    sealed_order+=("$artifact_name")
    require_direct_model_artifact "$model_dir/$artifact_name" \
      "Qwen checksummed artifact"
  done < "$checksum_path"

  local shard_count shard_number shard_name
  shard_count=$((10#$shard_count_text))
  (( shard_count >= 1 )) || die "Qwen main shard count must be positive"
  for ((shard_number = 1; shard_number <= shard_count; shard_number++)); do
    printf -v shard_name '%s-%05d-of-%05d.gguf' \
      "$shard_prefix" "$shard_number" "$shard_count"
    [[ -n "${sealed_names[$shard_name]+x}" ]] ||
      die "Qwen SHA256SUMS omits required main shard $shard_name"
  done
  [[ -n "${sealed_names[$mtp_name]+x}" ]] ||
    die "Qwen SHA256SUMS omits the selected MTP companion $mtp_name"
  [[ -n "${sealed_names[$mmproj_name]+x}" ]] ||
    die "Qwen SHA256SUMS omits the selected BF16 mmproj $mmproj_name"
  [[ -n "${sealed_names[$vision_text_model_name]+x}" ]] ||
    die "Qwen SHA256SUMS omits the selected vision vocab companion $vision_text_model_name"

  for artifact_name in "${sealed_order[@]}"; do
    verify_cached_sha256 "$model_dir/$artifact_name" \
      "${sealed_names[$artifact_name]}" "Qwen checksummed artifact"
  done

  export DFLASH_QWEN_MTP="$mtp"
  export DFLASH_QWEN_MTP_DEPTH="$mtp_depth"
  export DFLASH_QWEN_VISION_MMPROJ="$mmproj"
  export DFLASH_QWEN_VISION_TEXT_MODEL="$vision_text_model"
  export DFLASH_QWEN_VISION_PROVIDER="$provider"
  echo "ember: sealed Qwen3.8-Flash-Next artifact set verified ($shard_count main shards, MTP depth $mtp_depth)"
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
  persist_integrity_record "$destination" "$artifact_sha256" \
    "$(artifact_identity "$destination")"
  downloaded_artifact="$destination"
}

case "$deployment_mode" in
  deepseek-v4-flash)
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
      verify_existing_artifact "$model" "$expected_sha256" model
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
      verify_existing_artifact "$draft" "$draft_expected_sha256" "draft model"
    fi
    # Compose is authoritative for tuning knobs. DFLASH_DS4_DRAFT resolves to
    # the pinned, verified artifact unless an operator explicitly overrides it.
    export DFLASH_DS4_SPEC="${DFLASH_DS4_SPEC:-1}"
    export DFLASH_DS4_DRAFT="${DFLASH_DS4_DRAFT:-$draft}"

    if [[ ! -r "$mmproj" && "$mmproj" == "$model_dir/$mmproj_file" && \
          "${EMBER_AUTO_DOWNLOAD:-1}" == 1 ]]; then
      download_artifact "$mmproj_file" "$mmproj_expected_size" \
        "$mmproj_expected_sha256" "vision tower"
      mmproj_verified=1
      mmproj="$downloaded_artifact"
    fi

    # A missing tower is a hard error, not a silent downgrade to text. An image
    # request against a server without one must not look like a model that
    # answered without looking.
    if [[ ! -r "$mmproj" ]]; then
      echo "ember: vision tower is not readable: $mmproj" >&2
      exit 66
    fi

    if [[ "$mmproj_verified" != 1 ]]; then
      verify_existing_artifact "$mmproj" "$mmproj_expected_sha256" "vision tower"
    fi

    deepseek_vision_args=(--vision-mmproj "$mmproj")
    ;;
  qwen3.8-flash-next)
    prepare_qwen
    ;;
  *)
    die "EMBER_DEPLOYMENT_MODE must be deepseek-v4-flash or qwen3.8-flash-next"
    ;;
esac

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
  ${deepseek_vision_args[@]+"${deepseek_vision_args[@]}"} \
  --kv-cache-dir "${EMBER_KV_CACHE_DIR:-/cache}" \
  --host "${EMBER_HOST:-127.0.0.1}" \
  --port "${EMBER_PORT:-8080}" \
  "${server_args[@]}" \
  "$@"
