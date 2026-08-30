#!/usr/bin/env bash
# Production-safe, provenance-bound Qwen3.8 real-weight release gate.
#
# This driver owns the gfx1151 for its complete run. It first proves q=1 vs
# native-batch MTP token identity (fresh and snapshot-restored), then runs the
# unprofiled 2026.8.24 hard-gate protocol, and only afterward launches separate
# trace/PMC passes. Counter collection never contributes timing evidence.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_LOCK=/usr/local/sbin/ember-gpu-lock
PRODUCTION=/usr/local/sbin/ember-cert-production
PRODUCTION_HEALTH=http://127.0.0.1:8000/health
PROFILE_SCRIPT="$REPO/scripts/profile_gpu.sh"
PROFILE_REPORT="$REPO/scripts/profile_report.py"
COUNTER_CALIBRATION="$REPO/share/benchmark/gfx1151-rocm10-counter-calibration.json"
BENCHMARK="$REPO/scripts/bench/benchmark.py"
DISPATCH_EVIDENCE="$REPO/scripts/qwen_w4a8_dispatch_evidence.py"

IMAGE=""; IMAGE_DIGEST=""; PROFILE_IMAGE=""; PROFILE_IMAGE_DIGEST=""
MODEL=""; MODEL_SHA256=""; MODEL_BUILD_RECORD=""; MODEL_BUILD_RECORD_SHA256=""
MTP=""; MTP_SHA256=""; OUT_DIR=""
INTEGRITY_CACHE=/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/.artifact-integrity-v1.json
BINARY=/usr/local/bin/ember-dflash
PORT=18086; MTP_DEPTH=4; DRY_RUN=0
MEASUREMENT_ONLY=0
CONTAINER=""; LOCK_HELD=0; MASKED=0; RESTORE_SERVICE=0
PRODUCTION_STATE_CAPTURED=0; PRODUCTION_WAS_ACTIVE=0

die() { printf 'qwen-real-weight-gate: %s\n' "$*" >&2; exit 1; }
log() { printf 'qwen-real-weight-gate: %s\n' "$*"; }

usage() {
  cat <<'EOF'
usage: scripts/qwen_real_weight_gate.sh [options]

required:
  --image REF                 exact local candidate image reference
  --image-digest sha256:HEX   exact image ID or RepoDigest suffix
  --profile-image REF         exact matching-revision ROCm dev image
  --profile-image-digest      exact profiler image ID or RepoDigest suffix
  --model ABS_PATH            exact Qwen target GGUF
  --model-sha256 HEX          expected first target-shard SHA-256
  --model-build-record PATH   completed quant build record for all target shards
  --model-build-record-sha256 expected quant build-record SHA-256
  --mtp ABS_PATH              exact matching-quant MTP GGUF
  --mtp-sha256 HEX            expected MTP SHA-256
  --out-dir ABS_PATH          new evidence directory (must not exist)

options:
  --binary PATH               ember-dflash path in image
  --port N                    temporary loopback port (default 18086)
  --mtp-depth N               proposal depth 1..4 (default 4)
  --integrity-cache PATH      persistent identity-bound verification cache
  --measurement-only          retain complete measurements below hard thresholds
  --dry-run                   validate syntax and print plan; touch nothing
EOF
}

while (( $# )); do
  case "$1" in
    --image) IMAGE="${2:?--image needs a value}"; shift 2 ;;
    --image-digest) IMAGE_DIGEST="${2:?--image-digest needs a value}"; shift 2 ;;
    --profile-image) PROFILE_IMAGE="${2:?--profile-image needs a value}"; shift 2 ;;
    --profile-image-digest) PROFILE_IMAGE_DIGEST="${2:?--profile-image-digest needs a value}"; shift 2 ;;
    --model) MODEL="${2:?--model needs a path}"; shift 2 ;;
    --model-sha256) MODEL_SHA256="${2:?--model-sha256 needs a value}"; shift 2 ;;
    --model-build-record) MODEL_BUILD_RECORD="${2:?--model-build-record needs a path}"; shift 2 ;;
    --model-build-record-sha256) MODEL_BUILD_RECORD_SHA256="${2:?--model-build-record-sha256 needs a value}"; shift 2 ;;
    --mtp) MTP="${2:?--mtp needs a path}"; shift 2 ;;
    --mtp-sha256) MTP_SHA256="${2:?--mtp-sha256 needs a value}"; shift 2 ;;
    --integrity-cache) INTEGRITY_CACHE="${2:?--integrity-cache needs a path}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?--out-dir needs a path}"; shift 2 ;;
    --binary) BINARY="${2:?--binary needs a path}"; shift 2 ;;
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --mtp-depth) MTP_DEPTH="${2:?--mtp-depth needs a value}"; shift 2 ;;
    --measurement-only) MEASUREMENT_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ -n "$IMAGE" && -n "$IMAGE_DIGEST" && -n "$PROFILE_IMAGE" &&
   -n "$PROFILE_IMAGE_DIGEST" && -n "$MODEL" &&
   -n "$MODEL_SHA256" && -n "$MODEL_BUILD_RECORD" &&
   -n "$MODEL_BUILD_RECORD_SHA256" && -n "$MTP" && -n "$MTP_SHA256" &&
   -n "$OUT_DIR" ]] || die "all image/model/MTP paths and digests plus --out-dir are required"
[[ "$IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] ||
  die "--image-digest must be sha256: plus 64 lowercase hex characters"
[[ "$PROFILE_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] ||
  die "--profile-image-digest must be sha256: plus 64 lowercase hex characters"
[[ "$MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  die "--model-sha256 must be 64 lowercase hex characters"
[[ "$MODEL_BUILD_RECORD_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  die "--model-build-record-sha256 must be 64 lowercase hex characters"
[[ "$MTP_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  die "--mtp-sha256 must be 64 lowercase hex characters"
[[ "$MODEL" = /* && "$MODEL_BUILD_RECORD" = /* && "$MTP" = /* &&
   "$OUT_DIR" = /* && "$INTEGRITY_CACHE" = /* ]] ||
  die "--model, --model-build-record, --mtp, --out-dir and --integrity-cache must be absolute paths"
[[ "$BINARY" = /* ]] || die "--binary must be an absolute in-image path"
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1024 && PORT <= 65535 )) ||
  die "--port must be 1024..65535"
[[ "$MTP_DEPTH" =~ ^[0-9]+$ ]] && (( MTP_DEPTH >= 1 && MTP_DEPTH <= 4 )) ||
  die "--mtp-depth must be 1..4"

print_plan() {
  cat <<EOF
plan:
  candidate image     $IMAGE
  image digest        $IMAGE_DIGEST
  profiler image      $PROFILE_IMAGE
  profiler digest     $PROFILE_IMAGE_DIGEST
  target model        $MODEL
  target shard-1 sha  $MODEL_SHA256
  quant build record  $MODEL_BUILD_RECORD
  build record sha    $MODEL_BUILD_RECORD_SHA256
  MTP companion       $MTP
  MTP sha256          $MTP_SHA256
  integrity cache     $INTEGRITY_CACHE
  MTP depth           $MTP_DEPTH
  evidence            $OUT_DIR
  ownership           $GPU_LOCK acquire/release
  production          $PRODUCTION stop + mask; unconditional unmask/restore
  correctness         q=1 vs native MTP, fresh + snapshot restore + rejection evidence
  timing              unprofiled 3x exact-2074 prefill and 3x decode-256
  hard gates          prefill peak >=412.0 tok/s; decode median >=39.49 tok/s
  profiling           later, separate trace and one-counter-per-PMC passes
  publication         forbidden; text+MTP hardware marker is not release approval
  result mode         $([[ "$MEASUREMENT_ONLY" = 1 ]] && printf measurement || printf certification)
EOF
}

print_plan
if (( DRY_RUN )); then
  log "dry run: no files, GPU, docker, sudo, lock, or production state touched"
  exit 0
fi

command -v docker >/dev/null || die "docker is required"
command -v curl >/dev/null || die "curl is required"
command -v python3 >/dev/null || die "python3 is required"
command -v dd >/dev/null || die "dd is required for O_DIRECT integrity reads"
command -v stat >/dev/null || die "stat is required for numeric GPU device groups"
[[ -x "$GPU_LOCK" ]] || die "missing fixed-purpose GPU lock wrapper: $GPU_LOCK"
[[ -x "$PRODUCTION" ]] || die "missing production wrapper: $PRODUCTION"
[[ -x "$PROFILE_SCRIPT" && -f "$PROFILE_REPORT" &&
   -f "$COUNTER_CALIBRATION" && -f "$BENCHMARK" &&
   -f "$DISPATCH_EVIDENCE" ]] ||
  die "gate dependencies are missing"
[[ -f "$MODEL" && -f "$MODEL_BUILD_RECORD" && -f "$MTP" ]] ||
  die "model, quant build record, or MTP companion does not exist"
[[ -r /dev/kfd && -d /dev/dri ]] || die "this gate must run on the gfx1151 host"
[[ ! -e "$OUT_DIR" ]] || die "--out-dir must not already exist (prevents stale approval reuse)"
if grep -Eq '(^| )(iommu|amd_iommu)=off( |$)' /proc/cmdline; then
  die "IOMMU is disabled"
fi
if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  die "port $PORT is already serving"
fi

mkdir -p "$OUT_DIR"
python3 - "$MODEL_BUILD_RECORD" "$MODEL_BUILD_RECORD_SHA256" \
  "$MODEL" "$MODEL_SHA256" "$OUT_DIR/model-inventory.json" \
  "$OUT_DIR/qwen-quant-build-record.json" "$BENCHMARK" <<'PY'
import importlib.util, json, sys
from pathlib import Path
record_path, record_sha, model, model_sha, inventory_out, record_out, module_path = sys.argv[1:]
spec = importlib.util.spec_from_file_location("ember_benchmark_gate", module_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load benchmark inventory verifier")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
try:
    inventory, raw = module.model_inventory_from_build_record(
        Path(record_path), record_sha, Path(model), model_sha)
except (OSError, ValueError) as exc:
    raise SystemExit(str(exc)) from exc
with open(inventory_out, "x", encoding="utf-8") as stream:
    json.dump(inventory, stream, indent=2, sort_keys=True)
    stream.write("\n")
with open(record_out, "xb") as stream:
    stream.write(raw)
PY
docker image inspect "$IMAGE" >"$OUT_DIR/image-inspect.json"
docker image inspect "$PROFILE_IMAGE" >"$OUT_DIR/profile-image-inspect.json"
python3 - "$OUT_DIR/image-inspect.json" "$IMAGE_DIGEST" \
  "$OUT_DIR/profile-image-inspect.json" "$PROFILE_IMAGE_DIGEST" <<'PY'
import json, sys
for label, path, expected in (
        ("candidate", sys.argv[1], sys.argv[2]),
        ("profiler", sys.argv[3], sys.argv[4])):
    records = json.load(open(path, encoding="utf-8"))
    if len(records) != 1:
        raise SystemExit(f"{label} image reference did not resolve exactly once")
    record = records[0]
    observed = {record.get("Id")}
    observed.update(item.rsplit("@", 1)[-1]
                    for item in record.get("RepoDigests") or [])
    if expected not in observed:
        raise SystemExit(
            f"{label} image digest mismatch: expected {expected}, "
            f"observed {sorted(observed)}")
PY

# Fail before taking the machine lock if either digest-pinned image is
# incomplete.  The release image owns correctness/timing; the matching dev
# image owns only profiler collection.
docker run --rm --entrypoint /bin/sh "$IMAGE" -c 'test -x "$1"' \
  qwen-real-weight-gate-candidate-preflight "$BINARY" ||
  die "candidate image lacks the requested binary"
docker run --rm --entrypoint /bin/sh "$PROFILE_IMAGE" -c '
  test -x "$1" &&
  command -v rocprofv3 >/dev/null &&
  command -v rocprof-compute >/dev/null
' qwen-real-weight-gate-preflight "$BINARY" ||
  die "profiler image lacks the matching binary or ROCm profiler tools"

# The dev stage deliberately has no OCI labels, so prove the release/dev
# relationship using the release label plus the revision embedded in the dev
# CMake cache.  Also require byte-identical executable payloads: profiling a
# merely source-similar binary would not be evidence for the candidate.
candidate_revision="$(python3 - "$OUT_DIR/image-inspect.json" <<'PY'
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))[0]
print(((record.get("Config") or {}).get("Labels") or {}).get(
    "org.opencontainers.image.revision", ""))
PY
)"
[[ "$candidate_revision" =~ ^[0-9a-f]{40}$ ]] ||
  die "candidate image has no pinned 40-hex OCI source revision"
profile_revision="$(docker run --rm --entrypoint /bin/sh "$PROFILE_IMAGE" -c '
  awk -F= '\''$1 == "EMBER_CONFIGURED_GIT_HEAD:STRING" { print $2 }'\'' \
    /ember/build-rocm/CMakeCache.txt
')"
[[ "$profile_revision" == "$candidate_revision" ]] ||
  die "candidate/profiler images were not built from the same Ember revision"
candidate_binary_sha="$(docker run --rm --entrypoint sha256sum "$IMAGE" "$BINARY" | awk '{print $1}')"
profile_binary_sha="$(docker run --rm --entrypoint sha256sum "$PROFILE_IMAGE" "$BINARY" | awk '{print $1}')"
[[ "$candidate_binary_sha" =~ ^[0-9a-f]{64}$ &&
   "$candidate_binary_sha" == "$profile_binary_sha" ]] ||
  die "candidate/profiler ember-dflash binaries are not byte-identical"
python3 - "$OUT_DIR/binary-identity.json" "$candidate_revision" \
  "$candidate_binary_sha" <<'PY'
import json, sys
with open(sys.argv[1], "x", encoding="utf-8") as stream:
    json.dump({"ember_revision": sys.argv[2],
               "candidate_and_profiler_binary_sha256": sys.argv[3],
               "byte_identical": True}, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

# Bind any enabled W4A8 runtime to the saved production assembly and encoded
# object from the byte-identical dev image. A startup marker cannot substitute
# for this ISA gate, and compile-resource output is never timing evidence.
mapfile -t w4a8_build < <(docker run --rm --entrypoint /bin/sh "$PROFILE_IMAGE" -c '
  for key in GGML_HIP_ROCMI4_W4A4 GGML_HIP_ROCMI4_W4A8_IU4 GGML_HIP_ROCMI4_W4A8_IU4_PREPACK GGML_HIP_EXPORT_METRICS; do
    awk -F= -v key="$key:BOOL" '\''$1 == key { print $2 }'\'' /ember/build-rocm/CMakeCache.txt
  done
')
((${#w4a8_build[@]} == 4)) || die "profiler image lacks ROCMI4 CMake identity"
case "${w4a8_build[0]}:${w4a8_build[1]}:${w4a8_build[2]}:${w4a8_build[3]}" in
  OFF:OFF:OFF:*) w4a8_build_mode=exact_int8_mmq_control ;;
  ON:OFF:OFF:*) w4a8_build_mode=lossy_w4a4_mmq ;;
  OFF:ON:OFF:ON) w4a8_build_mode=w4a8_iu4_register_pack ;;
  OFF:ON:ON:ON) w4a8_build_mode=w4a8_iu4_prepack ;;
  *) die "profiler image has an ineligible W4A8/export-metrics build combination" ;;
esac
if [[ "$w4a8_build_mode" == w4a8_* ]]; then
  case "$w4a8_build_mode" in
    w4a8_iu4_register_pack) w4a8_variant=register ;;
    w4a8_iu4_prepack) w4a8_variant=prepack ;;
  esac
  docker run --rm --entrypoint /bin/bash "$PROFILE_IMAGE" -c '
    set -euo pipefail
    base=/ember/build-rocm/engine/ggml/src/ggml-hip/mmq-instance-q4_0_rocmi4-hip-amdgcn-amd-amdhsa-gfx1151
    /opt/rocm/llvm/bin/llvm-objdump --disassemble --mcpu=gfx1151 "$base.o" >/tmp/rocmi4-w4a8.disasm
    python3 /ember/scripts/check_rocmi4_w4a8_isa.py "$base.s" \
      --disassembly /tmp/rocmi4-w4a8.disasm \
      --cmake-cache /ember/build-rocm/CMakeCache.txt --variant "$1"
  ' qwen-w4a8-isa-gate "$w4a8_variant" >"$OUT_DIR/w4a8-isa-gate.txt" ||
    die "byte-identical profiler image failed the exact gfx1151 W4A8 ISA gate"
else
  printf '%s\n' 'not applicable: exact int8 MMQ control build' >"$OUT_DIR/w4a8-isa-gate.txt"
fi
python3 - "$OUT_DIR/w4a8-build-evidence.json" "$w4a8_build_mode" \
  "$OUT_DIR/w4a8-isa-gate.txt" "$candidate_binary_sha" <<'PY'
import hashlib, json, sys
output, mode, report, binary_sha = sys.argv[1:]
raw = open(report, "rb").read()
with open(output, "x", encoding="utf-8") as stream:
    json.dump({
        "schema": "ember.qwen3.8.w4a8-build-evidence.v1",
        "build_mode": mode,
        "candidate_and_profiler_binary_sha256": binary_sha,
        "saved_isa_gate": {
            "path": "w4a8-isa-gate.txt",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "passed": mode.startswith("w4a8_iu4_"),
        },
    }, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
W4A8_RUNTIME_ARGS=()
W4A8_DISPATCH_ARGS=()
W4A8_PROFILE_ARGS=()
candidate_kernel_capability="$(python3 - "$OUT_DIR/qwen-quant-build-record.json" <<'PY'
import json,sys
arm=(json.load(open(sys.argv[1],encoding="utf-8")).get("quantization_recipe") or {}).get("id")
mapping={
 "profile-default-rocmi4":"rocmi4_dense_and_routed",
 "rocmi4-control":"rocmi4_dense_and_routed",
 "rocmi4-q6k-embedding-head":"rocmi4_dense_and_routed",
 "rocmfp4-fast-routed-experts-q6k-embedding-head":"rocmi4_dense_only",
 "rocmfp4-fast-matrix-q6k-embedding-head":"no_eligible_rocmi4_mmq",
 "rocmfp4-fast-matrix-q3-ple-q6k-embedding-head":"no_eligible_rocmi4_mmq",
}
if arm not in mapping: raise SystemExit(f"unknown candidate kernel capability: {arm!r}")
print(mapping[arm])
PY
)"
if [[ "$w4a8_build_mode" == w4a8_* ]]; then
  W4A8_DISPATCH_ARGS=(-e DFLASH_ROCMI4_W4A8_DISPATCH_EVIDENCE=1)
  if [[ "$candidate_kernel_capability" != no_eligible_rocmi4_mmq ]]; then
    W4A8_RUNTIME_ARGS=(-e DFLASH_ROCMI4_W4A8_IU4=1)
    W4A8_PROFILE_ARGS=(--rocmi4-w4a8-iu4)
  fi
fi

GPU_ARGS=(
  --device /dev/kfd --device /dev/dri
  --ipc host --security-opt seccomp=unconfined
  --ulimit memlock=-1:-1
)
declare -A GPU_GIDS=()
for node in /dev/kfd /dev/dri/*; do
  [[ -c "$node" ]] || continue
  gid="$(stat -c %g -- "$node")"
  [[ "$gid" =~ ^[0-9]+$ ]] || die "GPU device has a nonnumeric group: $node"
  GPU_GIDS["$gid"]=1
done
((${#GPU_GIDS[@]} > 0)) || die "no GPU character-device groups found"
for gid in "${!GPU_GIDS[@]}"; do GPU_ARGS+=(--group-add "$gid"); done

remove_container() {
  if [[ -n "$CONTAINER" ]]; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    CONTAINER=""
  fi
}

restore_exclusive() {
  local failed=0 attempt healthy=0
  remove_container
  if (( MASKED )); then
    for attempt in 1 2 3; do
      if sudo -n "$PRODUCTION" unmask >/dev/null 2>&1; then
        MASKED=0
        break
      fi
      sleep 1
    done
    (( MASKED == 0 )) || failed=1
  fi
  if (( RESTORE_SERVICE )); then
    # Starting while the mask is still present only obscures the real restore
    # failure.  Release the ownership lock regardless so another recovery
    # operator is never locked out by this process.
    if (( MASKED == 0 )); then
      for attempt in 1 2 3; do
        if sudo -n "$PRODUCTION" start >/dev/null 2>&1; then
          RESTORE_SERVICE=0
          break
        fi
        sleep 1
      done
    fi
    (( RESTORE_SERVICE == 0 )) || failed=1
  fi
  if (( PRODUCTION_STATE_CAPTURED && PRODUCTION_WAS_ACTIVE &&
        MASKED == 0 && RESTORE_SERVICE == 0 )); then
    for attempt in $(seq 1 300); do
      if sudo -n "$PRODUCTION" is-active >/dev/null 2>&1 &&
         curl --fail --silent --max-time 2 "$PRODUCTION_HEALTH" >/dev/null 2>&1; then
        healthy=1
        break
      fi
      sleep 2
    done
    (( healthy )) || failed=1
  fi
  if (( LOCK_HELD )); then
    for attempt in 1 2 3; do
      if sudo -n "$GPU_LOCK" release >/dev/null 2>&1; then
        LOCK_HELD=0
        break
      fi
      sleep 1
    done
    (( LOCK_HELD == 0 )) || failed=1
  fi
  return "$failed"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  restore_exclusive || rc=1
  exit "$rc"
}
trap cleanup EXIT INT TERM

sudo -n "$GPU_LOCK" acquire
LOCK_HELD=1
if sudo -n "$PRODUCTION" is-active >/dev/null 2>&1; then
  PRODUCTION_WAS_ACTIVE=1
  RESTORE_SERVICE=1
  sudo -n "$PRODUCTION" stop
fi
PRODUCTION_STATE_CAPTURED=1
sudo -n "$PRODUCTION" mask
MASKED=1

log "verifying every ordered target shard and MTP through the identity-bound cache"
PYTHONPATH="$REPO/scripts" python3 - "$OUT_DIR/model-inventory.json" \
    "$INTEGRITY_CACHE" "$MTP" "$MTP_SHA256" <<'PY'
import json, pathlib, sys
from qwen_integrity_cache import IntegrityCache
inventory = json.load(open(sys.argv[1], encoding="utf-8"))
with IntegrityCache(pathlib.Path(sys.argv[2])) as cache:
    for row in inventory["shards"]:
        cache.verify(pathlib.Path(row["path"]), row["sha256"])
    cache.verify(pathlib.Path(sys.argv[3]), sys.argv[4])
PY

for _ in $(seq 1 60); do
  available_kib="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
  (( available_kib >= 100 * 1024 * 1024 )) && break
  sleep 5
done
(( available_kib >= 100 * 1024 * 1024 )) ||
  die "less than 100 GiB available after integrity verification"

python3 - "$OUT_DIR/validation-prompt.txt" <<'PY'
from pathlib import Path
import sys
text = "\n".join(
    f"Constraint {i}: retain marker {i * 17 % 101} while reasoning deterministically."
    for i in range(180))
Path(sys.argv[1]).write_text(
    text + "\nExplain the invariant in numbered steps without skipping details.\n",
    encoding="utf-8")
PY
mkdir -m 700 "$OUT_DIR/validation-kv-cache"

log "running q=1/native-batch snapshot differential"
CONTAINER="qwen-real-gate-validate-$$"
docker run --name "$CONTAINER" "${GPU_ARGS[@]}" \
  "${W4A8_RUNTIME_ARGS[@]}" "${W4A8_DISPATCH_ARGS[@]}" \
  -v "$(dirname "$MODEL"):/gate/model:ro" \
  -v "$MTP:/gate/mtp.gguf:ro" \
  -v "$OUT_DIR/validation-prompt.txt:/gate/prompt.txt:ro" \
  -v "$OUT_DIR/validation-kv-cache:/gate/cache" \
  -e DFLASH_QWEN_MTP=/gate/mtp.gguf \
  -e "DFLASH_QWEN_MTP_DEPTH=$MTP_DEPTH" \
  --entrypoint "$BINARY" "$IMAGE" \
  -m "/gate/model/$(basename "$MODEL")" --max-ctx 8192 \
  --kv-cache-dir /gate/cache \
  --validate-prompt /gate/prompt.txt --validate-tokens 64 \
  >"$OUT_DIR/differential.json" \
  2>"$OUT_DIR/differential-dispatch-server.log"
docker rm "$CONTAINER" >/dev/null
CONTAINER=""
python3 - "$OUT_DIR/differential.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
spec = report.get("spec") or {}
if not (report.get("ok") and report.get("snapshot_ok") and
        spec.get("checked") and spec.get("exact")):
    raise SystemExit(f"q=1/native MTP differential failed: {report}")
rate = spec.get("accept_rate")
if not isinstance(rate, (int, float)) or not 0.0 <= rate < 1.0:
    raise SystemExit(
        "differential did not exercise a rejected candidate/strict replay; "
        f"accept_rate={rate!r}")
PY
python3 "$DISPATCH_EVIDENCE" \
  --log "$OUT_DIR/differential-dispatch-server.log" \
  --output "$OUT_DIR/kernel-runtime-evidence.json" \
  --expected-capability "$candidate_kernel_capability"
python3 - "$OUT_DIR/w4a8-build-evidence.json" \
  "$OUT_DIR/kernel-runtime-evidence.json" <<'PY'
import json, sys
build = json.load(open(sys.argv[1], encoding="utf-8"))
runtime = json.load(open(sys.argv[2], encoding="utf-8"))
not_applicable = (runtime.get("candidate_kernel_capability") ==
                  "no_eligible_rocmi4_mmq")
if not not_applicable and build["build_mode"] != runtime["configured_mmq_mode"]:
    raise SystemExit(
        "W4A8 build/runtime evidence mismatch: "
        f"{build['build_mode']} != {runtime['configured_mmq_mode']}")
if (not not_applicable and build["build_mode"].startswith("w4a8_iu4_") and not (
        build["saved_isa_gate"]["passed"] and runtime["passed"])):
    raise SystemExit("enabled W4A8 variant lacks ISA or dispatch proof")
PY

log "starting exact candidate for clean hard-gate timing"
CONTAINER="qwen-real-gate-timing-$$"
docker run -d --name "$CONTAINER" --network host "${GPU_ARGS[@]}" \
  "${W4A8_RUNTIME_ARGS[@]}" \
  -v "$(dirname "$MODEL"):/gate/model:ro" -v "$MTP:/gate/mtp.gguf:ro" \
  -e DFLASH_QWEN_MTP=/gate/mtp.gguf \
  -e "DFLASH_QWEN_MTP_DEPTH=$MTP_DEPTH" \
  -e EMBER_IDLE_RECLAIM_SECS=0 \
  --entrypoint "$BINARY" "$IMAGE" \
  -m "/gate/model/$(basename "$MODEL")" --host 127.0.0.1 --port "$PORT" --max-ctx 8192 \
  >/dev/null
TIMING_HOST_PID="$(docker inspect --format '{{.State.Pid}}' "$CONTAINER")"
[[ "$TIMING_HOST_PID" =~ ^[0-9]+$ && "$TIMING_HOST_PID" -gt 1 &&
   -r "/proc/$TIMING_HOST_PID/status" ]] ||
  die "timing container host PID is missing or not readable"

benchmark_args=(
  --endpoint "http://127.0.0.1:$PORT/v1/chat/completions"
  --health-endpoint "http://127.0.0.1:$PORT/health" --health-timeout 1800
  --model qwen3.8-flash-next --output "$OUT_DIR/timing.jsonl"
  --protocol hard-gate --prefill-target 412.0 --decode-target 39.49
  --calibrate-qwen-shapes
  --server-pid "$TIMING_HOST_PID" --gtt-cap-bytes 133143986176
)
if (( ! MEASUREMENT_ONLY )); then
  benchmark_args+=(--require-gate --require-memory-gate)
fi
if ! python3 "$BENCHMARK" "${benchmark_args[@]}"; then
  docker logs --tail 80 "$CONTAINER" >"$OUT_DIR/timing-server-failure.log" 2>&1 || true
  die "timing, performance, or memory gate failed"
fi
docker logs "$CONTAINER" >"$OUT_DIR/timing-server.log" 2>&1 ||
  die "could not capture the timing server log"
python3 - "$OUT_DIR/timing-server.log" "$OUT_DIR/kernel-runtime-evidence.json" \
  "$OUT_DIR/timing-kernel-mode.json" <<'PY'
import json, re, sys
log = open(sys.argv[1], encoding="utf-8", errors="replace").read()
marker_lines = [line[line.index("ROCmI4"):].strip()
                for line in log.splitlines() if "ROCmI4 W4A" in line]
known = re.compile(
    r"ROCmI4 W4A8 IU4: exact experimental MMQ enabled for device [0-9]+; "
    r"activation_prepack=(?:on|off)|"
    r"ROCmI4 W4A4: enabled for device [0-9]+ "
    r"\(lossy prompt-processing path\)")
unknown = [line for line in marker_lines if known.fullmatch(line) is None]
if unknown:
    raise SystemExit(f"timing server logged unrecognized ROCmI4 markers: {unknown}")
states = set(re.findall(
    r"ROCmI4 W4A8 IU4: exact experimental MMQ enabled for device [0-9]+; "
    r"activation_prepack=(on|off)", log))
if len(states) > 1:
    raise SystemExit(f"timing server logged conflicting W4A8 variants: {states}")
w4a4 = bool(re.search(r"ROCmI4 W4A4: enabled for device [0-9]+", log))
if states and w4a4:
    raise SystemExit("timing server logged mutually exclusive W4A4 and W4A8")
runtime = json.load(open(sys.argv[2], encoding="utf-8"))
capability = runtime.get("candidate_kernel_capability")
if capability == "no_eligible_rocmi4_mmq":
    if states or w4a4:
        raise SystemExit("not-applicable candidate enabled a ROCMI4 timing kernel")
    mode = "not_applicable_no_eligible_rocmi4_mmq"
else:
    state = next(iter(states), None)
    mode = ({"on": "w4a8_iu4_prepack",
             "off": "w4a8_iu4_register_pack"}.get(
                state, "lossy_w4a4_mmq" if w4a4 else
                "exact_int8_mmq_control"))
if mode != runtime["candidate_timing_kernel_mode"]:
    raise SystemExit(
        f"clean timing used {mode}, expected candidate mode "
        f"{runtime['candidate_timing_kernel_mode']}")
with open(sys.argv[3], "x", encoding="utf-8") as stream:
    json.dump({"configured_mmq_mode": mode,
               "confirmation": "clean_timing_startup_marker",
               "passed": True}, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
python3 - "$OUT_DIR/timing.jsonl" "$OUT_DIR/memory-evidence.json" \
  "$MEASUREMENT_ONLY" <<'PY'
import json, math, statistics, sys
measurement_only = sys.argv[3] == "1"
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
decode = [row for row in rows if row.get("kind") == "request" and
          row.get("group") == "decode-256" and row.get("ok")]
if len(decode) != 3 or not all(row.get("spec_ran") is True for row in decode):
    raise SystemExit("hard-gate decode samples did not all run native MTP")
timing_keys = ("spec_provider_age_ms", "spec_provider_block_ms",
               "spec_head_ms", "spec_verify_ms")
if any(not isinstance(row.get("spec_cycles"), int)
       or isinstance(row.get("spec_cycles"), bool)
       or row["spec_cycles"] <= 0
       or any(not isinstance(row.get(key), (int, float))
              or isinstance(row.get(key), bool)
              or not math.isfinite(float(row[key])) or float(row[key]) < 0.0
              for key in timing_keys)
       or isinstance(row.get("accept_rate"), bool)
       or not isinstance(row.get("accept_rate"), (int, float))
       or not 0.0 < float(row["accept_rate"]) < 1.0
       for row in decode):
    raise SystemExit("hard-gate decode samples lack complete native-MTP timing evidence")
summaries = [row for row in rows if row.get("kind") == "summary"]
if len(summaries) != 1 or not isinstance(summaries[0].get("hard_gate"), dict):
    raise SystemExit("machine-readable hard gate is absent")
summary = summaries[0]
cycles = sum(row["spec_cycles"] for row in decode)
speculation = {
    "samples": len(decode),
    "timing_complete": True,
    "cycles": cycles,
    "accept_rate_mean": statistics.fmean(
        float(row["accept_rate"]) for row in decode),
}
for key in timing_keys:
    total = sum(float(row[key]) for row in decode)
    speculation[f"{key}_total"] = total
    speculation[f"{key}_per_cycle"] = total / cycles
if ((summary.get("groups") or {}).get("decode-256") or {}).get(
        "speculation") != speculation:
    raise SystemExit("native-MTP summary differs from its request timing rows")
memory = summary.get("memory_gate") or {}
resources = summary.get("resources") or {}
metadata = [row for row in rows if row.get("kind") == "metadata"]
if (len(metadata) != 1 or metadata[0].get("server_pid_source") != "explicit" or
        metadata[0].get("container_pid") != resources.get("server_host_pid")):
    raise SystemExit("timing evidence is not bound to one explicit container host PID")
if not measurement_only and not memory.get("passed"):
    raise SystemExit(f"runner RSS/GTT/UMA hard fit failed: {memory}")
if not measurement_only and not summary["hard_gate"].get("passed"):
    raise SystemExit(f"performance hard gate failed: {summary['hard_gate']}")
if (resources.get("peak_memory_measurement_method") !=
        "runner_rss_gtt_sampler_v1"):
    raise SystemExit("timing run lacks runner_rss_gtt_sampler_v1 evidence")
with open(sys.argv[2], "x", encoding="utf-8") as stream:
    json.dump({"resources": resources, "hard_fit": memory,
               "performance": summary["hard_gate"],
               "speculation": speculation}, stream,
              indent=2, sort_keys=True)
    stream.write("\n")
PY
remove_container

for _ in $(seq 1 60); do
  available_kib="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
  (( available_kib >= 100 * 1024 * 1024 )) && break
  sleep 5
done
(( available_kib >= 100 * 1024 * 1024 )) ||
  die "UMA did not drain before the separate profiler passes"

log "running separate trace/counter passes (never timing evidence)"
"$PROFILE_SCRIPT" --no-quiesce --image "$PROFILE_IMAGE" --binary "$BINARY" \
  --model "$MODEL" --mtp "$MTP" --mtp-depth "$MTP_DEPTH" --port "$PORT" \
  --prefill-words 2048 --decode-tokens 256 --gap-secs 3 \
  "${W4A8_PROFILE_ARGS[@]}" --out-dir "$OUT_DIR/profile"
cp "$COUNTER_CALIBRATION" "$OUT_DIR/profile/counter-calibration.json"
python3 "$PROFILE_REPORT" "$OUT_DIR/profile" \
  --counter-calibration "$OUT_DIR/profile/counter-calibration.json" --json \
  >"$OUT_DIR/profile/report.json"

# Restore production and release the ownership lock before granting approval.
# A failed restore is a failed gate and therefore cannot leave a publish marker.
restore_exclusive || die "failed to restore production or release the GPU lock"

python3 - "$OUT_DIR" "$IMAGE" "$IMAGE_DIGEST" "$PROFILE_IMAGE" \
  "$PROFILE_IMAGE_DIGEST" "$candidate_revision" "$candidate_binary_sha" \
  "$MODEL" "$MODEL_SHA256" "$MTP" "$MTP_SHA256" "$MTP_DEPTH" \
  "$MEASUREMENT_ONLY" <<'PY'
import hashlib, json, os, sys, time
(out, image, image_digest, profile_image, profile_image_digest, revision,
 binary_sha, model, model_sha, mtp, mtp_sha, depth, measurement_only) = sys.argv[1:]
def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
inventory = json.load(open(os.path.join(out, "model-inventory.json"), encoding="utf-8"))
memory = json.load(open(os.path.join(out, "memory-evidence.json"), encoding="utf-8"))
kernel_runtime = json.load(open(os.path.join(out, "kernel-runtime-evidence.json"), encoding="utf-8"))
kernel_build = json.load(open(os.path.join(out, "w4a8-build-evidence.json"), encoding="utf-8"))
timing_kernel_mode = json.load(open(os.path.join(out, "timing-kernel-mode.json"), encoding="utf-8"))
passed = bool(memory["performance"].get("passed") and memory["hard_fit"].get("passed"))
record = {
    "schema": "ember.qwen3.8.real-weight-gate.v2",
    "passed": passed,
    "hardware_certified": passed,
    "publish_approved": False,
    "certification_scope": ("text_model_plus_mtp_only" if passed
                            else "measurement_only_not_certified"),
    "completed_unix": time.time(),
    "image": {"ref": image, "digest": image_digest},
    "profile_image": {"ref": profile_image, "digest": profile_image_digest,
                      "ember_revision": revision,
                      "candidate_binary_sha256": binary_sha,
                      "candidate_binary_byte_identical": True},
    "model": {"path": model, "sha256": model_sha,
              "ordered_inventory": inventory},
    "mtp": {"path": mtp, "sha256": mtp_sha, "depth": int(depth)},
    "hard_gates": {"prefill_2074_peak_tps": 412.0,
                   "decode_256_median_tps": 39.49, "samples": 3,
                   "performance": memory["performance"],
                   "memory": memory["hard_fit"]},
    "resources": memory["resources"],
    "speculation": memory["speculation"],
    "kernel_runtime": kernel_runtime,
    "kernel_build": kernel_build,
    "timing_kernel_mode": timing_kernel_mode,
    "evidence": {
        "differential": {"path": "differential.json",
                         "sha256": sha(os.path.join(out, "differential.json"))},
        "timing": {"path": "timing.jsonl",
                   "sha256": sha(os.path.join(out, "timing.jsonl"))},
        "memory": {"path": "memory-evidence.json",
                   "sha256": sha(os.path.join(out, "memory-evidence.json"))},
        "kernel_runtime": {"path": "kernel-runtime-evidence.json",
                   "sha256": sha(os.path.join(out, "kernel-runtime-evidence.json"))},
        "kernel_build": {"path": "w4a8-build-evidence.json",
                   "sha256": sha(os.path.join(out, "w4a8-build-evidence.json"))},
        "kernel_isa_gate": {"path": "w4a8-isa-gate.txt",
                   "sha256": sha(os.path.join(out, "w4a8-isa-gate.txt"))},
        "timing_kernel_mode": {"path": "timing-kernel-mode.json",
                   "sha256": sha(os.path.join(out, "timing-kernel-mode.json"))},
        "dispatch_server_log": {"path": "differential-dispatch-server.log",
                   "sha256": sha(os.path.join(out, "differential-dispatch-server.log"))},
        "timing_server_log": {"path": "timing-server.log",
                   "sha256": sha(os.path.join(out, "timing-server.log"))},
        "model_inventory": {"path": "model-inventory.json",
                   "sha256": sha(os.path.join(out, "model-inventory.json"))},
        "quant_build_record": {"path": "qwen-quant-build-record.json",
                   "sha256": sha(os.path.join(out, "qwen-quant-build-record.json"))},
        "profile": {"path": "profile/manifest.json",
                    "sha256": sha(os.path.join(out, "profile/manifest.json"))},
        "profile_report": {"path": "profile/report.json",
                    "sha256": sha(os.path.join(out, "profile/report.json"))},
        "counter_calibration": {"path": "profile/counter-calibration.json",
                    "sha256": sha(os.path.join(out, "profile/counter-calibration.json"))},
    },
    "methodology": "clean timing and profiler/counter passes are separate",
}
temporary = os.path.join(out, f".hardware-certified.{os.getpid()}.tmp")
with open(temporary, "x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush(); os.fsync(stream.fileno())
measured = os.path.join(out, "hardware-measured.json")
os.replace(temporary, measured)
if passed:
    os.link(measured, os.path.join(out, "hardware-certified.json"))
print(json.dumps(record, indent=2, sort_keys=True))
PY
if [[ -f "$OUT_DIR/hardware-certified.json" ]]; then
  log "PASS: text+MTP hardware certification recorded after correctness, timing, profiling, and restore"
elif (( MEASUREMENT_ONLY )); then
  log "MEASURED: complete below-threshold result retained; hardware certification was not granted"
else
  die "hard gates did not pass"
fi
