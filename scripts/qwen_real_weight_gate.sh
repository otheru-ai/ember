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
PROFILE_SCRIPT="$REPO/scripts/profile_gpu.sh"
BENCHMARK="$REPO/scripts/bench/benchmark.py"

IMAGE=""; IMAGE_DIGEST=""; PROFILE_IMAGE=""; PROFILE_IMAGE_DIGEST=""
MODEL=""; MODEL_SHA256=""
MTP=""; MTP_SHA256=""; OUT_DIR=""
BINARY=/usr/local/bin/ember-dflash
PORT=18086; MTP_DEPTH=4; DRY_RUN=0
CONTAINER=""; LOCK_HELD=0; MASKED=0; RESTORE_SERVICE=0

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
  --model-sha256 HEX          expected target SHA-256
  --mtp ABS_PATH              exact matching-quant MTP GGUF
  --mtp-sha256 HEX            expected MTP SHA-256
  --out-dir ABS_PATH          new evidence directory (must not exist)

options:
  --binary PATH               ember-dflash path in image
  --port N                    temporary loopback port (default 18086)
  --mtp-depth N               proposal depth 1..4 (default 4)
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
    --mtp) MTP="${2:?--mtp needs a path}"; shift 2 ;;
    --mtp-sha256) MTP_SHA256="${2:?--mtp-sha256 needs a value}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?--out-dir needs a path}"; shift 2 ;;
    --binary) BINARY="${2:?--binary needs a path}"; shift 2 ;;
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --mtp-depth) MTP_DEPTH="${2:?--mtp-depth needs a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ -n "$IMAGE" && -n "$IMAGE_DIGEST" && -n "$PROFILE_IMAGE" &&
   -n "$PROFILE_IMAGE_DIGEST" && -n "$MODEL" &&
   -n "$MODEL_SHA256" && -n "$MTP" && -n "$MTP_SHA256" &&
   -n "$OUT_DIR" ]] || die "all image/model/MTP paths and digests plus --out-dir are required"
[[ "$IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] ||
  die "--image-digest must be sha256: plus 64 lowercase hex characters"
[[ "$PROFILE_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] ||
  die "--profile-image-digest must be sha256: plus 64 lowercase hex characters"
[[ "$MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  die "--model-sha256 must be 64 lowercase hex characters"
[[ "$MTP_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  die "--mtp-sha256 must be 64 lowercase hex characters"
[[ "$MODEL" = /* && "$MTP" = /* && "$OUT_DIR" = /* ]] ||
  die "--model, --mtp and --out-dir must be absolute paths"
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
  target sha256       $MODEL_SHA256
  MTP companion       $MTP
  MTP sha256          $MTP_SHA256
  MTP depth           $MTP_DEPTH
  evidence            $OUT_DIR
  ownership           $GPU_LOCK acquire/release
  production          $PRODUCTION stop + mask; unconditional unmask/restore
  correctness         q=1 vs native MTP, fresh + snapshot restore + rejection evidence
  timing              unprofiled 3x exact-2074 prefill and 3x decode-256
  hard gates          prefill median >=412.0 tok/s; decode median >=39.49 tok/s
  profiling           later, separate trace and one-counter-per-PMC passes
  publication         approval marker written only after restore and every gate passes
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
[[ -x "$GPU_LOCK" ]] || die "missing fixed-purpose GPU lock wrapper: $GPU_LOCK"
[[ -x "$PRODUCTION" ]] || die "missing production wrapper: $PRODUCTION"
[[ -x "$PROFILE_SCRIPT" && -f "$BENCHMARK" ]] || die "gate dependencies are missing"
[[ -f "$MODEL" && -f "$MTP" ]] || die "model or MTP companion does not exist"
[[ -r /dev/kfd && -d /dev/dri ]] || die "this gate must run on the gfx1151 host"
[[ ! -e "$OUT_DIR" ]] || die "--out-dir must not already exist (prevents stale approval reuse)"
if grep -Eq '(^| )(iommu|amd_iommu)=off( |$)' /proc/cmdline; then
  die "IOMMU is disabled"
fi
if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  die "port $PORT is already serving"
fi

mkdir -p "$OUT_DIR"
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

GPU_ARGS=(
  --device /dev/kfd --device /dev/dri
  --group-add video --group-add render
  --ipc host --security-opt seccomp=unconfined
  --ulimit memlock=-1:-1
)

remove_container() {
  if [[ -n "$CONTAINER" ]]; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    CONTAINER=""
  fi
}

restore_exclusive() {
  local failed=0 attempt
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
  RESTORE_SERVICE=1
  sudo -n "$PRODUCTION" stop
fi
sudo -n "$PRODUCTION" mask
MASKED=1

direct_sha256() {
  python3 - "$1" <<'PY'
import hashlib, subprocess, sys
path = sys.argv[1]
digest = hashlib.sha256()
process = subprocess.Popen(
    ["dd", f"if={path}", "iflag=direct", "bs=8M", "status=none"],
    stdout=subprocess.PIPE)
assert process.stdout is not None
while chunk := process.stdout.read(8 * 1024 * 1024):
    digest.update(chunk)
if process.wait() != 0:
    raise SystemExit(f"O_DIRECT integrity read failed: {path}")
print(digest.hexdigest())
PY
}

log "verifying target and MTP with O_DIRECT reads"
[[ "$(direct_sha256 "$MODEL")" == "$MODEL_SHA256" ]] || die "target model digest mismatch"
[[ "$(direct_sha256 "$MTP")" == "$MTP_SHA256" ]] || die "MTP companion digest mismatch"

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

log "running q=1/native-batch snapshot differential"
CONTAINER="qwen-real-gate-validate-$$"
docker run --name "$CONTAINER" "${GPU_ARGS[@]}" \
  -v "$MODEL:/gate/model.gguf:ro" \
  -v "$MTP:/gate/mtp.gguf:ro" \
  -v "$OUT_DIR/validation-prompt.txt:/gate/prompt.txt:ro" \
  -e DFLASH_QWEN_MTP=/gate/mtp.gguf \
  -e "DFLASH_QWEN_MTP_DEPTH=$MTP_DEPTH" \
  --entrypoint "$BINARY" "$IMAGE" \
  -m /gate/model.gguf --max-ctx 8192 \
  --validate-prompt /gate/prompt.txt --validate-tokens 64 \
  >"$OUT_DIR/differential.json"
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

log "starting exact candidate for clean hard-gate timing"
CONTAINER="qwen-real-gate-timing-$$"
docker run -d --name "$CONTAINER" --network host "${GPU_ARGS[@]}" \
  -v "$MODEL:/gate/model.gguf:ro" -v "$MTP:/gate/mtp.gguf:ro" \
  -e DFLASH_QWEN_MTP=/gate/mtp.gguf \
  -e "DFLASH_QWEN_MTP_DEPTH=$MTP_DEPTH" \
  -e EMBER_IDLE_RECLAIM_SECS=0 \
  --entrypoint "$BINARY" "$IMAGE" \
  -m /gate/model.gguf --host 127.0.0.1 --port "$PORT" --max-ctx 8192 \
  >/dev/null
for _ in $(seq 1 360); do
  if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true || {
    docker logs --tail 80 "$CONTAINER" >"$OUT_DIR/timing-server-failure.log" 2>&1 || true
    die "timing server exited during load"
  }
  sleep 5
done
curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null ||
  die "timing server did not become healthy within 30 minutes"

python3 "$BENCHMARK" \
  --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
  --model qwen3.8-flash-next --output "$OUT_DIR/timing.jsonl" \
  --protocol hard-gate --prefill-target 412.0 --decode-target 39.49 \
  --require-gate
python3 - "$OUT_DIR/timing.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
decode = [row for row in rows if row.get("kind") == "request" and
          row.get("group") == "decode-256" and row.get("ok")]
if len(decode) != 3 or not all(row.get("spec_ran") is True for row in decode):
    raise SystemExit("hard-gate decode samples did not all run native MTP")
summaries = [row for row in rows if row.get("kind") == "summary"]
if len(summaries) != 1 or not (summaries[0].get("hard_gate") or {}).get("passed"):
    raise SystemExit("machine-readable hard gate is absent or failed")
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
  --out-dir "$OUT_DIR/profile"

# Restore production and release the ownership lock before granting approval.
# A failed restore is a failed gate and therefore cannot leave a publish marker.
restore_exclusive || die "failed to restore production or release the GPU lock"

python3 - "$OUT_DIR" "$IMAGE" "$IMAGE_DIGEST" "$PROFILE_IMAGE" \
  "$PROFILE_IMAGE_DIGEST" "$candidate_revision" "$candidate_binary_sha" \
  "$MODEL" "$MODEL_SHA256" "$MTP" "$MTP_SHA256" "$MTP_DEPTH" <<'PY'
import hashlib, json, os, sys, time
(out, image, image_digest, profile_image, profile_image_digest, revision,
 binary_sha, model, model_sha, mtp, mtp_sha, depth) = sys.argv[1:]
def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
record = {
    "schema": "ember.qwen3.8.real-weight-gate.v1",
    "passed": True,
    "publish_approved": True,
    "completed_unix": time.time(),
    "image": {"ref": image, "digest": image_digest},
    "profile_image": {"ref": profile_image, "digest": profile_image_digest,
                      "ember_revision": revision,
                      "candidate_binary_sha256": binary_sha,
                      "candidate_binary_byte_identical": True},
    "model": {"path": model, "sha256": model_sha},
    "mtp": {"path": mtp, "sha256": mtp_sha, "depth": int(depth)},
    "hard_gates": {"prefill_2074_median_tps": 412.0,
                   "decode_256_median_tps": 39.49, "samples": 3},
    "evidence": {
        "differential": {"path": "differential.json",
                         "sha256": sha(os.path.join(out, "differential.json"))},
        "timing": {"path": "timing.jsonl",
                   "sha256": sha(os.path.join(out, "timing.jsonl"))},
        "profile": {"path": "profile/manifest.json",
                    "sha256": sha(os.path.join(out, "profile/manifest.json"))},
    },
    "methodology": "clean timing and profiler/counter passes are separate",
}
temporary = os.path.join(out, f".publish-approved.{os.getpid()}.tmp")
with open(temporary, "x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush(); os.fsync(stream.fileno())
os.replace(temporary, os.path.join(out, "publish-approved.json"))
print(json.dumps(record, indent=2, sort_keys=True))
PY
log "PASS: publish approval recorded only after correctness, timing, profiling, and restore"
