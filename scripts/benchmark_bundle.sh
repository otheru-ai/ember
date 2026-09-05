#!/usr/bin/env bash
# Assemble a publishable performance bundle for one Ember release.
#
# Produces benchmarks/ember-<date>/ containing the harnesses that were run, the
# raw JSONL they emitted, a generated chart, machine-readable environment and
# summary JSON, and a README. The bundle is the unit published alongside the
# Hugging Face model card, so everything needed to re-run it is inside.
#
#   # a repo build
#   scripts/benchmark_bundle.sh --out /tmp/bundle --release 2026.8.24
#
#   # the published image, which is what certification measures
#   scripts/benchmark_bundle.sh --out /tmp/bundle --release 2026.8.24 \
#     --image ghcr.io/otheru-ai/ember:sha-abc123456789 \
#     --binary /usr/local/bin/ember-dflash --no-repo-mount
#
# Requires the GPU, the model pair, and exclusive use of the box. By default it
# takes /root/gpu.lock and stops production for the duration, restoring it on
# every exit path. Pass --no-exclusive when the caller already owns the machine
# and manages production itself -- the certification job does, and a trap that
# restarted production mid-job would hand the GPU away underneath it.
set -uo pipefail

# Saved before parsing: the argument loop shifts $@ empty, so the flock re-exec
# below must replay these rather than "$@".
ORIG_ARGS=("$@")

OUT=""; RELEASE=""; MODEL_DIR=${MODEL_DIR:-/srv/models}
IMAGE=${IMAGE:-ember-rocm:10.0}
BIN=${BIN:-/ember/build-rocm/ember-dflash}
# Bind-mounted only when benchmarking a repo build. Certification measures the
# published image, which carries its own binary -- benchmarking anything else
# would attribute numbers to something that did not ship. Set REPO_MOUNT= to
# disable.
REPO_MOUNT=${REPO_MOUNT-/root/ember}
TARGET=${TARGET:-$MODEL_DIR/DeepSeek-V4-Flash-0731-ablit1042-v2.gguf}
DRAFT=${DRAFT:-$MODEL_DIR/DeepSeek-V4-Flash-0731-ablit1042-DSpark-draft.gguf}
MMPROJ=${MMPROJ:-}
MODEL_NAME=${EMBER_BENCH_MODEL:-deepseek-v4-flash}
EXPERT_TOP_K=${EXPERT_TOP_K:-4}
DS4_PREFILL=${DS4_PREFILL:-}
SERVER_LD_LIBRARY_PATH=${SERVER_LD_LIBRARY_PATH:-}
PORT=${PORT:-18083}
SKIP_CTX=0
# Validate everything this job needs and stop before the first server starts.
# The benchmark job has now failed twice on setup alone -- a digest asserted
# without its provenance field, and a tower never exported -- and each of those
# was discovered only after certification had already taken the box and
# quiesced production. A dry run reaches the same checks in seconds, on any
# machine, with no GPU and no downtime.
DRY_RUN=0
EXCLUSIVE=${EXCLUSIVE:-1}
LOCK=${LOCK:-/root/gpu.lock}

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --release) RELEASE="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --binary) BIN="$2"; shift 2 ;;
    --no-repo-mount) REPO_MOUNT=""; shift ;;
    --no-exclusive) EXCLUSIVE=0; shift ;;
    --skip-context-sweep) SKIP_CTX=1; shift ;;
    # Implies --no-exclusive: a validation pass must never quiesce production.
    --dry-run) DRY_RUN=1; EXCLUSIVE=0; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "--out is required" >&2; exit 2; }
[ -n "$RELEASE" ] || { echo "--release is required" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Pinned probe image for the vision measurement. Committed and digest-bound so
# the number describes the same request every time; content does not matter for
# timing, resolution does. Defined after HERE, which it depends on.
VISION_PROBE=${VISION_PROBE:-$HERE/share/bench/vision-probe.png}
DATE="$(date -u +%Y-%m-%d)"
BUNDLE="$OUT/ember-$DATE"
mkdir -p "$BUNDLE"

# Dry-run validates inputs without consulting GPU groups or Docker.
if [ "$DRY_RUN" = 0 ]; then
  VIDEO_GID=$(getent group video | cut -d: -f3)
  RENDER_GID=$(getent group render | cut -d: -f3)
  [[ "$VIDEO_GID" =~ ^[0-9]+$ && "$RENDER_GID" =~ ^[0-9]+$ ]] \
    || { echo "video/render host GIDs unavailable" >&2; exit 1; }
  # Docker resolves symbolic --group-add values inside the image: the dev image
  # has no matching entries, while the release image's entries use GIDs that do
  # not match this host. Bind the host GIDs that actually own /dev/kfd instead.
  DOCKER_GPU="--device /dev/kfd --device /dev/dri --group-add $VIDEO_GID --group-add $RENDER_GID"
  DOCKER_RUN="$DOCKER_GPU --ipc host --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit core=-1"
fi
NAME="bench-bundle-$$"

if [ "$EXCLUSIVE" = 1 ]; then
  # Re-exec under the documented lock so a concurrent agent cannot perturb the
  # numbers. Guard against looping if flock is unavailable.
  if [ -z "${BENCH_BUNDLE_LOCKED:-}" ] && command -v flock >/dev/null 2>&1; then
    export BENCH_BUNDLE_LOCKED=1
    exec flock -w 7200 "$LOCK" "$0" "${ORIG_ARGS[@]}"
  fi
  # Order matters: production holds /dev/kfd, so trap first, then quiesce, THEN
  # wait for the device to drain. Waiting before quiescing deadlocks.
  restore() {
    docker rm -f "$NAME" >/dev/null 2>&1
    sudo -n /usr/local/sbin/ember-cert-production start >/dev/null 2>&1 && echo "  production restored"
  }
  trap restore EXIT INT TERM
  sudo -n /usr/local/sbin/ember-cert-production stop >/dev/null 2>&1 && echo "  production quiesced"
elif [ "$DRY_RUN" = 0 ]; then
  # Caller owns the machine and production; only clean up our own container.
  trap 'docker rm -f "$NAME" >/dev/null 2>&1' EXIT INT TERM
fi
# Waiting for a previous 85 GiB resident model to be released. Pointless under
# --dry-run, which loads nothing and would otherwise sit here for five minutes
# on any machine that is not the box.
if [ "$DRY_RUN" = 0 ]; then
  for _ in $(seq 1 60); do [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 100 ] && break; sleep 5; done
fi

echo "=== model integrity ==="
# Hashing the pair is ~96 GiB of reads. The certification job already verifies
# both digests against pinned constants before this runs, so accept them from
# the environment when the caller has them and only hash when it does not.
if [ -z "${TARGET_SHA:-}" ]; then
  echo "  hashing target (85 GiB, this takes a while)..."
  TARGET_SHA=$(sha256sum "$TARGET" | cut -d' ' -f1)
  TARGET_SHA_SOURCE=computed
  TARGET_SHA_ASSERTED_BY=""
else
  TARGET_SHA_SOURCE=asserted
  [ -n "${TARGET_SHA_ASSERTED_BY:-}" ] \
    || { echo "  TARGET_SHA_ASSERTED_BY is required with TARGET_SHA" >&2; exit 1; }
fi
if [ -z "${DRAFT_SHA:-}" ]; then
  DRAFT_SHA=$(sha256sum "$DRAFT" | cut -d' ' -f1)
  DRAFT_SHA_SOURCE=computed
  DRAFT_SHA_ASSERTED_BY=""
else
  DRAFT_SHA_SOURCE=asserted
  [ -n "${DRAFT_SHA_ASSERTED_BY:-}" ] \
    || { echo "  DRAFT_SHA_ASSERTED_BY is required with DRAFT_SHA" >&2; exit 1; }
fi
[ -n "$TARGET_SHA" ] && [ -n "$DRAFT_SHA" ] || { echo "  model digests unavailable" >&2; exit 1; }
echo "  target $TARGET_SHA"
echo "  draft  $DRAFT_SHA"
if [ -n "$MMPROJ" ]; then
  if [ -z "${MMPROJ_SHA:-}" ]; then
    MMPROJ_SHA=$(sha256sum "$MMPROJ" | cut -d' ' -f1)
    MMPROJ_SHA_SOURCE=computed
    MMPROJ_SHA_ASSERTED_BY=""
  else
    MMPROJ_SHA_SOURCE=asserted
    [ -n "${MMPROJ_SHA_ASSERTED_BY:-}" ] \
      || { echo "  MMPROJ_SHA_ASSERTED_BY is required with MMPROJ_SHA" >&2; exit 1; }
  fi
  [ -n "$MMPROJ_SHA" ] || { echo "  mmproj digest unavailable" >&2; exit 1; }
  echo "  mmproj $MMPROJ_SHA"
fi

EMBER_VERIFY_EXISTING_SHA256=${EMBER_VERIFY_EXISTING_SHA256:-1}
COMMON_ENV="-e EMBER_VERIFY_EXISTING_SHA256=$EMBER_VERIFY_EXISTING_SHA256"
[ -n "$SERVER_LD_LIBRARY_PATH" ] \
  && COMMON_ENV="$COMMON_ENV -e LD_LIBRARY_PATH=$SERVER_LD_LIBRARY_PATH"
VISION_ARGS=""
[ -n "$MMPROJ" ] && VISION_ARGS="--vision-mmproj $MMPROJ"
PREFILL_ARGS=""
[ -n "$DS4_PREFILL" ] && PREFILL_ARGS="--ds4-prefill $DS4_PREFILL"

SPEC_ENV="-e DFLASH_DS4_SPEC=1 -e DFLASH_DS4_DRAFT=$DRAFT -e DFLASH_DS4_Q5_VERIFY=1
 -e DFLASH_DS4_SPEC_Q=6 -e DFLASH_DS4_FUSED_VERIFY=1 -e DFLASH_DS4_BATCH_VERIFY=1
 -e DFLASH_DS4_DECODE_FLASH=1 -e DFLASH_DS4_SPEC_MAX_CTX=49152
 -e DFLASH_DS4_BATCH_WARMUP_TOKENS=48 -e LUCE_MMVQ_MAX_NCOLS=6
 -e DFLASH_DS4_TIMING=0 -e EMBER_GTT_TRACE=0
 -e GGML_CUDA_DISABLE_GRAPHS=1 -e GGML_CUDA_POOL_MAX_MB=8192"

# Older releases do not all take --host; it was added partway through, and a
# build that predates it binds loopback anyway. Benchmarking a shipped release
# means running it as it shipped, so probe rather than assume. The usage text is
# printed on any unrecognised option, so this works even where --help is not one.
HOST_ARG="--host 127.0.0.1"
# A host-staged binary lives under MODEL_DIR and needs the same library bundle
# as the measured server. Probe with both mounted, otherwise exec failure is
# misreported as an old release that predates --host.
# Skipped under --dry-run: it probes the IMAGE's CLI, which is not what a
# wiring check is for, and pulling a 20 GiB image to read --help would make the
# cheap check expensive.
# shellcheck disable=SC2086
if [ "$DRY_RUN" = 0 ] && ! docker run --rm -v "$MODEL_DIR:$MODEL_DIR" $COMMON_ENV \
    --entrypoint "$BIN" "$IMAGE" --help 2>&1 | grep -q -- '--host'; then
  HOST_ARG=""
  echo "  note: $IMAGE predates --host; it binds loopback by default"
fi

start_server() { # $1 = extra env, $2 = max_ctx
  docker rm -f "$NAME" >/dev/null 2>&1
  local mount=""
  [ -n "$REPO_MOUNT" ] && mount="-v $REPO_MOUNT:/ember"
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --network host $DOCKER_RUN \
    $mount -v "$MODEL_DIR:$MODEL_DIR" $COMMON_ENV $1 \
    --entrypoint "$BIN" "$IMAGE" \
    -m "$TARGET" $VISION_ARGS --model-name "$MODEL_NAME" \
    $HOST_ARG --port "$PORT" --max-ctx "$2" $PREFILL_ARGS \
    --ds4-expert-top-k "$EXPERT_TOP_K" --default-temperature 0.6 >/dev/null
  for _ in $(seq 1 400); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    # Keep the reason before the next docker rm -f discards it. Reporting only
    # "server failed to start" cost a full re-run to learn the server had
    # rejected a command-line flag.
    docker ps --format '{{.Names}}' | grep -q "^$NAME$" || {
      docker logs "$NAME" >"$BUNDLE/server-failure.log" 2>&1 || true
      sed 's/^/    /' "$BUNDLE/server-failure.log" | head -8
      return 1
    }
    sleep 5
  done
  docker logs "$NAME" >"$BUNDLE/server-failure.log" 2>&1 || true
  return 1
}
stop_server() {
  docker stop --timeout 90 "$NAME" >/dev/null 2>&1; docker rm -f "$NAME" >/dev/null 2>&1
  for _ in $(seq 1 60); do [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 100 ] && break; sleep 5; done
}

EP="http://127.0.0.1:$PORT/v1/chat/completions"

# The image path is measured in the same server as the text throughput group,
# so the two describe one process rather than two. It needs the tower: without
# --vision-mmproj the engine refuses image input, which is correct but would
# make the block silently absent.
VISION_BENCH_ARGS=""
if [ -n "$MMPROJ" ] && [ -n "$VISION_PROBE" ]; then
  [ -r "$VISION_PROBE" ] || { echo "  vision probe unreadable: $VISION_PROBE" >&2; exit 1; }
  VISION_BENCH_ARGS="--vision-image $VISION_PROBE"
  echo "  vision probe $(sha256sum "$VISION_PROBE" | cut -d' ' -f1)"
elif [ -z "$MMPROJ" ]; then
  echo "  no MMPROJ: skipping the vision block"
fi

if [ "$DRY_RUN" = 1 ]; then
  echo "=== dry run: inputs validated, stopping before the first server ==="
  echo "  image        $IMAGE"
  echo "  binary       $BIN"
  echo "  target       $TARGET ($TARGET_SHA_SOURCE)"
  echo "  draft        $DRAFT ($DRAFT_SHA_SOURCE)"
  echo "  mmproj       ${MMPROJ:-<none>}${MMPROJ:+ (${MMPROJ_SHA_SOURCE:-})}"
  echo "  vision       ${VISION_BENCH_ARGS:-<none>}"
  echo "  bundle       $BUNDLE"
  echo "  context sweep $([ "$SKIP_CTX" = 1 ] && echo skipped || echo included)"
  echo DRY_RUN_OK
  exit 0
fi

echo "=== 1/3 throughput groups (prefill + decode) ==="
start_server "$SPEC_ENV" 65536 || { echo "  server failed to start"; exit 1; }
# shellcheck disable=SC2086  # deliberately word-split: empty means no flag
python3 "$HERE/scripts/bench/benchmark.py" --endpoint "$EP" \
  --model "$MODEL_NAME" --output "$BUNDLE/raw-results.jsonl" \
  $VISION_BENCH_ARGS >/dev/null 2>&1
echo "  $(wc -l < "$BUNDLE/raw-results.jsonl") rows"

echo "=== 2/3 decode by workload (speculation on) ==="
python3 "$HERE/scripts/bench/accept_sweep.py" "$EP" "$BUNDLE/workloads-spec-on.jsonl" >/dev/null 2>&1
stop_server
echo "=== 2/3 decode by workload (autoregressive baseline) ==="
start_server "-e DFLASH_DS4_SPEC=0 -e DFLASH_DS4_DECODE_FLASH=1 -e GGML_CUDA_DISABLE_GRAPHS=1" 65536 \
  && python3 "$HERE/scripts/bench/accept_sweep.py" "$EP" "$BUNDLE/workloads-spec-off.jsonl" >/dev/null 2>&1
stop_server

if [ "$SKIP_CTX" = 0 ]; then
  echo "=== 3/3 context sweep (this is the long one) ==="
  SIZES=0,1024,4096,16384,32768,65536,98304
  : > "$BUNDLE/context-sweep.jsonl"
  start_server "$SPEC_ENV" 131072 \
    && python3 "$HERE/scripts/bench/sweep_probe.py" "$EP" spec-on "$SIZES" "$BUNDLE/context-sweep.jsonl"
  stop_server
  start_server "-e DFLASH_DS4_SPEC=0 -e DFLASH_DS4_DECODE_FLASH=1 -e GGML_CUDA_DISABLE_GRAPHS=1" 131072 \
    && python3 "$HERE/scripts/bench/sweep_probe.py" "$EP" spec-off "$SIZES" "$BUNDLE/context-sweep.jsonl"
  stop_server
  python3 "$HERE/tools/plot_context_scaling.py" \
    "$BUNDLE/context-sweep.jsonl" "$BUNDLE/ember-context-scaling.svg"
fi

echo "=== capturing host facts ==="
# Captured here, on the machine that ran the benchmark. Assembly may happen
# elsewhere; detecting the host there would attribute numbers to the wrong box.
python3 - "$BUNDLE/host.json" <<'PY'
import json, platform, subprocess, sys
def run(cmd, default=""):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=10).stdout.strip() or default
    except Exception:
        return default
mem = run("awk '/MemTotal/{print $2}' /proc/meminfo", "0")
json.dump({
    "cpu": run("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-").strip(),
    "gpu": "AMD Radeon 8060S (gfx1151), integrated",
    "kernel": platform.release(),
    "memory_gib": round(int(mem) / 1024 / 1024) if mem.isdigit() else None,
    "os": run(". /etc/os-release 2>/dev/null && echo $PRETTY_NAME"),
}, open(sys.argv[1], "w"), indent=2, sort_keys=True)
PY
cat "$BUNDLE/host.json"

echo "=== assembling bundle ==="
cp "$HERE/scripts/bench/benchmark.py" "$HERE/scripts/bench/accept_sweep.py" \
   "$HERE/scripts/bench/sweep_probe.py" "$BUNDLE/" 2>/dev/null
TARGET_SHA="$TARGET_SHA" DRAFT_SHA="$DRAFT_SHA" RELEASE="$RELEASE" \
TARGET_SHA_SOURCE="$TARGET_SHA_SOURCE" \
TARGET_SHA_ASSERTED_BY="${TARGET_SHA_ASSERTED_BY:-}" \
DRAFT_SHA_SOURCE="$DRAFT_SHA_SOURCE" \
DRAFT_SHA_ASSERTED_BY="${DRAFT_SHA_ASSERTED_BY:-}" \
MMPROJ_SHA="${MMPROJ_SHA:-}" MMPROJ="$MMPROJ" \
MMPROJ_SHA_SOURCE="${MMPROJ_SHA_SOURCE:-}" \
MMPROJ_SHA_ASSERTED_BY="${MMPROJ_SHA_ASSERTED_BY:-}" \
MODEL_NAME="$MODEL_NAME" EXPERT_TOP_K="$EXPERT_TOP_K" DS4_PREFILL="$DS4_PREFILL" \
SERVER_LD_LIBRARY_PATH="$SERVER_LD_LIBRARY_PATH" \
EMBER_VERIFY_EXISTING_SHA256="$EMBER_VERIFY_EXISTING_SHA256" \
EXPECTED_WORKLOADS="$([ "${EMBER_BENCH_VISION:-}" = 1 ] && echo 12 || echo 10)" \
IMAGE="$IMAGE" BIN="$BIN" TARGET="$TARGET" DRAFT="$DRAFT" PORT="$PORT" \
  python3 "$HERE/scripts/bench/assemble_bundle.py" "$BUNDLE"

echo "=== bundle at $BUNDLE ==="
ls -la "$BUNDLE"
