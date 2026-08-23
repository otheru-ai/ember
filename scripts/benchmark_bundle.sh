#!/usr/bin/env bash
# Assemble a publishable performance bundle for one Ember release.
#
# Produces benchmarks/ember-<date>/ containing the harnesses that were run, the
# raw JSONL they emitted, a generated chart, machine-readable environment and
# summary JSON, and a README. The bundle is the unit published alongside the
# Hugging Face model card, so everything needed to re-run it is inside.
#
#   scripts/benchmark_bundle.sh --out /tmp/bundle --release 2026.8.24
#
# Requires the GPU, the model pair, and exclusive use of the box. Takes the
# documented flock so a concurrent agent does not perturb the numbers.
set -uo pipefail

OUT=""; RELEASE=""; MODEL_DIR=${MODEL_DIR:-/srv/models}
IMAGE=${IMAGE:-ember-rocm:7.14}
BIN=${BIN:-/ember/build-rocm/ember-dflash}
TARGET=${TARGET:-$MODEL_DIR/DeepSeek-V4-Flash-0731-ablit1042-v2.gguf}
DRAFT=${DRAFT:-$MODEL_DIR/DeepSeek-V4-Flash-0731-ablit1042-DSpark-draft.gguf}
PORT=${PORT:-18083}
SKIP_CTX=0

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --release) RELEASE="$2"; shift 2 ;;
    --skip-context-sweep) SKIP_CTX=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "--out is required" >&2; exit 2; }
[ -n "$RELEASE" ] || { echo "--release is required" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE="$(date -u +%Y-%m-%d)"
BUNDLE="$OUT/ember-$DATE"
mkdir -p "$BUNDLE"

DOCKER_GPU="--device /dev/kfd --device /dev/dri --group-add video --group-add render"
DOCKER_RUN="$DOCKER_GPU --ipc host --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit core=-1"
NAME="bench-bundle-$$"

# Order matters: production holds /dev/kfd, so trap first, then quiesce, THEN
# wait for the device to drain. Waiting before quiescing deadlocks.
restore() {
  docker rm -f "$NAME" >/dev/null 2>&1
  sudo -n /usr/local/sbin/ember-cert-production start >/dev/null 2>&1 && echo "  production restored"
}
trap restore EXIT INT TERM
sudo -n /usr/local/sbin/ember-cert-production stop >/dev/null 2>&1 && echo "  production quiesced"
for _ in $(seq 1 60); do [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 100 ] && break; sleep 5; done

echo "=== model integrity ==="
TARGET_SHA=$(sha256sum "$TARGET" | cut -d' ' -f1)
DRAFT_SHA=$(sha256sum "$DRAFT" | cut -d' ' -f1)
echo "  target $TARGET_SHA"
echo "  draft  $DRAFT_SHA"

SPEC_ENV="-e DFLASH_DS4_SPEC=1 -e DFLASH_DS4_DRAFT=$DRAFT -e DFLASH_DS4_Q5_VERIFY=1
 -e DFLASH_DS4_SPEC_Q=6 -e DFLASH_DS4_FUSED_VERIFY=1 -e DFLASH_DS4_BATCH_VERIFY=1
 -e DFLASH_DS4_DECODE_FLASH=1 -e DFLASH_DS4_SPEC_MAX_CTX=49152
 -e DFLASH_DS4_BATCH_WARMUP_TOKENS=48 -e LUCE_MMVQ_MAX_NCOLS=6
 -e DFLASH_DS4_TIMING=0 -e EMBER_GTT_TRACE=0
 -e GGML_CUDA_DISABLE_GRAPHS=1 -e GGML_CUDA_POOL_MAX_MB=8192"

start_server() { # $1 = extra env, $2 = max_ctx
  docker rm -f "$NAME" >/dev/null 2>&1
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --network host $DOCKER_RUN \
    -v /root/ember:/ember -v "$MODEL_DIR:$MODEL_DIR" $1 \
    --entrypoint "$BIN" "$IMAGE" \
    -m "$TARGET" --host 127.0.0.1 --port "$PORT" --max-ctx "$2" \
    --ds4-expert-top-k 4 --default-temperature 0.6 >/dev/null
  for _ in $(seq 1 400); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps --format '{{.Names}}' | grep -q "^$NAME$" || return 1
    sleep 5
  done
  return 1
}
stop_server() {
  docker stop --timeout 90 "$NAME" >/dev/null 2>&1; docker rm -f "$NAME" >/dev/null 2>&1
  for _ in $(seq 1 60); do [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 100 ] && break; sleep 5; done
}

EP="http://127.0.0.1:$PORT/v1/chat/completions"

echo "=== 1/3 throughput groups (prefill + decode) ==="
start_server "$SPEC_ENV" 65536 || { echo "  server failed to start"; exit 1; }
python3 "$HERE/scripts/bench/benchmark.py" --endpoint "$EP" \
  --output "$BUNDLE/raw-results.jsonl" >/dev/null 2>&1
echo "  $(wc -l < "$BUNDLE/raw-results.jsonl") rows"

echo "=== 2/3 decode by workload (speculation on) ==="
python3 "$HERE/scripts/bench/accept_sweep.py" "$BUNDLE/workloads-spec-on.jsonl" >/dev/null 2>&1
stop_server
echo "=== 2/3 decode by workload (autoregressive baseline) ==="
start_server "-e DFLASH_DS4_SPEC=0 -e DFLASH_DS4_DECODE_FLASH=1 -e GGML_CUDA_DISABLE_GRAPHS=1" 65536 \
  && python3 "$HERE/scripts/bench/accept_sweep.py" "$BUNDLE/workloads-spec-off.jsonl" >/dev/null 2>&1
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
IMAGE="$IMAGE" BIN="$BIN" TARGET="$TARGET" DRAFT="$DRAFT" PORT="$PORT" \
  python3 "$HERE/scripts/bench/assemble_bundle.py" "$BUNDLE"

echo "=== bundle at $BUNDLE ==="
ls -la "$BUNDLE"
