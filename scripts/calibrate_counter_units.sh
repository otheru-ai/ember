#!/usr/bin/env bash
# Collect known-traffic ROCm counter samples on an exclusively held gfx1151.
# The resulting samples.jsonl is analyzed by calibrate_counter_units.py.
set -euo pipefail

IMAGE="${EMBER_CALIBRATION_IMAGE:-ember-rocm:10.0-dev}"
OUT_DIR=""
SIZES=(268435456 536870912 1073741824)
REPS=4
DRY_RUN=0
PRODUCTION_WRAPPER="/usr/local/sbin/ember-cert-production"
GPU_LOCK=0
RESTORE=0
MASKED=0

die() { printf 'counter-calibration: %s\n' "$*" >&2; exit 1; }
log() { printf 'counter-calibration: %s\n' "$*"; }
usage() {
  cat <<'EOF'
usage: scripts/calibrate_counter_units.sh --out-dir DIR [options]
  --image REF       ROCm 10 image containing hipcc/rocprofv3
  --out-dir DIR     fresh directory for raw CSVs and calibration record
  --reps N          repetitions per sample (default 4)
  --no-quiesce      caller already owns the GPU and stopped production
  --dry-run         print the plan without touching the GPU
EOF
}
QUIESCE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="${2:?--image needs a value}"; shift 2;;
    --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2;;
    --reps) REPS="${2:?--reps needs a value}"; shift 2;;
    --no-quiesce) QUIESCE=0; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) die "unknown option: $1";;
  esac
done
[[ -n "$OUT_DIR" && "$OUT_DIR" = /* ]] || die "--out-dir must be absolute"
[[ "$REPS" =~ ^[1-9][0-9]*$ ]] || die "--reps must be positive"
[[ ! -e "$OUT_DIR" ]] || die "output directory already exists: $OUT_DIR"

cleanup() {
  local status=$?
  set +e
  (( MASKED )) && sudo -n "$PRODUCTION_WRAPPER" unmask
  (( RESTORE )) && sudo -n "$PRODUCTION_WRAPPER" start
  (( GPU_LOCK )) && sudo -n /usr/local/sbin/ember-gpu-lock release
  exit "$status"
}
trap cleanup EXIT INT TERM

log "image=$IMAGE sizes=${SIZES[*]} reps=$REPS"
(( DRY_RUN )) && { log "dry run: no GPU touched"; exit 0; }
command -v docker >/dev/null || die "docker is required"
command -v python3 >/dev/null || die "python3 is required"
[[ -r /dev/kfd && -d /dev/dri ]] || die "gfx1151 GPU devices are unavailable"
mkdir -p "$OUT_DIR"

if (( QUIESCE )); then
  sudo -n /usr/local/sbin/ember-gpu-lock acquire; GPU_LOCK=1
  if sudo -n "$PRODUCTION_WRAPPER" is-active >/dev/null 2>&1; then
    sudo -n "$PRODUCTION_WRAPPER" stop; RESTORE=1
  fi
  sudo -n "$PRODUCTION_WRAPPER" mask; MASKED=1
fi

docker run --rm -v "$PWD:/ember:ro" -v "$OUT_DIR:/out" \
  --entrypoint hipcc "$IMAGE" -O3 --offload-arch=gfx1151 \
  -o /out/bench_counter_traffic /ember/tools/bench_counter_traffic.hip

GPU_ARGS=(--device /dev/kfd --device /dev/dri --ipc host --security-opt seccomp=unconfined)
for counter in FETCH_SIZE WRITE_SIZE; do
  mode=read; [[ "$counter" == WRITE_SIZE ]] && mode=write
  for bytes in "${SIZES[@]}"; do
    for kind in baseline traffic; do
      tag="${kind,,}-${counter,,}-${bytes}"
      selected_mode=noop; [[ "$kind" == traffic ]] && selected_mode="$mode"
      docker run --rm "${GPU_ARGS[@]}" -v "$OUT_DIR:/out" \
        --entrypoint rocprofv3 "$IMAGE" --output-format csv -d /out -o "$tag" \
        --pmc "$counter" -- /out/bench_counter_traffic \
        --mode "$selected_mode" --bytes "$bytes" --reps "$REPS" --warmup 0
    done
  done
done

python3 - "$OUT_DIR" "$REPS" "${SIZES[@]}" <<'PY'
import csv, glob, json, pathlib, sys
out, reps, *sizes = sys.argv[1:]
def total(tag, counter):
    paths = glob.glob(str(pathlib.Path(out) / f"*{tag}*counter_collection.csv"))
    if len(paths) != 1:
        raise SystemExit(f"expected one counter CSV for {tag}, found {paths}")
    with open(paths[0], newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        value = 0.0
        for row in rows:
            key = next((k for k in row if k.lower().replace('_','') == 'countername'), None)
            val = next((k for k in row if k.lower().replace('_','') == 'countervalue'), None)
            if key and val and row[key] == counter:
                value += float(row[val])
        return value
with open(pathlib.Path(out) / "samples.jsonl", "w", encoding="utf-8") as stream:
    for counter in ("FETCH_SIZE", "WRITE_SIZE"):
        for size in map(int, sizes):
            mode = "read" if counter == "FETCH_SIZE" else "write"
            traffic = total(f"traffic-{counter.lower()}-{size}", counter)
            baseline = total(f"baseline-{counter.lower()}-{size}", counter)
            json.dump({"counter": counter, "mode": mode,
                       "bytes_per_rep": size, "repetitions": int(reps),
                       "expected_bytes": size * int(reps),
                       "traffic": traffic, "baseline": baseline}, stream)
            stream.write("\n")
PY
python3 "$(dirname "$0")/calibrate_counter_units.py" \
  "$OUT_DIR/samples.jsonl" --output "$OUT_DIR/calibration.json"
log "calibration complete: $OUT_DIR/calibration.json"
