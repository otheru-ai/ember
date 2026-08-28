#!/usr/bin/env bash
# Six-slot fresh-process A/B confirmation for the two format finalists.
# Timing is deliberately clean: profiler/counter collection is forbidden here.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_LOCK=/usr/local/sbin/ember-gpu-lock
PRODUCTION=/usr/local/sbin/ember-cert-production
PRODUCTION_HEALTH=http://127.0.0.1:8000/health
PREPARE="$REPO/scripts/qwen_balanced_confirmation.py"
SLOT="$REPO/scripts/qwen_balanced_slot.py"
BENCHMARK="$REPO/scripts/bench/benchmark.py"

PLAN=""; PLAN_SHA256=""; ACCUMULATOR=""; ACCUMULATOR_SHA256=""
EVIDENCE_ROOT=""; IMAGE=""; IMAGE_DIGEST=""; RUNTIME_REVISION=""
ENGINE_BINARY_SHA256=""; FORMAT_SHA256=""; OUT_DIR=""
BINARY=/usr/local/bin/ember-dflash; PORT=18087; DRY_RUN=0
CONTAINER=""; LOCK_HELD=0; MASKED=0; RESTORE_SERVICE=0
PRODUCTION_STATE_CAPTURED=0; PRODUCTION_WAS_ACTIVE=0

die() { printf 'qwen-balanced-confirmation: %s\n' "$*" >&2; exit 1; }
log() { printf 'qwen-balanced-confirmation: %s\n' "$*"; }

usage() {
  cat <<'EOF'
usage: scripts/qwen_balanced_confirmation.sh [options]

required:
  --plan ABS_PATH --plan-sha256 HEX
  --accumulator ABS_PATH --accumulator-sha256 HEX
  --evidence-root ABS_PATH
  --image REF --image-digest sha256:HEX
  --runtime-revision HEX40 --engine-binary-sha256 HEX
  --tensor-format-contract-sha256 HEX
  --out-dir ABS_PATH

options:
  --binary ABS_PATH             default /usr/local/bin/ember-dflash
  --port N                      default 18087
  --dry-run                     print a side-effect-free plan
EOF
}

while (( $# )); do
  case "$1" in
    --plan) PLAN="${2:?--plan needs a path}"; shift 2 ;;
    --plan-sha256) PLAN_SHA256="${2:?--plan-sha256 needs a value}"; shift 2 ;;
    --accumulator) ACCUMULATOR="${2:?--accumulator needs a path}"; shift 2 ;;
    --accumulator-sha256) ACCUMULATOR_SHA256="${2:?--accumulator-sha256 needs a value}"; shift 2 ;;
    --evidence-root) EVIDENCE_ROOT="${2:?--evidence-root needs a path}"; shift 2 ;;
    --image) IMAGE="${2:?--image needs a value}"; shift 2 ;;
    --image-digest) IMAGE_DIGEST="${2:?--image-digest needs a value}"; shift 2 ;;
    --runtime-revision) RUNTIME_REVISION="${2:?--runtime-revision needs a value}"; shift 2 ;;
    --engine-binary-sha256) ENGINE_BINARY_SHA256="${2:?--engine-binary-sha256 needs a value}"; shift 2 ;;
    --tensor-format-contract-sha256) FORMAT_SHA256="${2:?--tensor-format-contract-sha256 needs a value}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?--out-dir needs a path}"; shift 2 ;;
    --binary) BINARY="${2:?--binary needs a path}"; shift 2 ;;
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ -n "$PLAN" && -n "$PLAN_SHA256" && -n "$ACCUMULATOR" &&
   -n "$ACCUMULATOR_SHA256" && -n "$EVIDENCE_ROOT" && -n "$IMAGE" &&
   -n "$IMAGE_DIGEST" && -n "$RUNTIME_REVISION" &&
   -n "$ENGINE_BINARY_SHA256" && -n "$FORMAT_SHA256" && -n "$OUT_DIR" ]] ||
  die "all plan/accumulator/runtime/evidence arguments are required"
[[ "$PLAN" = /* && "$ACCUMULATOR" = /* && "$EVIDENCE_ROOT" = /* &&
   "$OUT_DIR" = /* && "$BINARY" = /* ]] || die "all paths must be absolute"
[[ "$PLAN_SHA256" =~ ^[0-9a-f]{64}$ && "$ACCUMULATOR_SHA256" =~ ^[0-9a-f]{64}$ &&
   "$ENGINE_BINARY_SHA256" =~ ^[0-9a-f]{64}$ && "$FORMAT_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  die "file/runtime SHA-256 values must be lowercase hexadecimal"
[[ "$IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid image digest"
[[ "$RUNTIME_REVISION" =~ ^[0-9a-f]{40}$ ]] || die "invalid runtime revision"
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1024 && PORT <= 65535 )) ||
  die "--port must be 1024..65535"

cat <<EOF
plan:
  mode                balanced format-finalist confirmation
  persisted order     counterbalanced A/B, B/A, A/B (ABBAAB)
  server lifecycle    six fresh processes (one per slot)
  workload per slot   one exact 2074-token prefill + one 256-token decode
  samples per arm     exactly 3
  timing              clean; profiling/counters forbidden during confirmation
  production          fixed GPU lock plus stop/mask and unconditional restore
  publication         forbidden
EOF
if (( DRY_RUN )); then
  log "dry run: no files, GPU, docker, sudo, lock, or production state touched"
  exit 0
fi

for command in docker curl python3 dd stat realpath; do
  command -v "$command" >/dev/null || die "$command is required"
done
[[ -x "$GPU_LOCK" && -x "$PRODUCTION" ]] || die "fixed-purpose host wrappers are missing"
[[ -f "$PREPARE" && -f "$SLOT" && -f "$BENCHMARK" ]] || die "runner dependencies are missing"
[[ -f "$PLAN" && ! -L "$PLAN" && -f "$ACCUMULATOR" && ! -L "$ACCUMULATOR" ]] ||
  die "plan or accumulator is missing/unsafe"
[[ -d "$EVIDENCE_ROOT" && ! -L "$EVIDENCE_ROOT" ]] || die "evidence root is missing/unsafe"
[[ ! -e "$OUT_DIR" && ! -L "$OUT_DIR" ]] || die "--out-dir must not already exist"
test "$(realpath -e -- "$(dirname -- "$OUT_DIR")")" = "$(realpath -e -- "$EVIDENCE_ROOT")" ||
  die "--out-dir must be a direct child of the evidence root"
[[ -r /dev/kfd && -d /dev/dri ]] || die "this runner must execute on gfx1151"
if grep -Eq '(^| )(iommu|amd_iommu)=off( |$)' /proc/cmdline; then die "IOMMU is disabled"; fi
if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  die "port $PORT is already serving"
fi

mkdir -m 0700 -- "$OUT_DIR"
RUNNER_PLAN="$OUT_DIR/runner-plan.json"
python3 "$PREPARE" prepare \
  --plan "$PLAN" --plan-sha256 "$PLAN_SHA256" \
  --accumulator "$ACCUMULATOR" --accumulator-sha256 "$ACCUMULATOR_SHA256" \
  --evidence-root "$EVIDENCE_ROOT" --runtime-revision "$RUNTIME_REVISION" \
  --engine-binary-sha256 "$ENGINE_BINARY_SHA256" --container-digest "$IMAGE_DIGEST" \
  --tensor-format-contract-sha256 "$FORMAT_SHA256" --output "$RUNNER_PLAN"

docker image inspect "$IMAGE" >"$OUT_DIR/image-inspect.json"
LOCAL_IMAGE_ID="$(python3 - "$OUT_DIR/image-inspect.json" "$IMAGE_DIGEST" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
if len(rows)!=1: raise SystemExit("runtime image did not resolve exactly once")
observed={rows[0].get("Id")}
observed.update(item.rsplit("@",1)[-1] for item in rows[0].get("RepoDigests") or [])
if sys.argv[2] not in observed: raise SystemExit("runtime image digest differs")
image_id=rows[0].get("Id")
if not isinstance(image_id,str) or not image_id.startswith("sha256:"):
    raise SystemExit("runtime image has no immutable local image ID")
print(image_id)
PY
)"
[[ "$LOCAL_IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid local image ID"
test "$(docker run --rm --entrypoint sha256sum "$LOCAL_IMAGE_ID" "$BINARY" | awk '{print $1}')" = \
  "$ENGINE_BINARY_SHA256"

python3 - "$RUNNER_PLAN" "$BENCHMARK" "$OUT_DIR" <<'PY'
import importlib.util,json,sys
from pathlib import Path
plan_path,module_path,out=sys.argv[1:]
spec=importlib.util.spec_from_file_location("balanced_inventory",module_path)
if spec is None or spec.loader is None: raise SystemExit("cannot load inventory verifier")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
plan=json.load(open(plan_path,encoding="utf-8"))
for index,row in enumerate(plan["bindings"]):
    inventory,_=module.model_inventory_from_build_record(
        Path(row["build_record"]),row["build_record_sha256"],
        Path(row["model"]),row["model_sha256"])
    Path(out,f"inventory-{index}.json").write_text(
        json.dumps(inventory,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY

declare -A GPU_GIDS=()
GPU_ARGS=(--device /dev/kfd --device /dev/dri
  --ipc host --security-opt seccomp=unconfined --ulimit memlock=-1:-1)
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
      sudo -n "$PRODUCTION" unmask >/dev/null 2>&1 && { MASKED=0; break; }
      sleep 1
    done
    (( MASKED == 0 )) || failed=1
  fi
  if (( RESTORE_SERVICE )) && (( MASKED == 0 )); then
    for attempt in 1 2 3; do
      sudo -n "$PRODUCTION" start >/dev/null 2>&1 && { RESTORE_SERVICE=0; break; }
      sleep 1
    done
  fi
  (( RESTORE_SERVICE == 0 )) || failed=1
  if (( PRODUCTION_STATE_CAPTURED && PRODUCTION_WAS_ACTIVE &&
        MASKED == 0 && RESTORE_SERVICE == 0 )); then
    for attempt in $(seq 1 300); do
      if sudo -n "$PRODUCTION" is-active >/dev/null 2>&1 &&
         curl --fail --silent --max-time 2 "$PRODUCTION_HEALTH" >/dev/null 2>&1; then
        healthy=1; break
      fi
      sleep 2
    done
    (( healthy )) || failed=1
  fi
  if (( LOCK_HELD )); then
    for attempt in 1 2 3; do
      sudo -n "$GPU_LOCK" release >/dev/null 2>&1 && { LOCK_HELD=0; break; }
      sleep 1
    done
    (( LOCK_HELD == 0 )) || failed=1
  fi
  return "$failed"
}
cleanup() { local rc=$?; trap - EXIT INT TERM; restore_exclusive || rc=1; exit "$rc"; }
trap cleanup EXIT INT TERM

sudo -n "$GPU_LOCK" acquire; LOCK_HELD=1
if sudo -n "$PRODUCTION" is-active >/dev/null 2>&1; then
  PRODUCTION_WAS_ACTIVE=1; RESTORE_SERVICE=1; sudo -n "$PRODUCTION" stop
fi
PRODUCTION_STATE_CAPTURED=1
sudo -n "$PRODUCTION" mask; MASKED=1

log "verifying both finalist shard inventories with O_DIRECT after quiescing production"
for inventory in "$OUT_DIR"/inventory-*.json; do
  python3 - "$inventory" <<'PY'
import hashlib,json,subprocess,sys
for row in json.load(open(sys.argv[1],encoding="utf-8"))["shards"]:
    digest=hashlib.sha256()
    process=subprocess.Popen(["dd",f"if={row['path']}","iflag=direct","bs=8M","status=none"],stdout=subprocess.PIPE)
    assert process.stdout is not None
    while chunk:=process.stdout.read(8*1024*1024): digest.update(chunk)
    if process.wait()!=0 or digest.hexdigest()!=row["sha256"]:
        raise SystemExit(f"O_DIRECT finalist shard integrity failed: {row['path']}")
PY
done
python3 - "$RUNNER_PLAN" <<'PY'
import hashlib,json,subprocess,sys
seen=set()
for row in json.load(open(sys.argv[1],encoding="utf-8"))["bindings"]:
    path=row["mtp"]
    if path in seen: continue
    seen.add(path); digest=hashlib.sha256()
    process=subprocess.Popen(["dd",f"if={path}","iflag=direct","bs=8M","status=none"],stdout=subprocess.PIPE)
    assert process.stdout is not None
    while chunk:=process.stdout.read(8*1024*1024): digest.update(chunk)
    if process.wait()!=0 or digest.hexdigest()!=row["mtp_sha256"]:
        raise SystemExit(f"O_DIRECT finalist MTP integrity failed: {path}")
PY

mapfile -t ORDER < <(python3 - "$RUNNER_PLAN" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
by={row["arm_id"]:row for row in p["bindings"]}
workloads={row["workload_id"]:row for row in p["confirmation_plan"]["workloads"]}
for arm,workload_id in zip(p["confirmation_plan"]["run_order"],p["confirmation_plan"]["workload_order"]):
    row=by[arm]
    workload=workloads[workload_id]
    values=(arm,row["model"],row["model_sha256"],row["mtp"],row["mtp_sha256"],
            str(row["mtp_depth"]),row["candidate_id"],row["model_inventory_sha256"],
            row["companion_inventory_sha256"],row["candidate_binding"]["sha256"],
            row["candidate_kernel_capability"],
            workload_id,workload["marker"],workload["recipe_sha256"])
    for value in values:
        if "\n" in value: raise SystemExit("runner plan contains newline")
        print(value)
PY
)
((${#ORDER[@]} == 84)) || die "runner plan did not yield six exact slots"

SLOT_ARGS=()
for index in $(seq 0 5); do
  offset=$((index * 14))
  arm="${ORDER[offset]}"; model="${ORDER[offset+1]}"; model_sha="${ORDER[offset+2]}"
  mtp="${ORDER[offset+3]}"; mtp_sha="${ORDER[offset+4]}"; mtp_depth="${ORDER[offset+5]}"
  candidate_id="${ORDER[offset+6]}"; model_inventory_sha="${ORDER[offset+7]}"
  companion_inventory_sha="${ORDER[offset+8]}"; candidate_binding_sha="${ORDER[offset+9]}"
  kernel_capability="${ORDER[offset+10]}"; workload_id="${ORDER[offset+11]}"
  workload_marker="${ORDER[offset+12]}"; workload_recipe_sha="${ORDER[offset+13]}"
  case "$kernel_capability" in
    rocmi4_dense_and_routed|rocmi4_dense_only)
      W4A8_ARGS=(-e DFLASH_ROCMI4_W4A8_IU4=1); w4a8_requested=true ;;
    no_eligible_rocmi4_mmq)
      W4A8_ARGS=(); w4a8_requested=false ;;
    *) die "slot $index has unknown candidate kernel capability" ;;
  esac
  for _ in $(seq 1 60); do
    available_kib="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
    (( available_kib >= 100 * 1024 * 1024 )) && break
    sleep 5
  done
  (( available_kib >= 100 * 1024 * 1024 )) || die "UMA did not drain before slot $index"
  CONTAINER="qwen-balanced-$PPID-$$-$index"
  docker run -d --name "$CONTAINER" --network host "${GPU_ARGS[@]}" \
    -v "$(dirname "$model"):/gate/model:ro" -e EMBER_IDLE_RECLAIM_SECS=0 \
    -v "$mtp:/gate/mtp.gguf:ro" -e DFLASH_QWEN_MTP=/gate/mtp.gguf \
    -e "DFLASH_QWEN_MTP_DEPTH=$mtp_depth" "${W4A8_ARGS[@]}" \
    --entrypoint "$BINARY" "$LOCAL_IMAGE_ID" \
    -m "/gate/model/$(basename "$model")" --host 127.0.0.1 --port "$PORT" \
    --max-ctx 8192 --prefix-cache-slots 1 >/dev/null
  host_pid="$(docker inspect --format '{{.State.Pid}}' "$CONTAINER")"
  container_id="$(docker inspect --format '{{.Id}}' "$CONTAINER")"
  [[ "$host_pid" =~ ^[0-9]+$ && "$host_pid" -gt 1 && "$container_id" =~ ^[0-9a-f]{64}$ ]] ||
    die "slot $index has no exact fresh process identity"
  start_ticks="$(awk '{print $22}' "/proc/$host_pid/stat")"
  process_file="$OUT_DIR/slot-$index-process.json"
  python3 - "$process_file" "$index" "$arm" "$candidate_id" "$container_id" \
    "$host_pid" "$start_ticks" "$RUNTIME_REVISION" "$IMAGE_DIGEST" \
    "$ENGINE_BINARY_SHA256" "$FORMAT_SHA256" "$candidate_binding_sha" "$model_sha" \
    "$model_inventory_sha" "$companion_inventory_sha" "$mtp_sha" "$mtp_depth" \
    "$kernel_capability" "$w4a8_requested" <<'PY'
import json,pathlib,sys
args=sys.argv[1:]
(output,index,arm,candidate,container,pid,start,revision,image,binary,format_sha,
 binding_sha,model_sha,model_inventory,companion_inventory,mtp_sha,mtp_depth,
 kernel_capability,w4a8_requested)=args
value={"schema":"ember.qwen3.8.fresh-server-process.v2","run_index":int(index),
       "arm_id":arm,"candidate_id":candidate,"container_id":container,
       "host_pid":int(pid),"proc_start_ticks":int(start),"ember_revision":revision,
       "container_digest":image,"engine_binary_sha256":binary,
       "tensor_format_contract_sha256":format_sha,
       "candidate_kernel_capability":kernel_capability,
       "rocmi4_w4a8_iu4_requested":w4a8_requested=="true",
       "candidate_binding_sha256":binding_sha,"model_first_shard_sha256":model_sha,
       "model_inventory_sha256":model_inventory,
       "companion_inventory_sha256":companion_inventory,
       "mtp_sha256":mtp_sha,"mtp_depth":int(mtp_depth)}
raw=json.dumps(value,sort_keys=True,separators=(",",":"))
pathlib.Path(output).write_text(raw,encoding="utf-8")
PY
  raw_slot_file="$OUT_DIR/slot-$index-raw.json"
  slot_file="$OUT_DIR/slot-$index.json"
  python3 "$SLOT" --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --health-endpoint "http://127.0.0.1:$PORT/health" --health-timeout 1800 \
    --run-index "$index" --arm-id "$arm" --process-identity "$process_file" \
    --workload-id "$workload_id" --workload-marker "$workload_marker" \
    --workload-recipe-sha256 "$workload_recipe_sha" \
    --server-pid "$host_pid" --timing-output "$OUT_DIR/slot-$index-timing.jsonl" \
    --output "$raw_slot_file"
  startup_log="$OUT_DIR/slot-$index-server.log"
  docker logs "$CONTAINER" >"$startup_log" 2>&1 || die "could not retain slot $index server log"
  python3 - "$raw_slot_file" "$startup_log" "$slot_file" \
    "$kernel_capability" <<'PY'
import hashlib,json,pathlib,re,sys
raw_path,log_path,output=map(pathlib.Path,sys.argv[1:4])
capability=sys.argv[4]
log=log_path.read_text(encoding="utf-8",errors="replace")
states=set(re.findall(
    r"ROCmI4 W4A8 IU4: exact experimental MMQ enabled for device [0-9]+; "
    r"activation_prepack=(on|off)",log))
if re.search(r"ROCmI4 W4A4: enabled for device [0-9]+",log):
    raise SystemExit("clean timing unexpectedly enabled lossy W4A4")
if capability == "no_eligible_rocmi4_mmq":
    if states:
        raise SystemExit("not-applicable finalist unexpectedly requested W4A8")
    mode="not_applicable_no_eligible_rocmi4_mmq"
elif capability in {"rocmi4_dense_and_routed","rocmi4_dense_only"}:
    if len(states)!=1:
        raise SystemExit(
            f"eligible finalist did not retain one W4A8 startup mode: {states}")
    mode={"on":"w4a8_iu4_prepack","off":"w4a8_iu4_register_pack"}[
        next(iter(states))]
else:
    raise SystemExit(f"unknown candidate kernel capability: {capability}")
value=json.loads(raw_path.read_text(encoding="utf-8"))
value["startup_kernel_mode"]=mode
value["startup_log_sha256"]=hashlib.sha256(log_path.read_bytes()).hexdigest()
output.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  test "$(sha256sum "$process_file" | awk '{print $1}')" = "$(python3 - "$slot_file" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding="utf-8"))["process_instance_sha256"])
PY
  )" || die "slot $index process-instance digest differs from retained identity"
  SLOT_ARGS+=(--slot "$slot_file")
  remove_container
done

python3 "$PREPARE" assemble --runner-plan "$RUNNER_PLAN" "${SLOT_ARGS[@]}" \
  --output "$OUT_DIR/balanced-confirmation.json"
restore_exclusive || die "failed to restore production or release the GPU lock"
log "PASS: six fresh-process timing slots recorded; no profiler or publisher ran"
