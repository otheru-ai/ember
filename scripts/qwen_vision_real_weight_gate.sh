#!/usr/bin/env bash
# Exclusive-gfx1151 Qwen image-text differential and cold/warm UMA gate.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_LOCK=/usr/local/sbin/ember-gpu-lock
PRODUCTION=/usr/local/sbin/ember-cert-production
PRODUCTION_HEALTH=http://127.0.0.1:8000/health
CORPUS="$REPO/share/quant_eval/qwen3.8-vision-differential-v2.json"
COMPARE="$REPO/scripts/qwen_vision_differential.py"
SAMPLER="$REPO/scripts/qwen_vision_residency.py"
IMAGE=""; DEV_IMAGE=""; MODEL=""; MODEL_SHA256=""
BUILD_RECORD=""; BUILD_RECORD_SHA256=""; MTP=""; MTP_SHA256=""
MMPROJ=""; MMPROJ_SHA256=""; VISION_VOCAB=""; VISION_VOCAB_SHA256=""
OUT_DIR=""; MTP_DEPTH=4; PORT=18089; DRY_RUN=0
INTEGRITY_CACHE=/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/.artifact-integrity-v1.json
CONTAINER=""; LOCK_HELD=0; MASKED=0; RESTORE_SERVICE=0
PRODUCTION_STATE_CAPTURED=0; PRODUCTION_WAS_ACTIVE=0; SAMPLER_PID=""

die() { printf 'qwen-vision-real-weight-gate: %s\n' "$*" >&2; exit 1; }
usage() {
  cat <<'EOF'
usage: scripts/qwen_vision_real_weight_gate.sh [options]
  --image REPOSITORY@sha256:HEX       exact release image
  --dev-image REPOSITORY@sha256:HEX   exact matching dev image
  --model ABS_PATH --model-sha256 HEX
  --model-build-record ABS_PATH --model-build-record-sha256 HEX
  --mtp ABS_PATH --mtp-sha256 HEX [--mtp-depth 1..4]
  --mmproj ABS_PATH --mmproj-sha256 HEX
  --vision-vocab ABS_PATH --vision-vocab-sha256 HEX
  --out-dir ABS_NEW_PATH [--port N] [--dry-run]
  --integrity-cache ABS_PATH          persistent identity-bound verification cache
EOF
}
while (( $# )); do
  case "$1" in
    --image) IMAGE="$2"; shift 2;; --dev-image) DEV_IMAGE="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;; --model-sha256) MODEL_SHA256="$2"; shift 2;;
    --model-build-record) BUILD_RECORD="$2"; shift 2;;
    --model-build-record-sha256) BUILD_RECORD_SHA256="$2"; shift 2;;
    --mtp) MTP="$2"; shift 2;; --mtp-sha256) MTP_SHA256="$2"; shift 2;;
    --mtp-depth) MTP_DEPTH="$2"; shift 2;;
    --mmproj) MMPROJ="$2"; shift 2;; --mmproj-sha256) MMPROJ_SHA256="$2"; shift 2;;
    --vision-vocab) VISION_VOCAB="$2"; shift 2;;
    --vision-vocab-sha256) VISION_VOCAB_SHA256="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;; --port) PORT="$2"; shift 2;;
    --integrity-cache) INTEGRITY_CACHE="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;; -h|--help) usage; exit 0;; *) die "unknown option $1";;
  esac
done
[[ "$IMAGE" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ &&
   "$DEV_IMAGE" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]] ||
  die "release and dev images must be immutable repository@sha256 references"
for digest in "$MODEL_SHA256" "$BUILD_RECORD_SHA256" "$MTP_SHA256" "$MMPROJ_SHA256" "$VISION_VOCAB_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || die "all artifact digests must be lowercase SHA-256"
done
for path in "$MODEL" "$BUILD_RECORD" "$MTP" "$MMPROJ" "$VISION_VOCAB" \
            "$OUT_DIR" "$INTEGRITY_CACHE"; do
  [[ "$path" = /* ]] || die "all artifact/evidence paths must be absolute"
done
[[ "$MTP_DEPTH" =~ ^[1-4]$ ]] || die "MTP depth must be 1..4"
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1024 && PORT <= 65535 )) || die "unsafe port"
[[ -d "$(dirname "$OUT_DIR")" && ! -L "$(dirname "$OUT_DIR")" ]] ||
  die "evidence parent must be an existing non-symlink directory"
corpus_sha="$(sha256sum "$CORPUS" | awk '{print $1}')"
cat <<EOF
plan:
  runtime          $IMAGE
  reference        $DEV_IMAGE / llama.cpp abdc7a0bf815d3b83e26dd523c6960e4dd597e82
  model            $MODEL
  MTP              $MTP (depth $MTP_DEPTH)
  BF16 mmproj      $MMPROJ
  vocab-only GGUF  $VISION_VOCAB
  integrity cache  $INTEGRITY_CACHE
  corpus           $CORPUS ($corpus_sha)
  differential     two images, cold/warm float32 atol=1e-5 rtol=1e-5, image-grounded answers
  residency        phase-separated host RSS, amdgpu GTT, and UMA
  production       stop+mask; unconditional unmask/restore and health proof
  publication      forbidden; evidence is consumed by the protected envelope
EOF
if (( DRY_RUN )); then
  echo 'qwen-vision-real-weight-gate: dry run touches no GPU, production, Docker, or output files'
  exit 0
fi

for tool in docker curl python3 sha256sum dd stat sed; do command -v "$tool" >/dev/null || die "$tool is required"; done
[[ -x "$GPU_LOCK" && -x "$PRODUCTION" && -f "$COMPARE" && -f "$SAMPLER" ]] || die "gate dependencies missing"
[[ -f "$MODEL" && ! -L "$MODEL" && -f "$BUILD_RECORD" && ! -L "$BUILD_RECORD" &&
   -f "$MTP" && ! -L "$MTP" && -f "$MMPROJ" && ! -L "$MMPROJ" ]] || die "artifact missing or symlinked"
[[ -f "$VISION_VOCAB" && ! -L "$VISION_VOCAB" ]] || die "vision vocab is missing or symlinked"
[[ -r /dev/kfd && -d /dev/dri ]] || die "gfx1151 device is unavailable"
[[ ! -e "$OUT_DIR" ]] || die "output directory already exists"
curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && die "port already in use"

mkdir -m 700 "$OUT_DIR"
python3 "$COMPARE" materialize --corpus "$CORPUS" --corpus-sha256 "$corpus_sha" \
  --output-dir "$OUT_DIR/corpus"
python3 - "$BUILD_RECORD" "$BUILD_RECORD_SHA256" "$MODEL" "$MODEL_SHA256" \
  "$OUT_DIR/model-inventory.json" "$REPO/scripts/bench/benchmark.py" <<'PY'
import importlib.util,json,sys
from pathlib import Path
record,record_sha,model,model_sha,out,module_path=sys.argv[1:]
spec=importlib.util.spec_from_file_location("vision_inventory",module_path)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
value,_raw=module.model_inventory_from_build_record(
    Path(record),record_sha,Path(model),model_sha)
Path(out).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
PY
docker pull "$IMAGE"; docker pull "$DEV_IMAGE"
docker image inspect "$IMAGE" >"$OUT_DIR/image-inspect.json"
docker image inspect "$DEV_IMAGE" >"$OUT_DIR/dev-image-inspect.json"
runtime_revision="$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMAGE")"
dev_revision="$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$DEV_IMAGE")"
[[ "$runtime_revision" =~ ^[0-9a-f]{40}$ && "$dev_revision" = "$runtime_revision" ]] || die "image revisions differ"
runtime_binary_sha="$(docker run --rm --entrypoint sha256sum "$IMAGE" /usr/local/bin/ember-dflash | awk '{print $1}')"
dev_binary_sha="$(docker run --rm --entrypoint sha256sum "$DEV_IMAGE" /usr/local/bin/ember-dflash | awk '{print $1}')"
[[ "$runtime_binary_sha" = "$dev_binary_sha" && "$runtime_binary_sha" =~ ^[0-9a-f]{64}$ ]] || die "runtime/dev binaries differ"
provider=/opt/qwen-vision-provider/lib/libember_qwen4exp_vision_provider.so
provider_sha="$(docker run --rm --entrypoint sha256sum "$IMAGE" "$provider" | awk '{print $1}')"
dev_provider_sha="$(docker run --rm --entrypoint sha256sum "$DEV_IMAGE" "$provider" | awk '{print $1}')"
[[ "$provider_sha" = "$dev_provider_sha" && "$provider_sha" =~ ^[0-9a-f]{64}$ ]] ||
  die "runtime/dev vision providers differ"
runtime_llama_revision="$(docker run --rm --entrypoint cat "$IMAGE" /opt/qwen-vision-provider/LLAMA_CPP_REVISION)"
dev_llama_revision="$(docker run --rm --entrypoint cat "$DEV_IMAGE" /opt/qwen-vision-provider/LLAMA_CPP_REVISION)"
[[ "$runtime_llama_revision" = abdc7a0bf815d3b83e26dd523c6960e4dd597e82 &&
   "$dev_llama_revision" = "$runtime_llama_revision" ]] || die "runtime/dev llama.cpp revisions differ"
runtime_rocm_full="$(docker run --rm --entrypoint cat "$IMAGE" /opt/rocm/.info/version)"
dev_rocm_full="$(docker run --rm --entrypoint cat "$DEV_IMAGE" /opt/rocm/.info/version)"
[[ "$runtime_rocm_full" = "$dev_rocm_full" && "$runtime_rocm_full" =~ ^10\.0\.0([.-]|$) ]] ||
  die "runtime/dev ROCm versions differ from 10.0.0"
rocm_version="$(printf '%s' "$runtime_rocm_full" | sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/')"
runner_uid="$(id -u)"; runner_gid="$(id -g)"
docker run --rm --user "$runner_uid:$runner_gid" -e CCACHE_DISABLE=1 \
  -v "$OUT_DIR:/gate/evidence" --entrypoint /bin/bash "$DEV_IMAGE" -c '
  set -euo pipefail
  mtmd_lib="$(readlink -f /opt/qwen-vision-provider/lib/libmtmd.so)"
  llama_lib="$(readlink -f /opt/qwen-vision-provider/lib/libllama.so)"
  test -f "$mtmd_lib" -a -f "$llama_lib"
  c++ -std=c++17 -O2 -Wall -Wextra -Werror -Wshadow -Wconversion -Wsign-conversion \
    -isystem /opt/qwen-vision-provider/include \
    /ember/tools/qwen4exp_vision_reference.cpp \
    -L/opt/qwen-vision-provider/lib -Wl,-rpath,/opt/qwen-vision-provider/lib \
    "$mtmd_lib" "$llama_lib" -o /gate/evidence/qwen4exp-vision-reference
'
reference_sha="$(sha256sum "$OUT_DIR/qwen4exp-vision-reference" | awk '{print $1}')"

declare -A GPU_GIDS=()
for node in /dev/kfd /dev/dri/*; do [[ -c "$node" ]] || continue; GPU_GIDS["$(stat -c %g "$node")"]=1; done
GPU_ARGS=(--device /dev/kfd --device /dev/dri --ipc host --security-opt seccomp=unconfined --ulimit memlock=-1:-1)
for gid in "${!GPU_GIDS[@]}"; do GPU_ARGS+=(--group-add "$gid"); done

remove_container() { [[ -z "$CONTAINER" ]] || { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; CONTAINER=""; }; }
restore_exclusive() {
  local failed=0 healthy=0
  if [[ -n "$SAMPLER_PID" ]]; then touch "$OUT_DIR/stop-sampler"; wait "$SAMPLER_PID" || failed=1; SAMPLER_PID=""; fi
  remove_container
  if (( MASKED )); then sudo -n "$PRODUCTION" unmask >/dev/null 2>&1 && MASKED=0 || failed=1; fi
  if (( RESTORE_SERVICE && ! MASKED )); then sudo -n "$PRODUCTION" start >/dev/null 2>&1 && RESTORE_SERVICE=0 || failed=1; fi
  if (( PRODUCTION_STATE_CAPTURED && PRODUCTION_WAS_ACTIVE && ! MASKED && ! RESTORE_SERVICE )); then
    for _ in $(seq 1 300); do
      if sudo -n "$PRODUCTION" is-active >/dev/null 2>&1 && curl --fail --silent --max-time 2 "$PRODUCTION_HEALTH" >/dev/null 2>&1; then healthy=1; break; fi
      sleep 2
    done
    (( healthy )) || failed=1
  fi
  if (( LOCK_HELD )); then sudo -n "$GPU_LOCK" release >/dev/null 2>&1 && LOCK_HELD=0 || failed=1; fi
  return "$failed"
}
cleanup() { rc=$?; trap - EXIT INT TERM; restore_exclusive || rc=1; exit "$rc"; }
trap cleanup EXIT INT TERM

sudo -n "$GPU_LOCK" acquire; LOCK_HELD=1
if sudo -n "$PRODUCTION" is-active >/dev/null 2>&1; then PRODUCTION_WAS_ACTIVE=1; RESTORE_SERVICE=1; sudo -n "$PRODUCTION" stop; fi
PRODUCTION_STATE_CAPTURED=1; sudo -n "$PRODUCTION" mask; MASKED=1

actual_gpu_arch="$(docker run --rm "${GPU_ARGS[@]}" --entrypoint /opt/rocm/bin/rocminfo \
  "$IMAGE" | awk '$1 == "Name:" && $2 ~ /^gfx/ {print $2}' | sort -u)"
[[ "$actual_gpu_arch" = gfx1151 ]] || die "runtime rocminfo did not measure exactly gfx1151"

PYTHONPATH="$REPO/scripts" python3 - "$OUT_DIR/model-inventory.json" \
 "$MTP" "$MTP_SHA256" "$MMPROJ" "$MMPROJ_SHA256" \
 "$VISION_VOCAB" "$VISION_VOCAB_SHA256" "$INTEGRITY_CACHE" <<'PY'
import json,pathlib,sys
from qwen_integrity_cache import IntegrityCache
inventory=json.load(open(sys.argv[1]))
rows=[(r["path"],r["sha256"]) for r in inventory["shards"]]+[(sys.argv[2],sys.argv[3]),(sys.argv[4],sys.argv[5]),(sys.argv[6],sys.argv[7])]
with IntegrityCache(pathlib.Path(sys.argv[8])) as cache:
 for path,wanted in rows: cache.verify(pathlib.Path(path),wanted)
PY
vision_vocab_metadata_sha="$(python3 - "$VISION_VOCAB" "$REPO/scripts/qwen_quantize.py" <<'PY'
import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("vision_vocab_contract",sys.argv[2])
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
print(module.validate_qwen_vocab_only_gguf(Path(sys.argv[1]))["metadata_sha256"])
PY
)"
[[ "$vision_vocab_metadata_sha" =~ ^[0-9a-f]{64}$ ]] || die "vision vocab metadata identity is malformed"
available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_kib >= 100 * 1024 * 1024 )) || die "less than 100 GiB available"

CONTAINER="qwen-vision-gate-$$"
docker run -d --name "$CONTAINER" --network host --user "$runner_uid:$runner_gid" "${GPU_ARGS[@]}" \
  -v "$(dirname "$MODEL"):/gate/model:ro" -v "$MTP:/gate/mtp.gguf:ro" \
  -v "$MMPROJ:/gate/mmproj.gguf:ro" -v "$VISION_VOCAB:/gate/vision-vocab.gguf:ro" \
  -v "$OUT_DIR:/gate/evidence" \
  -e DFLASH_QWEN_MTP=/gate/mtp.gguf -e "DFLASH_QWEN_MTP_DEPTH=$MTP_DEPTH" \
  -e DFLASH_QWEN_VISION_MMPROJ=/gate/mmproj.gguf -e DFLASH_QWEN_VISION_PROVIDER="$provider" \
  -e DFLASH_QWEN_VISION_TEXT_MODEL=/gate/vision-vocab.gguf \
  -e DFLASH_QWEN_VISION_CAPTURE_PREFIX=/gate/evidence/ember \
  --entrypoint /usr/local/bin/ember-dflash "$IMAGE" \
  -m "/gate/model/$(basename "$MODEL")" --host 127.0.0.1 --port "$PORT" --max-ctx 8192 >/dev/null
host_pid="$(docker inspect --format '{{.State.Pid}}' "$CONTAINER")"
for _ in $(seq 1 1800); do curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null || die "vision server did not become healthy"
printf idle >"$OUT_DIR/phase"
python3 "$SAMPLER" --pid "$host_pid" --phase-file "$OUT_DIR/phase" \
  --stop-file "$OUT_DIR/stop-sampler" --output "$OUT_DIR/residency.json" --interval 0.05 &
SAMPLER_PID=$!
sleep 0.25
mkdir "$OUT_DIR/responses" "$OUT_DIR/reference"
for run in cold warm; do
  printf '%s' "$run" >"$OUT_DIR/phase"; sleep 0.15
  for request in "$OUT_DIR"/corpus/*.request.json; do
    id="$(basename "$request" .request.json)"
    curl --fail-with-body --silent --show-error --max-time 1800 \
      -H 'Content-Type: application/json' --data-binary "@$request" \
      "http://127.0.0.1:$PORT/v1/chat/completions" >"$OUT_DIR/responses/$run-$id.json"
  done
  sleep 0.15
done
printf settled >"$OUT_DIR/phase"; sleep 0.30
touch "$OUT_DIR/stop-sampler"; wait "$SAMPLER_PID"; SAMPLER_PID=""
docker logs "$CONTAINER" >"$OUT_DIR/server.log" 2>&1
remove_container

for image_path in "$OUT_DIR"/corpus/*.png; do
  id="$(basename "$image_path" .png)"
  docker run --rm --user "$runner_uid:$runner_gid" "${GPU_ARGS[@]}" \
    -v "$VISION_VOCAB:/gate/vision-vocab.gguf:ro" -v "$MMPROJ:/gate/mmproj.gguf:ro" \
    -v "$image_path:/gate/image.png:ro" -v "$OUT_DIR:/gate/evidence" \
    --entrypoint /gate/evidence/qwen4exp-vision-reference "$DEV_IMAGE" \
    /gate/vision-vocab.gguf /gate/mmproj.gguf /gate/image.png "/gate/evidence/reference/$id.f32"
done

inventory_sha="$(sha256sum "$OUT_DIR/model-inventory.json" | awk '{print $1}')"
provider_path="$provider"
python3 - "$OUT_DIR/identity.json" "$IMAGE" "$runtime_revision" "$runtime_binary_sha" \
 "$OUT_DIR/model-inventory.json" "$inventory_sha" "$BUILD_RECORD" "$BUILD_RECORD_SHA256" \
 "$MTP" "$MTP_SHA256" "$MTP_DEPTH" \
 "$MMPROJ" "$MMPROJ_SHA256" "$provider_path" "$provider_sha" \
 "$OUT_DIR/qwen4exp-vision-reference" "$reference_sha" "$DEV_IMAGE" "$CORPUS" "$corpus_sha" \
 "$actual_gpu_arch" "$rocm_version" "$runtime_llama_revision" "$dev_llama_revision" \
 "$VISION_VOCAB" "$VISION_VOCAB_SHA256" "$vision_vocab_metadata_sha" <<'PY'
import json,pathlib,sys
(out,image,rev,binary,model,modelsha,build_record,build_sha,mtp,mtpsha,depth,mmproj,mmsha,provider,providersha,
 reference,refsha,dev_image,corpus,corpussha,gpu_arch,rocm_version,runtime_llama,dev_llama,
 vision_vocab,vision_vocab_sha,vision_vocab_metadata_sha)=sys.argv[1:]
inventory=json.loads(pathlib.Path(model).read_text())
ordered=[{"index":i,"sha256":row["sha256"],"bytes":row["size_bytes"]}
         for i,row in enumerate(inventory["shards"],1)]
import hashlib
canonical=(json.dumps(ordered,sort_keys=True,separators=(",",":"))+"\n").encode()
value={"schema":"ember.qwen3.8.vision-runtime-identity.v1",
 "runtime":{"image":image,"ember_revision":rev,"engine_binary_sha256":binary},
 "model":{"path":str(pathlib.Path(model).resolve()),"sha256":modelsha,
          "model_inventory_sha256":hashlib.sha256(canonical).hexdigest(),
          "first_shard_path":inventory["shards"][0]["path"],
          "first_shard_sha256":inventory["shards"][0]["sha256"],
          "build_record_path":build_record,"build_record_sha256":build_sha},
 "mtp":{"path":mtp,"sha256":mtpsha,"size_bytes":pathlib.Path(mtp).stat().st_size,
        "depth":int(depth)},
 "vision_mmproj":{"path":mmproj,"sha256":mmsha,
                  "size_bytes":pathlib.Path(mmproj).stat().st_size,"format":"BF16"},
 "vision_vocab":{"path":vision_vocab,"sha256":vision_vocab_sha,
                  "size_bytes":pathlib.Path(vision_vocab).stat().st_size,
                  "format":"GGUF_VOCAB_ONLY","metadata_sha256":vision_vocab_metadata_sha},
 "provider":{"path":provider,"sha256":providersha,"abi_version":1,
             "llama_cpp_revision":runtime_llama},
 "reference":{"path":str(pathlib.Path(reference).resolve()),"sha256":refsha,
              "image":dev_image,
              "llama_cpp_revision":dev_llama},
 "hardware":{"gpu_arch":gpu_arch,"rocm_version":rocm_version},
 "corpus":{"path":str(pathlib.Path(corpus).resolve()),"sha256":corpussha}}
pathlib.Path(out).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
PY
identity_sha="$(sha256sum "$OUT_DIR/identity.json" | awk '{print $1}')"
residency_sha="$(sha256sum "$OUT_DIR/residency.json" | awk '{print $1}')"

restore_exclusive || die "production restore or GPU lock release failed"
python3 "$COMPARE" finalize --corpus "$CORPUS" --corpus-sha256 "$corpus_sha" \
  --identity "$OUT_DIR/identity.json" --identity-sha256 "$identity_sha" \
  --residency "$OUT_DIR/residency.json" --residency-sha256 "$residency_sha" \
  --ember-dir "$OUT_DIR" --reference-dir "$OUT_DIR/reference" \
  --response-dir "$OUT_DIR/responses" --output "$OUT_DIR/vision-certified.json"
ln "$OUT_DIR/vision-certified.json" "$OUT_DIR/complete.json"
echo "qwen-vision-real-weight-gate: PASS $(sha256sum "$OUT_DIR/vision-certified.json" | awk '{print $1}')"
