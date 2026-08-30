#!/usr/bin/env bash
# Provenance-bound target-only baseline for the Qwen3.8 sequential bakeoff.
#
# This deliberately does not grant release approval.  It measures the same
# exact 2074-token/three-sample protocol as the MTP hard gate, then collects
# profiler evidence in a separate pass. The matching-MTP path must still
# produce complete correctness/timing/memory evidence; the adjudicator, not
# this target-only baseline, applies the hard performance thresholds.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_LOCK=/usr/local/sbin/ember-gpu-lock
PRODUCTION=/usr/local/sbin/ember-cert-production
PRODUCTION_HEALTH=http://127.0.0.1:8000/health
PROFILE_SCRIPT="$REPO/scripts/profile_gpu.sh"
PROFILE_REPORT="$REPO/scripts/profile_report.py"
COUNTER_CALIBRATION="$REPO/share/benchmark/gfx1151-rocm10-counter-calibration.json"
BENCHMARK="$REPO/scripts/bench/benchmark.py"

IMAGE=""; IMAGE_DIGEST=""; PROFILE_IMAGE=""; PROFILE_IMAGE_DIGEST=""
MODEL=""; MODEL_SHA256=""; MODEL_BUILD_RECORD=""; MODEL_BUILD_RECORD_SHA256=""
BAKEOFF_MANIFEST=""; BAKEOFF_MANIFEST_SHA256=""; CANDIDATE_ID=""
OUT_DIR=""; BINARY=/usr/local/bin/ember-dflash; PORT=18085; DRY_RUN=0
CONTAINER=""; LOCK_HELD=0; MASKED=0; RESTORE_SERVICE=0
PRODUCTION_STATE_CAPTURED=0; PRODUCTION_WAS_ACTIVE=0

die() { printf 'qwen-target-only-gate: %s\n' "$*" >&2; exit 1; }
log() { printf 'qwen-target-only-gate: %s\n' "$*"; }

usage() {
  cat <<'EOF'
usage: scripts/qwen_target_only_gate.sh [options]

required:
  --image REF
  --image-digest sha256:HEX
  --profile-image REF
  --profile-image-digest sha256:HEX
  --model ABS_PATH
  --model-sha256 HEX
  --model-build-record ABS_PATH
  --model-build-record-sha256 HEX
  --bakeoff-manifest ABS_PATH
  --bakeoff-manifest-sha256 HEX
  --candidate-id ID
  --out-dir ABS_PATH

options:
  --binary ABS_PATH
  --port N                    default 18085
  --dry-run                   validate syntax and print a side-effect-free plan
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
    --bakeoff-manifest) BAKEOFF_MANIFEST="${2:?--bakeoff-manifest needs a path}"; shift 2 ;;
    --bakeoff-manifest-sha256) BAKEOFF_MANIFEST_SHA256="${2:?--bakeoff-manifest-sha256 needs a value}"; shift 2 ;;
    --candidate-id) CANDIDATE_ID="${2:?--candidate-id needs a value}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?--out-dir needs a path}"; shift 2 ;;
    --binary) BINARY="${2:?--binary needs a path}"; shift 2 ;;
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ -n "$IMAGE" && -n "$IMAGE_DIGEST" && -n "$PROFILE_IMAGE" &&
   -n "$PROFILE_IMAGE_DIGEST" && -n "$MODEL" && -n "$MODEL_SHA256" &&
   -n "$MODEL_BUILD_RECORD" && -n "$MODEL_BUILD_RECORD_SHA256" &&
   -n "$BAKEOFF_MANIFEST" && -n "$BAKEOFF_MANIFEST_SHA256" &&
   -n "$CANDIDATE_ID" && -n "$OUT_DIR" ]] ||
  die "all image/model/bakeoff paths and digests plus --candidate-id and --out-dir are required"
[[ "$IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid --image-digest"
[[ "$PROFILE_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid --profile-image-digest"
[[ "$MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid --model-sha256"
[[ "$MODEL_BUILD_RECORD_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid --model-build-record-sha256"
[[ "$BAKEOFF_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid --bakeoff-manifest-sha256"
[[ "$CANDIDATE_ID" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]] || die "invalid --candidate-id"
[[ "$MODEL" = /* && "$MODEL_BUILD_RECORD" = /* && "$BAKEOFF_MANIFEST" = /* &&
   "$OUT_DIR" = /* && "$BINARY" = /* ]] ||
  die "model, build record, bakeoff manifest, output, and in-image binary paths must be absolute"
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1024 && PORT <= 65535 )) ||
  die "--port must be 1024..65535"

cat <<EOF
plan:
  mode                target-only baseline (MTP disabled)
  candidate image     $IMAGE
  image digest        $IMAGE_DIGEST
  profiler image      $PROFILE_IMAGE
  profiler digest     $PROFILE_IMAGE_DIGEST
  target model        $MODEL
  target shard-1 sha  $MODEL_SHA256
  quant build record  $MODEL_BUILD_RECORD
  build record sha    $MODEL_BUILD_RECORD_SHA256
  bakeoff manifest    $BAKEOFF_MANIFEST
  bakeoff manifest sha $BAKEOFF_MANIFEST_SHA256
  candidate id        $CANDIDATE_ID
  evidence            $OUT_DIR
  timing              unprofiled 3x exact-2074 prefill and 3x decode-256
  thresholds reported prefill peak 412.0 tok/s; decode median 39.49 tok/s (not approval)
  profiling           later separate trace and one-counter-per-PMC passes
  publication         forbidden; no approval marker is written
EOF
if (( DRY_RUN )); then
  log "dry run: no files, GPU, docker, sudo, lock, or production state touched"
  exit 0
fi

for command in docker curl python3 dd; do command -v "$command" >/dev/null || die "$command is required"; done
[[ -x "$GPU_LOCK" && -x "$PRODUCTION" ]] || die "fixed-purpose host wrappers are missing"
[[ -x "$PROFILE_SCRIPT" && -f "$PROFILE_REPORT" &&
   -f "$COUNTER_CALIBRATION" && -f "$BENCHMARK" ]] ||
  die "measurement dependencies are missing"
[[ -f "$MODEL" && -f "$MODEL_BUILD_RECORD" && -f "$BAKEOFF_MANIFEST" ]] ||
  die "model, build record, or bakeoff manifest is missing"
[[ -r /dev/kfd && -d /dev/dri ]] || die "this gate must run on the gfx1151 host"
[[ ! -e "$OUT_DIR" && ! -L "$OUT_DIR" ]] || die "--out-dir must not already exist"
if grep -Eq '(^| )(iommu|amd_iommu)=off( |$)' /proc/cmdline; then die "IOMMU is disabled"; fi
if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  die "port $PORT is already serving"
fi

mkdir -p "$OUT_DIR"
python3 - "$BAKEOFF_MANIFEST" "$BAKEOFF_MANIFEST_SHA256" "$CANDIDATE_ID" \
  "$MODEL" "$MODEL_SHA256" "$MODEL_BUILD_RECORD" "$MODEL_BUILD_RECORD_SHA256" \
  "$OUT_DIR/candidate-binding.json" "$IMAGE" "$IMAGE_DIGEST" \
  "$PROFILE_IMAGE" "$PROFILE_IMAGE_DIGEST" <<'PY'
import hashlib, json, math, os, re, subprocess, sys
from pathlib import Path

(manifest_path, manifest_sha, candidate_id, model_path, model_sha,
 build_path, build_sha, output_path, image, image_digest,
 profile_image, profile_image_digest) = sys.argv[1:]
HEX = re.compile(r"[0-9a-f]{64}")

def fail(message):
    raise SystemExit(message)

def exact_file(path_value, expected_sha, expected_bytes=None, label="file"):
    path = Path(path_value)
    if not path.is_absolute(): fail(f"{label} path must be absolute")
    before = path.lstat()
    if path.is_symlink() or not path.is_file(): fail(f"{label} must be a regular non-symlink file")
    digest = hashlib.sha256()
    count = 0
    if before.st_size >= 512 * 1024 * 1024:
        process = subprocess.Popen(
            ["dd", f"if={path}", "iflag=direct", "bs=8M", "status=none"],
            stdout=subprocess.PIPE)
        assert process.stdout is not None
        for block in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
            digest.update(block); count += len(block)
        if process.wait() != 0: fail(f"{label} O_DIRECT hash failed")
    else:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block); count += len(block)
    after = path.lstat()
    if (count != before.st_size or
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        fail(f"{label} changed while hashed")
    if not isinstance(expected_sha, str) or HEX.fullmatch(expected_sha) is None or digest.hexdigest() != expected_sha:
        fail(f"{label} SHA-256 mismatch")
    if expected_bytes is not None and (expected_bytes != before.st_size or isinstance(expected_bytes, bool)):
        fail(f"{label} byte count mismatch")
    return {"path": str(path), "bytes": before.st_size, "sha256": expected_sha}

manifest_file = exact_file(manifest_path, manifest_sha, label="bakeoff manifest")
manifest = json.load(open(manifest_path, encoding="utf-8"))
if set(manifest) != {"schema", "ember_revision", "images", "release_profile", "bakeoff_plan", "candidates", "evidence_root"}:
    fail("bakeoff manifest keys differ from the v2 contract")
if manifest["schema"] != "ember.qwen3.8.sequential-bakeoff-candidates.v2":
    fail("unsupported bakeoff manifest schema")
if re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("ember_revision"))) is None:
    fail("bakeoff manifest Ember revision is malformed")
rows = [row for row in manifest["candidates"] if isinstance(row, dict) and row.get("id") == candidate_id]
if len(rows) != 1: fail("candidate id must occur exactly once")
row = rows[0]
required = {
    "id", "candidate_id", "stage", "configuration_id", "arm_id", "corpus_sha256",
    "quantization_arm", "model", "model_sha256", "build_record",
    "build_record_sha256", "intervention_configuration_id",
    "intervention_manifest", "intervention_manifest_sha256", "profile_sha256",
    "override_sha256", "companion_inventory", "companion_inventory_sha256",
    "mtp", "mtp_bytes", "mtp_sha256", "mtp_export_manifest",
    "mtp_export_manifest_sha256", "mtp_matrix_quant_contract", "vision_mmproj",
    "vision_mmproj_bytes", "vision_mmproj_sha256", "vision_mmproj_format",
    "vision_vocab", "vision_vocab_bytes", "vision_vocab_sha256",
    "vision_vocab_format", "vision_vocab_metadata_sha256",
    "quality_contract", "quality_contract_sha256", "mtp_depth",
    "runtime_mode", "image", "image_digest", "profile_image",
    "profile_image_digest", "final_release_eligible", "model_inventory_sha256",
    "tensor_format_compatibility_sha256", "artifact_bytes",
}
if set(row) != required: fail("candidate keys differ from the v2 contract")
plan_desc = manifest["bakeoff_plan"]
if not isinstance(plan_desc, dict) or set(plan_desc) != {"path", "sha256"}:
    fail("bakeoff plan descriptor is malformed")
exact_file(plan_desc["path"], plan_desc["sha256"], label="bakeoff plan")
plan = json.load(open(plan_desc["path"], encoding="utf-8"))
sweep_config = {item.get("id"): item for item in plan.get("sweep_configurations", [])}.get(row["id"])
is_stock = (row["id"] == "stock-rocmi4-exact" and row["stage"] == "stock"
            and row["quantization_arm"] in {"profile-default-rocmi4", "rocmi4-control"}
            and row["runtime_mode"] == "exact_dequant"
            and row["final_release_eligible"] is False)
is_w4a4_control = (row["stage"] == "format" and row["arm_id"] == "rocmi4-w4a4"
                    and row["quantization_arm"] == "rocmi4-q6k-embedding-head"
                    and row["runtime_mode"] == "w4a4_opt_in"
                    and row["final_release_eligible"] is False)
is_sweep_candidate = (row["stage"] == "sweep" and isinstance(sweep_config, dict)
                      and row["quantization_arm"] == sweep_config.get("quantization_arm")
                      and row["override_sha256"] == sweep_config.get("quantization_overrides_sha256")
                      and row["runtime_mode"] == sweep_config.get("runtime_mode")
                      and row["final_release_eligible"] is sweep_config.get("final_release_eligible") is False)
is_release_candidate = (row["stage"] in {"format", "mtp-depth", "final"}
                        and row["quantization_arm"] not in {"profile-default-rocmi4", "rocmi4-control"}
                        and row["runtime_mode"] == "exact_dequant"
                        and row["final_release_eligible"] is True)
if not (is_stock or is_sweep_candidate or is_w4a4_control or is_release_candidate):
    fail("only canonical stock/sweep/W4A4 or final-eligible format/final candidates are allowed")
if row["model"] != model_path or row["model_sha256"] != model_sha:
    fail("candidate model differs from gate arguments")
if row["build_record"] != build_path or row["build_record_sha256"] != build_sha:
    fail("candidate build record differs from gate arguments")
if (row["image"] != image or row["image_digest"] != image_digest
        or row["profile_image"] != profile_image
        or row["profile_image_digest"] != profile_image_digest):
    fail("candidate image pair differs from gate arguments")
for field in ("corpus_sha256", "profile_sha256", "override_sha256",
              "companion_inventory_sha256",
              "mtp_sha256", "mtp_export_manifest_sha256", "vision_mmproj_sha256",
              "vision_vocab_sha256", "vision_vocab_metadata_sha256",
              "quality_contract_sha256", "model_inventory_sha256",
              "tensor_format_compatibility_sha256"):
    if is_stock and field == "quality_contract_sha256":
        continue
    if not isinstance(row[field], str) or HEX.fullmatch(row[field]) is None:
        fail(f"candidate {field} is malformed")
if (not isinstance(row["candidate_id"], str) or not row["candidate_id"]
        or isinstance(row["artifact_bytes"], bool)
        or not isinstance(row["artifact_bytes"], int) or row["artifact_bytes"] < 1):
    fail("candidate artifact identity is malformed")
if not isinstance(row["mtp_depth"], int) or isinstance(row["mtp_depth"], bool) or not 1 <= row["mtp_depth"] <= 4:
    fail("candidate mtp_depth must be 1..4")

profile_desc = manifest["release_profile"]
if not isinstance(profile_desc, dict) or set(profile_desc) != {"path", "sha256"}:
    fail("release_profile descriptor is malformed")
profile_file = exact_file(profile_desc["path"], profile_desc["sha256"], label="release profile")
if row["profile_sha256"] != profile_desc["sha256"]: fail("candidate profile SHA differs")
profile = json.load(open(profile_desc["path"], encoding="utf-8"))
arms = {item.get("id"): item for item in profile["quantization"]["performance_bakeoff"]["arms"]}
arm = arms.get(row["quantization_arm"])
if not isinstance(arm, dict): fail("quantization arm is absent from the release profile")

build_file = exact_file(build_path, build_sha, label="quant build record")
build = json.load(open(build_path, encoding="utf-8"))
if build.get("status") != "complete" or build.get("mode") != "execute": fail("quant build record is not complete execute evidence")
if (build.get("tools") or {}).get("ember_revision") != manifest["ember_revision"]:
    fail("quant build record Ember revision differs from bakeoff manifest")
if (build.get("profile") or {}).get("sha256") != row["profile_sha256"]: fail("build-record profile SHA mismatch")
recipe = build.get("quantization_recipe") or {}
if (recipe.get("id") != row["quantization_arm"] or
        recipe.get("per_tensor_overrides_sha256") != row["override_sha256"] or
        recipe.get("ple_override_preserved") is not True):
    fail("build-record quantization recipe/override mismatch")
experiment = build.get("experiment") or {}
if is_stock:
    if (experiment.get("kind") != "stock_control" or experiment.get("stock_weights_unchanged") is not True
            or experiment.get("purpose") != "activation_capture_and_bakeoff_baseline"):
        fail("stock-control build-record experiment differs")
else:
    if (experiment.get("kind") != "directional_ablation" or
            experiment.get("stock_weights_unchanged") is not False or
            experiment.get("purpose") != "measured_bakeoff_candidate"):
        fail("intervention build-record experiment differs")
eligibility = experiment.get("eligibility_status")
if ((is_stock and eligibility != "ineligible_stock_control") or
        (not is_stock and eligibility != "pending_measured_bakeoff_and_hardware_certification")):
    fail("build-record experiment still has missing/pending artifact eligibility")
intervention = build.get("intervention") or {}
if is_stock:
    if (build.get("intervention") is not None or row["intervention_configuration_id"] is not None
            or row["intervention_manifest"] is not None
            or row["intervention_manifest_sha256"] != "0" * 64):
        fail("stock control must not claim intervention evidence")
    intervention_file = None
else:
    if (not isinstance(row["intervention_manifest_sha256"], str)
            or HEX.fullmatch(row["intervention_manifest_sha256"]) is None
            or intervention.get("manifest_sha256") != row["intervention_manifest_sha256"]
            or intervention.get("weight_intervention") is not True or intervention.get("prompt_only") is not False):
        fail("build-record intervention evidence mismatch")
    if row["intervention_configuration_id"] != row["configuration_id"]:
        fail("intervention configuration id differs from bakeoff configuration")
    intervention_file = exact_file(row["intervention_manifest"], row["intervention_manifest_sha256"], label="intervention manifest")

inventory_file = exact_file(row["companion_inventory"], row["companion_inventory_sha256"], label="companion inventory")
companions = build.get("companion_inventory") or {}
if ((companions.get("manifest") or {}).get("sha256") != row["companion_inventory_sha256"] or
        companions.get("status") != "verified_exact" or
        companions.get("fit_status") != "verified_exact_fit" or
        companions.get("release_companions_complete") is not True or
        companions.get("enabled_roles") != ["mtp", "vision_mmproj"] or
        companions.get("disabled_roles") != [] or
        companions.get("final_release_eligibility") != "pending_measured_bakeoff_and_hardware_certification"):
    fail("build-record companion inventory is incomplete or pending")
role_map = {item.get("role"): item for item in companions.get("roles") or []}
if set(role_map) != {"mtp", "vision_mmproj"}: fail("build-record companion roles are not exact")
mtp = role_map["mtp"]
mmproj = role_map["vision_mmproj"]
mtp_file = exact_file(row["mtp"], row["mtp_sha256"], row["mtp_bytes"], "MTP companion")
mtp_manifest_file = exact_file(row["mtp_export_manifest"], row["mtp_export_manifest_sha256"], label="MTP export manifest")
mmproj_file = exact_file(row["vision_mmproj"], row["vision_mmproj_sha256"], row["vision_mmproj_bytes"], "vision mmproj")
vocab_file = exact_file(row["vision_vocab"], row["vision_vocab_sha256"], row["vision_vocab_bytes"], "vision vocab")
if (mtp.get("path") != row["mtp"] or mtp.get("sha256") != row["mtp_sha256"] or
        mtp.get("size_bytes") != row["mtp_bytes"] or
        mtp.get("matrix_quant_contract") != row["mtp_matrix_quant_contract"] or
        (mtp.get("export_manifest") or {}).get("path") != row["mtp_export_manifest"] or
        (mtp.get("export_manifest") or {}).get("sha256") != row["mtp_export_manifest_sha256"]):
    fail("build-record MTP/export contract differs from candidate")
if ((mtp.get("export_manifest") or {}).get("quantizer_build_info") or {}).get("ember_revision") != manifest["ember_revision"]:
    fail("MTP export manifest Ember revision differs from bakeoff manifest")
if (mmproj.get("path") != row["vision_mmproj"] or mmproj.get("sha256") != row["vision_mmproj_sha256"] or
        mmproj.get("size_bytes") != row["vision_mmproj_bytes"] or mmproj.get("format") != row["vision_mmproj_format"] or
        row["vision_mmproj_format"] != "BF16"):
    fail("build-record vision mmproj contract differs from candidate")
vocab = mmproj.get("text_model") or {}
if (vocab.get("path") != row["vision_vocab"] or vocab.get("sha256") != row["vision_vocab_sha256"] or
        vocab.get("size_bytes") != row["vision_vocab_bytes"] or vocab.get("format") != row["vision_vocab_format"] or
        row["vision_vocab_format"] != "GGUF_VOCAB_ONLY" or
        (vocab.get("gguf_contract") or {}).get("metadata_sha256") != row["vision_vocab_metadata_sha256"]):
    fail("build-record vision vocab contract differs from candidate")
memory = build.get("memory_preflight") or {}
if (memory.get("combined_fits") is not True or memory.get("companion_artifact_fit_status") != "verified_exact_fit" or
        memory.get("enabled_companion_artifact_bytes") != row["mtp_bytes"] + row["vision_mmproj_bytes"] + row["vision_vocab_bytes"]):
    fail("combined main+MTP+mmproj+vision-vocab memory preflight is absent or inconsistent")

quality_file = None
if not is_stock:
    quality_file = exact_file(row["quality_contract"], row["quality_contract_sha256"], label="quality contract")

binding = {
    "schema": "ember.qwen3.8.candidate-binding.v2", "candidate": row,
    "bakeoff_manifest": manifest_file, "release_profile": profile_file,
    "quant_build_record": build_file, "intervention_manifest": intervention_file,
    "companion_inventory": inventory_file, "mtp": mtp_file,
    "mtp_export_manifest": mtp_manifest_file, "vision_mmproj": mmproj_file,
    "vision_vocab": vocab_file,
    "quality_contract": quality_file,
    "performance_scope": "text_only_pending_vision_differential",
}
with open(output_path, "x", encoding="utf-8") as stream:
    json.dump(binding, stream, indent=2, sort_keys=True); stream.write("\n")
PY
python3 - "$MODEL_BUILD_RECORD" "$MODEL_BUILD_RECORD_SHA256" "$MODEL" \
  "$MODEL_SHA256" "$OUT_DIR/model-inventory.json" "$BENCHMARK" <<'PY'
import importlib.util, json, sys
from pathlib import Path
record, record_sha, model, model_sha, output, module_path = sys.argv[1:]
spec = importlib.util.spec_from_file_location("ember_benchmark_gate", module_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load benchmark inventory verifier")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
inventory, _raw = module.model_inventory_from_build_record(
    Path(record), record_sha, Path(model), model_sha)
with open(output, "x", encoding="utf-8") as stream:
    json.dump(inventory, stream, indent=2, sort_keys=True); stream.write("\n")
PY

docker image inspect "$IMAGE" >"$OUT_DIR/image-inspect.json"
docker image inspect "$PROFILE_IMAGE" >"$OUT_DIR/profile-image-inspect.json"
python3 - "$OUT_DIR/image-inspect.json" "$IMAGE_DIGEST" \
  "$OUT_DIR/profile-image-inspect.json" "$PROFILE_IMAGE_DIGEST" <<'PY'
import json, sys
for label, path, expected in (("candidate", sys.argv[1], sys.argv[2]),
                              ("profiler", sys.argv[3], sys.argv[4])):
    rows = json.load(open(path, encoding="utf-8"))
    if len(rows) != 1:
        raise SystemExit(f"{label} image did not resolve exactly once")
    observed = {rows[0].get("Id")}
    observed.update(item.rsplit("@", 1)[-1] for item in rows[0].get("RepoDigests") or [])
    if expected not in observed:
        raise SystemExit(f"{label} image digest mismatch")
PY

# Fail without taking the machine lock when either exact image cannot perform
# its assigned role. The release image owns timing; the dev image owns only
# the later profiler/counter passes.
docker run --rm --entrypoint /bin/sh "$IMAGE" -c 'test -x "$1"' \
  qwen-target-only-candidate-preflight "$BINARY" ||
  die "candidate image lacks the requested binary"
docker run --rm --entrypoint /bin/sh "$PROFILE_IMAGE" -c '
  test -x "$1" && command -v rocprofv3 >/dev/null &&
  command -v rocprof-compute >/dev/null && command -v rocprofv3-avail >/dev/null
' qwen-target-only-profiler-preflight "$BINARY" ||
  die "profiler image lacks the matching binary or ROCm profiler tools"

candidate_revision="$(python3 - "$OUT_DIR/image-inspect.json" <<'PY'
import json, sys
row = json.load(open(sys.argv[1], encoding="utf-8"))[0]
print(((row.get("Config") or {}).get("Labels") or {}).get("org.opencontainers.image.revision", ""))
PY
)"
[[ "$candidate_revision" =~ ^[0-9a-f]{40}$ ]] || die "candidate image lacks a pinned source revision"
profile_revision="$(docker run --rm --entrypoint /bin/sh "$PROFILE_IMAGE" -c '
  awk -F= '\''$1 == "EMBER_CONFIGURED_GIT_HEAD:STRING" { print $2 }'\'' /ember/build-rocm/CMakeCache.txt
')"
[[ "$profile_revision" == "$candidate_revision" ]] || die "candidate/profiler source revisions differ"
candidate_binary_sha="$(docker run --rm --entrypoint sha256sum "$IMAGE" "$BINARY" | awk '{print $1}')"
profile_binary_sha="$(docker run --rm --entrypoint sha256sum "$PROFILE_IMAGE" "$BINARY" | awk '{print $1}')"
[[ "$candidate_binary_sha" =~ ^[0-9a-f]{64}$ && "$candidate_binary_sha" == "$profile_binary_sha" ]] ||
  die "candidate/profiler binaries are not byte-identical"

command -v stat >/dev/null || die "stat is required for numeric GPU device groups"
GPU_ARGS=(--device /dev/kfd --device /dev/dri
  --ipc host --security-opt seccomp=unconfined --ulimit memlock=-1:-1)
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
  if [[ -n "$CONTAINER" ]]; then docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; CONTAINER=""; fi
}
restore_exclusive() {
  local failed=0 attempt healthy=0
  remove_container
  if (( MASKED )); then
    for attempt in 1 2 3; do sudo -n "$PRODUCTION" unmask >/dev/null 2>&1 && { MASKED=0; break; }; sleep 1; done
    (( MASKED == 0 )) || failed=1
  fi
  if (( RESTORE_SERVICE )) && (( MASKED == 0 )); then
    for attempt in 1 2 3; do sudo -n "$PRODUCTION" start >/dev/null 2>&1 && { RESTORE_SERVICE=0; break; }; sleep 1; done
  fi
  (( RESTORE_SERVICE == 0 )) || failed=1
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
    for attempt in 1 2 3; do sudo -n "$GPU_LOCK" release >/dev/null 2>&1 && { LOCK_HELD=0; break; }; sleep 1; done
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

log "verifying every ordered target shard through O_DIRECT"
python3 - "$OUT_DIR/model-inventory.json" <<'PY'
import hashlib, json, subprocess, sys
for row in json.load(open(sys.argv[1], encoding="utf-8"))["shards"]:
    digest = hashlib.sha256()
    process = subprocess.Popen(["dd", f"if={row['path']}", "iflag=direct", "bs=8M", "status=none"], stdout=subprocess.PIPE)
    assert process.stdout is not None
    while chunk := process.stdout.read(8 * 1024 * 1024): digest.update(chunk)
    if process.wait() != 0 or digest.hexdigest() != row["sha256"]:
        raise SystemExit(f"O_DIRECT shard integrity failed: {row['path']}")
PY

for _ in $(seq 1 60); do
  available_kib="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
  (( available_kib >= 100 * 1024 * 1024 )) && break
  sleep 5
done
(( available_kib >= 100 * 1024 * 1024 )) || die "less than 100 GiB available"

CONTAINER="qwen-target-only-$PPID-$$"
docker run -d --name "$CONTAINER" --network host "${GPU_ARGS[@]}" \
  -v "$(dirname "$MODEL"):/gate/model:ro" -e EMBER_IDLE_RECLAIM_SECS=0 \
  --entrypoint "$BINARY" "$IMAGE" \
  -m "/gate/model/$(basename "$MODEL")" --host 127.0.0.1 --port "$PORT" --max-ctx 8192 >/dev/null
TIMING_HOST_PID="$(docker inspect --format '{{.State.Pid}}' "$CONTAINER")"
[[ "$TIMING_HOST_PID" =~ ^[0-9]+$ && "$TIMING_HOST_PID" -gt 1 ]] || die "invalid timing host PID"

python3 "$BENCHMARK" \
  --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
  --health-endpoint "http://127.0.0.1:$PORT/health" --health-timeout 1800 \
  --model qwen3.8-flash-next --output "$OUT_DIR/timing.jsonl" \
  --protocol hard-gate --prefill-target 412.0 --decode-target 39.49 \
  --server-pid "$TIMING_HOST_PID" --gtt-cap-bytes 133143986176 \
  --require-memory-gate
python3 - "$OUT_DIR/timing.jsonl" "$OUT_DIR/target-only-summary.json" \
  "$candidate_revision" "$candidate_binary_sha" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
metadata = [row for row in rows if row.get("kind") == "metadata"]
summaries = [row for row in rows if row.get("kind") == "summary"]
requests = [row for row in rows if row.get("kind") == "request" and row.get("ok")]
prefill = [row for row in requests if row.get("group") == "prefill-2048"]
decode = [row for row in requests if row.get("group") == "decode-256"]
if len(metadata) != 1 or len(summaries) != 1 or len(prefill) != 3 or len(decode) != 3:
    raise SystemExit("target-only protocol did not produce exactly three samples per shape")
if not all(row.get("spec_ran") is False for row in decode):
    raise SystemExit("target-only baseline did not explicitly report MTP disabled")
summary = summaries[0]
gate = summary.get("hard_gate") or {}
prefill_gate = gate.get("prefill_2048") or {}
if (prefill_gate.get("expected_evaluated_tokens") != 2074 or
        prefill_gate.get("evaluated_tokens") != [2074, 2074, 2074]):
    raise SystemExit("target-only prefill shape was not exactly 2074 evaluated tokens")
if prefill_gate.get("statistic") != "peak":
    raise SystemExit("target-only prefill observation did not use the required peak statistic")
if not (summary.get("memory_gate") or {}).get("passed"):
    raise SystemExit("target-only memory gate failed")
record = {"schema": "ember.qwen3.8.target-only-baseline.v1", "passed": True,
          "release_approval": False, "publishes": False,
          "ember_revision": sys.argv[3], "binary_sha256": sys.argv[4],
          "exact_prompt_tokens": 2074, "samples_per_shape": 3,
          "prefill_gate_statistic": "peak", "decode_gate_statistic": "median",
          "hard_gate_observation": summary.get("hard_gate"),
          "resources": summary.get("resources"), "memory_gate": summary.get("memory_gate")}
with open(sys.argv[2], "x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n")
PY
remove_container

for _ in $(seq 1 60); do
  available_kib="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
  (( available_kib >= 100 * 1024 * 1024 )) && break
  sleep 5
done
(( available_kib >= 100 * 1024 * 1024 )) || die "UMA did not drain before profiling"

log "running profiler passes separately from timing"
"$PROFILE_SCRIPT" --no-quiesce --image "$PROFILE_IMAGE" --binary "$BINARY" \
  --model "$MODEL" --port "$PORT" \
  --prefill-words 2048 --decode-tokens 256 --gap-secs 3 \
  --out-dir "$OUT_DIR/profile"
cp "$COUNTER_CALIBRATION" "$OUT_DIR/profile/counter-calibration.json"
python3 "$PROFILE_REPORT" "$OUT_DIR/profile" \
  --counter-calibration "$OUT_DIR/profile/counter-calibration.json" --json \
  >"$OUT_DIR/profile/report.json"

restore_exclusive || die "failed to restore production or release the GPU lock"
python3 - "$OUT_DIR" "$IMAGE" "$IMAGE_DIGEST" "$PROFILE_IMAGE" \
  "$PROFILE_IMAGE_DIGEST" "$candidate_revision" "$candidate_binary_sha" \
  "$MODEL" "$MODEL_SHA256" "$MODEL_BUILD_RECORD" \
  "$MODEL_BUILD_RECORD_SHA256" <<'PY'
import hashlib, json, os, sys
(out, image, image_digest, profile_image, profile_digest, revision, binary_sha,
 model, model_sha, build_record, build_record_sha) = sys.argv[1:]
def digest(relative):
    value = hashlib.sha256()
    with open(os.path.join(out, relative), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()
record = {
    "schema": "ember.qwen3.8.target-only-gate.v1", "passed": True,
    "release_approval": False, "publishes": False,
    "image": {"ref": image, "digest": image_digest},
    "profile_image": {"ref": profile_image, "digest": profile_digest,
                      "ember_revision": revision,
                      "candidate_binary_sha256": binary_sha,
                      "candidate_binary_byte_identical": True},
    "model": {"path": model, "first_shard_sha256": model_sha,
              "build_record_path": build_record,
              "build_record_sha256": build_record_sha},
    "protocol": {"exact_prefill_tokens": 2074, "samples_per_shape": 3,
                 "prefill_peak_threshold_observed": 412.0,
                 "decode_median_threshold_observed": 39.49,
                 "thresholds_are_release_gates": False},
    "performance_scope": "text_only_pending_vision_differential",
    "methodology": "clean target-only text timing and profiler/counter passes are separate; vision differential is pending",
    "evidence": {
        "timing": {"path": "timing.jsonl", "sha256": digest("timing.jsonl")},
        "summary": {"path": "target-only-summary.json",
                    "sha256": digest("target-only-summary.json")},
        "model_inventory": {"path": "model-inventory.json",
                            "sha256": digest("model-inventory.json")},
        "candidate_binding": {"path": "candidate-binding.json",
                              "sha256": digest("candidate-binding.json")},
        "profile": {"path": "profile/manifest.json",
                    "sha256": digest("profile/manifest.json")},
        "profile_report": {"path": "profile/report.json",
                    "sha256": digest("profile/report.json")},
        "counter_calibration": {"path": "profile/counter-calibration.json",
                    "sha256": digest("profile/counter-calibration.json")},
    },
}
with open(os.path.join(out, "complete.json"), "x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n")
PY
log "PASS: target-only baseline recorded; no release approval was granted"
