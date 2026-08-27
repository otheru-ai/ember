#!/usr/bin/env python3
"""Generate one fail-closed Qwen quality capture plan and phase descriptor.

This is a GPU-free handoff tool.  It verifies the already-unlocked phase,
completed stock/candidate build records, every GGUF shard, the judge inventory,
and immutable runtime images.  It does not start a runtime, open the new capture
root, measure, publish, or delete anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any

import qwen_bakeoff as bakeoff


ROOT = Path(__file__).resolve().parents[1]
RUBRIC = ROOT / "share" / "quant_eval" / "qwen3.8-quality-rubric.json"
AGENTIC_CASES = ROOT / "share" / "quant_eval" / "agentic_cases.jsonl"
RUBRIC_SHA256 = "ce26e1eb276bc9730aba2ee3ef5915eceeb8a81b699b609b0f0ec3ae972d6fb2"
AGENTIC_SHA256 = "079f73454951e0089094ff28df480cbf5d79616b921211f241a2dde74de1ec6a"
FINAL_SHA256 = "493ad8c9ed44fe635697ebdb27308ac0d3be80ec322ee28e1fd89c441b0df515"
SWEEP_SHA256 = "e50105eeff560e9cd6695f52ea240dbbfa64aa6288b51f33b4999422bb9fa451"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._:-]{0,127}")
JUDGE_SCHEMA = "ember.qwen3.8.quality-judge-inventory.v1"
PLAN_SCHEMA = "ember.qwen3.8.quality-capture-plan.v1"
DESCRIPTOR_SCHEMA = "ember.qwen3.8.quality-phase-descriptor.v1"


class DescriptorError(ValueError):
    pass


def fail(message: str) -> None:
    raise DescriptorError(message)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    if path.stat().st_size >= 512 * 1024 * 1024:
        try:
            process = subprocess.Popen(
                ["dd", f"if={path}", "iflag=direct", "bs=8M", "status=none"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            fail(f"cannot start direct-I/O hash for {path}: {exc}")
        assert process.stdout is not None
        for block in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
            digest.update(block)
        process.stdout.close()
        assert process.stderr is not None
        detail = process.stderr.read().decode("utf-8", "replace").strip()
        process.stderr.close()
        if process.wait() != 0:
            fail(f"direct-I/O hash failed for {path}: {detail}")
    else:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def exact_file(value: Any, expected: Any, label: str,
               expected_bytes: Any | None = None) -> Path:
    if (not isinstance(value, str) or not value.startswith("/") or "\0" in value
            or "\n" in value or HEX64.fullmatch(str(expected)) is None):
        fail(f"{label} descriptor is malformed")
    path = Path(value)
    try:
        before = path.lstat()
    except OSError as exc:
        fail(f"cannot stat {label}: {exc}")
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be an absolute regular non-symlink file")
    if expected_bytes is not None and (isinstance(expected_bytes, bool)
                                       or not isinstance(expected_bytes, int)
                                       or expected_bytes < 1
                                       or before.st_size != expected_bytes):
        fail(f"{label} byte count differs")
    try:
        digest = sha256_file(path)
        after = path.lstat()
    except OSError as exc:
        fail(f"cannot hash {label}: {exc}")
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                            row.st_mtime_ns, row.st_ctime_ns)
    if identity(before) != identity(after):
        fail(f"{label} changed while verified")
    if digest != expected:
        fail(f"{label} SHA-256 differs")
    return path


def exact_json(path: Path, expected: str, label: str) -> tuple[dict[str, Any], Path]:
    value = str(path)
    if (not path.is_absolute() or "\0" in value or "\n" in value
            or HEX64.fullmatch(str(expected)) is None):
        fail(f"{label} descriptor is malformed")
    try:
        before = path.lstat()
    except OSError as exc:
        fail(f"cannot stat {label}: {exc}")
    if not stat.S_ISREG(before.st_mode):
        fail(f"{label} must be an absolute regular non-symlink file")
    digest = hashlib.sha256()
    raw = bytearray()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(path, flags)
        try:
            opened = os.fstat(file_descriptor)
            for block in iter(lambda: os.read(file_descriptor, 1024 * 1024), b""):
                digest.update(block)
                raw.extend(block)
            read_complete = os.fstat(file_descriptor)
        finally:
            os.close(file_descriptor)
        after = path.lstat()
    except OSError as exc:
        fail(f"cannot read {label}: {exc}")
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                            row.st_mtime_ns, row.st_ctime_ns)
    if (len(raw) != before.st_size or identity(before) != identity(opened)
            or identity(opened) != identity(read_complete)
            or identity(read_complete) != identity(after)):
        fail(f"{label} changed while verified")
    if digest.hexdigest() != expected:
        fail(f"{label} SHA-256 differs")
    try:
        parsed = json.loads(bytes(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot parse {label}: {exc}")
    if not isinstance(parsed, dict):
        fail(f"{label} must be a JSON object")
    return parsed, path


def descriptor(path: Path, expected: str) -> dict[str, str]:
    return {"path": str(path), "sha256": expected}


def checked_descriptor(path: Path, expected: str, label: str) -> dict[str, str]:
    exact_file(str(path), expected, label)
    return descriptor(path, expected)


def output_path(path: Path, label: str) -> Path:
    if (not path.is_absolute() or path == Path("/") or path.exists()
            or not path.parent.is_dir() or path.parent.is_symlink()
            or path.resolve() != path):
        fail(f"{label} must be a new absolute path below a non-symlink directory")
    return path


def new_capture_root(path: Path) -> Path:
    if (not path.is_absolute() or path == Path("/") or path.exists()
            or not path.parent.is_dir() or path.parent.is_symlink()
            or path.resolve() != path):
        fail("quality output root must be a new absolute runner path")
    return path


def write_new(path: Path, value: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor_fd = os.open(path, flags, 0o600)
    except OSError as exc:
        fail(f"refusing {path}: {exc}")
    try:
        with os.fdopen(descriptor_fd, "wb", closefd=True) as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def image(value: str, label: str) -> str:
    if not isinstance(value, str) or IMAGE.fullmatch(value) is None:
        fail(f"{label} must be an immutable repository@sha256 image reference")
    return value


def plan_phase(raw: dict[str, Any], requested: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        plan = bakeoff.verify_plan(raw)
    except (bakeoff.BakeoffError, KeyError, OSError, TypeError, ValueError) as exc:
        fail(f"phase plan does not reproduce: {exc}")
    if requested == "sweep":
        if (plan.get("phase_scope") != "selection"
                or plan.get("status") != "planned_unmeasured"
                or "final_corpus_manifest" in plan):
            fail("sweep requires the final-blind selection plan")
        corpus = (plan.get("corpora") or {}).get("sweep-validation.jsonl")
        expected = SWEEP_SHA256
        selection = plan
    else:
        if (plan.get("phase_scope") != "final_confirmation"
                or plan.get("status") !=
                "final_heldout_unlocked_after_mtp_depth_selection"):
            fail("final requires the MTP-depth-unlocked final plan")
        corpus = (plan.get("corpora") or {}).get("final-heldout.jsonl")
        expected = FINAL_SHA256
        selection = plan.get("selection_plan")
    if (not isinstance(corpus, dict) or set(corpus) < {"path", "sha256"}
            or corpus.get("sha256") != expected or not isinstance(selection, dict)):
        fail("phase plan does not bind the pinned quality corpus")
    exact_file(corpus["path"], corpus["sha256"], f"{requested} quality corpus")
    return plan, selection


def ordered_shards(record: dict[str, Any], label: str) -> tuple[list[dict[str, Any]], str, int]:
    rows = (record.get("output") or {}).get("shards")
    if not isinstance(rows, list) or not rows:
        fail(f"{label} build record has no ordered output shards")
    result: list[dict[str, Any]] = []
    identity: list[dict[str, Any]] = []
    parents: set[Path] = set()
    seen: set[Path] = set()
    for index, row in enumerate(rows, 1):
        if (not isinstance(row, dict)
                or set(row) != {"path", "sha256", "size_bytes"}):
            fail(f"{label} shard {index} descriptor is malformed")
        path = exact_file(row["path"], row["sha256"], f"{label} shard {index}",
                          row["size_bytes"])
        if path in seen:
            fail(f"{label} repeats a shard path")
        seen.add(path)
        parents.add(path.parent.resolve())
        result.append({"path": str(path), "sha256": row["sha256"],
                       "bytes": row["size_bytes"]})
        identity.append({"index": index, "sha256": row["sha256"],
                         "bytes": row["size_bytes"]})
    if len(parents) != 1:
        fail(f"{label} shards must share one directory")
    return result, hashlib.sha256(canonical(identity)).hexdigest(), sum(
        row["bytes"] for row in result)


def model_from_build(record: dict[str, Any], record_path: Path, record_sha: str,
                     label: str, candidate_id: str, selection: dict[str, Any],
                     stock: bool) -> tuple[dict[str, Any], str, int]:
    if (record.get("status") != "complete" or record.get("mode") != "execute"
            or record.get("publishes") is not False
            or record.get("credentials_accessed") is not False
            or record.get("compute_mode") != "exact_dequant"
            or record.get("w4a4_enabled") is not False):
        fail(f"{label} build record is not complete nonpublishing exact-dequant evidence")
    if SAFE_ID.fullmatch(candidate_id) is None:
        fail(f"{label} candidate id is malformed")
    profile = record.get("profile")
    if profile != selection.get("release_profile") or not isinstance(profile, dict):
        fail(f"{label} build record uses a different release profile")
    profile_value, _ = exact_json(
        Path(str(profile.get("path"))), profile.get("sha256"), f"{label} release profile")
    recipe = record.get("quantization_recipe")
    if (not isinstance(recipe, dict) or not isinstance(recipe.get("id"), str)
            or HEX64.fullmatch(str(recipe.get("per_tensor_overrides_sha256"))) is None):
        fail(f"{label} quantization recipe is incomplete")
    experiment = record.get("experiment")
    expected_experiment = {
        "kind": "stock_control" if stock else "directional_ablation",
        "stock_weights_unchanged": stock,
        "final_release_eligible": False,
        "eligibility_status": ("ineligible_stock_control" if stock else
                               "pending_measured_bakeoff_and_hardware_certification"),
        "purpose": ("activation_capture_and_bakeoff_baseline" if stock else
                    "measured_bakeoff_candidate"),
    }
    if experiment != expected_experiment:
        fail(f"{label} experiment semantics differ")
    intervention = record.get("intervention")
    if stock:
        if intervention is not None:
            fail("stock build record unexpectedly carries an intervention")
        intervention_sha = "0" * 64
    else:
        intervention_sha = ((intervention or {}).get("manifest_sha256")
                            if isinstance(intervention, dict) else None)
        if HEX64.fullmatch(str(intervention_sha)) is None:
            fail("candidate build record lacks exact intervention evidence")
        manifest_name = ((profile_value.get("intervention") or {}).get("manifest_filename"))
        if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
            fail("release profile intervention filename is malformed")
        exact_file(str(record_path.parent / manifest_name), intervention_sha,
                   "candidate intervention manifest")
    shards, inventory_sha, artifact_bytes = ordered_shards(record, label)
    return ({"candidate_id": candidate_id, "build_record_sha256": record_sha,
             "intervention_manifest_sha256": intervention_sha,
             "profile_sha256": profile["sha256"],
             "quantization_overrides_sha256": recipe["per_tensor_overrides_sha256"],
             "shards": shards}, inventory_sha, artifact_bytes)


def candidate_phase_binding(phase: str, plan: dict[str, Any], selection: dict[str, Any],
                            record: dict[str, Any], record_sha: str,
                            model: dict[str, Any], inventory_sha: str,
                            artifact_bytes: int) -> None:
    recipe = record["quantization_recipe"]
    if phase == "sweep":
        rows = [*(selection.get("sweep_configurations") or []),
                *(selection.get("format_arms") or [])]
        matches = [row for row in rows if isinstance(row, dict)
                   and row.get("quantization_arm") == recipe.get("id")
                   and row.get("quantization_overrides_sha256") ==
                   recipe.get("per_tensor_overrides_sha256")
                   and ("mtp_matrix_quant_contract" not in row
                        or row.get("mtp_matrix_quant_contract") ==
                        recipe.get("selected_mtp_matrix_quant_contract"))]
        authorization = record.get("sweep_authorization")
        if isinstance(authorization, dict):
            matches = [row for row in matches
                       if row.get("id") == authorization.get("configuration_id")]
        if len(matches) != 1:
            fail("candidate build record does not identify one selection-plan row")
        return
    sealed = plan.get("sealed_recipe_ledger") or {}
    try:
        prior, _ = bakeoff.read_prior_ledger(sealed.get("descriptor"), "mtp-depth", selection)
    except (bakeoff.BakeoffError, KeyError, OSError, TypeError, ValueError) as exc:
        fail(f"final plan sealed ledger does not reproduce: {exc}")
    identity = prior.get("selected_artifact_identity")
    if not isinstance(identity, dict):
        fail("final plan does not expose the sealed artifact identity")
    expected = {
        "candidate_id": model["candidate_id"],
        "build_record_sha256": record_sha,
        "intervention_manifest_sha256": model["intervention_manifest_sha256"],
        "profile_sha256": model["profile_sha256"],
        "quantization_overrides_sha256": model["quantization_overrides_sha256"],
        "model_inventory_sha256": inventory_sha,
        "artifact_bytes": artifact_bytes,
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        fail("candidate build record differs from the final plan's sealed winner")


def judge_artifact(path: Path, expected_sha: str) -> dict[str, Any]:
    inventory, _ = exact_json(path, expected_sha, "judge inventory")
    if set(inventory) != {"schema", "artifact"} or inventory.get("schema") != JUDGE_SCHEMA:
        fail("judge inventory schema differs")
    artifact = inventory.get("artifact")
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256", "bytes"}:
        fail("judge inventory artifact descriptor differs")
    exact_file(artifact["path"], artifact["sha256"], "judge artifact", artifact["bytes"])
    return artifact


def generate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if HEX40.fullmatch(str(args.ember_revision)) is None:
        fail("Ember revision must be an exact lowercase 40-hex commit")
    capture_output = new_capture_root(args.quality_output_root)
    plan_output = output_path(args.capture_plan_output, "capture plan output")
    phase_output = output_path(args.output, "phase descriptor output")
    if len({capture_output, plan_output, phase_output}) != 3:
        fail("capture and descriptor outputs must be distinct")
    phase_raw, phase_path = exact_json(args.phase_plan, args.phase_plan_sha256,
                                       "unlocked phase plan")
    phase_plan, selection = plan_phase(phase_raw, args.phase)
    stock_record, stock_path = exact_json(
        args.stock_build_record, args.stock_build_record_sha256, "stock build record")
    candidate_record, candidate_path = exact_json(
        args.candidate_build_record, args.candidate_build_record_sha256,
        "candidate build record")
    if stock_path == candidate_path:
        fail("stock and candidate build records must differ")
    stock_id = (selection.get("stock_control") or {}).get("id")
    if not isinstance(stock_id, str):
        fail("selection plan stock-control id is malformed")
    stock, stock_inventory, _ = model_from_build(
        stock_record, stock_path, args.stock_build_record_sha256, "stock", stock_id,
        selection, True)
    candidate, candidate_inventory, candidate_bytes = model_from_build(
        candidate_record, candidate_path, args.candidate_build_record_sha256,
        "candidate", args.candidate_id, selection, False)
    if stock_inventory == candidate_inventory:
        fail("stock and candidate model inventories must differ")
    candidate_phase_binding(args.phase, phase_plan, selection, candidate_record,
                            args.candidate_build_record_sha256, candidate,
                            candidate_inventory, candidate_bytes)
    judge = judge_artifact(args.judge_inventory, args.judge_inventory_sha256)
    model_image = image(args.model_runtime_image, "model runtime image")
    judge_image = image(args.judge_runtime_image, "judge runtime image")
    corpus_name = ("sweep-validation.jsonl" if args.phase == "sweep"
                   else "final-heldout.jsonl")
    corpus = phase_plan["corpora"][corpus_name]
    corpus_desc = {"path": corpus["path"], "sha256": corpus["sha256"]}
    suffix = args.candidate_build_record_sha256[:12]
    contract_id = f"qwen3.8-{args.phase}-{suffix}"
    roles = ("stock", "candidate", "judge")
    artifacts = {"stock": stock, "candidate": candidate, "judge": {"shards": [judge]}}
    images = {"stock": model_image, "candidate": model_image, "judge": judge_image}
    ports = {"stock": 18080, "candidate": 18081, "judge": 18082}
    runs = {
        role: {
            "container": f"qwen-quality-{role}-{suffix}",
            "endpoint": f"http://127.0.0.1:{ports[role]}/v1/chat/completions",
            "image": images[role],
            "loaded_path": f"/models/{Path(artifacts[role]['shards'][0]['path']).name}",
        }
        for role in roles
    }
    capture_plan = {
        "schema": PLAN_SCHEMA,
        "contract_id": contract_id,
        "corpus": corpus_desc,
        "rubric": checked_descriptor(RUBRIC, RUBRIC_SHA256, "checked quality rubric"),
        "agentic_cases": checked_descriptor(
            AGENTIC_CASES, AGENTIC_SHA256, "checked agentic cases"),
        "models": {"stock": stock, "candidate": candidate},
        "judge": {"artifact": judge, "settings": {
            "temperature": 0, "seed": 7301, "batch_size": 1,
            "target_only": True, "speculative_decode": False,
            "required_tool": "submit_verdict",
        }},
        "runs": runs,
    }
    write_new(plan_output, capture_plan)
    plan_sha = sha256_file(plan_output)
    phase_descriptor = {
        "schema": DESCRIPTOR_SCHEMA,
        "ember_revision": args.ember_revision,
        "phase": args.phase,
        "unlocked_plan": descriptor(phase_path, args.phase_plan_sha256),
        "capture_plan": descriptor(plan_output, plan_sha),
        "output_dir": str(capture_output),
        "launches": {role: {"extra_arguments": [],
                             "startup_timeout_seconds": 1200} for role in roles},
        "release_scope": {"modality": "text_only",
                          "multimodal_release_claim": False,
                          "vision_mmproj_differential_pass": False},
    }
    try:
        write_new(phase_output, phase_descriptor)
    except Exception:
        plan_output.unlink(missing_ok=True)
        raise
    return capture_plan, phase_descriptor


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase", choices=("sweep", "final"), required=True)
    result.add_argument("--phase-plan", type=Path, required=True)
    result.add_argument("--phase-plan-sha256", required=True)
    result.add_argument("--stock-build-record", type=Path, required=True)
    result.add_argument("--stock-build-record-sha256", required=True)
    result.add_argument("--candidate-build-record", type=Path, required=True)
    result.add_argument("--candidate-build-record-sha256", required=True)
    result.add_argument("--candidate-id", required=True)
    result.add_argument("--judge-inventory", type=Path, required=True)
    result.add_argument("--judge-inventory-sha256", required=True)
    result.add_argument("--ember-revision", required=True)
    result.add_argument("--model-runtime-image", required=True)
    result.add_argument("--judge-runtime-image", required=True)
    result.add_argument("--quality-output-root", type=Path, required=True)
    result.add_argument("--capture-plan-output", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        capture, phase = generate(args)
    except (DescriptorError, OSError, ValueError) as exc:
        print(f"qwen-quality-descriptor: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "capture_plan": {"path": str(args.capture_plan_output),
                         "sha256": sha256_file(args.capture_plan_output)},
        "phase_descriptor": {"path": str(args.output),
                             "sha256": sha256_file(args.output)},
        "contract_id": capture["contract_id"], "phase": phase["phase"],
        "quality_output_root": phase["output_dir"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
