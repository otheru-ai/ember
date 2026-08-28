#!/usr/bin/env python3
"""Prepare and assemble fail-closed Qwen format-finalist confirmation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_bakeoff as bakeoff  # noqa: E402


PLAN_SCHEMA = "ember.qwen3.8.balanced-confirmation-runner-plan.v1"
HEX64 = re.compile(r"[0-9a-f]{64}")


class ConfirmationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file(value: Any, expected: Any, label: str) -> Path:
    if (not isinstance(value, str) or not value.startswith("/") or "\n" in value
            or not isinstance(expected, str) or HEX64.fullmatch(expected) is None):
        raise ConfirmationError(f"{label} descriptor is malformed")
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise ConfirmationError(f"{label} is not a regular non-symlink file")
    if sha256_file(path) != expected:
        raise ConfirmationError(f"{label} SHA-256 differs")
    return path.resolve()


def declared_regular_file(value: Any, expected: Any, label: str) -> Path:
    """Bind a large artifact without polluting UMA before production quiesces."""
    if (not isinstance(value, str) or not value.startswith("/") or "\n" in value
            or not isinstance(expected, str) or HEX64.fullmatch(expected) is None):
        raise ConfirmationError(f"{label} descriptor is malformed")
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise ConfirmationError(f"{label} is not a regular non-symlink file")
    return path.resolve()


def exact_json(value: Any, expected: Any, label: str) -> tuple[dict[str, Any], Path]:
    path = exact_file(value, expected, label)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfirmationError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfirmationError(f"{label} must contain one JSON object")
    return parsed, path


def beneath(path: Path, root: Path, label: str) -> None:
    if path != root and root not in path.parents:
        raise ConfirmationError(f"{label} escapes the declared evidence root")


def _binding_from_assessment(row: dict[str, Any], evidence_root: Path) -> dict[str, Any]:
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", row_id) is None:
        raise ConfirmationError("format finalist row id is unsafe")
    measurement_path = evidence_root / f"format-{row_id}-measurement.evidence.json"
    measurement, measurement_path = exact_json(
        str(measurement_path), row.get("measurement_manifest_sha256"),
        "format finalist measurement manifest")
    beneath(measurement_path, evidence_root, "format finalist measurement manifest")
    if (measurement.get("schema") != bakeoff.RESULT_SCHEMA
            or measurement.get("candidate_id") !=
            (row.get("artifact_identity") or {}).get("candidate_id")):
        raise ConfirmationError("format finalist measurement identity differs")

    target_desc = (measurement.get("evidence") or {}).get("target_complete")
    if not isinstance(target_desc, dict) or set(target_desc) != {"path", "sha256"}:
        raise ConfirmationError("format finalist target completion descriptor differs")
    target, target_path = exact_json(
        target_desc.get("path"), target_desc.get("sha256"),
        "format finalist target completion")
    beneath(target_path, evidence_root, "format finalist target completion")
    if (target.get("schema") != "ember.qwen3.8.target-only-gate.v1"
            or target.get("passed") is not True or target.get("publishes") is not False):
        raise ConfirmationError("format finalist target completion did not pass")

    binding_desc = (target.get("evidence") or {}).get("candidate_binding")
    if not isinstance(binding_desc, dict) or set(binding_desc) != {"path", "sha256"}:
        raise ConfirmationError("format finalist candidate-binding descriptor differs")
    binding_path = (target_path.parent / str(binding_desc["path"])).resolve()
    binding, binding_path = exact_json(
        str(binding_path), binding_desc.get("sha256"),
        "format finalist candidate binding")
    beneath(binding_path, evidence_root, "format finalist candidate binding")
    if binding.get("schema") != "ember.qwen3.8.candidate-binding.v2":
        raise ConfirmationError("format finalist candidate-binding schema differs")
    candidate = binding.get("candidate")
    artifact = row.get("artifact_identity")
    if not isinstance(candidate, dict) or not isinstance(artifact, dict):
        raise ConfirmationError("format finalist lacks candidate/artifact identity")
    comparisons = {
        "candidate_id": "candidate_id",
        "build_record_sha256": "build_record_sha256",
        "intervention_manifest_sha256": "intervention_manifest_sha256",
        "profile_sha256": "profile_sha256",
        "override_sha256": "quantization_overrides_sha256",
        "quantization_arm": "quantization_arm",
        "model_inventory_sha256": "model_inventory_sha256",
        "companion_inventory_sha256": "companion_inventory_sha256",
        "mtp_matrix_quant_contract": "mtp_matrix_quant_contract",
        "mtp_depth": "mtp_depth",
        "intervention_configuration_id": "intervention_configuration_id",
        "tensor_format_compatibility_sha256": "tensor_format_compatibility_sha256",
        "artifact_bytes": "artifact_bytes",
    }
    companion_bytes = {"mtp": candidate.get("mtp_bytes"),
                       "vision_mmproj": candidate.get("vision_mmproj_bytes")}
    if (candidate.get("stage") != "format"
            or candidate.get("final_release_eligible") is not True
            or candidate.get("arm_id") != row.get("arm_id")
            or companion_bytes != artifact.get("companion_artifact_bytes")
            or any(candidate.get(candidate_key) != artifact.get(artifact_key)
                   for candidate_key, artifact_key in comparisons.items())):
        raise ConfirmationError("format finalist candidate binding differs from assessment")
    model = declared_regular_file(candidate.get("model"), candidate.get("model_sha256"),
                                  "format finalist first shard")
    mtp = declared_regular_file(candidate.get("mtp"), candidate.get("mtp_sha256"),
                                "format finalist MTP companion")
    build = exact_file(candidate.get("build_record"), candidate.get("build_record_sha256"),
                       "format finalist build record")
    return {
        "arm_id": row["arm_id"],
        "candidate_id": candidate["candidate_id"],
        "quantization_arm": candidate["quantization_arm"],
        "candidate_kernel_capability": bakeoff.candidate_kernel_capability(
            candidate["quantization_arm"]),
        "model": str(model),
        "model_sha256": candidate["model_sha256"],
        "model_inventory_sha256": candidate["model_inventory_sha256"],
        "companion_inventory_sha256": candidate["companion_inventory_sha256"],
        "mtp": str(mtp),
        "mtp_sha256": candidate["mtp_sha256"],
        "mtp_depth": candidate["mtp_depth"],
        "build_record": str(build),
        "build_record_sha256": candidate["build_record_sha256"],
        "measurement_manifest": {"path": str(measurement_path),
                                 "sha256": row["measurement_manifest_sha256"]},
        "target_completion": {"path": str(target_path), "sha256": target_desc["sha256"]},
        "candidate_binding": {"path": str(binding_path), "sha256": binding_desc["sha256"]},
    }


def prepare(args: argparse.Namespace) -> None:
    plan, _plan_path = exact_json(args.plan, args.plan_sha256, "selection plan")
    accumulator, _accumulator_path = exact_json(
        args.accumulator, args.accumulator_sha256, "format assessment accumulator")
    if (accumulator.get("schema") != "ember.qwen3.8.sequential-bakeoff-accumulator.v2"
            or accumulator.get("phase") != "format"
            or accumulator.get("contains_raw_measurements") is not False
            or accumulator.get("publication_allowed") is not False):
        raise ConfirmationError("format assessment accumulator contract differs")
    verified = bakeoff.verify_plan(plan)
    rows, _digests = bakeoff.assessment_rows(accumulator, {"format"}, verified)
    eligible = [row for row in rows if row.get("final_release_eligible") is True]
    confirmation_plan = bakeoff.balanced_confirmation_plan(verified, eligible)
    evidence_root = Path(args.evidence_root).resolve()
    if (not evidence_root.is_dir() or evidence_root.is_symlink()
            or not evidence_root.is_absolute()):
        raise ConfirmationError("evidence root must be an existing absolute non-symlink directory")
    by_arm = {row.get("arm_id"): row for row in eligible}
    bindings = [_binding_from_assessment(by_arm[item["arm_id"]], evidence_root)
                for item in confirmation_plan["finalists"]]
    value = {
        "schema": PLAN_SCHEMA,
        "confirmation_plan": confirmation_plan,
        "bindings": bindings,
        "runtime_identity": {
            "ember_revision": args.runtime_revision,
            "engine_binary_sha256": args.engine_binary_sha256,
            "container_digest": args.container_digest,
            "tensor_format_contract_sha256": args.tensor_format_contract_sha256,
        },
        "publishes": False,
        "profiles_during_timing": False,
    }
    expected_runtime = value["runtime_identity"]
    if any(row.get("runtime_identity") != expected_runtime for row in rows):
        raise ConfirmationError("format assessment runtime identity differs from confirmation runner")
    output = Path(args.output)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ConfirmationError("runner plan output must be a new absolute path")
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assemble(args: argparse.Namespace) -> None:
    runner = json.loads(Path(args.runner_plan).read_text(encoding="utf-8"))
    if runner.get("schema") != PLAN_SCHEMA:
        raise ConfirmationError("runner plan schema differs")
    slots = []
    for raw in args.slot:
        value = json.loads(Path(raw).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ConfirmationError("slot evidence must be an object")
        slots.append(value)
    expected = runner["confirmation_plan"]
    if len(slots) != 6:
        raise ConfirmationError("balanced confirmation requires exactly six slot files")
    process_ids: set[str] = set()
    container_ids: set[str] = set()
    host_process_ids: set[tuple[int, int]] = set()
    arm_counts: dict[str, int] = {}
    bindings = {item["arm_id"]: item for item in runner["bindings"]}
    runtime = runner["runtime_identity"]
    workload_runs: dict[str, list[dict[str, Any]]] = {
        item["workload_id"]: [] for item in expected["workloads"]}
    eligible_kernel_modes: set[str] = set()
    for index, (slot, arm_id) in enumerate(zip(slots, expected["run_order"], strict=True)):
        if (slot.get("run_index") != index or slot.get("arm_id") != arm_id
                or set(slot) != {"run_index", "arm_id", "process_instance",
                                 "process_instance_sha256", "workload_id",
                                 "workload_recipe_sha256", "prefill_prompt_sha256",
                                 "decode_prompt_sha256", "calibrated_prefill_words",
                                 "evaluated_prefill_tokens", "completion_tokens",
                                 "prefill_tps", "decode_tps", "spec_ran", "accept_rate",
                                 "startup_kernel_mode", "startup_log_sha256"}):
            raise ConfirmationError("slot evidence differs from persisted run order/schema")
        process = slot.get("process_instance")
        process_id = slot.get("process_instance_sha256")
        binding = bindings.get(arm_id) or {}
        capability = binding.get("candidate_kernel_capability")
        expected_finalist = next(
            (item for item in expected["finalists"] if item["arm_id"] == arm_id), {})
        if capability != expected_finalist.get("candidate_kernel_capability"):
            raise ConfirmationError(
                "runner binding changed the candidate timing-kernel capability")
        w4a8_required = capability != "no_eligible_rocmi4_mmq"
        if (not isinstance(process, dict) or set(process) != {
                "schema", "run_index", "arm_id", "candidate_id", "container_id",
                "host_pid", "proc_start_ticks", "ember_revision", "container_digest",
                "engine_binary_sha256", "tensor_format_contract_sha256",
                "candidate_kernel_capability", "rocmi4_w4a8_iu4_requested",
                "candidate_binding_sha256", "model_first_shard_sha256",
                "model_inventory_sha256", "companion_inventory_sha256",
                "mtp_sha256", "mtp_depth"}
                or process.get("schema") != bakeoff.BALANCED_PROCESS_SCHEMA
                or process.get("run_index") != index or process.get("arm_id") != arm_id
                or process.get("candidate_id") != binding.get("candidate_id")
                or HEX64.fullmatch(str(process.get("container_id", ""))) is None
                or isinstance(process.get("host_pid"), bool)
                or not isinstance(process.get("host_pid"), int)
                or process["host_pid"] <= 1
                or isinstance(process.get("proc_start_ticks"), bool)
                or not isinstance(process.get("proc_start_ticks"), int)
                or process["proc_start_ticks"] <= 0
                or process.get("ember_revision") != runtime.get("ember_revision")
                or process.get("container_digest") != runtime.get("container_digest")
                or process.get("engine_binary_sha256") != runtime.get("engine_binary_sha256")
                or process.get("tensor_format_contract_sha256") !=
                   runtime.get("tensor_format_contract_sha256")
                or process.get("candidate_kernel_capability") != capability
                or process.get("rocmi4_w4a8_iu4_requested") is not w4a8_required
                or process.get("candidate_binding_sha256") !=
                   (binding.get("candidate_binding") or {}).get("sha256")
                or process.get("model_first_shard_sha256") != binding.get("model_sha256")
                or process.get("model_inventory_sha256") != binding.get("model_inventory_sha256")
                or process.get("companion_inventory_sha256") !=
                   binding.get("companion_inventory_sha256")
                or process.get("mtp_sha256") != binding.get("mtp_sha256")
                or process.get("mtp_depth") != binding.get("mtp_depth")):
            raise ConfirmationError("slot process identity differs from candidate/image/binary/MTP")
        numbers = (slot.get("prefill_tps"), slot.get("decode_tps"), slot.get("accept_rate"))
        if (not isinstance(process_id, str) or HEX64.fullmatch(process_id) is None
                or process_id != bakeoff.canonical_sha256(process)
                or process_id in process_ids
                or process["container_id"] in container_ids
                or (process["host_pid"], process["proc_start_ticks"]) in host_process_ids):
            raise ConfirmationError("every slot must bind one unique fresh process")
        if (slot.get("evaluated_prefill_tokens") != 2074
                or slot.get("completion_tokens") != 256
                or slot.get("spec_ran") is not True
                or (w4a8_required and slot.get("startup_kernel_mode") not in {
                    "w4a8_iu4_register_pack", "w4a8_iu4_prepack"})
                or (not w4a8_required and slot.get("startup_kernel_mode") !=
                    "not_applicable_no_eligible_rocmi4_mmq")
                or HEX64.fullmatch(str(slot.get("startup_log_sha256", ""))) is None
                or not 0.0 < float(slot.get("accept_rate", 0.0)) < 1.0
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(float(value)) or float(value) <= 0
                       for value in numbers)):
            raise ConfirmationError("slot workload/throughput evidence differs")
        workload = next((item for item in expected["workloads"]
                         if item["workload_id"] == slot.get("workload_id")), None)
        if (workload is None or slot["workload_id"] != expected["workload_order"][index]
                or slot.get("workload_recipe_sha256") != workload["recipe_sha256"]
                or any(HEX64.fullmatch(str(slot.get(key, ""))) is None
                       for key in ("prefill_prompt_sha256", "decode_prompt_sha256"))
                or isinstance(slot.get("calibrated_prefill_words"), bool)
                or not isinstance(slot.get("calibrated_prefill_words"), int)
                or slot["calibrated_prefill_words"] < 1):
            raise ConfirmationError("slot does not bind its sealed paired workload")
        workload_runs[slot["workload_id"]].append(slot)
        if w4a8_required:
            eligible_kernel_modes.add(slot["startup_kernel_mode"])
        process_ids.add(process_id)
        container_ids.add(process["container_id"])
        host_process_ids.add((process["host_pid"], process["proc_start_ticks"]))
        arm_counts[arm_id] = arm_counts.get(arm_id, 0) + 1
    if sorted(arm_counts.values()) != [3, 3]:
        raise ConfirmationError("balanced confirmation requires exactly three slots per arm")
    if len(eligible_kernel_modes) > 1:
        raise ConfirmationError("balanced confirmation changed W4A8 startup kernel mode")
    for workload_id, pair in workload_runs.items():
        if (len(pair) != 2 or {item["arm_id"] for item in pair} != set(bindings)
                or pair[0]["prefill_prompt_sha256"] != pair[1]["prefill_prompt_sha256"]
                or pair[0]["decode_prompt_sha256"] != pair[1]["decode_prompt_sha256"]
                or pair[0]["calibrated_prefill_words"] != pair[1]["calibrated_prefill_words"]):
            raise ConfirmationError(
                f"paired workload {workload_id} differs between finalists")
    value = {"schema": bakeoff.BALANCED_CONFIRMATION_SCHEMA,
             "confirmation_plan": expected, "runs": slots}
    output = Path(args.output)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ConfirmationError("confirmation output must be a new absolute path")
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--plan", required=True)
    prep.add_argument("--plan-sha256", required=True)
    prep.add_argument("--accumulator", required=True)
    prep.add_argument("--accumulator-sha256", required=True)
    prep.add_argument("--evidence-root", required=True)
    prep.add_argument("--runtime-revision", required=True)
    prep.add_argument("--engine-binary-sha256", required=True)
    prep.add_argument("--container-digest", required=True)
    prep.add_argument("--tensor-format-contract-sha256", required=True)
    prep.add_argument("--output", required=True)
    prep.set_defaults(function=prepare)
    assembly = commands.add_parser("assemble")
    assembly.add_argument("--runner-plan", required=True)
    assembly.add_argument("--slot", action="append", required=True)
    assembly.add_argument("--output", required=True)
    assembly.set_defaults(function=assemble)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        args.function(args)
    except (ConfirmationError, bakeoff.BakeoffError, KeyError, OSError,
            TypeError, ValueError) as exc:
        print(f"qwen_balanced_confirmation.py: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
