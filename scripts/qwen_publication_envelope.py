#!/usr/bin/env python3
"""Build and verify the fail-closed Qwen Hugging Face publication handoff.

The package generator remains deliberately non-publishing.  This standard-
library tool binds that package to the final externally-attested bakeoff
ledger, the full measurement/quality/hardware evidence, and the exact runtime
image.  It never contacts Hugging Face and never reads a credential.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any

import qwen_bakeoff as bakeoff


SCHEMA = "ember.qwen3.8.hf-publication-envelope.v1"
RECEIPT_SCHEMA = "ember.qwen3.8.hf-candidate-verification.v1"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
IMAGE = re.compile(r"([^\s@]+)@sha256:([0-9a-f]{64})")
REVISION = re.compile(r"candidate/[A-Za-z0-9._-]+")
SAFE_RELEASE_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
EXPECTED_REPO = "otheru/Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-GGUF"
EXPECTED_ATTESTATION_REPOSITORY = "OtherU-AI/ember"
# `gh attestation verify --signer-workflow` requires the fully qualified
# owner/repository/workflow path, not merely the repository-relative filename.
EXPECTED_ATTESTATION_WORKFLOW = (
    "OtherU-AI/ember/.github/workflows/qwen-gfx1151-bakeoff.yml"
)


class EnvelopeError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def inspect_stable(path: Path, expected: str, label: str,
                   retain: bool) -> tuple[bytes | None, int]:
    if (not path.is_absolute() or HEX64.fullmatch(str(expected)) is None
            or "\0" in str(path) or "\n" in str(path)):
        raise EnvelopeError(f"{label} descriptor is malformed")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        named = path.lstat()
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            digest = hashlib.sha256()
            raw = bytearray() if retain else None
            byte_count = 0
            for block in iter(lambda: os.read(descriptor, 8 * 1024 * 1024), b""):
                digest.update(block)
                byte_count += len(block)
                if raw is not None:
                    raw.extend(block)
            complete = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = path.lstat()
    except OSError as exc:
        raise EnvelopeError(f"cannot read {label}: {exc}") from exc
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                            row.st_mtime_ns, row.st_ctime_ns)
    if (not stat.S_ISREG(named.st_mode) or identity(named) != identity(opened)
            or identity(opened) != identity(complete) or identity(complete) != identity(after)
            or byte_count != named.st_size):
        raise EnvelopeError(f"{label} is not one stable regular file")
    if digest.hexdigest() != expected:
        raise EnvelopeError(f"{label} SHA-256 differs")
    return (bytes(raw) if raw is not None else None), byte_count


def read_stable(path: Path, expected: str, label: str) -> bytes:
    raw, _size = inspect_stable(path, expected, label, True)
    assert raw is not None
    return raw


def verify_stable(path: Path, expected: str, label: str) -> int:
    _raw, size = inspect_stable(path, expected, label, False)
    return size


def read_json(path: Path, expected: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(read_stable(path, expected, label))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvelopeError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise EnvelopeError(f"{label} must be a JSON object")
    return value


def file_descriptor(path: Path, expected: str, label: str) -> dict[str, Any]:
    raw = read_stable(path, expected, label)
    return {"path": str(path), "sha256": expected, "size_bytes": len(raw)}


def exact_descriptor(value: Any, label: str) -> tuple[Path, str]:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
            or not isinstance(value.get("path"), str)
            or HEX64.fullmatch(str(value.get("sha256"))) is None):
        raise EnvelopeError(f"{label} descriptor must contain only path and sha256")
    return Path(value["path"]).resolve(), value["sha256"]


def validate_upload_plan(path: Path, expected: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = read_json(path, expected, "upload plan")
    required = {"schema_version", "action", "publishes", "repo_id", "repo_type",
                "revision", "files", "authentication", "promotion",
                "publication_blockers"}
    if (set(plan) != required or plan.get("schema_version") != 1
            or plan.get("action") != "candidate_upload_plan"
            or plan.get("publishes") is not False
            or plan.get("repo_id") != EXPECTED_REPO or plan.get("repo_type") != "model"
            or REVISION.fullmatch(str(plan.get("revision"))) is None
            or plan.get("publication_blockers") != []):
        raise EnvelopeError("upload plan is not the exact unblocked Qwen candidate plan")
    if (plan.get("authentication") != {
            "preferred": "Hugging Face Trusted Publisher via GitHub OIDC",
            "fallback": "fine-grained HF_TOKEN with write access only to the target model repository",
            "token_embedded": False}):
        raise EnvelopeError("upload plan authentication policy differs")
    if plan.get("promotion") != {
            "allowed": False,
            "requires": "all documented release gates and exact candidate commit verification"}:
        raise EnvelopeError("upload plan must prohibit promotion")
    rows = plan.get("files")
    if not isinstance(rows, list) or not rows:
        raise EnvelopeError("upload plan has no files")
    checked: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    seen_destinations: set[str] = set()
    package_parents: set[Path] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {
                "local_path", "path_in_repo", "size_bytes", "sha256"}:
            raise EnvelopeError(f"upload-plan file {index} has unexpected fields")
        local = Path(str(row.get("local_path")))
        destination = row.get("path_in_repo")
        size = row.get("size_bytes")
        digest = row.get("sha256")
        if (not local.is_absolute() or not isinstance(destination, str)
                or Path(destination).name != destination or destination in {"", ".", ".."}
                or isinstance(size, bool) or not isinstance(size, int) or size < 1
                or HEX64.fullmatch(str(digest)) is None):
            raise EnvelopeError(f"upload-plan file {index} is malformed")
        local = local.absolute()
        if local in seen_paths or destination in seen_destinations:
            raise EnvelopeError("upload plan repeats a local or destination path")
        actual_size = verify_stable(local, digest, f"planned file {destination}")
        if actual_size != size:
            raise EnvelopeError(f"planned file {destination} byte count differs")
        seen_paths.add(local); seen_destinations.add(destination)
        package_parents.add(local.parent.resolve())
        checked.append({"local_path": str(local), "path_in_repo": destination,
                        "size_bytes": size, "sha256": digest})
    if len(package_parents) != 1 or path.resolve().parent not in package_parents:
        raise EnvelopeError("all planned files and upload-plan.json must share one package directory")
    required_names = {"README.md", "LICENSE", "artifact-manifest.json", "SHA256SUMS",
                      "release-profile.json", "qwen-quant-build-record.json",
                      "qwen-intervention-manifest.json"}
    if not required_names.issubset(seen_destinations):
        raise EnvelopeError("upload plan omits required release evidence")
    return plan, checked


def planned_json(files: list[dict[str, Any]], name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = [row for row in files if row["path_in_repo"] == name]
    if len(matches) != 1:
        raise EnvelopeError(f"package must contain exactly one {name}")
    row = matches[0]
    return read_json(Path(row["local_path"]), row["sha256"], name), row


def planned_file(files: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [row for row in files if row["path_in_repo"] == name]
    if len(matches) != 1:
        raise EnvelopeError(f"package must contain exactly one {name}")
    return matches[0]


def reconstruct_measurement(assessment: dict[str, Any], manifest_path: Path,
                            manifest_sha: str, quality_path: Path,
                            quality_sha: str) -> dict[str, Any]:
    identity = assessment.get("artifact_identity") or {}
    inputs = assessment.get("decision_inputs") or {}
    runtime = assessment.get("runtime_identity")
    row = {
        "id": assessment.get("row_id"), "candidate_id": identity.get("candidate_id"),
        "stage": "final", "status": "complete", "measurement_kind": "measured",
        "final_release_eligible": assessment.get("final_release_eligible"),
        "configuration_id": assessment.get("configuration_id"),
        "quantization_arm": identity.get("quantization_arm"),
        "quantization_overrides_sha256": identity.get("quantization_overrides_sha256"),
        "intervention_configuration_id": identity.get("intervention_configuration_id"),
        "intervention_manifest_sha256": identity.get("intervention_manifest_sha256"),
        "build_record_sha256": identity.get("build_record_sha256"),
        "companion_inventory_sha256": identity.get("companion_inventory_sha256"),
        "model_inventory_sha256": identity.get("model_inventory_sha256"),
        "profile_sha256": identity.get("profile_sha256"),
        "builder_identity": identity.get("builder_identity"),
        "runtime_identity": runtime,
        "tensor_format_compatibility_sha256": identity.get(
            "tensor_format_compatibility_sha256"),
        "mtp_matrix_quant_contract": assessment.get("mtp_matrix_quant_contract"),
        "mtp_depth": assessment.get("mtp_depth"),
        "corpus_sha256": assessment.get("corpus_sha256"),
        "evidence_manifest": {"path": str(manifest_path), "sha256": manifest_sha,
                              "schema": bakeoff.RESULT_SCHEMA},
        "quality_contract": {"path": str(quality_path), "sha256": quality_sha},
        "enabled_companions": inputs.get("enabled_companions"),
    }
    for key in ("prefill_tps_samples", "decode_tps_samples", "evaluated_prefill_tokens",
                "completion_tokens", "mtp_spec_ran", "artifact_bytes",
                "companion_artifact_bytes"):
        row[key] = inputs.get(key)
    resources = inputs.get("resources") or {}
    for key in ("runner_memtotal_bytes", "runner_gtt_pages_limit",
                "peak_memory_measurement_method", "measured_peak_rss_bytes",
                "measured_peak_gtt_bytes", "measured_peak_uma_bytes"):
        row[key] = resources.get(key)
    return row


def validate_evidence(args: argparse.Namespace, runtime_image: str) -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    ledger_path = args.final_ledger.absolute()
    ledger = read_json(ledger_path, args.final_ledger_sha256, "final ledger")
    expected_ledger_keys = {
        "schema", "phase", "status", "plan_sha256", "assessments", "assessment_sha256",
        "external_attestation_verified", "local_sha256_authenticates_authorship",
        "publication_allowed", "prior_mtp_depth_ledger_sha256",
        "selected_configuration_id", "selected_arm_id", "selected_mtp_matrix_quant_contract",
        "selected_mtp_depth", "selected_artifact_identity", "final_metrics",
        "final_heldout_used_for_selection",
    }
    if (set(ledger) != expected_ledger_keys or ledger.get("schema") != bakeoff.LEDGER_SCHEMA
            or ledger.get("phase") != "final" or ledger.get("status") != "final_selection_complete"
            or ledger.get("external_attestation_verified") is not True
            or ledger.get("local_sha256_authenticates_authorship") is not False
            or ledger.get("publication_allowed") is not False
            or ledger.get("final_heldout_used_for_selection") is not False):
        raise EnvelopeError("final ledger is not a sealed, externally-attested final confirmation")
    assessments = ledger.get("assessments")
    digests = ledger.get("assessment_sha256")
    if not isinstance(assessments, list) or len(assessments) != 1 or not isinstance(digests, list):
        raise EnvelopeError("final ledger must contain exactly one final assessment")
    assessment = assessments[0]
    if (not isinstance(assessment, dict) or digests != [sha256_bytes(canonical(assessment))]
            or assessment.get("schema") != bakeoff.ASSESSMENT_SCHEMA
            or assessment.get("status") != "complete" or assessment.get("stage") != "final"
            or assessment.get("row_id") != "final-confirmation"
            or assessment.get("final_release_eligible") is not True
            or assessment.get("external_attestation_required") is not True
            or assessment.get("local_sha256_authenticates_authorship") is not False
            or assessment.get("publication_allowed") is not False):
        raise EnvelopeError("embedded final assessment is not the exact externally-attested row")

    manifest_path = args.measurement_manifest.absolute()
    manifest = read_json(manifest_path, args.measurement_manifest_sha256,
                         "measurement manifest")
    quality_path = args.quality_contract.absolute()
    read_stable(quality_path, args.quality_contract_sha256, "quality contract")
    hardware_path = args.hardware_evidence.absolute()
    hardware = read_json(hardware_path, args.hardware_evidence_sha256, "hardware evidence")
    if assessment.get("measurement_manifest_sha256") != args.measurement_manifest_sha256:
        raise EnvelopeError("final assessment names a different measurement manifest")
    if assessment.get("quality_contract_sha256") != args.quality_contract_sha256:
        raise EnvelopeError("final assessment names a different quality contract")
    evidence = manifest.get("evidence") or {}
    quality_desc = evidence.get("quality_contract")
    hardware_desc = evidence.get("matching_mtp_hardware_measurement")
    q_path, q_sha = exact_descriptor(quality_desc, "measurement quality contract")
    h_path, h_sha = exact_descriptor(hardware_desc, "measurement hardware evidence")
    if q_path != quality_path or q_sha != args.quality_contract_sha256:
        raise EnvelopeError("measurement manifest substitutes a different quality contract")
    if h_path != hardware_path or h_sha != args.hardware_evidence_sha256:
        raise EnvelopeError("measurement manifest substitutes different hardware evidence")

    row = reconstruct_measurement(assessment, manifest_path,
                                  args.measurement_manifest_sha256, quality_path,
                                  args.quality_contract_sha256)
    try:
        hard_gates = bakeoff.read_object(
            bakeoff.DEFAULT_RECIPE, "bakeoff recipe")["hard_gates"]
        validated_metrics = bakeoff.assess(
            row, hard_gates, assessment["corpus_sha256"])
        metrics = bakeoff.derive_assessment(
            assessment.get("decision_inputs") or {}, hard_gates)
        artifact_identity = bakeoff.candidate_artifact_identity(row, validated_metrics)
    except (bakeoff.BakeoffError, KeyError, OSError, TypeError, ValueError) as exc:
        raise EnvelopeError(f"final quality/hardware evidence does not reproduce: {exc}") from exc
    if (metrics != ledger.get("final_metrics") or metrics.get("passes") is not True
            or artifact_identity != ledger.get("selected_artifact_identity")
            or assessment.get("artifact_identity") != artifact_identity
            or assessment.get("observed_decision") != metrics):
        raise EnvelopeError("final ledger decision does not reproduce from quality/hardware evidence")
    runtime = assessment.get("runtime_identity") or {}
    match = IMAGE.fullmatch(runtime_image)
    if (match is None or runtime.get("container_digest") != f"sha256:{match.group(2)}"
            or HEX40.fullmatch(str(runtime.get("ember_revision"))) is None
            or HEX64.fullmatch(str(runtime.get("engine_binary_sha256"))) is None):
        raise EnvelopeError("runtime image/revision/binary differs from the final measurement")
    if (hardware.get("schema") != "ember.qwen3.8.real-weight-gate.v2"
            or hardware.get("publish_approved") is not False):
        raise EnvelopeError("hardware evidence is not the non-promoting real-weight gate")
    return ledger, assessment, manifest, hardware


def assemble(args: argparse.Namespace, created_at: str) -> dict[str, Any]:
    match = IMAGE.fullmatch(args.runtime_image)
    if match is None:
        raise EnvelopeError("runtime image must be immutable repository@sha256")
    try:
        parsed_time = dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EnvelopeError("created-at must be an ISO-8601 timestamp") from exc
    if parsed_time.tzinfo is None:
        raise EnvelopeError("created-at must include a timezone")
    plan, files = validate_upload_plan(args.upload_plan.absolute(), args.upload_plan_sha256)
    ledger, assessment, measurement_manifest, _hardware = validate_evidence(
        args, args.runtime_image)
    artifact_manifest, _ = planned_json(files, "artifact-manifest.json")
    build_record, build_row = planned_json(files, "qwen-quant-build-record.json")
    intervention, intervention_row = planned_json(files, "qwen-intervention-manifest.json")
    _profile, profile_row = planned_json(files, "release-profile.json")
    checksums_row = planned_file(files, "SHA256SUMS")
    identity = ledger["selected_artifact_identity"]
    artifacts = artifact_manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise EnvelopeError("artifact manifest has no ordered model artifacts")
    model_inventory = []
    planned = {row["path_in_repo"]: row for row in files}
    for index, item in enumerate(artifacts, 1):
        if not isinstance(item, dict) or set(item) != {"filename", "size_bytes", "sha256"}:
            raise EnvelopeError("artifact manifest model inventory is malformed")
        row = planned.get(item["filename"])
        if row is None or (row["size_bytes"], row["sha256"]) != (
                item["size_bytes"], item["sha256"]):
            raise EnvelopeError("artifact manifest differs from planned model bytes")
        model_inventory.append({"index": index, "sha256": item["sha256"],
                                "bytes": item["size_bytes"]})
    companions = artifact_manifest.get("companion_artifacts")
    if not isinstance(companions, list) or len(companions) != 2:
        raise EnvelopeError("artifact manifest must contain exactly MTP and vision-mmproj companions")
    by_role = {
        item.get("role"): item for item in companions if isinstance(item, dict)
    }
    if len(by_role) != 2 or set(by_role) != {"mtp", "vision_mmproj"}:
        raise EnvelopeError("artifact manifest companion roles must be unique MTP and vision_mmproj")
    mtp = by_role["mtp"]
    vision = by_role["vision_mmproj"]
    if set(mtp) != {"role", "filename", "size_bytes", "sha256"}:
        raise EnvelopeError("artifact manifest MTP companion contract differs")
    if (not isinstance(vision, dict) or vision.get("role") != "vision_mmproj"
            or vision.get("format") != "BF16" or vision.get("required_for") != "multimodal"):
        raise EnvelopeError("artifact manifest vision companion contract differs")
    mtp_row = planned.get(mtp.get("filename"))
    vision_row = planned.get(vision.get("filename"))
    measured_mtp = ((measurement_manifest.get("artifacts") or {}).get("mtp")
                    if isinstance(measurement_manifest, dict) else None)
    measured_vision = ((measurement_manifest.get("artifacts") or {}).get("vision_mmproj")
                       if isinstance(measurement_manifest, dict) else None)
    if (mtp_row is None or not isinstance(measured_mtp, dict)
            or (mtp_row["sha256"], mtp_row["size_bytes"]) !=
            (mtp.get("sha256"), mtp.get("size_bytes"))
            or (mtp_row["sha256"], mtp_row["size_bytes"]) !=
            (measured_mtp.get("sha256"), measured_mtp.get("bytes"))):
        raise EnvelopeError("packaged MTP companion differs from measured hardware evidence")
    if (vision_row is None or not isinstance(measured_vision, dict)
            or (vision_row["sha256"], vision_row["size_bytes"]) !=
            (vision.get("sha256"), vision.get("size_bytes"))
            or (vision_row["sha256"], vision_row["size_bytes"]) !=
            (measured_vision.get("sha256"), measured_vision.get("bytes"))):
        raise EnvelopeError("packaged vision companion differs from measured hardware evidence")
    selected_names = [item["filename"] for item in artifacts]
    selected_names.extend((mtp["filename"], vision["filename"]))
    if (len(selected_names) != len(set(selected_names))
            or any(SAFE_RELEASE_BASENAME.fullmatch(str(name)) is None
                   for name in selected_names)):
        raise EnvelopeError("selected package artifact names are unsafe or duplicated")
    integrity = artifact_manifest.get("model_artifact_integrity")
    expected_integrity = {
        "checksum_filename": "SHA256SUMS",
        "checksum_format": "gnu_sha256sum_text",
        "sha256": checksums_row["sha256"],
        "basenames_only": True,
        "ordered_filenames": selected_names,
        "entry_count": len(selected_names),
    }
    if integrity != expected_integrity:
        raise EnvelopeError("artifact manifest SHA256SUMS release evidence differs")
    selected_rows = [*artifacts, mtp, vision]
    expected_checksums = "".join(
        f"{item['sha256']}  {item['filename']}\n" for item in selected_rows
    ).encode("utf-8")
    actual_checksums = read_stable(
        Path(checksums_row["local_path"]), checksums_row["sha256"], "SHA256SUMS"
    )
    if actual_checksums != expected_checksums:
        raise EnvelopeError("SHA256SUMS does not exactly cover the selected runtime artifacts")
    if (sha256_bytes(canonical(model_inventory)) != identity.get("model_inventory_sha256")
            or sum(item["bytes"] for item in model_inventory) != identity.get("artifact_bytes")
            or build_row["sha256"] != identity.get("build_record_sha256")
            or intervention_row["sha256"] != identity.get("intervention_manifest_sha256")
            or profile_row["sha256"] != identity.get("profile_sha256")):
        raise EnvelopeError("package bytes differ from the final selected artifact identity")
    if (build_record.get("status") != "complete" or build_record.get("publishes") is not False
            or intervention.get("status") != "complete"):
        raise EnvelopeError("package build/intervention evidence is incomplete")
    package_build = artifact_manifest.get("build") or {}
    runtime = assessment["runtime_identity"]
    if (artifact_manifest.get("candidate") != {
            "repo_id": plan["repo_id"], "repo_type": plan["repo_type"],
            "revision": plan["revision"], "published": False}
            or package_build.get("engine_revision") != runtime.get("ember_revision")
            or package_build.get("container_image") != args.runtime_image):
        raise EnvelopeError("package manifest candidate/runtime binding differs")
    inventory = [{key: row[key] for key in ("path_in_repo", "size_bytes", "sha256")}
                 for row in files]
    return {
        "schema": SCHEMA,
        "status": "candidate_certified",
        "created_at": created_at,
        "publishes": False,
        "package": {
            "upload_plan": file_descriptor(args.upload_plan.absolute(),
                                           args.upload_plan_sha256, "upload plan"),
            "inventory_sha256": sha256_bytes(canonical(inventory)),
            "files": files,
        },
        "candidate": {"repo_id": plan["repo_id"], "repo_type": plan["repo_type"],
                      "revision": plan["revision"], "base_revision": "main"},
        "runtime": {"image": args.runtime_image, "image_digest": f"sha256:{match.group(2)}",
                    "ember_revision": runtime["ember_revision"],
                    "engine_binary_sha256": runtime["engine_binary_sha256"],
                    "tensor_format_contract_sha256": runtime[
                        "tensor_format_contract_sha256"]},
        "evidence": {
            "final_ledger": file_descriptor(args.final_ledger.absolute(),
                                            args.final_ledger_sha256, "final ledger"),
            "final_ledger_attestation": file_descriptor(
                args.final_ledger_attestation.absolute(),
                args.final_ledger_attestation_sha256, "final ledger attestation"),
            "measurement_manifest": file_descriptor(
                args.measurement_manifest.absolute(), args.measurement_manifest_sha256,
                "measurement manifest"),
            "quality_contract": file_descriptor(
                args.quality_contract.absolute(), args.quality_contract_sha256,
                "quality contract"),
            "hardware": file_descriptor(args.hardware_evidence.absolute(),
                                        args.hardware_evidence_sha256, "hardware evidence"),
        },
        "attestation_policy": {
            "repository": EXPECTED_ATTESTATION_REPOSITORY,
            "signer_workflow": EXPECTED_ATTESTATION_WORKFLOW,
            "verification_required_before_oidc_exchange": True,
        },
        "authorization": {
            "candidate_upload": True,
            "only_revision": plan["revision"],
            "immutable_commit_verification_required": True,
            "promotion": False,
            "promotion_workflow_present": False,
            "requires_separate_environment_approval": True,
        },
    }


def write_new(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute() or path == Path("/") or path.parent.is_symlink():
        raise EnvelopeError("output must be an absolute new path below a non-symlink directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    except OSError as exc:
        raise EnvelopeError(f"refusing publication-envelope output: {exc}") from exc


def arguments_from_envelope(value: dict[str, Any]) -> argparse.Namespace:
    evidence = value.get("evidence") or {}
    package = value.get("package") or {}
    runtime = value.get("runtime") or {}
    def source(name: str) -> tuple[Path, str]:
        row = evidence.get(name) if name != "upload_plan" else package.get(name)
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes"}:
            raise EnvelopeError(f"envelope {name} descriptor differs")
        return Path(str(row["path"])), str(row["sha256"])
    upload, upload_sha = source("upload_plan")
    ledger, ledger_sha = source("final_ledger")
    attestation, attestation_sha = source("final_ledger_attestation")
    measurement, measurement_sha = source("measurement_manifest")
    quality, quality_sha = source("quality_contract")
    hardware, hardware_sha = source("hardware")
    return argparse.Namespace(
        upload_plan=upload, upload_plan_sha256=upload_sha,
        final_ledger=ledger, final_ledger_sha256=ledger_sha,
        final_ledger_attestation=attestation,
        final_ledger_attestation_sha256=attestation_sha,
        measurement_manifest=measurement, measurement_manifest_sha256=measurement_sha,
        quality_contract=quality, quality_contract_sha256=quality_sha,
        hardware_evidence=hardware, hardware_evidence_sha256=hardware_sha,
        runtime_image=runtime.get("image"))


def verify_envelope(path: Path, expected: str,
                    expected_engine_revision: str | None = None) -> dict[str, Any]:
    value = read_json(path.absolute(), expected, "publication envelope")
    if set(value) != {"schema", "status", "created_at", "publishes", "package",
                      "candidate", "runtime", "evidence", "attestation_policy",
                      "authorization"} or value.get("schema") != SCHEMA:
        raise EnvelopeError("publication envelope schema/fields differ")
    rebuilt = assemble(arguments_from_envelope(value), value.get("created_at"))
    if rebuilt != value:
        raise EnvelopeError("publication envelope does not reproduce from its bound evidence")
    if (expected_engine_revision is not None
            and value["runtime"]["ember_revision"] != expected_engine_revision):
        raise EnvelopeError("publication envelope was produced for another Ember commit")
    return value


def verify_remote(value: dict[str, Any], root: Path, commit: str) -> dict[str, Any]:
    if HEX40.fullmatch(commit) is None:
        raise EnvelopeError("candidate commit must be an immutable 40-hex Hub commit")
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise EnvelopeError("remote verification root must be an absolute non-symlink directory")
    verified = []
    for row in value["package"]["files"]:
        destination = row["path_in_repo"]
        path = root / destination
        actual_size = verify_stable(
            path.absolute(), row["sha256"], f"remote file {destination}")
        if path.resolve().parent != root.resolve() or actual_size != row["size_bytes"]:
            raise EnvelopeError(f"remote file {destination} escaped or changed size")
        verified.append({"path_in_repo": destination, "size_bytes": actual_size,
                         "sha256": row["sha256"]})
    return {"schema": RECEIPT_SCHEMA, "status": "candidate_commit_verified",
            "repo_id": value["candidate"]["repo_id"],
            "candidate_revision": value["candidate"]["revision"],
            "candidate_commit": commit,
            "envelope_sha256": sha256_bytes(canonical(value)),
            "inventory_sha256": sha256_bytes(canonical(verified)),
            "files": verified, "promotion_allowed": False}


def add_sources(parser: argparse.ArgumentParser) -> None:
    for name in ("upload-plan", "final-ledger", "final-ledger-attestation",
                 "measurement-manifest", "quality-contract", "hardware-evidence"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--runtime-image", required=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    add_sources(build)
    build.add_argument("--created-at", required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--envelope", type=Path, required=True)
    verify.add_argument("--envelope-sha256", required=True)
    verify.add_argument("--expected-engine-revision")
    emit = sub.add_parser("emit-upload-files")
    emit.add_argument("--envelope", type=Path, required=True)
    emit.add_argument("--envelope-sha256", required=True)
    remote = sub.add_parser("verify-remote")
    remote.add_argument("--envelope", type=Path, required=True)
    remote.add_argument("--envelope-sha256", required=True)
    remote.add_argument("--root", type=Path, required=True)
    remote.add_argument("--candidate-commit", required=True)
    remote.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        if args.command == "build":
            value = assemble(args, args.created_at)
            write_new(args.output.absolute(), value)
            result = {"status": value["status"], "publishes": False,
                      "output": str(args.output.absolute())}
        else:
            value = verify_envelope(args.envelope, args.envelope_sha256,
                                    getattr(args, "expected_engine_revision", None))
            if args.command == "emit-upload-files":
                for row in value["package"]["files"]:
                    sys.stdout.buffer.write(row["local_path"].encode() + b"\0")
                    sys.stdout.buffer.write(row["path_in_repo"].encode() + b"\0")
                return 0
            if args.command == "verify-remote":
                receipt = verify_remote(value, args.root.absolute(), args.candidate_commit)
                write_new(args.output.absolute(), receipt)
                result = {"status": receipt["status"], "candidate_commit":
                          receipt["candidate_commit"], "promotion_allowed": False}
            else:
                result = {"status": value["status"], "publishes": False,
                          "repo_id": value["candidate"]["repo_id"],
                          "revision": value["candidate"]["revision"]}
    except (EnvelopeError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"qwen_publication_envelope.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
