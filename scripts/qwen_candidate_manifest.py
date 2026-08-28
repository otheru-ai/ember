#!/usr/bin/env python3
"""Normalize one audited Qwen construction receipt into bakeoff manifest v3.

The normalizer does not build, measure, publish, or delete anything.  It binds
the already-built bytes to one canonical serial row and refuses to infer phase
authorization, MTP pairing, depth, quality, or runtime identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import qwen_bakeoff as bakeoff
import qwen_vision_inventory as vision_inventory


CONSTRUCTION_SCHEMA = "ember.qwen3.8.candidate-construction.v1"
MANIFEST_SCHEMA = "ember.qwen3.8.sequential-bakeoff-candidate.v3"
ACCUMULATOR_SCHEMA = "ember.qwen3.8.sequential-bakeoff-accumulator.v2"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._:-]{0,127}")
MTP_CONTRACTS = {"Q4_0_ROCMI4", "Q4_0_ROCMFP4_FAST"}
MTP_ALIASES = {
    "Q4_0_ROCMI4": ("Q4_0_ROCMI4", "ROCMI4"),
    "Q4_0_ROCMFP4_FAST": ("Q4_0_ROCMFP4_FAST", "ROCMFP4-FAST"),
}
REQUEST_SCHEMA = "ember.qwen3.8.candidate-normalization-request.v1"


class ManifestError(ValueError):
    pass


def fail(message: str) -> None:
    raise ManifestError(message)


def sha256_file(path: Path) -> str:
    """Hash large model files through direct I/O so UMA remains reclaimable."""
    digest = hashlib.sha256()
    if path.stat().st_size >= 512 * 1024 * 1024:
        process = subprocess.Popen(
            ["dd", f"if={path}", "iflag=direct", "bs=8M", "status=none"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
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
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def exact_file(value: Any, expected: Any, label: str,
               expected_bytes: Any | None = None) -> Path:
    if (not isinstance(value, str) or not value.startswith("/")
            or "\0" in value or "\n" in value or HEX64.fullmatch(str(expected)) is None):
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
                                       or before.st_size != expected_bytes):
        fail(f"{label} byte count differs")
    if sha256_file(path) != expected:
        fail(f"{label} SHA-256 differs")
    after = path.lstat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                            row.st_mtime_ns, row.st_ctime_ns)
    if identity(before) != identity(after):
        fail(f"{label} changed while verified")
    return path


def descriptor(value: Any, label: str) -> tuple[dict[str, Any], Path]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        fail(f"{label} must contain exactly path and sha256")
    path = exact_file(value["path"], value["sha256"], label)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot parse {label}: {exc}")
    if not isinstance(parsed, dict):
        fail(f"{label} must be a JSON object")
    return parsed, path


def exact_descriptor(path: Path, expected_sha: str, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    absolute = exact_file(str(path.absolute()), expected_sha, label)
    try:
        value = json.loads(absolute.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot parse {label}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value, {"path": str(absolute), "sha256": expected_sha}


def write_new(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute() or path == Path("/") or path.parent.is_symlink():
        fail("output must be a new absolute path below a non-symlink directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        file_descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        fail(f"refusing output path: {exc}")
    with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def exact_image(ref: str, digest: str, label: str) -> dict[str, str]:
    if (not isinstance(ref, str) or not ref.startswith("ghcr.io/")
            or "\n" in ref or "@sha256:" not in ref
            or DIGEST.fullmatch(str(digest)) is None or not ref.endswith("@" + digest)):
        fail(f"{label} must be one exact GHCR repository@digest identity")
    return {"ref": ref, "digest": digest}


def selected_companion(construction: dict[str, Any], contract: str) -> dict[str, Any]:
    rows = construction.get("shared_companions")
    if not isinstance(rows, dict):
        fail("construction descriptor has no shared companion map")
    matches = [rows[key] for key in MTP_ALIASES[contract] if key in rows]
    if not matches or any(value != matches[0] for value in matches[1:]):
        fail("construction companion aliases are missing or disagree")
    if not isinstance(matches[0], dict):
        fail("selected construction companion descriptor is malformed")
    return matches[0]


def validate_companion_inventory(
    construction: dict[str, Any], contract: str, expected_source: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any], Path]:
    raw = selected_companion(construction, contract)
    inventory, inventory_path = descriptor(raw, f"{contract} companion inventory")
    if (inventory.get("schema") != "ember.qwen3.8-flash-next.companion-inventory.v1"
            or set(inventory) != {"schema", "source", "companions"}
            or inventory.get("source") != expected_source):
        fail("canonical companion inventory schema differs")
    rows = inventory.get("companions")
    if not isinstance(rows, list) or len(rows) != 2:
        fail("companion inventory must contain exactly MTP and vision mmproj")
    by_role = {row.get("role"): row for row in rows if isinstance(row, dict)}
    if set(by_role) != {"mtp", "vision_mmproj"}:
        fail("companion inventory roles differ")
    mtp, mmproj = by_role["mtp"], by_role["vision_mmproj"]
    if mtp.get("enabled") is not True or mtp.get("matrix_quant_contract") != contract:
        fail("selected MTP matrix contract differs")
    exact_file(mtp.get("path"), mtp.get("sha256"), "MTP companion", mtp.get("size_bytes"))
    export = exact_file(mtp.get("export_manifest_path"), mtp.get("export_manifest_sha256"),
                        "MTP export manifest")
    vision_contract = vision_inventory.load_contract()
    if (mmproj.get("enabled") is not True or mmproj.get("format") != "BF16"
            or mmproj.get("tensor_inventory_sha256") !=
               vision_contract["tensor_inventory_sha256"]):
        fail("vision companion must be enabled BF16")
    exact_file(mmproj.get("path"), mmproj.get("sha256"), "vision mmproj",
               mmproj.get("size_bytes"))
    text_model = mmproj.get("text_model")
    if (not isinstance(text_model, dict)
            or set(text_model) != {"path", "size_bytes", "sha256", "format",
                                  "metadata_sha256"}
            or text_model.get("format") != "GGUF_VOCAB_ONLY"
            or HEX64.fullmatch(str(text_model.get("metadata_sha256", ""))) is None):
        fail("vision companion lacks its exact vocab-only text model")
    exact_file(text_model.get("path"), text_model.get("sha256"),
               "vision vocab companion", text_model.get("size_bytes"))
    return ({"path": str(inventory_path), "sha256": raw["sha256"]},
            mtp, mmproj, export)


def validate_build_companion(
    inventory_desc: dict[str, str], mtp: dict[str, Any], mmproj: dict[str, Any],
    export: Path, contract: str, build: dict[str, Any],
) -> None:
    bound = build.get("companion_inventory") or {}
    if ((bound.get("manifest") or {}).get("path") != inventory_desc["path"]
            or (bound.get("manifest") or {}).get("sha256") != inventory_desc["sha256"]
            or bound.get("status") != "verified_exact"
            or bound.get("fit_status") != "verified_exact_fit"
            or bound.get("enabled_roles") != ["mtp", "vision_mmproj"]):
        fail("build record is not bound to the selected exact companion inventory")
    build_roles = {row.get("role"): row for row in bound.get("roles") or []
                   if isinstance(row, dict)}
    if set(build_roles) != {"mtp", "vision_mmproj"}:
        fail("build-record companion roles differ")
    expected_mtp = build_roles["mtp"]
    if (expected_mtp.get("path") != mtp.get("path")
            or expected_mtp.get("sha256") != mtp.get("sha256")
            or expected_mtp.get("size_bytes") != mtp.get("size_bytes")
            or expected_mtp.get("matrix_quant_contract") != contract
            or (expected_mtp.get("export_manifest") or {}).get("path") != str(export)
            or (expected_mtp.get("export_manifest") or {}).get("sha256") !=
            mtp.get("export_manifest_sha256")):
        fail("build-record MTP provenance differs from canonical inventory")
    expected_mmproj = build_roles["vision_mmproj"]
    if (any(expected_mmproj.get(key) != mmproj.get(key)
            for key in ("path", "sha256", "size_bytes", "format"))
            or (expected_mmproj.get("gguf_contract") or {}).get(
                "tensor_inventory_sha256") != mmproj.get("tensor_inventory_sha256")
            or any((expected_mmproj.get("text_model") or {}).get(key) !=
                   mmproj["text_model"].get(key)
                   for key in ("path", "size_bytes", "sha256", "format"))
            or ((expected_mmproj.get("text_model") or {}).get("gguf_contract") or {}).get(
                "metadata_sha256") != mmproj["text_model"].get("metadata_sha256")):
        fail("build-record mmproj provenance differs from canonical inventory")


def phase_row(plan: dict[str, Any], construction: dict[str, Any], args: argparse.Namespace,
              prior: dict[str, Any] | None) -> dict[str, Any]:
    stage, row_id = args.stage, args.row_id
    if SAFE_ID.fullmatch(row_id) is None:
        fail("row id is malformed")
    if args.mtp_matrix_quant_contract not in MTP_CONTRACTS:
        fail("MTP matrix contract is unsupported")
    if isinstance(args.mtp_depth, bool) or not 1 <= args.mtp_depth <= 4:
        fail("MTP depth must be 1..4")
    selection = plan.get("selection_plan", plan)
    if stage == "stock":
        expected = selection["stock_control"]
        if (row_id != expected["id"] or construction.get("kind") != "stock"
                or construction.get("intended_stage") != "stock"
                or construction.get("row_id") != row_id
                or args.mtp_matrix_quant_contract != "Q4_0_ROCMI4"
                or args.mtp_depth != 3 or args.runtime_mode != expected["runtime_mode"]):
            fail("stock normalization differs from the canonical stock row")
        return {"configuration_id": None, "arm_id": None,
                "final_release_eligible": False}
    if construction.get("kind") != "intervention":
        fail("non-stock stages require an intervention construction descriptor")
    configuration = construction.get("intervention_configuration_id")
    if stage == "sweep":
        rows = {row["id"]: row for row in selection["sweep_configurations"]}
        expected = rows.get(row_id)
        if (expected is None or construction.get("intended_stage") != "sweep"
                or construction.get("row_id") != row_id or configuration != row_id
                or args.mtp_matrix_quant_contract != "Q4_0_ROCMI4"
                or args.mtp_depth != 3 or args.runtime_mode != expected["runtime_mode"]
                or construction.get("quantization_arm") != expected["quantization_arm"]):
            fail("sweep normalization differs from its canonical configuration")
        return {"configuration_id": row_id, "arm_id": None,
                "final_release_eligible": expected["final_release_eligible"],
                "expected": expected}
    if prior is None:
        fail(f"{stage} normalization requires its prior phase ledger")
    if stage == "format":
        rows = {row["id"]: row for row in selection["format_arms"]}
        expected = rows.get(row_id)
        if (expected is None or construction.get("intended_stage") != "format"
                or construction.get("row_id") != row_id
                or configuration != prior.get("selected_configuration_id")
                or construction.get("quantization_arm") != expected["quantization_arm"]
                or args.mtp_matrix_quant_contract != expected["mtp_matrix_quant_contract"]
                or args.mtp_depth != expected["mtp_depth"]
                or args.runtime_mode != expected["runtime_mode"]):
            fail("format normalization differs from selected intervention/arm semantics")
        return {"configuration_id": configuration, "arm_id": row_id,
                "final_release_eligible": expected["final_release_eligible"],
                "expected": expected}
    if construction.get("intended_stage") != "format":
        fail(f"{stage} must reuse a format-stage construction descriptor")
    selected_arm = prior.get("selected_arm")
    if stage == "mtp-depth":
        expected = {row["id"]: row for row in selection["mtp_depth_configurations"]}.get(row_id)
        if (expected is None or not isinstance(selected_arm, dict)
                or construction.get("row_id") != prior.get("selected_arm_id")
                or configuration != prior.get("selected_configuration_id")
                or construction.get("quantization_arm") != selected_arm.get("quantization_arm")
                or args.mtp_matrix_quant_contract != selected_arm.get("mtp_matrix_quant_contract")
                or args.mtp_depth != expected["mtp_depth"]
                or args.runtime_mode != expected["runtime_mode"]):
            fail("MTP-depth normalization changed the selected format artifact or pairing")
        return {"configuration_id": configuration, "arm_id": prior["selected_arm_id"],
                "final_release_eligible": expected["final_release_eligible"],
                "expected": expected}
    if stage == "final":
        if (row_id != "final-confirmation" or construction.get("row_id") != prior.get("selected_arm_id")
                or configuration != prior.get("selected_configuration_id")
                or construction.get("quantization_arm") !=
                (prior.get("selected_arm") or {}).get("quantization_arm")
                or args.mtp_matrix_quant_contract != prior.get("selected_mtp_matrix_quant_contract")
                or args.mtp_depth != prior.get("selected_mtp_depth")
                or args.runtime_mode != "exact_dequant"):
            fail("final normalization changed the MTP-depth-selected artifact")
        return {"configuration_id": configuration, "arm_id": prior["selected_arm_id"],
                "final_release_eligible": True}
    fail("unsupported candidate stage")


def prior_ledger(args: argparse.Namespace, selection_plan: dict[str, Any]) -> dict[str, Any] | None:
    supplied = (args.prior_ledger is not None, args.prior_ledger_sha256 is not None)
    needs = args.stage in {"format", "mtp-depth", "final"}
    if supplied[0] != supplied[1] or supplied[0] != needs:
        fail(f"{args.stage} prior-ledger inputs are incomplete or forbidden")
    if not needs:
        return None
    ledger, _ = exact_descriptor(args.prior_ledger, args.prior_ledger_sha256, "prior phase ledger")
    expected_phase = {"format": "sweep", "mtp-depth": "format", "final": "mtp-depth"}[
        args.stage]
    if (ledger.get("schema") != bakeoff.LEDGER_SCHEMA or ledger.get("phase") != expected_phase
            or ledger.get("plan_sha256") != bakeoff.canonical_sha256(selection_plan)
            or ledger.get("external_attestation_verified") is not True
            or ledger.get("publication_allowed") is not False):
        fail("prior ledger is not bound to the canonical preceding phase")
    try:
        bakeoff.verify_ledger_semantics(selection_plan, ledger)
    except (bakeoff.BakeoffError, KeyError, TypeError, ValueError) as exc:
        fail(f"prior ledger semantics do not reproduce: {exc}")
    return ledger


def verify_serial_position(args: argparse.Namespace, plan: dict[str, Any], plan_sha: str) -> None:
    pair = (args.prior_accumulator is not None,
            args.prior_accumulator_sha256 is not None)
    if pair[0] != pair[1]:
        fail("prior accumulator path and SHA-256 are required together")
    phase = "sweep" if args.stage in {"stock", "sweep"} else args.stage
    selection = plan.get("selection_plan", plan)
    expected = ([selection["stock_control"]["id"],
                 *[row["id"] for row in selection["sweep_configurations"]]]
                if phase == "sweep" else
                [row["id"] for row in selection["format_arms"]] if phase == "format" else
                [row["id"] for row in selection["mtp_depth_configurations"]]
                if phase == "mtp-depth" else ["final-confirmation"])
    count = 0
    if pair[0]:
        accumulator, _ = exact_descriptor(
            args.prior_accumulator, args.prior_accumulator_sha256, "prior phase accumulator")
        rows = accumulator.get("assessments")
        if (accumulator.get("schema") != ACCUMULATOR_SCHEMA
                or accumulator.get("phase") != phase
                or accumulator.get("plan_sha256") != plan_sha
                or not isinstance(rows, list)
                or accumulator.get("contains_raw_measurements") is not False
                or accumulator.get("external_attestation_required") is not True
                or accumulator.get("publication_allowed") is not False):
            fail("prior accumulator is not an exact compact descriptor for this phase")
        count = len(rows)
    if count >= len(expected) or args.row_id != expected[count]:
        fail("candidate is not the next row in canonical serial phase order")


def normalize(args: argparse.Namespace) -> dict[str, Any]:
    for label, path in (
        ("construction", args.construction), ("quality contract", args.quality_contract),
        ("prior accumulator", args.prior_accumulator), ("prior ledger", args.prior_ledger),
        ("final plan", args.final_plan),
    ):
        if path is not None and not path.is_absolute():
            fail(f"{label} path must be absolute")
    construction, construction_desc = exact_descriptor(
        args.construction, args.construction_sha256, "construction descriptor")
    required = {
        "schema", "status", "publishes", "deletes", "candidate_id", "kind",
        "intended_stage", "row_id", "intervention_configuration_id",
        "quantization_arm", "mtp_matrix_quant_contract", "runtime_mode",
        "builder_revision", "runtime_revision", "images", "capture", "stock_capture",
        "bf16_cache", "shared_companions", "selection_plan", "build_record",
        "builder_attestation", "intervention_manifest", "artifacts",
        "v3_candidate_manifest",
    }
    if (set(construction) != required or construction.get("schema") != CONSTRUCTION_SCHEMA
            or construction.get("status") != "complete"
            or construction.get("publishes") is not False
            or construction.get("deletes") is not False):
        fail("construction descriptor schema or lifecycle state differs")
    builder_revision = construction.get("builder_revision")
    construction_runtime_revision = construction.get("runtime_revision")
    if (HEX40.fullmatch(str(builder_revision)) is None
            or HEX40.fullmatch(str(construction_runtime_revision)) is None):
        fail("construction builder/runtime revision is malformed")
    plan_raw, plan_path = descriptor(construction["selection_plan"], "selection plan")
    try:
        selection_plan = bakeoff.verify_plan(plan_raw)
    except (bakeoff.BakeoffError, KeyError, TypeError, ValueError) as exc:
        fail(f"selection plan does not reproduce: {exc}")
    plan_for_quality = selection_plan
    if args.stage == "final":
        if args.final_plan is None or args.final_plan_sha256 is None:
            fail("final normalization requires the separately unlocked final plan")
        final_raw, _ = exact_descriptor(args.final_plan, args.final_plan_sha256, "final phase plan")
        try:
            plan_for_quality = bakeoff.verify_plan(final_raw)
        except (bakeoff.BakeoffError, KeyError, TypeError, ValueError) as exc:
            fail(f"final phase plan does not reproduce: {exc}")
        if plan_for_quality.get("selection_plan") != selection_plan:
            fail("final phase plan embeds a different selection plan")
    elif args.final_plan is not None or args.final_plan_sha256 is not None:
        fail("only final normalization may consume a final phase plan")
    prior = prior_ledger(args, selection_plan)
    if args.stage == "final":
        sealed = plan_for_quality.get("sealed_recipe_ledger") or {}
        subject = (sealed.get("descriptor") or {}).get("subject") or {}
        if (sealed.get("sha256") != args.prior_ledger_sha256
                or subject.get("path") != str(args.prior_ledger.absolute())
                or subject.get("sha256") != args.prior_ledger_sha256):
            fail("final plan was not unlocked by the supplied MTP-depth ledger")
    semantics = phase_row(selection_plan, construction, args, prior)
    verify_serial_position(args, selection_plan, construction["selection_plan"]["sha256"])

    build, build_path = descriptor(construction["build_record"], "candidate build record")
    attestation, attestation_path = descriptor(
        construction["builder_attestation"], "candidate builder attestation")
    if (build.get("status") != "complete" or build.get("mode") != "execute"
            or (build.get("tools") or {}).get("ember_revision") != builder_revision
            or attestation.get("schema") != "ember.qwen3.8.candidate-workset-attestation.v1"
            or attestation.get("candidate_id") != construction.get("candidate_id")
            or attestation.get("build_record_sha256") != construction["build_record"]["sha256"]):
        fail("build record or builder attestation identity differs")
    experiment = build.get("experiment") or {}
    if construction.get("kind") == "stock":
        expected_experiment = {
            "kind": "stock_control", "stock_weights_unchanged": True,
            "final_release_eligible": False,
            "eligibility_status": "ineligible_stock_control",
            "purpose": "activation_capture_and_bakeoff_baseline",
        }
    else:
        expected_experiment = {
            "kind": "directional_ablation", "stock_weights_unchanged": False,
            "final_release_eligible": False,
            "eligibility_status": "pending_measured_bakeoff_and_hardware_certification",
            "purpose": "measured_bakeoff_candidate",
        }
    if experiment != expected_experiment:
        fail("build record experiment semantics differ from construction kind")
    builder_identity = attestation.get("builder_identity")
    if (not isinstance(builder_identity, dict)
            or builder_identity.get("ember_revision") != builder_revision
            or attestation.get("tensor_format_compatibility_sha256") !=
            builder_identity.get("tensor_format_contract_sha256")):
        fail("builder tensor-format identity differs")
    builder_image = (construction.get("images") or {}).get("builder") or {}
    exact_image(builder_image.get("ref"), builder_image.get("digest"), "builder image")
    if builder_identity.get("container_digest") != builder_image.get("digest"):
        fail("builder image digest differs from builder attestation")
    recipe = build.get("quantization_recipe") or {}
    quantization_arm = recipe.get("id")
    if (quantization_arm != construction.get("quantization_arm")
            or recipe.get("selected_mtp_matrix_quant_contract") !=
            args.mtp_matrix_quant_contract
            or HEX64.fullmatch(str(recipe.get("per_tensor_overrides_sha256"))) is None):
        fail("build-record quantization/MTP arm differs from construction")
    profile = build.get("profile") or {}
    release_profile = selection_plan.get("release_profile") or {}
    if profile != release_profile:
        fail("build record uses a different checked release profile")

    cache, cache_path = descriptor(construction["bf16_cache"], "BF16 cache manifest")
    del cache
    build_cache = (build.get("bf16_cache") or {}).get("manifest") or {}
    if (build_cache.get("path") != str(cache_path)
            or build_cache.get("sha256") != construction["bf16_cache"]["sha256"]
            or attestation.get("bf16_cache_manifest_sha256") !=
            construction["bf16_cache"]["sha256"]):
        fail("candidate was not built from the declared BF16 cache")
    companion_rows = {
        contract: validate_companion_inventory(
            construction, contract, (selection_plan.get("direction_basis") or {}).get("source"))
        for contract in sorted(MTP_CONTRACTS)
    }
    mmproj_identities = [{key: row[2].get(key) for key in (
        "path", "sha256", "size_bytes", "format", "enabled", "text_model")}
        for row in companion_rows.values()]
    if any(identity != mmproj_identities[0] for identity in mmproj_identities[1:]):
        fail("companion inventories do not bind one shared BF16 vision mmproj")
    inventory_desc, mtp, mmproj, mtp_export = companion_rows[
        args.mtp_matrix_quant_contract]
    validate_build_companion(inventory_desc, mtp, mmproj, mtp_export,
                             args.mtp_matrix_quant_contract, build)

    output_rows = (build.get("output") or {}).get("shards")
    declared_rows = (construction.get("artifacts") or {}).get("shards")
    attested_rows = (attestation.get("artifact_identity") or {}).get("quantized_shards")
    if (not isinstance(output_rows, list) or not output_rows
            or output_rows != declared_rows or output_rows != attested_rows):
        fail("construction/build/attestation shard inventories differ")
    shards = []
    for index, row in enumerate(output_rows, 1):
        if not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}:
            fail("candidate shard row is malformed")
        path = exact_file(row["path"], row["sha256"], f"candidate shard {index}",
                          row["size_bytes"])
        shards.append({"path": str(path), "size_bytes": row["size_bytes"],
                       "sha256": row["sha256"]})
    artifact_bytes = sum(row["size_bytes"] for row in shards)
    if construction["artifacts"].get("total_bytes") != artifact_bytes:
        fail("construction artifact total differs")
    identities = [{"index": index, "sha256": row["sha256"], "bytes": row["size_bytes"]}
                  for index, row in enumerate(shards, 1)]
    inventory_sha = bakeoff.canonical_sha256(identities)

    is_stock = args.stage == "stock"
    capture_value, capture_path = descriptor(construction["capture"], "activation capture")
    del capture_value
    if is_stock:
        stock = construction.get("stock_capture")
        if stock != construction.get("capture") or construction.get("intervention_manifest") is not None:
            fail("stock construction does not separately bind its activation capture")
        stock_capture = {"path": str(capture_path), "sha256": stock["sha256"]}
        intervention_path, intervention_sha = None, "0" * 64
        source_sha = stock_capture["sha256"]
        captured = attestation.get("stock_capture") or {}
        if (captured.get("path") != stock_capture["path"]
                or captured.get("sha256") != stock_capture["sha256"]
                or captured.get("byte_identical") is not True
                or build.get("intervention") is not None
                or quantization_arm not in {"profile-default-rocmi4", "rocmi4-control"}):
            fail("stock builder attestation is not byte-identical capture evidence")
    else:
        if construction.get("stock_capture") is not None:
            fail("intervention construction unexpectedly carries stock capture")
        intervention, intervention_file = descriptor(
            construction["intervention_manifest"], "intervention manifest")
        del intervention
        intervention_path, intervention_sha = str(intervention_file), construction[
            "intervention_manifest"]["sha256"]
        stock_capture = None
        source_sha = intervention_sha
        if (attestation.get("stock_capture") is not None
                or attestation.get("intervention_manifest_sha256") != intervention_sha
                or (build.get("intervention") or {}).get("manifest_sha256") != intervention_sha):
            fail("builder intervention provenance differs")
    if attestation.get("intervention_manifest_sha256") != source_sha:
        fail("builder weight-source SHA differs")

    runtime_release = exact_image(args.runtime_release_ref, args.runtime_release_digest,
                                  "runtime release image")
    runtime_dev = exact_image(args.runtime_dev_ref, args.runtime_dev_digest,
                              "runtime development image")
    if HEX40.fullmatch(args.runtime_revision) is None:
        fail("runtime revision is malformed")
    format_sha = builder_identity.get("tensor_format_contract_sha256")
    construction_runtime = (construction.get("images") or {}).get("runtime") or {}
    if set(construction_runtime) != {
            "release_ref", "release_digest", "dev_ref", "dev_digest",
            "tensor_format_contract_sha256"}:
        fail("construction runtime image identity is malformed")
    exact_image(construction_runtime["release_ref"], construction_runtime["release_digest"],
                "construction runtime release image")
    exact_image(construction_runtime["dev_ref"], construction_runtime["dev_digest"],
                "construction runtime development image")
    if construction_runtime["tensor_format_contract_sha256"] != format_sha:
        fail("construction runtime tensor-format contract differs from its builder")
    if args.runtime_tensor_format_contract_sha256 != format_sha:
        fail("runtime cannot decode the builder tensor-format contract")
    runtime_identity = {
        "ember_revision": args.runtime_revision,
        "container_digest": args.runtime_release_digest,
        "tensor_format_contract_sha256": format_sha,
    }
    if prior is not None:
        old_runtime = prior.get("runtime_identity") or {}
        for key, value in runtime_identity.items():
            if old_runtime.get(key) != value:
                fail("normalization changed the prior phase runtime identity")

    candidate = {
        "id": args.row_id, "candidate_id": construction["candidate_id"],
        "stage": args.stage, "configuration_id": semantics["configuration_id"],
        "arm_id": semantics["arm_id"],
        "corpus_sha256": None if args.stage == "final" else
            selection_plan["corpora"]["sweep-validation.jsonl"]["sha256"],
        "quantization_arm": quantization_arm, "model": shards[0]["path"],
        "model_sha256": shards[0]["sha256"], "build_record": str(build_path),
        "build_record_sha256": construction["build_record"]["sha256"],
        "builder_attestation": {"path": str(attestation_path),
                                "sha256": construction["builder_attestation"]["sha256"]},
        "intervention_configuration_id": semantics["configuration_id"],
        "intervention_manifest": intervention_path,
        "intervention_manifest_sha256": intervention_sha,
        "stock_capture": stock_capture,
        "profile_sha256": profile["sha256"],
        "override_sha256": recipe.get("per_tensor_overrides_sha256"),
        "companion_inventory": inventory_desc["path"],
        "companion_inventory_sha256": inventory_desc["sha256"],
        "mtp": mtp["path"], "mtp_bytes": mtp["size_bytes"],
        "mtp_sha256": mtp["sha256"],
        "mtp_export_manifest": mtp["export_manifest_path"],
        "mtp_export_manifest_sha256": mtp["export_manifest_sha256"],
        "mtp_matrix_quant_contract": args.mtp_matrix_quant_contract,
        "vision_mmproj": mmproj["path"], "vision_mmproj_bytes": mmproj["size_bytes"],
        "vision_mmproj_sha256": mmproj["sha256"],
        "vision_mmproj_format": mmproj["format"],
        "vision_vocab": mmproj["text_model"]["path"],
        "vision_vocab_bytes": mmproj["text_model"]["size_bytes"],
        "vision_vocab_sha256": mmproj["text_model"]["sha256"],
        "vision_vocab_format": mmproj["text_model"]["format"],
        "vision_vocab_metadata_sha256": mmproj["text_model"]["metadata_sha256"],
        "quality_contract": None, "quality_contract_sha256": None,
        "mtp_depth": args.mtp_depth, "runtime_mode": args.runtime_mode,
        "final_release_eligible": semantics["final_release_eligible"],
        "tensor_format_compatibility_sha256": format_sha,
        "model_inventory_sha256": inventory_sha, "artifact_bytes": artifact_bytes,
    }
    if is_stock:
        if args.quality_contract is not None or args.quality_contract_sha256 is not None:
            fail("stock normalization cannot claim an intervention quality contract")
    else:
        if args.quality_contract is None or args.quality_contract_sha256 is None:
            fail("intervention normalization requires an audited quality contract")
        quality_path = exact_file(str(args.quality_contract.absolute()),
                                  args.quality_contract_sha256, "quality contract")
        candidate["quality_contract"] = str(quality_path)
        candidate["quality_contract_sha256"] = args.quality_contract_sha256
        corpus_sha = plan_for_quality["corpora"][
            "final-heldout.jsonl" if args.stage == "final" else "sweep-validation.jsonl"]["sha256"]
        try:
            bakeoff.audited_quality({
                "quality_contract": {"path": str(quality_path),
                                     "sha256": args.quality_contract_sha256},
                "candidate_id": candidate["candidate_id"],
                "build_record_sha256": candidate["build_record_sha256"],
                "intervention_manifest_sha256": intervention_sha,
                "profile_sha256": candidate["profile_sha256"],
                "quantization_overrides_sha256": candidate["override_sha256"],
                "model_inventory_sha256": inventory_sha,
                "artifact_bytes": artifact_bytes,
            }, corpus_sha)
        except (bakeoff.BakeoffError, OSError, ValueError) as exc:
            fail(f"audited quality provenance differs: {exc}")

    artifact_identity = {
        "candidate_id": candidate["candidate_id"],
        "build_record_sha256": candidate["build_record_sha256"],
        "intervention_manifest_sha256": candidate["intervention_manifest_sha256"],
        "profile_sha256": candidate["profile_sha256"],
        "quantization_overrides_sha256": candidate["override_sha256"],
        "model_inventory_sha256": inventory_sha,
        "companion_inventory_sha256": inventory_desc["sha256"],
        "mtp_matrix_quant_contract": args.mtp_matrix_quant_contract,
        "mtp_depth": args.mtp_depth, "artifact_bytes": artifact_bytes,
        "companion_artifact_bytes": {"mtp": mtp["size_bytes"],
                                     "vision_mmproj": mmproj["size_bytes"],
                                     "vision_vocab": mmproj["text_model"]["size_bytes"]},
        "quantization_arm": quantization_arm,
        "intervention_configuration_id": semantics["configuration_id"],
        "builder_identity": builder_identity,
        "tensor_format_compatibility_sha256": format_sha,
    }
    if args.stage == "mtp-depth":
        previous = prior.get("selected_artifact_identity") or {}
        without_depth = lambda row: {key: value for key, value in row.items()
                                     if key != "mtp_depth"}
        if without_depth(artifact_identity) != without_depth(previous):
            fail("MTP-depth row does not reuse the exact format-selected artifact")
    if args.stage == "final" and artifact_identity != prior.get("selected_artifact_identity"):
        fail("final row is not the exact MTP-depth-selected artifact")

    cache_root = cache_path.parent.parent
    if cache_root.name != "bf16-cache" or cache_root.is_symlink() or not cache_root.is_dir():
        fail("BF16 cache is not content-addressed beneath its immutable cache root")
    evidence_root = construction_desc["path"]
    evidence_dir = Path(evidence_root).parent
    if evidence_dir.is_symlink() or not evidence_dir.is_dir():
        fail("construction evidence root is unsafe")
    canonical_companions = {
        contract: companion_rows[contract][0] for contract in sorted(MTP_CONTRACTS)
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "builder_revision": builder_revision, "runtime_revision": args.runtime_revision,
        "images": {
            "builder": builder_image,
            "runtime": {"release_ref": runtime_release["ref"],
                        "release_digest": runtime_release["digest"],
                        "dev_ref": runtime_dev["ref"], "dev_digest": runtime_dev["digest"],
                        "tensor_format_contract_sha256": format_sha},
        },
        "selection_plan": {"path": str(plan_path),
                           "sha256": construction["selection_plan"]["sha256"]},
        "release_profile": release_profile,
        "bf16_cache": {"path": str(cache_path),
                       "sha256": construction["bf16_cache"]["sha256"]},
        "workset_root": str(cache_root),
        "corpus_dir": str(Path(selection_plan["corpus_manifest"]["path"]).parent.resolve()),
        "evidence_root": str(evidence_dir.resolve()),
        "shared_companions": canonical_companions,
        "build_record": {"path": str(build_path),
                         "sha256": construction["build_record"]["sha256"]},
        "construction": construction_desc,
        "candidate": candidate,
        "publication_allowed": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--construction", type=Path, required=True)
    result.add_argument("--construction-sha256", required=True)
    result.add_argument("--stage", choices=("stock", "sweep", "format", "mtp-depth", "final"),
                        required=True)
    result.add_argument("--row-id", required=True)
    result.add_argument("--mtp-matrix-quant-contract", choices=sorted(MTP_CONTRACTS), required=True)
    result.add_argument("--mtp-depth", type=int, required=True)
    result.add_argument("--runtime-mode", choices=("exact_dequant", "w4a4_opt_in"), required=True)
    result.add_argument("--runtime-revision", required=True)
    result.add_argument("--runtime-release-ref", required=True)
    result.add_argument("--runtime-release-digest", required=True)
    result.add_argument("--runtime-dev-ref", required=True)
    result.add_argument("--runtime-dev-digest", required=True)
    result.add_argument("--runtime-tensor-format-contract-sha256", required=True)
    result.add_argument("--quality-contract", type=Path)
    result.add_argument("--quality-contract-sha256")
    result.add_argument("--prior-accumulator", type=Path)
    result.add_argument("--prior-accumulator-sha256")
    result.add_argument("--prior-ledger", type=Path)
    result.add_argument("--prior-ledger-sha256")
    result.add_argument("--final-plan", type=Path)
    result.add_argument("--final-plan-sha256")
    result.add_argument("--output", type=Path, required=True)
    return result


def request_namespace(path: Path, expected_sha: str) -> argparse.Namespace:
    value, _ = exact_descriptor(path, expected_sha, "candidate normalization request")
    required = {
        "schema", "construction", "stage", "row_id", "mtp_matrix_quant_contract",
        "mtp_depth", "runtime_mode", "runtime", "quality_contract",
        "prior_accumulator", "prior_ledger", "final_plan", "output",
        "publishes", "deletes",
    }
    if (set(value) != required or value.get("schema") != REQUEST_SCHEMA
            or value.get("publishes") is not False or value.get("deletes") is not False):
        fail("normalization request schema or lifecycle policy differs")
    construction = value.get("construction")
    runtime = value.get("runtime")
    if (not isinstance(construction, dict) or set(construction) != {"path", "sha256"}
            or not isinstance(runtime, dict) or set(runtime) != {
                "revision", "release_ref", "release_digest", "dev_ref", "dev_digest",
                "tensor_format_contract_sha256"}):
        fail("normalization request construction/runtime identity is malformed")

    def exact_request_descriptor(item: dict[str, Any], label: str) -> None:
        if (not isinstance(item.get("path"), str) or not item["path"].startswith("/")
                or "\n" in item["path"] or "\0" in item["path"]
                or HEX64.fullmatch(str(item.get("sha256"))) is None):
            fail(f"normalization request {label} descriptor is malformed")

    exact_request_descriptor(construction, "construction")

    def optional(name: str) -> tuple[Path | None, str | None]:
        item = value.get(name)
        if item is None:
            return None, None
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            fail(f"normalization request {name} descriptor is malformed")
        exact_request_descriptor(item, name)
        return Path(item["path"]), item["sha256"]

    quality, quality_sha = optional("quality_contract")
    accumulator, accumulator_sha = optional("prior_accumulator")
    ledger, ledger_sha = optional("prior_ledger")
    final_plan, final_plan_sha = optional("final_plan")
    output = value.get("output")
    if (not isinstance(output, str) or not output.startswith("/")
            or "\n" in output or "\0" in output):
        fail("normalization request output path is malformed")
    return argparse.Namespace(
        construction=Path(construction["path"]),
        construction_sha256=construction["sha256"],
        stage=value["stage"], row_id=value["row_id"],
        mtp_matrix_quant_contract=value["mtp_matrix_quant_contract"],
        mtp_depth=value["mtp_depth"], runtime_mode=value["runtime_mode"],
        runtime_revision=runtime["revision"], runtime_release_ref=runtime["release_ref"],
        runtime_release_digest=runtime["release_digest"], runtime_dev_ref=runtime["dev_ref"],
        runtime_dev_digest=runtime["dev_digest"],
        runtime_tensor_format_contract_sha256=runtime["tensor_format_contract_sha256"],
        quality_contract=quality, quality_contract_sha256=quality_sha,
        prior_accumulator=accumulator, prior_accumulator_sha256=accumulator_sha,
        prior_ledger=ledger, prior_ledger_sha256=ledger_sha,
        final_plan=final_plan, final_plan_sha256=final_plan_sha,
        output=Path(output),
    )


def main(argv: list[str] | None = None) -> int:
    effective = sys.argv[1:] if argv is None else argv
    if effective and effective[0] == "from-request":
        request_parser = argparse.ArgumentParser(description=__doc__)
        request_parser.add_argument("from_request")
        request_parser.add_argument("--request", type=Path, required=True)
        request_parser.add_argument("--request-sha256", required=True)
        request = request_parser.parse_args(effective)
        try:
            args = request_namespace(request.request, request.request_sha256)
        except (ManifestError, KeyError, OSError, TypeError, ValueError) as exc:
            print(f"qwen-candidate-manifest: {exc}", file=sys.stderr)
            return 1
    else:
        args = parser().parse_args(effective)
    try:
        value = normalize(args)
        if not args.output.is_absolute():
            fail("output must be an absolute path")
        write_new(args.output, value)
    except (ManifestError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"qwen-candidate-manifest: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "manifest": str(args.output),
                      "manifest_sha256": sha256_file(args.output),
                      "publishes": False, "deletes": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
