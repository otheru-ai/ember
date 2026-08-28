#!/usr/bin/env python3
"""Build an offline Hugging Face candidate package plan for Qwen ROCmI4.

This program deliberately has no upload implementation and uses only the
Python standard library.  The generated upload-plan.json names local inputs and
their destination paths, but never reads Hugging Face credentials or contacts
the network.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import sys
import tempfile
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import qwen_vision_inventory as vision_inventory


HEX40 = re.compile(r"^[0-9a-f]{40}$")
OCI_DIGEST = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
CANDIDATE_REVISION = re.compile(r"^candidate/[A-Za-z0-9._-]+$")
QWEN_QSA_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47}
INTERVENTION_TARGET_RE = re.compile(r"^blk\.([0-9]+)\.(attn_output|ssm_out)\.weight$")


class PackageError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_stable_file(source: Path, destination: Path) -> str:
    """Copy one regular input through a pinned descriptor and sync the copy."""
    try:
        named = os.lstat(source)
    except OSError as exc:
        raise PackageError(f"cannot inspect package input {source}: {exc}") from exc
    if not stat.S_ISREG(named.st_mode):
        raise PackageError(f"package input is not a regular file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_stream:
            before = os.fstat(input_stream.fileno())
            if (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
                raise PackageError(f"package input identity changed before copy: {source}")
            with destination.open("xb") as output_stream:
                for chunk in iter(lambda: input_stream.read(8 * 1024 * 1024), b""):
                    output_stream.write(chunk)
                    digest.update(chunk)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            after = os.fstat(input_stream.fileno())
    except OSError as exc:
        raise PackageError(f"cannot snapshot package input {source}: {exc}") from exc
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise PackageError(f"package input changed while it was copied: {source}")
    return digest.hexdigest()


def rename_directory_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise PackageError("Linux renameat2 is required for atomic package publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise PackageError(f"refusing to overwrite existing package directory: {destination}")
        raise PackageError(f"cannot atomically publish package directory: {os.strerror(error)}")
    fsync_directory(destination.parent)


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PackageError(f"{field} must be an object")
    return value


def inspect_bf16_qwen_mmproj(path: Path) -> dict[str, Any]:
    """Read the bounded GGUF header needed to identify the pinned Qwen tower."""
    scalar_formats = {
        0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
        6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
    }

    with path.open("rb") as stream:
        def exact(size: int) -> bytes:
            value = stream.read(size)
            if len(value) != size:
                raise PackageError("vision mmproj has a truncated GGUF header")
            return value

        def integer(fmt: str) -> int:
            return int(struct.unpack(fmt, exact(struct.calcsize(fmt)))[0])

        def string() -> str:
            size = integer("<Q")
            if size > 16 * 1024 * 1024:
                raise PackageError("vision mmproj GGUF string exceeds the audit bound")
            try:
                return exact(size).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PackageError("vision mmproj has invalid UTF-8 metadata") from exc

        def value(kind: int, retain: bool = True) -> Any:
            if kind in scalar_formats:
                unpacked = struct.unpack(
                    scalar_formats[kind], exact(struct.calcsize(scalar_formats[kind]))
                )[0]
                return unpacked if retain else None
            if kind == 8:
                parsed = string()
                return parsed if retain else None
            if kind == 9:
                element_kind = integer("<I")
                count = integer("<Q")
                if count > 16 * 1024 * 1024:
                    raise PackageError("vision mmproj GGUF array exceeds the audit bound")
                parsed = [] if retain and count <= 4096 else None
                for _ in range(count):
                    item = value(element_kind, parsed is not None)
                    if parsed is not None:
                        parsed.append(item)
                return parsed
            raise PackageError(f"vision mmproj has unsupported GGUF value type {kind}")

        if exact(4) != b"GGUF":
            raise PackageError("vision mmproj is not a GGUF file")
        version = integer("<I")
        if version not in (2, 3):
            raise PackageError(f"vision mmproj uses unsupported GGUF version {version}")
        tensor_count = integer("<Q")
        metadata_count = integer("<Q")
        if tensor_count < 1 or tensor_count > 1_000_000 or metadata_count > 1_000_000:
            raise PackageError("vision mmproj GGUF header counts are outside audit bounds")
        wanted = {
            "general.architecture", "general.file_type", "clip.projector_type",
            "clip.has_vision_encoder", "clip.vision.projection_dim",
            "clip.vision.spatial_merge_size",
        }
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = string()
            kind = integer("<I")
            parsed = value(kind, key in wanted)
            if key in wanted:
                metadata[key] = parsed
        tensors: list[dict[str, Any]] = []
        tensor_types: set[int] = set()
        offsets: list[int] = []
        for _ in range(tensor_count):
            name = string()
            dimensions = integer("<I")
            if dimensions > 8:
                raise PackageError("vision mmproj tensor rank exceeds the audit bound")
            shape = [integer("<Q") for _ in range(dimensions)]
            tensor_types.add(integer("<I"))
            offsets.append(integer("<Q"))
            tensors.append({"name": name, "shape": shape})
        data_start = (stream.tell() + 31) & ~31
        file_size = path.stat().st_size
        if data_start >= file_size or any(offset >= file_size - data_start for offset in offsets):
            raise PackageError("vision mmproj tensor offsets exceed the GGUF payload")

    expected = {
        "general.architecture": "clip",
        "general.file_type": 32,
        "clip.projector_type": "qwen3vl_merger",
        "clip.has_vision_encoder": True,
        "clip.vision.projection_dim": 2560,
        "clip.vision.spatial_merge_size": 2,
    }
    if any(metadata.get(key) != expected_value for key, expected_value in expected.items()):
        raise PackageError("vision mmproj metadata is not the pinned Qwen3.8 BF16 tower")
    try:
        inventory = vision_inventory.validate_inventory(tensors)
    except vision_inventory.VisionInventoryError as exc:
        raise PackageError(str(exc)) from exc
    if 30 not in tensor_types or not tensor_types.issubset({0, 30}):
        raise PackageError("vision mmproj tensor inventory is not unquantized BF16/F32")
    return {
        "gguf_version": version,
        "tensor_count": tensor_count,
        "tensor_types": sorted(tensor_types),
        "metadata": metadata,
        "tensor_inventory_contract": inventory["contract_schema"],
        "tensor_inventory_sha256": inventory["tensor_inventory_sha256"],
    }


def load_profile(path: Path) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read release profile {path}: {exc}") from exc
    require_mapping(profile, "profile")
    if profile.get("schema_version") != 1:
        raise PackageError("unsupported profile schema_version")
    source = require_mapping(profile.get("source"), "source")
    conversion = require_mapping(profile.get("conversion"), "conversion")
    quantizer = require_mapping(profile.get("quantizer"), "quantizer")
    intervention = require_mapping(profile.get("intervention"), "intervention")
    quantization = require_mapping(profile.get("quantization"), "quantization")
    artifact = require_mapping(profile.get("artifact"), "artifact")
    release = require_mapping(profile.get("release"), "release")
    license_info = require_mapping(source.get("license"), "source.license")
    for field, value in (
        ("source.revision", source.get("revision")),
        ("quantizer.revision", quantizer.get("revision")),
    ):
        if not isinstance(value, str) or not HEX40.fullmatch(value):
            raise PackageError(f"{field} must be a lowercase 40-character commit")
    license_hash = license_info.get("sha256")
    if not isinstance(license_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", license_hash):
        raise PackageError("source.license.sha256 must be a lowercase SHA-256")
    if quantizer.get("format") != "Q4_0_ROCMI4":
        raise PackageError("quantizer.format must be Q4_0_ROCMI4")
    if (
        intervention.get("required") is not True
        or intervention.get("kind") != "directional_ablation"
        or intervention.get("manifest_schema_version") != 1
        or intervention.get("application_stage") != "pre_quantization_encoding"
        or intervention.get("weight_intervention_required") is not True
        or intervention.get("prompt_only_allowed") is not False
        or intervention.get("manifest_filename") != "qwen-intervention-manifest.json"
        or intervention.get("quantizer_application_report_required") is not True
        or intervention.get("required_evidence") != [
            "source", "tooling", "corpora", "directions", "targets", "tensor_map"
        ]
        or intervention.get("otheru_pipeline") != {
            "repository": "https://git.otheru.ai/akadmin/otheru-quant-pipeline",
            "branch": "ember-contract-and-drafter-fix",
            "revision": "a3c6a728510f91394e991504951ac316cd3a89af",
        }
        or intervention.get("upstream_heretic") != {
            "repository": "https://github.com/p-e-w/heretic",
            "revision": "bedb94ef117a271532ac2058447fbc165d5051bd",
        }
    ):
        raise PackageError("profile does not require the pinned Heretic weight intervention")
    memory_gate = require_mapping(
        quantization.get("native_262k_memory_gate"),
        "quantization.native_262k_memory_gate",
    )
    if (
        memory_gate.get("native_context_tokens") != 262144
        or memory_gate.get("device_budget_bytes") != 137438953472
        or memory_gate.get("runtime_reserve_bytes") != 34359738368
        or memory_gate.get("rule") != "artifact_bytes + runtime_reserve_bytes <= device_budget_bytes"
        or memory_gate.get("yarn_1m_math_oracle_passed") is not True
        or memory_gate.get("yarn_1m_runtime_certified") is not False
        or memory_gate.get("yarn_1m_fit_claim") is not False
    ):
        raise PackageError("quantization.native_262k_memory_gate does not match the audited contract")
    vision = require_mapping(
        quantization.get("vision_artifact"), "quantization.vision_artifact"
    )
    provider = require_mapping(
        vision.get("runtime_provider"),
        "quantization.vision_artifact.runtime_provider",
    )
    companions = artifact.get("required_companion_artifacts")
    if not isinstance(companions, list) or len(companions) != 1:
        raise PackageError("artifact must declare exactly one required vision companion")
    companion = require_mapping(companions[0], "artifact.required_companion_artifacts[0]")
    if (
        vision.get("layout") != "separate_mmproj_gguf"
        or vision.get("required_for_multimodal_runtime") is not True
        or vision.get("storage_format") != "BF16"
        or vision.get("converter_option") != "--mmproj"
        or vision.get("included_in_text_shard_budget") is not False
        or vision.get("runtime_implementation_status")
            != "provider_built_gpu_free_abi_smoke_only"
        or provider.get("kind") != "dynamic_llama_cpp_mtmd"
        or provider.get("llama_cpp_revision") != conversion.get("revision")
        or provider.get("shared_object") != "libember_qwen4exp_vision_provider.so"
        or provider.get("build_script") != "scripts/build_qwen_vision_provider.sh"
        or provider.get("lazy_load") is not True
        or provider.get("text_model_view") != "vocab_only"
        or provider.get("duplicates_text_tensor_weights") is not False
        or provider.get("real_weight_differential_certified") is not False
        or provider.get("gfx1151_runtime_certified") is not False
        or companion.get("role") != "vision_mmproj"
        or companion.get("filename") != "Qwen3.8-Flash-Next-BF16-mmproj.gguf"
        or companion.get("format") != "BF16"
        or companion.get("required_for") != "multimodal"
    ):
        raise PackageError("profile does not carry the audited separate BF16 vision contract")
    required_artifacts = release.get("required_artifacts")
    if (
        not isinstance(required_artifacts, list)
        or companion["filename"] not in required_artifacts
    ):
        raise PackageError("release.required_artifacts must include the BF16 mmproj")
    q1_memory = require_mapping(release.get("q1_correctness_memory"), "release.q1_correctness_memory")
    if (
        q1_memory.get("classification") != "correctness_first_no_performance_claim"
        or q1_memory.get("native_cache_bytes") != 14495514624
        or q1_memory.get("gdn_state_bytes") != 117669888
        or q1_memory.get("additional_runtime_reserve_bytes") != 8589934592
        or q1_memory.get("accounted_total_bytes") != 23203119104
        or q1_memory.get("copy_on_write_accounting") is not True
        or q1_memory.get("performance_claim") is not False
    ):
        raise PackageError("release.q1_correctness_memory does not match the audited contract")
    runner = require_mapping(
        release.get("conversion_runner_requirements"),
        "release.conversion_runner_requirements",
    )
    if runner.get("minimum_free_disk_gib") != 1152 or runner.get("minimum_physical_ram_gib") != 256:
        raise PackageError("release conversion runner must require 1152 GiB disk and 256 GiB RAM")
    layout_gate = require_mapping(release.get("artifact_layout_gate"), "release.artifact_layout_gate")
    if (
        layout_gate.get("current_quantizer_multi_shard_supported") is not True
        or layout_gate.get("default_split_max_size") != "48G"
        or layout_gate.get("aggregate_preflight_before_output") is not True
        or layout_gate.get("transactional_no_clobber_shards") is not True
        or layout_gate.get("atomic_committed_work_directory") is not True
        or layout_gate.get("directory_commit_method") != "renameat2(RENAME_NOREPLACE)"
        or layout_gate.get("conditional_rollback_unlink") is not False
        or layout_gate.get("package_atomic_no_clobber_directory") is not True
        or layout_gate.get("single_large_file_publication_accepted") is not False
        or layout_gate.get("single_large_file_roundtrip_hash_tested") is not False
        or layout_gate.get("publication_blocked") is not False
    ):
        raise PackageError("release multi-shard artifact-layout contract is missing")
    for field in ("repo_id", "filename", "repo_type"):
        if not isinstance(artifact.get(field), str) or not artifact[field]:
            raise PackageError(f"artifact.{field} must be a non-empty string")
    if "Heretic" not in artifact["repo_id"] or "Heretic" not in artifact["filename"]:
        raise PackageError("directional-ablation release must use an explicit Heretic artifact name")
    return profile


def snapshot_intervention_evidence(
    profile: dict[str, Any], build_record: dict[str, Any],
    source_build_record_path: Path, out_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Verify and snapshot the manifest that the quantizer actually applied."""
    contract = profile["intervention"]
    evidence = require_mapping(build_record.get("intervention"), "build_record.intervention")
    manifest_hash = evidence.get("manifest_sha256")
    target_hash = evidence.get("target_names_sha256")
    target_count = evidence.get("target_count")
    targets = evidence.get("targets")
    if (
        evidence.get("kind") != "directional_ablation"
        or evidence.get("application_stage") != "pre_quantization_encoding"
        or evidence.get("weight_intervention") is not True
        or evidence.get("prompt_only") is not False
        or evidence.get("manifest_filename") != contract["manifest_filename"]
        or not isinstance(manifest_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest_hash) is None
        or not isinstance(target_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", target_hash) is None
        or not isinstance(target_count, int) or isinstance(target_count, bool) or target_count < 1
        or not isinstance(targets, list) or len(targets) != target_count
        or any(not isinstance(name, str) or not name for name in targets)
        or targets != sorted(targets)
        or evidence.get("tooling") != {
            "otheru_quant_pipeline": contract["otheru_pipeline"],
            "upstream_heretic": contract["upstream_heretic"],
        }
    ):
        raise PackageError("quantization build record lacks complete Heretic intervention evidence")
    if hashlib.sha256("\n".join(sorted(targets)).encode("utf-8")).hexdigest() != target_hash:
        raise PackageError("quantization build record intervention tensor-map digest is inconsistent")
    for stage, applied in (("quantizer_preflight", False), ("quantizer_application", True)):
        report = require_mapping(evidence.get(stage), f"build_record.intervention.{stage}")
        if (
            report.get("manifest_sha256") != manifest_hash
            or report.get("target_names_sha256") != target_hash
            or report.get("target_count") != target_count
            or report.get("targets") != targets
            or report.get("validated") is not True
            or report.get("applied") is not applied
        ):
            raise PackageError(f"quantization build record {stage} evidence is inconsistent")
    application = evidence["quantizer_application"]
    metrics = application.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != target_count:
        raise PackageError("quantization build record lacks per-target intervention metrics")
    nonnegative_fields = (
        "source_projection_l2", "stored_projection_l2", "stored_projection_ratio",
        "relative_frobenius_delta", "row_norm_relative_rmse", "row_norm_relative_max",
    )
    for expected_name, metric_value in zip(targets, metrics, strict=True):
        metric = require_mapping(metric_value, "build_record intervention metric")
        if metric.get("tensor_name") != expected_name:
            raise PackageError("intervention metrics do not match the exact target order")
        for field in nonnegative_fields:
            value = metric.get(field)
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise PackageError(f"intervention metric {field} must be finite and non-negative")
        signed = metric.get("signed_projection_coefficient")
        if isinstance(signed, bool) or not isinstance(signed, (int, float)) or not math.isfinite(float(signed)):
            raise PackageError("intervention signed projection coefficient must be finite")

    source_manifest = source_build_record_path.parent / contract["manifest_filename"]
    destination = out_dir / contract["manifest_filename"]
    copied_hash = copy_stable_file(source_manifest, destination)
    if copied_hash != manifest_hash:
        raise PackageError("applied intervention manifest SHA-256 differs from the build record")
    try:
        manifest = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read applied intervention manifest: {exc}") from exc
    require_mapping(manifest, "intervention manifest")
    if (
        manifest.get("schema_version") != contract["manifest_schema_version"]
        or manifest.get("kind") != contract["kind"]
        or manifest.get("status") != "complete"
        or manifest.get("weight_intervention") is not True
        or manifest.get("prompt_only") is not False
        or manifest.get("application_stage") != contract["application_stage"]
    ):
        raise PackageError("packaged intervention manifest is not a completed weight intervention")
    source = require_mapping(manifest.get("source"), "intervention.source")
    if any(source.get(key) != profile["source"].get(key) for key in (
        "repo_id", "revision", "snapshot_inventory_sha256"
    )):
        raise PackageError("packaged intervention manifest source differs from the release profile")
    if require_mapping(manifest.get("tooling"), "intervention.tooling") != evidence["tooling"]:
        raise PackageError("packaged intervention tooling differs from the build record")

    corpora = manifest.get("corpora")
    if not isinstance(corpora, list) or not corpora:
        raise PackageError("packaged intervention manifest lacks corpus evidence")
    for corpus in corpora:
        require_mapping(corpus, "intervention corpus")
        if (
            corpus.get("role") != "direction_extraction"
            or not isinstance(corpus.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", corpus["sha256"]) is None
            or not isinstance(corpus.get("record_count"), int)
            or isinstance(corpus.get("record_count"), bool)
            or corpus["record_count"] < 1
            or corpus.get("held_out_evaluation_overlap_count") != 0
        ):
            raise PackageError("packaged intervention corpus evidence is incomplete")

    directions = manifest.get("directions")
    if not isinstance(directions, list) or not directions:
        raise PackageError("packaged intervention manifest lacks inline direction evidence")
    direction_ids: set[str] = set()
    direction_dimensions: dict[str, int] = {}
    for direction in directions:
        require_mapping(direction, "intervention direction")
        values = direction.get("values")
        direction_id = direction.get("id")
        if (
            not isinstance(direction_id, str) or not direction_id or direction_id in direction_ids
            or direction.get("dtype") != "F32"
            or not isinstance(values, list) or not values
        ):
            raise PackageError("packaged intervention direction evidence is incomplete")
        direction_ids.add(direction_id)
        direction_dimensions[direction_id] = len(values)
        packed = bytearray()
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise PackageError("packaged intervention direction contains a non-finite value")
            try:
                packed.extend(struct.pack("<f", float(value)))
            except (OverflowError, struct.error) as exc:
                raise PackageError("packaged intervention direction value is outside F32 range") from exc
        if direction.get("sha256") != hashlib.sha256(packed).hexdigest():
            raise PackageError("packaged intervention direction digest is invalid")

    manifest_targets = manifest.get("targets")
    if not isinstance(manifest_targets, list) or [item.get("tensor_name") for item in manifest_targets if isinstance(item, dict)] != targets:
        raise PackageError("packaged intervention exact tensor targets differ from the build record")
    for target in manifest_targets:
        require_mapping(target, "intervention target")
        name = target.get("tensor_name")
        match = INTERVENTION_TARGET_RE.fullmatch(name) if isinstance(name, str) else None
        shape = target.get("expected_shape")
        scale = target.get("scale")
        if (
            match is None
            or target.get("direction_id") not in direction_ids
            or target.get("normalization") != "row_norm_preserve"
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in shape)
            or direction_dimensions.get(target.get("direction_id")) != shape[1]
            or isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale)) or float(scale) == 0.0
        ):
            raise PackageError("packaged intervention target evidence is incomplete")
        layer = int(match.group(1))
        writer = match.group(2)
        if layer > 47 or (
            (layer in QWEN_QSA_LAYERS and writer != "attn_output")
            or (layer not in QWEN_QSA_LAYERS and writer != "ssm_out")
        ):
            raise PackageError("packaged intervention target does not match the Qwen residual writer")
    tensor_map = require_mapping(manifest.get("tensor_map"), "intervention.tensor_map")
    if (
        tensor_map.get("kind") != "exact_tensor_names"
        or tensor_map.get("target_count") != target_count
        or tensor_map.get("target_names_sha256") != target_hash
    ):
        raise PackageError("packaged intervention tensor-map evidence is inconsistent")
    return destination, {
        "kind": evidence["kind"],
        "application_stage": evidence["application_stage"],
        "weight_intervention": True,
        "prompt_only": False,
        "manifest_filename": contract["manifest_filename"],
        "manifest_sha256": manifest_hash,
        "target_count": target_count,
        "target_names_sha256": target_hash,
        "tooling": evidence["tooling"],
        "quantizer_validated": True,
        "quantizer_applied": True,
        "metrics": metrics,
    }


def created_at(value: str | None) -> str:
    if value is None:
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PackageError("--created-at must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise PackageError("--created-at must include a timezone")
    return parsed.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def render_card(profile: dict[str, Any], manifest: dict[str, Any]) -> str:
    source = profile["source"]
    quantizer = profile["quantizer"]
    artifact = profile["artifact"]
    model_card = profile["model_card"]
    tags = "\n".join(f"- {tag}" for tag in model_card["tags"])
    external_baselines = json.dumps(
        profile.get("external_comparison_baselines", []), indent=2, sort_keys=True
    )
    artifact_rows = "\n".join(
        f"| `{item['filename']}` | {item.get('role', 'text_model')} | "
        f"{item['size_bytes']} | `{item['sha256']}` |"
        for item in [*manifest["artifacts"], *manifest["companion_artifacts"]]
    )
    return f"""---
license: {source['license']['id']}
license_name: {source['license']['name']}
license_link: {source['license']['url']}
base_model: {source['repo_id']}
base_model_relation: quantized
pipeline_tag: {source['pipeline_tag']}
library_name: {model_card['library_name']}
tags:
{tags}
---

# {artifact['repo_id'].split('/', 1)[-1]}

This is a candidate `{quantizer['format']}` GGUF derived from
`{source['repo_id']}@{source['revision']}` for AMD Strix Halo (`gfx1151`).
It is not a certification claim until the release gates and target-hardware
benchmark record are attached and approved.

This candidate is **not** a prompt-only jailbreak or a stock quant relabeled as
Heretic. Its weights were changed by the completed `directional_ablation`
manifest before ROCmI4 encoding. The exact inline F32 directions, extraction
corpus hashes, pinned tools, exact tensor map, scales, shapes, and normalization
policy are published in `qwen-intervention-manifest.json`. The quantizer's
preflight and application reports agree on manifest
`{manifest['intervention']['manifest_sha256']}` and all
`{manifest['intervention']['target_count']}` exact tensor targets; the same
evidence is embedded in every GGUF shard and retained in the build record.

## Artifact

| File | Role | Bytes | SHA-256 |
| --- | --- | ---: | --- |
{artifact_rows}

The BF16 `vision_mmproj` is a separate llama.cpp `--mmproj` artifact. Ember
loads its dynamic mtmd provider only on the first image request and opens the
text GGUF as a vocab-only view, so it does not duplicate the quantized text
tensor weights. The provider has passed GPU-free ABI and dependency checks;
real-weight image-text differential correctness and gfx1151 runtime behavior
remain uncertified.

## Provenance

| Input | Pinned revision |
| --- | --- |
| Base model | `{source['repo_id']}@{source['revision']}` |
| Qwen4Exp converter | `{profile['conversion']['repository']}@{profile['conversion']['revision']}` |
| Quantizer | `{quantizer['repository']}@{quantizer['revision']}` |
| OtherU intervention pipeline | `{profile['intervention']['otheru_pipeline']['repository']}@{profile['intervention']['otheru_pipeline']['revision']}` |
| Upstream Heretic | `{profile['intervention']['upstream_heretic']['repository']}@{profile['intervention']['upstream_heretic']['revision']}` |
| Ember | `{manifest['build']['engine_revision']}` |
| Build image | `{manifest['build']['container_image']}` |

`ROCmI4` names the `{quantizer['format']}` storage format. `IU4` names the
gfx1151 instruction path. The exact-dequant result and the optional W4A4 path
(`{quantizer['w4a4_cmake_option']}=ON`) are separate quality and performance
claims; W4A4 is off by default and must be certified independently.

The completed build record passed the native-262K dry-size gate: artifact bytes
plus the provisional 32 GiB non-artifact reserve fit within the 128 GiB UMA
budget. This is a packaging gate, not a measured peak-RSS result. The 1M YaRN
math oracle has passed, but real-weight target runtime is not certified and no
1M fit claim is made.

The release recipe preserves llama.cpp's ordered 48G split layout, validates
the complete global tensor inventory, and preflights the aggregate quantized
size before creating output shards. Each final shard is promoted with
transactional no-clobber handling. See the artifact-layout gate in
`artifact-manifest.json`.

## License

The upstream Qwen Community License 1.0 is copied into this candidate as
`LICENSE` and its pinned-source digest is recorded in `artifact-manifest.json`.
Review the license, including its separate commercial-use conditions, before
publication or service deployment.

See `artifact-manifest.json` and `SHA256SUMS` for machine-readable provenance
and integrity data. External framework results, when present, are comparison
baselines with their original workloads and offload settings; they are not
Ember throughput claims.

## External comparison recipes

The following settings are copied for reproducibility and comparison only. No
throughput or quality value from these recipes is an Ember result.

```json
{external_baselines}
```
"""


def build_package_in_stage(
    args: argparse.Namespace, out_dir: Path, final_out_dir: Path
) -> dict[str, Any]:
    profile_path = args.profile.resolve()
    source_artifact_paths = [path.resolve() for path in args.artifact]
    source_license_path = args.license.resolve()
    source_build_record_path = args.build_record.resolve()
    profile_snapshot = out_dir / "release-profile.json"
    copy_stable_file(profile_path, profile_snapshot)
    profile = load_profile(profile_snapshot)
    companion_contract = profile["artifact"]["required_companion_artifacts"][0]
    source_mmproj_path = (
        args.mmproj.absolute()
        if args.mmproj is not None
        else source_build_record_path.parent / companion_contract["filename"]
    )
    if not HEX40.fullmatch(args.engine_revision):
        raise PackageError("--engine-revision must be a lowercase 40-character commit")
    if not OCI_DIGEST.fullmatch(args.container_image):
        raise PackageError("--container-image must use an immutable @sha256:<64 hex> reference")
    timestamp = created_at(args.created_at)
    expected_license_hash = profile["source"]["license"]["sha256"]
    license_path = out_dir / "LICENSE"
    actual_license_hash = copy_stable_file(source_license_path, license_path)
    if actual_license_hash != expected_license_hash:
        raise PackageError(
            "license SHA-256 does not match the copy pinned by the release profile: "
            f"expected {expected_license_hash}, got {actual_license_hash}"
        )

    template = profile["release"]["candidate_revision_template"]
    revision = args.candidate_revision or template.format(
        source_revision_short=profile["source"]["revision"][:12],
        engine_revision_short=args.engine_revision[:12],
    )
    if not CANDIDATE_REVISION.fullmatch(revision):
        raise PackageError("candidate revision must start with candidate/ and use ref-safe characters")

    expected_name = profile["artifact"]["filename"]
    if len(source_artifact_paths) == 1:
        destination_names = [expected_name]
    else:
        stem = Path(expected_name).stem
        destination_names = [path.name for path in source_artifact_paths]
        pattern = re.compile(rf"^{re.escape(stem)}-([0-9]{{5}})-of-([0-9]{{5}})\.gguf$")
        matches = [pattern.fullmatch(name) for name in destination_names]
        counts = {int(match.group(2)) for match in matches if match is not None}
        numbers = [int(match.group(1)) for match in matches if match is not None]
        if any(match is None for match in matches) or counts != {len(source_artifact_paths)} or numbers != list(range(1, len(source_artifact_paths) + 1)):
            raise PackageError("multi-file artifacts must be a complete, ordered GGUF shard sequence")
    artifact_paths = [out_dir / destination for destination in destination_names]
    artifact_hashes = [
        copy_stable_file(source, destination)
        for source, destination in zip(source_artifact_paths, artifact_paths, strict=True)
    ]
    artifact_records = [
        {
            "filename": destination,
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }
        for path, destination, digest in zip(
            artifact_paths, destination_names, artifact_hashes, strict=True
        )
    ]
    mmproj_path = out_dir / companion_contract["filename"]
    mmproj_hash = copy_stable_file(source_mmproj_path, mmproj_path)
    mmproj_inspection = inspect_bf16_qwen_mmproj(mmproj_path)
    companion_records = [{
        "role": companion_contract["role"],
        "filename": companion_contract["filename"],
        "format": companion_contract["format"],
        "required_for": companion_contract["required_for"],
        "size_bytes": mmproj_path.stat().st_size,
        "sha256": mmproj_hash,
        "inspection": mmproj_inspection,
    }]
    build_record_path = out_dir / "qwen-quant-build-record.json"
    copy_stable_file(source_build_record_path, build_record_path)
    try:
        build_record = json.loads(build_record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read quantization build record: {exc}") from exc
    if not isinstance(build_record, dict) or build_record.get("status") != "complete":
        raise PackageError("quantization build record must have status complete")
    if (
        build_record.get("schema_version") != 1
        or build_record.get("mode") != "execute"
        or build_record.get("compute_mode") != "exact_dequant"
        or build_record.get("publishes") is not False
        or build_record.get("credentials_accessed") is not False
        or build_record.get("w4a4_enabled") is not False
    ):
        raise PackageError("quantization build record must be nonpublishing exact-dequant")
    recorded_profile = require_mapping(build_record.get("profile"), "build_record.profile")
    if recorded_profile.get("sha256") != sha256_file(profile_snapshot):
        raise PackageError("quantization build record profile SHA-256 does not match --profile")
    recorded_inventory = require_mapping(
        build_record.get("snapshot_inventory"), "build_record.snapshot_inventory"
    )
    if recorded_inventory.get("sha256") != profile["source"].get("snapshot_inventory_sha256"):
        raise PackageError("quantization build record snapshot inventory does not match the profile")
    recorded_snapshot = require_mapping(build_record.get("snapshot"), "build_record.snapshot")
    if recorded_snapshot.get("revision") != profile["source"]["revision"]:
        raise PackageError("quantization build record source revision does not match the profile")
    recorded_tools = require_mapping(build_record.get("tools"), "build_record.tools")
    expected_tool_revisions = {
        "ember_revision": args.engine_revision,
        "llama_cpp_revision": profile["conversion"]["revision"],
        "llama_cpp_base_revision": profile["conversion"]["base_revision"],
        "rocmfpx_revision": profile["quantizer"]["revision"],
    }
    if any(recorded_tools.get(key) != value for key, value in expected_tool_revisions.items()):
        raise PackageError(
            "quantization build record tool revisions do not match --engine-revision/profile"
        )
    build_info = require_mapping(
        recorded_tools.get("quantizer_build_info"),
        "build_record.tools.quantizer_build_info",
    )
    expected_build_info = {
        "tool": profile["quantization"]["tool"],
        "ember_revision": args.engine_revision,
        "rocmfpx_revision": profile["quantizer"]["revision"],
        "format": profile["quantization"]["format"],
        "ggml_tensor_type": profile["quantization"]["ggml_tensor_type"],
        "intervention_manifest_schema": profile["intervention"]["manifest_schema_version"],
    }
    if any(build_info.get(key) != value for key, value in expected_build_info.items()):
        raise PackageError("quantization build record quantizer build identity is inconsistent")
    if build_record.get("native_262k_memory_gate") != profile["quantization"]["native_262k_memory_gate"]:
        raise PackageError("quantization build record native-262K gate does not match the profile")
    build_resources = require_mapping(build_record.get("resources"), "build_record.resources")
    runner_requirements = profile["release"]["conversion_runner_requirements"]
    if (
        build_resources.get("minimum_free_gib", -1) < runner_requirements["minimum_free_disk_gib"]
        or build_resources.get("minimum_ram_gib", -1) < runner_requirements["minimum_physical_ram_gib"]
        or build_resources.get("free_disk_bytes", -1) < runner_requirements["minimum_free_disk_gib"] * 1024**3
        or build_resources.get("physical_ram_bytes", -1) < runner_requirements["minimum_physical_ram_gib"] * 1024**3
    ):
        raise PackageError("quantization build record does not satisfy release runner resource floors")
    recorded = build_record.get("output", {}).get("shards", [])
    recorded_triples = [
        (Path(item["path"]).name, item["size_bytes"], item["sha256"])
        for item in recorded
    ]
    artifact_triples = [
        (item["filename"], item["size_bytes"], item["sha256"])
        for item in artifact_records
    ]
    if recorded_triples != artifact_triples:
        raise PackageError("artifact files/sizes/hashes do not match the quantization build record")
    transaction = require_mapping(
        build_record.get("staging_transaction"), "build_record.staging_transaction"
    )
    committed_directory = source_build_record_path.parent.resolve()
    expected_promoted = [str(committed_directory / name) for name in destination_names]
    expected_evidence_promoted = [
        str(committed_directory / profile["intervention"]["manifest_filename"])
    ]
    if (
        transaction.get("boundary") != "atomic_directory"
        or transaction.get("commit_method") != "renameat2(RENAME_NOREPLACE)"
        or transaction.get("committed_directory") != str(committed_directory)
        or transaction.get("same_filesystem") is not True
        or transaction.get("verified_before_promotion") is not True
        or transaction.get("publication_state") != "committed_on_visibility"
        or transaction.get("promoted") != expected_promoted
        or transaction.get("evidence_promoted") != expected_evidence_promoted
        or any(path.parent.resolve() != committed_directory for path in source_artifact_paths)
    ):
        raise PackageError("quantization build record lacks committed-directory transaction evidence")
    intervention_path, intervention_evidence = snapshot_intervention_evidence(
        profile, build_record, source_build_record_path, out_dir
    )
    output_evidence = require_mapping(build_record.get("output"), "build_record.output")
    tensor_count = output_evidence.get("tensor_count")
    tensor_names_sha256 = output_evidence.get("tensor_names_sha256")
    tensor_type_counts = output_evidence.get("tensor_type_counts")
    if (
        not isinstance(tensor_count, int)
        or tensor_count < 1
        or not isinstance(tensor_names_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", tensor_names_sha256) is None
        or not isinstance(tensor_type_counts, dict)
        or not isinstance(tensor_type_counts.get("108"), int)
        or tensor_type_counts["108"] < 1
    ):
        raise PackageError("quantization build record lacks verified tensor inventory/type evidence")
    memory_gate = require_mapping(
        profile["quantization"].get("native_262k_memory_gate"),
        "quantization.native_262k_memory_gate",
    )
    memory_preflight = require_mapping(
        build_record.get("memory_preflight"), "build_record.memory_preflight"
    )
    artifact_bytes = sum(item["size_bytes"] for item in artifact_records)
    artifact_sizes = [item["size_bytes"] for item in artifact_records]
    budget_bytes = memory_gate.get("device_budget_bytes")
    reserve_bytes = memory_gate.get("runtime_reserve_bytes")
    if (
        memory_preflight.get("artifact_bytes") != artifact_bytes
        or memory_preflight.get("shard_count") != len(artifact_records)
        or memory_preflight.get("shard_bytes") != artifact_sizes
        or memory_preflight.get("budget_bytes") != budget_bytes
        or memory_preflight.get("runtime_reserve_bytes") != reserve_bytes
        or memory_preflight.get("total_bytes") != artifact_bytes + reserve_bytes
        or memory_preflight.get("headroom_bytes") != budget_bytes - artifact_bytes - reserve_bytes
        or memory_preflight.get("fits") is not True
        or artifact_bytes + reserve_bytes > budget_bytes
    ):
        raise PackageError("quantization build record does not satisfy the exact native-262K memory gate")
    if len(artifact_records) == 1:
        artifact_summary = {"kind": "single_file", **artifact_records[0]}
    else:
        artifact_summary = {
            "kind": "ordered_shard_set",
            "filename": None,
            "size_bytes": artifact_bytes,
            "sha256": None,
            "shard_count": len(artifact_records),
        }
    manifest = {
        "schema_version": 1,
        "created_at": timestamp,
        "profile_id": profile["profile_id"],
        "candidate": {
            "repo_id": profile["artifact"]["repo_id"],
            "repo_type": profile["artifact"]["repo_type"],
            "revision": revision,
            "published": False,
        },
        "source": profile["source"],
        "conversion": profile["conversion"],
        "quantizer": profile["quantizer"],
        "quantization": profile["quantization"],
        "intervention": intervention_evidence,
        "build": {
            "engine_revision": args.engine_revision,
            "container_image": args.container_image,
        },
        "artifact": artifact_summary,
        "artifacts": artifact_records,
        "companion_artifacts": companion_records,
        "license": {
            "filename": "LICENSE",
            "sha256": actual_license_hash,
            "copied_from": str(source_license_path),
        },
        "external_comparison_baselines": profile.get("external_comparison_baselines", []),
        "external_recipe_sources": profile.get("external_recipe_sources", {}),
        "external_quality_context": profile.get("external_quality_context", {}),
        "provisional_runtime_provenance": profile.get("provisional_runtime_provenance", []),
        "certification": {
            "status": "pending",
            "exact_dequant": "required",
            "w4a4_opt_in": "required_if_published_or_benchmarked",
            "vision": {
                "status": "pending_real_weight_gfx1151_differential",
                "gpu_free_provider_abi_smoke": "passed_at_image_build",
                "real_weight_differential_certified": False,
            },
            "publication_blockers": [],
        },
        "quantization_build_record": {
            "filename": "qwen-quant-build-record.json",
            "sha256": sha256_file(build_record_path),
        },
        "native_262k_memory_preflight": memory_preflight,
        "q1_correctness_memory": profile["release"]["q1_correctness_memory"],
        "conversion_runner_requirements": profile["release"]["conversion_runner_requirements"],
        "artifact_layout_gate": profile["release"]["artifact_layout_gate"],
    }

    readme_path = out_dir / "README.md"
    license_out = license_path
    manifest_path = out_dir / "artifact-manifest.json"
    checksums_path = out_dir / "SHA256SUMS"
    plan_path = out_dir / "upload-plan.json"
    build_record_out = build_record_path
    write_text(readme_path, render_card(profile, manifest))
    write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksums = [
        *((item["sha256"], item["filename"]) for item in artifact_records),
        *((item["sha256"], item["filename"]) for item in companion_records),
        (sha256_file(license_out), "LICENSE"),
        (sha256_file(readme_path), "README.md"),
        (sha256_file(manifest_path), "artifact-manifest.json"),
        (sha256_file(profile_snapshot), "release-profile.json"),
    ]
    checksums.append((sha256_file(build_record_out), "qwen-quant-build-record.json"))
    checksums.append((sha256_file(intervention_path), profile["intervention"]["manifest_filename"]))
    write_text(checksums_path, "".join(f"{digest}  {name}\n" for digest, name in checksums))

    upload_files = [
        *((path, destination) for path, destination in zip(artifact_paths, destination_names)),
        (mmproj_path, companion_contract["filename"]),
        (readme_path, "README.md"),
        (license_out, "LICENSE"),
        (manifest_path, "artifact-manifest.json"),
        (checksums_path, "SHA256SUMS"),
        (profile_snapshot, "release-profile.json"),
    ]
    upload_files.append((build_record_out, "qwen-quant-build-record.json"))
    upload_files.append((intervention_path, profile["intervention"]["manifest_filename"]))
    plan = {
        "schema_version": 1,
        "action": "candidate_upload_plan",
        "publishes": False,
        "repo_id": profile["artifact"]["repo_id"],
        "repo_type": profile["artifact"]["repo_type"],
        "revision": revision,
        "files": [
            {
                "local_path": str(final_out_dir / destination),
                "path_in_repo": destination,
                "size_bytes": local_path.stat().st_size,
                "sha256": sha256_file(local_path),
            }
            for local_path, destination in upload_files
        ],
        "authentication": {
            "preferred": "Hugging Face Trusted Publisher via GitHub OIDC",
            "fallback": "fine-grained HF_TOKEN with write access only to the target model repository",
            "token_embedded": False,
        },
        "promotion": {
            "allowed": False,
            "requires": "all documented release gates and exact candidate commit verification",
        },
        "publication_blockers": [],
    }
    write_text(plan_path, json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    final_out_dir = args.out_dir.absolute()
    try:
        os.lstat(final_out_dir)
    except FileNotFoundError:
        pass
    else:
        raise PackageError(f"refusing to overwrite existing package directory: {final_out_dir}")
    final_out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage: Path | None = Path(tempfile.mkdtemp(
        prefix=f".{final_out_dir.name}.transaction-", dir=final_out_dir.parent
    ))
    try:
        assert stage is not None
        plan = build_package_in_stage(args, stage, final_out_dir)
        for path in stage.iterdir():
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        fsync_directory(stage)
        rename_directory_noreplace(stage, final_out_dir)
        stage = None
        return plan
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True,
                        help="GGUF path; repeat in shard order for a split model")
    parser.add_argument(
        "--mmproj", type=Path,
        help=("separate BF16 vision GGUF; defaults to the required companion "
              "filename beside --build-record"),
    )
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--build-record", type=Path, required=True,
                        help="completed qwen_quantize.py build record to verify and include")
    parser.add_argument("--engine-revision", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-revision")
    parser.add_argument("--created-at")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        plan = build_package(args)
    except PackageError as exc:
        print(f"qwen_release_package.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "publishes": False,
        "repo_id": plan["repo_id"],
        "revision": plan["revision"],
        "upload_plan": str(args.out_dir.resolve() / "upload-plan.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
