#!/usr/bin/env python3
"""Plan or execute the pinned Qwen3.8 Flash Next ROCmI4 conversion.

Dry-run planning is the default.  ``--execute`` is the only path that invokes
the converter and quantizer.  The program uses no network client, constructs a
credential-free subprocess environment, and never publishes artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import datetime as dt
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from typing import Any, BinaryIO


GIB = 1024 ** 3
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<number>[0-9]{5})-of-(?P<count>[0-9]{5})\.gguf$")
GGUF_TYPES = {
    0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32",
    5: "INT32", 6: "FLOAT32", 7: "BOOL", 8: "STRING", 9: "ARRAY",
    10: "UINT64", 11: "INT64", 12: "FLOAT64",
}
FIXED_FORMATS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
    7: "?", 10: "Q", 11: "q", 12: "d",
}
PLE_SUFFIXES = {
    "ple_embedding.layer_multipliers": "qwen4exp.ple.layer_multipliers",
    "ple_embedding.ngram_heads_offsets": "qwen4exp.ple.head_offsets",
    "ple_embedding.ngram_heads_vocab_sizes": "qwen4exp.ple.head_vocab_sizes",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QWEN_QSA_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47}
INTERVENTION_TARGET_RE = re.compile(r"^blk\.([0-9]+)\.(attn_output|ssm_out)\.weight$")


class PipelineError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode("ascii"), usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"{label} must be a JSON object")
    return value


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineError(f"{field} must be a JSON object")
    return value


def validate_intervention_manifest(
    path: Path, profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate evidence for an actual pre-quantization weight intervention.

    The directions are inline so the exact bytes consumed by the quantizer are
    covered by the manifest digest.  Corpus rows need not be embedded, but a
    non-empty count and content digest make the extraction set identifiable.
    Tensor targets are exact names: allowing a pattern here would make the
    claimed intervention depend on a tool's regex dialect.
    """
    manifest = read_json(path, "intervention manifest")
    contract = require_mapping(profile.get("intervention"), "profile.intervention")
    if (
        contract.get("required") is not True
        or contract.get("kind") != "directional_ablation"
        or contract.get("manifest_schema_version") != 1
        or contract.get("application_stage") != "pre_quantization_encoding"
        or contract.get("weight_intervention_required") is not True
        or contract.get("prompt_only_allowed") is not False
        or contract.get("manifest_filename") != "qwen-intervention-manifest.json"
        or contract.get("quantizer_application_report_required") is not True
    ):
        raise PipelineError("release profile does not require the audited intervention contract")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "directional_ablation"
        or manifest.get("status") != "complete"
        or manifest.get("weight_intervention") is not True
        or manifest.get("prompt_only") is not False
        or manifest.get("application_stage") != "pre_quantization_encoding"
    ):
        raise PipelineError(
            "intervention manifest must describe a complete pre-quantization weight intervention, not prompt-only behavior"
        )

    source = require_mapping(manifest.get("source"), "intervention.source")
    expected_source = profile["source"]
    if any(source.get(field) != expected_source.get(field) for field in (
        "repo_id", "revision", "snapshot_inventory_sha256"
    )):
        raise PipelineError("intervention source evidence does not match the pinned base model")

    tooling = require_mapping(manifest.get("tooling"), "intervention.tooling")
    for manifest_key, profile_key in (
        ("otheru_quant_pipeline", "otheru_pipeline"),
        ("upstream_heretic", "upstream_heretic"),
    ):
        actual = require_mapping(tooling.get(manifest_key), f"intervention.tooling.{manifest_key}")
        expected = require_mapping(contract.get(profile_key), f"profile.intervention.{profile_key}")
        if actual != expected:
            raise PipelineError(f"intervention tooling evidence for {manifest_key} is not pinned to the release profile")

    corpora = manifest.get("corpora")
    if not isinstance(corpora, list) or not corpora:
        raise PipelineError("intervention.corpora must contain direction-extraction corpus evidence")
    corpus_ids: set[str] = set()
    corpus_records = 0
    for index, item in enumerate(corpora):
        corpus = require_mapping(item, f"intervention.corpora[{index}]")
        corpus_id = corpus.get("id")
        if not isinstance(corpus_id, str) or not corpus_id or corpus_id in corpus_ids:
            raise PipelineError("intervention corpus ids must be unique non-empty strings")
        corpus_ids.add(corpus_id)
        if corpus.get("role") != "direction_extraction":
            raise PipelineError("intervention corpus role must be direction_extraction")
        if not isinstance(corpus.get("sha256"), str) or not SHA256_RE.fullmatch(corpus["sha256"]):
            raise PipelineError("intervention corpus evidence requires a lowercase SHA-256")
        count = corpus.get("record_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise PipelineError("intervention corpus record_count must be positive")
        if corpus.get("held_out_evaluation_overlap_count") != 0:
            raise PipelineError("direction-extraction corpora must be disjoint from held-out evaluation")
        corpus_records += count

    directions = manifest.get("directions")
    if not isinstance(directions, list) or not directions:
        raise PipelineError("intervention.directions must contain inline F32 direction evidence")
    direction_ids: set[str] = set()
    direction_dimensions: dict[str, int] = {}
    for index, item in enumerate(directions):
        direction = require_mapping(item, f"intervention.directions[{index}]")
        direction_id = direction.get("id")
        if not isinstance(direction_id, str) or not direction_id or direction_id in direction_ids:
            raise PipelineError("intervention direction ids must be unique non-empty strings")
        direction_ids.add(direction_id)
        values = direction.get("values")
        if direction.get("dtype") != "F32" or not isinstance(values, list) or not values:
            raise PipelineError("intervention directions must contain non-empty inline F32 values")
        packed = bytearray()
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise PipelineError("intervention direction values must be finite numbers")
            try:
                packed.extend(struct.pack("<f", float(value)))
            except (OverflowError, struct.error) as exc:
                raise PipelineError("intervention direction value is outside F32 range") from exc
        digest = hashlib.sha256(packed).hexdigest()
        if direction.get("sha256") != digest:
            raise PipelineError("intervention direction SHA-256 does not match its packed little-endian F32 values")
        direction_dimensions[direction_id] = len(values)

    # The release model's ordinary Transformers hidden-state API mixes the
    # 10240-wide HC state with the final 2560-wide state and cannot be used to
    # construct these left-projection directions.  Require evidence from the
    # Qwen-specific mixed-input extractor.  Fixture profiles use another repo
    # id and intentionally retain small synthetic direction dimensions.
    if expected_source.get("repo_id") == "Qwen/Qwen3.8-Flash-Next":
        extraction = require_mapping(manifest.get("extraction"), "intervention.extraction")
        if (
            extraction.get("direction_scope") != "per_layer"
            or extraction.get("layer_count") != 48
            or extraction.get("activation_width") != 2560
            or extraction.get("semantic_capture_point")
            != "decoder_layer.attn_hyper_connection.mixed_input"
            or extraction.get("hidden_states_api_used") is not False
            or extraction.get("streaming") is not True
            or extraction.get("efficacy_evaluated") is not False
        ):
            raise PipelineError(
                "Qwen intervention extraction must use streamed 2560-wide per-layer HC mixed inputs"
            )
        expected_direction_ids = {
            f"layer-{layer:02d}-attn-mixed-input-r1" for layer in range(48)
        }
        if direction_ids != expected_direction_ids or any(
            direction_dimensions[direction_id] != 2560
            for direction_id in expected_direction_ids
        ):
            raise PipelineError("Qwen intervention must carry all 48 per-layer 2560-wide directions")
        classes = {corpus.get("class"): corpus for corpus in corpora}
        if set(classes) != {"good_control", "bad_target"}:
            raise PipelineError("Qwen intervention requires distinct good and bad extraction corpora")
        if (
            extraction.get("good_records_processed") != classes["good_control"].get("record_count")
            or extraction.get("bad_records_processed") != classes["bad_target"].get("record_count")
        ):
            raise PipelineError("Qwen intervention activation counts differ from corpus evidence")
        held_out = require_mapping(
            manifest.get("held_out_evaluation"), "intervention.held_out_evaluation"
        )
        if (
            not isinstance(held_out.get("sha256"), str)
            or SHA256_RE.fullmatch(held_out["sha256"]) is None
            or not isinstance(held_out.get("record_count"), int)
            or isinstance(held_out.get("record_count"), bool)
            or held_out["record_count"] < 1
            or held_out.get("overlap_count") != 0
            or held_out.get("comparison") != "canonical_text_chat_messages_sha256"
        ):
            raise PipelineError("Qwen intervention lacks disjoint held-out evaluation evidence")
        activation = require_mapping(
            extraction.get("activation_evidence"),
            "intervention.extraction.activation_evidence",
        )
        backend = activation.get("backend")
        if backend == "ember_qwen_runtime_f32_dump":
            if (
                activation.get("format") != "48x2560-little-endian-f32-records-v1"
                or activation.get("record_order") != "corpus_jsonl_order"
                or activation.get("artifact_sha256_verification")
                != "supplied_not_locally_rehashed"
                or any(
                    not isinstance(activation.get(field), str)
                    or SHA256_RE.fullmatch(activation[field]) is None
                    for field in (
                        "stock_rocmi4_artifact_sha256",
                        "good_dump_sha256",
                        "bad_dump_sha256",
                    )
                )
                or activation.get("good_dump_bytes")
                != extraction["good_records_processed"] * 48 * 2560 * 4
                or activation.get("bad_dump_bytes")
                != extraction["bad_records_processed"] * 48 * 2560 * 4
            ):
                raise PipelineError("Qwen runtime activation-dump evidence is incomplete")
        elif backend == "transformers_optional_reference":
            if activation.get("snapshot_revision") != expected_source.get("revision"):
                raise PipelineError("Qwen Transformers extraction did not use the pinned snapshot")
        else:
            raise PipelineError("Qwen intervention names an unsupported activation backend")

    targets = manifest.get("targets")
    if not isinstance(targets, list) or not targets:
        raise PipelineError("intervention.targets must contain at least one exact weight tensor")
    target_names: list[str] = []
    for index, item in enumerate(targets):
        target = require_mapping(item, f"intervention.targets[{index}]")
        name = target.get("tensor_name")
        match = INTERVENTION_TARGET_RE.fullmatch(name) if isinstance(name, str) else None
        if match is None or name in target_names:
            raise PipelineError(
                "intervention targets must be unique exact Qwen residual-writer tensor names"
            )
        layer = int(match.group(1))
        writer = match.group(2)
        if layer > 47 or (
            (layer in QWEN_QSA_LAYERS and writer != "attn_output")
            or (layer not in QWEN_QSA_LAYERS and writer != "ssm_out")
        ):
            raise PipelineError(
                f"intervention target {name} does not match the Qwen layer's residual writer"
            )
        target_names.append(name)
        direction_id = target.get("direction_id")
        if direction_id not in direction_ids:
            raise PipelineError(f"intervention target {name} refers to an unknown direction")
        scale = target.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) == 0.0:
            raise PipelineError(f"intervention target {name} requires a finite non-zero scale")
        if target.get("normalization") != "row_norm_preserve":
            raise PipelineError(f"intervention target {name} must use row_norm_preserve")
        shape = target.get("expected_shape")
        if (
            not isinstance(shape, list) or len(shape) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in shape)
            or direction_dimensions[str(direction_id)] != shape[1]
        ):
            raise PipelineError(
                f"intervention target {name} has invalid [ne0 columns, ne1 rows] direction/shape evidence"
            )
        if expected_source.get("repo_id") == "Qwen/Qwen3.8-Flash-Next" and (
            shape != [6144, 2560]
            or direction_id != f"layer-{layer:02d}-attn-mixed-input-r1"
        ):
            raise PipelineError(
                f"intervention target {name} must use its same-layer 2560-wide direction and [6144, 2560] GGUF shape"
            )

    tensor_map = require_mapping(manifest.get("tensor_map"), "intervention.tensor_map")
    if target_names != sorted(target_names):
        raise PipelineError("intervention targets must be lexicographically ordered")
    names_digest = hashlib.sha256("\n".join(sorted(target_names)).encode("utf-8")).hexdigest()
    if (
        tensor_map.get("kind") != "exact_tensor_names"
        or tensor_map.get("target_count") != len(target_names)
        or tensor_map.get("target_names_sha256") != names_digest
    ):
        raise PipelineError("intervention tensor-map evidence does not match the exact target list")

    manifest_hash = sha256_file(path)
    return manifest, {
        "manifest_filename": contract["manifest_filename"],
        "manifest_source_path": str(path),
        "manifest_sha256": manifest_hash,
        "kind": manifest["kind"],
        "application_stage": manifest["application_stage"],
        "weight_intervention": True,
        "prompt_only": False,
        "corpus_count": len(corpora),
        "corpus_record_count": corpus_records,
        "direction_count": len(directions),
        "target_count": len(target_names),
        "targets": target_names,
        "target_names_sha256": names_digest,
        "tooling": tooling,
    }


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def path_entry_exists(path: Path) -> bool:
    """Return true for every directory entry, including dangling symlinks."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def write_json_atomic(
    path: Path, value: dict[str, Any], *, create: bool,
) -> None:
    """Durably create or replace JSON inside an unpublished private directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if create:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise PipelineError(f"refusing to overwrite existing build record: {path}") from exc
        else:
            os.replace(temporary, path)
        fsync_directory(path.parent)
    except OSError as exc:
        raise PipelineError(f"cannot write build record {path}: {exc}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one same-filesystem directory without replacement."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise PipelineError("Linux renameat2 is required for atomic transaction publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    if renameat2(at_fdcwd, os.fsencode(source), at_fdcwd,
                 os.fsencode(destination), rename_noreplace) != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise PipelineError(f"refusing to overwrite existing transaction directory: {destination}")
        raise PipelineError(
            f"cannot atomically publish transaction directory {destination}: {os.strerror(error)}"
        )
    fsync_directory(destination.parent)


def sync_verified_outputs(paths: list[Path]) -> None:
    for path in paths:
        status = os.lstat(path)
        if not stat.S_ISREG(status.st_mode):
            raise PipelineError(f"staged GGUF is not a regular file: {path}")
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (status.st_dev, status.st_ino):
                raise PipelineError(f"staged GGUF identity changed before sync: {path}")
            os.fsync(stream.fileno())


def copy_verified_input(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy immutable evidence into the private transaction via a pinned fd."""
    try:
        named = os.lstat(source)
    except OSError as exc:
        raise PipelineError(f"cannot inspect intervention manifest {source}: {exc}") from exc
    if not stat.S_ISREG(named.st_mode):
        raise PipelineError(f"intervention manifest is not a regular file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            before = os.fstat(input_stream.fileno())
            if (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
                raise PipelineError("intervention manifest identity changed before transaction copy")
            for chunk in iter(lambda: input_stream.read(8 * 1024 * 1024), b""):
                output_stream.write(chunk)
                digest.update(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
            after = os.fstat(input_stream.fileno())
    except OSError as exc:
        raise PipelineError(f"cannot copy intervention manifest: {exc}") from exc
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise PipelineError("intervention manifest changed while copied into transaction")
    if digest.hexdigest() != expected_sha256:
        raise PipelineError("intervention manifest changed after validation")


def credential_free_env() -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
    }


def run_checked(
    command: list[str], cwd: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> str:
    environment = credential_free_env()
    if env_overrides:
        environment.update(env_overrides)
    result = subprocess.run(
        command, cwd=cwd, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise PipelineError(f"command failed ({result.returncode}): {shlex.join(command)}\n{detail}")
    return result.stdout.strip()


def run_checked_combined(command: list[str]) -> str:
    """Return stdout+stderr for tools such as llama.cpp that version on stderr."""
    result = subprocess.run(
        command, env=credential_free_env(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    combined = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise PipelineError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n{combined}"
        )
    return combined


def validate_profile(profile_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    profile = read_json(profile_path, "release profile")
    if profile.get("schema_version") != 1:
        raise PipelineError("unsupported release profile schema_version")
    for section in ("source", "conversion", "intervention", "quantizer", "quantization", "artifact", "release"):
        if not isinstance(profile.get(section), dict):
            raise PipelineError(f"profile section {section} is missing")
    source = profile["source"]
    conversion = profile["conversion"]
    quantizer = profile["quantizer"]
    quantization = profile["quantization"]
    intervention = profile["intervention"]
    for label, revision in (
        ("source", source.get("revision")),
        ("conversion base", conversion.get("base_revision")),
        ("conversion", conversion.get("revision")),
        ("quantizer", quantizer.get("revision")),
    ):
        if not isinstance(revision, str) or not HEX40.fullmatch(revision):
            raise PipelineError(f"{label} revision is not a pinned 40-character commit")
    if quantization.get("format") != "Q4_0_ROCMI4" or quantization.get("ggml_tensor_type") != 108:
        raise PipelineError("release profile must select Q4_0_ROCMI4 / GGML tensor type 108")
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
        raise PipelineError("release profile lacks the pinned Heretic weight-intervention contract")
    artifact_name = str(profile["artifact"].get("filename", ""))
    artifact_repo = str(profile["artifact"].get("repo_id", ""))
    if "Heretic" not in artifact_name or "Heretic" not in artifact_repo:
        raise PipelineError("required directional-ablation artifact must carry the Heretic release name")
    if quantization.get("w4a4_enabled") is not False or quantizer.get("w4a4_default") is not False:
        raise PipelineError("conversion pipeline is exact-dequant only; W4A4 must remain disabled")
    memory_gate = quantization.get("native_262k_memory_gate")
    if not isinstance(memory_gate, dict):
        raise PipelineError("quantization.native_262k_memory_gate is missing")
    if (
        memory_gate.get("native_context_tokens") != 262144
        or memory_gate.get("device_budget_bytes") != 137438953472
        or memory_gate.get("runtime_reserve_bytes") != 34359738368
        or memory_gate.get("certification_host_memtotal_bytes") != 134297894912
        or memory_gate.get("certification_host_rule")
        != "artifact_bytes + runtime_reserve_bytes + enabled_companion_artifact_bytes <= certification_host_memtotal_bytes"
        or memory_gate.get("companion_artifact_gate_status")
        != "pending_mtp_mmproj_inventory_and_measured_peak_rss_gtt"
        or memory_gate.get("rule") != "artifact_bytes + runtime_reserve_bytes <= device_budget_bytes"
        or memory_gate.get("yarn_1m_math_oracle_passed") is not True
        or memory_gate.get("yarn_1m_runtime_certified") is not False
        or memory_gate.get("yarn_1m_fit_claim") is not False
    ):
        raise PipelineError("native-262K memory gate does not match the audited 128 GiB contract")
    runner = profile["release"].get("conversion_runner_requirements")
    if (
        not isinstance(runner, dict)
        or runner.get("minimum_free_disk_gib") != 1024
        or runner.get("minimum_physical_ram_gib") != 256
        or runner.get("bounded_memory_minimum_physical_ram_gib") != 120
        or runner.get("bounded_memory_mode") != "llama_use_temp_file_then_gguf_split"
        or runner.get("bounded_memory_status") != "mechanically_bounded_unmeasured_on_target"
        or runner.get("certification_host_memtotal_bytes") != 134297894912
    ):
        raise PipelineError("release conversion runner must require 1 TiB disk and 256 GiB RAM")
    layout_gate = profile["release"].get("artifact_layout_gate")
    if (
        not isinstance(layout_gate, dict)
        or layout_gate.get("current_quantizer_multi_shard_supported") is not True
        or layout_gate.get("default_split_max_size") != "48G"
        or layout_gate.get("aggregate_preflight_before_output") is not True
        or layout_gate.get("transactional_no_clobber_shards") is not True
        or layout_gate.get("atomic_committed_work_directory") is not True
        or layout_gate.get("directory_commit_method") != "renameat2(RENAME_NOREPLACE)"
        or layout_gate.get("conditional_rollback_unlink") is not False
        or layout_gate.get("publication_blocked") is not False
    ):
        raise PipelineError("release multi-shard artifact-layout contract is missing")
    if profile["conversion"].get("default_split_max_size") != "48G":
        raise PipelineError("release conversion must default to 48G llama.cpp shards")
    inventory_path = profile_path.parent / str(source.get("snapshot_inventory", ""))
    expected_inventory_hash = source.get("snapshot_inventory_sha256")
    if not inventory_path.is_file() or sha256_file(inventory_path) != expected_inventory_hash:
        raise PipelineError("snapshot inventory file is missing or does not match its pinned SHA-256")
    inventory = read_json(inventory_path, "snapshot inventory")
    if inventory.get("repo_id") != source.get("repo_id") or inventory.get("revision") != source.get("revision"):
        raise PipelineError("snapshot inventory repository/revision differs from release profile")
    files = inventory.get("files")
    if not isinstance(files, list) or len(files) != inventory.get("file_count"):
        raise PipelineError("snapshot inventory file count is invalid")
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != inventory.get("inventory_sha256"):
        raise PipelineError("snapshot inventory canonical digest is invalid")
    weight_bytes = sum(row["size"] for row in files if str(row["path"]).endswith(".safetensors"))
    if weight_bytes != source.get("weight_bytes"):
        raise PipelineError("snapshot inventory weight bytes differ from the release profile")
    return profile, inventory, inventory_path


def validate_snapshot(snapshot: Path, revision: str, profile: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if revision != profile["source"]["revision"]:
        raise PipelineError("--snapshot-revision does not match the pinned Hugging Face revision")
    if not snapshot.is_dir():
        raise PipelineError(f"snapshot directory does not exist: {snapshot}")
    if snapshot.parent.name == "snapshots" and snapshot.name != revision:
        raise PipelineError("Hugging Face cache snapshot directory name does not match --snapshot-revision")
    actual_paths = sorted(
        path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()
    )
    expected_rows = inventory["files"]
    expected_paths = [row["path"] for row in expected_rows]
    if actual_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(actual_paths))[:10]
        extra = sorted(set(actual_paths) - set(expected_paths))[:10]
        raise PipelineError(f"snapshot file inventory mismatch; missing={missing}, extra={extra}")
    checked = []
    for row in expected_rows:
        rel = PurePosixPath(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise PipelineError(f"unsafe path in snapshot inventory: {row['path']}")
        path = snapshot / Path(*rel.parts)
        size = path.stat().st_size
        if size != row["size"]:
            raise PipelineError(f"snapshot size mismatch for {row['path']}: expected {row['size']}, got {size}")
        if "sha256" in row:
            actual_digest = sha256_file(path)
            if actual_digest != row["sha256"]:
                raise PipelineError(f"snapshot SHA-256 mismatch for {row['path']}")
            algorithm = "sha256"
        else:
            actual_digest = git_blob_sha1(path)
            if actual_digest != row["git_blob"]:
                raise PipelineError(f"snapshot Git blob mismatch for {row['path']}")
            algorithm = "git_blob_sha1"
        checked.append({"path": row["path"], "size_bytes": size, algorithm: actual_digest})
    license_path = snapshot / profile["source"]["license"]["path_in_source"]
    if sha256_file(license_path) != profile["source"]["license"]["sha256"]:
        raise PipelineError("snapshot license does not match the release profile")
    config = read_json(snapshot / "config.json", "snapshot config")
    architectures = config.get("architectures", [])
    if profile["source"]["architecture"] not in architectures:
        raise PipelineError("snapshot config architecture does not match the release profile")
    if config.get("model_type") != profile["source"]["model_type"]:
        raise PipelineError("snapshot config model_type does not match the release profile")
    return {"revision": revision, "files_verified": len(checked), "total_bytes": sum(x["size_bytes"] for x in checked)}


def git_revision(directory: Path) -> str:
    git = shutil.which("git", path=os.defpath)
    if git is None:
        raise PipelineError("git is required for source revision validation")
    if not directory.is_dir():
        raise PipelineError(f"source checkout does not exist: {directory}")
    return run_checked([git, "-C", str(directory), "rev-parse", "HEAD"])


def validate_tools(
    llama_dir: Path,
    rocmfpx_dir: Path,
    ember_dir: Path,
    ember_revision: str,
    quantizer_binary: Path,
    profile: dict[str, Any],
    gguf_splitter: Path | None = None,
) -> dict[str, Any]:
    if sys.version_info < (3, 10):
        raise PipelineError("Python 3.10 or newer is required by the pinned converter")
    conversion = profile["conversion"]
    quantizer = profile["quantizer"]
    llama_head = git_revision(llama_dir)
    rocmfpx_head = git_revision(rocmfpx_dir)
    ember_head = git_revision(ember_dir)
    if llama_head != conversion["revision"]:
        raise PipelineError(f"llama.cpp checkout must be at {conversion['revision']}, got {llama_head}")
    git = shutil.which("git", path=os.defpath)
    assert git is not None
    git_version = run_checked([git, "--version"])
    llama_parent = run_checked([git, "-C", str(llama_dir), "rev-parse", "HEAD^"])
    if llama_parent != conversion["base_revision"]:
        raise PipelineError("Qwen4Exp rotated-KV head is not directly based on the pinned PR #27742 head")
    if rocmfpx_head != quantizer["revision"]:
        raise PipelineError(f"ROCmFPX checkout must be at {quantizer['revision']}, got {rocmfpx_head}")
    if not HEX40.fullmatch(ember_revision) or ember_head != ember_revision:
        raise PipelineError(f"Ember checkout must be at requested revision {ember_revision}, got {ember_head}")
    for directory, label in ((llama_dir, "llama.cpp"), (rocmfpx_dir, "ROCmFPX"), (ember_dir, "Ember")):
        dirty = run_checked([git, "-C", str(directory), "status", "--porcelain", "--untracked-files=all"])
        if dirty:
            raise PipelineError(f"{label} checkout is dirty; pinned conversion requires a clean tree")
    converter = llama_dir / "convert_hf_to_gguf.py"
    qwen_converter = llama_dir / "conversion" / "qwen4exp.py"
    quantizer_source = rocmfpx_dir / "tools" / "quantize" / "quantize.cpp"
    for path in (converter, qwen_converter, quantizer_source):
        if not path.is_file():
            raise PipelineError(f"required tool source is missing: {path}")
    qwen_text = qwen_converter.read_text(encoding="utf-8")
    if "Qwen4ExpForConditionalGeneration" not in qwen_text or "_read_hash_constants" not in qwen_text:
        raise PipelineError("pinned converter lacks Qwen4Exp exact PLE metadata handling")
    quant_text = quantizer_source.read_text(encoding="utf-8")
    if "Q4_0_ROCMI4" not in quant_text or "arg_idx < argc && strncmp" not in quant_text:
        raise PipelineError("pinned ROCmFPX source lacks ROCMI4 or the audited option parser")
    quantizer_binary = quantizer_binary.resolve()
    if not quantizer_binary.is_file() or not os.access(quantizer_binary, os.X_OK):
        raise PipelineError(f"quantizer is not an executable file: {quantizer_binary}")
    try:
        build_info = json.loads(run_checked([str(quantizer_binary), "--build-info-json"]))
    except json.JSONDecodeError as exc:
        raise PipelineError("quantizer --build-info-json did not return a JSON object") from exc
    expected_info = {
        "tool": profile["quantization"]["tool"],
        "ember_revision": ember_revision,
        "rocmfpx_revision": quantizer["revision"],
        "format": profile["quantization"]["format"],
        "ggml_tensor_type": profile["quantization"]["ggml_tensor_type"],
        "intervention_manifest_schema": profile["intervention"]["manifest_schema_version"],
    }
    if not isinstance(build_info, dict) or any(build_info.get(key) != value for key, value in expected_info.items()):
        raise PipelineError(f"quantizer build provenance mismatch; required fields are {expected_info}")
    result = {
        "python": platform.python_version(),
        "git": git_version,
        "llama_cpp_revision": llama_head,
        "llama_cpp_base_revision": llama_parent,
        "converter_sha256": sha256_file(converter),
        "rocmfpx_revision": rocmfpx_head,
        "ember_revision": ember_head,
        "quantizer_binary": str(quantizer_binary),
        "quantizer_sha256": sha256_file(quantizer_binary),
        "quantizer_build_info": build_info,
        "composite_support": ["qwen4exp-converter", "architecture-agnostic-streaming", "Q4_0_ROCMI4"],
    }
    if gguf_splitter is not None:
        splitter = gguf_splitter.resolve()
        if not splitter.is_file() or not os.access(splitter, os.X_OK):
            raise PipelineError(f"GGUF splitter is not an executable file: {splitter}")
        if not splitter.is_relative_to(llama_dir.resolve()):
            raise PipelineError("GGUF splitter must come from the pinned llama.cpp checkout")
        version = run_checked_combined([str(splitter), "--version"])
        version_commit = re.search(r"\bcommit\s+([0-9a-f]{7,40})\b", version)
        if version_commit is None or not conversion["revision"].startswith(version_commit.group(1)):
            raise PipelineError("GGUF splitter build revision does not match pinned llama.cpp")
        result["gguf_splitter_binary"] = str(splitter)
        result["gguf_splitter_sha256"] = sha256_file(splitter)
        result["gguf_splitter_version"] = version
    return result


def physical_ram_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        raise PipelineError("cannot determine physical RAM on this host")


def existing_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists():
        if current.parent == current:
            raise PipelineError(f"cannot find an existing parent for {path}")
        current = current.parent
    return current


def validate_resources(work_dir: Path, minimum_free_gib: int, minimum_ram_gib: int) -> dict[str, int]:
    if minimum_free_gib < 0 or minimum_ram_gib < 0:
        raise PipelineError("resource minimums cannot be negative")
    free = shutil.disk_usage(existing_parent(work_dir)).free
    ram = physical_ram_bytes()
    if free < minimum_free_gib * GIB:
        raise PipelineError(f"insufficient free disk: need {minimum_free_gib} GiB, have {free / GIB:.1f} GiB")
    if ram < minimum_ram_gib * GIB:
        raise PipelineError(f"insufficient physical RAM: need {minimum_ram_gib} GiB, have {ram / GIB:.1f} GiB")
    return {"free_disk_bytes": free, "physical_ram_bytes": ram, "minimum_free_gib": minimum_free_gib, "minimum_ram_gib": minimum_ram_gib}


def read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise PipelineError("truncated GGUF structure")
    return value


def read_u32(stream: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(stream, 4))[0]


def read_u64(stream: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(stream, 8))[0]


def read_gguf_string(stream: BinaryIO) -> str:
    length = read_u64(stream)
    if length > 64 * 1024 * 1024:
        raise PipelineError("unreasonable GGUF string length")
    return read_exact(stream, length).decode("utf-8")


def read_gguf_value(stream: BinaryIO, value_type: int) -> tuple[Any, int | None]:
    if value_type in FIXED_FORMATS:
        fmt = "<" + FIXED_FORMATS[value_type]
        return struct.unpack(fmt, read_exact(stream, struct.calcsize(fmt)))[0], None
    if value_type == 8:
        return read_gguf_string(stream), None
    if value_type == 9:
        subtype = read_u32(stream)
        if subtype == 9 or subtype not in GGUF_TYPES:
            raise PipelineError(f"unsupported GGUF array subtype {subtype}")
        count = read_u64(stream)
        if count > 100_000_000:
            raise PipelineError("unreasonable GGUF array length")
        values = [read_gguf_value(stream, subtype)[0] for _ in range(count)]
        return values, subtype
    raise PipelineError(f"unknown GGUF metadata type {value_type}")


def inspect_gguf(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        if read_exact(stream, 4) != b"GGUF":
            raise PipelineError(f"not a GGUF file: {path}")
        version = read_u32(stream)
        if version not in (2, 3):
            raise PipelineError(f"unsupported GGUF version {version}: {path}")
        tensor_count = read_u64(stream)
        metadata_count = read_u64(stream)
        metadata: dict[str, dict[str, Any]] = {}
        for _ in range(metadata_count):
            key = read_gguf_string(stream)
            value_type = read_u32(stream)
            value, subtype = read_gguf_value(stream, value_type)
            metadata[key] = {"value": value, "type": value_type, "subtype": subtype}
        tensors = []
        for _ in range(tensor_count):
            name = read_gguf_string(stream)
            dimensions = read_u32(stream)
            shape = [read_u64(stream) for _ in range(dimensions)]
            tensor_type = read_u32(stream)
            offset = read_u64(stream)
            tensors.append({"name": name, "shape": shape, "type": tensor_type, "offset": offset})
    return {"path": str(path), "version": version, "metadata": metadata, "tensors": tensors}


def discover_gguf(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    matches = sorted(base.parent.glob(base.stem + "-?????-of-?????.gguf"))
    if not matches:
        raise PipelineError(f"GGUF output was not produced: {base}")
    parsed = [SHARD_RE.fullmatch(path.name) for path in matches]
    if any(match is None for match in parsed):
        raise PipelineError("invalid GGUF shard filename")
    counts = {int(match.group("count")) for match in parsed if match is not None}
    numbers = [int(match.group("number")) for match in parsed if match is not None]
    if len(counts) != 1 or numbers != list(range(1, len(matches) + 1)) or counts != {len(matches)}:
        raise PipelineError("GGUF shard sequence is incomplete")
    return matches


def expected_gguf_outputs(base: Path, shard_count: int) -> list[Path]:
    if shard_count < 1:
        raise PipelineError("GGUF output shard count must be positive")
    if shard_count == 1:
        return [base]
    return [
        base.with_name(f"{base.stem}-{index:05d}-of-{shard_count:05d}.gguf")
        for index in range(1, shard_count + 1)
    ]


def safetensor_ple_constants(snapshot: Path) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for path in sorted(snapshot.glob("model-*.safetensors")):
        with path.open("rb") as stream:
            header_length = struct.unpack("<Q", read_exact(stream, 8))[0]
            if header_length > 256 * 1024 * 1024:
                raise PipelineError(f"unreasonable safetensors header: {path}")
            header = json.loads(read_exact(stream, header_length).decode("utf-8"))
            data_start = 8 + header_length
            for name, info in header.items():
                if name == "__metadata__" or not isinstance(info, dict):
                    continue
                for suffix in PLE_SUFFIXES:
                    if not name.endswith(suffix):
                        continue
                    if info.get("dtype") != "I64":
                        raise PipelineError(f"PLE constant {name} must be I64 in the source snapshot")
                    start, end = info["data_offsets"]
                    if (end - start) % 8:
                        raise PipelineError(f"invalid I64 byte length for {name}")
                    stream.seek(data_start + start)
                    raw = read_exact(stream, end - start)
                    found[suffix] = list(struct.unpack("<" + "q" * ((end - start) // 8), raw))
    missing = sorted(set(PLE_SUFFIXES) - set(found))
    if missing:
        raise PipelineError(f"source snapshot lacks I64 PLE constants: {missing}")
    return found


def verify_gguf_set(
    paths: list[Path], expected_ple: dict[str, list[int]], *, quantized: bool,
    profile: dict[str, Any], intervention: dict[str, Any] | None = None,
    stock_control: bool = False,
) -> dict[str, Any]:
    inspected = [inspect_gguf(path) for path in paths]
    all_tensors = [tensor for item in inspected for tensor in item["tensors"]]
    names = [tensor["name"] for tensor in all_tensors]
    if len(names) != len(set(names)):
        raise PipelineError("GGUF tensor inventory contains duplicate names across shards")
    invariant_metadata: dict[str, dict[str, Any]] | None = None
    for shard_index, item in enumerate(inspected):
        metadata = item["metadata"]
        architecture = metadata.get("general.architecture", {}).get("value")
        if architecture != "qwen4exp":
            raise PipelineError(
                f"GGUF shard {shard_index + 1} architecture must be qwen4exp, got {architecture!r}"
            )
        for source_suffix, gguf_key in PLE_SUFFIXES.items():
            field = metadata.get(gguf_key)
            if field is None or field["type"] != 9 or field["subtype"] != 10:
                raise PipelineError(
                    f"GGUF shard {shard_index + 1} PLE metadata {gguf_key} must be ARRAY<UINT64>"
                )
            if field["value"] != expected_ple[source_suffix]:
                raise PipelineError(
                    f"GGUF shard {shard_index + 1} PLE metadata {gguf_key} does not match the source I64 values"
                )
        if quantized:
            if intervention is not None:
                intervention_fields = {
                    "ember.intervention.kind": (8, intervention["kind"]),
                    "ember.intervention.application_stage": (8, intervention["application_stage"]),
                    "ember.intervention.manifest_sha256": (8, intervention["manifest_sha256"]),
                    "ember.intervention.target_names_sha256": (8, intervention["target_names_sha256"]),
                    "ember.intervention.target_count": (4, intervention["target_count"]),
                }
                for key, (expected_type, expected_value) in intervention_fields.items():
                    field = metadata.get(key)
                    if field is None or field.get("type") != expected_type or field.get("value") != expected_value:
                        raise PipelineError(
                            f"GGUF shard {shard_index + 1} intervention metadata {key} does not match the applied manifest"
                        )
            elif stock_control:
                if any(key.startswith("ember.intervention.") for key in metadata):
                    raise PipelineError(
                        f"stock-control GGUF shard {shard_index + 1} carries intervention metadata"
                    )
            else:
                raise PipelineError(
                    "quantized GGUF verification requires intervention evidence or explicit stock-control mode"
                )
        # llama.cpp split files intentionally vary only split.no. Every other
        # metadata value is release-critical provenance/configuration and must
        # remain byte-semantically identical across the set.
        current_invariants = {
            key: value for key, value in metadata.items() if key != "split.no"
        }
        if invariant_metadata is None:
            invariant_metadata = current_invariants
        elif current_invariants != invariant_metadata:
            differing = sorted(
                key for key in set(invariant_metadata) | set(current_invariants)
                if invariant_metadata.get(key) != current_invariants.get(key)
            )
            raise PipelineError(
                f"GGUF shard {shard_index + 1} invariant metadata differs from shard 1: {differing}"
            )
    if len(paths) > 1:
        for index, item in enumerate(inspected):
            metadata = item["metadata"]
            if metadata.get("split.count", {}).get("value") != len(paths):
                raise PipelineError("GGUF split.count does not match discovered shard count")
            if metadata.get("split.no", {}).get("value") != index:
                raise PipelineError("GGUF split.no is not contiguous")
            if metadata.get("split.tensors.count", {}).get("value") != len(all_tensors):
                raise PipelineError("GGUF split.tensors.count does not match tensor inventory")
    type_counts = Counter(tensor["type"] for tensor in all_tensors)
    if quantized:
        ple_name = profile["quantization"]["ple_tensor_name"]
        ple = next((tensor for tensor in all_tensors if tensor["name"] == ple_name), None)
        expected_type = profile["quantization"]["ggml_tensor_type"]
        if ple is None or ple["type"] != expected_type:
            raise PipelineError(f"{ple_name} must use GGML tensor type {expected_type}")
        if type_counts[expected_type] == 0:
            raise PipelineError(f"quantized GGUF contains no tensor type {expected_type}")
    return {
        "shards": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        ],
        "tensor_count": len(all_tensors),
        "tensor_names_sha256": hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest(),
        "tensor_type_counts": {str(key): value for key, value in sorted(type_counts.items())},
    }


def planned_commands(
    args: argparse.Namespace, profile: dict[str, Any], work_dir: Path,
    intervention_manifest: Path | None,
) -> tuple[list[str], list[str] | None, list[str], list[str], Path, Path, Path | None]:
    intermediate = work_dir / "Qwen3.8-Flash-Next-BF16.gguf"
    unsplit = (
        work_dir / "Qwen3.8-Flash-Next-BF16.unsplit.gguf"
        if args.bounded_memory_temp and args.split_max_size != "0" else None
    )
    output_name = (
        "Qwen3.8-Flash-Next-Stock-Control-ROCmI4-Strix-Halo.gguf"
        if args.stock_control else profile["artifact"]["filename"]
    )
    output = work_dir / output_name
    converter = args.llama_cpp_dir.resolve() / "convert_hf_to_gguf.py"
    convert = [
        sys.executable, str(converter), str(args.snapshot_dir.resolve()),
        "--outfile", str(unsplit or intermediate), "--outtype", "bf16",
    ]
    split: list[str] | None = None
    if unsplit is not None:
        convert.append("--use-temp-file")
        assert args.gguf_splitter is not None
        split = [
            str(args.gguf_splitter.resolve()), "--split-max-size",
            args.split_max_size, str(unsplit), str(intermediate),
        ]
    elif args.split_max_size != "0":
        convert.extend(["--split-max-size", args.split_max_size])
    elif args.bounded_memory_temp:
        convert.append("--use-temp-file")
    quantize_options = [
        "--tensor-type", profile["quantization"]["ple_tensor_override"],
    ]
    if intervention_manifest is not None:
        quantize_options.extend(["--intervention-manifest", str(intervention_manifest)])
    if args.split_max_size != "0":
        quantize_options.append("--keep-split")
    memory_gate = profile["quantization"]["native_262k_memory_gate"]
    quantize_options.extend([
        "--device-budget-bytes", str(memory_gate["device_budget_bytes"]),
        "--runtime-reserve-bytes", str(memory_gate["runtime_reserve_bytes"]),
    ])
    # All options intentionally precede all positional arguments.  ROCmFPX's
    # parser stops scanning options at the first positional and otherwise drops
    # later flags without applying them.
    quantize = [
        str(args.quantizer.resolve()), *quantize_options,
        "<converted-first-shard>" if args.split_max_size != "0" else str(intermediate),
        "<private-same-filesystem-staging-output>",
        profile["quantization"]["format"], str(args.threads),
    ]
    preflight = [
        quantize[0], *quantize[1:-4], "--dry-size-json",
        quantize[-4], str(output), *quantize[-2:],
    ]
    return convert, split, preflight, quantize, intermediate, output, unsplit


def validate_memory_preflight(value: Any, profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineError("quantizer --dry-size-json did not return a JSON object")
    gate = profile["quantization"]["native_262k_memory_gate"]
    budget = gate["device_budget_bytes"]
    reserve = gate["runtime_reserve_bytes"]
    for key in (
        "artifact_bytes", "shard_count", "shard_bytes", "runtime_reserve_bytes",
        "budget_bytes", "total_bytes", "headroom_bytes", "fits",
    ):
        if key not in value:
            raise PipelineError(f"quantizer size preflight omitted {key}")
    if value["budget_bytes"] != budget or value["runtime_reserve_bytes"] != reserve:
        raise PipelineError("quantizer size preflight did not use the profile's exact budget and reserve")
    if not isinstance(value["artifact_bytes"], int) or value["artifact_bytes"] < 1:
        raise PipelineError("quantizer size preflight returned an invalid artifact size")
    if (
        not isinstance(value["shard_count"], int)
        or value["shard_count"] < 1
        or not isinstance(value["shard_bytes"], list)
        or len(value["shard_bytes"]) != value["shard_count"]
        or any(not isinstance(size, int) or size < 1 for size in value["shard_bytes"])
        or sum(value["shard_bytes"]) != value["artifact_bytes"]
    ):
        raise PipelineError("quantizer size preflight returned an invalid ordered shard inventory")
    if value["total_bytes"] != value["artifact_bytes"] + reserve:
        raise PipelineError("quantizer size preflight total is inconsistent")
    expected_fits = value["total_bytes"] <= budget
    if value["headroom_bytes"] != budget - value["total_bytes"]:
        raise PipelineError("quantizer size preflight headroom is inconsistent")
    if not isinstance(value["fits"], bool) or value["fits"] != expected_fits:
        raise PipelineError("quantizer size preflight fit result is inconsistent")
    if not expected_fits:
        raise PipelineError(
            "quantized artifact plus provisional non-artifact reserve does not fit the 128 GiB UMA budget"
        )
    certification_memtotal = gate["certification_host_memtotal_bytes"]
    certification_main_only_total = value["artifact_bytes"] + reserve
    if certification_main_only_total > certification_memtotal:
        raise PipelineError(
            "quantized main artifact plus reserve does not fit measured certification-host MemTotal"
        )
    result = dict(value)
    result["certification_host_memtotal_bytes"] = certification_memtotal
    result["certification_main_only_total_bytes"] = certification_main_only_total
    result["certification_main_only_headroom_bytes"] = (
        certification_memtotal - certification_main_only_total
    )
    result["companion_artifact_fit_status"] = "pending_mtp_mmproj_inventory"
    result["measured_peak_rss_gtt_status"] = "pending_target_run"
    return result


def validate_intervention_report(
    value: Any, evidence: dict[str, Any], *, applied: bool
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineError("quantizer intervention report is not a JSON object")
    if (
        value.get("intervention_manifest_sha256") != evidence["manifest_sha256"]
        or value.get("intervention_target_names_sha256") != evidence["target_names_sha256"]
        or value.get("intervention_target_count") != evidence["target_count"]
        or value.get("intervention_targets") != evidence["targets"]
        or value.get("intervention_validated") is not True
        or value.get("intervention_applied") is not applied
    ):
        stage = "application" if applied else "preflight"
        raise PipelineError(f"quantizer {stage} intervention evidence does not match the manifest")
    result = {
        "manifest_sha256": value["intervention_manifest_sha256"],
        "target_names_sha256": value["intervention_target_names_sha256"],
        "target_count": value["intervention_target_count"],
        "targets": value["intervention_targets"],
        "validated": True,
        "applied": applied,
    }
    if applied:
        metrics = value.get("intervention_metrics")
        if not isinstance(metrics, list) or len(metrics) != evidence["target_count"]:
            raise PipelineError("quantizer application omitted per-target intervention metrics")
        metric_fields = (
            "source_projection_l2", "stored_projection_l2",
            "stored_projection_ratio", "relative_frobenius_delta",
            "row_norm_relative_rmse", "row_norm_relative_max",
        )
        validated_metrics: list[dict[str, Any]] = []
        for expected_name, metric_value in zip(evidence["targets"], metrics, strict=True):
            metric = require_mapping(metric_value, "quantizer intervention metric")
            if metric.get("tensor_name") != expected_name:
                raise PipelineError("quantizer intervention metrics are not keyed to the exact target order")
            for field in metric_fields:
                number = metric.get(field)
                if (
                    isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(float(number)) or float(number) < 0.0
                ):
                    raise PipelineError(f"quantizer intervention metric {field} must be finite and non-negative")
            signed = metric.get("signed_projection_coefficient")
            if isinstance(signed, bool) or not isinstance(signed, (int, float)) or not math.isfinite(float(signed)):
                raise PipelineError("quantizer intervention signed projection coefficient must be finite")
            validated_metrics.append({
                "tensor_name": expected_name,
                **{field: metric[field] for field in metric_fields},
                "signed_projection_coefficient": signed,
            })
        result["metrics"] = validated_metrics
    elif "intervention_metrics" in value:
        raise PipelineError("quantizer preflight must not claim post-encoding intervention metrics")
    return result


def orchestrate(args: argparse.Namespace) -> dict[str, Any]:
    profile_path = args.profile.resolve()
    profile, inventory, inventory_path = validate_profile(profile_path)
    stock_control = bool(args.stock_control)
    intervention_source = (
        args.intervention_manifest.resolve()
        if args.intervention_manifest is not None else None
    )
    intervention: dict[str, Any] | None = None
    if intervention_source is not None:
        _intervention_manifest, intervention = validate_intervention_manifest(
            intervention_source, profile
        )
    final_work_dir = args.work_dir.parent.resolve() / args.work_dir.name
    runner = profile["release"]["conversion_runner_requirements"]
    minimum_ram_gib = args.min_ram_gib
    if minimum_ram_gib is None:
        minimum_ram_gib = (
            runner["bounded_memory_minimum_physical_ram_gib"]
            if args.bounded_memory_temp else runner["minimum_physical_ram_gib"]
        )
    resources = validate_resources(final_work_dir, args.min_free_gib, minimum_ram_gib)
    snapshot = validate_snapshot(args.snapshot_dir.resolve(), args.snapshot_revision, profile, inventory)
    tools = validate_tools(
        args.llama_cpp_dir.resolve(), args.rocmfpx_dir.resolve(), args.ember_dir.resolve(),
        args.ember_revision, args.quantizer.resolve(), profile,
        args.gguf_splitter.resolve() if args.gguf_splitter is not None else None,
    )
    expected_ple = safetensor_ple_constants(args.snapshot_dir.resolve())
    transaction_dir: Path | None = None
    if args.execute:
        requested_record = final_work_dir / "qwen-quant-build-record.json"
        if args.build_record is not None and args.build_record.absolute() != requested_record:
            raise PipelineError(
                "--build-record must name qwen-quant-build-record.json inside --work-dir in execute mode"
            )
        try:
            os.lstat(final_work_dir)
        except FileNotFoundError:
            pass
        else:
            raise PipelineError(
                f"refusing to overwrite existing transaction directory: {final_work_dir}"
            )
        final_work_dir.parent.mkdir(parents=True, exist_ok=True)
        transaction_dir = Path(tempfile.mkdtemp(
            prefix=f".{final_work_dir.name}.transaction-", dir=final_work_dir.parent
        ))
        work_dir = transaction_dir
        record_path = transaction_dir / "qwen-quant-build-record.json"
        if intervention_source is not None and intervention is not None:
            staged_manifest = transaction_dir / profile["intervention"]["manifest_filename"]
            try:
                copy_verified_input(
                    intervention_source, staged_manifest, intervention["manifest_sha256"]
                )
            except PipelineError:
                shutil.rmtree(transaction_dir, ignore_errors=True)
                transaction_dir = None
                raise
            intervention_command_path: Path | None = staged_manifest
        else:
            intervention_command_path = None
    else:
        work_dir = final_work_dir
        record_path = (
            args.build_record.absolute() if args.build_record
            else final_work_dir.with_name(final_work_dir.name + ".plan.json")
        )
        record_path.parent.mkdir(parents=True, exist_ok=True)
        if path_entry_exists(record_path):
            raise PipelineError(f"refusing to overwrite existing build record: {record_path}")
        intervention_command_path = intervention_source
    convert, split, preflight, quantize, intermediate, output, unsplit = planned_commands(
        args, profile, work_dir, intervention_command_path
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "profile": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
        "snapshot_inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        "snapshot": snapshot,
        "intervention": intervention,
        "experiment": {
            "kind": "stock_control" if stock_control else "directional_ablation",
            "stock_weights_unchanged": stock_control,
            "final_release_eligible": False,
            "eligibility_status": (
                "ineligible_stock_control" if stock_control
                else "pending_measured_bakeoff_and_hardware_certification"
            ),
            "purpose": (
                "activation_capture_and_bakeoff_baseline"
                if stock_control else "measured_bakeoff_candidate"
            ),
        },
        "tools": tools,
        "resources": resources,
        "mode": "execute" if args.execute else "dry-run",
        "publishes": False,
        "credentials_accessed": False,
        "compute_mode": "exact_dequant",
        "w4a4_enabled": False,
        "commands": {
            "convert": convert,
            "convert_shell": shlex.join(convert),
            "split": split,
            "split_shell": shlex.join(split) if split is not None else None,
            "quantize_preflight": preflight,
            "quantize_preflight_shell": shlex.join(preflight),
            "quantize": quantize,
            "quantize_shell": shlex.join(quantize),
            "quantizer_options_precede_positionals": True,
        },
        "ple": {
            "source_dtype": "I64",
            "gguf_metadata_dtype": "ARRAY<UINT64>",
            "keys": sorted(PLE_SUFFIXES.values()),
        },
        "native_262k_memory_gate": profile["quantization"]["native_262k_memory_gate"],
        "conversion_memory": {
            "mode": (
                "bounded_temp_file_then_split" if args.bounded_memory_temp
                else "ordinary_lazy_tensor_registry"
            ),
            "full_in_memory_tensor_registry": False if args.bounded_memory_temp else None,
            "temporary_directory": "private_transaction_directory" if args.bounded_memory_temp else None,
            "target_measurement_status": (
                "pending_peak_rss_and_wall_time" if args.bounded_memory_temp else "not_applicable"
            ),
        },
        "status": "planned",
    }
    if not args.execute:
        write_json_atomic(record_path, record, create=True)
        return record
    try:
        write_json_atomic(record_path, record, create=True)
        if args.bounded_memory_temp:
            converter_temp = work_dir / ".converter-tmp"
            converter_temp.mkdir(mode=0o700)
            run_checked(convert, env_overrides={"TMPDIR": str(converter_temp)})
            try:
                converter_temp.rmdir()
            except OSError as exc:
                raise PipelineError(f"converter left unexpected temporary files: {exc}") from exc
        else:
            run_checked(convert)
        if split is not None:
            assert unsplit is not None
            run_checked(split)
        intermediate_paths = discover_gguf(intermediate)
        record["intermediate"] = verify_gguf_set(intermediate_paths, expected_ple, quantized=False, profile=profile)
        if unsplit is not None:
            try:
                unsplit.unlink()
            except OSError as exc:
                raise PipelineError(f"cannot remove private unsplit BF16 intermediate: {exc}") from exc
        if "<converted-first-shard>" in quantize:
            first_shard = str(intermediate_paths[0])
            quantize[quantize.index("<converted-first-shard>")] = first_shard
            preflight[preflight.index("<converted-first-shard>")] = first_shard
        record["commands"]["quantize_preflight"] = preflight
        record["commands"]["quantize_preflight_shell"] = shlex.join(preflight)
        record["commands"]["quantize"] = quantize
        record["commands"]["quantize_shell"] = shlex.join(quantize)
        write_json_atomic(record_path, record, create=False)
        try:
            preflight_json = json.loads(run_checked(preflight))
        except json.JSONDecodeError as exc:
            raise PipelineError("quantizer --dry-size-json returned invalid JSON") from exc
        record["memory_preflight"] = validate_memory_preflight(preflight_json, profile)
        if intervention is not None:
            record["intervention"]["quantizer_preflight"] = validate_intervention_report(
                preflight_json, intervention, applied=False
            )
        write_json_atomic(record_path, record, create=False)
        staged_outputs = expected_gguf_outputs(
            output, record["memory_preflight"]["shard_count"]
        )
        quantize[quantize.index("<private-same-filesystem-staging-output>")] = str(output)
        record["commands"]["quantize"] = quantize
        record["commands"]["quantize_shell"] = shlex.join(quantize)
        write_json_atomic(record_path, record, create=False)
        quantize_stdout = run_checked(quantize)
        if intervention is not None:
            try:
                application_json = json.loads(quantize_stdout)
            except json.JSONDecodeError as exc:
                raise PipelineError("quantizer did not return intervention application JSON") from exc
            record["intervention"]["quantizer_application"] = validate_intervention_report(
                application_json, intervention, applied=True
            )
        elif quantize_stdout.strip():
            raise PipelineError("stock-control quantizer unexpectedly returned intervention evidence")
        staged_paths = discover_gguf(output)
        final = verify_gguf_set(
            staged_paths, expected_ple, quantized=True, profile=profile,
            intervention=intervention, stock_control=stock_control,
        )
        if final["tensor_names_sha256"] != record["intermediate"]["tensor_names_sha256"]:
            raise PipelineError("quantization changed the GGUF tensor inventory")
        final_sizes = [item["size_bytes"] for item in final["shards"]]
        if (
            len(final_sizes) != record["memory_preflight"]["shard_count"]
            or final_sizes != record["memory_preflight"]["shard_bytes"]
            or sum(final_sizes) != record["memory_preflight"]["artifact_bytes"]
        ):
            raise PipelineError("ordered quantized output shard sizes differ from the preflight")
        if staged_paths != staged_outputs:
            raise PipelineError("quantizer output paths differ from the preflight shard plan")
        sync_verified_outputs(staged_paths)
        final_outputs = [final_work_dir / path.name for path in staged_paths]
        for shard, path in zip(final["shards"], final_outputs, strict=True):
            shard["path"] = str(path)
        record["output"] = final
        record["staging_transaction"] = {
            "boundary": "atomic_directory",
            "commit_method": "renameat2(RENAME_NOREPLACE)",
            "committed_directory": str(final_work_dir),
            "same_filesystem": True,
            "verified_before_promotion": True,
            "publication_state": "committed_on_visibility",
            "promoted": [str(path) for path in final_outputs],
            "evidence_promoted": (
                [str(final_work_dir / profile["intervention"]["manifest_filename"])]
                if intervention is not None else []
            ),
        }
        record["status"] = "complete"
        try:
            for path in intermediate_paths:
                path.unlink()
        except OSError as exc:
            raise PipelineError(
                f"cannot remove private BF16 intermediate before commit: {exc}"
            ) from exc
        write_json_atomic(record_path, record, create=False)
        fsync_directory(work_dir)
        assert transaction_dir is not None
        rename_directory_noreplace(transaction_dir, final_work_dir)
        transaction_dir = None
    except PipelineError as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        try:
            write_json_atomic(record_path, record, create=False)
        except PipelineError:
            pass
        raise
    finally:
        if transaction_dir is not None:
            shutil.rmtree(transaction_dir, ignore_errors=True)
    return record


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_profile = Path(__file__).resolve().parents[1] / "share" / "release_profiles" / "qwen3.8-flash-next-rocmi4-strix-halo.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=default_profile)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--snapshot-revision", required=True)
    recipe = parser.add_mutually_exclusive_group(required=True)
    recipe.add_argument(
        "--intervention-manifest", type=Path,
        help="completed directional-ablation manifest applied by the quantizer",
    )
    recipe.add_argument(
        "--stock-control", action="store_true",
        help="leave weights unchanged; final-ineligible activation/bakeoff control only",
    )
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument("--rocmfpx-dir", type=Path, required=True)
    parser.add_argument("--ember-dir", type=Path, required=True)
    parser.add_argument("--ember-revision", required=True)
    parser.add_argument("--quantizer", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="execute mode atomically publishes this previously absent directory")
    parser.add_argument("--build-record", type=Path,
                        help="dry-run record path; execute mode only accepts WORK_DIR/qwen-quant-build-record.json")
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--split-max-size", default="48G", help="llama.cpp split size; release default is 48G, 0 writes one GGUF")
    parser.add_argument("--min-free-gib", type=int, default=1024)
    parser.add_argument(
        "--bounded-memory-temp", action="store_true",
        help="spill converter tensor payloads under WORK_DIR, then split the single BF16 GGUF",
    )
    parser.add_argument(
        "--gguf-splitter", type=Path,
        help="pinned llama-gguf-split executable; required by bounded mode with split output",
    )
    parser.add_argument(
        "--min-ram-gib", type=int,
        help="override the profile RAM floor (256 GiB ordinary, 120 GiB bounded)",
    )
    parser.add_argument("--execute", action="store_true", help="run conversion and quantization; default only writes a plan")
    return parser.parse_args(argv)


def reported_record_path(args: argparse.Namespace) -> Path:
    if args.execute:
        work_dir = args.work_dir.parent.resolve() / args.work_dir.name
        return work_dir / "qwen-quant-build-record.json"
    if args.build_record:
        return args.build_record.absolute()
    work_dir = args.work_dir.parent.resolve() / args.work_dir.name
    return work_dir.with_name(work_dir.name + ".plan.json")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        if args.threads < 1:
            raise PipelineError("--threads must be positive")
        if not re.fullmatch(r"0|[1-9][0-9]*[KMGT]", args.split_max_size):
            raise PipelineError("--split-max-size must be 0 or an integer followed by K, M, G, or T")
        if args.bounded_memory_temp and args.split_max_size != "0" and args.gguf_splitter is None:
            raise PipelineError("--bounded-memory-temp with split output requires --gguf-splitter")
        if not args.bounded_memory_temp and args.gguf_splitter is not None:
            raise PipelineError("--gguf-splitter is only valid with --bounded-memory-temp")
        record = orchestrate(args)
    except PipelineError as exc:
        print(f"qwen_quantize.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "mode": record["mode"], "publishes": False, "status": record["status"],
        "build_record": str(reported_record_path(args)),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
