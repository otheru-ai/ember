#!/usr/bin/env python3
"""Build and compare matched Qwen Q3/ROCMI4 hardware benchmark evidence.

The ordinary format bakeoff selects a complete main-model/MTP bundle.  This
helper answers the narrower question requested for the Q3 proof iteration:
what changed when only the main quant recipe changed from Q3 PLE to ROCMI4?
It rejects a comparison if the intervention, BF16 source, MTP companion,
runtime, exact workload, or measurement method differs.  The result is
descriptive only; sequential Q3-then-IU4 measurements do not replace the
counterbalanced format selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_bakeoff as bakeoff  # noqa: E402


CONTRACT_SCHEMA = "ember.qwen3.8.hardware-benchmark-contract.v1"
COMPARISON_SCHEMA = "ember.qwen3.8.q3-iu4-hardware-comparison.v1"
HARDWARE_SCHEMA = "ember.qwen3.8.real-weight-gate.v2"
KERNEL_RUNTIME_SCHEMA = "ember.qwen3.8.w4a8-dispatch-evidence.v1"
KERNEL_BUILD_SCHEMA = "ember.qwen3.8.w4a8-build-evidence.v1"
CONSTRUCTION_SCHEMA = "ember.qwen3.8.candidate-construction.v1"
COMPANION_SCHEMA = "ember.qwen3.8-flash-next.companion-inventory.v1"
Q3_ARM = "rocmfp4-fast-matrix-q3-ple-q6k-embedding-head"
IU4_ARM = "rocmi4-q6k-embedding-head"
MATCHED_MTP_CONTRACT = "Q4_0_ROCMFP4_FAST"
HEX64 = re.compile(r"[0-9a-f]{64}")


class ComparisonError(ValueError):
    pass


def canonical_sha256(value: Any) -> str:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_stable(path: Path, label: str) -> tuple[bytes, str]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ComparisonError(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ComparisonError(f"{label} is not one regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ComparisonError(f"cannot open {label}: {exc}") from exc
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        opened = os.fstat(fd)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                chunks.append(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                            row.st_mtime_ns, row.st_ctime_ns)
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise ComparisonError(f"{label} changed while it was hashed")
    return b"".join(chunks), digest.hexdigest()


def sha256_file(path: Path, label: str) -> str:
    return _read_stable(path, label)[1]


def exact_json(path: Path, expected: str, label: str) -> tuple[dict[str, Any], Path]:
    if not path.is_absolute() or HEX64.fullmatch(str(expected)) is None:
        raise ComparisonError(f"{label} path or digest is malformed")
    path = path.absolute()
    raw, digest = _read_stable(path, label)
    if digest != expected:
        raise ComparisonError(f"{label} SHA-256 differs")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"{label} must contain one JSON object")
    return value, path.resolve()


def exact_descriptor(value: Any, base: Path, label: str) -> tuple[dict[str, Any], Path]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ComparisonError(f"{label} descriptor is malformed")
    path = Path(str(value["path"]))
    if not path.is_absolute():
        path = base / path
    return exact_json(path, str(value["sha256"]), label)


def exact_relative_json(value: Any, base: Path,
                        label: str) -> tuple[dict[str, Any], Path]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ComparisonError(f"{label} descriptor is malformed")
    raw = Path(str(value["path"]))
    if raw.is_absolute():
        raise ComparisonError(f"{label} must be relative to its hardware evidence")
    root = base.resolve()
    path = (root / raw).resolve()
    if path != root and root not in path.parents:
        raise ComparisonError(f"{label} escapes its hardware evidence directory")
    return exact_json(path, str(value["sha256"]), label)


def write_new(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ComparisonError("output must be one new absolute path")
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _timing_rows(path: Path) -> list[dict[str, Any]]:
    try:
        raw, _ = _read_stable(path, "timing evidence")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()
                if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot parse timing evidence: {exc}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ComparisonError("timing evidence contains a non-object row")
    return rows


def derive_contract(timing: Path, benchmark_driver: Path,
                    gate_driver: Path) -> dict[str, Any]:
    if any(not path.is_absolute()
           for path in (timing, benchmark_driver, gate_driver)):
        raise ComparisonError("benchmark contract inputs must be absolute paths")
    timing_digest = sha256_file(timing, "timing evidence")
    try:
        facts = bakeoff.timing_facts(timing, True)
    except (OSError, ValueError) as exc:
        raise ComparisonError(f"timing evidence is not a complete Qwen hardware run: {exc}") from exc
    rows = _timing_rows(timing)
    if sha256_file(timing, "timing evidence") != timing_digest:
        raise ComparisonError("timing evidence changed while its contract was derived")
    metadata = [row for row in rows if row.get("kind") == "metadata"]
    summaries = [row for row in rows if row.get("kind") == "summary"]
    prefill = [row for row in rows if row.get("kind") == "request"
               and row.get("group") == "prefill-2048" and row.get("ok")]
    decode = [row for row in rows if row.get("kind") == "request"
              and row.get("group") == "decode-256" and row.get("ok")]
    if len(metadata) != 1 or len(summaries) != 1:
        raise ComparisonError("timing evidence lacks one metadata and one summary row")
    protocol = bakeoff.BENCHMARK_MODULE.QWEN_HARD_GATE_PROTOCOL
    if metadata[0].get("protocol") != protocol:
        raise ComparisonError("timing evidence used a different benchmark protocol")
    if [row.get("repeat") for row in prefill] != [1, 2, 3] or [
            row.get("repeat") for row in decode] != [1, 2, 3]:
        raise ComparisonError("timing evidence does not contain ordered repeats 1,2,3")
    prefill_calibration = summaries[0].get("prefill_calibration") or {}
    decode_calibration = summaries[0].get("decode_calibration") or {}
    words = prefill_calibration.get("selected_words")
    marker = decode_calibration.get("selected_marker")
    if (prefill_calibration.get("target_prompt_tokens") != 2074
            or isinstance(words, bool) or not isinstance(words, int) or words < 1
            or decode_calibration.get("target_completion_tokens") != 256
            or not isinstance(marker, str) or len(marker) != 1):
        raise ComparisonError("Qwen prompt-shape calibration is incomplete")
    prefill_prompts = [
        bakeoff.BENCHMARK_MODULE.make_prefill_prompt(item, words)
        for item in ("B", "C", "D")
    ]
    decode_prompt = bakeoff.BENCHMARK_MODULE.make_decode_prompt(marker)
    identity = {
        "protocol": protocol,
        "benchmark_driver_sha256": sha256_file(benchmark_driver, "benchmark driver"),
        "gate_driver_sha256": sha256_file(gate_driver, "hardware gate driver"),
        "prefill_generator": "benchmark.make_prefill_prompt.v1",
        "decode_generator": "benchmark.make_decode_prompt.v1",
        "prefill_calibration": {
            "target_prompt_tokens": 2074,
            "selected_words": words,
        },
        "decode_calibration": {
            "target_completion_tokens": 256,
            "selected_marker": marker,
        },
        "prefill_prompt_sha256": [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for prompt in prefill_prompts
        ],
        "decode_prompt_sha256": hashlib.sha256(
            decode_prompt.encode("utf-8")).hexdigest(),
        "evaluated_prefill_tokens": facts["evaluated_prefill_tokens"],
        "completion_tokens": facts["completion_tokens"],
        "samples_per_group": 3,
        "prefill_statistic": "peak",
        "decode_statistic": "median",
        "minimum_prefill_peak_tps": 412.0,
        "minimum_decode_median_tps": 39.49,
        "native_mtp_required": True,
        "clean_timing_unprofiled": True,
        "profile_and_counters_separate": True,
    }
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "complete",
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "publishes": False,
        "selection_allowed": False,
    }


def _construction(path: Path, digest: str, expected_arm: str,
                  label: str) -> dict[str, Any]:
    value, resolved = exact_json(path, digest, f"{label} construction")
    required = {
        "schema", "status", "publishes", "deletes", "candidate_id", "kind",
        "intended_stage", "row_id", "intervention_configuration_id",
        "quantization_arm", "mtp_matrix_quant_contract", "runtime_mode",
        "builder_revision", "runtime_revision", "images", "capture",
        "stock_capture", "bf16_cache", "shared_companions", "selection_plan",
        "build_record", "builder_attestation", "intervention_manifest",
        "artifacts", "v3_candidate_manifest",
    }
    if (set(value) != required or value.get("schema") != CONSTRUCTION_SCHEMA
            or value.get("status") != "complete" or value.get("publishes") is not False
            or value.get("deletes") is not False or value.get("kind") != "intervention"
            or value.get("intended_stage") != "format"
            or value.get("quantization_arm") != expected_arm
            or value.get("mtp_matrix_quant_contract") != MATCHED_MTP_CONTRACT
            or value.get("runtime_mode") != "exact_dequant"):
        raise ComparisonError(f"{label} construction is not the required exact format arm")
    build, build_path = exact_descriptor(
        value.get("build_record"), resolved.parent, f"{label} build record")
    intervention, _ = exact_descriptor(
        value.get("intervention_manifest"), resolved.parent,
        f"{label} intervention manifest")
    exact_descriptor(value.get("capture"), resolved.parent, f"{label} capture manifest")
    exact_descriptor(value.get("selection_plan"), resolved.parent,
                     f"{label} selection plan")
    exact_descriptor(value.get("bf16_cache"), resolved.parent,
                     f"{label} BF16 cache manifest")
    attestation, _ = exact_descriptor(
        value.get("builder_attestation"), resolved.parent,
        f"{label} builder attestation")
    companion_descriptors = value.get("shared_companions")
    if not isinstance(companion_descriptors, dict) or set(companion_descriptors) != {
            "Q4_0_ROCMI4", "Q4_0_ROCMFP4_FAST"}:
        raise ComparisonError(f"{label} shared companion inventory differs")
    companion, _ = exact_descriptor(
        companion_descriptors[MATCHED_MTP_CONTRACT], resolved.parent,
        f"{label} matched MTP companion inventory")
    mtp_rows = [row for row in companion.get("companions", [])
                if isinstance(row, dict) and row.get("role") == "mtp"
                and row.get("enabled") is True]
    if (companion.get("schema") != COMPANION_SCHEMA or len(mtp_rows) != 1
            or mtp_rows[0].get("matrix_quant_contract") != MATCHED_MTP_CONTRACT
            or HEX64.fullmatch(str(mtp_rows[0].get("sha256", ""))) is None
            or isinstance(mtp_rows[0].get("size_bytes"), bool)
            or not isinstance(mtp_rows[0].get("size_bytes"), int)
            or mtp_rows[0]["size_bytes"] < 1):
        raise ComparisonError(f"{label} lacks one exact matched FAST MTP")
    recipe = build.get("quantization_recipe") or {}
    bf16 = build.get("bf16_cache") or {}
    profile = build.get("profile") or {}
    build_intervention = build.get("intervention") or {}
    artifacts = value.get("artifacts") or {}
    rows = artifacts.get("shards")
    expected_formats = ({"Q3_0_ROCMFPX", "Q4_0_ROCMFP4_FAST", "Q6_K"}
                        if expected_arm == Q3_ARM else {"Q4_0_ROCMI4", "Q6_K"})
    format_contract = attestation.get("tensor_format_compatibility_sha256")
    builder_identity = attestation.get("builder_identity") or {}
    valid_rows = bool(
        isinstance(rows, list) and rows
        and all(isinstance(row, dict)
                and set(row) == {"path", "size_bytes", "sha256"}
                and isinstance(row["size_bytes"], int)
                and not isinstance(row["size_bytes"], bool)
                and row["size_bytes"] > 0
                and HEX64.fullmatch(str(row["sha256"])) is not None
                for row in rows))
    if (build.get("status") != "complete" or build.get("mode") != "execute"
            or build.get("compute_mode") != "exact_dequant"
            or build.get("w4a4_enabled") is not False
            or (build.get("tools") or {}).get("ember_revision") !=
               value.get("builder_revision")
            or recipe.get("id") != expected_arm
            or set(recipe.get("formats") or []) != expected_formats
            or recipe.get("selected_mtp_matrix_quant_contract") != MATCHED_MTP_CONTRACT
            or recipe.get("ple_override_preserved") is not True
            or build_intervention.get("manifest_sha256") !=
               value["intervention_manifest"]["sha256"]
            or not isinstance(bf16.get("cache_id"), str)
            or (bf16.get("manifest") or {}).get("sha256") !=
               value["bf16_cache"]["sha256"]
            or HEX64.fullmatch(str(profile.get("sha256", ""))) is None
            or attestation.get("schema") !=
               "ember.qwen3.8.candidate-workset-attestation.v1"
            or attestation.get("candidate_id") != value.get("candidate_id")
            or attestation.get("build_record_sha256") != value["build_record"]["sha256"]
            or builder_identity.get("ember_revision") != value.get("builder_revision")
            or builder_identity.get("tensor_format_contract_sha256") != format_contract
            or format_contract != ((value.get("images") or {}).get("runtime") or {}).get(
                "tensor_format_contract_sha256")
            or not valid_rows
            or rows != (build.get("output") or {}).get("shards")
            or sum(row["size_bytes"] for row in rows if isinstance(row, dict)) !=
               artifacts.get("total_bytes")):
        raise ComparisonError(f"{label} build/source/intervention contract differs")
    return {
        "descriptor": value,
        "descriptor_path": str(resolved),
        "descriptor_sha256": digest,
        "build": build,
        "build_path": str(build_path),
        "intervention": intervention,
        "mtp": mtp_rows[0],
    }


def _exact_path_descriptor(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ComparisonError(f"{label} descriptor is malformed")
    path = Path(str(value["path"]))
    if path.is_absolute():
        raise ComparisonError(f"{label} must be relative to its hardware evidence")
    root = base.resolve()
    path = (root / path).resolve()
    if path != root and root not in path.parents:
        raise ComparisonError(f"{label} escapes its hardware evidence directory")
    expected = str(value["sha256"])
    if HEX64.fullmatch(expected) is None or sha256_file(path, label) != expected:
        raise ComparisonError(f"{label} SHA-256 differs")
    return path.resolve()


def validate_hardware(path: Path, digest: str, construction: dict[str, Any],
                      label: str) -> dict[str, Any]:
    value, resolved = exact_json(path, digest, f"{label} hardware evidence")
    descriptor = construction["descriptor"]
    runtime = (descriptor.get("images") or {}).get("runtime") or {}
    evidence = value.get("evidence") or {}
    contract, _ = exact_relative_json(
        evidence.get("benchmark_contract"), resolved.parent,
        f"{label} benchmark contract")
    kernel_runtime, _ = exact_relative_json(
        evidence.get("kernel_runtime"), resolved.parent,
        f"{label} kernel runtime evidence")
    kernel_build, _ = exact_relative_json(
        evidence.get("kernel_build"), resolved.parent,
        f"{label} kernel build evidence")
    timing_mode_record, _ = exact_relative_json(
        evidence.get("timing_kernel_mode"), resolved.parent,
        f"{label} timing kernel mode evidence")
    timing_path = _exact_path_descriptor(
        evidence.get("timing"), resolved.parent, f"{label} clean timing")
    timing_digest = str((evidence.get("timing") or {}).get("sha256"))
    try:
        facts = bakeoff.timing_facts(timing_path, True)
    except (OSError, ValueError) as exc:
        raise ComparisonError(f"{label} timing evidence differs: {exc}") from exc
    identity = contract.get("identity")
    identity_keys = {
        "protocol", "benchmark_driver_sha256", "gate_driver_sha256",
        "prefill_generator", "decode_generator", "prefill_calibration",
        "decode_calibration", "prefill_prompt_sha256", "decode_prompt_sha256",
        "evaluated_prefill_tokens", "completion_tokens", "samples_per_group",
        "prefill_statistic", "decode_statistic", "minimum_prefill_peak_tps",
        "minimum_decode_median_tps", "native_mtp_required",
        "clean_timing_unprofiled", "profile_and_counters_separate",
    }
    if (contract.get("schema") != CONTRACT_SCHEMA or contract.get("status") != "complete"
            or contract.get("publishes") is not False
            or contract.get("selection_allowed") is not False
            or not isinstance(identity, dict) or set(identity) != identity_keys
            or contract.get("identity_sha256") != canonical_sha256(identity)
            or value.get("benchmark_contract") != contract
            or any(HEX64.fullmatch(str(identity.get(key, ""))) is None
                   for key in ("benchmark_driver_sha256", "gate_driver_sha256",
                               "decode_prompt_sha256"))
            or not isinstance(identity.get("prefill_prompt_sha256"), list)
            or len(identity["prefill_prompt_sha256"]) != 3
            or any(HEX64.fullmatch(str(item)) is None
                   for item in identity["prefill_prompt_sha256"])
            or identity.get("samples_per_group") != 3
            or identity.get("prefill_statistic") != "peak"
            or identity.get("decode_statistic") != "median"
            or identity.get("minimum_prefill_peak_tps") != 412.0
            or identity.get("minimum_decode_median_tps") != 39.49
            or identity.get("native_mtp_required") is not True
            or identity.get("clean_timing_unprofiled") is not True
            or identity.get("profile_and_counters_separate") is not True):
        raise ComparisonError(f"{label} benchmark contract is not exact")
    rows = _timing_rows(timing_path)
    if sha256_file(timing_path, f"{label} clean timing") != timing_digest:
        raise ComparisonError(f"{label} timing changed while it was validated")
    metadata = next(row for row in rows if row.get("kind") == "metadata")
    summary = next(row for row in rows if row.get("kind") == "summary")
    if (identity.get("protocol") != metadata.get("protocol")
            or identity.get("prefill_calibration") != {
                "target_prompt_tokens": 2074,
                "selected_words": (summary.get("prefill_calibration") or {}).get(
                    "selected_words"),
            }
            or identity.get("decode_calibration") != {
                "target_completion_tokens": 256,
                "selected_marker": (summary.get("decode_calibration") or {}).get(
                    "selected_marker"),
            }
            or identity.get("evaluated_prefill_tokens") !=
               facts["evaluated_prefill_tokens"]
            or identity.get("completion_tokens") != facts["completion_tokens"]):
        raise ComparisonError(f"{label} timing rows differ from their benchmark contract")
    hardware_model = (value.get("model") or {}).get("ordered_inventory") or {}
    hardware_rows = hardware_model.get("shards")
    construction_rows = (descriptor.get("artifacts") or {}).get("shards")
    def identities(rows_value: Any) -> list[dict[str, Any]]:
        if not isinstance(rows_value, list):
            return []
        return [{"sha256": row.get("sha256"), "size_bytes": row.get("size_bytes")}
                for row in rows_value if isinstance(row, dict)]
    profile = value.get("profile_image") or {}
    hardware_mtp = value.get("mtp") or {}
    performance = (value.get("hard_gates") or {}).get("performance")
    memory = (value.get("hard_gates") or {}).get("memory")
    timing_mode = value.get("timing_kernel_mode") or {}
    kernel_capability = kernel_runtime.get("candidate_kernel_capability")
    if (value.get("schema") != HARDWARE_SCHEMA or value.get("publish_approved") is not False
            or value.get("methodology") != "clean timing and profiler/counter passes are separate"
            or value.get("image") != {
                "ref": runtime.get("release_ref"), "digest": runtime.get("release_digest")}
            or profile.get("ref") != runtime.get("dev_ref")
            or profile.get("digest") != runtime.get("dev_digest")
            or profile.get("ember_revision") != descriptor.get("runtime_revision")
            or profile.get("candidate_binary_byte_identical") is not True
            or (hardware_model.get("build_record") or {}).get("sha256") !=
               descriptor["build_record"]["sha256"]
            or identities(hardware_rows) != identities(construction_rows)
            or hardware_mtp.get("sha256") != construction["mtp"]["sha256"]
            or hardware_mtp.get("depth") != 3
            or performance != facts["hard_gate"]
            or memory != facts["memory_gate"]
            or value.get("resources") != facts["resources"]
            or value.get("speculation") != facts["mtp_speculation"]
            or value.get("kernel_runtime") != kernel_runtime
            or value.get("kernel_build") != kernel_build
            or timing_mode != timing_mode_record
            or kernel_runtime.get("schema") != KERNEL_RUNTIME_SCHEMA
            or kernel_runtime.get("passed") is not True
            or kernel_runtime.get("candidate_timing_kernel_mode") !=
               timing_mode.get("configured_mmq_mode")
            or kernel_build.get("schema") != KERNEL_BUILD_SCHEMA
            or (kernel_capability != "no_eligible_rocmi4_mmq"
                and (kernel_build.get("saved_isa_gate") or {}).get("passed") is not True)
            or timing_mode.get("passed") is not True
            or timing_mode.get("configured_mmq_mode") == "lossy_w4a4_mmq"
            or kernel_build.get("candidate_and_profiler_binary_sha256") !=
               profile.get("candidate_binary_sha256")):
        raise ComparisonError(f"{label} hardware evidence is not bound to its candidate/runtime")
    return {
        "value": value,
        "path": str(resolved),
        "sha256": digest,
        "contract": contract,
        "facts": facts,
        "artifact_bytes": sum(row["size_bytes"] for row in construction_rows),
        "runtime_identity": {
            "runtime_revision": descriptor["runtime_revision"],
            "release_image": runtime["release_ref"],
            "release_digest": runtime["release_digest"],
            "dev_image": runtime["dev_ref"],
            "dev_digest": runtime["dev_digest"],
            "engine_binary_sha256": profile["candidate_binary_sha256"],
            "tensor_format_contract_sha256": runtime[
                "tensor_format_contract_sha256"],
        },
    }


def _finite(value: Any, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise ComparisonError(f"{label} is not finite")
    return float(value)


def make_comparison(q3_construction: dict[str, Any], q3_hardware: dict[str, Any],
                    iu4_construction: dict[str, Any], iu4_hardware: dict[str, Any]) -> dict[str, Any]:
    q3_desc = q3_construction["descriptor"]
    iu4_desc = iu4_construction["descriptor"]
    q3_build = q3_construction["build"]
    iu4_build = iu4_construction["build"]
    source_identity = lambda desc, build: {
        "bf16_cache_id": (build.get("bf16_cache") or {}).get("cache_id"),
        "bf16_cache_manifest_sha256": ((build.get("bf16_cache") or {}).get(
            "manifest") or {}).get("sha256"),
        "profile_sha256": (build.get("profile") or {}).get("sha256"),
        "selection_plan_sha256": (desc.get("selection_plan") or {}).get("sha256"),
        "capture_sha256": (desc.get("capture") or {}).get("sha256"),
        "intervention_configuration_id": desc.get("intervention_configuration_id"),
        "intervention_manifest_sha256": (desc.get("intervention_manifest") or {}).get(
            "sha256"),
    }
    q3_source = source_identity(q3_desc, q3_build)
    iu4_source = source_identity(iu4_desc, iu4_build)
    q3_mtp = {key: q3_construction["mtp"].get(key) for key in (
        "sha256", "size_bytes", "matrix_quant_contract")}
    iu4_mtp = {key: iu4_construction["mtp"].get(key) for key in (
        "sha256", "size_bytes", "matrix_quant_contract")}
    if q3_source != iu4_source:
        raise ComparisonError("Q3 and IU4 do not share one BF16/intervention source identity")
    if q3_mtp != iu4_mtp:
        raise ComparisonError("Q3 and IU4 do not use one exact FAST MTP companion")
    if q3_hardware["runtime_identity"] != iu4_hardware["runtime_identity"]:
        raise ComparisonError("Q3 and IU4 do not use one exact runtime engine/image")
    if q3_hardware["contract"] != iu4_hardware["contract"]:
        raise ComparisonError("Q3 and IU4 do not use one exact benchmark workload contract")
    if q3_desc.get("candidate_id") == iu4_desc.get("candidate_id"):
        raise ComparisonError("Q3 and IU4 candidate identities must be distinct")
    q3_kernel = (q3_hardware["value"].get("kernel_runtime") or {}).get(
        "candidate_kernel_capability")
    q3_mode = (q3_hardware["value"].get("timing_kernel_mode") or {}).get(
        "configured_mmq_mode")
    iu4_kernel = (iu4_hardware["value"].get("kernel_runtime") or {}).get(
        "candidate_kernel_capability")
    iu4_mode = (iu4_hardware["value"].get("timing_kernel_mode") or {}).get(
        "configured_mmq_mode")
    if (q3_kernel != "no_eligible_rocmi4_mmq"
            or q3_mode != "not_applicable_no_eligible_rocmi4_mmq"):
        raise ComparisonError("Q3 arm did not use its exact no-ROCMI4 runtime mode")
    if (iu4_kernel != "rocmi4_dense_and_routed"
            or iu4_mode not in {"w4a8_iu4_register_pack", "w4a8_iu4_prepack"}):
        raise ComparisonError("IU4 arm did not use one exact W4A8 IU4 runtime mode")
    q3_facts = q3_hardware["facts"]
    iu4_facts = iu4_hardware["facts"]

    def metrics(hardware: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
        value = hardware["value"]
        prefill = [_finite(item, "prefill sample") for item in facts["prefill_tps_samples"]]
        decode = [_finite(item, "decode sample") for item in facts["decode_tps_samples"]]
        resources = facts["resources"]
        return {
            "prefill_tps_samples": prefill,
            "prefill_peak_tps": max(prefill),
            "prefill_median_tps": statistics.median(prefill),
            "decode_tps_samples": decode,
            "decode_median_tps": statistics.median(decode),
            "artifact_bytes": hardware["artifact_bytes"],
            "measured_peak_rss_bytes": resources.get("measured_peak_rss_bytes"),
            "measured_peak_gtt_bytes": resources.get("measured_peak_gtt_bytes"),
            "measured_peak_uma_bytes": resources.get("measured_peak_uma_bytes"),
            "hard_gate_passed": bool((value.get("hard_gates") or {}).get(
                "performance", {}).get("passed") and
                (value.get("hard_gates") or {}).get("memory", {}).get("passed")),
            "kernel_capability": (value.get("kernel_runtime") or {}).get(
                "candidate_kernel_capability"),
            "timing_kernel_mode": (value.get("timing_kernel_mode") or {}).get(
                "configured_mmq_mode"),
        }

    q3_metrics = metrics(q3_hardware, q3_facts)
    iu4_metrics = metrics(iu4_hardware, iu4_facts)
    delta_fields = (
        "prefill_peak_tps", "prefill_median_tps", "decode_median_tps",
        "artifact_bytes", "measured_peak_rss_bytes", "measured_peak_gtt_bytes",
        "measured_peak_uma_bytes",
    )
    deltas: dict[str, Any] = {}
    for field in delta_fields:
        left = q3_metrics.get(field)
        right = iu4_metrics.get(field)
        if (isinstance(left, bool) or isinstance(right, bool)
                or not isinstance(left, (int, float))
                or not isinstance(right, (int, float))):
            raise ComparisonError(f"matched comparison lacks numeric {field}")
        delta = right - left
        deltas[f"iu4_minus_q3_{field}"] = delta
        deltas[f"iu4_vs_q3_{field}_percent"] = (
            delta * 100.0 / left if left != 0 else None)
    matched = {
        "source": q3_source,
        "mtp": {**q3_mtp, "depth": 3},
        "runtime": q3_hardware["runtime_identity"],
        "benchmark_contract_sha256": q3_hardware["contract"]["identity_sha256"],
    }
    return {
        "schema": COMPARISON_SCHEMA,
        "status": "complete",
        "comparison": "Q3_0_ROCMFPX_PLE_main_vs_Q4_0_ROCMI4_main",
        "matched_identity": matched,
        "allowed_differences": [
            "candidate_id", "main_quantization_arm", "main_model_inventory",
            "main_artifact_bytes", "exact_runtime_kernel_capability_and_mode",
        ],
        "arms": {
            "q3": {
                "candidate_id": q3_desc["candidate_id"],
                "quantization_arm": q3_desc["quantization_arm"],
                "construction": {"path": q3_construction["descriptor_path"],
                                 "sha256": q3_construction["descriptor_sha256"]},
                "hardware": {"path": q3_hardware["path"],
                             "sha256": q3_hardware["sha256"]},
                "metrics": q3_metrics,
            },
            "iu4": {
                "candidate_id": iu4_desc["candidate_id"],
                "quantization_arm": iu4_desc["quantization_arm"],
                "construction": {"path": iu4_construction["descriptor_path"],
                                 "sha256": iu4_construction["descriptor_sha256"]},
                "hardware": {"path": iu4_hardware["path"],
                             "sha256": iu4_hardware["sha256"]},
                "metrics": iu4_metrics,
            },
        },
        "deltas": deltas,
        "interpretation": "descriptive_sequential_comparison_not_counterbalanced_selection",
        "selection_allowed": False,
        "publishes": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    derive = commands.add_parser("derive-contract")
    derive.add_argument("--timing", type=Path, required=True)
    derive.add_argument("--benchmark-driver", type=Path, required=True)
    derive.add_argument("--gate-driver", type=Path, required=True)
    derive.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    for arm in ("q3", "iu4"):
        compare.add_argument(f"--{arm}-construction", type=Path, required=True)
        compare.add_argument(f"--{arm}-construction-sha256", required=True)
        compare.add_argument(f"--{arm}-hardware", type=Path, required=True)
        compare.add_argument(f"--{arm}-hardware-sha256", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "derive-contract":
            output = derive_contract(
                args.timing, args.benchmark_driver, args.gate_driver)
        else:
            q3_construction = _construction(
                args.q3_construction, args.q3_construction_sha256, Q3_ARM, "Q3")
            iu4_construction = _construction(
                args.iu4_construction, args.iu4_construction_sha256, IU4_ARM, "IU4")
            q3_hardware = validate_hardware(
                args.q3_hardware, args.q3_hardware_sha256, q3_construction, "Q3")
            iu4_hardware = validate_hardware(
                args.iu4_hardware, args.iu4_hardware_sha256, iu4_construction, "IU4")
            output = make_comparison(
                q3_construction, q3_hardware, iu4_construction, iu4_hardware)
        write_new(args.output, output)
    except (ComparisonError, OSError) as exc:
        print(f"qwen_quant_comparison.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.absolute()),
                      "status": output["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
