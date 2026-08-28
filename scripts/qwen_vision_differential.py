#!/usr/bin/env python3
"""Materialize, compare, and verify Qwen real-weight vision evidence."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any


CORPUS_SCHEMA = "ember.qwen3.8.vision-differential-corpus.v1"
IDENTITY_SCHEMA = "ember.qwen3.8.vision-runtime-identity.v1"
EVIDENCE_SCHEMA = "ember.qwen3.8.vision-differential-evidence.v1"
MODEL_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
LLAMA_REVISION = "abdc7a0bf815d3b83e26dd523c6960e4dd597e82"
CORPUS_SHA256 = "ea723d93b6f6f0b36b608ca42fcbe3653d5f4388a04ab046e0e3bff0ec2f724c"
CASES = {
    "checkerboard-56": {
        "image_sha256": "339bb609557896a9de3eef73d08de863e94a596e9a0efa061d03562112fa1d75",
        "prompt_sha256": "77215e48a3a67b27202d14922ce6bb6e85ead5dde86c71099de21cc7ba459c3d"},
    "rgb-bands-84x56": {
        "image_sha256": "e883a8dafa672931b3e55ee8456efd2fbc0c3e68b59993529ad2a140d243f132",
        "prompt_sha256": "06325b7b90fa4f7f1f92fb1af5d53401c2851fbd90b1f335ea74c7acbedcb831"},
}
ATOL = 1.0e-5
RTOL = 1.0e-5
RUNNER_MEMTOTAL_BYTES = 134_297_894_912
RUNNER_GTT_PAGES_LIMIT = 32_505_856
MAX_CERTIFIED_BYTES = 133_143_986_176
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
CASE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


class VisionEvidenceError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stable_bytes(path: Path, expected: str | None, label: str) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        raise VisionEvidenceError(f"{label} must be an absolute non-symlink file")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise VisionEvidenceError(f"{label} is not regular")
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                            row.st_mtime_ns, row.st_ctime_ns)
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise VisionEvidenceError(f"{label} changed while read")
    if expected is not None and sha256_bytes(raw) != expected:
        raise VisionEvidenceError(f"{label} SHA-256 differs")
    return raw


def read_json(path: Path, expected: str | None, label: str) -> dict[str, Any]:
    try:
        value = json.loads(stable_bytes(path, expected, label))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VisionEvidenceError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise VisionEvidenceError(f"{label} must be an object")
    return value


def descriptor(path: Path, label: str) -> dict[str, Any]:
    raw = stable_bytes(path.absolute(), None, label)
    return {"path": str(path.absolute()), "size_bytes": len(raw),
            "sha256": sha256_bytes(raw)}


def validate_descriptor(value: Any, label: str, *, expected_sha256: str | None = None,
                        expected_size: int | None = None,
                        minimum_size: int = 1) -> None:
    if (not isinstance(value, dict)
            or set(value) != {"path", "size_bytes", "sha256"}
            or not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or type(value.get("size_bytes")) is not int
            or value["size_bytes"] < minimum_size
            or HEX64.fullmatch(str(value.get("sha256"))) is None):
        raise VisionEvidenceError(f"vision {label} descriptor is malformed")
    if expected_sha256 is not None and value["sha256"] != expected_sha256:
        raise VisionEvidenceError(f"vision {label} SHA-256 differs")
    if expected_size is not None and value["size_bytes"] != expected_size:
        raise VisionEvidenceError(f"vision {label} byte count differs")


def load_corpus(path: Path, expected: str) -> tuple[dict, list[tuple[dict, bytes]]]:
    if expected != CORPUS_SHA256:
        raise VisionEvidenceError("vision corpus is not the pinned release corpus")
    corpus = read_json(path.absolute(), expected, "vision corpus")
    if (set(corpus) != {"schema", "source", "cases"} or
            corpus.get("schema") != CORPUS_SCHEMA or corpus.get("source") != {
                "model_repo": "Qwen/Qwen3.8-Flash-Next",
                "model_revision": MODEL_REVISION,
                "llama_cpp_repo": "ggml-org/llama.cpp",
                "llama_cpp_revision": LLAMA_REVISION}):
        raise VisionEvidenceError("vision corpus source contract differs")
    cases = corpus.get("cases")
    if not isinstance(cases, list) or len(cases) < 2:
        raise VisionEvidenceError("vision corpus requires at least two cases")
    result = []
    seen = set()
    for row in cases:
        if (not isinstance(row, dict) or set(row) != {
                "id", "prompt", "mime_type", "image_sha256", "image_base64"}
                or CASE_ID.fullmatch(str(row.get("id"))) is None
                or row["id"] in seen or row.get("mime_type") != "image/png"
                or not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
                or HEX64.fullmatch(str(row.get("image_sha256"))) is None):
            raise VisionEvidenceError("vision corpus case is malformed")
        try:
            image = base64.b64decode(row["image_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise VisionEvidenceError("vision corpus image is not strict base64") from exc
        if sha256_bytes(image) != row["image_sha256"] or not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise VisionEvidenceError("vision corpus image identity differs")
        seen.add(row["id"]); result.append((row, image))
    return corpus, result


def write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    _corpus, cases = load_corpus(args.corpus, args.corpus_sha256)
    root = args.output_dir.absolute()
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for row, image in cases:
        image_path = root / f"{row['id']}.png"
        request_path = root / f"{row['id']}.request.json"
        write_new(image_path, image)
        request = {"model": "qwen3.8-flash-next", "temperature": 0,
                   "max_tokens": 8, "stream": False, "messages": [{
                       "role": "user", "content": [
                           {"type": "image_url", "image_url": {"url":
                            f"data:{row['mime_type']};base64,{row['image_base64']}"}},
                           {"type": "text", "text": row["prompt"]}]}]}
        write_new(request_path, canonical(request))
        rows.append({"id": row["id"], "image": descriptor(image_path, "case image"),
                     "request": descriptor(request_path, "case request")})
    return {"status": "materialized", "cases": rows}


def capture(path: Path, label: str) -> tuple[dict[str, int], tuple[float, ...], bytes]:
    raw = stable_bytes(path.absolute(), None, label)
    if len(raw) < 56 or raw[:8] != b"EVISF32\0":
        raise VisionEvidenceError(f"{label} header differs")
    version, grid_t, grid_h, grid_w, width, rows = struct.unpack("<6Q", raw[8:56])
    if (version != 1 or grid_t != 1 or width != 2560 or rows == 0 or
            grid_h == 0 or grid_w == 0 or grid_h % 2 or grid_w % 2 or
            rows != (grid_h // 2) * (grid_w // 2) or
            len(raw) != 56 + rows * width * 4):
        raise VisionEvidenceError(f"{label} shape or byte count differs")
    values = struct.unpack(f"<{rows * width}f", raw[56:])
    if not all(math.isfinite(value) for value in values):
        raise VisionEvidenceError(f"{label} contains non-finite values")
    return ({"grid_t": grid_t, "grid_h": grid_h, "grid_w": grid_w,
             "embedding_width": width, "rows": rows}, values, raw)


def comparison(observed_path: Path, reference_path: Path, label: str) -> dict:
    observed_shape, observed, observed_raw = capture(observed_path, f"{label} Ember capture")
    reference_shape, reference, reference_raw = capture(reference_path, f"{label} reference")
    if observed_shape != reference_shape or len(observed) != len(reference):
        raise VisionEvidenceError(f"{label} Ember/reference shape differs")
    max_abs = 0.0; max_rel = 0.0; max_tolerance_ratio = 0.0; mismatch = 0
    for actual, wanted in zip(observed, reference):
        absolute = abs(actual - wanted)
        relative = absolute / max(abs(wanted), ATOL)
        tolerance_ratio = absolute / (ATOL + RTOL * abs(wanted))
        max_abs = max(max_abs, absolute); max_rel = max(max_rel, relative)
        max_tolerance_ratio = max(max_tolerance_ratio, tolerance_ratio)
        if absolute > ATOL + RTOL * abs(wanted):
            mismatch += 1
    if mismatch:
        raise VisionEvidenceError(
            f"{label} has {mismatch} embedding values outside tolerance")
    return {"shape": observed_shape, "values_compared": len(observed),
            "max_absolute_error": max_abs, "max_relative_error": max_rel,
            "max_tolerance_ratio": max_tolerance_ratio,
            "mismatches": mismatch,
            "ember": {"path": str(observed_path.absolute()),
                       "size_bytes": len(observed_raw),
                       "sha256": sha256_bytes(observed_raw)},
            "reference": {"path": str(reference_path.absolute()),
                           "size_bytes": len(reference_raw),
                           "sha256": sha256_bytes(reference_raw)}}


def validate_identity(value: dict) -> None:
    if set(value) != {"schema", "runtime", "model", "mtp", "vision_mmproj", "vision_vocab",
                      "provider", "reference", "hardware", "corpus"} or value.get("schema") != IDENTITY_SCHEMA:
        raise VisionEvidenceError("vision runtime identity fields differ")
    runtime = value["runtime"]
    if (not isinstance(runtime, dict)
            or set(runtime) != {"image", "ember_revision", "engine_binary_sha256"}
            or IMAGE.fullmatch(str(runtime.get("image"))) is None
            or HEX40.fullmatch(str(runtime.get("ember_revision"))) is None
            or HEX64.fullmatch(str(runtime.get("engine_binary_sha256"))) is None):
        raise VisionEvidenceError("vision runtime image identity is malformed")
    expected_fields = {
        "model": {"path", "sha256", "model_inventory_sha256", "first_shard_path",
                  "first_shard_sha256", "build_record_path", "build_record_sha256"},
        "mtp": {"path", "sha256", "size_bytes", "depth"},
        "vision_mmproj": {"path", "sha256", "size_bytes", "format"},
        "vision_vocab": {"path", "sha256", "size_bytes", "format", "metadata_sha256"},
        "provider": {"path", "sha256", "abi_version", "llama_cpp_revision"},
        "reference": {"path", "sha256", "image", "llama_cpp_revision"},
        "corpus": {"path", "sha256"},
    }
    for name in ("model", "mtp", "vision_mmproj", "vision_vocab", "provider", "reference", "corpus"):
        row = value[name]
        if (not isinstance(row, dict) or set(row) != expected_fields[name]
                or HEX64.fullmatch(str(row.get("sha256"))) is None
                or not isinstance(row.get("path"), str)
                or not Path(row["path"]).is_absolute()):
            raise VisionEvidenceError(f"vision {name} identity is malformed")
    if (not isinstance(value.get("hardware"), dict)
            or set(value["hardware"]) != {"gpu_arch", "rocm_version"}):
        raise VisionEvidenceError("vision hardware identity is malformed")
    if (HEX64.fullmatch(str(value["model"].get("model_inventory_sha256"))) is None
            or HEX64.fullmatch(str(value["model"].get("first_shard_sha256"))) is None
            or HEX64.fullmatch(str(value["model"].get("build_record_sha256"))) is None
            or not Path(str(value["model"].get("first_shard_path", ""))).is_absolute()
            or not Path(str(value["model"].get("build_record_path", ""))).is_absolute()):
        raise VisionEvidenceError("vision ordered model identity is malformed")
    if any(type(value[name].get("size_bytes")) is not int
           or value[name]["size_bytes"] <= 0 for name in ("mtp", "vision_mmproj", "vision_vocab")):
        raise VisionEvidenceError("vision companion byte count is malformed")
    if (value["mtp"].get("depth") not in (1, 2, 3, 4)
            or value["vision_mmproj"].get("format") != "BF16"
            or value["vision_vocab"].get("format") != "GGUF_VOCAB_ONLY"
            or HEX64.fullmatch(str(value["vision_vocab"].get("metadata_sha256"))) is None
            or value["provider"].get("abi_version") != 1
            or value["provider"].get("llama_cpp_revision") != LLAMA_REVISION
            or value["reference"].get("llama_cpp_revision") != LLAMA_REVISION
            or IMAGE.fullmatch(str(value["reference"].get("image"))) is None
            or value["hardware"].get("gpu_arch") != "gfx1151"
            or value["hardware"].get("rocm_version") != "10.0.0"):
        raise VisionEvidenceError("vision companion/reference/hardware identity differs")


def validate_residency(value: dict) -> None:
    phases = value.get("phases") if isinstance(value, dict) else None
    if (not isinstance(value, dict)
            or set(value) != {"schema", "method", "server_host_pid",
                              "sample_interval_seconds", "runner_memtotal_bytes",
                              "runner_gtt_pages_limit", "phases", "raw_samples"}
            or value.get("schema") != "ember.qwen3.8.vision-residency.v1"
            or value.get("method") != "runner_phase_rss_gtt_uma_sampler_v1"
            or not isinstance(phases, dict) or set(phases) != {"idle", "cold", "warm", "settled"}
            or type(value.get("server_host_pid")) is not int
            or value["server_host_pid"] <= 1
            or not isinstance(value.get("sample_interval_seconds"), (int, float))
            or isinstance(value["sample_interval_seconds"], bool)
            or not math.isfinite(value["sample_interval_seconds"])
            or value["sample_interval_seconds"] <= 0
            or type(value.get("runner_memtotal_bytes")) is not int
            or value["runner_memtotal_bytes"] != RUNNER_MEMTOTAL_BYTES
            or value.get("runner_gtt_pages_limit") != RUNNER_GTT_PAGES_LIMIT):
        raise VisionEvidenceError("vision residency differs from the exact 128-GiB runner profile")
    raw_samples = value.get("raw_samples")
    sample_fields = {"monotonic", "phase", "memtotal_bytes", "mem_available_bytes",
                     "uma_used_bytes", "rss_bytes", "hwm_bytes", "gtt_bytes"}
    if not isinstance(raw_samples, list) or len(raw_samples) < len(phases) * 2:
        raise VisionEvidenceError("vision residency lacks bound raw samples")
    previous_time = -math.inf
    raw_by_phase: dict[str, list[dict]] = {phase: [] for phase in phases}
    for sample in raw_samples:
        timestamp = sample.get("monotonic") if isinstance(sample, dict) else None
        if (not isinstance(sample, dict) or set(sample) != sample_fields
                or sample.get("phase") not in raw_by_phase
                or not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool)
                or not math.isfinite(timestamp) or timestamp < previous_time):
            raise VisionEvidenceError("vision residency raw sample is malformed")
        previous_time = timestamp
        for key in sample_fields - {"monotonic", "phase"}:
            if type(sample.get(key)) is not int or sample[key] < 0:
                raise VisionEvidenceError("vision residency raw counter is malformed")
        if (sample["memtotal_bytes"] != value["runner_memtotal_bytes"]
                or sample["mem_available_bytes"] > sample["memtotal_bytes"]
                or sample["uma_used_bytes"] !=
                   sample["memtotal_bytes"] - sample["mem_available_bytes"]):
            raise VisionEvidenceError("vision residency raw UMA counters are incoherent")
        raw_by_phase[sample["phase"]].append(sample)
    peaks = []
    for phase, row in phases.items():
        if (not isinstance(row, dict)
                or set(row) != {"samples", "rss_peak_bytes", "gtt_peak_bytes",
                                "uma_used_peak_bytes", "mem_available_floor_bytes"}
                or type(row.get("samples")) is not int or row["samples"] < 2):
            raise VisionEvidenceError(f"vision residency phase {phase} is undersampled")
        for key in ("rss_peak_bytes", "gtt_peak_bytes", "uma_used_peak_bytes",
                    "mem_available_floor_bytes"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise VisionEvidenceError(f"vision residency phase {phase} lacks {key}")
        selected = raw_by_phase[phase]
        expected = {
            "samples": len(selected),
            "rss_peak_bytes": max(max(sample["rss_bytes"], sample["hwm_bytes"])
                                  for sample in selected),
            "gtt_peak_bytes": max(sample["gtt_bytes"] for sample in selected),
            "uma_used_peak_bytes": max(sample["uma_used_bytes"] for sample in selected),
            "mem_available_floor_bytes": min(sample["mem_available_bytes"]
                                             for sample in selected),
        }
        if row != expected:
            raise VisionEvidenceError(f"vision residency phase {phase} summary differs from raw samples")
        # On a UMA APU RSS, GTT, and MemTotal-MemAvailable overlap; summing them
        # would double/triple count the same pages. The conservative resident
        # envelope is their maximum, and must include system-wide UMA use so a
        # small server RSS cannot hide pressure from the rest of the host.
        peaks.append(max(row["rss_peak_bytes"], row["gtt_peak_bytes"],
                         row["uma_used_peak_bytes"]))
    limit = value.get("runner_gtt_pages_limit")
    if type(limit) is not int or limit != RUNNER_GTT_PAGES_LIMIT:
        raise VisionEvidenceError("vision residency TTM pages_limit differs")
    if (MAX_CERTIFIED_BYTES > value["runner_memtotal_bytes"]
            or limit * 4096 != MAX_CERTIFIED_BYTES):
        raise VisionEvidenceError("vision certified cap differs from exact runner profile")
    if max(peaks) > MAX_CERTIFIED_BYTES:
        raise VisionEvidenceError("vision cold/warm residency exceeds the certified UMA cap")


def finalize(args: argparse.Namespace) -> dict:
    _corpus, cases = load_corpus(args.corpus, args.corpus_sha256)
    identity = read_json(args.identity.absolute(), args.identity_sha256, "vision identity")
    validate_identity(identity)
    if identity["corpus"]["sha256"] != args.corpus_sha256:
        raise VisionEvidenceError("runtime identity names a different vision corpus")
    residency = read_json(args.residency.absolute(), args.residency_sha256, "vision residency")
    validate_residency(residency)
    comparisons = []
    case_count = len(cases)
    for run_index, run_name in enumerate(("cold", "warm")):
        for case_index, (row, _image) in enumerate(cases):
            observed = args.ember_dir / f"ember.{run_index * case_count + case_index:06d}.f32"
            reference = args.reference_dir / f"{row['id']}.f32"
            response = args.response_dir / f"{run_name}-{row['id']}.json"
            response_value = read_json(response.absolute(), None, "Ember image-text response")
            choices = response_value.get("choices")
            message = (choices[0].get("message") if isinstance(choices, list)
                       and len(choices) == 1 and isinstance(choices[0], dict) else None)
            visible = ((message or {}).get("content") or "")
            reasoning = ((message or {}).get("reasoning_content") or "")
            if (not isinstance(choices, list) or len(choices) != 1
                    or not isinstance(choices[0], dict)
                    or not isinstance(message, dict)
                    or not isinstance(visible, str) or not isinstance(reasoning, str)
                    or not (visible.strip() or reasoning.strip())):
                raise VisionEvidenceError("Ember image-text response is incomplete")
            result = comparison(observed, reference, f"{run_name}/{row['id']}")
            result.update({"run": run_name, "case_id": row["id"],
                           "image_sha256": row["image_sha256"],
                           "prompt_sha256": sha256_bytes(row["prompt"].encode()),
                           "response": descriptor(response.absolute(), "Ember response")})
            comparisons.append(result)
    return {"schema": EVIDENCE_SCHEMA, "status": "passed", "passed": True,
            "publishes": False, "certification_scope": "qwen_image_text_gfx1151",
            "identity": identity,
            "corpus": descriptor(args.corpus.absolute(), "vision corpus"),
            "tolerances": {"comparison": "elementwise_float32",
                           "absolute": ATOL, "relative": RTOL,
                           "non_finite_allowed": False},
            "runs": ["cold", "warm"], "comparisons": comparisons,
            "residency": {"evidence": descriptor(args.residency.absolute(), "vision residency"),
                          "measurement": residency,
                          "certified_peak_cap_bytes": MAX_CERTIFIED_BYTES},
            "release_gate": {"real_weight_differential": True,
                             "cold_warm_residency_captured": True,
                             "publication_requires_exact_evidence_sha256": True}}


def verify(value: dict, expected_revision: str | None = None) -> dict:
    if (set(value) != {"schema", "status", "passed", "publishes", "certification_scope",
                       "identity", "corpus", "tolerances", "runs", "comparisons",
                       "residency", "release_gate"} or value.get("schema") != EVIDENCE_SCHEMA
            or value.get("status") != "passed" or value.get("passed") is not True
            or value.get("publishes") is not False
            or value.get("certification_scope") != "qwen_image_text_gfx1151"):
        raise VisionEvidenceError("vision evidence is not a passing nonpublishing gate")
    validate_identity(value["identity"])
    residency = value.get("residency")
    if (not isinstance(residency, dict)
            or set(residency) != {"evidence", "measurement", "certified_peak_cap_bytes"}
            or residency.get("certified_peak_cap_bytes") != MAX_CERTIFIED_BYTES):
        raise VisionEvidenceError("vision residency evidence/cap contract differs")
    validate_descriptor(residency.get("evidence"), "residency evidence")
    validate_residency(residency.get("measurement"))
    if (value.get("tolerances") != {"comparison": "elementwise_float32",
                                    "absolute": ATOL, "relative": RTOL,
                                    "non_finite_allowed": False}
            or value.get("runs") != ["cold", "warm"]
            or value.get("release_gate") != {
                "real_weight_differential": True,
                "cold_warm_residency_captured": True,
                "publication_requires_exact_evidence_sha256": True}):
        raise VisionEvidenceError("vision evidence gate/tolerance contract differs")
    rows = value.get("comparisons")
    if not isinstance(rows, list) or len(rows) != 4:
        raise VisionEvidenceError("vision evidence lacks cold/warm corpus comparisons")
    expected_row_fields = {
        "shape", "values_compared", "max_absolute_error", "max_relative_error",
        "max_tolerance_ratio", "mismatches", "ember", "reference", "run",
        "case_id", "image_sha256", "prompt_sha256", "response"}
    if any(not isinstance(row, dict) or set(row) != expected_row_fields for row in rows):
        raise VisionEvidenceError("vision comparison row fields differ")
    keys = {(row["run"], row["case_id"]) for row in rows}
    if keys != {(run, case_id) for run in ("cold", "warm") for case_id in CASES}:
        raise VisionEvidenceError("vision evidence cold/warm case coverage differs")
    for row in rows:
        shape = row["shape"]
        if (not isinstance(shape, dict)
                or set(shape) != {"grid_t", "grid_h", "grid_w", "embedding_width", "rows"}
                or any(type(shape.get(key)) is not int or shape[key] <= 0 for key in shape)
                or shape["grid_t"] != 1 or shape["embedding_width"] != 2560
                or shape["grid_h"] % 2 or shape["grid_w"] % 2
                or shape["rows"] != (shape["grid_h"] // 2) * (shape["grid_w"] // 2)
                or type(row["values_compared"]) is not int
                or row["values_compared"] != shape["rows"] * 2560
                or row["values_compared"] <= 0
                or row["mismatches"] != 0):
            raise VisionEvidenceError("vision comparison shape/count/result differs")
        for field in ("max_absolute_error", "max_relative_error", "max_tolerance_ratio"):
            number = row[field]
            if (not isinstance(number, (int, float)) or isinstance(number, bool)
                    or not math.isfinite(number) or number < 0):
                raise VisionEvidenceError("vision comparison error statistic is malformed")
        if row["max_tolerance_ratio"] > 1.0:
            raise VisionEvidenceError("vision comparison exceeds elementwise tolerance")
        capture_size = 56 + row["values_compared"] * 4
        validate_descriptor(row["ember"], "Ember capture", expected_size=capture_size)
        validate_descriptor(row["reference"], "reference capture", expected_size=capture_size)
        validate_descriptor(row["response"], "Ember response")
    validate_descriptor(value.get("corpus"), "corpus", expected_sha256=CORPUS_SHA256,
                        expected_size=1306)
    if (value["identity"]["corpus"].get("sha256") != CORPUS_SHA256
            or any(row.get("image_sha256") != CASES[row["case_id"]]["image_sha256"]
                   or row.get("prompt_sha256") != CASES[row["case_id"]]["prompt_sha256"]
                   for row in rows)):
        raise VisionEvidenceError("vision evidence case/corpus identity differs")
    if expected_revision is not None and value["identity"]["runtime"].get(
            "ember_revision") != expected_revision:
        raise VisionEvidenceError("vision evidence belongs to another Ember revision")
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("materialize")
    prep.add_argument("--corpus", type=Path, required=True)
    prep.add_argument("--corpus-sha256", required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    final = sub.add_parser("finalize")
    for name in ("corpus", "identity", "residency"):
        final.add_argument(f"--{name}", type=Path, required=True)
        final.add_argument(f"--{name}-sha256", required=True)
    for name in ("ember-dir", "reference-dir", "response-dir"):
        final.add_argument(f"--{name}", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--evidence", type=Path, required=True)
    check.add_argument("--evidence-sha256", required=True)
    check.add_argument("--expected-ember-revision")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        if args.command == "materialize":
            result = materialize(args)
        elif args.command == "finalize":
            result = finalize(args)
            write_new(args.output.absolute(), canonical(result))
            result = {"status": "passed", "output": str(args.output.absolute())}
        else:
            value = read_json(args.evidence.absolute(), args.evidence_sha256,
                              "vision evidence")
            verify(value, args.expected_ember_revision)
            result = {"status": "passed", "publishes": False}
    except (OSError, KeyError, TypeError, ValueError, VisionEvidenceError) as exc:
        print(f"qwen_vision_differential.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
