#!/usr/bin/env python3
"""Derive Q3 first-token evidence from Ember's full serial differential.

The real-weight gate already pays for a 64-token baseline, production-prefill,
restored-speculative, fresh-speculative, and disk-restore comparison.  This
tool turns those retained artifacts into the narrower first-token attestation
without loading the model again.  It is deliberately stdlib-only and
create-only so CI can exercise the exact post-processing path without a GPU.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import sys
from typing import Any


SCHEMA = "ember.qwen3.8.q3-ple-first-token.v1"
EXPECTED_TOKENS = 64
REQUIRED_PATHS = {"baseline", "prefill", "restored", "fresh", "disk"}
TRACE = re.compile(
    r"^\[ember-validate-token\] path=(\w+) index=(\d+) id=(-?\d+)$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IMAGE = re.compile(r"^ghcr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$")


class EvidenceError(ValueError):
    """The retained differential cannot support the requested evidence."""


def _open_regular(path: pathlib.Path, label: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(f"cannot open {label}: {exc}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise EvidenceError(f"{label} is not a regular file: {path}")
    return descriptor, metadata


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def _require_unchanged(
        before: os.stat_result, after: os.stat_result, label: str) -> None:
    if _identity(before) != _identity(after):
        raise EvidenceError(f"{label} changed while it was being consumed")


def _hex(value: str, pattern: re.Pattern[str], label: str) -> str:
    if pattern.fullmatch(value) is None:
        raise EvidenceError(f"{label} is not an exact lowercase digest")
    return value


def _absolute(value: str, label: str) -> str:
    if not pathlib.Path(value).is_absolute():
        raise EvidenceError(f"{label} must be absolute")
    return value


def _load_differential(path: pathlib.Path) -> tuple[dict[str, Any], str]:
    descriptor, before = _open_regular(path, "differential JSON")
    try:
        with os.fdopen(descriptor, "rb") as stream:
            if before.st_size > 1024 * 1024:
                raise EvidenceError("differential JSON exceeds the 1 MiB bound")
            raw = stream.read()
            after = os.fstat(stream.fileno())
        _require_unchanged(before, after, "differential JSON")
        value = json.loads(raw.decode("utf-8"))
    except EvidenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read differential JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError("differential JSON is not an object")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_differential(value: dict[str, Any]) -> int:
    prefill = value.get("prefill") or {}
    spec = value.get("spec") or {}
    disk = value.get("disk") or {}
    n_tokens = value.get("baseline_tokens")
    if (value.get("ok") is not True or value.get("snapshot_ok") is not True
            or value.get("requested_tokens") != EXPECTED_TOKENS
            or type(n_tokens) is not int or n_tokens != EXPECTED_TOKENS
            or not isinstance(prefill, dict)
            or prefill.get("checked") is not True
            or type(prefill.get("exact")) is not bool
            or prefill.get("accepted") is not True
            or prefill.get("tv_checked") is not True
            or prefill.get("tv_within_bound") is not True
            or prefill.get("tokens") != n_tokens
            or not isinstance(spec, dict) or spec.get("checked") is not True
            or spec.get("exact") is not True or spec.get("tokens") != n_tokens
            or not isinstance(disk, dict) or disk.get("checked") is not True
            or disk.get("exact") is not True or disk.get("tokens") != n_tokens):
        raise EvidenceError(
            "64-token differential did not prove accepted prefill plus exact "
            f"authority paths: {value}")
    tv_distance = prefill.get("tv_distance")
    tv_threshold = prefill.get("tv_threshold")
    if (isinstance(tv_distance, bool) or
            not isinstance(tv_distance, (int, float)) or
            not math.isfinite(float(tv_distance)) or
            isinstance(tv_threshold, bool) or
            not isinstance(tv_threshold, (int, float)) or
            not math.isfinite(float(tv_threshold)) or
            float(tv_distance) > float(tv_threshold)):
        raise EvidenceError(
            f"prefill TV evidence is missing or inconsistent: {prefill}")
    return n_tokens


def _load_traces(
        path: pathlib.Path, n_tokens: int) -> tuple[dict[str, list[int]], str]:
    descriptor, before = _open_regular(path, "differential engine log")
    found: dict[str, dict[int, int]] = {}
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb") as stream:
            for raw_line in stream:
                if len(raw_line) > 1024 * 1024:
                    raise EvidenceError(
                        "differential engine log contains an oversized line")
                digest.update(raw_line)
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                match = TRACE.fullmatch(line)
                if not match:
                    continue
                name = match.group(1)
                index = int(match.group(2))
                path_tokens = found.setdefault(name, {})
                if index in path_tokens:
                    raise EvidenceError(
                        f"duplicate token trace for {name} index {index}")
                path_tokens[index] = int(match.group(3))
            after = os.fstat(stream.fileno())
        _require_unchanged(before, after, "differential engine log")
    except OSError as exc:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise EvidenceError(f"cannot read differential engine log: {exc}") from exc

    expected_indices = set(range(n_tokens))
    if (set(found) != REQUIRED_PATHS or any(
            set(found[name]) != expected_indices for name in REQUIRED_PATHS)):
        raise EvidenceError(f"token trace paths or indices differ: {found}")
    traces = {
        name: [found[name][index] for index in range(n_tokens)]
        for name in sorted(REQUIRED_PATHS)
    }
    baseline = traces["baseline"]
    authority_paths = ("restored", "fresh", "disk")
    if any(traces[name] != baseline for name in authority_paths):
        raise EvidenceError(
            f"token traces differ across exact authority paths: {traces}")
    if baseline[0] < 0:
        raise EvidenceError(f"first token is invalid: {baseline[0]}")
    return traces, digest.hexdigest()


def derive(args: argparse.Namespace, *, created_at: str | None = None) -> dict[str, Any]:
    root = pathlib.Path(args.root)
    if root.is_symlink() or not root.is_dir():
        raise EvidenceError(f"evidence root is not a directory: {root}")
    differential = root / "full-benchmark" / "differential.json"
    engine_log = root / "full-benchmark" / "differential-dispatch-server.log"
    differential_value, differential_sha = _load_differential(differential)
    n_tokens = _validate_differential(differential_value)
    traces, engine_log_sha = _load_traces(engine_log, n_tokens)
    prefill = differential_value["prefill"]
    trace_exact = traces["prefill"] == traces["baseline"]
    if trace_exact != prefill["exact"]:
        raise EvidenceError(
            "prefill exact verdict disagrees with the retained token trace")

    revision = _hex(args.ember_revision, HEX40, "Ember revision")
    descriptor_sha = _hex(
        args.construction_descriptor_sha256, HEX64,
        "construction descriptor SHA-256")
    model_sha = _hex(args.model_sha256, HEX64, "model SHA-256")
    mtp_sha = _hex(args.mtp_sha256, HEX64, "MTP SHA-256")
    engine_sha = _hex(
        args.runtime_engine_sha256, HEX64, "runtime engine SHA-256")
    if IMAGE.fullmatch(args.runtime_image) is None:
        raise EvidenceError("runtime image is not one exact GHCR digest reference")
    image_digest = args.runtime_image_digest
    if (re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest) is None
            or not args.runtime_image.endswith("@" + image_digest)):
        raise EvidenceError("runtime image digest differs from its exact reference")
    if not args.candidate_id or len(args.candidate_id) > 256:
        raise EvidenceError("candidate ID is empty or unreasonably long")
    if args.model_total_bytes <= 0:
        raise EvidenceError("model total bytes must be positive")
    if args.mtp_depth != 3:
        raise EvidenceError("Q3 first-token evidence requires MTP depth 3")

    timestamp = created_at or dt.datetime.now(dt.timezone.utc).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")
    first = traces["baseline"][0]
    return {
        "schema": SCHEMA,
        "status": "complete",
        "publishes": False,
        "deletes": False,
        "created_at": timestamp,
        "ember_revision": revision,
        "candidate_id": args.candidate_id,
        "construction_descriptor": {
            "path": _absolute(
                args.construction_descriptor, "construction descriptor"),
            "sha256": descriptor_sha,
        },
        "model": {
            "first_shard": _absolute(args.model, "model"),
            "first_shard_sha256": model_sha,
            "total_bytes": args.model_total_bytes,
        },
        "mtp": {
            "path": _absolute(args.mtp, "MTP"),
            "sha256": mtp_sha,
            "depth": args.mtp_depth,
            "matrix_quant_contract": "Q4_0_ROCMFP4_FAST",
        },
        "quantization_arm": "rocmfp4-fast-matrix-q3-ple-q6k-embedding-head",
        "runtime": {
            "image": args.runtime_image,
            "image_digest": image_digest,
            "engine_sha256": engine_sha,
            "device": "gfx1151",
        },
        "generation": {
            "requested_tokens": n_tokens,
            "first_token_id": first,
            "token_ids_by_path": traces,
            "snapshot_exact": True,
            "production_prefill_exact": prefill["exact"],
            "production_prefill_accepted": True,
            "production_prefill_tv": {
                "checked": True,
                "within_bound": True,
                "distance": prefill["tv_distance"],
                "threshold": prefill["tv_threshold"],
                "row": prefill.get("tv_index", -1),
            },
            "speculative_exact": True,
            "disk_restore_exact": True,
        },
        "evidence": {
            "validation_json_sha256": differential_sha,
            "engine_log_sha256": engine_log_sha,
        },
        "claim_scope": (
            "first sampled token plus 64-token serial differential; "
            "no quality or performance claim"),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", required=True)
    value.add_argument("--ember-revision", required=True)
    value.add_argument("--candidate-id", required=True)
    value.add_argument("--construction-descriptor", required=True)
    value.add_argument("--construction-descriptor-sha256", required=True)
    value.add_argument("--model", required=True)
    value.add_argument("--model-sha256", required=True)
    value.add_argument("--model-total-bytes", required=True, type=int)
    value.add_argument("--mtp", required=True)
    value.add_argument("--mtp-sha256", required=True)
    value.add_argument("--mtp-depth", required=True, type=int)
    value.add_argument("--runtime-image", required=True)
    value.add_argument("--runtime-image-digest", required=True)
    value.add_argument("--runtime-engine-sha256", required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        evidence = derive(args)
        output = pathlib.Path(args.root) / "first-token-evidence.json"
        with output.open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(evidence["generation"]["first_token_id"])
        return 0
    except (EvidenceError, FileExistsError, OSError) as exc:
        print(f"qwen-first-token-evidence: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
