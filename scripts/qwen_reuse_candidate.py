#!/usr/bin/env python3
"""Verify and rebind an immutable Qwen candidate to a new runtime revision."""

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


HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
CONSTRUCTION_SCHEMA = "ember.qwen3.8.candidate-construction.v1"


class ReuseError(RuntimeError):
    pass


def sha256_file(path: Path, *, direct_threshold: int = 512 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    if before.st_size >= direct_threshold:
        process = subprocess.Popen(
            ["dd", f"if={path}", "iflag=direct", "bs=8M", "status=none"],
            stdout=subprocess.PIPE,
        )
        assert process.stdout is not None
        for block in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
            digest.update(block)
        if process.wait() != 0:
            raise ReuseError(f"O_DIRECT hash failed: {path}")
    else:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ReuseError(f"file changed while hashed: {path}")
    return digest.hexdigest()


def exact_file(path_raw: str, expected: str, label: str) -> Path:
    path = Path(path_raw)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or HEX64.fullmatch(expected) is None):
        raise ReuseError(f"{label} is not one exact absolute regular file")
    if sha256_file(path) != expected:
        raise ReuseError(f"{label} digest differs")
    return path


def descriptor(value: Any, label: str) -> tuple[dict[str, Any], Path]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ReuseError(f"{label} descriptor is malformed")
    path = exact_file(str(value["path"]), str(value["sha256"]), label)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReuseError(f"{label} is not JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ReuseError(f"{label} JSON is not an object")
    return loaded, path


def exact_image(ref: str, digest: str, label: str) -> None:
    if (re.fullmatch(r"ghcr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}", ref) is None
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            or not ref.endswith("@" + digest)):
        raise ReuseError(f"{label} is not one digest-pinned GHCR image")


def rebind(args: argparse.Namespace) -> dict[str, Any]:
    construction_path = exact_file(
        str(args.construction), args.construction_sha256,
        "source construction descriptor",
    )
    construction = json.loads(construction_path.read_text(encoding="utf-8"))
    required = {
        "schema", "status", "publishes", "deletes", "candidate_id", "kind",
        "intended_stage", "row_id", "intervention_configuration_id",
        "quantization_arm", "mtp_matrix_quant_contract", "runtime_mode",
        "builder_revision", "runtime_revision", "images", "capture",
        "stock_capture", "bf16_cache", "shared_companions", "selection_plan",
        "build_record", "builder_attestation", "intervention_manifest",
        "artifacts", "v3_candidate_manifest",
    }
    if (not isinstance(construction, dict) or set(construction) != required
            or construction.get("schema") != CONSTRUCTION_SCHEMA
            or construction.get("status") != "complete"
            or construction.get("publishes") is not False
            or construction.get("deletes") is not False
            or construction.get("runtime_mode") != "exact_dequant"):
        raise ReuseError("source construction descriptor contract differs")
    builder_revision = str(construction.get("builder_revision", ""))
    if (HEX40.fullmatch(builder_revision) is None
            or HEX40.fullmatch(args.runtime_revision) is None):
        raise ReuseError("builder or runtime revision is malformed")

    # Every recipe/source descriptor is rehashed before artifact reuse.
    for key in ("capture", "bf16_cache", "selection_plan"):
        descriptor(construction[key], key.replace("_", " "))
    if construction["intervention_manifest"] is not None:
        descriptor(construction["intervention_manifest"], "intervention manifest")
    companions = construction.get("shared_companions")
    if not isinstance(companions, dict) or set(companions) != {
        "Q4_0_ROCMI4", "Q4_0_ROCMFP4_FAST"
    }:
        raise ReuseError("shared companion set differs")
    for name, value in companions.items():
        descriptor(value, f"{name} companion inventory")

    build, _ = descriptor(construction["build_record"], "build record")
    attestation, _ = descriptor(
        construction["builder_attestation"], "builder attestation")
    if ((build.get("tools") or {}).get("ember_revision") != builder_revision
            or build.get("status") != "complete" or build.get("mode") != "execute"
            or attestation.get("schema") !=
                "ember.qwen3.8.candidate-workset-attestation.v1"
            or attestation.get("candidate_id") != construction.get("candidate_id")
            or attestation.get("build_record_sha256") !=
                construction["build_record"]["sha256"]):
        raise ReuseError("builder record or attestation identity differs")
    recipe = build.get("quantization_recipe") or {}
    if (recipe.get("id") != construction.get("quantization_arm")
            or recipe.get("selected_mtp_matrix_quant_contract") !=
                construction.get("mtp_matrix_quant_contract")
            or recipe.get("ple_override_preserved") is not True):
        raise ReuseError("quantization recipe differs")
    builder_identity = attestation.get("builder_identity") or {}
    format_contract = attestation.get("tensor_format_compatibility_sha256")
    if (builder_identity.get("ember_revision") != builder_revision
            or builder_identity.get("tensor_format_contract_sha256") != format_contract
            or format_contract != args.tensor_format_contract_sha256):
        raise ReuseError("current runtime tensor-format contract differs")

    recorded = (build.get("output") or {}).get("shards")
    artifacts = (construction.get("artifacts") or {}).get("shards")
    if not isinstance(recorded, list) or not recorded or artifacts != recorded:
        raise ReuseError("construction and build-record shard inventories differ")
    total = 0
    for index, row in enumerate(recorded):
        if (not isinstance(row, dict)
                or set(row) != {"path", "size_bytes", "sha256"}
                or not isinstance(row["size_bytes"], int)
                or isinstance(row["size_bytes"], bool) or row["size_bytes"] < 1
                or HEX64.fullmatch(str(row["sha256"])) is None):
            raise ReuseError(f"candidate shard {index} metadata is malformed")
        path = Path(str(row["path"]))
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or path.stat().st_size != row["size_bytes"]
                or sha256_file(path) != row["sha256"]):
            raise ReuseError(f"candidate shard {index} differs")
        total += row["size_bytes"]
    if total != (construction.get("artifacts") or {}).get("total_bytes"):
        raise ReuseError("candidate shard byte total differs")

    exact_image(args.runtime_release_ref, args.runtime_release_digest,
                "runtime release image")
    exact_image(args.runtime_dev_ref, args.runtime_dev_digest,
                "runtime development image")
    rebound = dict(construction)
    rebound["runtime_revision"] = args.runtime_revision
    rebound["images"] = dict(construction["images"])
    rebound["images"]["runtime"] = {
        "release_ref": args.runtime_release_ref,
        "release_digest": args.runtime_release_digest,
        "dev_ref": args.runtime_dev_ref,
        "dev_digest": args.runtime_dev_digest,
        "tensor_format_contract_sha256": format_contract,
    }
    return rebound


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction", type=Path, required=True)
    parser.add_argument("--construction-sha256", required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--runtime-release-ref", required=True)
    parser.add_argument("--runtime-release-digest", required=True)
    parser.add_argument("--runtime-dev-ref", required=True)
    parser.add_argument("--runtime-dev-digest", required=True)
    parser.add_argument("--tensor-format-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        value = rebind(args)
        if not args.output.is_absolute() or args.output.exists() or args.output.is_symlink():
            raise ReuseError("output must be one new absolute path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(args.output, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps({"path": str(args.output),
                          "sha256": sha256_file(args.output)}, sort_keys=True))
        return 0
    except (OSError, ReuseError, json.JSONDecodeError) as exc:
        print(f"qwen-reuse-candidate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
