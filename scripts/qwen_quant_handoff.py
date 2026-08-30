#!/usr/bin/env python3
"""Create a non-executing branch-dispatch envelope for the next quant stage.

The gfx1151 evidence lives on its persistent runner, while the workflow dispatch
is submitted through GitHub.  This helper removes the error-prone manual step of
copying paths, digests, and base64.  It only writes an envelope; it never invokes
GitHub, acquires the GPU, or changes runner artifacts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


ENVELOPE_SCHEMA = "ember.qwen3.8.branch-dispatch-envelope.v1"
HANDOFF_SCHEMA = "ember.qwen3.8.quant-next-operation-handoff.v1"
REQUEST_ROOT = Path(
    "/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/"
    "evidence/operation-requests")
COMPARISON_ROOT = Path(
    "/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/"
    "evidence/comparisons")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


class HandoffError(ValueError):
    pass


def exact_revision(value: str) -> str:
    if HEX40.fullmatch(value) is None:
        raise HandoffError("Ember revision must be one full lowercase SHA")
    return value


def exact_digest(value: str, label: str) -> str:
    if HEX64.fullmatch(value) is None:
        raise HandoffError(f"{label} digest must be one lowercase SHA-256")
    return value


def normalized_absolute(value: Path, label: str) -> str:
    raw = str(value)
    if not value.is_absolute() or os.path.normpath(raw) != raw or "\n" in raw or "\0" in raw:
        raise HandoffError(f"{label} must be one normalized absolute path")
    return raw


def fixed_child(value: Path, root: Path, pattern: str, label: str) -> str:
    raw = normalized_absolute(value, label)
    if value.parent != root or re.fullmatch(pattern, value.name) is None:
        raise HandoffError(f"{label} is not one safe child of {root}")
    return raw


def make_envelope(args: argparse.Namespace) -> dict[str, Any]:
    revision = exact_revision(args.ember_revision)
    common = {
        "q3_construction": normalized_absolute(
            args.q3_construction, "Q3 construction"),
        "q3_construction_sha256": exact_digest(
            args.q3_construction_sha256, "Q3 construction"),
        "q3_hardware": normalized_absolute(args.q3_hardware, "Q3 hardware"),
        "q3_hardware_sha256": exact_digest(
            args.q3_hardware_sha256, "Q3 hardware"),
    }
    if args.command == "matched-iu4":
        if SAFE_ID.fullmatch(args.candidate_id) is None:
            raise HandoffError("matched IU4 candidate id is malformed")
        inputs = {
            **common,
            "candidate_id": args.candidate_id,
            "construction_request_output": fixed_child(
                args.construction_request_output, REQUEST_ROOT,
                r"construction-[A-Za-z0-9._-]{1,80}\.json",
                "matched IU4 construction request"),
        }
        operation = "matched-iu4-plan"
    else:
        inputs = {
            **common,
            "iu4_construction": normalized_absolute(
                args.iu4_construction, "IU4 construction"),
            "iu4_construction_sha256": exact_digest(
                args.iu4_construction_sha256, "IU4 construction"),
            "output": fixed_child(
                args.comparison_output, COMPARISON_ROOT,
                r"q3-iu4-[a-z0-9][a-z0-9._-]{0,79}",
                "Q3/IU4 comparison output"),
        }
        operation = "quant-compare"
    return {
        "schema": ENVELOPE_SCHEMA,
        "ember_revision": revision,
        "operation": operation,
        "inputs": inputs,
        "publishes": False,
        "deletes": False,
    }


def write_new(path: Path, value: dict[str, Any]) -> tuple[str, str]:
    normalized_absolute(path, "handoff output")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise HandoffError("handoff output parent must be one existing real directory")
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode("ascii")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--ember-revision", required=True)
        command.add_argument("--q3-construction", type=Path, required=True)
        command.add_argument("--q3-construction-sha256", required=True)
        command.add_argument("--q3-hardware", type=Path, required=True)
        command.add_argument("--q3-hardware-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)

    matched = commands.add_parser("matched-iu4")
    common(matched)
    matched.add_argument("--candidate-id", required=True)
    matched.add_argument("--construction-request-output", type=Path, required=True)

    compare = commands.add_parser("quant-compare")
    common(compare)
    compare.add_argument("--iu4-construction", type=Path, required=True)
    compare.add_argument("--iu4-construction-sha256", required=True)
    compare.add_argument("--comparison-output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        envelope = make_envelope(args)
        digest, encoded = write_new(args.output, envelope)
    except (HandoffError, OSError) as exc:
        print(f"qwen_quant_handoff.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "schema": HANDOFF_SCHEMA,
        "operation": envelope["operation"],
        "envelope": str(args.output),
        "envelope_sha256": digest,
        "envelope_base64": encoded,
        "release_version": "qwen-dispatch",
        "workflow": ".github/workflows/gfx1151-certify.yml",
        "workflow_dispatch_inputs": {
            "commit_sha": envelope["ember_revision"],
            "release_version": "qwen-dispatch",
            "qwen_dispatch_envelope_base64": encoded,
            "qwen_dispatch_envelope_sha256": digest,
        },
        "dispatches": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
