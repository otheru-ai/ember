#!/usr/bin/env python3
"""Stage and verify only Qwen's digest-pinned selection-phase corpus.

The authoritative OtherU checkout is root-readable on the gfx1151 host.  This
helper is run once in a capability-minimal root container to copy four exact
files into a durable runner-owned directory, then unprivileged to bind those
bytes to Ember's pinned corpus contract.  The final-heldout partition is never
opened or named by this selection-only capability boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


MANIFEST_NAME = "qwen-selection-corpora-manifest.json"
ARTIFACT_NAMES = (
    "extraction-good.jsonl",
    "extraction-bad.jsonl",
    "sweep-validation.jsonl",
)
STAGED_NAMES = (MANIFEST_NAME, *ARTIFACT_NAMES)
COPY_CHUNK = 1024 * 1024


class SelectionCorpusError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * COPY_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionCorpusError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise SelectionCorpusError(f"{label} must be a JSON object")
    return value


def regular_file(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise SelectionCorpusError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
        raise SelectionCorpusError(f"{label} is not one regular non-symlink file")
    return info


def exact_directory(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise SelectionCorpusError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SelectionCorpusError(f"{label} is not one non-symlink directory")
    return info


def write_all(descriptor: int, block: bytes) -> None:
    remaining = memoryview(block)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise SelectionCorpusError("selection corpus copy made no progress")
        remaining = remaining[written:]


def stage(source: Path, destination: Path, runner_uid: int, runner_gid: int) -> dict[str, Any]:
    """Copy the exact selection inventory or validate a complete prior stage."""
    exact_directory(source, "protected selection corpus")
    exact_directory(destination, "durable selection corpus")
    try:
        present = list(destination.iterdir())
    except OSError as exc:
        raise SelectionCorpusError(f"cannot inventory durable selection corpus: {exc}") from exc
    if present and {path.name for path in present} != set(STAGED_NAMES):
        raise SelectionCorpusError(
            "durable selection staging inventory is partial or unexpected")

    for name in STAGED_NAMES:
        source_path = source / name
        source_info = regular_file(source_path, f"protected selection source {name}")
        output = destination / name
        if not present:
            source_fd = -1
            output_fd = -1
            try:
                source_fd = os.open(
                    source_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                opened = os.fstat(source_fd)
                if ((opened.st_dev, opened.st_ino) !=
                        (source_info.st_dev, source_info.st_ino)
                        or not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1):
                    raise SelectionCorpusError(
                        f"protected selection source changed while opening: {name}")
                output_fd = os.open(
                    output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
                while block := os.read(source_fd, COPY_CHUNK):
                    write_all(output_fd, block)
                os.fchown(output_fd, runner_uid, runner_gid)
                os.fchmod(output_fd, 0o400)
                os.fsync(output_fd)
            finally:
                if source_fd >= 0:
                    os.close(source_fd)
                if output_fd >= 0:
                    os.close(output_fd)

        staged = regular_file(output, f"durable selection artifact {name}")
        if (staged.st_uid != runner_uid or staged.st_gid != runner_gid
                or stat.S_IMODE(staged.st_mode) != 0o400):
            raise SelectionCorpusError(
                f"durable selection artifact metadata differs: {name}")

    os.chown(destination, runner_uid, runner_gid, follow_symlinks=False)
    os.chmod(destination, 0o500, follow_symlinks=False)
    directory = os.open(destination, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"status": "complete", "directory": str(destination),
            "files": list(STAGED_NAMES), "reused": bool(present)}


def verify(contract_path: Path, contract_sha256: str, corpus: Path,
           revision: str) -> dict[str, Any]:
    """Bind the staged selection bytes to the exact checked-in contract."""
    regular_file(contract_path, "corpus contract")
    if sha256(contract_path) != contract_sha256:
        raise SelectionCorpusError("corpus contract digest differs")
    exact_directory(corpus, "durable selection corpus")
    entries = list(corpus.iterdir())
    if {path.name for path in entries} != set(STAGED_NAMES) or len(entries) != len(STAGED_NAMES):
        raise SelectionCorpusError("selection staging exposed an unexpected corpus entry")
    for path in entries:
        regular_file(path, f"staged selection entry {path.name}")

    contract = read_object(contract_path, "corpus contract")
    manifest_path = corpus / MANIFEST_NAME
    manifest = read_object(manifest_path, "selection corpus manifest")
    derived = contract.get("derived_artifacts")
    if not isinstance(derived, dict):
        raise SelectionCorpusError("corpus contract has no derived artifact inventory")
    try:
        expected = {name: derived[name] for name in ARTIFACT_NAMES}
    except KeyError as exc:
        raise SelectionCorpusError(
            f"corpus contract lacks selection artifact: {exc.args[0]}") from exc
    rows = manifest.get("artifacts")
    by_name = ({row.get("filename"): row for row in rows if isinstance(row, dict)}
               if isinstance(rows, list) else {})
    if (contract.get("source", {}).get("revision") != revision
            or manifest.get("source", {}).get("revision") != revision
            or manifest.get("partition", {}).get("pairwise_request_overlap_count") != 0
            or set(by_name) != set(expected) or len(rows) != len(expected)):
        raise SelectionCorpusError(
            "selection corpus manifest does not match the pinned phase boundary")

    evidence: dict[str, Any] = {}
    for name, item in expected.items():
        if not isinstance(item, dict):
            raise SelectionCorpusError(f"corpus contract artifact is malformed: {name}")
        row = by_name[name]
        path = corpus / name
        digest = sha256(path)
        if (row.get("sha256") != item.get("sha256") or digest != item.get("sha256")
                or row.get("record_count") != item.get("record_count")):
            raise SelectionCorpusError(f"selection corpus artifact differs: {name}")
        evidence[name] = {"path": str(path), "sha256": digest,
                          "record_count": row["record_count"]}
    return {"status": "complete", "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path), "artifacts": evidence}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage")
    staging.add_argument("--source", type=Path, required=True)
    staging.add_argument("--destination", type=Path, required=True)
    staging.add_argument("--runner-uid", type=int, required=True)
    staging.add_argument("--runner-gid", type=int, required=True)
    validation = commands.add_parser("verify")
    validation.add_argument("--contract", type=Path, required=True)
    validation.add_argument("--contract-sha256", required=True)
    validation.add_argument("--corpus", type=Path, required=True)
    validation.add_argument("--revision", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "stage":
            result = stage(args.source.absolute(), args.destination.absolute(),
                           args.runner_uid, args.runner_gid)
        else:
            result = verify(args.contract.absolute(), args.contract_sha256,
                            args.corpus.absolute(), args.revision)
    except (OSError, SelectionCorpusError) as exc:
        print(f"qwen_selection_corpus_stage.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
