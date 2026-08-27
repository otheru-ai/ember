#!/usr/bin/env python3
"""Narrow ownership handoff for authorized Qwen stock-shard retirement.

The retirement builder deliberately runs as the unprivileged runner user.  A
captured control can have root-owned shard inodes because conversion ran in a
container, so deletion needs one privileged handoff first.  This helper treats
the already-fsynced workflow authorization as the complete authority: it opens
only the fixed stock directory and its listed shard inodes with no-follow,
descriptor-relative operations, validates the whole scope before changing any
metadata, and never enumerates or opens retained files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


WORKSPACE = Path("/var/tmp/ember-qwen3.8-flash-next")
AUTH_SCHEMA = "ember.qwen3.8.stock-retirement-workflow-authorization.v1"
ACKNOWLEDGEMENT = "RETIRE_CAPTURED_STOCK_SHARDS"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
SHARD_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,253}\.gguf")


class HandoffError(RuntimeError):
    """Authorization or filesystem state is unsafe for a metadata handoff."""


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise HandoffError(f"authorization contains duplicate key: {key}")
        value[key] = item
    return value


def _normalized_absolute(path: Path, label: str) -> Path:
    raw = os.fspath(path)
    if not path.is_absolute() or os.path.normpath(raw) != raw:
        raise HandoffError(f"{label} is not one normalized absolute path")
    return path


def _reject_symlink_ancestors(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise HandoffError(f"{label} does not exist: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise HandoffError(f"{label} traverses a symlink: {current}")


def _open_directory(path: Path, label: str) -> int:
    _normalized_absolute(path, label)
    _reject_symlink_ancestors(path, label)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open(path, flags)
    except OSError as error:
        raise HandoffError(f"cannot open {label} as one real directory: {path}") from error


def _open_child_directory(parent: int, name: str, label: str) -> int:
    if Path(name).name != name or name in ("", ".", ".."):
        raise HandoffError(f"invalid {label} component")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except OSError as error:
        raise HandoffError(f"cannot open {label} without following links") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise HandoffError(f"{label} is not one directory")
    return descriptor


def _load_authorization(directory: int, name: str) -> tuple[dict[str, Any], int]:
    if Path(name).name != name or not name.endswith(".workflow.json"):
        raise HandoffError("workflow authorization has an unsafe filename")
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    except OSError as error:
        raise HandoffError("cannot open workflow authorization without following links") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise HandoffError("workflow authorization is not one singly-linked regular file")
    try:
        with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_object_without_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        os.close(descriptor)
        raise HandoffError("workflow authorization is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        os.close(descriptor)
        raise HandoffError("workflow authorization root is not an object")
    return value, descriptor


def _validate_identity(
    value: dict[str, Any], *, source_revision: str, stock_revision: str,
    image_digest: str, workspace: Path, authorization: Path,
    builder_authorization: Path,
) -> Path:
    if REVISION_RE.fullmatch(source_revision) is None:
        raise HandoffError("source revision is malformed")
    if REVISION_RE.fullmatch(stock_revision) is None:
        raise HandoffError("stock artifact revision is malformed")
    if IMAGE_DIGEST_RE.fullmatch(image_digest) is None:
        raise HandoffError("builder image digest is malformed")
    expected_workset = workspace / "artifacts" / "qwen-workset"
    expected_auth_parent = expected_workset / "evidence" / "stock-retirement"
    expected_stock = workspace / "artifacts" / f"stock-rocmi4-{stock_revision[:12]}"
    if authorization != Path(f"{builder_authorization}.workflow.json"):
        raise HandoffError("workflow authorization is not paired with builder authorization")
    if authorization.parent != expected_auth_parent:
        raise HandoffError("workflow authorization escapes the fixed evidence directory")
    if builder_authorization.parent != expected_auth_parent:
        raise HandoffError("builder authorization escapes the fixed evidence directory")
    if (value.get("schema") != AUTH_SCHEMA
            or value.get("status") != "authorized_before_quiesce"
            or value.get("authorization_phrase") != ACKNOWLEDGEMENT
            or value.get("publishes") is not False):
        raise HandoffError("workflow authorization state is not destructive-authorized")
    if value.get("source") != {"revision": source_revision}:
        raise HandoffError("workflow authorization source revision differs")
    if value.get("stock_artifact") != {"revision": stock_revision}:
        raise HandoffError("workflow authorization stock revision differs")
    image = value.get("image")
    if not isinstance(image, dict) or image.get("digest") != image_digest:
        raise HandoffError("workflow authorization image digest differs")
    if value.get("workset_root") != str(expected_workset):
        raise HandoffError("workflow authorization workset differs from the fixed path")
    if value.get("stock_dir") != str(expected_stock):
        raise HandoffError("workflow authorization stock directory differs from the fixed path")
    if value.get("builder_authorization") != str(builder_authorization):
        raise HandoffError("workflow authorization builder output differs")
    return expected_stock


def handoff_authorized_shards(
    *, authorization: Path, builder_authorization: Path, source_revision: str,
    stock_revision: str, image_digest: str, runner_uid: int, runner_gid: int,
    verify_only: bool = False, workspace: Path = WORKSPACE,
) -> list[tuple[str, int]]:
    """Validate and hand off exactly the shard inodes named by authorization."""
    workspace = _normalized_absolute(workspace, "workspace")
    authorization = _normalized_absolute(authorization, "workflow authorization")
    builder_authorization = _normalized_absolute(
        builder_authorization, "builder authorization")
    if runner_uid < 0 or runner_gid < 0:
        raise HandoffError("runner uid/gid must be non-negative")

    descriptors: list[int] = []
    shard_descriptors: list[tuple[str, int, int]] = []
    try:
        artifacts = _open_directory(workspace / "artifacts", "artifacts directory")
        descriptors.append(artifacts)
        workset = _open_child_directory(artifacts, "qwen-workset", "workset directory")
        descriptors.append(workset)
        evidence = _open_child_directory(workset, "evidence", "evidence directory")
        descriptors.append(evidence)
        retirement = _open_child_directory(
            evidence, "stock-retirement", "retirement evidence directory")
        descriptors.append(retirement)
        value, authorization_descriptor = _load_authorization(
            retirement, authorization.name)
        descriptors.append(authorization_descriptor)
        expected_stock = _validate_identity(
            value, source_revision=source_revision, stock_revision=stock_revision,
            image_digest=image_digest, workspace=workspace,
            authorization=authorization, builder_authorization=builder_authorization)
        stock = _open_child_directory(
            artifacts, expected_stock.name, "fixed stock directory")
        descriptors.append(stock)

        scope = value.get("deletion_scope")
        if not isinstance(scope, list) or not scope:
            raise HandoffError("workflow authorization has no deletion scope")
        names: set[str] = set()
        inodes: set[tuple[int, int]] = set()
        for index, row in enumerate(scope):
            if not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}:
                raise HandoffError(f"deletion scope row {index} is malformed")
            path_value = row["path"]
            size = row["size_bytes"]
            digest = row["sha256"]
            if not isinstance(path_value, str):
                raise HandoffError(f"deletion scope path {index} is not a string")
            path = _normalized_absolute(Path(path_value), f"deletion scope path {index}")
            if path.parent != expected_stock or SHARD_NAME_RE.fullmatch(path.name) is None:
                raise HandoffError(f"deletion scope path {index} is outside the fixed stock directory")
            if path.name in names:
                raise HandoffError("deletion scope repeats one shard filename")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise HandoffError(f"deletion scope size {index} is invalid")
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                raise HandoffError(f"deletion scope digest {index} is invalid")
            try:
                descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=stock)
            except OSError as error:
                raise HandoffError(f"cannot open authorized shard without following links: {path}") from error
            metadata = os.fstat(descriptor)
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
                    or metadata.st_size != size):
                os.close(descriptor)
                raise HandoffError(f"authorized shard inode differs: {path}")
            inode = (metadata.st_dev, metadata.st_ino)
            if inode in inodes:
                os.close(descriptor)
                raise HandoffError("deletion scope aliases one shard inode")
            names.add(path.name)
            inodes.add(inode)
            shard_descriptors.append((path.name, descriptor, size))

        # Every descriptor is validated before the first metadata mutation.
        # Do not stream the ~85 GiB shard set through page cache here: the
        # unprivileged retirement builder hashes every shard against the same
        # authorized digest immediately before unlinking it.  This handoff
        # validates the digest shape plus the descriptor's identity and size.
        if not verify_only:
            for _, descriptor, _ in shard_descriptors:
                os.fchown(descriptor, runner_uid, runner_gid)
                os.fchmod(descriptor, 0o600)
            os.fchown(stock, runner_uid, runner_gid)
            os.fchmod(stock, 0o700)
            os.fsync(stock)

        stock_metadata = os.fstat(stock)
        if (stock_metadata.st_uid, stock_metadata.st_gid,
                stat.S_IMODE(stock_metadata.st_mode)) != (runner_uid, runner_gid, 0o700):
            raise HandoffError("fixed stock directory ownership handoff did not persist")
        for name, descriptor, size in shard_descriptors:
            metadata = os.fstat(descriptor)
            if (metadata.st_uid, metadata.st_gid, stat.S_IMODE(metadata.st_mode),
                    metadata.st_size) != (runner_uid, runner_gid, 0o600, size):
                raise HandoffError(f"authorized shard ownership handoff did not persist: {name}")
        return [(name, size) for name, _, size in shard_descriptors]
    finally:
        for _, descriptor, _ in reversed(shard_descriptors):
            os.close(descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--builder-authorization", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--stock-artifact-revision", required=True)
    parser.add_argument("--builder-image-digest", required=True)
    parser.add_argument("--runner-uid", type=int, required=True)
    parser.add_argument("--runner-gid", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = handoff_authorized_shards(
            authorization=args.authorization,
            builder_authorization=args.builder_authorization,
            source_revision=args.source_revision,
            stock_revision=args.stock_artifact_revision,
            image_digest=args.builder_image_digest,
            runner_uid=args.runner_uid, runner_gid=args.runner_gid,
            verify_only=args.verify_only)
    except HandoffError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"verified_shards": len(rows),
                      "verified_bytes": sum(size for _, size in rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
