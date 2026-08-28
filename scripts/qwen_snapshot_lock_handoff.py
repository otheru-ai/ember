#!/usr/bin/env python3
"""Hand one completed Qwen snapshot coordination inode to the runner.

The resumable fetch can run as root while candidate construction deliberately
runs as the unprivileged self-hosted-runner user.  Both operations coordinate
through ``.ember-fetch.lock``.  This helper changes metadata on that one inode
only: it opens the snapshot directory and lock without following symlinks,
proves that the name still denotes the singly-linked regular inode it opened,
and takes a nonblocking exclusive flock before changing owner/mode through the
descriptor.  Holding that flock preserves the fetch/read-lease protocol across
the handoff and makes an active or newly-starting fetch fail closed.

No inventory entry or model payload is enumerated, opened, read, or written.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import stat
from typing import Any


LOCK_NAME = ".ember-fetch.lock"


class HandoffError(RuntimeError):
    """The named coordination inode is unsafe or currently leased."""


def _normalized_absolute(path: Path, label: str) -> Path:
    raw = os.fspath(path)
    if not path.is_absolute() or os.path.normpath(raw) != raw:
        raise HandoffError(f"{label} is not one normalized absolute path")
    return path


def _open_directory_chain(path: Path) -> int:
    """Resolve every absolute component with descriptor-relative no-follow opens."""
    flags = (os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
             | getattr(os, "O_CLOEXEC", 0))
    descriptor = -1
    try:
        descriptor = os.open(path.anchor, flags)
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise HandoffError("snapshot is not one directory")
        result = descriptor
        descriptor = -1
        return result
    except HandoffError:
        raise
    except OSError as error:
        raise HandoffError(
            "cannot resolve snapshot without following directory symlinks"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _named_lock_status(directory: int) -> os.stat_result:
    try:
        return os.stat(LOCK_NAME, dir_fd=directory, follow_symlinks=False)
    except OSError as error:
        raise HandoffError(f"cannot inspect snapshot coordination inode: {error}") from error


def _validate_lock(metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise HandoffError("snapshot coordination inode is not a regular non-symlink file")
    if metadata.st_nlink != 1:
        raise HandoffError("snapshot coordination inode must have exactly one hard link")


def handoff_snapshot_lock(
    snapshot: Path, *, runner_uid: int, runner_gid: int, verify_only: bool = False,
) -> dict[str, Any]:
    """Validate, exclusively lease, and optionally hand off the exact lock inode."""
    snapshot = _normalized_absolute(snapshot, "snapshot directory")
    if runner_uid <= 0 or runner_gid <= 0:
        raise HandoffError("runner uid/gid must identify an unprivileged account")

    directory = -1
    descriptor = -1
    locked = False
    try:
        directory = _open_directory_chain(snapshot)

        # O_RDWR is intentional: the fetch writer and the shared construction
        # lease both use this inode, and the privileged handoff must be able to
        # change metadata through the already-validated descriptor.
        lock_flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(LOCK_NAME, lock_flags, dir_fd=directory)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise HandoffError(
                    "snapshot coordination inode is not a regular non-symlink file"
                ) from error
            raise HandoffError("cannot open snapshot coordination inode") from error

        opened = os.fstat(descriptor)
        named = _named_lock_status(directory)
        _validate_lock(opened)
        _validate_lock(named)
        if not _same_inode(opened, named):
            raise HandoffError("snapshot coordination name changed while opening it")

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as error:
            raise HandoffError(
                "snapshot coordination inode is held by an active fetch or read lease"
            ) from error

        # Re-resolve only the lock name after taking exclusive ownership.  A
        # rename cannot redirect the descriptor, but it must prevent mutation
        # because the workflow authorized the exact named coordination inode.
        named_locked = _named_lock_status(directory)
        _validate_lock(named_locked)
        if not _same_inode(opened, named_locked):
            raise HandoffError("snapshot coordination name changed before metadata handoff")

        if not verify_only:
            os.fchown(descriptor, runner_uid, runner_gid)
            os.fchmod(descriptor, 0o600)

        final = os.fstat(descriptor)
        named_final = _named_lock_status(directory)
        _validate_lock(final)
        _validate_lock(named_final)
        if not _same_inode(final, named_final):
            raise HandoffError("snapshot coordination name changed during metadata handoff")
        if (final.st_uid, final.st_gid, stat.S_IMODE(final.st_mode)) != (
                runner_uid, runner_gid, 0o600):
            raise HandoffError("snapshot coordination ownership handoff did not persist")
        return {
            "schema": "ember.qwen3.8.snapshot-lock-handoff.v1",
            "status": "verified" if verify_only else "handed_off",
            "path": str(snapshot / LOCK_NAME),
            "device": final.st_dev,
            "inode": final.st_ino,
            "links": final.st_nlink,
            "uid": final.st_uid,
            "gid": final.st_gid,
            "mode": f"{stat.S_IMODE(final.st_mode):04o}",
            "exclusive_idle_proof": True,
            "model_bytes_touched": 0,
        }
    finally:
        if descriptor >= 0:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        if directory >= 0:
            os.close(directory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--runner-uid", type=int, required=True)
    parser.add_argument("--runner-gid", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = handoff_snapshot_lock(
            args.snapshot, runner_uid=args.runner_uid, runner_gid=args.runner_gid,
            verify_only=args.verify_only,
        )
    except HandoffError as error:
        raise SystemExit(f"qwen_snapshot_lock_handoff: {error}") from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
