#!/usr/bin/env python3
"""Cache recent direct-I/O verification of immutable Qwen artifacts.

The cache never replaces first verification.  A hit requires the requested
digest plus the file's device, inode, size, mtime, and ctime to match a recent
record.  Any uncertainty falls back to a full content hash.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


SCHEMA = "ember.qwen3.8.artifact-integrity-cache.v1"
HEX64 = re.compile(r"[0-9a-f]{64}")
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60
DIRECT_THRESHOLD = 512 * 1024 * 1024


class IntegrityError(RuntimeError):
    pass


def identity(status: os.stat_result) -> dict[str, int]:
    return {
        "device": status.st_dev,
        "inode": status.st_ino,
        "size": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "ctime_ns": status.st_ctime_ns,
    }


def hash_file(path: Path, direct_threshold: int = DIRECT_THRESHOLD) -> str:
    before = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"artifact is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    count = 0
    if before.st_size >= direct_threshold:
        process = subprocess.Popen(
            ["dd", f"if={path}", "iflag=direct", "bs=8M", "status=none"],
            stdout=subprocess.PIPE,
        )
        assert process.stdout is not None
        for block in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
            digest.update(block)
            count += len(block)
        if process.wait() != 0:
            raise IntegrityError(f"O_DIRECT hash failed: {path}")
    else:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                count += len(block)
    after = path.lstat()
    if count != before.st_size or identity(before) != identity(after):
        raise IntegrityError(f"artifact changed while hashed: {path}")
    return digest.hexdigest()


class IntegrityCache:
    def __init__(self, path: Path, ttl_seconds: int = DEFAULT_TTL_SECONDS,
                 now: int | None = None) -> None:
        if not path.is_absolute() or ttl_seconds <= 0:
            raise IntegrityError("cache path must be absolute and TTL must be positive")
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.now = int(time.time()) if now is None else now
        self.records: dict[str, Any] = {}
        self.lock_fd = -1
        self.dirty = False

    def __enter__(self) -> "IntegrityCache":
        if not self.path.parent.is_dir() or self.path.parent.is_symlink():
            raise IntegrityError("integrity-cache parent must be an existing directory")
        if ((self.path.exists() or self.path.is_symlink())
                and (self.path.is_symlink() or not self.path.is_file())):
            raise IntegrityError("integrity cache must be a regular non-symlink file")
        lock = self.path.with_name(self.path.name + ".lock")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        self.lock_fd = os.open(lock, flags, 0o600)
        fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if (not isinstance(value, dict) or set(value) != {"schema", "records"}
                    or value.get("schema") != SCHEMA
                    or not isinstance(value.get("records"), dict)):
                raise ValueError("cache contract differs")
            self.records = value["records"]
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError,
                ValueError):
            self.records = {}
            self.dirty = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            if exc_type is None and self.dirty:
                self._flush()
        finally:
            if self.lock_fd >= 0:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
                self.lock_fd = -1

    def verify(self, path: Path, expected: str,
               direct_threshold: int = DIRECT_THRESHOLD) -> bool:
        if (self.lock_fd < 0 or not path.is_absolute()
                or HEX64.fullmatch(expected) is None):
            raise IntegrityError("verification requires an open cache and exact path/digest")
        status = path.lstat()
        if path.is_symlink() or not path.is_file():
            raise IntegrityError(f"artifact is not a regular non-symlink file: {path}")
        file_identity = identity(status)
        record = self.records.get(str(path), {})
        if not isinstance(record, dict):
            record = {}
        try:
            age = self.now - int(record.get("verified_at", 0))
        except (TypeError, ValueError):
            age = self.ttl_seconds
        if (record.get("digest") == expected
                and record.get("identity") == file_identity
                and 0 <= age < self.ttl_seconds):
            print(f"{path}: OK (integrity cache hit, age {age}s)", file=sys.stderr)
            return True
        actual = hash_file(path, direct_threshold)
        if actual != expected:
            raise IntegrityError(f"artifact digest differs: {path}")
        self.records[str(path)] = {
            "digest": actual,
            "identity": file_identity,
            "verified_at": self.now,
        }
        self.dirty = True
        print(f"{path}: OK (fresh {'O_DIRECT ' if status.st_size >= direct_threshold else ''}hash)",
              file=sys.stderr)
        return False

    def _flush(self) -> None:
        raw = (json.dumps({"schema": SCHEMA, "records": self.records},
                          indent=2, sort_keys=True) + "\n").encode()
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=True) as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent,
                                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary.exists():
                temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)
    parser.add_argument("--file", nargs=2, action="append", default=[],
                        metavar=("PATH", "SHA256"))
    args = parser.parse_args(argv)
    if not args.file:
        parser.error("at least one --file PATH SHA256 pair is required")
    try:
        with IntegrityCache(args.cache, args.ttl_seconds) as cache:
            for path, digest in args.file:
                cache.verify(Path(path), digest)
        return 0
    except (IntegrityError, OSError, ValueError) as error:
        print(f"qwen-integrity-cache: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
