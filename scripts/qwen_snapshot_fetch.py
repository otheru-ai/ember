#!/usr/bin/env python3
"""Resumably fetch and verify the pinned Qwen snapshot using bounded memory."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import urllib.error
import urllib.parse
import urllib.request


CHUNK = 8 * 1024 * 1024


class FetchError(ValueError):
    pass


def _load_inventory(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FetchError(f"cannot read inventory {path}: {exc}") from exc
    files = value.get("files")
    if not isinstance(files, list) or value.get("file_count") != len(files):
        raise FetchError("inventory file_count is invalid")
    total = 0
    seen: set[str] = set()
    for row in files:
        if not isinstance(row, dict):
            raise FetchError("inventory rows must be objects")
        raw = row.get("path")
        rel = PurePosixPath(raw) if isinstance(raw, str) else PurePosixPath("/")
        if rel.is_absolute() or not rel.parts or any(part in ("", ".", "..") for part in rel.parts):
            raise FetchError(f"unsafe inventory path: {raw!r}")
        if raw in seen:
            raise FetchError(f"duplicate inventory path: {raw}")
        seen.add(raw)
        size = row.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise FetchError(f"invalid size for {raw}")
        sha256 = row.get("sha256")
        git_blob = row.get("git_blob")
        if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != 64):
            raise FetchError(f"invalid SHA-256 for {raw}")
        if not isinstance(git_blob, str) or len(git_blob) != 40:
            raise FetchError(f"invalid Git blob id for {raw}")
        total += size
    if value.get("total_bytes") != total:
        raise FetchError("inventory total_bytes is invalid")
    return value


def _digests(path: Path, size: int) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1(f"blob {size}\0".encode("ascii"), usedforsecurity=False)
    with path.open("rb") as source:
        while block := source.read(CHUNK):
            sha256.update(block)
            sha1.update(block)
    return sha256.hexdigest(), sha1.hexdigest()


def _verified(path: Path, row: dict) -> tuple[bool, str]:
    try:
        if path.stat().st_size != row["size"]:
            return False, "size"
        sha256, git_blob = _digests(path, row["size"])
    except OSError as exc:
        return False, type(exc).__name__
    expected_sha = row.get("sha256")
    if expected_sha is not None and sha256 != expected_sha:
        return False, "sha256"
    if expected_sha is None and git_blob != row["git_blob"]:
        return False, "git_blob"
    return True, sha256 if expected_sha is not None else git_blob


def _fetch_one(base_url: str, output: Path, row: dict) -> dict:
    target = output.joinpath(*PurePosixPath(row["path"]).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, digest = _verified(target, row)
    if ok:
        return {"path": row["path"], "bytes": row["size"], "digest": digest, "cached": True}

    partial = target.with_name(f".{target.name}.partial")
    try:
        offset = partial.stat().st_size
    except FileNotFoundError:
        offset = 0
    if offset > row["size"]:
        partial.unlink()
        offset = 0

    url = base_url.rstrip("/") + "/" + urllib.parse.quote(row["path"], safe="/")
    headers = {"User-Agent": "ember-qwen-snapshot-fetch/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset == row["size"]:
            response = None
        else:
            raise FetchError(f"download failed for {row['path']}: HTTP {exc.code}") from exc

    if response is not None:
        status = getattr(response, "status", response.getcode())
        mode = "ab" if offset and status == 206 else "wb"
        if mode == "wb":
            offset = 0
        if status not in (200, 206):
            response.close()
            raise FetchError(f"download failed for {row['path']}: HTTP {status}")
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                response.close()
                raise FetchError(f"invalid resume response for {row['path']}")
        try:
            with partial.open(mode) as destination:
                while block := response.read(CHUNK):
                    destination.write(block)
                destination.flush()
                os.fsync(destination.fileno())
        finally:
            response.close()

    ok, digest = _verified(partial, row)
    if not ok:
        raise FetchError(f"verification failed for {row['path']}: {digest}")
    os.replace(partial, target)
    try:
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        with target.open("rb") as source:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    return {"path": row["path"], "bytes": row["size"], "digest": digest, "cached": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    try:
        inventory = _load_inventory(args.inventory)
        if args.workers < 1 or args.workers > 16:
            raise FetchError("--workers must be in [1, 16]")
        files = inventory["files"]
        if args.max_files is not None:
            if args.max_files < 0:
                raise FetchError("--max-files must be non-negative")
            files = files[:args.max_files]
        base_url = args.base_url or (
            f"https://huggingface.co/{inventory['repo_id']}/resolve/{inventory['revision']}"
        )
        plan = {
            "schema": "ember.qwen3.8.snapshot-fetch.v1",
            "repo_id": inventory["repo_id"],
            "revision": inventory["revision"],
            "output": str(args.output),
            "file_count": len(files),
            "bytes": sum(row["size"] for row in files),
            "workers": args.workers,
        }
        if args.plan:
            print(json.dumps(plan, sort_keys=True))
            return 0
        args.output.mkdir(parents=True, exist_ok=True)
        lock_path = args.output / ".ember-fetch.lock"
        with lock_path.open("w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            results = []
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(_fetch_one, base_url, args.output, row): row
                    for row in files
                }
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(json.dumps(result, sort_keys=True), flush=True)
        summary = dict(plan)
        summary.update({
            "verified_files": len(results),
            "verified_bytes": sum(item["bytes"] for item in results),
            "downloaded_files": sum(not item["cached"] for item in results),
            "complete": len(files) == inventory["file_count"],
        })
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (FetchError, OSError, KeyError) as exc:
        print(f"qwen_snapshot_fetch: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
