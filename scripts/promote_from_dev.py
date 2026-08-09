#!/usr/bin/env python3
"""Promote reviewed implementation changes from ember-dev without its history.

The two repositories intentionally have unrelated histories. This tool performs
a content-level three-way merge using the last promoted ember-dev commit as the
base, the requested ember-dev commit as "theirs", and the release worktree as
"ours". It never creates commits, stages files, fetches, or pushes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_FILE = pathlib.Path(".release/ember-dev-base")
LOG_FILE = pathlib.Path(".release/promotion-log.md")


@dataclass
class Change:
    status: str
    path: str
    policy: str
    decision: str = ""
    merge: str = ""
    content: bytes | None = None
    mode: int | None = None


def git(repo: pathlib.Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False
    )
    if check and result.returncode:
        message = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(message or f"git {' '.join(args)} failed")
    return result


def git_text(repo: pathlib.Path, *args: str) -> str:
    return git(repo, *args).stdout.decode(errors="strict").strip()


def classify(path: str) -> str:
    blocked = (
        path == ".env"
        or path == "scripts/capture_live_requests.py"
        or path == "test/fixtures_real_failures.h"
        or path.startswith("reports/")
        or "/__pycache__/" in f"/{path}"
        or path.endswith((".pyc", ".pyo"))
    )
    if blocked:
        return "blocked"
    if path == "engine/VENDOR.md":
        return "manual"
    automatic = (
        path == "CMakeLists.txt"
        or path.startswith(("src/", "engine/", "vendor/", "share/", "tools/"))
        or (
            path.startswith("test/")
            and pathlib.PurePosixPath(path).suffix
            in {".c", ".cc", ".cpp", ".h", ".py", ".json", ".jsonl"}
        )
    )
    return "automatic" if automatic else "manual"


def changed_paths(dev: pathlib.Path, base: str, target: str) -> list[Change]:
    raw = git(dev, "diff", "--name-status", "--no-renames", "-z", base, target, "--").stdout
    fields = raw.decode(errors="strict").split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    if len(fields) % 2:
        raise RuntimeError("unexpected git diff --name-status output")
    changes: list[Change] = []
    for i in range(0, len(fields), 2):
        status, path = fields[i], fields[i + 1]
        if status not in {"A", "M", "D"}:
            raise RuntimeError(f"unsupported change status {status} for {path}")
        changes.append(Change(status=status, path=path, policy=classify(path)))
    return changes


def blob(dev: pathlib.Path, ref: str, path: str) -> bytes:
    return git(dev, "show", f"{ref}:{path}").stdout


def target_mode(dev: pathlib.Path, ref: str, path: str) -> int:
    line = git_text(dev, "ls-tree", ref, "--", path)
    if not line:
        raise RuntimeError(f"cannot determine mode for {path} at {ref}")
    mode = line.split(None, 1)[0]
    return 0o755 if mode == "100755" else 0o644


def merge_change(release: pathlib.Path, dev: pathlib.Path, base: str,
                 target: str, change: Change) -> None:
    current_path = release / change.path
    current = current_path.read_bytes() if current_path.is_file() else None
    base_data = None if change.status == "A" else blob(dev, base, change.path)
    other = None if change.status == "D" else blob(dev, target, change.path)
    change.mode = None if other is None else target_mode(dev, target, change.path)

    if change.status == "A":
        if current is None:
            change.merge, change.content = "clean", other
        elif current == other:
            change.merge = "already-present"
        else:
            change.merge = "conflict"
        return
    if change.status == "D":
        if current is None:
            change.merge = "already-absent"
        elif current == base_data:
            change.merge, change.content = "clean-delete", None
        else:
            change.merge = "conflict"
        return
    if current is None:
        change.merge = "conflict"
        return
    if current == other:
        change.merge = "already-present"
        return
    if current == base_data:
        change.merge, change.content = "clean", other
        return
    if other == base_data:
        change.merge = "release-only"
        return
    if b"\0" in current or b"\0" in base_data or b"\0" in other:
        change.merge = "binary-conflict"
        return

    with tempfile.TemporaryDirectory(prefix="ember-promotion-") as directory:
        root = pathlib.Path(directory)
        ours, ancestor, theirs = root / "ours", root / "base", root / "theirs"
        ours.write_bytes(current)
        ancestor.write_bytes(base_data)
        theirs.write_bytes(other)
        result = subprocess.run(
            ["git", "merge-file", "-p", str(ours), str(ancestor), str(theirs)],
            capture_output=True,
            check=False,
        )
    if result.returncode == 0:
        change.merge, change.content = "clean-merge", result.stdout
    # git-merge-file reports the number of conflict regions (capped at 127),
    # so a file with several conflicts is still an expected merge result.
    elif 1 <= result.returncode <= 127:
        change.merge = "conflict"
    else:
        raise RuntimeError(
            result.stderr.decode(errors="replace").strip()
            or f"git merge-file failed for {change.path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=pathlib.Path, default=ROOT.parent / "ember-dev")
    parser.add_argument("--release", type=pathlib.Path, default=ROOT,
                        help=argparse.SUPPRESS)
    parser.add_argument("--target", default="HEAD", help="ember-dev commit to promote")
    parser.add_argument("--include", action="append", default=[], metavar="PATH",
                        help="include a manual path (repeatable)")
    parser.add_argument("--skip", action="append", default=[], metavar="PATH",
                        help="record a path as manually handled or intentionally omitted")
    parser.add_argument("--apply", action="store_true",
                        help="write the clean merge and advance the promotion boundary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    release = args.release.resolve()
    dev = args.dev.resolve()
    base_path = release / BASE_FILE
    if not (release / ".git").exists() or not (dev / ".git").exists():
        print("error: --release and --dev must identify Git repositories", file=sys.stderr)
        return 2
    if not base_path.is_file():
        print(f"error: missing promotion boundary {base_path}", file=sys.stderr)
        return 2
    base = base_path.read_text().strip()
    try:
        base = git_text(dev, "rev-parse", f"{base}^{{commit}}")
        target = git_text(dev, "rev-parse", f"{args.target}^{{commit}}")
        if git(dev, "merge-base", "--is-ancestor", base, target, check=False).returncode:
            raise RuntimeError("promotion target is not a descendant of the recorded base")
        changes = changed_paths(dev, base, target)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    known = {change.path for change in changes}
    includes, skips = set(args.include), set(args.skip)
    unknown = (includes | skips) - known
    overlap = includes & skips
    if unknown or overlap:
        if unknown:
            print(f"error: decision paths are unchanged: {', '.join(sorted(unknown))}", file=sys.stderr)
        if overlap:
            print(f"error: paths cannot be included and skipped: {', '.join(sorted(overlap))}", file=sys.stderr)
        return 2

    unresolved: list[str] = []
    conflicts: list[str] = []
    for change in changes:
        if change.path in skips:
            change.decision = "skip"
        elif change.path in includes:
            if change.policy == "blocked":
                print(f"error: blocked path cannot be included: {change.path}", file=sys.stderr)
                return 2
            change.decision = "include"
        elif change.policy == "automatic":
            change.decision = "include"
        else:
            change.decision = "unresolved"
            unresolved.append(change.path)
        if change.decision == "include":
            try:
                merge_change(release, dev, base, target, change)
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            if "conflict" in change.merge:
                conflicts.append(change.path)

    print(f"ember-dev range: {base[:12]}..{target[:12]}")
    commits = git_text(dev, "log", "--reverse", "--format=%h %s", f"{base}..{target}")
    if commits:
        print("commits:")
        for line in commits.splitlines():
            print(f"  {line}")
    print("paths:")
    for change in changes:
        outcome = change.merge or "-"
        print(f"  {change.status} {change.policy:9} {change.decision:10} {outcome:15} {change.path}")

    if not args.apply:
        if unresolved:
            print("plan: manual decisions required; use --include or --skip for each path")
        if conflicts:
            print("plan: conflicts require a manual port followed by --skip acknowledgement")
        return 0

    if unresolved or conflicts:
        if unresolved:
            print(f"error: unresolved manual paths: {', '.join(unresolved)}", file=sys.stderr)
        if conflicts:
            print(f"error: merge conflicts: {', '.join(conflicts)}", file=sys.stderr)
        return 2
    if git(release, "status", "--porcelain", "--untracked-files=all").stdout:
        print("error: release worktree must be clean before --apply", file=sys.stderr)
        return 2
    if git(dev, "status", "--porcelain", "--untracked-files=all").stdout:
        print("error: ember-dev worktree must be clean before --apply", file=sys.stderr)
        return 2

    for change in changes:
        if change.decision != "include" or change.merge in {
            "already-present", "already-absent", "release-only"
        }:
            continue
        path = release / change.path
        if change.merge == "clean-delete":
            path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(change.content if change.content is not None else b"")
        if change.mode is not None:
            os.chmod(path, change.mode)

    base_path.write_text(target + "\n")
    log_path = release / LOG_FILE
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    skipped = [change.path for change in changes if change.decision == "skip"]
    included = [change.path for change in changes if change.decision == "include"]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n## {timestamp}: `{base[:12]}..{target[:12]}`\n\n")
        log.write(f"- Source commit: `{target}`\n")
        log.write(f"- Automatically or explicitly included: {', '.join(f'`{p}`' for p in included) or 'none'}\n")
        log.write(f"- Manually handled or omitted: {', '.join(f'`{p}`' for p in skipped) or 'none'}\n")
    print("applied: review the worktree, run release validation, then commit as one promotion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
