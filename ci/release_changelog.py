#!/usr/bin/env python3
"""Prepare deterministic CalVer release metadata and extract release notes."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
VERSION = ROOT / "VERSION"
COMPOSE = ROOT / "compose.yaml"

VERSION_RE = re.compile(r"^(\d{4})\.([1-9]|1[0-2])\.([1-9]|[12]\d|3[01])$")
COMMIT_RE = re.compile(r"^([a-z]+)(?:\(([^)]+)\))?(!)?:\s+(.+)$")
GROUPS = (
    ("Added", {"feat"}),
    ("Fixed", {"fix"}),
    ("Performance", {"perf"}),
    ("Changed", {"refactor"}),
    ("Build and CI", {"build", "ci"}),
    ("Documentation", {"docs"}),
    ("Testing", {"test"}),
    ("Maintenance", {"chore"}),
)


class ReleaseError(RuntimeError):
    pass


def run_git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.PIPE
        ).strip()
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise ReleaseError(f"git {' '.join(args)} failed: {detail}") from exc


def validate_version(version: str) -> None:
    match = VERSION_RE.fullmatch(version)
    if not match:
        raise ReleaseError(f"invalid CalVer {version!r}; expected YEAR.M.D")
    try:
        dt.date(*(int(part) for part in match.groups()))
    except ValueError as exc:
        raise ReleaseError(f"invalid calendar date {version!r}: {exc}") from exc


def commits_since(previous_tag: str) -> list[tuple[str, str]]:
    raw = subprocess.check_output(
        ["git", "log", "-z", "--format=%H%x00%s", f"{previous_tag}..HEAD"],
        cwd=ROOT,
    ).decode("utf-8", "replace")
    fields = [field for field in raw.split("\0") if field]
    if len(fields) % 2:
        raise ReleaseError("git log returned an incomplete commit record")
    return list(zip(fields[0::2], fields[1::2]))


def markdown_subject(subject: str) -> str:
    return subject.replace("\\", "\\\\").replace("`", "\\`")


def generated_sections(commits: list[tuple[str, str]]) -> str:
    grouped: dict[str, list[str]] = {title: [] for title, _ in GROUPS}
    grouped["Other"] = []
    type_to_group = {
        commit_type: title for title, types in GROUPS for commit_type in types
    }
    for sha, subject in commits:
        parsed = COMMIT_RE.match(subject)
        if parsed:
            commit_type, scope, breaking, summary = parsed.groups()
            title = type_to_group.get(commit_type, "Other")
            prefix = f"**{scope}:** " if scope else ""
            if breaking:
                prefix = "**Breaking:** " + prefix
            text = prefix + markdown_subject(summary)
        else:
            title = "Other"
            text = markdown_subject(subject)
        grouped[title].append(f"- {text} (`{sha[:8]}`)")

    sections: list[str] = []
    for title, _ in GROUPS:
        if grouped[title]:
            sections.append(f"### {title}\n\n" + "\n".join(grouped[title]))
    if grouped["Other"]:
        sections.append("### Other\n\n" + "\n".join(grouped["Other"]))
    return "\n\n".join(sections)


def split_unreleased(text: str) -> tuple[str, str, str]:
    marker = "## Unreleased"
    start = text.find(marker)
    if start < 0:
        raise ReleaseError("CHANGELOG.md has no ## Unreleased section")
    content_start = start + len(marker)
    next_release = text.find("\n## ", content_start)
    if next_release < 0:
        next_release = len(text)
    return text[:start], text[content_start:next_release].strip(), text[next_release:]


def replace_image_version(path: pathlib.Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    needle = f"ghcr.io/otheru-ai/ember:{old}"
    count = text.count(needle)
    if count != 1:
        raise ReleaseError(f"{path.name} contains {count} occurrences of {needle!r}")
    path.write_text(text.replace(needle, f"ghcr.io/otheru-ai/ember:{new}"), encoding="utf-8")


def prepare(version: str, previous_tag: str | None) -> None:
    validate_version(version)
    old_version = VERSION.read_text(encoding="utf-8").strip()
    validate_version(old_version)
    if tuple(map(int, version.split("."))) <= tuple(map(int, old_version.split("."))):
        raise ReleaseError(f"release {version} must be newer than VERSION {old_version}")
    if run_git("tag", "--list", f"v{version}"):
        raise ReleaseError(f"tag v{version} already exists")

    previous_tag = previous_tag or run_git("describe", "--tags", "--abbrev=0", "--match", "v[0-9]*")
    commits = commits_since(previous_tag)
    if not commits:
        raise ReleaseError(f"no commits found after {previous_tag}")

    prefix, curated, suffix = split_unreleased(CHANGELOG.read_text(encoding="utf-8"))
    parts = [f"## {version}"]
    if curated:
        parts.append("### Curated notes\n\n" + curated)
    parts.append(generated_sections(commits))
    release_section = "\n\n".join(part for part in parts if part)
    CHANGELOG.write_text(
        prefix + "## Unreleased\n\n" + release_section + suffix,
        encoding="utf-8",
    )
    VERSION.write_text(version + "\n", encoding="utf-8")
    replace_image_version(COMPOSE, old_version, version)


def notes(version: str) -> str:
    validate_version(version)
    text = CHANGELOG.read_text(encoding="utf-8")
    marker = f"## {version}"
    start = text.find(marker)
    if start < 0:
        raise ReleaseError(f"CHANGELOG.md has no {marker} section")
    content_start = start + len(marker)
    end = text.find("\n## ", content_start)
    if end < 0:
        end = len(text)
    return text[content_start:end].strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="update release metadata in place")
    prep.add_argument("--version", required=True)
    prep.add_argument("--previous-tag")
    extract = sub.add_parser("notes", help="print one changelog release section")
    extract.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.version, args.previous_tag)
        else:
            sys.stdout.write(notes(args.version))
    except ReleaseError as exc:
        sys.stderr.write(f"release error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
