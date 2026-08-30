#!/usr/bin/env python3
"""Classify candidate-builder changes for safe immutable-artifact reuse.

Qwen candidate weights depend on the builder's construction path, but not on
its post-build retirement and reconstruction bookkeeping.  This helper strips
only an explicit set of retirement-only top-level definitions before comparing
two revisions.  Every other syntax change is content-affecting by default.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys


HEX40 = re.compile(r"[0-9a-f]{40}")
BUILDER_PATH = "scripts/qwen_candidate_builder.py"
RETENTION_ONLY_CONSTANTS = frozenset({
    "RECONSTRUCTABLE_RETIREMENT_SCHEMA",
    "RECONSTRUCTABLE_RETIREMENT_COMPLETE_SCHEMA",
})
RETENTION_ONLY_FUNCTIONS = frozenset({
    "_validate_reconstruction_intervention",
    "_record_reconstruction_contract",
    "retire_reconstructable",
    "restore_reconstructable",
})
ROUTING_FUNCTIONS = frozenset({"main", "parser"})


class ScopeError(ValueError):
    pass


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: set[str] = set()
    for target in targets:
        if not isinstance(target, ast.Name):
            return set()
        names.add(target.id)
    return names


def content_fingerprint(source: str) -> str:
    """Return a location-independent AST for content-affecting builder code."""
    try:
        module = ast.parse(source, filename=BUILDER_PATH)
    except SyntaxError as exc:
        raise ScopeError(f"candidate builder is not valid Python: {exc}") from exc

    retained: list[ast.stmt] = []
    omitted: dict[str, ast.stmt] = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in RETENTION_ONLY_FUNCTIONS:
                omitted[node.name] = node
                continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = _assigned_names(node)
            if names and names <= RETENTION_ONLY_CONSTANTS:
                for name in names:
                    omitted[name] = node
                continue
        retained.append(node)

    # If construction code ever starts referring to an otherwise omitted
    # definition, that definition immediately rejoins the fingerprint.
    referenced: set[str] = set()
    for parent in retained:
        # CLI construction and dispatch necessarily name every subcommand.
        # Their own AST remains in the fingerprint, but those references do
        # not make a retirement implementation content-affecting.
        if (isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                and parent.name in ROUTING_FUNCTIONS):
            continue
        referenced.update(
            node.id for node in ast.walk(parent) if isinstance(node, ast.Name))
    reinstate_ids = {id(omitted[name]) for name in referenced if name in omitted}
    normalized = [
        node for node in module.body
        if node in retained or id(node) in reinstate_ids
    ]
    return ast.dump(ast.Module(body=normalized, type_ignores=[]),
                    annotate_fields=True, include_attributes=False)


def revision_source(repository: Path, revision: str) -> str:
    if not repository.is_absolute() or not repository.is_dir():
        raise ScopeError("repository must be one absolute directory")
    if HEX40.fullmatch(revision) is None:
        raise ScopeError("revision must be one full 40-character SHA")
    result = subprocess.run(
        ["git", "-C", str(repository), "show", f"{revision}:{BUILDER_PATH}"],
        check=False, capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ScopeError(f"cannot read candidate builder at {revision}: {detail}")
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ScopeError("candidate builder is not UTF-8") from exc


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository", type=Path, required=True)
    result.add_argument("--base", required=True)
    result.add_argument("--target", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        before = content_fingerprint(revision_source(args.repository, args.base))
        after = content_fingerprint(revision_source(args.repository, args.target))
    except (OSError, ScopeError) as exc:
        print(f"qwen_builder_change_scope.py: error: {exc}", file=sys.stderr)
        return 2
    changed = before != after
    print(json.dumps({
        "base": args.base,
        "target": args.target,
        "builder_path": BUILDER_PATH,
        "artifact_content_changed": changed,
        "classification": ("content_affecting_rebuild_required" if changed
                           else "unchanged_or_retention_only_reuse_allowed"),
    }, sort_keys=True))
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
