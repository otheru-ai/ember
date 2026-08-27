#!/usr/bin/env python3
"""Adapt the pinned OtherU prompt fixtures into disjoint Qwen corpora.

The output is deterministic and contains no timestamps or source paths. The
upstream overtrigger fixture is pinned as excluded evidence because that file
contains environment-derived material; a separately reviewed sanitized
tool-use corpus is required by the bakeoff instead.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from typing import Any, Iterable


REPOSITORY = "https://git.otheru.ai/akadmin/otheru-quant-pipeline"
BRANCH = "ember-contract-and-drafter-fix"
REVISION = "a3c6a728510f91394e991504951ac316cd3a89af"
RAW_FILES = {
    "benign_draft.json": "646629c7b5f280cad4da035decd81babaec57f912fa22fecf2c51490ed761385",
    "coherence.json": "339941410fc32c2d024cb1de21cc0105ad0a97ea52e58cce3819366342f767bb",
    "harmful.txt": "fa2036d1d556e22cccb99c3a7004759fd37c00e0404c858b32463dcf9a722a36",
    "harmful32.txt": "302f2d93e5bebfbcd891fd27cc67507565ae5e6153d648ecd04ebbbfd6fc9e50",
    "harmless.txt": "64832649b46b34c08a125621f629b91f300ccf21b71e558d6301e1c932ca8888",
    "harmless32.txt": "0cf3980114ecf5f8cce432efb36648468237719f93f5c2ae9990dcfa0f70119c",
    "overrefusal.txt": "e082ef7f3b6d513399a0ee97f2979b70726ae52d72c4781daf5ae0122cf24232",
    "overtrigger.json": "acd30da1ee1260d24acdc189cc3c15d4b242f924c33aa2f1ecaf5f3f8749f09d",
}
OUTPUT_NAMES = (
    "extraction-good.jsonl", "extraction-bad.jsonl",
    "sweep-validation.jsonl", "final-heldout.jsonl",
)


class CorpusError(ValueError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).strip()
    if not normalized or "\0" in normalized:
        raise CorpusError("upstream corpus contains an empty or NUL prompt")
    return normalized


def request_fingerprint(messages: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        {"messages": messages}, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def record(source: str, index: int, prompt: str, suite: str,
           expected: dict[str, Any]) -> dict[str, Any]:
    messages = [{"role": "user", "content": canonical_text(prompt)}]
    return {
        "id": f"otheru-{source}-{index:03d}",
        "messages": messages,
        "suite": suite,
        "expected": expected,
        "source": {"file": source, "index": index},
        "request_sha256": request_fingerprint(messages),
    }


def lines(path: Path) -> list[str]:
    values = [canonical_text(item) for item in path.read_text(encoding="utf-8").splitlines()
              if item.strip()]
    if len(values) != len(set(values)):
        raise CorpusError(f"upstream corpus has duplicate lines: {path.name}")
    return values


def split_half(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda item: (item["request_sha256"], item["id"]))
    if len(ordered) % 2:
        raise CorpusError("held-out suite must have an even record count")
    midpoint = len(ordered) // 2
    return ordered[:midpoint], ordered[midpoint:]


def json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"cannot parse pinned source {path.name}: {exc}") from exc


def verify_source(source_dir: Path) -> Path:
    try:
        revision = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise CorpusError("--source-dir must be a git checkout") from exc
    if revision != REVISION:
        raise CorpusError(f"OtherU checkout must be exactly {REVISION}, got {revision}")
    prompt_dir = source_dir / "tools" / "abliterate" / "prompts"
    for name, expected in RAW_FILES.items():
        path = prompt_dir / name
        if not path.is_file() or path.is_symlink():
            raise CorpusError(f"pinned source must be a regular non-symlink file: {name}")
        actual = sha256_bytes(path.read_bytes())
        if actual != expected:
            raise CorpusError(f"pinned source digest mismatch: {name}")
    return prompt_dir


def build_rows(prompt_dir: Path) -> dict[str, list[dict[str, Any]]]:
    harmful = lines(prompt_dir / "harmful.txt")
    harmful32 = lines(prompt_dir / "harmful32.txt")
    harmless = lines(prompt_dir / "harmless.txt")
    harmless32 = lines(prompt_dir / "harmless32.txt")
    if harmful[:32] != harmful32 or harmless[:32] != harmless32:
        raise CorpusError("pinned 32-record extraction sets are not exact source prefixes")

    extraction_good = [
        record("harmless32", index, prompt, "direction_extraction_good",
               {"behavior": "helpful_control"})
        for index, prompt in enumerate(harmless32)
    ]
    extraction_bad = [
        record("harmful32", index, prompt, "direction_extraction_bad",
               {"behavior": "refusal_target"})
        for index, prompt in enumerate(harmful32)
    ]
    evaluation_suites: list[list[dict[str, Any]]] = [
        [record("harmful", index, prompt, "refusal_bypass",
                {"behavior": "answer_without_stock_refusal"})
         for index, prompt in enumerate(harmful[32:], start=32)],
        [record("harmless", index, prompt, "helpfulness_control",
                {"behavior": "helpful"})
         for index, prompt in enumerate(harmless[32:], start=32)],
        [record("overrefusal", index, prompt, "overrefusal",
                {"behavior": "helpful"})
         for index, prompt in enumerate(lines(prompt_dir / "overrefusal.txt"))],
    ]

    coherence = json_value(prompt_dir / "coherence.json")
    if not isinstance(coherence, list) or len(coherence) != 8:
        raise CorpusError("pinned coherence corpus must contain eight cases")
    coherence_rows = []
    for index, item in enumerate(coherence):
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise CorpusError("pinned coherence case is malformed")
        expected = {key: value for key, value in item.items()
                    if key not in ("prompt", "_why", "id")}
        expected["behavior"] = "coherent"
        coherence_rows.append(record("coherence", index, item["prompt"],
                                     "coherence", expected))
    evaluation_suites.append(coherence_rows)

    benign = json_value(prompt_dir / "benign_draft.json")
    prompts = benign.get("benign_prompts") if isinstance(benign, dict) else None
    if not isinstance(prompts, list) or len(prompts) != 36 or not all(
        isinstance(item, str) for item in prompts
    ):
        raise CorpusError("pinned benign capability corpus must contain 36 prompts")
    evaluation_suites.append([
        record("benign-draft", index, prompt, "benign_capability",
               {"behavior": "coherent"})
        for index, prompt in enumerate(prompts)
    ])

    sweep: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    for suite in evaluation_suites:
        left, right = split_half(suite)
        sweep.extend(left)
        final.extend(right)
    sweep.sort(key=lambda item: (item["suite"], item["request_sha256"]))
    final.sort(key=lambda item: (item["suite"], item["request_sha256"]))
    return {
        "extraction-good.jsonl": extraction_good,
        "extraction-bad.jsonl": extraction_bad,
        "sweep-validation.jsonl": sweep,
        "final-heldout.jsonl": final,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def publish_noreplace(stage: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CorpusError("Linux renameat2 is required for no-clobber publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(stage), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise CorpusError(f"refusing to overwrite corpus directory: {destination}")
        raise CorpusError(f"cannot publish corpus directory: {os.strerror(error)}")


def generate(source_dir: Path, destination: Path) -> dict[str, Any]:
    prompt_dir = verify_source(source_dir)
    rows = build_rows(prompt_dir)
    fingerprints = {
        name: {item["request_sha256"] for item in items}
        for name, items in rows.items()
    }
    names = list(fingerprints)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if fingerprints[left] & fingerprints[right]:
                raise CorpusError(f"derived corpora overlap: {left}/{right}")
    destination = destination.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage: Path | None = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.transaction-", dir=destination.parent
    ))
    try:
        artifacts = []
        for name in OUTPUT_NAMES:
            path = stage / name
            write_jsonl(path, rows[name])
            artifacts.append({
                "filename": name, "record_count": len(rows[name]),
                "sha256": sha256_bytes(path.read_bytes()),
                "suite_counts": {
                    suite: sum(item["suite"] == suite for item in rows[name])
                    for suite in sorted({item["suite"] for item in rows[name]})
                },
            })
        manifest = {
            "schema_version": 1,
            "source": {"repository": REPOSITORY, "branch": BRANCH,
                       "revision": REVISION},
            "raw_files": [
                {"path": f"tools/abliterate/prompts/{name}", "sha256": digest,
                 "included": name != "overtrigger.json"}
                for name, digest in sorted(RAW_FILES.items())
            ],
            "excluded_sources": [{
                "path": "tools/abliterate/prompts/overtrigger.json",
                "reason": "environment-derived private material; require a separately reviewed sanitized tool-use corpus",
            }],
            "partition": {
                "algorithm": "sha256(canonical_messages), lexical sort, equal halves",
                "extraction_sets": "exact pinned 32-record prefixes",
                "pairwise_request_overlap_count": 0,
                "sweep_data_may_select_recipe": True,
                "final_heldout_may_select_recipe": False,
            },
            "artifacts": artifacts,
        }
        manifest_path = stage / "qwen-corpora-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        directory = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        publish_noreplace(stage, destination)
        stage = None
        return manifest
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = generate(args.source_dir, args.output_dir)
    except (CorpusError, OSError) as exc:
        print(f"qwen_corpus_adapter.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.output_dir.absolute()),
                      "artifacts": manifest["artifacts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
