#!/usr/bin/env python3
"""Render a Dwarfstar frozen-continuation manifest as a deterministic text corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").rstrip("\r\n")
    except OSError as exc:
        raise SystemExit(f"{path}: {exc}") from exc


def render(manifest: Path, shuffle_seed: int | None = None) -> tuple[str, int]:
    rows = []
    try:
        with manifest.open(encoding="utf-8", newline="") as fp:
            reader = csv.reader((line for line in fp if not line.startswith("#")), delimiter="\t")
            for line_no, fields in enumerate(reader, 1):
                if not fields:
                    continue
                if len(fields) < 3:
                    raise SystemExit(f"{manifest}:{line_no}: expected id, prompt, continuation")
                case_id, prompt_file, continuation_file = fields[:3]
                prompt = read_text(Path(prompt_file))
                continuation = read_text(Path(continuation_file))
                rows.append(
                    f"Evaluation case {case_id}\n\nPrompt:\n{prompt}\n\nResponse:\n{continuation}"
                )
    except OSError as exc:
        raise SystemExit(f"{manifest}: {exc}") from exc
    if not rows:
        raise SystemExit(f"{manifest}: no cases")
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    return "\n\n---\n\n".join(rows) + "\n", len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shuffle-seed", type=int)
    args = parser.parse_args()
    corpus, cases = render(args.manifest, args.shuffle_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = corpus.encode("utf-8")
    args.out.write_bytes(data)
    print(f"corpus: {args.out}")
    print(f"cases: {cases}")
    print(f"bytes: {len(data)}")
    print(f"shuffle_seed: {args.shuffle_seed}")
    print(f"sha256: {hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
