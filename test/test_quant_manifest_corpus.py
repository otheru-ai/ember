#!/usr/bin/env python3
"""Checks deterministic rendering of a frozen continuation manifest."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quant_manifest_corpus", ROOT / "scripts" / "quant_manifest_corpus.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "p.txt").write_text("question\n", encoding="utf-8")
        (root / "c.txt").write_text("answer\n", encoding="utf-8")
        manifest = root / "manifest.tsv"
        manifest.write_text(
            "# id\tprompt_file\tcontinuation_file\tresponse_file\n"
            f"case_000\t{root / 'p.txt'}\t{root / 'c.txt'}\tunused.json\n",
            encoding="utf-8",
        )
        corpus, cases = MODULE.render(manifest)
        assert cases == 1
        assert corpus == "Evaluation case case_000\n\nPrompt:\nquestion\n\nResponse:\nanswer\n"
        assert MODULE.render(manifest, 7301) == (corpus, cases)
    print("quant manifest corpus: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
