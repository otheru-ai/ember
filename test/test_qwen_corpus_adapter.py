#!/usr/bin/env python3
"""GPU-free contract tests for the pinned OtherU corpus adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen_corpus_adapter", ROOT / "scripts" / "qwen_corpus_adapter.py"
)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
PINNED_REVISION = adapter.REVISION
PINNED_RAW_FILES = dict(adapter.RAW_FILES)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CorpusAdapterTest(unittest.TestCase):
    def tearDown(self) -> None:
        adapter.REVISION = PINNED_REVISION
        adapter.RAW_FILES = dict(PINNED_RAW_FILES)

    def test_checked_in_contract_matches_adapter_pin(self) -> None:
        contract = json.loads(
            (ROOT / "share" / "quant_eval" /
             "qwen3.8-otheru-corpus-contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(contract["source"]["repository"], adapter.REPOSITORY)
        self.assertEqual(contract["source"]["branch"], adapter.BRANCH)
        self.assertEqual(contract["source"]["revision"], adapter.REVISION)
        self.assertEqual(contract["raw_files"], adapter.RAW_FILES)
        self.assertEqual(contract["pairwise_request_overlap_count"], 0)

    def make_source(self, root: Path) -> Path:
        source = root / "source"
        prompts = source / "tools" / "abliterate" / "prompts"
        prompts.mkdir(parents=True)
        harmful = [f"harmful request {index}" for index in range(128)]
        harmless = [f"harmless request {index}" for index in range(128)]
        files = {
            "harmful.txt": "\n".join(harmful) + "\n",
            "harmful32.txt": "\n".join(harmful[:32]) + "\n",
            "harmless.txt": "\n".join(harmless) + "\n",
            "harmless32.txt": "\n".join(harmless[:32]) + "\n",
            "overrefusal.txt": "\n".join(
                f"overrefusal request {index}" for index in range(32)
            ) + "\n",
            "coherence.json": json.dumps([
                {"prompt": f"coherence request {index}", "minimum_words": index + 1}
                for index in range(8)
            ]),
            "benign_draft.json": json.dumps({
                "benign_prompts": [f"benign request {index}" for index in range(36)]
            }),
            "overtrigger.json": json.dumps({"excluded_private_marker": "must-not-escape"}),
        }
        for name, value in files.items():
            (prompts / name).write_text(value, encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(source), "-c", "user.name=Test",
             "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
            check=True,
        )
        adapter.REVISION = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True, text=True, stdout=subprocess.PIPE,
        ).stdout.strip()
        adapter.RAW_FILES = {name: sha256(prompts / name) for name in files}
        return source

    @staticmethod
    def rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_deterministic_disjoint_partition_and_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            first = root / "first"
            second = root / "second"
            manifest = adapter.generate(source, first)
            adapter.generate(source, second)

            expected = {
                "extraction-good.jsonl": 32,
                "extraction-bad.jsonl": 32,
                "sweep-validation.jsonl": 134,
                "final-heldout.jsonl": 134,
            }
            fingerprints = []
            for name, count in expected.items():
                left = first / name
                right = second / name
                self.assertEqual(left.read_bytes(), right.read_bytes())
                rows = self.rows(left)
                self.assertEqual(len(rows), count)
                fingerprints.append({row["request_sha256"] for row in rows})
            for index, left in enumerate(fingerprints):
                for right in fingerprints[index + 1:]:
                    self.assertFalse(left & right)
            self.assertEqual(manifest["partition"]["pairwise_request_overlap_count"], 0)
            self.assertNotIn(
                "must-not-escape",
                (first / "qwen-corpora-manifest.json").read_text(encoding="utf-8"),
            )

    def test_refuses_existing_destination_and_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            destination = root / "corpora"
            adapter.generate(source, destination)
            with self.assertRaisesRegex(adapter.CorpusError, "refusing to overwrite"):
                adapter.generate(source, destination)
            prompt = source / "tools" / "abliterate" / "prompts" / "harmful.txt"
            prompt.write_text(prompt.read_text(encoding="utf-8") + "drift\n", encoding="utf-8")
            with self.assertRaisesRegex(adapter.CorpusError, "digest mismatch"):
                adapter.generate(source, root / "drift")


if __name__ == "__main__":
    unittest.main()
