#!/usr/bin/env python3
"""GPU-free contract tests for the fixed 2026.8.24 performance gate."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load_module("ember_benchmark", ROOT / "scripts/bench/benchmark.py")
assemble = load_module("ember_assemble", ROOT / "scripts/bench/assemble_bundle.py")


def row(group: str, rate: float, repeat: int) -> dict:
    result = {
        "kind": "request",
        "group": group,
        "repeat": repeat,
        "ok": True,
    }
    if group == "prefill-2048":
        result.update({
            "evaluated_prefill_tokens": 2074,
            "prefill_tokens_per_second": rate,
        })
    else:
        result.update({
            "completion_tokens": 256,
            "decode_tokens_per_second": rate,
        })
    return result


class BenchmarkGateTest(unittest.TestCase):
    def records(self) -> list[dict]:
        return [
            row("prefill-2048", value, index + 1)
            for index, value in enumerate((411.0, 412.0, 413.0))
        ] + [
            row("decode-256", value, index + 1)
            for index, value in enumerate((39.29, 39.49, 40.44))
        ]

    def test_exact_reference_medians_pass(self) -> None:
        gate = benchmark.evaluate_hard_gate(
            self.records(), prefill_target=412.0, decode_target=39.49)
        self.assertEqual(gate["protocol"], benchmark.HARD_GATE_PROTOCOL)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["prefill_2048"]["median_tps"], 412.0)
        self.assertEqual(gate["decode_256_counting"]["median_tps"], 39.49)

    def test_wrong_shape_or_missing_sample_fails(self) -> None:
        records = self.records()
        records[0]["evaluated_prefill_tokens"] = 2073
        records[-1]["completion_tokens"] = 255
        gate = benchmark.evaluate_hard_gate(
            records, prefill_target=412.0, decode_target=39.49)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["prefill_2048"]["shape_match"])
        self.assertFalse(gate["decode_256_counting"]["shape_match"])
        self.assertFalse(benchmark.evaluate_hard_gate(
            records[:-1], prefill_target=1.0, decode_target=1.0)["passed"])

    def test_bundle_assembly_preserves_machine_gate(self) -> None:
        gate = benchmark.evaluate_hard_gate(
            self.records(), prefill_target=412.0, decode_target=39.49)
        with tempfile.TemporaryDirectory() as raw:
            bundle = Path(raw)
            records = self.records() + [{"kind": "summary", "hard_gate": gate}]
            (bundle / "raw-results.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records))
            loaded = assemble.jsonl(bundle / "raw-results.jsonl")
            summaries = [item for item in loaded if item.get("kind") == "summary"]
            self.assertEqual(summaries[-1]["hard_gate"], gate)

    def test_qwen_shape_calibration_uses_prompt_tokens(self) -> None:
        class FakeSuite:
            def __init__(self) -> None:
                self.words: list[int] = []

            def request(self, _label: str, prompt: str, _max_tokens: int,
                        *, group: str, repeat: int) -> dict:
                del group, repeat
                words = prompt.count(" alpha")
                self.words.append(words)
                # Simulate a Qwen template with 31 tokens of overhead rather
                # than the DeepSeek protocol's 26.
                return {"ok": True, "prompt_tokens": words + 31,
                        "evaluated_prefill_tokens": 1}

        fake = FakeSuite()
        words, attempts = benchmark.calibrate_prefill_words(fake, 2074)
        self.assertEqual(words, 2043)
        self.assertEqual(fake.words, [2048, 2043])
        self.assertEqual(attempts[-1]["prompt_tokens"], 2074)


if __name__ == "__main__":
    unittest.main()
