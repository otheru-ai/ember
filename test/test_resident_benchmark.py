#!/usr/bin/env python3
"""Unit coverage for the resident hardware benchmark's report contract."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "benchmark_resident", ROOT / "scripts" / "benchmark_resident.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class ResidentBenchmarkTests(unittest.TestCase):
    def test_response_metrics_supports_completion_payload(self):
        payload = {
            "choices": [{"text": "answer"}],
            "usage": {
                "completion_tokens": 4,
                "accept_rate": 0.75,
                "timings": {
                    "prefill_ms": 12.5,
                    "prefill_tokens_per_sec": 1000.0,
                    "decode_ms": 20.0,
                    "decode_tokens_per_sec": 200.0,
                },
                "backend": {"spec_ran": True, "prefill_mode": "exact"},
            },
        }
        row = BENCH.response_metrics(payload, 0.025, "prompt")
        self.assertEqual(row["completion_tokens"], 4)
        self.assertEqual(row["request_wall_ms"], 25.0)
        self.assertEqual(row["prefill_ms"], 12.5)
        self.assertTrue(row["spec_ran"])
        self.assertEqual(row["prefill_mode"], "exact")

    def test_summary_uses_concurrent_round_wall_time(self):
        rounds = [
            {
                "wall_ms": 100.0,
                "completion_tokens": 8,
                "aggregate_tokens_per_second": 80.0,
                "rows": [
                    {
                        "request_wall_ms": 90.0,
                        "prefill_ms": 10.0,
                        "decode_tokens_per_second": 50.0,
                        "accept_rate": 0.5,
                        "spec_ran": True,
                    },
                    {
                        "request_wall_ms": 100.0,
                        "prefill_ms": 20.0,
                        "decode_tokens_per_second": 40.0,
                        "accept_rate": 0.75,
                        "spec_ran": True,
                    },
                ],
            },
            {
                "wall_ms": 200.0,
                "completion_tokens": 8,
                "aggregate_tokens_per_second": 40.0,
                "rows": [
                    {
                        "request_wall_ms": 180.0,
                        "prefill_ms": 30.0,
                        "decode_tokens_per_second": 30.0,
                        "accept_rate": 0.25,
                        "spec_ran": True,
                    },
                    {
                        "request_wall_ms": 200.0,
                        "prefill_ms": 40.0,
                        "decode_tokens_per_second": 20.0,
                        "accept_rate": 0.0,
                        "spec_ran": False,
                    },
                ],
            },
        ]
        status_before = {"continuous_batching": {
            "max_decode_batch": 1, "decode_batches": 10, "decode_rows": 10,
        }}
        status_after = {"continuous_batching": {
            "max_decode_batch": 2, "decode_batches": 12, "decode_rows": 14,
        }}
        summary = BENCH.summarize(rounds, status_before, status_after)
        self.assertAlmostEqual(summary["aggregate_tokens_per_second"],
                               16 / 0.3)
        self.assertEqual(summary["spec_rows"], 3)
        self.assertAlmostEqual(summary["accept_rate_mean"], 0.5)
        self.assertEqual(summary["max_decode_batch_after"], 2)
        self.assertEqual(summary["decode_batches_delta"], 2)
        self.assertEqual(summary["mean_decode_batch"], 2.0)
        self.assertEqual(summary["backend_prefill_ms_mean"], 25.0)

    def test_reference_requires_identical_output_hashes(self):
        outputs = {"prompt": ["output"]}
        BENCH.compare_reference(outputs, {"outputs": outputs})
        with self.assertRaises(RuntimeError):
            BENCH.compare_reference(outputs,
                                    {"outputs": {"prompt": ["different"]}})

    def test_report_is_json_serializable(self):
        report = {"schema_version": 1, "outputs": {"p": ["o"]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            self.assertEqual(json.loads(path.read_text()), report)


if __name__ == "__main__":
    unittest.main()
