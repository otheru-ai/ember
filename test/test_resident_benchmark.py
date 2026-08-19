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
                    "spec_cycles": 2,
                    "spec_provider_age_ms": 160.0,
                    "spec_provider_block_ms": 20.0,
                    "spec_head_ms": 6.0,
                    "spec_verify_ms": 120.0,
                },
                "backend": {"spec_ran": True, "prefill_mode": "exact"},
            },
        }
        row = BENCH.response_metrics(payload, 0.025, "prompt")
        self.assertEqual(row["completion_tokens"], 4)
        self.assertEqual(row["request_wall_ms"], 25.0)
        self.assertEqual(row["prefill_ms"], 12.5)
        self.assertEqual(row["spec_provider_block_ms"], 20.0)
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
                        "spec_cycles": 2,
                        "spec_provider_age_ms": 160.0,
                        "spec_provider_block_ms": 20.0,
                        "spec_head_ms": 6.0,
                        "spec_verify_ms": 120.0,
                    },
                    {
                        "request_wall_ms": 100.0,
                        "prefill_ms": 20.0,
                        "decode_tokens_per_second": 40.0,
                        "accept_rate": 0.75,
                        "spec_ran": True,
                        "spec_cycles": 2,
                        "spec_provider_age_ms": 180.0,
                        "spec_provider_block_ms": 40.0,
                        "spec_head_ms": 8.0,
                        "spec_verify_ms": 140.0,
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
                        "spec_cycles": 2,
                        "spec_provider_age_ms": 200.0,
                        "spec_provider_block_ms": 60.0,
                        "spec_head_ms": 10.0,
                        "spec_verify_ms": 160.0,
                    },
                    {
                        "request_wall_ms": 200.0,
                        "prefill_ms": 40.0,
                        "decode_tokens_per_second": 20.0,
                        "accept_rate": 0.0,
                        "spec_ran": False,
                        "spec_cycles": 0,
                        "spec_provider_age_ms": 0.0,
                        "spec_provider_block_ms": 0.0,
                        "spec_head_ms": 0.0,
                        "spec_verify_ms": 0.0,
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
        self.assertAlmostEqual(
            summary["round_tokens_per_second_stdev"], 28.284271247461902)
        self.assertEqual(summary["spec_cycles"], 6)
        self.assertEqual(summary["spec_provider_block_ms_total"], 120.0)
        self.assertEqual(summary["spec_provider_block_ms_per_cycle"], 20.0)
        self.assertEqual(summary["spec_verify_ms_per_cycle"], 70.0)

    @staticmethod
    def _round(wall_ms):
        return {
            "wall_ms": wall_ms,
            "completion_tokens": 128,
            "aggregate_tokens_per_second": 128000.0 / wall_ms,
            "rows": [],
        }

    def test_reference_promotes_only_confident_exact_speedup(self):
        outputs = {"prompt": ["output"]}
        config = {
            "model": "model",
            "prompt_sha256": "prompt",
            "max_tokens": 64,
            "concurrency": 2,
        }
        baseline = [self._round(value) for value in (200.0, 210.0, 190.0)]
        candidate = [self._round(value) for value in (100.0, 105.0, 95.0)]
        reference = {
            "config": config,
            "outputs": outputs,
            "round_results": baseline,
        }
        comparison = BENCH.compare_reference(
            outputs, config, candidate, reference, 1.5, 0.95, 2000)
        self.assertTrue(comparison["promoted"])
        self.assertTrue(comparison["config_exact"])
        self.assertTrue(comparison["outputs_exact"])
        self.assertAlmostEqual(comparison["observed_speedup"], 2.0)
        self.assertGreater(comparison["speedup_lower_bound"], 1.5)

        mismatch_config = dict(config)
        mismatch_config["concurrency"] = 1
        rejected = BENCH.compare_reference(
            {"prompt": ["different"]}, mismatch_config, candidate,
            reference, 2.1, 0.95, 2000)
        self.assertFalse(rejected["promoted"])
        self.assertFalse(rejected["config_exact"])
        self.assertFalse(rejected["outputs_exact"])
        self.assertEqual(len(rejected["failures"]), 3)

    def test_bootstrap_speedup_is_deterministic(self):
        baseline = [self._round(value) for value in (100.0, 105.0, 95.0)]
        candidate = [self._round(value) for value in (90.0, 92.0, 88.0)]
        first = BENCH.bootstrap_speedup(
            candidate, baseline, 0.95, 1000)
        second = BENCH.bootstrap_speedup(
            candidate, baseline, 0.95, 1000)
        self.assertEqual(first, second)
        self.assertGreater(first["observed_speedup"], 1.0)
        self.assertGreater(first["speedup_lower_bound"], 1.0)

    def test_report_is_json_serializable(self):
        report = {"schema_version": 1, "outputs": {"p": ["o"]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            self.assertEqual(json.loads(path.read_text()), report)


if __name__ == "__main__":
    unittest.main()
