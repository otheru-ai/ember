#!/usr/bin/env python3
"""GPU-free contract tests for the fixed 2026.8.24 performance gate."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from unittest import mock
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
        milliseconds = 2074 * 1000.0 / rate
        result.update({
            "evaluated_prefill_tokens": 2074,
            "prefill_ms": milliseconds,
            "declared_prefill_tokens_per_second": round(rate, 1),
            "prefill_tokens_per_second": 2074 * 1000.0 / milliseconds,
            "prefill_tps_rounding_consistent": True,
        })
    else:
        milliseconds = 256 * 1000.0 / rate
        result.update({
            "completion_tokens": 256,
            "decode_ms": milliseconds,
            "declared_decode_tokens_per_second": round(rate, 2),
            "decode_tokens_per_second": 256 * 1000.0 / milliseconds,
            "decode_tps_rounding_consistent": True,
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

    def test_exact_reference_prefill_peak_and_decode_median_pass(self) -> None:
        gate = benchmark.evaluate_hard_gate(
            self.records(), prefill_target=412.0, decode_target=39.49)
        self.assertEqual(gate["protocol"], benchmark.HARD_GATE_PROTOCOL)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["prefill_2048"]["statistic"], "peak")
        self.assertEqual(gate["prefill_2048"]["peak_tps"], 413.0)
        self.assertEqual(gate["prefill_2048"]["median_tps"], 412.0)
        self.assertEqual(gate["decode_256_counting"]["statistic"], "median")
        self.assertEqual(gate["decode_256_counting"]["median_tps"], 39.49)

    def test_prefill_peak_can_pass_when_median_is_below_target(self) -> None:
        records = self.records()
        for row_value, value in zip(records[:3], (390.0, 400.0, 412.0), strict=True):
            row_value.update(row("prefill-2048", value, row_value["repeat"]))
        gate = benchmark.evaluate_hard_gate(
            records, prefill_target=412.0, decode_target=39.49)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["prefill_2048"]["peak_tps"], 412.0)
        self.assertLess(gate["prefill_2048"]["median_tps"], 412.0)

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

    def test_rates_are_derived_and_declared_values_are_only_audit_evidence(self) -> None:
        derived, consistent = benchmark.derived_tps(
            2074, 5000.0, 414.8, declared_decimals=1)
        self.assertAlmostEqual(derived, 414.8)
        self.assertTrue(consistent)
        derived, consistent = benchmark.derived_tps(
            256, 6400.0, 999.0, declared_decimals=2)
        self.assertEqual(derived, 40.0)
        self.assertFalse(consistent)
        self.assertEqual(
            benchmark.derived_tps(256, 0.0, 40.0, declared_decimals=2),
            (None, False))

        records = self.records()
        records[2]["prefill_tokens_per_second"] = 900.0
        records[2]["prefill_tps_rounding_consistent"] = False
        gate = benchmark.evaluate_hard_gate(
            records, prefill_target=412.0, decode_target=39.49)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["prefill_2048"]["shape_match"])

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

    def test_qwen_decode_calibration_records_and_selects_exact_shape(self) -> None:
        class FakeSuite:
            def __init__(self) -> None:
                self.markers: list[str] = []

            def request(self, _label: str, prompt: str, _max_tokens: int,
                        *, group: str, repeat: int) -> dict:
                del group, repeat
                marker = prompt.split("Marker ", 1)[1].split(".", 1)[0]
                self.markers.append(marker)
                lengths = {"D": 25, "E": 1, "F": 256}
                length = lengths[marker]
                return {"ok": True, "completion_tokens": length,
                        "finish_reason": "length" if length == 256 else "stop"}

        fake = FakeSuite()
        marker, attempts = benchmark.calibrate_decode_marker(fake, 256)
        self.assertEqual(marker, "F")
        self.assertEqual(fake.markers, ["D", "E", "F"])
        self.assertEqual([row["completion_tokens"] for row in attempts],
                         [25, 1, 256])

    def test_qwen_gate_protocol_is_bound_in_machine_evidence(self) -> None:
        gate = benchmark.evaluate_hard_gate(
            self.records(), prefill_target=412.0, decode_target=39.49,
            protocol=benchmark.QWEN_HARD_GATE_PROTOCOL)
        self.assertEqual(gate["protocol"], benchmark.QWEN_HARD_GATE_PROTOCOL)

    def test_http_error_preserves_bounded_backend_response(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            suite = benchmark.Suite(
                "http://127.0.0.1:1/v1/chat/completions",
                Path(raw) / "timing.jsonl", 1.0, "test")
            body = b'{"error":{"message":"backend failed"}}' + b"x" * 5000
            error = urllib.error.HTTPError(
                suite.endpoint, 500, "Internal Server Error", {}, io.BytesIO(body))
            with mock.patch.object(benchmark.urllib.request, "urlopen",
                                   side_effect=error), \
                    mock.patch.object(suite, "emit"):
                record = suite.request(
                    "repro", "prompt", 1, group="diagnostic", repeat=1)
            self.assertFalse(record["ok"])
            self.assertEqual(record["http_status"], 500)
            self.assertTrue(record["response_body"].startswith(
                '{"error":{"message":"backend failed"}}'))
            self.assertEqual(len(record["response_body"].encode()), 4096)
            self.assertTrue(record["response_body_truncated"])

    def test_request_preserves_and_summarizes_speculative_timings(self) -> None:
        body = {
            "choices": [{"finish_reason": "length"}],
            "usage": {
                "prompt_tokens": 64,
                "completion_tokens": 256,
                "accept_rate": 0.25,
                "timings": {
                    "prefill_tokens": 64,
                    "prefill_ms": 100.0,
                    "prefill_tokens_per_sec": 640.0,
                    "decode_ms": 6400.0,
                    "decode_tokens_per_sec": 40.0,
                    "spec_cycles": 32,
                    "spec_provider_age_ms": 320.0,
                    "spec_provider_block_ms": 64.0,
                    "spec_head_ms": 96.0,
                    "spec_verify_ms": 3200.0,
                },
                "backend": {"spec_ran": True},
            },
        }
        with tempfile.TemporaryDirectory() as raw:
            suite = benchmark.Suite(
                "http://127.0.0.1:1/v1/chat/completions",
                Path(raw) / "timing.jsonl", 1.0, "test")
            response = io.BytesIO(json.dumps(body).encode())
            with mock.patch.object(benchmark.urllib.request, "urlopen",
                                   return_value=response):
                record = suite.request(
                    "decode", "prompt", 256, group="decode-256", repeat=1)
            self.assertEqual(record["spec_cycles"], 32)
            self.assertEqual(record["spec_head_ms"], 96.0)
            self.assertEqual(record["spec_verify_ms"], 3200.0)
            speculation = suite.summarize()["decode-256"]["speculation"]
            self.assertTrue(speculation["timing_complete"])
            self.assertEqual(speculation["cycles"], 32)
            self.assertEqual(speculation["accept_rate_mean"], 0.25)
            self.assertEqual(speculation["spec_head_ms_per_cycle"], 3.0)
            self.assertEqual(speculation["spec_verify_ms_per_cycle"], 100.0)

    def test_quant_build_record_binds_complete_ordered_shards(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            shards = []
            for index, data in enumerate((b"one", b"two"), 1):
                name = f"qwen-rocmi4-{index:05d}-of-00002.gguf"
                path = root / name
                path.write_bytes(data)
                shards.append({
                    "path": f"/qwen-work/final/{name}",
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                })
            record = {
                "status": "complete", "mode": "execute",
                "output": {"shards": shards},
                "memory_preflight": {
                    "shard_count": 2, "shard_bytes": [3, 3],
                    "artifact_bytes": 6,
                },
            }
            record_path = root / "qwen-quant-build-record.json"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            record_sha = hashlib.sha256(record_path.read_bytes()).hexdigest()
            inventory, copied = benchmark.model_inventory_from_build_record(
                record_path, record_sha, root / shards[0]["path"].split("/")[-1],
                shards[0]["sha256"])
            self.assertEqual(copied, record_path.read_bytes())
            self.assertEqual(inventory["shard_count"], 2)
            self.assertEqual(inventory["aggregate_bytes"], 6)
            self.assertEqual(
                [row["filename"] for row in inventory["shards"]],
                [Path(row["path"]).name for row in shards])
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                benchmark.model_inventory_from_build_record(
                    record_path, "0" * 64,
                    root / shards[0]["path"].split("/")[-1],
                    shards[0]["sha256"])

            record["output"]["shards"].reverse()
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete ordered"):
                benchmark.model_inventory_from_build_record(
                    record_path,
                    hashlib.sha256(record_path.read_bytes()).hexdigest(),
                    root / shards[0]["path"].split("/")[-1],
                    shards[0]["sha256"])

    def test_runner_sampler_records_live_rss_gtt_uma_and_pages_limit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            status = root / "proc" / "123" / "status"
            status.parent.mkdir(parents=True)
            status.write_text("VmRSS:\t1024 kB\nVmHWM:\t2048 kB\n")
            meminfo = root / "meminfo"
            meminfo.write_text(
                "MemTotal:       131150288 kB\n"
                "MemAvailable:    10485760 kB\n")
            pages = root / "pages_limit"
            pages.write_text("32505856\n")
            gtt = root / "gtt"
            gtt.write_text("3145728\n")
            sampler = benchmark.ResourceSampler(
                123, proc_root=root / "proc", meminfo_path=meminfo,
                pages_limit_path=pages, gtt_paths=[gtt])
            sampler.samples.append(sampler._sample())
            resources = sampler.summary()
            self.assertEqual(
                resources["peak_memory_measurement_method"],
                "runner_rss_gtt_sampler_v1")
            self.assertEqual(resources["server_host_pid"], 123)
            self.assertEqual(resources["runner_memtotal_bytes"], 134297894912)
            self.assertEqual(resources["runner_gtt_pages_limit"], 32505856)
            self.assertEqual(resources["runner_gtt_cap_bytes"], 133143986176)
            self.assertEqual(resources["measured_peak_rss_bytes"], 2048 * 1024)
            self.assertEqual(resources["measured_peak_gtt_bytes"], 3145728)
            gate = benchmark.evaluate_memory_gate(
                resources, gtt_cap_bytes=133143986176)
            self.assertTrue(gate["passed"], gate)

    def test_memory_gate_rejects_gtt_or_uma_over_live_limits(self) -> None:
        resources = {
            "peak_memory_measurement_method": "runner_rss_gtt_sampler_v1",
            "samples": 10,
            "server_host_pid": 123,
            "runner_memtotal_bytes": 134297894912,
            "runner_gtt_pages_limit": 32505856,
            "runner_gtt_cap_bytes": 133143986176,
            "measured_peak_rss_bytes": 100_000_000_000,
            "measured_peak_gtt_bytes": 100_000_000_000,
            "measured_peak_uma_bytes": 110_000_000_000,
        }
        self.assertTrue(benchmark.evaluate_memory_gate(
            resources, gtt_cap_bytes=133143986176)["passed"])
        resources["measured_peak_gtt_bytes"] = 133143986177
        resources["measured_peak_uma_bytes"] = 133143986177
        gate = benchmark.evaluate_memory_gate(
            resources, gtt_cap_bytes=133143986176)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["gtt_fits_required_cap"])
        resources["measured_peak_gtt_bytes"] = 100_000_000_000
        resources["measured_peak_uma_bytes"] = 134297894913
        gate = benchmark.evaluate_memory_gate(
            resources, gtt_cap_bytes=133143986176)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["uma_fits_host_memtotal"])


if __name__ == "__main__":
    unittest.main()
