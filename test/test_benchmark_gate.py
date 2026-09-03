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

    def test_workload_summary_preserves_prefill_rates(self) -> None:
        on = [{"label": "image_short", "decode_tps": 20.0,
               "prefill_tps": 31.25}]
        off = [{"label": "image_short", "decode_tps": 18.0,
                "prefill_tps": 30.75}]
        row = assemble.summarise_workloads(on, off)["image_short"]
        self.assertEqual(row["prefill_tok_s"], 31.2)
        self.assertEqual(row["autoregressive_prefill_tok_s"], 30.8)

    def test_workload_baseline_rejects_partial_or_error_rows(self) -> None:
        on = [{"label": f"w{i}", "decode_tps": 20.0,
               "accept_rate": 0.5, "spec_ran": True, "spec_cycles": 32}
              for i in range(12)]
        off = [{"label": f"w{i}", "decode_tps": 18.0,
                "accept_rate": 0.0, "spec_ran": False, "spec_cycles": 0}
               for i in range(12)]
        assemble.validate_workload_rows(on, off, 12)
        with self.assertRaisesRegex(SystemExit, "requires 12 unique rows"):
            assemble.validate_workload_rows(on[:-1], off, 12)
        broken = [dict(row) for row in on]
        broken[3] = {"label": "w3", "error": "HTTP Error 422"}
        with self.assertRaisesRegex(SystemExit, "unusable rows"):
            assemble.validate_workload_rows(broken, off, 12)
        with self.assertRaisesRegex(SystemExit, "requires 12 unique rows"):
            assemble.validate_workload_rows([], [], 12)

    def test_workload_baseline_requires_complete_speculative_evidence(self) -> None:
        on = [{"label": f"w{i}", "decode_tps": 20.0,
               "accept_rate": 0.5, "spec_ran": True, "spec_cycles": 32}
              for i in range(12)]
        off = [{"label": f"w{i}", "decode_tps": 18.0,
                "accept_rate": 0.0, "spec_ran": False, "spec_cycles": 0}
               for i in range(12)]
        for field, value in (("accept_rate", None), ("accept_rate", True),
                             ("accept_rate", float("nan")), ("spec_ran", None),
                             ("spec_ran", 0), ("spec_cycles", None),
                             ("spec_cycles", False),
                             ("spec_cycles", float("inf"))):
            broken = [dict(row) for row in on]
            broken[3][field] = value
            with self.subTest(field=field, value=value), \
                    self.assertRaisesRegex(SystemExit, "invalid speculative evidence"):
                assemble.validate_workload_rows(broken, off, 12)

    def test_model_provenance_binds_mmproj_when_present(self) -> None:
        model = assemble.model_provenance({
            "TARGET": "/srv/models/vision.gguf", "TARGET_SHA": "target-sha",
            "TARGET_SHA_SOURCE": "computed",
            "DRAFT": "/srv/models/draft.gguf", "DRAFT_SHA": "draft-sha",
            "DRAFT_SHA_SOURCE": "asserted",
            "DRAFT_SHA_ASSERTED_BY": "run-123",
            "MMPROJ": "/srv/models/mmproj.gguf", "MMPROJ_SHA": "mmproj-sha",
            "MMPROJ_SHA_SOURCE": "asserted",
            "MMPROJ_SHA_ASSERTED_BY": "run-456",
        })
        self.assertEqual(model["target"], "vision.gguf")
        self.assertEqual(model["target_sha256"], "target-sha")
        self.assertEqual(model["mmproj"], "mmproj.gguf")
        self.assertEqual(model["mmproj_sha256"], "mmproj-sha")
        self.assertEqual(model["target_sha256_source"], "computed")
        self.assertEqual(model["drafter_sha256_source"], "asserted")
        self.assertEqual(model["drafter_sha256_asserted_by"], "run-123")
        with self.assertRaisesRegex(SystemExit, "SHA-256 is missing"):
            assemble.model_provenance({
                "TARGET_SHA": "target-sha", "TARGET_SHA_SOURCE": "computed",
                "DRAFT_SHA": "draft-sha", "DRAFT_SHA_SOURCE": "computed",
                "MMPROJ": "/srv/models/mmproj.gguf"})

    def test_model_provenance_rejects_unbound_asserted_digest(self) -> None:
        with self.assertRaisesRegex(SystemExit, "missing its evidence reference"):
            assemble.model_provenance({
                "TARGET_SHA": "target-sha", "TARGET_SHA_SOURCE": "asserted",
                "DRAFT_SHA": "draft-sha", "DRAFT_SHA_SOURCE": "computed"})

    def test_shape_calibration_uses_prompt_tokens(self) -> None:
        class FakeSuite:
            def __init__(self) -> None:
                self.words: list[int] = []

            def request(self, _label: str, prompt: str, _max_tokens: int,
                        *, group: str, repeat: int) -> dict:
                del group, repeat
                words = prompt.count(" alpha")
                self.words.append(words)
                # Simulate a template with 31 tokens of overhead rather
                # than the DeepSeek protocol's 26.
                return {"ok": True, "prompt_tokens": words + 31,
                        "evaluated_prefill_tokens": 1}

        fake = FakeSuite()
        words, attempts = benchmark.calibrate_prefill_words(fake, 2074)
        self.assertEqual(words, 2043)
        self.assertEqual(fake.words, [2048, 2043])
        self.assertEqual(attempts[-1]["prompt_tokens"], 2074)

    def test_decode_calibration_records_and_selects_exact_shape(self) -> None:
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

    def test_calibrated_gate_protocol_is_bound_in_machine_evidence(self) -> None:
        gate = benchmark.evaluate_hard_gate(
            self.records(), prefill_target=412.0, decode_target=39.49,
            protocol=benchmark.CALIBRATED_HARD_GATE_PROTOCOL)
        self.assertEqual(gate["protocol"], benchmark.CALIBRATED_HARD_GATE_PROTOCOL)

    def test_differential_decode_rates_are_independently_derived(self) -> None:
        evidence = {
            "schema": benchmark.DIFFERENTIAL_DECODE_SCHEMA,
            "purpose": benchmark.DIFFERENTIAL_DECODE_PURPOSE,
            "tokens_per_path": benchmark.DIFFERENTIAL_DECODE_TOKENS,
            "ar": {"decode_seconds": 2.0, "tokens_per_second": 32.0},
            "mtp": {
                "accept_rate": 0.5,
                "restored_decode_seconds": 1.7,
                "warm_fresh_decode_seconds": 1.6,
                "warm_fresh_tokens_per_second": 40.0,
                "warm_speedup_vs_ar": 1.25,
            },
        }
        self.assertIs(benchmark.validate_differential_decode(evidence), evidence)
        evidence["mtp"]["warm_speedup_vs_ar"] = 1.0
        with self.assertRaisesRegex(ValueError, "derivation differs"):
            benchmark.validate_differential_decode(evidence)
        evidence["mtp"]["warm_speedup_vs_ar"] = 1.25
        evidence["mtp"]["accept_rate"] = 1.0
        with self.assertRaisesRegex(ValueError, "acceptance is out of range"):
            benchmark.validate_differential_decode(evidence)

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
                name = f"model-rocmi4-{index:05d}-of-00002.gguf"
                path = root / name
                path.write_bytes(data)
                shards.append({
                    "path": f"/model-work/final/{name}",
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
            record_path = root / "model-quant-build-record.json"
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
