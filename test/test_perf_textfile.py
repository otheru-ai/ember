"""perf_textfile.py exports docs/perf/data.json as Prometheus textfile gauges;
this pins the exposition contract the Grafana benchmarks dashboard reads."""
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "perf_textfile", ROOT / "scripts" / "bench" / "perf_textfile.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

SAMPLE = {"schema": 1, "releases": [
    {"id": "2026.9.8", "image": "ghcr.io/otheru-ai/ember:2026.9.8", "measured": "2026-09-09",
     "provenance": {"certified": False, "bundle": "b"}, "model": {"target": "t.gguf", "drafter": "d.gguf"},
     "throughput": {"median_tps": 39.44, "min_tps": 39.4, "max_tps": 40.41, "median_accept_rate": 0.981,
                    "min_accept_rate": 0.981, "samples": 3, "spec_ran": 3},
     "workloads": {"code": {"tok_s": 29.36, "autoregressive_tok_s": 23.98, "prefill_tok_s": 116.0,
                            "autoregressive_prefill_tok_s": 116.3, "speedup": 1.224}},
     "prefill_groups": {"prefill-2048": {"evaluated_prompt_tokens": 2074, "median_tps": 416.6,
                                         "min_tps": 384.7, "max_tps": 418.0, "samples": 3}},
     "depths": [{"depth": 0, "decode_tok_s": 38.03, "prefill_tok_s": 69.7, "autoregressive_tok_s": 23.47,
                 "total_tok_s": 30.0, "accept_rate": 1.0, "speedup": 1.62, "prompt_tokens": 43}],
     "vision": {"cold_prefill_ms": 1571.2, "warm_prefill_ms": 1563.8, "warm_decode_tps": 23.9,
                "cold_wall_seconds": 5.311, "warm_wall_seconds": 4.51, "prompt_tokens": 361,
                "image_bytes": 117091, "samples": 4, "spec_ran": [False]}},
    {"id": "2026.9.11", "measured": "2026-09-11", "throughput": {"median_tps": 40.0}},
]}


class PerfTextfileTests(unittest.TestCase):
    def setUp(self):
        self.text = MOD.export(SAMPLE)
        self.samples = [l for l in self.text.splitlines() if l and not l.startswith("#")]

    def test_release_ordinal_sorts_calver(self):
        # 2026.9.11 sorts before 2026.9.8 lexically; seq keeps data.json order.
        self.assertIn('ember_bench_release_info{release="2026.9.8",seq="000",', self.text)
        self.assertIn('ember_bench_measured_timestamp_seconds{release="2026.9.11",seq="001"}', self.text)

    def test_every_family_present_and_unit_named(self):
        for name in ("ember_bench_throughput_tokens_per_second", "ember_bench_accept_rate",
                     "ember_bench_workload_tokens_per_second", "ember_bench_workload_speedup",
                     "ember_bench_prefill_group_tokens_per_second", "ember_bench_depth_tokens_per_second",
                     "ember_bench_depth_accept_rate", "ember_bench_vision_cold_prefill_seconds"):
            self.assertIn(f"# TYPE {name} gauge", self.text, name)
        self.assertNotIn("_tok_s{", self.text)   # promtool: no abbreviated units
        self.assertNotIn("_ms{", self.text)

    def test_values_copied_not_computed(self):
        self.assertIn('ember_bench_workload_tokens_per_second{release="2026.9.8",seq="000",workload="code",mode="spec"} 29.36', self.text)
        self.assertIn('ember_bench_depth_tokens_per_second{release="2026.9.8",seq="000",depth="0",kind="decode"} 38.03', self.text)
        self.assertIn('ember_bench_vision_cold_prefill_seconds{release="2026.9.8",seq="000"} 1.5712', self.text)
        self.assertIn('ember_bench_prefill_group_tokens_per_second{release="2026.9.8",seq="000",group="prefill-2048",stat="median",prompt_tokens="2074"} 416.6', self.text)

    def test_partial_release_emits_only_what_it_has(self):
        mine = [s for s in self.samples if 'release="2026.9.11"' in s]
        self.assertTrue(any(s.startswith("ember_bench_throughput_tokens_per_second") for s in mine))
        self.assertFalse(any(s.startswith("ember_bench_workload") for s in mine))

    def test_help_and_type_once_per_family(self):
        self.assertEqual(self.text.count("# TYPE ember_bench_workload_tokens_per_second gauge"), 1)

    def test_main_writes_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = pathlib.Path(tmp) / "data.json"; data.write_text(json.dumps(SAMPLE))
            out = pathlib.Path(tmp) / "bench.prom"
            self.assertEqual(MOD.main(["--data", str(data), "--out", str(out)]), 0)
            self.assertTrue(out.exists()); self.assertFalse(out.with_suffix(".prom.tmp").exists())
            self.assertEqual(out.read_text(), self.text)

    def test_real_data_json_exports(self):
        real = json.loads((ROOT / "docs" / "perf" / "data.json").read_text())
        text = MOD.export(real)
        self.assertEqual(text.count("ember_bench_release_info{"), len(real["releases"]))


if __name__ == "__main__":
    unittest.main()
