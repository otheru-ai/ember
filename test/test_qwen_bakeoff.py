#!/usr/bin/env python3
"""Offline tests for measured-only Qwen bakeoff selection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qwen_bakeoff", ROOT / "scripts" / "qwen_bakeoff.py")
assert SPEC is not None and SPEC.loader is not None
qb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qb)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BakeoffTest(unittest.TestCase):
    def make_corpora(self, root: Path) -> Path:
        directory = root / "corpora"
        directory.mkdir()
        names = ("extraction-good.jsonl", "extraction-bad.jsonl",
                 "sweep-validation.jsonl", "final-heldout.jsonl")
        artifacts = []
        for index, name in enumerate(names):
            path = directory / name
            path.write_text(json.dumps({"id": f"case-{index}"}) + "\n", encoding="utf-8")
            artifacts.append({"filename": name, "sha256": digest(path), "record_count": 1})
        (directory / "qwen-corpora-manifest.json").write_text(json.dumps({
            "partition": {"pairwise_request_overlap_count": 0}, "artifacts": artifacts,
        }), encoding="utf-8")
        return directory

    @staticmethod
    def measured(identifier: str, stage: str, corpus_sha: str, **extra: object) -> dict:
        return {
            "id": identifier, "stage": stage, "measurement_kind": "measured",
            "status": "complete", "corpus_sha256": corpus_sha,
            "differential_correctness_pass": True, "audited_quality_pass": True,
            "artifact_bytes": 90_000_000_000, "quality_score": 0.9,
            "prefill_tps_samples": [412.0, 413.0, 414.0],
            "decode_tps_samples": [39.49, 40.0, 40.5], **extra,
            "companion_artifact_bytes": {"mtp": 2_000_000_000, "vision_mmproj": 901_943_132},
            "enabled_companions": ["mtp", "vision_mmproj"],
            "runner_memtotal_bytes": 134_297_894_912,
            "runner_gtt_pages_limit": 32_505_856,
            "peak_memory_measurement_method": "runner_rss_gtt_sampler_v1",
            "measured_peak_rss_bytes": 120_000_000_000,
            "measured_peak_gtt_bytes": 100_000_000_000,
            "measured_peak_uma_bytes": 125_000_000_000,
        }

    def complete_results(self, plan: dict) -> dict:
        sweep_sha = plan["corpora"]["sweep-validation.jsonl"]["sha256"]
        final_sha = plan["corpora"]["final-heldout.jsonl"]["sha256"]
        rows = [self.measured(
            "stock-rocmi4-exact", "stock", sweep_sha, final_release_eligible=False,
        )]
        for index, configuration in enumerate(plan["sweep_configurations"]):
            rows.append(self.measured(
                configuration["id"], "sweep", sweep_sha,
                quality_score=1.0 if index == 0 else 0.8,
            ))
        winner = plan["sweep_configurations"][0]["id"]
        for arm in plan["format_arms"]:
            rows.append(self.measured(
                f"{winner}:{arm['id']}", "format", sweep_sha,
                configuration_id=winner, arm_id=arm["id"],
                final_release_eligible=arm["final_release_eligible"],
                quality_score=1.0 if arm["id"] == "rocmi4-exact" else 0.8,
            ))
        rows.append(self.measured(
            f"{winner}:rocmi4-exact", "final", final_sha,
            configuration_id=winner, arm_id="rocmi4-exact",
        ))
        return {"results": rows}

    def test_plan_is_complete_and_final_partition_cannot_select(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpora = self.make_corpora(Path(temporary))
            plan = qb.make_plan(qb.DEFAULT_RECIPE, corpora)
            self.assertEqual(len(plan["sweep_configurations"]), 16)
            self.assertEqual({arm["id"] for arm in plan["format_arms"]}, {
                "rocmi4-exact", "rocmfp4-fast-exact", "rocmi4-w4a4",
            })
            self.assertFalse(plan["publication_allowed"])
            for configuration in plan["sweep_configurations"]:
                self.assertEqual(len(configuration["layer_scales"]), 48)
            decision = qb.decide(plan, self.complete_results(plan))
            self.assertEqual(decision["selected_configuration_id"],
                             plan["sweep_configurations"][0]["id"])
            self.assertEqual(decision["selected_arm_id"], "rocmi4-exact")
            self.assertFalse(decision["final_heldout_used_for_selection"])
            self.assertFalse(decision["publication_allowed"])

    def test_missing_estimated_or_final_reselection_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = qb.make_plan(qb.DEFAULT_RECIPE, self.make_corpora(Path(temporary)))
            results = self.complete_results(plan)
            results["results"][1]["measurement_kind"] = "estimated"
            with self.assertRaisesRegex(qb.BakeoffError, "complete measured"):
                qb.decide(plan, results)
            results = self.complete_results(plan)
            results["results"][-1]["arm_id"] = "rocmfp4-fast-exact"
            with self.assertRaisesRegex(qb.BakeoffError, "attempted to change"):
                qb.decide(plan, results)

    def test_hard_perf_and_128g_fit_gates_are_inclusive(self) -> None:
        recipe = json.loads(qb.DEFAULT_RECIPE.read_text(encoding="utf-8"))
        row = self.measured("x", "sweep", "a" * 64)
        passed = qb.assess(row, recipe["hard_gates"], "a" * 64)
        self.assertTrue(passed["passes"])
        row["artifact_bytes"] = (
            recipe["hard_gates"]["device_budget_bytes"]
            - recipe["hard_gates"]["runtime_reserve_bytes"] + 1
        )
        self.assertFalse(qb.assess(row, recipe["hard_gates"], "a" * 64)["passes"])
        row = self.measured("x", "sweep", "a" * 64)
        row["measured_peak_gtt_bytes"] = recipe["hard_gates"]["certification_host_gtt_cap_bytes"] + 1
        row["measured_peak_uma_bytes"] = row["measured_peak_gtt_bytes"]
        self.assertFalse(qb.assess(row, recipe["hard_gates"], "a" * 64)["passes"])


if __name__ == "__main__":
    unittest.main()
