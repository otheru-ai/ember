#!/usr/bin/env python3
"""Adversarial GPU-free tests for Qwen quality descriptor generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_quality_descriptor as quality  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.revision = "1" * 40
        self.image = "ghcr.io/otheru-ai/ember@sha256:" + "2" * 64
        self.judge_image = "ghcr.io/otheru-ai/judge@sha256:" + "3" * 64
        self.corpus = write(root / "corpora" / "sweep-validation.jsonl", b"fixture\n")
        self.profile = write(root / "profile.json", {
            "intervention": {"manifest_filename": "qwen-intervention-manifest.json"},
        })
        self.selection = {
            "phase_scope": "selection", "status": "planned_unmeasured",
            "release_profile": {"path": str(self.profile), "sha256": digest(self.profile)},
            "stock_control": {"id": "stock-rocmi4-exact"},
            "corpora": {"sweep-validation.jsonl": {
                "path": str(self.corpus), "sha256": digest(self.corpus)}},
            "sweep_configurations": [{
                "id": "lambda-0.25-all-48", "quantization_arm": "rocmi4-control",
                "quantization_overrides_sha256": "4" * 64,
            }],
            "format_arms": [],
        }
        self.plan = write(root / "selection-plan.json", self.selection)
        self.stock_record = self.build("stock", stock=True)
        self.candidate_record = self.build("candidate", stock=False)
        judge_artifact = write(root / "judge" / "judge.gguf", b"judge")
        self.judge_inventory = write(root / "judge-inventory.json", {
            "schema": quality.JUDGE_SCHEMA,
            "artifact": {"path": str(judge_artifact), "sha256": digest(judge_artifact),
                         "bytes": judge_artifact.stat().st_size},
        })

    def build(self, name: str, *, stock: bool) -> Path:
        directory = self.root / name
        shard = write(directory / "model.gguf", name.encode())
        if not stock:
            write(directory / "qwen-intervention-manifest.json", b"intervention")
        record = {
            "status": "complete", "mode": "execute", "publishes": False,
            "credentials_accessed": False, "compute_mode": "exact_dequant",
            "w4a4_enabled": False,
            "profile": {"path": str(self.profile), "sha256": digest(self.profile)},
            "experiment": {
                "kind": "stock_control" if stock else "directional_ablation",
                "stock_weights_unchanged": stock, "final_release_eligible": False,
                "eligibility_status": ("ineligible_stock_control" if stock else
                                       "pending_measured_bakeoff_and_hardware_certification"),
                "purpose": ("activation_capture_and_bakeoff_baseline" if stock else
                            "measured_bakeoff_candidate"),
            },
            "intervention": None if stock else {
                "manifest_sha256": digest(directory / "qwen-intervention-manifest.json")},
            "quantization_recipe": {
                "id": "profile-default-rocmi4" if stock else "rocmi4-control",
                "per_tensor_overrides_sha256": "5" * 64 if stock else "4" * 64,
                "selected_mtp_matrix_quant_contract": "Q4_0_ROCMI4",
            },
            "sweep_authorization": None if stock else {
                "configuration_id": "lambda-0.25-all-48"},
            "output": {"shards": [{"path": str(shard), "sha256": digest(shard),
                                    "size_bytes": shard.stat().st_size}]},
        }
        return write(directory / "qwen-quant-build-record.json", record)

    def args(self, **changes: object) -> argparse.Namespace:
        values = {
            "phase": "sweep", "phase_plan": self.plan,
            "phase_plan_sha256": digest(self.plan),
            "stock_build_record": self.stock_record,
            "stock_build_record_sha256": digest(self.stock_record),
            "candidate_build_record": self.candidate_record,
            "candidate_build_record_sha256": digest(self.candidate_record),
            "candidate_id": "candidate-fixture",
            "judge_inventory": self.judge_inventory,
            "judge_inventory_sha256": digest(self.judge_inventory),
            "ember_revision": self.revision,
            "model_runtime_image": self.image, "judge_runtime_image": self.judge_image,
            "quality_output_root": self.root / "quality-capture",
            "capture_plan_output": self.root / "capture-plan.json",
            "output": self.root / "phase-descriptor.json",
        }
        values.update(changes)
        return argparse.Namespace(**values)


class QualityDescriptorTests(unittest.TestCase):
    def generate(self, fixture: Fixture, **changes: object) -> tuple[dict, dict]:
        with (mock.patch.object(quality.bakeoff, "verify_plan",
                               return_value=fixture.selection),
              mock.patch.object(quality, "SWEEP_SHA256", digest(fixture.corpus))):
            return quality.generate(fixture.args(**changes))

    def test_generates_exact_workflow_schemas_without_opening_capture_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            capture, phase = self.generate(fixture)
            self.assertEqual(capture["schema"], quality.PLAN_SCHEMA)
            self.assertEqual(phase["schema"], quality.DESCRIPTOR_SCHEMA)
            self.assertEqual(phase["phase"], "sweep")
            self.assertFalse(fixture.args().quality_output_root.exists())
            self.assertEqual(capture["models"]["stock"]["intervention_manifest_sha256"],
                             "0" * 64)
            self.assertEqual(capture["models"]["candidate"]["candidate_id"],
                             "candidate-fixture")
            self.assertEqual(capture["runs"]["stock"]["image"],
                             capture["runs"]["candidate"]["image"])
            self.assertEqual(set(phase["launches"]), {"stock", "candidate", "judge"})
            self.assertEqual(phase["capture_plan"]["sha256"],
                             digest(fixture.args().capture_plan_output))

    def test_tampered_shard_fails_before_outputs_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            (fixture.root / "candidate" / "model.gguf").write_bytes(b"changed")
            with self.assertRaisesRegex(quality.DescriptorError, "byte count differs"):
                self.generate(fixture)
            self.assertFalse(fixture.args().capture_plan_output.exists())
            self.assertFalse(fixture.args().output.exists())

    def test_judge_inventory_rejects_symlink_and_unexpected_keys(self) -> None:
        for attack in ("symlink", "extra-key"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as raw:
                fixture = Fixture(Path(raw))
                inventory = fixture.judge_inventory
                if attack == "symlink":
                    real = inventory.with_name("real-inventory.json")
                    inventory.rename(real)
                    inventory.symlink_to(real)
                    expected = digest(real)
                else:
                    value = json.loads(inventory.read_text())
                    value["runtime"] = "untrusted"
                    write(inventory, value)
                    expected = digest(inventory)
                with self.assertRaises(quality.DescriptorError):
                    self.generate(fixture, judge_inventory_sha256=expected)

    def test_json_mutation_during_the_verified_read_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            target = write(Path(raw) / "input.json", {"before": True})
            expected = digest(target)
            original_read = os.read
            mutated = False

            def adversarial_read(file_descriptor: int, size: int) -> bytes:
                nonlocal mutated
                data = original_read(file_descriptor, size)
                if data and not mutated:
                    mutated = True
                    target.write_bytes(b'{"after":"identity changed during read"}\n')
                return data

            with (mock.patch.object(quality.os, "read", adversarial_read),
                  self.assertRaisesRegex(quality.DescriptorError, "changed while verified")):
                quality.exact_json(target, expected, "mutating JSON")

    def test_bad_hash_unsafe_image_and_existing_capture_root_fail_closed(self) -> None:
        attacks = {
            "hash": {"candidate_build_record_sha256": "0" * 64},
            "image": {"model_runtime_image": "ghcr.io/otheru-ai/ember:latest"},
            "root": {},
        }
        for attack, changes in attacks.items():
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as raw:
                fixture = Fixture(Path(raw))
                if attack == "root":
                    fixture.args().quality_output_root.mkdir()
                with self.assertRaises(quality.DescriptorError):
                    self.generate(fixture, **changes)

    def test_final_requires_exact_sealed_candidate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            candidate_record = json.loads(fixture.candidate_record.read_text())
            shards, inventory_sha, artifact_bytes = quality.ordered_shards(
                candidate_record, "candidate")
            identity = {
                "candidate_id": "candidate-fixture",
                "build_record_sha256": digest(fixture.candidate_record),
                "intervention_manifest_sha256": candidate_record["intervention"][
                    "manifest_sha256"],
                "profile_sha256": digest(fixture.profile),
                "quantization_overrides_sha256": "4" * 64,
                "model_inventory_sha256": inventory_sha,
                "artifact_bytes": artifact_bytes,
            }
            final_corpus = write(fixture.root / "corpora" / "final-heldout.jsonl", b"final\n")
            final = {
                "phase_scope": "final_confirmation",
                "status": "final_heldout_unlocked_after_mtp_depth_selection",
                "selection_plan": fixture.selection,
                "corpora": {"final-heldout.jsonl": {
                    "path": str(final_corpus), "sha256": digest(final_corpus)}},
                "sealed_recipe_ledger": {"descriptor": {"subject": "fixture"}},
            }
            final_path = write(fixture.root / "final-plan.json", final)
            args = fixture.args(phase="final", phase_plan=final_path,
                                phase_plan_sha256=digest(final_path))
            with (mock.patch.object(quality.bakeoff, "verify_plan", return_value=final),
                  mock.patch.object(quality.bakeoff, "read_prior_ledger",
                                    return_value=({"selected_artifact_identity": identity}, "a" * 64)),
                  mock.patch.object(quality, "FINAL_SHA256", digest(final_corpus))):
                quality.generate(args)
            self.assertTrue(args.output.is_file())

            identity["build_record_sha256"] = "f" * 64
            args = fixture.args(phase="final", phase_plan=final_path,
                                phase_plan_sha256=digest(final_path),
                                capture_plan_output=fixture.root / "capture-plan-bad.json",
                                output=fixture.root / "phase-descriptor-bad.json",
                                quality_output_root=fixture.root / "quality-capture-bad")
            with (mock.patch.object(quality.bakeoff, "verify_plan", return_value=final),
                  mock.patch.object(quality.bakeoff, "read_prior_ledger",
                                    return_value=({"selected_artifact_identity": identity}, "a" * 64)),
                  mock.patch.object(quality, "FINAL_SHA256", digest(final_corpus)),
                  self.assertRaisesRegex(quality.DescriptorError, "sealed winner")):
                quality.generate(args)


if __name__ == "__main__":
    unittest.main()
