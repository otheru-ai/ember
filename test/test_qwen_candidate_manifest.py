#!/usr/bin/env python3
"""Adversarial GPU-free tests for Qwen bakeoff manifest normalization."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_candidate_manifest as manifest  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def desc(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest(path)}


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.hex40 = "1" * 40
        self.format_sha = "2" * 64
        self.runtime_digest = "sha256:" + "3" * 64
        self.dev_digest = "sha256:" + "4" * 64
        self.builder_digest = "sha256:" + "5" * 64
        self.release_ref = "ghcr.io/otheru-ai/ember@" + self.runtime_digest
        self.dev_ref = "ghcr.io/otheru-ai/ember@" + self.dev_digest
        self.builder_ref = "ghcr.io/otheru-ai/ember@" + self.builder_digest
        self.source = {"repo_id": "Qwen/Qwen3.8-Flash-Next",
                       "revision": "source", "snapshot_inventory_sha256": "0" * 64}
        corpus = write(root / "corpora" / "qwen-selection-corpora-manifest.json", {})
        profile = write(root / "profile.json", {})
        self.plan = {
            "phase_scope": "selection",
            "release_profile": desc(profile),
            "corpus_manifest": desc(corpus),
            "direction_basis": {"source": self.source},
            "corpora": {"sweep-validation.jsonl": {"sha256": "6" * 64}},
            "stock_control": {"id": "stock-rocmi4-exact", "runtime_mode": "exact_dequant"},
            "sweep_configurations": [{
                "id": "lambda-0.25-all-48", "quantization_arm": "rocmi4-control",
                "quantization_overrides_sha256": "7" * 64,
                "runtime_mode": "exact_dequant", "final_release_eligible": False,
            }],
            "format_arms": [{
                "id": "rocmi4-q6k-main-rocmi4-mtp-d3",
                "quantization_arm": "rocmi4-q6k-embedding-head",
                "quantization_overrides_sha256": "8" * 64,
                "mtp_matrix_quant_contract": "Q4_0_ROCMI4", "mtp_depth": 3,
                "runtime_mode": "exact_dequant", "final_release_eligible": True,
            }],
            "mtp_depth_configurations": [
                {"id": f"mtp-depth-{depth}", "mtp_depth": depth,
                 "runtime_mode": "exact_dequant", "final_release_eligible": True}
                for depth in range(1, 5)
            ],
        }
        self.plan_path = write(root / "selection-plan.json", self.plan)
        cache_dir = root / "workset" / "bf16-cache" / "bf16-fixture"
        self.cache = write(cache_dir / "bf16-cache-manifest.json", {"cache_id": "cache"})
        self.capture = write(root / "capture.json", {"status": "complete"})
        self.intervention = write(root / "intervention.json", {"status": "complete"})
        self.mtp = root / "companions" / "mtp.gguf"; self.mtp.parent.mkdir(parents=True)
        self.mtp.write_bytes(b"mtp")
        self.mmproj = root / "companions" / "mmproj.gguf"; self.mmproj.write_bytes(b"mmproj")
        self.mmproj_inventory_sha = manifest.vision_inventory.load_contract()[
            "tensor_inventory_sha256"]
        self.export = write(root / "companions" / "export.json", {"status": "complete"})
        self.mtp_fast = root / "companions" / "mtp-fast.gguf"
        self.mtp_fast.write_bytes(b"mtp-fast")
        self.export_fast = write(
            root / "companions" / "export-fast.json", {"status": "complete"})
        self.companion = write(root / "companions" / "rocmi4.json", {
            "schema": "ember.qwen3.8-flash-next.companion-inventory.v1",
            "source": self.source,
            "companions": [
                {"role": "mtp", "enabled": True, "path": str(self.mtp),
                 "size_bytes": self.mtp.stat().st_size, "sha256": digest(self.mtp),
                 "matrix_quant_contract": "Q4_0_ROCMI4",
                 "export_manifest_path": str(self.export),
                 "export_manifest_sha256": digest(self.export)},
                {"role": "vision_mmproj", "enabled": True, "path": str(self.mmproj),
                 "size_bytes": self.mmproj.stat().st_size, "sha256": digest(self.mmproj),
                 "format": "BF16",
                 "tensor_inventory_sha256": self.mmproj_inventory_sha},
            ],
        })
        self.fast = write(root / "companions" / "fast.json", {
            "schema": "ember.qwen3.8-flash-next.companion-inventory.v1",
            "source": self.source,
            "companions": [
                {"role": "mtp", "enabled": True, "path": str(self.mtp_fast),
                 "size_bytes": self.mtp_fast.stat().st_size,
                 "sha256": digest(self.mtp_fast),
                 "matrix_quant_contract": "Q4_0_ROCMFP4_FAST",
                 "export_manifest_path": str(self.export_fast),
                 "export_manifest_sha256": digest(self.export_fast)},
                {"role": "vision_mmproj", "enabled": True, "path": str(self.mmproj),
                 "size_bytes": self.mmproj.stat().st_size, "sha256": digest(self.mmproj),
                 "format": "BF16",
                 "tensor_inventory_sha256": self.mmproj_inventory_sha},
            ],
        })
        self.quality = write(root / "quality.json", {"audited": True})

    def set_companion_source(self, source: dict) -> None:
        for path in (self.companion, self.fast):
            value = json.loads(path.read_text())
            value["source"] = source
            write(path, value)

    def construction(self, stage: str, row_id: str, *, stock: bool = False) -> tuple[Path, dict]:
        candidate = self.root / f"candidate-{stage}-{row_id}"
        candidate.mkdir()
        shard = candidate / "model.gguf"; shard.write_bytes((stage + row_id).encode())
        arm = "rocmi4-control" if stock or stage == "sweep" else "rocmi4-q6k-embedding-head"
        override = "7" * 64 if stage == "sweep" else "8" * 64
        source_sha = digest(self.capture if stock else self.intervention)
        roles = [
            {"role": "mtp", "path": str(self.mtp), "size_bytes": self.mtp.stat().st_size,
             "sha256": digest(self.mtp), "matrix_quant_contract": "Q4_0_ROCMI4",
             "export_manifest": {"path": str(self.export), "sha256": digest(self.export)}},
            {"role": "vision_mmproj", "path": str(self.mmproj),
             "size_bytes": self.mmproj.stat().st_size, "sha256": digest(self.mmproj),
             "format": "BF16", "gguf_contract": {
                 "tensor_inventory_sha256": self.mmproj_inventory_sha}},
        ]
        record = write(candidate / "qwen-quant-build-record.json", {
            "status": "complete", "mode": "execute", "tools": {"ember_revision": self.hex40},
            "experiment": {
                "kind": "stock_control" if stock else "directional_ablation",
                "stock_weights_unchanged": stock, "final_release_eligible": False,
                "eligibility_status": ("ineligible_stock_control" if stock else
                    "pending_measured_bakeoff_and_hardware_certification"),
                "purpose": ("activation_capture_and_bakeoff_baseline" if stock else
                    "measured_bakeoff_candidate"),
            },
            "profile": self.plan["release_profile"],
            "bf16_cache": {"manifest": desc(self.cache)},
            "companion_inventory": {"manifest": desc(self.companion),
                "status": "verified_exact", "fit_status": "verified_exact_fit",
                "enabled_roles": ["mtp", "vision_mmproj"], "roles": roles},
            "intervention": None if stock else {"manifest_sha256": source_sha},
            "quantization_recipe": {"id": arm,
                "selected_mtp_matrix_quant_contract": "Q4_0_ROCMI4",
                "per_tensor_overrides_sha256": override},
            "output": {"shards": [{"path": str(shard), "size_bytes": shard.stat().st_size,
                                     "sha256": digest(shard)}]},
        })
        builder_identity = {"ember_revision": self.hex40, "quantizer_tool_sha256": "9" * 64,
                            "container_digest": self.builder_digest,
                            "tensor_format_contract_sha256": self.format_sha}
        attestation = write(candidate / "qwen-candidate-workset-attestation.json", {
            "schema": "ember.qwen3.8.candidate-workset-attestation.v1",
            "candidate_id": "artifact-1", "build_record_sha256": digest(record),
            "bf16_cache_manifest_sha256": digest(self.cache),
            "intervention_manifest_sha256": source_sha,
            "stock_capture": ({**desc(self.capture), "byte_identical": True} if stock else None),
            "builder_identity": builder_identity,
            "tensor_format_compatibility_sha256": self.format_sha,
            "artifact_identity": {"quantized_shards": json.loads(record.read_text())["output"]["shards"]},
        })
        configuration = None if stock else (row_id if stage == "sweep" else "lambda-0.25-all-48")
        value = {
            "schema": manifest.CONSTRUCTION_SCHEMA, "status": "complete",
            "publishes": False, "deletes": False, "candidate_id": "artifact-1",
            "kind": "stock" if stock else "intervention", "intended_stage": stage,
            "row_id": row_id, "intervention_configuration_id": configuration,
            "quantization_arm": arm, "mtp_matrix_quant_contract": "Q4_0_ROCMI4",
            "runtime_mode": "exact_dequant", "builder_revision": self.hex40,
            "runtime_revision": self.hex40,
            "images": {"builder": {"ref": self.builder_ref, "digest": self.builder_digest},
                       "runtime": {
                           "release_ref": self.release_ref,
                           "release_digest": self.runtime_digest,
                           "dev_ref": self.dev_ref, "dev_digest": self.dev_digest,
                           "tensor_format_contract_sha256": self.format_sha}},
            "capture": desc(self.capture), "stock_capture": desc(self.capture) if stock else None,
            "bf16_cache": desc(self.cache),
            "shared_companions": {"Q4_0_ROCMI4": desc(self.companion),
                                  "Q4_0_ROCMFP4_FAST": desc(self.fast)},
            "selection_plan": desc(self.plan_path), "build_record": desc(record),
            "builder_attestation": desc(attestation),
            "intervention_manifest": None if stock else desc(self.intervention),
            "artifacts": {"shards": json.loads(record.read_text())["output"]["shards"],
                          "total_bytes": shard.stat().st_size},
            "v3_candidate_manifest": {"ready": False, "blocked_on": ["normalizer"]},
        }
        path = write(self.root / f"construction-{stage}-{row_id}.json", value)
        return path, value

    def args(self, construction: Path, stage: str, row_id: str, depth: int,
             **extra: object) -> argparse.Namespace:
        values = dict(
            construction=construction, construction_sha256=digest(construction),
            stage=stage, row_id=row_id, mtp_matrix_quant_contract="Q4_0_ROCMI4",
            mtp_depth=depth, runtime_mode="exact_dequant", runtime_revision=self.hex40,
            runtime_release_ref=self.release_ref, runtime_release_digest=self.runtime_digest,
            runtime_dev_ref=self.dev_ref, runtime_dev_digest=self.dev_digest,
            runtime_tensor_format_contract_sha256=self.format_sha,
            quality_contract=None if stage == "stock" else self.quality,
            quality_contract_sha256=None if stage == "stock" else digest(self.quality),
            prior_accumulator=None, prior_accumulator_sha256=None,
            prior_ledger=None, prior_ledger_sha256=None,
            final_plan=None, final_plan_sha256=None,
            output=self.root / f"manifest-{stage}-{row_id}.json",
        )
        values.update(extra)
        return argparse.Namespace(**values)

    def artifact_identity(self, construction: dict, depth: int) -> dict:
        record = json.loads(Path(construction["build_record"]["path"]).read_text())
        attestation = json.loads(Path(construction["builder_attestation"]["path"]).read_text())
        shard = record["output"]["shards"][0]
        identities = [{"index": 1, "sha256": shard["sha256"], "bytes": shard["size_bytes"]}]
        return {
            "candidate_id": construction["candidate_id"],
            "build_record_sha256": construction["build_record"]["sha256"],
            "intervention_manifest_sha256": construction["intervention_manifest"]["sha256"],
            "profile_sha256": self.plan["release_profile"]["sha256"],
            "quantization_overrides_sha256": "8" * 64,
            "model_inventory_sha256": manifest.bakeoff.canonical_sha256(identities),
            "companion_inventory_sha256": digest(self.companion),
            "mtp_matrix_quant_contract": "Q4_0_ROCMI4", "mtp_depth": depth,
            "artifact_bytes": shard["size_bytes"],
            "companion_artifact_bytes": {"mtp": self.mtp.stat().st_size,
                                         "vision_mmproj": self.mmproj.stat().st_size},
            "quantization_arm": "rocmi4-q6k-embedding-head",
            "intervention_configuration_id": "lambda-0.25-all-48",
            "builder_identity": attestation["builder_identity"],
            "tensor_format_compatibility_sha256": self.format_sha,
        }

    def ledger(self, phase: str, **extra: object) -> Path:
        value = {"schema": manifest.bakeoff.LEDGER_SCHEMA, "phase": phase,
                 "plan_sha256": manifest.bakeoff.canonical_sha256(self.plan),
                 "external_attestation_verified": True, "publication_allowed": False,
                 **extra}
        return write(self.root / f"{phase}-ledger.json", value)


class CandidateManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.verify_plan = manifest.bakeoff.verify_plan
        self.verify_ledger = manifest.bakeoff.verify_ledger_semantics
        self.quality = manifest.bakeoff.audited_quality
        manifest.bakeoff.verify_plan = lambda value: value
        manifest.bakeoff.verify_ledger_semantics = lambda plan, ledger: None
        manifest.bakeoff.audited_quality = lambda row, corpus: {"passes": True}

    def tearDown(self) -> None:
        manifest.bakeoff.verify_plan = self.verify_plan
        manifest.bakeoff.verify_ledger_semantics = self.verify_ledger
        manifest.bakeoff.audited_quality = self.quality

    def test_stock_and_sweep_emit_distinct_weight_source_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            stock, _ = fixture.construction("stock", "stock-rocmi4-exact", stock=True)
            normalized = manifest.normalize(fixture.args(stock, "stock", "stock-rocmi4-exact", 3))
            row = normalized["candidate"]
            self.assertIsNone(row["intervention_manifest"])
            self.assertEqual(row["intervention_manifest_sha256"], "0" * 64)
            self.assertEqual(row["stock_capture"], desc(fixture.capture))
            accumulator = write(fixture.root / "sweep-acc.json", {
                "schema": manifest.ACCUMULATOR_SCHEMA, "phase": "sweep",
                "plan_sha256": digest(fixture.plan_path), "assessments": [{}],
                "contains_raw_measurements": False, "external_attestation_required": True,
                "publication_allowed": False,
            })
            sweep, _ = fixture.construction("sweep", "lambda-0.25-all-48")
            args = fixture.args(sweep, "sweep", "lambda-0.25-all-48", 3,
                                prior_accumulator=accumulator,
                                prior_accumulator_sha256=digest(accumulator))
            row = manifest.normalize(args)["candidate"]
            self.assertIsNone(row["stock_capture"])
            self.assertEqual(row["intervention_manifest"], str(fixture.intervention))

    def test_format_depth_and_final_reuse_one_exact_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            row_id = "rocmi4-q6k-main-rocmi4-mtp-d3"
            construction, value = fixture.construction("format", row_id)
            runtime = {"ember_revision": fixture.hex40,
                       "container_digest": fixture.runtime_digest,
                       "tensor_format_contract_sha256": fixture.format_sha,
                       "engine_binary_sha256": "a" * 64}
            sweep = fixture.ledger("sweep", selected_configuration_id="lambda-0.25-all-48",
                                   runtime_identity=runtime)
            format_args = fixture.args(construction, "format", row_id, 3,
                                       prior_ledger=sweep, prior_ledger_sha256=digest(sweep))
            format_row = manifest.normalize(format_args)["candidate"]
            selected = fixture.artifact_identity(value, 3)
            format_ledger = fixture.ledger(
                "format", selected_configuration_id="lambda-0.25-all-48",
                selected_arm_id=row_id, selected_arm=fixture.plan["format_arms"][0],
                selected_artifact_identity=selected, runtime_identity=runtime)
            depth_args = fixture.args(
                construction, "mtp-depth", "mtp-depth-1", 1,
                prior_ledger=format_ledger, prior_ledger_sha256=digest(format_ledger))
            depth_row = manifest.normalize(depth_args)["candidate"]
            self.assertEqual(depth_row["candidate_id"], format_row["candidate_id"])
            self.assertEqual(depth_row["model_sha256"], format_row["model_sha256"])
            selected_depth = fixture.artifact_identity(value, 1)
            depth_ledger = fixture.ledger(
                "mtp-depth", selected_configuration_id="lambda-0.25-all-48",
                selected_arm_id=row_id, selected_arm=fixture.plan["format_arms"][0],
                selected_mtp_matrix_quant_contract="Q4_0_ROCMI4", selected_mtp_depth=1,
                selected_artifact_identity=selected_depth, runtime_identity=runtime)
            final_plan = write(fixture.root / "final-plan.json", {
                "selection_plan": fixture.plan,
                "sealed_recipe_ledger": {
                    "sha256": digest(depth_ledger),
                    "descriptor": {"subject": desc(depth_ledger)},
                },
                "corpora": {"final-heldout.jsonl": {"sha256": "b" * 64}},
            })
            final_args = fixture.args(
                construction, "final", "final-confirmation", 1,
                prior_ledger=depth_ledger, prior_ledger_sha256=digest(depth_ledger),
                final_plan=final_plan, final_plan_sha256=digest(final_plan))
            final_row = manifest.normalize(final_args)["candidate"]
            self.assertIsNone(final_row["corpus_sha256"])
            self.assertEqual(final_row["model_sha256"], depth_row["model_sha256"])

    def test_tampered_companion_depth_runtime_and_serial_order_fail(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            construction, _ = fixture.construction("sweep", "lambda-0.25-all-48")
            base = fixture.args(construction, "sweep", "lambda-0.25-all-48", 3)
            with self.assertRaisesRegex(manifest.ManifestError, "next row"):
                manifest.normalize(base)
            accumulator = write(fixture.root / "acc.json", {
                "schema": manifest.ACCUMULATOR_SCHEMA, "phase": "sweep",
                "plan_sha256": digest(fixture.plan_path), "assessments": [{}],
                "contains_raw_measurements": False, "external_attestation_required": True,
                "publication_allowed": False,
            })
            base.prior_accumulator = accumulator
            base.prior_accumulator_sha256 = digest(accumulator)
            base.mtp_depth = 4
            with self.assertRaisesRegex(manifest.ManifestError, "sweep normalization"):
                manifest.normalize(base)
            base.mtp_depth = 3
            base.runtime_tensor_format_contract_sha256 = "f" * 64
            with self.assertRaisesRegex(manifest.ManifestError, "cannot decode"):
                manifest.normalize(base)
            base.runtime_tensor_format_contract_sha256 = fixture.format_sha
            fixture.mtp.write_bytes(b"tampered")
            with self.assertRaisesRegex(manifest.ManifestError, "MTP companion .*differs"):
                manifest.normalize(base)

    def test_cli_refuses_overwrite_and_never_mentions_publish_or_delete_commands(self) -> None:
        source = (ROOT / "scripts" / "qwen_candidate_manifest.py").read_text(encoding="utf-8")
        for forbidden in ("docker push", "hf upload", "delete-loser", "retire-captured-stock"):
            self.assertNotIn(forbidden, source)
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            construction, _ = fixture.construction("stock", "stock-rocmi4-exact", stock=True)
            args = fixture.args(construction, "stock", "stock-rocmi4-exact", 3)
            args.output.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(manifest.ManifestError, "refusing output"):
                manifest.write_new(args.output, manifest.normalize(args))
            self.assertEqual(args.output.read_text(encoding="utf-8"), "keep")

    def test_from_request_is_exact_and_writes_one_new_v3_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            construction, _ = fixture.construction(
                "stock", "stock-rocmi4-exact", stock=True)
            output = fixture.root / "from-request-v3.json"
            request = write(fixture.root / "normalization-request.json", {
                "schema": manifest.REQUEST_SCHEMA,
                "construction": desc(construction), "stage": "stock",
                "row_id": "stock-rocmi4-exact",
                "mtp_matrix_quant_contract": "Q4_0_ROCMI4", "mtp_depth": 3,
                "runtime_mode": "exact_dequant",
                "runtime": {"revision": fixture.hex40,
                    "release_ref": fixture.release_ref,
                    "release_digest": fixture.runtime_digest,
                    "dev_ref": fixture.dev_ref, "dev_digest": fixture.dev_digest,
                    "tensor_format_contract_sha256": fixture.format_sha},
                "quality_contract": None, "prior_accumulator": None,
                "prior_ledger": None, "final_plan": None,
                "output": str(output), "publishes": False, "deletes": False,
            })
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(manifest.main([
                    "from-request", "--request", str(request),
                    "--request-sha256", digest(request),
                ]), 0)
            self.assertEqual(json.loads(output.read_text())["schema"], manifest.MANIFEST_SCHEMA)
            malformed = json.loads(request.read_text())
            malformed["publishes"] = True
            write(request, malformed)
            with self.assertRaisesRegex(manifest.ManifestError, "lifecycle"):
                manifest.request_namespace(request, digest(request))


class RealPlanIntegrationTests(unittest.TestCase):
    def test_stock_normalizes_against_reproduced_checked_policy_plan(self) -> None:
        """Exercise the real verifier instead of the unit fixture monkeypatches."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            corpus_dir = root / "real-corpora"
            corpus_dir.mkdir()
            names = ("extraction-good.jsonl", "extraction-bad.jsonl",
                     "sweep-validation.jsonl", "final-heldout.jsonl")
            artifacts = []
            for index, name in enumerate(names):
                path = corpus_dir / name
                path.write_text(json.dumps({"id": f"case-{index}"}) + "\n", encoding="utf-8")
                artifacts.append({"filename": name, "sha256": digest(path),
                                  "record_count": 1})
            for name, rows in (("qwen-selection-corpora-manifest.json", artifacts[:3]),
                               ("qwen-final-corpus-manifest.json", artifacts[3:])):
                write(corpus_dir / name, {
                    "partition": {"pairwise_request_overlap_count": 0},
                    "artifacts": rows,
                })
            fixture = Fixture(root)
            fixture.plan = manifest.bakeoff.make_plan(
                manifest.bakeoff.DEFAULT_RECIPE, corpus_dir)
            fixture.plan_path = write(root / "real-selection-plan.json", fixture.plan)
            fixture.set_companion_source(fixture.plan["direction_basis"]["source"])
            construction, _ = fixture.construction(
                "stock", fixture.plan["stock_control"]["id"], stock=True)
            result = manifest.normalize(fixture.args(
                construction, "stock", fixture.plan["stock_control"]["id"], 3))
            self.assertEqual(result["schema"], manifest.MANIFEST_SCHEMA)
            self.assertEqual(manifest.bakeoff.verify_plan(fixture.plan), fixture.plan)
            self.assertEqual(set(result["shared_companions"]), manifest.MTP_CONTRACTS)
            self.assertEqual(result["candidate"]["stock_capture"], desc(fixture.capture))


if __name__ == "__main__":
    unittest.main()
