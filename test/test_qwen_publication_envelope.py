#!/usr/bin/env python3
"""GPU-free tests for the Qwen protected publication handoff."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "qwen_publication_envelope.py"
WORKFLOW = ROOT / ".github" / "workflows" / "qwen-hf-candidate.yml"
SPEC = importlib.util.spec_from_file_location("qwen_publication_envelope", SCRIPT)
assert SPEC and SPEC.loader
qpe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qpe)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QwenPublicationEnvelopeTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[argparse.Namespace, dict, dict, dict]:
        package = root / "package"
        package.mkdir()
        runtime_image = "ghcr.io/otheru-ai/ember@sha256:" + "2" * 64
        engine_revision = "1" * 40
        model = package / "model.gguf"
        model.write_bytes(b"model-bytes")
        mmproj = package / "Qwen3.8-Flash-Next-BF16-mmproj.gguf"
        mmproj.write_bytes(b"vision-bytes")
        model_sha = digest(model)
        model_inventory = [{"index": 1, "sha256": model_sha,
                            "bytes": model.stat().st_size}]
        identity = {
            "candidate_id": "winner", "build_record_sha256": "",
            "intervention_manifest_sha256": "", "profile_sha256": "",
            "quantization_overrides_sha256": "3" * 64,
            "model_inventory_sha256": qpe.sha256_bytes(qpe.canonical(model_inventory)),
            "companion_inventory_sha256": "4" * 64,
            "mtp_matrix_quant_contract": "Q4_0_ROCMI4", "mtp_depth": 2,
            "artifact_bytes": model.stat().st_size,
            "companion_artifact_bytes": {"mtp": 1, "vision_mmproj": 1},
            "quantization_arm": "rocm-i4", "intervention_configuration_id": "lambda-1",
            "builder_identity": {"ember_revision": engine_revision,
                                 "quantizer_tool_sha256": "5" * 64,
                                 "container_digest": "sha256:" + "6" * 64,
                                 "tensor_format_contract_sha256": "7" * 64},
            "tensor_format_compatibility_sha256": "7" * 64,
        }
        runtime = {"ember_revision": engine_revision, "engine_binary_sha256": "8" * 64,
                   "container_digest": "sha256:" + "2" * 64,
                   "tensor_format_contract_sha256": "7" * 64}
        files = {
            "README.md": b"card", "LICENSE": b"license", "SHA256SUMS": b"sums",
            "release-profile.json": b"{}", "qwen-intervention-manifest.json":
                json.dumps({"status": "complete"}).encode(),
            "qwen-quant-build-record.json":
                json.dumps({"status": "complete", "publishes": False}).encode(),
        }
        for name, raw in files.items():
            (package / name).write_bytes(raw)
        identity["build_record_sha256"] = digest(package / "qwen-quant-build-record.json")
        identity["intervention_manifest_sha256"] = digest(
            package / "qwen-intervention-manifest.json")
        identity["profile_sha256"] = digest(package / "release-profile.json")
        revision = "candidate/source-engine"
        artifact_manifest = {
            "candidate": {"repo_id": qpe.EXPECTED_REPO, "repo_type": "model",
                          "revision": revision, "published": False},
            "build": {"engine_revision": engine_revision,
                      "container_image": runtime_image},
            "artifacts": [{"filename": model.name, "size_bytes": model.stat().st_size,
                           "sha256": model_sha}],
            "companion_artifacts": [{
                "role": "vision_mmproj", "filename": mmproj.name, "format": "BF16",
                "required_for": "multimodal", "size_bytes": mmproj.stat().st_size,
                "sha256": digest(mmproj)}],
        }
        (package / "artifact-manifest.json").write_text(json.dumps(artifact_manifest))
        planned = []
        for path in sorted(package.iterdir()):
            planned.append({"local_path": str(path.resolve()), "path_in_repo": path.name,
                            "size_bytes": path.stat().st_size, "sha256": digest(path)})
        plan = {
            "schema_version": 1, "action": "candidate_upload_plan", "publishes": False,
            "repo_id": qpe.EXPECTED_REPO, "repo_type": "model", "revision": revision,
            "files": planned,
            "authentication": {
                "preferred": "Hugging Face Trusted Publisher via GitHub OIDC",
                "fallback": "fine-grained HF_TOKEN with write access only to the target model repository",
                "token_embedded": False},
            "promotion": {"allowed": False,
                          "requires": "all documented release gates and exact candidate commit verification"},
            "publication_blockers": [],
        }
        plan_path = package / "upload-plan.json"
        plan_path.write_text(json.dumps(plan))
        evidence_paths = {}
        for name in ("ledger", "attestation", "measurement", "quality", "hardware"):
            path = root / f"{name}.json"
            path.write_text("{}")
            evidence_paths[name] = path
        metrics = {"passes": True}
        ledger = {"selected_artifact_identity": identity, "final_metrics": metrics}
        assessment = {"runtime_identity": runtime, "artifact_identity": identity}
        args = argparse.Namespace(
            upload_plan=plan_path, upload_plan_sha256=digest(plan_path),
            final_ledger=evidence_paths["ledger"],
            final_ledger_sha256=digest(evidence_paths["ledger"]),
            final_ledger_attestation=evidence_paths["attestation"],
            final_ledger_attestation_sha256=digest(evidence_paths["attestation"]),
            measurement_manifest=evidence_paths["measurement"],
            measurement_manifest_sha256=digest(evidence_paths["measurement"]),
            quality_contract=evidence_paths["quality"],
            quality_contract_sha256=digest(evidence_paths["quality"]),
            hardware_evidence=evidence_paths["hardware"],
            hardware_evidence_sha256=digest(evidence_paths["hardware"]),
            runtime_image=runtime_image)
        measurement = {"artifacts": {"vision_mmproj": {
            "sha256": digest(mmproj), "bytes": mmproj.stat().st_size}}}
        return args, ledger, assessment, measurement

    def test_envelope_binds_package_runtime_and_forbids_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args, ledger, assessment, measurement = self.fixture(Path(raw))
            with mock.patch.object(qpe, "validate_evidence",
                                   return_value=(ledger, assessment, measurement, {})):
                value = qpe.assemble(args, "2026-08-27T12:00:00Z")
            self.assertEqual(value["schema"], qpe.SCHEMA)
            self.assertTrue(value["authorization"]["candidate_upload"])
            self.assertFalse(value["authorization"]["promotion"])
            self.assertFalse(value["publishes"])
            self.assertEqual(value["runtime"]["ember_revision"], "1" * 40)
            self.assertEqual(value["candidate"]["revision"], "candidate/source-engine")
            self.assertEqual(
                value["attestation_policy"]["signer_workflow"],
                "OtherU-AI/ember/.github/workflows/qwen-gfx1151-bakeoff.yml")

    def test_plan_rejects_destination_escape_and_changed_bytes(self) -> None:
        for mutation, expected in (("escape", "malformed"), ("bytes", "SHA-256")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                args, _, _, _ = self.fixture(Path(raw))
                plan = json.loads(args.upload_plan.read_text())
                if mutation == "escape":
                    plan["files"][0]["path_in_repo"] = "../outside"
                    args.upload_plan.write_text(json.dumps(plan))
                    args.upload_plan_sha256 = digest(args.upload_plan)
                else:
                    Path(plan["files"][0]["local_path"]).write_bytes(b"changed")
                with self.assertRaisesRegex(qpe.EnvelopeError, expected):
                    qpe.validate_upload_plan(args.upload_plan, args.upload_plan_sha256)

    def test_remote_verification_is_commit_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args, ledger, assessment, measurement = self.fixture(root)
            with mock.patch.object(qpe, "validate_evidence",
                                   return_value=(ledger, assessment, measurement, {})):
                value = qpe.assemble(args, "2026-08-27T12:00:00Z")
            remote = root / "remote"
            remote.mkdir()
            for row in value["package"]["files"]:
                (remote / row["path_in_repo"]).write_bytes(Path(row["local_path"]).read_bytes())
            receipt = qpe.verify_remote(value, remote, "9" * 40)
            self.assertEqual(receipt["candidate_commit"], "9" * 40)
            self.assertFalse(receipt["promotion_allowed"])
            (remote / value["package"]["files"][0]["path_in_repo"]).write_bytes(b"tamper")
            with self.assertRaisesRegex(qpe.EnvelopeError, "SHA-256"):
                qpe.verify_remote(value, remote, "9" * 40)

    def test_publisher_is_oidc_candidate_only_and_no_runtime_installer(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        for required in ("id-token: write", "environment: qwen-hf-candidate",
                         "HF_OIDC_RESOURCE", "hf auth token", "--no-exist-ok",
                         "verify-remote", "candidate_commit",
                         "git merge-base --is-ancestor",
                         "Promotion was not requested or performed"):
            self.assertIn(required, body)
        for forbidden in ("HF_TOKEN: ${{ secrets", "pip install", "curl -LsSf",
                          "hf repo branch delete", "--revision main --commit-message"):
            self.assertNotIn(forbidden, body)
        self.assertNotIn("workflow_call", body)
        self.assertNotIn("push:", body)
        self.assertNotIn("promotion:", body)

    def test_offline_envelope_tool_has_no_network_or_credentials(self) -> None:
        body = SCRIPT.read_text(encoding="utf-8")
        for forbidden in ("urllib", "requests", "huggingface_hub", "os.environ",
                          "os.getenv", "subprocess"):
            self.assertNotIn(forbidden, body)


if __name__ == "__main__":
    unittest.main()
