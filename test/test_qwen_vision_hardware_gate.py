#!/usr/bin/env python3
"""GPU-free contract tests for the protected Qwen vision hardware gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_vision_differential.py"
GATE = ROOT / "scripts" / "qwen_vision_real_weight_gate.sh"
SAMPLER = ROOT / "scripts" / "qwen_vision_residency.py"
CORPUS = ROOT / "share" / "quant_eval" / "qwen3.8-vision-differential-v2.json"
WORKFLOW = ROOT / ".github" / "workflows" / "qwen-gfx1151-vision.yml"
DISPATCH = ROOT / ".github" / "workflows" / "gfx1151-certify.yml"
PUBLICATION = ROOT / "scripts" / "qwen_publication_envelope.py"
SPEC = importlib.util.spec_from_file_location("qwen_vision_differential", SCRIPT)
assert SPEC and SPEC.loader
vision = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(vision)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_capture(path: Path, values: list[float], *, rows: int = 1,
                  grid_h: int = 2, grid_w: int = 2) -> None:
    assert len(values) == rows * 2560
    path.write_bytes(b"EVISF32\0" + struct.pack(
        "<6Q", 1, 1, grid_h, grid_w, 2560, rows) +
        struct.pack(f"<{len(values)}f", *values))


class QwenVisionHardwareGateTest(unittest.TestCase):
    def test_corpus_materializes_exact_images_and_requests(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = argparse.Namespace(corpus=CORPUS.resolve(), corpus_sha256=sha(CORPUS),
                                      output_dir=root / "corpus")
            result = vision.materialize(args)
            self.assertEqual(len(result["cases"]), 2)
            for row in result["cases"]:
                self.assertEqual(sha(Path(row["image"]["path"])), row["image"]["sha256"])
                request = json.loads(Path(row["request"]["path"]).read_text())
                self.assertEqual(request["temperature"], 0)
                self.assertEqual(request["reasoning_effort"], "none")
                self.assertEqual(request["max_tokens"], 32)
                self.assertTrue(request["messages"][0]["content"][0]
                                ["image_url"]["url"].startswith("data:image/png;base64,"))

    def fixture(self, root: Path) -> argparse.Namespace:
        corpus_value, cases = vision.load_corpus(CORPUS.resolve(), sha(CORPUS))
        del corpus_value
        ember = root / "ember"; reference = root / "reference"; responses = root / "responses"
        ember.mkdir(); reference.mkdir(); responses.mkdir()
        values = [float(i % 17) / 16 for i in range(2560)]
        response_texts = {
            "checkerboard-56": "A blue and white checkerboard pattern.",
            "rgb-bands-84x56": "red, green, blue",
        }
        for case_index, (row, _image) in enumerate(cases):
            write_capture(reference / f"{row['id']}.f32", values)
            for run_index, run in enumerate(("cold", "warm")):
                write_capture(ember / f"ember.{run_index * len(cases) + case_index:06d}.f32",
                              [value + (1e-7 if run == "warm" else 0) for value in values])
                (responses / f"{run}-{row['id']}.json").write_text(json.dumps({
                    "choices": [{"message": {"content": response_texts[row["id"]]}}]}))
        identity_path = root / "identity.json"
        identity_path.write_text(json.dumps({
            "schema": vision.IDENTITY_SCHEMA,
            "runtime": {"image": "ghcr.io/otheru-ai/ember@sha256:" + "1" * 64,
                        "ember_revision": "2" * 40, "engine_binary_sha256": "3" * 64},
            "model": {"path": "/evidence/model-inventory.json", "sha256": "4" * 64,
                      "model_inventory_sha256": "5" * 64,
                      "first_shard_path": "/models/model.gguf",
                      "first_shard_sha256": "6" * 64,
                      "build_record_path": "/models/build.json",
                      "build_record_sha256": "c" * 64},
            "mtp": {"path": "/models/mtp.gguf", "sha256": "7" * 64,
                    "size_bytes": 100, "depth": 2},
            "vision_mmproj": {"path": "/models/mmproj.gguf", "sha256": "8" * 64,
                              "size_bytes": 200, "format": "BF16"},
            "vision_vocab": {"path": "/models/vision-vocab.gguf", "sha256": "d" * 64,
                              "size_bytes": 300, "format": "GGUF_VOCAB_ONLY",
                              "metadata_sha256": "e" * 64},
            "provider": {"path": "/opt/provider.so", "sha256": "9" * 64,
                         "abi_version": 1, "llama_cpp_revision": vision.LLAMA_REVISION},
            "reference": {"path": "/evidence/reference", "sha256": "a" * 64,
                          "image": "ghcr.io/otheru-ai/ember-dev@sha256:" + "b" * 64,
                          "llama_cpp_revision": vision.LLAMA_REVISION},
            "hardware": {"gpu_arch": "gfx1151", "rocm_version": "10.0.0"},
            "corpus": {"path": str(CORPUS.resolve()), "sha256": sha(CORPUS)},
        }))
        memtotal = vision.RUNNER_MEMTOTAL_BYTES
        mem_available = 30 * 1024**3
        uma_used = memtotal - mem_available
        phases = {name: {"samples": 3, "rss_peak_bytes": 80 * 1024**3,
                         "gtt_peak_bytes": 82 * 1024**3,
                         "uma_used_peak_bytes": uma_used,
                         "mem_available_floor_bytes": mem_available}
                  for name in ("idle", "cold", "warm", "settled")}
        raw_samples = []
        for phase_index, name in enumerate(("idle", "cold", "warm", "settled")):
            for sample_index in range(3):
                raw_samples.append({
                    "monotonic": float(phase_index * 3 + sample_index), "phase": name,
                    "memtotal_bytes": memtotal,
                    "mem_available_bytes": mem_available,
                    "uma_used_bytes": uma_used,
                    "rss_bytes": 80 * 1024**3, "hwm_bytes": 80 * 1024**3,
                    "gtt_bytes": 82 * 1024**3})
        residency = root / "residency.json"
        residency.write_text(json.dumps({
            "schema": "ember.qwen3.8.vision-residency.v1",
            "method": "runner_phase_rss_gtt_uma_sampler_v1", "server_host_pid": 42,
            "sample_interval_seconds": 0.05, "runner_memtotal_bytes": memtotal,
            "runner_gtt_pages_limit": vision.RUNNER_GTT_PAGES_LIMIT, "phases": phases,
            "raw_samples": raw_samples}))
        return argparse.Namespace(
            corpus=CORPUS.resolve(), corpus_sha256=sha(CORPUS),
            identity=identity_path, identity_sha256=sha(identity_path),
            residency=residency, residency_sha256=sha(residency),
            ember_dir=ember, reference_dir=reference, response_dir=responses,
            output=root / "vision-certified.json")

    def test_float_differential_and_cold_warm_residency_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self.fixture(Path(raw))
            value = vision.finalize(args)
            self.assertTrue(value["passed"])
            self.assertEqual(len(value["comparisons"]), 4)
            self.assertTrue(all(row["mismatches"] == 0 for row in value["comparisons"]))
            vision.verify(value, "2" * 40)

    def test_mismatch_and_under_sampled_residency_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self.fixture(Path(raw))
            first = sorted(args.ember_dir.glob("*.f32"))[0]
            write_capture(first, [1.0] * 2560)
            with self.assertRaisesRegex(vision.VisionEvidenceError, "outside tolerance"):
                vision.finalize(args)
        with tempfile.TemporaryDirectory() as raw:
            args = self.fixture(Path(raw)); value = json.loads(args.residency.read_text())
            value["phases"]["cold"]["samples"] = 1
            args.residency.write_text(json.dumps(value)); args.residency_sha256 = sha(args.residency)
            with self.assertRaisesRegex(vision.VisionEvidenceError, "undersampled"):
                vision.finalize(args)
        with tempfile.TemporaryDirectory() as raw:
            args = self.fixture(Path(raw)); value = json.loads(args.residency.read_text())
            value["phases"]["warm"]["uma_used_peak_bytes"] = vision.RUNNER_MEMTOTAL_BYTES
            value["phases"]["warm"]["mem_available_floor_bytes"] = 0
            for sample in value["raw_samples"]:
                if sample["phase"] == "warm":
                    sample["mem_available_bytes"] = 0
                    sample["uma_used_bytes"] = vision.RUNNER_MEMTOTAL_BYTES
            args.residency.write_text(json.dumps(value)); args.residency_sha256 = sha(args.residency)
            with self.assertRaisesRegex(vision.VisionEvidenceError, "UMA cap"):
                vision.finalize(args)
        for field, replacement in (
                ("runner_memtotal_bytes", vision.RUNNER_MEMTOTAL_BYTES + 4096),
                ("runner_gtt_pages_limit", vision.RUNNER_GTT_PAGES_LIMIT - 1)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                args = self.fixture(Path(raw)); value = json.loads(args.residency.read_text())
                value[field] = replacement
                args.residency.write_text(json.dumps(value))
                args.residency_sha256 = sha(args.residency)
                with self.assertRaisesRegex(vision.VisionEvidenceError, "exact 128-GiB"):
                    vision.finalize(args)

    def test_verify_rejects_minimal_duplicate_and_tampered_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self.fixture(Path(raw))
            baseline = vision.finalize(args)
            mutations = []

            duplicate = copy.deepcopy(baseline)
            duplicate["comparisons"][-1] = copy.deepcopy(duplicate["comparisons"][0])
            mutations.append(("coverage", duplicate))

            extra = copy.deepcopy(baseline)
            extra["comparisons"].append(copy.deepcopy(extra["comparisons"][0]))
            mutations.append(("lacks", extra))

            for field, replacement in (("values_compared", 1), ("mismatches", 1),
                                       ("max_tolerance_ratio", 1.01),
                                       ("max_absolute_error", float("nan"))):
                changed = copy.deepcopy(baseline)
                changed["comparisons"][0][field] = replacement
                mutations.append(("comparison", changed))

            changed = copy.deepcopy(baseline)
            changed["comparisons"][0]["ember"]["size_bytes"] += 4
            mutations.append(("byte count", changed))
            changed = copy.deepcopy(baseline)
            changed["comparisons"][0]["response"]["path"] = "relative.json"
            mutations.append(("descriptor", changed))
            changed = copy.deepcopy(baseline)
            changed["comparisons"][0]["response_semantics"]["passed"] = False
            mutations.append(("semantic proof", changed))
            changed = copy.deepcopy(baseline)
            changed["comparisons"][0]["response_semantics"]["ordered_matches"] = ["blue"]
            mutations.append(("semantic proof", changed))
            changed = copy.deepcopy(baseline)
            changed["corpus"]["sha256"] = "0" * 64
            mutations.append(("corpus SHA", changed))
            changed = copy.deepcopy(baseline)
            changed["residency"]["evidence"]["sha256"] = "x" * 64
            mutations.append(("descriptor", changed))
            changed = copy.deepcopy(baseline)
            changed["residency"]["certified_peak_cap_bytes"] -= 1
            mutations.append(("cap", changed))
            changed = copy.deepcopy(baseline)
            changed["residency"]["measurement"]["raw_samples"][0]["uma_used_bytes"] = 1
            mutations.append(("incoherent", changed))

            for expected, changed in mutations:
                with self.subTest(expected=expected):
                    with self.assertRaisesRegex(vision.VisionEvidenceError, expected):
                        vision.verify(changed, "2" * 40)

    def test_empty_image_text_response_is_not_proof(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self.fixture(Path(raw))
            response = sorted(args.response_dir.glob("*.json"))[0]
            response.write_text(json.dumps({"choices": [{"message": {
                "content": "", "reasoning_content": ""}}]}))
            with self.assertRaisesRegex(vision.VisionEvidenceError, "incomplete"):
                vision.finalize(args)

    def test_ungrounded_or_misordered_response_is_not_proof(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self.fixture(Path(raw))
            response = args.response_dir / "cold-checkerboard-56.json"
            response.write_text(json.dumps({"choices": [{"message": {
                "content": "A geometric pattern."}}]}))
            with self.assertRaisesRegex(vision.VisionEvidenceError,
                                        "image-grounded response term"):
                vision.finalize(args)
        with tempfile.TemporaryDirectory() as raw:
            args = self.fixture(Path(raw))
            response = args.response_dir / "warm-rgb-bands-84x56.json"
            response.write_text(json.dumps({"choices": [{"message": {
                "content": "blue, green, red"}}]}))
            with self.assertRaisesRegex(vision.VisionEvidenceError,
                                        "misorders image-grounded response terms"):
                vision.finalize(args)

    def test_gate_is_exclusive_unprivileged_and_restores_production(self) -> None:
        body = GATE.read_text(encoding="utf-8")
        for required in ('trap cleanup EXIT INT TERM', 'sudo -n "$GPU_LOCK" acquire',
                         'sudo -n "$PRODUCTION" stop', 'sudo -n "$PRODUCTION" mask',
                         'sudo -n "$PRODUCTION" unmask', 'sudo -n "$PRODUCTION" start',
                         '--user "$runner_uid:$runner_gid"',
                         'from qwen_integrity_cache import IntegrityCache',
                         '--integrity-cache',
                         'DFLASH_QWEN_VISION_CAPTURE_PREFIX', 'printf idle',
                         'DFLASH_QWEN_VISION_TEXT_MODEL=/gate/vision-vocab.gguf',
                         'printf settled', 'restore_exclusive || die'):
            self.assertIn(required, body)
        self.assertIn('-v "$(dirname "$MODEL"):/gate/model:ro"', body)
        self.assertLess(body.index('sudo -n "$GPU_LOCK" acquire'),
                        body.index("from qwen_integrity_cache import IntegrityCache"))
        self.assertIn('/gate/vision-vocab.gguf /gate/mmproj.gguf', body)
        self.assertNotIn("hf upload", body)
        subprocess.run(["bash", "-n", str(GATE)], check=True)

    def test_workflow_and_publication_are_connected_fail_closed(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        dispatch = DISPATCH.read_text(encoding="utf-8")
        publication = PUBLICATION.read_text(encoding="utf-8")
        self.assertIn("qwen_vision_real_weight_gate.sh", workflow)
        self.assertIn("qwen_vision_differential.py verify", workflow)
        self.assertIn("workflow_call:", workflow)
        self.assertIn("contains(github.workflow_ref, '/.github/workflows/gfx1151-certify.yml@')",
                      workflow)
        self.assertIn('"vision": {"runtime_image", "runtime_dev_image"', dispatch)
        self.assertIn("qwen-call-vision:", dispatch)
        self.assertIn("uses: ./.github/workflows/qwen-gfx1151-vision.yml", dispatch)
        self.assertIn("vision_evidence", publication)
        self.assertIn('source("vision")', publication)
        self.assertIn("vision evidence substitutes runtime or companion artifact bytes", publication)
        self.assertIn("ember.qwen3.8.hf-publication-envelope.v3", publication)
        global_permissions = dispatch.split("permissions:\n", 1)[1].split("\nconcurrency:", 1)[0]
        for permission in ("id-token: write", "attestations: write",
                           "artifact-metadata: write"):
            self.assertNotIn(permission, global_permissions)
        vision_job = dispatch.split("  qwen-call-vision:\n", 1)[1].split(
            "\n  qwen-inspect-control-residue:", 1)[0]
        for permission in ("contents: read", "packages: read", "id-token: write",
                           "attestations: write", "artifact-metadata: write"):
            self.assertIn(permission, vision_job)
        for required in ("id-token: write", "attestations: write",
                         "subject-path: ${{ steps.gate.outputs.output }}/vision-certified.json",
                         "gh attestation verify", "vision-certified.sigstore.json",
                         '--source-digest "$TARGET_SHA"', '--signer-digest "$TARGET_SHA"',
                         'test "$GITHUB_SHA" = "$TARGET_SHA"', "model-inventory.json",
                         "ember.*.f32", "reference/*.f32"):
            self.assertIn(required, workflow)

    def test_reference_uses_public_mtmd_not_provider_adapter(self) -> None:
        source = (ROOT / "tools" / "qwen4exp_vision_reference.cpp").read_text()
        self.assertIn("mtmd_encode_chunk", source)
        self.assertIn("mtmd_get_output_embd", source)
        self.assertNotIn("qwen4exp_vision_provider_get_v1", source)
        provider = (ROOT / "tools" / "qwen4exp_vision_provider_llamacpp.cpp").read_text()
        self.assertIn("could not durably write vision capture", provider)
        self.assertIn("O_EXCL", provider)


if __name__ == "__main__":
    unittest.main()
