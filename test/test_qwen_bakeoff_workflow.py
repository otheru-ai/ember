#!/usr/bin/env python3
"""Static/adversarial contracts for the dedicated gfx1151 Qwen bakeoff."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/qwen-gfx1151-bakeoff.yml"
TARGET_GATE = ROOT / "scripts/qwen_target_only_gate.sh"
HEX = "1" * 64
ATTEST_SHA = "1e69f48acb82d1966a394da916b4c1698aa569d6"


def workflow_run_blocks(text: str) -> list[str]:
    """Extract YAML literal run blocks without needing a PyYAML dependency."""
    lines = text.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        match = re.match(r"^(\s*)run:\s*\|\s*$", lines[index])
        if match is None:
            index += 1
            continue
        base = len(match.group(1)); index += 1; body: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line and len(line) - len(line.lstrip(" ")) <= base:
                break
            body.append(line[base + 2:] if line else "")
            index += 1
        blocks.append("\n".join(body) + "\n")
    return blocks


class QwenBakeoffWorkflowTest(unittest.TestCase):
    def test_target_gate_syntax_and_side_effect_free_dry_run(self) -> None:
        subprocess.run(["bash", "-n", str(TARGET_GATE)], check=True)
        with tempfile.TemporaryDirectory() as temporary:
            for tool in ("docker", "curl", "sudo", "dd", "python3"):
                path = Path(temporary) / tool
                path.write_text(f"#!/bin/sh\necho FORBIDDEN:{tool} >&2\nexit 97\n")
                path.chmod(0o755)
            env = os.environ | {"PATH": temporary + os.pathsep + os.environ["PATH"]}
            result = subprocess.run([
                "bash", str(TARGET_GATE), "--dry-run",
                "--image", "candidate:exact", "--image-digest", f"sha256:{HEX}",
                "--profile-image", "candidate-dev:exact",
                "--profile-image-digest", f"sha256:{HEX}",
                "--model", "/models/qwen.gguf", "--model-sha256", HEX,
                "--model-build-record", "/models/qwen-quant-build-record.json",
                "--model-build-record-sha256", HEX,
                "--bakeoff-manifest", "/models/bakeoff-candidates.json",
                "--bakeoff-manifest-sha256", HEX,
                "--candidate-id", "lambda-0.50-all-48",
                "--out-dir", "/tmp/qwen-target-only-never-created",
            ], cwd=ROOT, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("FORBIDDEN", result.stderr)
        self.assertIn("target-only baseline (MTP disabled)", result.stdout)
        self.assertIn("exact-2074", result.stdout)
        self.assertIn("prefill peak 412.0", result.stdout)
        self.assertIn("decode median 39.49", result.stdout)

    def test_workflow_yaml_shell_and_python_heredocs_parse(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        ruby = shutil.which("ruby")
        if ruby:
            subprocess.run([ruby, "-e", "require 'yaml'; YAML.parse_file(ARGV[0])",
                            str(WORKFLOW)], check=True)
        blocks = workflow_run_blocks(body)
        self.assertGreaterEqual(len(blocks), 10)
        for index, block in enumerate(blocks):
            neutral = re.sub(r"\$\{\{.*?\}\}", "github-expression", block)
            result = subprocess.run(["bash", "-n"], input=neutral, text=True,
                                    capture_output=True)
            self.assertEqual(result.returncode, 0, f"run block {index}: {result.stderr}")
            for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", neutral, re.S):
                compile(script, f"workflow-run-{index}-heredoc.py", "exec")

    def test_workflow_uses_v3_serial_immutable_workset_contract(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", body)
        self.assertNotIn("workflow_call:", body)
        self.assertIn("runs-on: [self-hosted, linux, x64, gfx1151]", body)
        self.assertIn("one-candidate-per-dispatch is required", body)
        self.assertIn("ember.qwen3.8.sequential-bakeoff-candidate.v3", body)
        self.assertIn("BF16 cache is not content-addressed beneath the workset", body)
        self.assertIn('{"ROCMI4", "ROCMFP4-FAST"}', body)
        self.assertIn("shared BF16 mmproj inventory differs", body)
        self.assertIn("candidate is not the next row in the canonical serial phase order", body)
        self.assertIn("candidate-workset-attestation.v1", body)
        self.assertIn("tensor_format_compatibility_sha256", body)
        self.assertIn("quantizer_tool_sha256", body)
        self.assertIn("engine_binary_sha256", body)

    def test_builder_and_runtime_revisions_are_not_conflated(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("builder_revision", body)
        self.assertIn("runtime_revision", body)
        self.assertIn('builder["ember_revision"] != builder_sha', body)
        self.assertIn('m.get("runtime_revision") != runtime_revision', body)
        self.assertIn('RUNTIME_ENGINE_SHA256="$(docker run', body)
        self.assertIn("comparative phase attempted to change the exact runtime engine identity", body)
        self.assertIn("Artifact builder revision", body)
        self.assertIn("Runtime engine revision", body)
        self.assertNotIn("Exact 40-character Ember revision used by both images", body)

    def test_final_data_is_sealed_until_attested_format_unlock(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        reject = body.index("final corpus must remain unavailable in the candidate manifest")
        unlock = body.index("--stage unlock-final")
        final_read = body.index('["final-heldout.jsonl"]["sha256"]')
        self.assertLess(reject, unlock)
        self.assertLess(unlock, final_read)
        self.assertIn("--prior-attestation-bundle", body)
        self.assertIn("final-confirmation is not the exact attested format winner", body)
        self.assertIn('c.get("id") != "final-confirmation"', body)
        self.assertIn("selection-only corpus", body)

    def test_every_retained_decision_is_externally_attested_and_verified(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("id-token: write", body)
        self.assertIn("attestations: write", body)
        self.assertIn("artifact-metadata: write", body)
        self.assertGreaterEqual(body.count(f"actions/attest@{ATTEST_SHA}"), 3)
        self.assertGreaterEqual(body.count("gh attestation verify"), 4)
        self.assertIn("ember.qwen3.8.candidate-assessment.v1", body)
        self.assertIn("ember.qwen3.8.sequential-bakeoff-accumulator.v2", body)
        self.assertIn('"contains_raw_measurements":False', body)
        self.assertIn("ember.qwen3.8.sequential-bakeoff-ledger.v2", body)
        self.assertIn("Retain and verify completed phase ledger", body)
        self.assertIn("subject-path: ${{ env.QWEN_ASSESSMENT }}", body)

    def test_measure_assess_attest_then_delete_order_is_fail_closed(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        target = body.index("scripts/qwen_target_only_gate.sh")
        mtp = body.index("scripts/qwen_real_weight_gate.sh")
        cleanup = body.index("Restore production and release exclusive GPU ownership")
        assess = body.index("--stage assess")
        attest = body.index("GitHub-attest candidate assessment")
        durable = body.index("externally-attested-candidate-assessment.v1")
        delete = body.index("scripts/qwen_candidate_builder.py delete-loser")
        self.assertLess(target, mtp)
        self.assertLess(mtp, cleanup)
        self.assertLess(cleanup, assess)
        self.assertLess(assess, attest)
        self.assertLess(attest, durable)
        self.assertLess(durable, delete)
        self.assertIn("provisional-retain", body)
        self.assertIn("it is not a loser until the phase ledger exists", body)
        self.assertIn("--measurement-only", body)

    def test_cleanup_accumulates_failures_and_requires_real_health(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("cleanup_failed=0", body)
        self.assertGreaterEqual(body.count("cleanup_failed=1"), 5)
        self.assertIn("ember-cert-production unmask", body)
        self.assertIn("ember-cert-production start", body)
        self.assertIn("ember-cert-production is-active", body)
        self.assertIn("http://127.0.0.1:8000/health", body)
        self.assertIn('exit "$cleanup_failed"', body)
        self.assertIn("ember-gpu-lock release", body)
        for forbidden in ("docker push", "huggingface-cli", "hf upload",
                          "actions/upload-artifact", "gh release"):
            self.assertNotIn(forbidden, body)


if __name__ == "__main__":
    unittest.main()
