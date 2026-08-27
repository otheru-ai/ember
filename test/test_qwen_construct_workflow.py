#!/usr/bin/env python3
"""Static safety contracts for serial Qwen candidate construction on gfx1151."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/qwen-gfx1151-construct.yml"


def workflow_run_blocks(text: str) -> list[str]:
    """Extract YAML literal run blocks without adding a PyYAML dependency."""
    lines = text.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        match = re.match(r"^(\s*)run:\s*\|\s*$", lines[index])
        if match is None:
            index += 1
            continue
        base = len(match.group(1))
        index += 1
        body: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line and len(line) - len(line.lstrip(" ")) <= base:
                break
            body.append(line[base + 2:] if line else "")
            index += 1
        blocks.append("\n".join(body) + "\n")
    return blocks


class QwenConstructWorkflowTest(unittest.TestCase):
    def test_yaml_shell_and_embedded_python_parse(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        ruby = shutil.which("ruby")
        if ruby:
            subprocess.run(
                [ruby, "-e", "require 'yaml'; YAML.parse_file(ARGV[0])", str(WORKFLOW)],
                check=True,
            )
        blocks = workflow_run_blocks(body)
        self.assertGreaterEqual(len(blocks), 10)
        for index, block in enumerate(blocks):
            neutral = re.sub(r"\$\{\{.*?\}\}", "github-expression", block)
            result = subprocess.run(
                ["bash", "-n"], input=neutral, text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 0, f"run block {index}: {result.stderr}")
            for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", neutral, re.S):
                compile(script, f"construction-run-{index}-heredoc.py", "exec")

    def test_exact_source_image_and_toolchain_bindings(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        for revision in (
            "f5d08274bafd880402bd16f5e3e6c514136ec06c",
            "abdc7a0bf815d3b83e26dd523c6960e4dd597e82",
            "c49ebdbd5c9f01ec242369f9e7f7967855f80cba",
        ):
            self.assertIn(revision, body)
        self.assertIn("PLE_PATCH_SHA256: 606880dd1e23", body)
        self.assertIn('builder="$repository@$builder_digest"', body)
        self.assertIn('runtime="$repository@$runtime_digest"', body)
        self.assertIn("converter-requirements.freeze.txt", body)
        self.assertIn("QWEN_CONVERTER_LOCK_SHA256", body)
        self.assertIn('test "$(git rev-parse HEAD)" = "$TARGET_SHA"', body)
        self.assertIn("org.opencontainers.image.revision", body)

    def test_construction_is_serial_same_path_uid_and_bounded(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("group: gfx1151-certification", body)
        self.assertIn("cancel-in-progress: false", body)
        self.assertIn("runs-on: [self-hosted, linux, x64, gfx1151]", body)
        self.assertIn("workspace=/var/tmp/ember-qwen3.8-flash-next", body)
        self.assertIn('-v "$QWEN_WORKSPACE:$QWEN_WORKSPACE"', body)
        self.assertIn('-v "$GITHUB_WORKSPACE:$GITHUB_WORKSPACE:ro"', body)
        self.assertEqual(body.count("docker run"), 9)
        self.assertEqual(body.count("--memory 125g"), body.count("docker run"))
        self.assertEqual(body.count("--memory-swap 125g"), body.count("docker run"))
        self.assertEqual(body.count('--user "$uid:$gid"'), body.count("docker run"))
        self.assertGreaterEqual(body.count("--memory-limit-bytes 134217728000"), 3)
        self.assertNotIn("/qwen-work/", body)
        self.assertNotIn("--user root", body)

    def test_modes_prepare_shared_inputs_and_build_one_candidate(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "options: [prepare-cache, prepare-companions, build-candidate, normalize-candidate]",
            body,
        )
        dispatch = re.search(r"workflow_dispatch:\n    inputs:\n(.*?)\npermissions:", body, re.S)
        self.assertIsNotNone(dispatch)
        inputs = re.findall(r"^      ([a-z0-9_]+):$", dispatch.group(1), re.M)
        self.assertEqual(inputs, ["commit_sha", "mode", "operation_request",
                                  "operation_request_sha256"])
        self.assertLessEqual(len(inputs), 10)
        self.assertIn("ember.qwen3.8.candidate-construction-request.v1", body)
        self.assertIn("scripts/qwen_candidate_builder.py\" prepare-cache", body)
        self.assertIn("Q4_0_ROCMI4", body)
        self.assertIn("Q4_0_ROCMFP4_FAST", body)
        self.assertIn("scripts/qwen_candidate_builder.py\" make-companion-inventory", body)
        self.assertIn('"schema":"ember.qwen3.8.companion-construction.v1"', body)
        self.assertEqual(body.count('scripts/qwen_candidate_builder.py" build-candidate'), 2)
        self.assertIn("--stock-control", body)
        self.assertIn("--stock-capture-manifest", body)
        self.assertIn("Rebuild the stock control from the immutable cache", body)
        self.assertIn("--intervention-manifest", body)
        self.assertIn("scripts/qwen_candidate_manifest.py from-request", body)
        self.assertNotIn("qwen_quantize.py", body)

    def test_expressions_reference_only_declared_dispatch_inputs(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        dispatch = re.search(r"workflow_dispatch:\n    inputs:\n(.*?)\npermissions:", body, re.S)
        self.assertIsNotNone(dispatch)
        declared = set(re.findall(r"^      ([a-z0-9_]+):$", dispatch.group(1), re.M))
        referenced = set(re.findall(r"\binputs\.([a-zA-Z0-9_]+)\b", body))
        self.assertEqual(referenced - declared, set(),
                         "workflow expressions reference undeclared dispatch inputs")

        # Candidate kind is parsed from the digest-bound operation request and
        # exported through GITHUB_ENV. It is deliberately not a dispatch input.
        self.assertIn(
            "if: inputs.mode == 'build-candidate' && "
            "env.QWEN_CANDIDATE_KIND == 'intervention'",
            body,
        )
        self.assertIn(
            "if: inputs.mode == 'build-candidate' && env.QWEN_CANDIDATE_KIND == 'stock'",
            body,
        )

    def test_embedded_operation_request_parser_accepts_only_exact_mode_shape(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        scripts = [script for block in workflow_run_blocks(body)
                   for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", block, re.S)]
        parser = next(script for script in scripts
                      if "candidate-construction-request.v1" in script)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            request = root / "request.json"
            value = {"schema": "ember.qwen3.8.candidate-construction-request.v1",
                     "mode": "prepare-cache", "parameters": {},
                     "publishes": False, "deletes": False}
            request.write_text(json.dumps(value) + "\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-", str(request), "prepare-cache"], input=parser,
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(result.stdout.splitlines()), 21)
            value["parameters"]["unexpected"] = None
            request.write_text(json.dumps(value) + "\n", encoding="utf-8")
            malformed = subprocess.run(
                [sys.executable, "-", str(request), "prepare-cache"], input=parser,
                text=True, capture_output=True,
            )
            self.assertNotEqual(malformed.returncode, 0)

    def test_lock_restore_and_health_are_fail_closed(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        acquire = body.index("ember-gpu-lock acquire")
        stop = body.index("ember-cert-production stop")
        construct = body.index("Prepare one immutable content-addressed BF16 cache")
        cleanup = body.index("Restore production, release ownership, and prove health")
        self.assertLess(acquire, stop)
        self.assertLess(stop, construct)
        self.assertLess(construct, cleanup)
        self.assertIn("if: always() && steps.exclusive.outputs.armed == 'yes'", body)
        self.assertIn("ember-cert-production unmask", body)
        self.assertIn("ember-cert-production start", body)
        self.assertIn("ember-gpu-lock release", body)
        self.assertGreaterEqual(body.count("http://127.0.0.1:8000/health"), 2)
        self.assertIn('exit "$cleanup_failed"', body)

    def test_handoff_is_durable_but_does_not_claim_v3_readiness(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("ember.qwen3.8.candidate-construction.v1", body)
        self.assertIn('"v3_candidate_manifest":{"ready":False', body)
        for blocker in (
            "audited_quality_contract",
            "phase_accumulator_state",
            "normalization_request",
        ):
            self.assertIn(blocker, body)
        self.assertIn('"builder_attestation":item(attestation)', body)
        self.assertIn('"runtime_revision":revision', body)
        self.assertIn('"stock_capture":item(capture_path) if kind=="stock" else None', body)
        self.assertIn('"shared_companions":{"Q4_0_ROCMI4":item(rocmi4_path)', body)
        self.assertIn('"publishes":False,"deletes":False', body)
        for forbidden in (
            "docker push",
            "hf upload",
            "huggingface-cli",
            "actions/upload-artifact",
            "delete-loser",
            "retire-captured-stock",
            "shutil.rmtree",
            "Path.unlink",
        ):
            self.assertNotIn(forbidden, body)


if __name__ == "__main__":
    unittest.main()
