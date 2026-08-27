#!/usr/bin/env python3
"""GPU-free safety and methodology tests for the Qwen real-weight gate."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts/qwen_real_weight_gate.sh"
PROFILE = ROOT / "scripts/profile_gpu.sh"
HEX = "1" * 64


def dry_args(out: str = "/tmp/qwen-real-gate-never-created") -> list[str]:
    return [
        "--dry-run", "--image", "candidate:exact",
        "--image-digest", f"sha256:{HEX}",
        "--profile-image", "candidate-dev:exact",
        "--profile-image-digest", f"sha256:{HEX}",
        "--model", "/models/qwen.gguf", "--model-sha256", HEX,
        "--mtp", "/models/qwen-mtp.gguf", "--mtp-sha256", HEX,
        "--out-dir", out,
    ]


def sabotaged_path(directory: str) -> dict[str, str]:
    for tool in ("docker", "curl", "sudo", "dd", "python3"):
        path = Path(directory) / tool
        path.write_text(
            f"#!/bin/sh\necho FORBIDDEN:{tool} >&2\nexit 97\n",
            encoding="utf-8")
        path.chmod(0o755)
    return os.environ | {"PATH": directory + os.pathsep + os.environ["PATH"]}


class QwenRealWeightGateTest(unittest.TestCase):
    def run_gate(self, args: list[str], *, env=None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(GATE), *args], cwd=ROOT, env=env,
            text=True, capture_output=True)

    def test_shell_contracts_are_syntactically_valid(self) -> None:
        for path in (GATE, PROFILE):
            subprocess.run(["bash", "-n", str(path)], check=True,
                           capture_output=True)

    def test_dry_run_touches_nothing_and_records_exact_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = str(Path(temporary) / "evidence")
            result = self.run_gate(dry_args(out), env=sabotaged_path(temporary))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(Path(out).exists())
        self.assertIn("candidate:exact", result.stdout)
        self.assertIn("candidate-dev:exact", result.stdout)
        self.assertIn(f"sha256:{HEX}", result.stdout)
        self.assertIn("/models/qwen.gguf", result.stdout)
        self.assertIn("/models/qwen-mtp.gguf", result.stdout)
        self.assertIn("no files, GPU, docker, sudo", result.stdout)
        self.assertNotIn("FORBIDDEN", result.stderr)

    def test_missing_or_unsafe_identity_is_rejected(self) -> None:
        cases = [
            dry_args()[:-2],
            [value if value != f"sha256:{HEX}" else "sha256:bad"
             for value in dry_args()],
            [value if value != "/models/qwen.gguf" else "relative.gguf"
             for value in dry_args()],
            [*dry_args(), "--mtp-depth", "5"],
            [*dry_args(), "--port", "80"],
        ]
        for args in cases:
            with self.subTest(args=args):
                result = self.run_gate(args)
                self.assertNotEqual(result.returncode, 0)

    def test_exclusive_restore_and_success_only_publication(self) -> None:
        body = GATE.read_text(encoding="utf-8")
        self.assertIn("trap cleanup EXIT INT TERM", body)
        self.assertIn('sudo -n "$GPU_LOCK" acquire', body)
        self.assertIn('sudo -n "$GPU_LOCK" release', body)
        self.assertIn('sudo -n "$PRODUCTION" stop', body)
        self.assertIn('sudo -n "$PRODUCTION" mask', body)
        self.assertIn('sudo -n "$PRODUCTION" unmask', body)
        self.assertIn('sudo -n "$PRODUCTION" start', body)
        restore = body.index("restore_exclusive || die")
        approval = body.index(".publish-approved.")
        self.assertLess(restore, approval)
        self.assertNotIn("huggingface-cli", body)
        self.assertNotIn("hf upload", body)

    def test_gate_uses_exact_shapes_and_separate_profile_pass(self) -> None:
        body = GATE.read_text(encoding="utf-8")
        self.assertIn("--protocol hard-gate", body)
        self.assertIn("--prefill-target 412.0", body)
        self.assertIn("--decode-target 39.49", body)
        self.assertIn("--require-gate", body)
        self.assertIn('row.get("spec_ran") is True', body)
        self.assertIn("0.0 <= rate < 1.0", body)
        timing = body.index('python3 "$BENCHMARK"')
        profile = body.index('"$PROFILE_SCRIPT" --no-quiesce')
        self.assertLess(timing, profile)
        self.assertIn("never timing evidence", body)
        self.assertIn('--image "$PROFILE_IMAGE"', body)

    def test_candidate_and_profiler_images_are_exactly_bound(self) -> None:
        body = GATE.read_text(encoding="utf-8")
        self.assertIn("--profile-image-digest", body)
        self.assertIn("profile-image-inspect.json", body)
        self.assertIn("org.opencontainers.image.revision", body)
        self.assertIn("EMBER_CONFIGURED_GIT_HEAD:STRING", body)
        self.assertIn('candidate_binary_sha="$(docker run', body)
        self.assertIn('profile_binary_sha="$(docker run', body)
        self.assertIn('"$candidate_binary_sha" == "$profile_binary_sha"', body)
        self.assertIn('"profile_image": {"ref": profile_image', body)

    def test_integrity_is_direct_and_profile_supports_qwen_mtp(self) -> None:
        gate = GATE.read_text(encoding="utf-8")
        profile = PROFILE.read_text(encoding="utf-8")
        self.assertIn('"iflag=direct"', gate)
        self.assertIn("DFLASH_QWEN_MTP=/gate/mtp.gguf", gate)
        self.assertIn("--mtp PATH", profile)
        self.assertIn("DFLASH_QWEN_MTP=/pmtp/", profile)
        self.assertIn("--draft and --mtp are mutually exclusive", profile)
        self.assertIn('--pmc "$counter"', profile)

    def test_profile_mtp_dry_run_is_gpu_free_and_rejects_two_drafters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = sabotaged_path(temporary)
            result = subprocess.run(
                ["bash", str(PROFILE), "--dry-run",
                 "--model", "/models/qwen.gguf",
                 "--mtp", "/models/qwen-mtp.gguf", "--mtp-depth", "4"],
                cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("FORBIDDEN", result.stderr)
            self.assertIn("mtp             /models/qwen-mtp.gguf", result.stdout)
            self.assertIn("mtp depth       4", result.stdout)

        result = subprocess.run(
            ["bash", str(PROFILE), "--dry-run",
             "--model", "/models/qwen.gguf",
             "--draft", "/models/dspark.gguf",
             "--mtp", "/models/qwen-mtp.gguf"],
            cwd=ROOT, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutually exclusive", result.stderr)


if __name__ == "__main__":
    unittest.main()
