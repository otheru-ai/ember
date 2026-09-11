#!/usr/bin/env python3
"""Run the certify workflow's benchmark wiring without a GPU.

The benchmark job has failed twice on setup alone: once for asserting the model
digests without the provenance field benchmark_bundle.sh requires, and once for
never exporting the vision tower, which would have left the vision block
silently absent. Both are visible before a single byte of model is loaded, but
both were found only after certification had taken the box and quiesced
production for two hours.

So this extracts the environment the workflow's benchmark step actually sets,
points it at fixture files, and runs the real script with --dry-run. It is the
same script and the same variables; only the artifacts are fakes and nothing is
loaded.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
CERTIFY = ROOT / ".github" / "workflows" / "gfx1151-certify.yml"
BUNDLE = ROOT / "scripts" / "benchmark_bundle.sh"
DIGEST = "a" * 64


def _block(lines: list[str], start: int) -> list[str]:
    """Lines of the block nested under lines[start], up to its dedent."""
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = []
    for line in lines[start + 1:]:
        if line.strip() and not line.startswith("#"):
            depth = len(line) - len(line.lstrip())
            if depth <= indent:
                break
        body.append(line)
    return body


def _env_map(lines: list[str]) -> dict[str, str]:
    """`KEY: value` pairs from an `env:` block; quotes stripped, comments dropped."""
    env = {}
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or ":" not in text:
            continue
        key, _, value = text.partition(":")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        env[key.strip()] = value
    return env


def benchmark_step() -> tuple[str, dict[str, str]]:
    """The workflow's benchmark step body and the env visible to it.

    A stdlib-only reader for the two shapes this file uses -- `env:` maps of
    scalars and a `run: |` literal block -- because AGENTS.md promises a
    stdlib-only test suite and this was the one PyYAML import (#21). It
    deliberately understands nothing else: a workflow edit that needs more
    YAML than this should fail here loudly, not be parsed wrong quietly.
    """
    lines = CERTIFY.read_text().splitlines()
    job_start = next(i for i, l in enumerate(lines) if l == "  benchmark:")
    job = _block(lines, job_start)
    env: dict[str, str] = {}
    for i, line in enumerate(job):
        if line == "    env:":
            env.update(_env_map(_block(job, i)))
        if line.strip() == "- name: Run the performance bundle":
            step = _block(job, i)
            run = None
            for k, sline in enumerate(step):
                if sline.strip() == "env:":
                    env.update(_env_map(_block(step, k)))
                if sline.strip() == "run: |":
                    body = _block(step, k)
                    dedent = min(len(b) - len(b.lstrip()) for b in body if b.strip())
                    run = "\n".join(b[dedent:] for b in body).rstrip("\n") + "\n"
            if run is None:
                raise AssertionError("benchmark step has no `run: |` block")
            return run, env
    raise AssertionError("benchmark step not found")


class BenchmarkDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = pathlib.Path(directory.name)
        self.docker_called = root / "docker-called"
        for name, body in {
            "getent": "exit 2",
            "docker": f'touch "{self.docker_called}"; exit 99',
        }.items():
            command = root / name
            command.write_text("#!/bin/sh\n" + body + "\n")
            command.chmod(0o755)
        env = patch.dict(os.environ, {"PATH": f"{root}:{os.environ['PATH']}"})
        env.start()
        self.addCleanup(env.stop)

    def tearDown(self) -> None:
        self.assertFalse(self.docker_called.exists(), "CPU validation invoked Docker")

    def test_real_benchmark_still_requires_gpu_groups(self) -> None:
        result = subprocess.run(
            [str(BUNDLE), "--out", str(self.docker_called.parent / "out"),
             "--release", "9999.1.1", "--no-exclusive"],
            text=True, capture_output=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("video/render host GIDs unavailable", result.stderr)

    def test_step_exports_every_digest_with_its_provenance(self) -> None:
        """benchmark_bundle.sh refuses an asserted digest with no source.

        Hashing 96 GiB is skipped by asserting the digests certification
        already checked, and the script requires the caller to say who asserted
        them so the bundle records where the value came from. Forgetting the
        second half is what failed the job.
        """
        body, _ = benchmark_step()
        for name in ("TARGET_SHA", "DRAFT_SHA", "MMPROJ_SHA"):
            self.assertRegex(
                body, rf"\b{name}=",
                f"the step asserts nothing for {name}")
            self.assertRegex(
                body, rf"\b{name}_ASSERTED_BY=",
                f"{name} is asserted without provenance; the script exits 1")
            self.assertRegex(
                body, rf"export[^\n]*\b{name}_ASSERTED_BY\b",
                f"{name}_ASSERTED_BY is set but never exported")

    def test_step_exports_the_tower(self) -> None:
        """Without MMPROJ the engine refuses images and vision goes missing."""
        body, _ = benchmark_step()
        self.assertRegex(body, r"\bMMPROJ=",
                         "no tower: the vision block would be absent, and "
                         "absent reads the same as never measured")
        self.assertRegex(body, r"export[^\n]*\bMMPROJ\b")

    def test_the_real_script_accepts_the_workflow_environment(self) -> None:
        """End to end: the workflow's own variables through the real script."""
        body, job_env = benchmark_step()

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for name in ("model.gguf", "draft.gguf", "mmproj.gguf"):
                (root / name).write_bytes(b"fixture")

            # Resolve the workflow's env, substituting fixtures for the paths
            # that would otherwise point at 96 GiB on the box.
            env = os.environ | {
                "MODEL_PATH": str(root / "model.gguf"),
                "DRAFT_PATH": str(root / "draft.gguf"),
                "MMPROJ_PATH": str(root / "mmproj.gguf"),
                "RELEASE_VERSION": "9999.1.1",
                "GITHUB_RUN_ID": "0",
                "GITHUB_REPOSITORY": "otheru-ai/ember",
                "RUNNER_TEMP": str(root),
            }
            for key, value in job_env.items():
                if "${{" not in str(value):
                    env[key] = str(value)

            # Take the step's own assignments, minus the literal digests, which
            # would need real 96 GiB artifacts to match.
            script_lines = []
            for line in body.splitlines():
                stripped = line.strip()
                if re.match(r"^(export |[A-Z_]+=)", stripped) and "SHA=" not in stripped:
                    script_lines.append(stripped)
            script = "\n".join(script_lines)

            probe = ROOT / "share" / "bench" / "vision-probe.png"
            self.assertTrue(probe.is_file(), "the pinned probe image is missing")

            result = subprocess.run(
                ["bash", "-c", f"""
set -euo pipefail
{script}
TARGET_SHA={DIGEST}
DRAFT_SHA={DIGEST}
MMPROJ_SHA={DIGEST}
export TARGET_SHA DRAFT_SHA MMPROJ_SHA
exec {BUNDLE} --out "$RUNNER_TEMP/out" --release "$RELEASE_VERSION" \\
  --image example:latest --binary /bin/true --no-repo-mount --dry-run
"""],
                env=env, text=True, capture_output=True, timeout=120,
            )
            self.assertEqual(
                result.returncode, 0,
                f"the workflow's environment does not satisfy the script\n"
                f"--- stderr ---\n{result.stderr}\n--- stdout ---\n{result.stdout}")
            self.assertIn("DRY_RUN_OK", result.stdout)
            self.assertIn("--vision-image", result.stdout,
                          "the tower reached the script but vision was not armed")

    def test_a_missing_provenance_field_still_fails(self) -> None:
        """The guard has to be able to fail, or it proves nothing."""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for name in ("model.gguf", "draft.gguf"):
                (root / name).write_bytes(b"fixture")
            env = os.environ | {
                "MODEL_DIR": str(root),
                "TARGET": str(root / "model.gguf"),
                "DRAFT": str(root / "draft.gguf"),
                "TARGET_SHA": DIGEST,
                "DRAFT_SHA": DIGEST,
                # TARGET_SHA_ASSERTED_BY deliberately absent.
            }
            env.pop("TARGET_SHA_ASSERTED_BY", None)
            result = subprocess.run(
                [str(BUNDLE), "--out", str(root / "out"), "--release", "9999.1.1",
                 "--image", "example:latest", "--binary", "/bin/true",
                 "--no-repo-mount", "--dry-run"],
                env=env, text=True, capture_output=True, timeout=120,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("TARGET_SHA_ASSERTED_BY is required",
                          result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=0)
