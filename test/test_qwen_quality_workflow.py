#!/usr/bin/env python3
"""Adversarial static contract for staged gfx1151 quality capture."""

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/qwen-quality-capture.yml"
PLAN_WORKFLOW = ROOT / ".github/workflows/qwen-quality-plan.yml"
DISPATCH_WORKFLOW = ROOT / ".github/workflows/gfx1151-certify.yml"
JUDGE_SOURCE = ROOT / "share/release_profiles/deepseek-v4-flash-strix-halo-quality-judge.json"


def shell_blocks(text: str) -> list[str]:
    """Extract literal workflow run blocks without needing a YAML dependency."""
    lines = text.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index] == "        run: |":
            index += 1
            body: list[str] = []
            while index < len(lines) and (not lines[index] or lines[index].startswith("          ")):
                line = lines[index]
                body.append(line[10:] if line else "")
                index += 1
            blocks.append("\n".join(body) + "\n")
            continue
        index += 1
    return blocks


class QwenQualityWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.body = WORKFLOW.read_text(encoding="utf-8")

    def test_every_shell_step_parses(self) -> None:
        blocks = shell_blocks(self.body)
        self.assertGreaterEqual(len(blocks), 6)
        with tempfile.TemporaryDirectory() as temporary:
            for index, block in enumerate(blocks):
                # GitHub expressions are replaced before the runner invokes bash.
                parsed = re.sub(r"\$\{\{.*?\}\}", "github-expression", block)
                path = Path(temporary) / f"step-{index}.sh"
                path.write_text(parsed, encoding="utf-8")
                result = subprocess.run(["bash", "-n", str(path)], text=True,
                                        capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_inline_python_program_compiles(self) -> None:
        programs: list[str] = []
        for block in shell_blocks(self.body):
            programs.extend(re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", block, re.DOTALL))
        self.assertGreaterEqual(len(programs), 4)
        for index, program in enumerate(programs):
            compile(program, f"workflow-heredoc-{index}.py", "exec")

    def test_manual_serial_runner_and_permissions(self) -> None:
        body = self.body
        self.assertIn("workflow_dispatch:", body)
        self.assertIn("workflow_call:", body)
        called = re.search(
            r"workflow_call:\n    inputs:\n(.*?)\n  workflow_dispatch:", body, re.S)
        self.assertIsNotNone(called)
        self.assertEqual(re.findall(r"^      ([a-z0-9_]+):$", called.group(1), re.M),
                         ["commit_sha", "phase_descriptor",
                          "phase_descriptor_sha256"])
        self.assertIn("runs-on: [self-hosted, linux, x64, gfx1151]", body)
        self.assertIn(
            "contains(github.workflow_ref, '/.github/workflows/gfx1151-certify.yml@')",
            body)
        self.assertIn("qwen-quality-called-{0}", body)
        self.assertIn("|| 'gfx1151-certification'", body)
        self.assertIn("cancel-in-progress: false", body)
        for permission in ("id-token: write", "attestations: write",
                           "artifact-metadata: write", "packages: read"):
            self.assertIn(permission, body)
        self.assertIn('[[ "${GITHUB_REPOSITORY,,}" = otheru-ai/ember ]]', body)
        self.assertNotIn('[[ "$GITHUB_REPOSITORY" = OtherU-AI/ember ]]', body)
        self.assertIn('test "$(git rev-parse HEAD)" = "$TARGET_SHA"', body)
        self.assertIn("git diff --quiet --exit-code", body)
        self.assertIn("qwen-quality-binding-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT", body)
        self.assertIn('chmod 400 "$binding_dir"/*.json', body)
        self.assertIn('test "${#binding[@]}" -eq 9', body)

    def test_phase_capability_boundary_is_fail_closed(self) -> None:
        body = self.body
        self.assertIn("ember.qwen3.8.quality-phase-descriptor.v1", body)
        self.assertIn('phase not in {"sweep", "final"}', body)
        self.assertIn("phase descriptor keys differ", body)
        self.assertIn("path must be absolute and single-line", body)
        self.assertIn("output_dir must be a new absolute runner path", body)
        self.assertIn('unlocked.get("phase_scope") != "selection"', body)
        self.assertIn('"final_corpus_manifest" in unlocked', body)
        self.assertIn("final-heldout quality is inaccessible to the sweep workflow", body)
        self.assertIn("sweep capture plan references final-heldout capability", body)
        self.assertIn('unlocked.get("phase_scope") != "final_confirmation"', body)
        self.assertIn("final_heldout_unlocked_after_mtp_depth_selection", body)
        self.assertIn("capture corpus differs from the already-unlocked phase corpus", body)
        self.assertIn("scripts/qwen_bakeoff.py --plan \"$UNLOCKED_PLAN\" --stage verify", body)
        self.assertIn("canonical_plan_verified", body)
        self.assertNotIn("inputs.final", body)
        self.assertNotIn("/models/", body)

    def test_immutable_identity_and_no_credential_capture(self) -> None:
        body = self.body
        self.assertIn("model descriptor keys differ", body)
        self.assertIn("stock and candidate inventories must differ", body)
        self.assertIn("judge settings are not the deterministic target-only contract", body)
        self.assertIn("judge rubric is not the pinned checked rubric", body)
        self.assertIn("agentic cases are not the pinned 15-case corpus", body)
        self.assertIn("image must be an immutable digest reference", body)
        self.assertIn("org.opencontainers.image.revision", body)
        self.assertIn("stock and candidate must use the same runtime image and launch policy", body)
        self.assertIn("launch arguments override identity or may expose credentials", body)
        self.assertIn('"--env", "DFLASH_DS4_SPEC=0"', body)
        self.assertIn('"--env", "DFLASH_DSPARK_XDNA_PLUGIN="', body)
        self.assertNotIn("Config.Env", body)
        self.assertNotIn("secrets.GITHUB_TOKEN", body)
        self.assertIn("set +x", body)

    def test_residency_is_strictly_serial_and_verified(self) -> None:
        body = self.body
        points = [
            body.index("start_runtime stock"),
            body.index('responses --variant stock'),
            body.index('agentic --variant stock'),
            body.index("start_runtime candidate"),
            body.index('responses --variant candidate'),
            body.index('agentic --variant candidate'),
            body.index("start_runtime judge"),
            body.index('output-dir "$QUALITY_OUTPUT" judge'),
            body.index('output-dir "$QUALITY_OUTPUT" assemble'),
            body.index("scripts/qwen_quality_judge.py --contract"),
        ]
        self.assertEqual(points, sorted(points))
        segment = body[points[0]:points[-1]]
        self.assertGreaterEqual(segment.count("remove_current"), 3)
        self.assertIn("available_kib >= 100 * 1024 * 1024", body)
        self.assertIn("less than 100 GiB is available before the $role residency", body)
        self.assertIn('(( judge_status == 0 || judge_status == 1 ))', body)
        self.assertIn("judge_status == 0 || judge_status == 1", body)
        self.assertIn("runtime/qwen_quality_judge.py", body)

    def test_gpu_exclusivity_restores_health_and_accumulates_failure(self) -> None:
        body = self.body
        lock = body.index("ember-gpu-lock acquire")
        stop = body.index("ember-cert-production stop")
        mask = body.index("ember-cert-production mask")
        capture = body.index("start_runtime stock")
        restore = body.index("restore_failed=0")
        release = body.index("ember-gpu-lock release")
        self.assertLess(lock, stop)
        self.assertLess(stop, mask)
        self.assertLess(mask, capture)
        self.assertLess(capture, restore)
        self.assertLess(restore, release)
        self.assertIn("always() && steps.exclusive.outputs.lock_acquired == 'yes'", body)
        self.assertIn("ember-cert-production unmask || restore_failed=1", body)
        self.assertIn("ember-cert-production start || restore_failed=1", body)
        self.assertIn("ember-cert-production is-active", body)
        self.assertIn("http://127.0.0.1:8000/health", body)
        self.assertIn("production_healthy=0", body)
        self.assertIn("one or more capture containers could not be removed", body)
        self.assertIn('exit "$restore_failed"', body)

    def test_exact_attestation_is_retained_and_verified_before_handoff(self) -> None:
        body = self.body
        action = "actions/attest@1e69f48acb82d1966a394da916b4c1698aa569d6 # v4.2.2"
        self.assertIn(action, body)
        self.assertIn("subject-path: ${{ env.QUALITY_CONTRACT }}", body)
        self.assertNotIn("subject-path: |", body)
        attest = body.index(action)
        verify = body.index('gh attestation verify "$QUALITY_CONTRACT"')
        handoff = body.index("ember.qwen3.8.quality-contract-handoff.v1")
        upload = body.index("actions/upload-artifact@")
        self.assertLess(attest, verify)
        self.assertLess(verify, handoff)
        self.assertLess(handoff, upload)
        self.assertIn("steps.attest.outputs.bundle-path", body)
        self.assertIn("quality-contract.sigstore.json", body)
        self.assertIn("--signer-workflow .github/workflows/qwen-quality-capture.yml", body)
        self.assertIn('"external_attestation_verified": True', body)
        self.assertIn('"ready_for_bakeoff_adjudication": True', body)
        self.assertIn("if-no-files-found: error", body)

    def test_scope_is_text_only_and_workflow_cannot_publish(self) -> None:
        body = self.body
        self.assertGreaterEqual(body.count('"modality": "text_only"'), 2)
        self.assertGreaterEqual(body.count('"multimodal_release_claim": False'), 2)
        self.assertGreaterEqual(body.count('"vision_mmproj_differential_pass": False'), 2)
        self.assertIn('"publication_allowed": False', body)
        for forbidden in ("docker push", "huggingface-cli", "hf upload", "gh release"):
            self.assertNotIn(forbidden, body)

    def test_runner_local_planner_is_create_only_and_chains_exact_outputs(self) -> None:
        body = PLAN_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_call:", body)
        self.assertIn("runs-on: [self-hosted, linux, x64, gfx1151]", body)
        self.assertIn("qwen-quality-plan-${{ github.run_id }}", body)
        self.assertIn("O_EXCL", body)
        self.assertIn("decoded quality request must contain 1..32768 bytes", body)
        self.assertIn("scripts/qwen_quality_request.py", body)
        self.assertIn("phase_descriptor_sha256", body)
        self.assertNotIn("ember-gpu-lock acquire", body)
        for forbidden in ("docker push", "huggingface-cli", "hf upload", "gh release"):
            self.assertNotIn(forbidden, body)
        for block in shell_blocks(body):
            parsed = re.sub(r"\$\{\{.*?\}\}", "github-expression", block)
            result = subprocess.run(["bash", "-n"], input=parsed, text=True,
                                    capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for program in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", parsed, re.S):
                compile(program, "quality-plan-heredoc.py", "exec")

    def test_judge_staging_is_exact_create_only_and_keeps_production_online(self) -> None:
        body = DISPATCH_WORKFLOW.read_text(encoding="utf-8")
        start = body.index("  qwen-stage-quality-judge:")
        end = body.index("\n  qwen-convert-control:", start)
        stage = body[start:end]
        self.assertIn(
            "if: inputs.release_version == 'qwen-stage-quality-judge-v1-20260828'",
            stage)
        self.assertNotIn("startsWith(inputs.release_version", stage)
        self.assertIn("75a4bed8e6762986d7b4169e1e1afbb57c482704", stage)
        self.assertIn("a936e0a514385c8ae964c0f42263a4314a34fbc6efea9d9aced5320f320a3d54",
                      stage)
        self.assertIn("JUDGE_BYTES: '91547243200'", stage)
        self.assertIn("--workers 1", stage)
        self.assertIn("ionice -c 3 nice -n 19", stage)
        self.assertIn("iflag=direct", stage)
        self.assertIn("os.O_EXCL", stage)
        self.assertIn("ember.qwen3.8.quality-judge-inventory.v1", stage)
        self.assertIn('[[ "${GITHUB_REPOSITORY,,}" = otheru-ai/ember ]]', stage)
        self.assertGreaterEqual(stage.count("ember-cert-production is-active"), 2)
        self.assertGreaterEqual(stage.count("http://127.0.0.1:8000/health"), 2)
        for forbidden in ("ember-gpu-lock acquire", "ember-cert-production stop",
                          "docker push", "huggingface-cli", "hf upload", "gh release",
                          "/models/"):
            self.assertNotIn(forbidden, stage)
        for block in shell_blocks(stage):
            parsed = re.sub(r"\$\{\{.*?\}\}", "github-expression", block)
            result = subprocess.run(["bash", "-n"], input=parsed, text=True,
                                    capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for program in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", parsed, re.S):
                compile(program, "judge-stage-heredoc.py", "exec")

    def test_judge_source_inventory_is_one_exact_independent_artifact(self) -> None:
        import json

        value = json.loads(JUDGE_SOURCE.read_text(encoding="utf-8"))
        self.assertEqual(value, {
            "schema": "ember.qwen3.8.quality-judge-source.v1",
            "artifact_role": "independent_quality_judge",
            "repo_id": "otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF",
            "revision": "75a4bed8e6762986d7b4169e1e1afbb57c482704",
            "file_count": 1,
            "total_bytes": 91547243200,
            "files": [{
                "path": "DeepSeek-V4-Flash-0731-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf",
                "size": 91547243200,
                "sha256": "a936e0a514385c8ae964c0f42263a4314a34fbc6efea9d9aced5320f320a3d54",
                "git_blob": "493783394e65e7be586050e3860af475ae1e87f8",
            }],
        })


if __name__ == "__main__":
    unittest.main()
