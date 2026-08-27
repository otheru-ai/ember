#!/usr/bin/env python3
"""Offline safety and contract tests for stock Qwen activation capture."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qwen_capture_control.py"
WORKFLOW = ROOT / ".github/workflows/gfx1151-certify.yml"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("qwen_capture_control", SCRIPT)
assert SPEC and SPEC.loader
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)
HEX = "1" * 64


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dry_args(output: str = "/tmp/qwen-capture-control-never-created") -> list[str]:
    return [
        "--dry-run", "--tool-revision", "0" * 40,
        "--artifact-revision", "2" * 40,
        "--image", "ember:exact",
        "--image-digest", f"sha256:{HEX}",
        "--model", "/models/control-00001-of-00002.gguf", "--model-sha256", HEX,
        "--control-record", "/models/qwen-quant-build-record.json",
        "--control-record-sha256", HEX,
        "--mtp", "/models/mtp.gguf", "--mtp-sha256", HEX,
        "--corpus-dir", "/corpora", "--corpus-contract-sha256", HEX,
        "--good-corpus-sha256", HEX, "--bad-corpus-sha256", HEX,
        "--recipe-sha256", HEX, "--output-dir", output,
    ]


def sabotaged_path(directory: str) -> dict[str, str]:
    for tool in ("docker", "sudo", "dd"):
        path = Path(directory) / tool
        path.write_text(f"#!/bin/sh\necho FORBIDDEN:{tool} >&2\nexit 97\n", encoding="utf-8")
        path.chmod(0o755)
    return os.environ | {"PATH": directory + os.pathsep + os.environ["PATH"]}


def write_rows(path: Path, prefix: str) -> str:
    with path.open("x", encoding="utf-8") as stream:
        for index in range(32):
            row = {"id": f"{prefix}-{index:02d}",
                   "messages": [{"role": "user", "content": f"{prefix} prompt {index}"}]}
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return file_sha(path)


class QwenCaptureControlTest(unittest.TestCase):
    def test_script_compiles(self) -> None:
        subprocess.run([sys.executable, "-m", "py_compile", str(SCRIPT)], check=True)

    def test_workflow_reports_bounded_retained_failure_log(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("Report retained activation-container failure log", body)
        self.assertIn('log="$QWEN_CAPTURE_OUTPUT/activation-container.log"', body)
        self.assertIn('[[ -f "$log" && ! -L "$log" ]]', body)
        self.assertIn('tail -c 1048576 -- "$log"', body)
        self.assertIn('dev-sha-${CAPTURE_TOOL_SHA:0:12}', body)
        self.assertIn('--artifact-revision "$TARGET_SHA"', body)
        self.assertIn('manifest["model"]["quantizer_ember_revision"]', body)
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("image_revision != args.tool_revision", script)
        self.assertIn("record_revision != args.artifact_revision", script)

    def test_gpu_groups_are_numeric_device_gids_not_image_names(self) -> None:
        body = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("*device_group_args()", body)
        self.assertIn('result.extend(("--group-add", str(gid)))', body)
        self.assertNotIn('"--group-add", "render"', body)
        self.assertNotIn('"--group-add", "video"', body)

    def test_dry_run_touches_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = str(Path(temporary) / "capture")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), *dry_args(output)], cwd=ROOT,
                env=sabotaged_path(temporary), text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(Path(output).exists())
            self.assertNotIn("FORBIDDEN", result.stderr)
            self.assertIn("exact pinned OtherU 32 good + 32 bad", result.stdout)
            self.assertIn("verified, not loaded", result.stdout)
            self.assertIn("selection/confirmation, never direction inputs", result.stdout)

    def test_invalid_identity_or_path_fails_before_side_effects(self) -> None:
        cases = [
            ["bad" if item == f"sha256:{HEX}" else item for item in dry_args()],
            ["bad" if item == "2" * 40 else item for item in dry_args()],
            ["relative.gguf" if item == "/models/mtp.gguf" else item for item in dry_args()],
            [*dry_args(), "--port", "80"],
        ]
        for args in cases:
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), *args], cwd=ROOT,
                    text=True, capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)

    def test_pinned_contract_accepts_only_exact_32_plus_32_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            good_sha = write_rows(directory / "extraction-good.jsonl", "good")
            bad_sha = write_rows(directory / "extraction-bad.jsonl", "bad")
            contract = {
                "schema_version": 1,
                "source": {
                    "repository": "https://git.otheru.ai/akadmin/otheru-quant-pipeline",
                    "revision": "a3c6a728510f91394e991504951ac316cd3a89af",
                },
                "derived_artifacts": {
                    "extraction-good.jsonl": {"record_count": 32, "sha256": good_sha},
                    "extraction-bad.jsonl": {"record_count": 32, "sha256": bad_sha},
                    "sweep-validation.jsonl": {"record_count": 134, "sha256": "2" * 64},
                    "final-heldout.jsonl": {"record_count": 134, "sha256": "3" * 64},
                },
                "pairwise_request_overlap_count": 0,
            }
            contract_path = directory / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            good, bad, parsed, artifacts = capture.validate_corpora(
                contract_path, file_sha(contract_path), directory, good_sha, bad_sha,
            )
            self.assertEqual((good.name, bad.name),
                             ("extraction-good.jsonl", "extraction-bad.jsonl"))
            self.assertEqual(parsed["source"]["revision"],
                             "a3c6a728510f91394e991504951ac316cd3a89af")
            self.assertEqual(artifacts["sweep-validation.jsonl"]["record_count"], 134)
            self.assertFalse((directory / "sweep-validation.jsonl").exists())
            self.assertFalse((directory / "final-heldout.jsonl").exists())

            with self.assertRaisesRegex(capture.CaptureError, "digest"):
                capture.validate_corpora(
                    contract_path, file_sha(contract_path), directory,
                    "0" * 64, bad_sha,
                )

    def test_policy_grid_is_exact_and_measurement_independent(self) -> None:
        recipe = json.loads(capture.DEFAULT_RECIPE.read_text(encoding="utf-8"))
        grid = capture.intervention_grid(recipe)
        self.assertEqual(len(grid), 16)
        self.assertEqual(len({identifier for identifier, *_ in grid}), 16)
        counts = {policy: sum(value != 0.0 for value in scales)
                  for _identifier, _scale, policy, scales in grid}
        self.assertEqual(counts,
                         {"all-48": 48, "upper-24": 24,
                          "upper-12": 12, "non-qsa": 36})
        self.assertNotIn("results", capture.intervention_grid.__code__.co_names)

    def test_dump_split_fails_closed_on_missing_row(self) -> None:
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(capture, "RECORD_BYTES", 4):
            directory = Path(raw)
            combined = directory / "combined.f32"
            combined.write_bytes(bytes(64 * 4 - 1))
            with self.assertRaisesRegex(capture.CaptureError, "expected"):
                capture.split_dump(combined, directory / "good.f32", directory / "bad.f32")
            self.assertFalse((directory / "good.f32").exists())

            combined.write_bytes(bytes(range(64 * 4)))
            capture.split_dump(combined, directory / "good.f32", directory / "bad.f32")
            self.assertEqual((directory / "good.f32").stat().st_size, 32 * 4)
            self.assertEqual((directory / "bad.f32").stat().st_size, 32 * 4)

    def test_stock_record_rejects_intervened_or_non_rocmi4_model(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            model = directory / "control.gguf"
            model.write_bytes(b"stock-control")
            digest = file_sha(model)
            record = {
                "status": "complete", "mode": "execute", "compute_mode": "exact_dequant",
                "w4a4_enabled": False, "intervention": None,
                "experiment": {"kind": "stock_control", "stock_weights_unchanged": True,
                               "final_release_eligible": False},
                "tools": {"quantizer_build_info": {
                    "format": "Q4_0_ROCMI4", "ggml_tensor_type": 108,
                    "ember_revision": "a" * 40,
                }},
                "output": {"shards": [{"path": "/qwen-work/artifacts/control.gguf",
                                         "size_bytes": model.stat().st_size,
                                         "sha256": digest}]},
            }
            record_path = directory / "qwen-quant-build-record.json"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            parsed, shards = capture.validate_stock_control(
                record_path, file_sha(record_path), model, digest,
            )
            self.assertEqual(parsed["experiment"]["kind"], "stock_control")
            self.assertEqual(shards[0]["sha256"], digest)
            self.assertEqual(shards[0]["path"], model)
            self.assertEqual(shards[0]["recorded_path"],
                             Path("/qwen-work/artifacts/control.gguf"))
            record["intervention"] = {"kind": "directional_ablation"}
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(capture.CaptureError, "stock ROCMI4"):
                capture.validate_stock_control(
                    record_path, file_sha(record_path), model, digest,
                )

    def test_restore_attempts_unmask_start_and_release(self) -> None:
        exclusive = capture.ExclusiveGPU()
        exclusive.masked = True
        exclusive.restore_service = True
        exclusive.locked = True
        calls: list[tuple[str, str]] = []

        def fake_sudo(wrapper: Path, action: str, *, check: bool = True):
            del check
            calls.append((wrapper.name, action))
            return subprocess.CompletedProcess([], 0)

        with mock.patch.object(capture.ExclusiveGPU, "sudo", side_effect=fake_sudo):
            exclusive.restore()
        self.assertEqual(calls, [
            ("ember-cert-production", "unmask"),
            ("ember-cert-production", "start"),
            ("ember-gpu-lock", "release"),
        ])
        self.assertFalse(exclusive.masked or exclusive.restore_service or exclusive.locked)

    def test_container_logs_are_create_only_and_byte_preserving(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            observed: list[list[str]] = []

            def fake_run(command, **kwargs):
                observed.append(list(command))
                self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
                self.assertEqual(kwargs["timeout"], 30)
                os.write(kwargs["stdout"], b"model load failed\nraw:\xff\n")
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(capture.subprocess, "run", side_effect=fake_run):
                output = capture.retain_container_logs("capture-owned", directory)
            self.assertEqual(observed, [[
                "docker", "logs", "--timestamps", "capture-owned",
            ]])
            self.assertEqual(output, directory / capture.CONTAINER_LOG_NAME)
            self.assertEqual(output.read_bytes(), b"model load failed\nraw:\xff\n")
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_container_log_retention_refuses_existing_file_or_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            output = directory / capture.CONTAINER_LOG_NAME
            output.write_bytes(b"existing evidence")
            with mock.patch.object(capture.subprocess, "run") as run:
                with self.assertRaisesRegex(capture.CaptureError, "refusing to overwrite"):
                    capture.retain_container_logs("capture-owned", directory)
                run.assert_not_called()
            self.assertEqual(output.read_bytes(), b"existing evidence")

            output.unlink()
            victim = directory / "victim"
            victim.write_bytes(b"must not change")
            output.symlink_to(victim)
            with mock.patch.object(capture.subprocess, "run") as run:
                with self.assertRaisesRegex(capture.CaptureError, "refusing to overwrite"):
                    capture.retain_container_logs("capture-owned", directory)
                run.assert_not_called()
            self.assertEqual(victim.read_bytes(), b"must not change")

            linked_directory = directory.parent / f"{directory.name}-link"
            linked_directory.symlink_to(directory, target_is_directory=True)
            try:
                with mock.patch.object(capture.subprocess, "run") as run:
                    with self.assertRaisesRegex(capture.CaptureError,
                                                "cannot open capture output directory"):
                        capture.retain_container_logs("capture-owned", linked_directory)
                    run.assert_not_called()
            finally:
                linked_directory.unlink()

    def test_failed_docker_logs_still_retains_diagnostic_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)

            def failed_logs(command, **kwargs):
                os.write(kwargs["stdout"], b"daemon log retrieval failed\n")
                return subprocess.CompletedProcess(command, 17)

            with mock.patch.object(capture.subprocess, "run", side_effect=failed_logs):
                with self.assertRaisesRegex(capture.CaptureError,
                                            "diagnostic output was retained"):
                    capture.retain_container_logs("capture-owned", directory)
            self.assertEqual(
                (directory / capture.CONTAINER_LOG_NAME).read_bytes(),
                b"daemon log retrieval failed\n",
            )

    def test_failure_cleanup_retains_logs_before_container_removal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events: list[str] = []
            original = capture.CaptureError("container exited during model load")

            def retain(_name: str, _output: Path) -> Path:
                events.append("retain")
                return Path(raw) / capture.CONTAINER_LOG_NAME

            def remove(_name: str) -> None:
                events.append("remove")

            with mock.patch.object(capture, "retain_container_logs", side_effect=retain), \
                    mock.patch.object(capture, "remove_container", side_effect=remove):
                result = capture.cleanup_capture_container(
                    "capture-owned", Path(raw), True, original,
                )
            self.assertIs(result, original)
            self.assertEqual(events, ["retain", "remove"])

            events.clear()
            with mock.patch.object(
                    capture, "retain_container_logs",
                    side_effect=capture.CaptureError("unsafe destination")), \
                    mock.patch.object(capture, "remove_container", side_effect=remove):
                result = capture.cleanup_capture_container(
                    "capture-owned", Path(raw), True, original,
                )
            self.assertIsInstance(result, capture.CaptureError)
            self.assertIn("container exited during model load", str(result))
            self.assertIn("failed to retain container logs", str(result))
            self.assertEqual(events, ["remove"])

    def test_durable_recovery_markers_track_only_owned_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            exclusive = capture.ExclusiveGPU(directory)

            def fake_sudo(_wrapper: Path, _action: str, *, check: bool = True):
                del check
                return subprocess.CompletedProcess([], 0)

            with mock.patch.object(capture.ExclusiveGPU, "sudo", side_effect=fake_sudo):
                exclusive.acquire()
                self.assertTrue((directory / ".gpu-lock-held").is_file())
                self.assertTrue((directory / ".production-was-active").is_file())
                self.assertTrue((directory / ".production-masked").is_file())
                exclusive.restore()
            self.assertFalse(any(directory.iterdir()))

    def test_source_keeps_mtp_off_and_sweep_final_unopened(self) -> None:
        body = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"-e", "DFLASH_QWEN_MTP="', body)
        self.assertIn('"-e", "DFLASH_QWEN_MTP_DEPTH="', body)
        self.assertIn("actual != expected", body)
        self.assertIn("exclusive.restore()", body)
        self.assertIn("selection_only_not_read_during_direction_generation", body)
        self.assertIn("single_winner_confirmation_only_not_read_during_direction_generation", body)
        self.assertNotIn('corpus_dir / "sweep-validation.jsonl"', body)
        self.assertNotIn('corpus_dir / "final-heldout.jsonl"', body)
        self.assertNotIn("huggingface", body.lower())


if __name__ == "__main__":
    unittest.main()
