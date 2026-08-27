#!/usr/bin/env python3
"""Static and adversarial contracts for durable captured-stock retirement."""

from __future__ import annotations

import argparse
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
WORKFLOW = ROOT / ".github/workflows/qwen-gfx1151-retire-stock.yml"
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_candidate_builder as builder  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


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


class QwenRetireStockWorkflowTest(unittest.TestCase):
    def test_yaml_shell_and_embedded_python_parse(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        ruby = shutil.which("ruby")
        if ruby:
            subprocess.run(
                [ruby, "-e", "require 'yaml'; YAML.parse_file(ARGV[0])", str(WORKFLOW)],
                check=True,
            )
        blocks = workflow_run_blocks(body)
        self.assertEqual(len(blocks), 7)
        for index, block in enumerate(blocks):
            neutral = re.sub(r"\$\{\{.*?\}\}", "github-expression", block)
            result = subprocess.run(
                ["bash", "-n"], input=neutral, text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 0, f"run block {index}: {result.stderr}")
            for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", neutral, re.S):
                compile(script, f"retirement-run-{index}-heredoc.py", "exec")

    def test_dispatch_binds_full_revision_digest_and_exact_persistent_paths(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        match = re.search(r"workflow_dispatch:\n    inputs:\n(.*?)\npermissions:", body, re.S)
        self.assertIsNotNone(match)
        dispatch = match.group(1)
        self.assertEqual(len(re.findall(r"(?m)^      [a-z0-9_]+:$", dispatch)), 10)
        self.assertNotRegex(dispatch, r"(?m)^      stock_dir:$")
        for required_input in (
            "commit_sha", "builder_image", "builder_image_digest", "workset_root",
            "stock_artifact_revision", "stock_build_record_sha256", "capture_manifest",
            "capture_manifest_sha256", "authorization_output", "authorize",
        ):
            self.assertRegex(body, rf"(?m)^      {required_input}:$")
        self.assertIn('[[ "$TARGET_SHA" =~ ^[0-9a-f]{40}$ ]]', body)
        self.assertIn('[[ "$QWEN_STOCK_ARTIFACT_REVISION" =~ ^[0-9a-f]{40}$ ]]', body)
        self.assertIn('[[ "$BUILDER_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]', body)
        self.assertIn('test "$(git rev-parse HEAD)" = "$TARGET_SHA"', body)
        self.assertIn('pinned_image="$BUILDER_IMAGE_REF@$BUILDER_IMAGE_DIGEST"', body)
        self.assertIn("org.opencontainers.image.revision", body)
        self.assertIn("workspace=/var/tmp/ember-qwen3.8-flash-next", body)
        self.assertIn('test "$QWEN_WORKSET_ROOT" = "$expected_workset"', body)
        self.assertIn('QWEN_STOCK_DIR="$expected_stock"', body)
        self.assertIn("builder authorization escapes the fixed retirement evidence directory", body)
        self.assertIn("filename is not one safe stock retirement id", body)
        self.assertIn("capture-manifest.json under workspace/evidence", body)
        self.assertIn("RETIRE_CAPTURED_STOCK_SHARDS", body)
        self.assertIn(
            "contains(github.workflow_ref, '/.github/workflows/gfx1151-certify.yml@')",
            body,
        )

    def test_stock_artifact_revision_is_independent_from_builder_revision(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            'expected_stock="$workspace/artifacts/stock-rocmi4-${QWEN_STOCK_ARTIFACT_REVISION:0:12}"',
            body,
        )
        self.assertNotIn(
            'expected_stock="$workspace/artifacts/stock-rocmi4-${TARGET_SHA:0:12}"',
            body,
        )
        self.assertIn("stock directory does not match the independent artifact revision", body)
        self.assertIn(
            'model.get("quantizer_ember_revision") != stock_artifact_revision', body,
        )
        self.assertIn('runtime_revision != capture_tool_revision', body)
        self.assertIn('"artifact-to-capture-runtime"', body)
        self.assertIn('"capture-runtime-to-retirement"', body)
        self.assertIn('"stock_artifact": {"revision": stock_artifact_revision}', body)
        self.assertIn('"stock_artifact_revision": stock_artifact_revision', body)

    def test_workset_ownership_handoff_is_exact_pinned_and_symlink_safe(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        image_verified = body.index('test "$embedded" = "$TARGET_SHA"')
        ownership_handoff = body.index(
            "The persistent artifacts directory is builder-owned on the runner"
        )
        authorization = body.index(
            'workflow_authorization="$QWEN_RETIRE_AUTHORIZATION.workflow.json"'
        )
        self.assertLess(image_verified, ownership_handoff)
        self.assertLess(ownership_handoff, authorization)
        self.assertNotIn('mkdir -m 700 "$QWEN_WORKSET_ROOT"', body)
        self.assertNotIn("chown -R", body)
        self.assertNotIn("chmod -R", body)
        self.assertNotIn("sudo -n chown", body)
        self.assertIn('-v "$workspace/artifacts:/qwen-artifacts"', body)
        self.assertIn("--cap-drop ALL --cap-add CHOWN --cap-add DAC_OVERRIDE --cap-add FOWNER", body)
        self.assertIn("--security-opt no-new-privileges --pids-limit 16 --user 0:0", body)
        self.assertIn(
            'for component in ("qwen-workset", "evidence", "stock-retirement"):', body,
        )
        self.assertIn("os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW", body)
        self.assertIn("dir_fd=parent, follow_symlinks=False", body)
        self.assertIn("os.fchown(child, uid, gid)", body)
        self.assertIn("os.fchmod(child, 0o700)", body)
        self.assertIn('test "$(stat -c \'%u:%g\' "$path")" = "$uid:$gid"', body)
        self.assertIn('test "$(stat -c \'%a\' "$path")" = 700', body)

    def test_lock_quiesce_and_unconditional_restore_are_fail_closed(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        acquire = body.index("ember-gpu-lock acquire")
        stop = body.index("ember-cert-production stop")
        mask = body.index("ember-cert-production mask")
        retire = body.index("Retire only the activation-captured stock shard inventory")
        cleanup = body.index("Restore production, release gfx1151 ownership, and prove health")
        self.assertLess(acquire, stop)
        self.assertLess(stop, mask)
        self.assertLess(mask, retire)
        self.assertLess(retire, cleanup)
        self.assertIn("if: always() && steps.exclusive.outputs.armed == 'yes'", body)
        self.assertIn("ember-cert-production unmask", body)
        self.assertIn("ember-cert-production start", body)
        self.assertIn("ember-gpu-lock release", body)
        self.assertGreaterEqual(body.count("http://127.0.0.1:8000/health"), 2)
        self.assertIn('exit "$cleanup_failed"', body)

    def test_retirement_is_exact_durable_reconstructive_and_nonpublishing(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(body.count("retire-captured-stock"), 1)
        for argument in (
            "--stock-dir", "--build-record-sha256", "--stock-capture-manifest",
            "--stock-capture-manifest-sha256", "--workset-root", "--output",
        ):
            self.assertIn(argument, body)
        self.assertIn("--network none --read-only", body)
        self.assertIn("--cap-drop ALL", body)
        self.assertIn("ember.qwen3.8.stock-retirement-workflow-authorization.v1", body)
        self.assertIn("ember.qwen3.8.stock-retirement-authorization.v1", body)
        self.assertIn("ember.qwen3.8.stock-retirement-complete.v1", body)
        self.assertIn("ember.qwen3.8.stock-retirement-workflow-complete.v1", body)
        self.assertIn("reconstructive_not_undelete", body)
        self.assertIn("not recoverable in place", body)
        self.assertIn("Capture manifest SHA-256:", body)
        self.assertIn("Retained stock build record SHA-256:", body)
        self.assertIn("Workflow completion SHA-256:", body)
        self.assertGreaterEqual(body.count('"publishes": False'), 2)
        self.assertGreaterEqual(body.count('get("publishes") is not False'), 2)
        self.assertIn("permissions:\n  contents: read\n  packages: read", body)
        for forbidden in (
            "rm -rf", "find -delete", "shutil.rmtree", "Path.unlink",
            "delete-loser", "docker push", "hf upload", "huggingface-cli",
            "actions/upload-artifact",
        ):
            self.assertNotIn(forbidden, body)

    def test_retirement_maps_only_exact_converter_namespace(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            'pathlib.PurePosixPath("/qwen-work/artifacts") / stock.name', body,
        )
        self.assertIn("os.path.normpath(raw) != raw", body)
        self.assertIn("recorded_path.parent != recorded_parent", body)
        self.assertIn("stock / recorded_path.name", body)
        self.assertIn("outside the exact converter namespace", body)

    def test_builder_converter_namespace_mapping_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stock = root / "stock-rocmi4-deadbeefcafe"
            stock.mkdir()
            first = stock / "stock-00001-of-00002.gguf"
            second = stock / "stock-00002-of-00002.gguf"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            prefix = f"/qwen-work/artifacts/{stock.name}"

            def row(path: str, artifact: Path) -> dict:
                return {"path": path, "size_bytes": artifact.stat().st_size,
                        "sha256": digest(artifact)}

            accepted = {"output": {"shards": [
                row(f"{prefix}/{first.name}", first),
                row(f"{prefix}/{second.name}", second),
            ]}}
            mapped = builder.exact_retirement_stock_shards(stock, accepted)
            self.assertEqual([Path(item["path"]).name for item in mapped],
                             [first.name, second.name])
            self.assertEqual([item["size_bytes"] for item in mapped],
                             [first.stat().st_size, second.stat().st_size])
            self.assertEqual([item["sha256"] for item in mapped],
                             [digest(first), digest(second)])

            bad_paths = (
                f"/qwen-work/artifacts/{stock.name}-sibling/{first.name}",
                f"{prefix}/../{stock.name}/{first.name}",
                str(first),
            )
            for path in bad_paths:
                with self.subTest(path=path):
                    rejected = {"output": {"shards": [row(path, first)]}}
                    with self.assertRaises(builder.BuilderError):
                        builder.exact_retirement_stock_shards(stock, rejected)

            mixed = {"output": {"shards": [
                row(f"{prefix}/{first.name}", first),
                row(f"/qwen-work/artifacts/{stock.name}-sibling/{second.name}", second),
            ]}}
            with self.assertRaises(builder.BuilderError):
                builder.exact_retirement_stock_shards(stock, mixed)

    def test_builder_rejects_bad_capture_and_preserves_uninventoried_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stock = root / "stock"
            workset = root / "workset"
            evidence = root / "evidence"
            stock.mkdir()
            shard = stock / "stock-00001.gguf"
            shard.write_bytes(b"captured-stock")
            sentinel = stock / "operator-note.txt"
            sentinel.write_bytes(b"must survive exact retirement")
            sentinel_sha = digest(sentinel)
            record = stock / "qwen-quant-build-record.json"
            write_json(record, {
                "status": "complete",
                "experiment": {"kind": "stock_control", "stock_weights_unchanged": True},
                "output": {"shards": [{"path": f"/qwen-work/artifacts/{stock.name}/{shard.name}",
                                          "size_bytes": shard.stat().st_size,
                                          "sha256": digest(shard)}]},
            })
            capture = root / "capture-manifest.json"
            write_json(capture, {
                "schema": "ember.qwen3.8.stock-control-activation-capture.v1",
                "status": "complete", "stock_rocmi4_only": True, "publishes": False,
                "model": {"build_record_sha256": digest(record), "shards": [{
                    "filename": shard.name, "size_bytes": shard.stat().st_size,
                    "sha256": digest(shard),
                }]},
            })
            args = argparse.Namespace(
                stock_dir=stock, build_record_sha256=digest(record),
                stock_capture_manifest=capture,
                stock_capture_manifest_sha256="0" * 64,
                workset_root=workset, output=evidence / "authorization.json",
            )
            with self.assertRaises((builder.BuilderError, ValueError)):
                builder.retire_captured_stock(args)
            self.assertTrue(shard.exists())
            self.assertFalse(args.output.exists())
            args.stock_capture_manifest_sha256 = digest(capture)
            result = builder.retire_captured_stock(args)
            self.assertFalse(shard.exists())
            self.assertTrue(record.exists())
            self.assertTrue(sentinel.exists())
            self.assertEqual(digest(sentinel), sentinel_sha)
            self.assertTrue(Path(result["completion"]).exists())


if __name__ == "__main__":
    unittest.main()
