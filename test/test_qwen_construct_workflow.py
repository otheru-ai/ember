#!/usr/bin/env python3
"""Static safety contracts for serial Qwen candidate construction on gfx1151."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/qwen-gfx1151-construct.yml"
REQUEST_BRIDGE = ROOT / ".github/workflows/qwen-gfx1151-request-bridge.yml"
DISPATCHER = ROOT / ".github/workflows/gfx1151-certify.yml"
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_snapshot_lock_handoff as snapshot_handoff  # noqa: E402


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
    def test_dispatcher_disk_reclaim_is_exact_and_stops_at_floor(self) -> None:
        body = DISPATCHER.read_text(encoding="utf-8")
        self.assertIn("qwen-reclaim-dangling-build-images-20260828", body)
        self.assertIn("required=$((1152 * 1024 * 1024 * 1024))", body)
        for digest in (
            "6bc8aa48fcf203d2d0a6b06c54df7b1816a1ad3127791fb64ecbbc5e3672ca16",
            "cdca5af61a921a29ca7632643f836f494e462e4b74e7e7603de97b27960912a6",
            "ffcb6c666ecc406bbbb579229602631d139c8518325e52d5ab19245f69ac1f80",
        ):
            self.assertIn(digest, body)
        self.assertIn('image.get("RepoTags") not in (None, [])', body)
        self.assertIn('docker ps -aq --filter ancestor="$image"', body)
        self.assertIn('(( available >= required )) && break', body)
        self.assertNotIn("docker system prune", body)
        self.assertNotIn("docker volume prune", body)
        for index, block in enumerate(workflow_run_blocks(body)):
            neutral = re.sub(r"\$\{\{.*?\}\}", "github-expression", block)
            result = subprocess.run(
                ["bash", "-n"], input=neutral, text=True, capture_output=True
            )
            self.assertEqual(
                result.returncode, 0,
                f"dispatcher run block {index}: {result.stderr}",
            )
            for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", neutral, re.S):
                compile(script, f"dispatcher-run-{index}-heredoc.py", "exec")

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
        self.assertIn("SPLITTER_PATCH_SHA256: b174f2b2a0b", body)
        self.assertIn("gguf-split-bounded-copy.patch", body)
        self.assertIn("fc6fc7103959763791d49e338916dd020429b1948ad357a5e5e54d54137be321", body)
        self.assertIn('builder="$repository@$builder_digest"', body)
        self.assertIn('runtime="$repository@$runtime_digest"', body)
        self.assertIn("converter-requirements.freeze.txt", body)
        self.assertIn("QWEN_CONVERTER_LOCK_SHA256", body)
        # The unprivileged construction container cannot use the dev image's
        # root-owned ccache directory. Conversion tooling is pinned and built
        # once, so disable auto-detected ccache instead of weakening ownership.
        self.assertIn("-DGGML_CCACHE=OFF", body)
        self.assertIn('test "$(git rev-parse HEAD)" = "$TARGET_SHA"', body)
        self.assertIn("org.opencontainers.image.revision", body)

    def test_construction_is_serial_same_path_uid_and_bounded(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("'gfx1151-certification'", body)
        self.assertIn("cancel-in-progress: false", body)
        self.assertIn("runs-on: [self-hosted, linux, x64, gfx1151]", body)
        self.assertIn("workspace=/var/tmp/ember-qwen3.8-flash-next", body)
        self.assertIn('-v "$QWEN_WORKSPACE:$QWEN_WORKSPACE"', body)
        self.assertIn('-v "$GITHUB_WORKSPACE:$GITHUB_WORKSPACE:ro"', body)
        # Nine build/runtime containers remain unprivileged and 125-GiB
        # bounded.  The one capability-minimal 256-MiB root helper can mutate
        # metadata on the completed fetch lock only; it is not a construction
        # container and mounts neither the workset nor output directories.
        self.assertEqual(body.count("docker run"), 10)
        self.assertEqual(body.count("--memory 125g"), 9)
        self.assertEqual(body.count("--memory-swap 125g"), 9)
        self.assertEqual(body.count('--user "$uid:$gid"'), 9)
        self.assertEqual(body.count("--user 0:0"), 1)
        self.assertIn("--memory 256m --memory-swap 256m", body)
        self.assertGreaterEqual(body.count("--memory-limit-bytes 134217728000"), 3)
        self.assertNotIn("/qwen-work/", body)
        self.assertNotIn("--user root", body)

    def test_snapshot_lock_handoff_is_exact_idle_and_precedes_construction(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        handoff = body.index("Hand off only the completed snapshot coordination inode")
        tools = body.index("Prepare exact pinned conversion tools")
        exclusive = body.index("Acquire exclusive UMA ownership")
        construction = body.index("Prepare one immutable content-addressed BF16 cache")
        self.assertLess(handoff, tools)
        self.assertLess(handoff, exclusive)
        self.assertLess(handoff, construction)
        self.assertIn("scripts/qwen_snapshot_lock_handoff.py", body)
        self.assertIn('-v "$QWEN_SNAPSHOT:$QWEN_SNAPSHOT"', body)
        self.assertNotIn('-v "$QWEN_WORKSPACE:$QWEN_WORKSPACE"', body[handoff:tools])
        self.assertIn("--cap-drop ALL --cap-add CHOWN --cap-add DAC_OVERRIDE", body)
        self.assertIn("--security-opt no-new-privileges --pids-limit 16 --user 0:0", body)
        self.assertIn("--verify-only", body[handoff:tools])

        source = (ROOT / "scripts/qwen_snapshot_lock_handoff.py").read_text(
            encoding="utf-8"
        )
        flock = source.index("fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)")
        mutation = source.index("os.fchown(descriptor, runner_uid, runner_gid)")
        self.assertLess(flock, mutation)
        for required in (
            "os.O_NOFOLLOW", "stat.S_ISREG", "metadata.st_nlink != 1",
            "os.fchmod(descriptor, 0o600)", "_same_inode(opened, named)",
            "model_bytes_touched\": 0",
        ):
            self.assertIn(required, source)
        for forbidden in ("os.walk", "os.listdir", "glob(", "rglob(", "read_bytes"):
            self.assertNotIn(forbidden, source)

    def test_snapshot_lock_handoff_rejects_leases_links_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            snapshot = Path(raw) / "snapshot"
            snapshot.mkdir()
            payload = snapshot / "model-00001-of-00001.safetensors"
            payload.write_bytes(b"model payload must remain untouched")
            payload_before = payload.stat()
            payload_digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            lock = snapshot / snapshot_handoff.LOCK_NAME
            lock.write_bytes(b"")
            lock.chmod(0o640)
            inode = (lock.stat().st_dev, lock.stat().st_ino)
            runner_uid = os.getuid() if os.getuid() > 0 else 12345
            runner_gid = os.getgid() if os.getgid() > 0 else 12345

            result = snapshot_handoff.handoff_snapshot_lock(
                snapshot, runner_uid=runner_uid, runner_gid=runner_gid,
            )
            self.assertEqual(result["status"], "handed_off")
            self.assertTrue(result["exclusive_idle_proof"])
            self.assertEqual(result["model_bytes_touched"], 0)
            self.assertEqual((lock.stat().st_dev, lock.stat().st_ino), inode)
            self.assertEqual(lock.stat().st_nlink, 1)
            self.assertEqual(stat.S_IMODE(lock.stat().st_mode), 0o600)
            verified = snapshot_handoff.handoff_snapshot_lock(
                snapshot, runner_uid=runner_uid, runner_gid=runner_gid, verify_only=True,
            )
            self.assertEqual(verified["inode"], inode[1])

            with lock.open("r+b") as active_fetch:
                fcntl.flock(active_fetch, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(
                        snapshot_handoff.HandoffError, "active fetch or read lease"):
                    snapshot_handoff.handoff_snapshot_lock(
                        snapshot, runner_uid=runner_uid, runner_gid=runner_gid)
                fcntl.flock(active_fetch, fcntl.LOCK_UN)

            with lock.open("r+b") as active_reader:
                fcntl.flock(active_reader, fcntl.LOCK_SH | fcntl.LOCK_NB)
                with self.assertRaisesRegex(
                        snapshot_handoff.HandoffError, "active fetch or read lease"):
                    snapshot_handoff.handoff_snapshot_lock(
                        snapshot, runner_uid=runner_uid, runner_gid=runner_gid)
                fcntl.flock(active_reader, fcntl.LOCK_UN)

            alias = snapshot / "lock-alias"
            os.link(lock, alias)
            with self.assertRaisesRegex(snapshot_handoff.HandoffError, "one hard link"):
                snapshot_handoff.handoff_snapshot_lock(
                    snapshot, runner_uid=runner_uid, runner_gid=runner_gid)
            alias.unlink()

            lock.unlink()
            lock.symlink_to(payload.name)
            with self.assertRaisesRegex(snapshot_handoff.HandoffError, "non-symlink"):
                snapshot_handoff.handoff_snapshot_lock(
                    snapshot, runner_uid=runner_uid, runner_gid=runner_gid)

            payload_after = payload.stat()
            self.assertEqual(hashlib.sha256(payload.read_bytes()).hexdigest(), payload_digest)
            self.assertEqual(
                (payload_after.st_dev, payload_after.st_ino, payload_after.st_size,
                 payload_after.st_mode, payload_after.st_mtime_ns),
                (payload_before.st_dev, payload_before.st_ino, payload_before.st_size,
                 payload_before.st_mode, payload_before.st_mtime_ns),
            )

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
        reusable = re.search(r"workflow_call:\n    inputs:\n(.*?)\n  workflow_dispatch:",
                             body, re.S)
        self.assertIsNotNone(reusable)
        declared.update(re.findall(r"^      ([a-z0-9_]+):$", reusable.group(1), re.M))
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

    def test_request_bridge_is_bounded_digest_pinned_and_non_mutating(self) -> None:
        body = REQUEST_BRIDGE.read_text(encoding="utf-8")
        ruby = shutil.which("ruby")
        if ruby:
            subprocess.run(
                [ruby, "-e", "require 'yaml'; YAML.parse_file(ARGV[0])", str(REQUEST_BRIDGE)],
                check=True,
            )
        dispatch = re.search(r"workflow_dispatch:\n    inputs:\n(.*?)\npermissions:", body, re.S)
        self.assertIsNotNone(dispatch)
        declared = re.findall(r"^      ([a-z0-9_]+):$", dispatch.group(1), re.M)
        self.assertEqual(declared, ["commit_sha", "mode", "request_payload_base64",
                                    "request_payload_sha256", "output"])
        self.assertLessEqual(len(declared), 10)
        reusable = re.search(r"workflow_call:\n    inputs:\n(.*?)\n  workflow_dispatch:",
                             body, re.S)
        self.assertIsNotNone(reusable)
        self.assertEqual(len(re.findall(r"^      [a-z0-9_]+:$", reusable.group(1), re.M)), 5)
        referenced = set(re.findall(r"\binputs\.([a-zA-Z0-9_]+)\b", body))
        self.assertEqual(referenced - set(declared), set())
        self.assertIn("base64.b64decode(encoded, validate=True)", body)
        self.assertIn("hashlib.sha256(payload).hexdigest() != expected_sha", body)
        self.assertIn("os.O_WRONLY | os.O_CREAT | os.O_EXCL", body)
        self.assertGreaterEqual(body.count("os.fsync("), 2)
        self.assertIn("evidence/operation-requests", body)
        self.assertIn("construction-[A-Za-z0-9._-]{1,80}", body)
        self.assertIn("sha256(path) != item[\"sha256\"]", body)
        self.assertIn("decoded request payload must contain 1..65536 bytes", body)
        for forbidden in ("docker run", "ember-gpu-lock", "ember-cert-production",
                          "rm ", "unlink(", "rmtree", "packages: write"):
            self.assertNotIn(forbidden, body)
        for block in workflow_run_blocks(body):
            neutral = re.sub(r"\$\{\{.*?\}\}", "github-expression", block)
            result = subprocess.run(
                ["bash", "-n"], input=neutral, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", neutral, re.S):
                compile(script, "request-bridge-heredoc.py", "exec")
        validator = next(
            script for block in workflow_run_blocks(body)
            for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", block, re.S)
            if "candidate-construction-request.v1" in script
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            env_path = root / "github-env"
            output = root / "must-not-exist.json"
            bad_base64 = subprocess.run(
                [sys.executable, "-", "prepare-cache", "%%%", "0" * 64,
                 str(output), str(env_path)],
                input=validator, text=True, capture_output=True,
            )
            self.assertNotEqual(bad_base64.returncode, 0)
            self.assertFalse(output.exists())

            payload = json.dumps({
                "schema": "ember.qwen3.8.candidate-construction-request.v1",
                "mode": "prepare-cache", "parameters": {"unexpected": None},
                "publishes": False, "deletes": False,
            }, separators=(",", ":")).encode()
            malformed = subprocess.run(
                [sys.executable, "-", "prepare-cache",
                 base64.b64encode(payload).decode(),
                 hashlib.sha256(payload).hexdigest(), str(output), str(env_path)],
                input=validator, text=True, capture_output=True,
            )
            self.assertNotEqual(malformed.returncode, 0)
            self.assertFalse(output.exists())

    def test_default_branch_dispatcher_calls_same_revision_reusable_workflows(self) -> None:
        body = DISPATCHER.read_text(encoding="utf-8")
        dispatch = re.search(r"workflow_dispatch:\n    inputs:\n(.*?)\npermissions:", body, re.S)
        self.assertIsNotNone(dispatch)
        inputs = re.findall(r"^      ([a-z0-9_]+):$", dispatch.group(1), re.M)
        self.assertEqual(inputs, ["commit_sha", "release_version",
                                  "qwen_dispatch_envelope_base64",
                                  "qwen_dispatch_envelope_sha256"])
        self.assertLessEqual(len(inputs), 10)
        for path, expected_count in ((REQUEST_BRIDGE, 5), (WORKFLOW, 4),
                                     (ROOT / ".github/workflows/qwen-gfx1151-retire-stock.yml", 10),
                                     (ROOT / ".github/workflows/qwen-gfx1151-vision.yml", 15)):
            called = re.search(r"workflow_call:\n    inputs:\n(.*?)(?:\n    outputs:|\n  workflow_dispatch:)",
                               path.read_text(encoding="utf-8"), re.S)
            self.assertIsNotNone(called)
            self.assertEqual(
                len(re.findall(r"^      [a-z0-9_]+:$", called.group(1), re.M)),
                expected_count,
            )
        self.assertIn("inputs.release_version == 'qwen-dispatch'", body)
        self.assertIn("CALLER_SHA: ${{ github.sha }}", body)
        self.assertIn('test "$CALLER_SHA" = "$TARGET_SHA"', body)
        for workflow in (
            "qwen-gfx1151-request-bridge.yml",
            "qwen-gfx1151-construct.yml",
            "qwen-gfx1151-retire-stock.yml",
            "qwen-gfx1151-vision.yml",
        ):
            self.assertIn(f"uses: ./.github/workflows/{workflow}", body)
        self.assertNotRegex(body, r"uses:.*\$\{\{")
        self.assertEqual(body.count("uses: ./.github/workflows/qwen-gfx1151-"), 4)
        self.assertIn("contains(github.workflow_ref, '/.github/workflows/gfx1151-certify.yml@')",
                      WORKFLOW.read_text(encoding="utf-8"))
        self.assertIn("ember.qwen3.8.branch-dispatch-envelope.v1", body)
        self.assertIn("decoded dispatch envelope must contain 1..32768 bytes", body)
        self.assertIn("len(nested_payload) > 16384", body)
        self.assertIn("RETIRE_CAPTURED_STOCK_SHARDS", body)
        self.assertIn("value.get(\"deletes\") is not (operation == \"retire\")", body)
        for index, block in enumerate(workflow_run_blocks(body)):
            neutral = re.sub(r"\$\{\{.*?\}\}", "github-expression", block)
            shell = subprocess.run(
                ["bash", "-n"], input=neutral, text=True, capture_output=True,
            )
            self.assertEqual(shell.returncode, 0,
                             f"dispatcher run block {index}: {shell.stderr}")
            for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", neutral, re.S):
                compile(script, f"dispatcher-run-{index}-heredoc.py", "exec")

        parser = next(
            script for block in workflow_run_blocks(body)
            for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", block, re.S)
            if "branch-dispatch-envelope.v1" in script
        )
        revision = "1" * 40
        nested = b"{}"
        valid = {
            "schema": "ember.qwen3.8.branch-dispatch-envelope.v1",
            "ember_revision": revision,
            "operation": "request",
            "inputs": {
                "mode": "prepare-cache",
                "request_payload_base64": base64.b64encode(nested).decode(),
                "request_payload_sha256": hashlib.sha256(nested).hexdigest(),
                "output": ("/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/"
                           "evidence/operation-requests/construction-cache.json"),
            },
            "publishes": False,
            "deletes": False,
        }
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "github-output"

            def invoke(value: dict) -> subprocess.CompletedProcess[str]:
                payload = json.dumps(value, separators=(",", ":")).encode()
                return subprocess.run(
                    [sys.executable, "-", revision, base64.b64encode(payload).decode(),
                     hashlib.sha256(payload).hexdigest(), str(output)],
                    input=parser, text=True, capture_output=True,
                )

            accepted = invoke(valid)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            emitted = output.read_text(encoding="utf-8")
            self.assertIn("operation=request\n", emitted)
            self.assertIn("mode=prepare-cache\n", emitted)

            output.unlink()
            vision = {
                "schema": "ember.qwen3.8.branch-dispatch-envelope.v1",
                "ember_revision": revision,
                "operation": "vision",
                "inputs": {
                    "runtime_image": "ghcr.io/otheru-ai/ember@sha256:" + "2" * 64,
                    "runtime_dev_image": "ghcr.io/otheru-ai/ember-dev@sha256:" + "3" * 64,
                    "model": "/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/model.gguf",
                    "model_sha256": "4" * 64,
                    "model_build_record": "/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/build.json",
                    "model_build_record_sha256": "5" * 64,
                    "mtp": "/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/mtp.gguf",
                    "mtp_sha256": "6" * 64, "mtp_depth": "2",
                    "mmproj": "/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/mmproj.gguf",
                    "mmproj_sha256": "7" * 64,
                    "vision_vocab": "/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/vocab.gguf",
                    "vision_vocab_sha256": "8" * 64,
                    "output": "/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/vision-test",
                },
                "publishes": False, "deletes": False,
            }
            accepted_vision = invoke(vision)
            self.assertEqual(accepted_vision.returncode, 0, accepted_vision.stderr)
            emitted = output.read_text(encoding="utf-8")
            self.assertIn("operation=vision\n", emitted)
            self.assertIn("mtp_depth=2\n", emitted)
            self.assertIn("vision_output=/var/tmp/ember-qwen3.8-flash-next/", emitted)

            output.unlink()
            invalid = dict(valid)
            invalid["deletes"] = True
            rejected = invoke(invalid)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(output.exists())

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
