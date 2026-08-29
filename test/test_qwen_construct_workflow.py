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
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/qwen-gfx1151-construct.yml"
REQUEST_BRIDGE = ROOT / ".github/workflows/qwen-gfx1151-request-bridge.yml"
DISPATCHER = ROOT / ".github/workflows/gfx1151-certify.yml"
Q3_FIRST_TOKEN_PLAN = ROOT / ".github/workflows/qwen-q3-first-token-plan.yml"
Q3_FIRST_TOKEN = ROOT / ".github/workflows/qwen-q3-first-token.yml"
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_snapshot_lock_handoff as snapshot_handoff  # noqa: E402
import qwen_selection_corpus_stage as selection_stage  # noqa: E402
import qwen_candidate_request as candidate_request  # noqa: E402
import qwen_reuse_candidate as reuse_candidate  # noqa: E402


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
    def test_candidate_planner_accepts_only_projection_equivalent_capture_recipe(self) -> None:
        predecessor = next(iter(candidate_request.CAPTURE_RECIPE_PROJECTIONS))
        intervention = {
            "lambdas": [0.25, 0.5, 0.75, 1.0],
            "layer_policies": {
                "band-10-42": "10-42; exploratory transfer hypothesis from the pinned OtherU DeepSeek result",
                "upper-24": "24-47", "upper-12": "36-47",
                "non-qsa-band-10-42": "non-QSA layers within the exploratory DeepSeek-derived 10-42 band",
            },
            "artifact_format": "Q4_0_ROCMI4", "runtime_mode": "exact_dequant",
            "may_select_recipe": True, "final_release_eligible": False,
        }
        recipe = {"sha256": "9" * 64,
                  "value": {"intervention_sweep": intervention}}
        self.assertTrue(candidate_request.capture_recipe_compatible(
            predecessor, recipe))
        self.assertFalse(candidate_request.capture_recipe_compatible(
            "8" * 64, recipe))
        changed = json.loads(json.dumps(recipe))
        changed["value"]["intervention_sweep"]["lambdas"][0] = 0.20
        self.assertFalse(candidate_request.capture_recipe_compatible(
            predecessor, changed))
        self.assertTrue(candidate_request.capture_recipe_compatible(
            recipe["sha256"], changed))

    def test_candidate_planner_derives_sweep_identity_from_pinned_capture(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            requests = root / "operation-requests"
            requests.mkdir()

            def write(name: str, value: object) -> tuple[Path, dict[str, str]]:
                path = root / name
                if isinstance(value, bytes):
                    path.write_bytes(value)
                else:
                    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
                return path, {"path": str(path),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

            profile_path, profile = write("profile.json", {"fixture": True})
            inventory_path, _inventory = write(
                "inventory.json", {"fixture": "snapshot inventory"})
            configuration = {
                "id": "lambda-0.25-band-10-42", "scale": 0.25,
                "layer_policy": "band-10-42", "quantization_arm": "rocmi4-control",
            }
            plan_value = {
                "recipe": {"sha256": "1" * 64},
                "release_profile": {"path": str(profile_path),
                                    "sha256": profile["sha256"]},
                "sweep_configurations": [configuration],
                "format_arms": [],
            }
            _plan_path, plan = write("plan.json", plan_value)
            intervention_path = root / "capture" / "interventions" / (
                configuration["id"] + ".json")
            intervention_path.parent.mkdir(parents=True)
            intervention_path.write_text('{"kind":"directional_ablation"}\n', encoding="utf-8")
            intervention_sha = hashlib.sha256(intervention_path.read_bytes()).hexdigest()
            capture_value = {
                "schema": candidate_request.CAPTURE_SCHEMA, "status": "complete",
                "publishes": False, "recipe_sha256": "1" * 64,
                "interventions": [{"id": configuration["id"], "lambda": 0.25,
                    "layer_policy": "band-10-42",
                    "filename": "interventions/" + intervention_path.name,
                    "sha256": intervention_sha, "target_count": 1}],
            }
            capture_path = root / "capture" / "capture-manifest.json"
            capture_path.write_text(json.dumps(capture_value) + "\n", encoding="utf-8")
            capture = {"path": str(capture_path),
                       "sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest()}
            descriptors = {}
            for name in ("cache", "rocmi4", "fast"):
                _path, descriptors[name] = write(name + ".json", {"name": name})
            output = requests / "construction-first-sweep.json"
            intent_value = {
                "schema": candidate_request.INTENT_SCHEMA,
                "ember_revision": "a" * 40, "stage": "sweep",
                "row_id": configuration["id"], "candidate_id": configuration["id"],
                "capture_manifest": capture, "cache_manifest": descriptors["cache"],
                "companion_rocmi4": descriptors["rocmi4"],
                "companion_fast": descriptors["fast"], "selection_plan": plan,
                "prior_ledger": None, "construction_request_output": str(output),
                "publishes": False, "deletes": False,
            }
            intent_path, intent = write("intent.json", intent_value)
            with (mock.patch.object(candidate_request, "REQUEST_ROOT", requests),
                  mock.patch.object(candidate_request.bakeoff, "verify_plan",
                                    return_value=plan_value),
                  mock.patch.object(candidate_request.quant, "validate_profile",
                                    return_value=({"intervention": {}}, {}, inventory_path)),
                  mock.patch.object(candidate_request.quant,
                                    "validate_intervention_manifest",
                                    return_value=({"kind": "directional_ablation"},
                                                  {"target_count": 1})),
                  mock.patch.object(candidate_request.quant,
                                    "validated_quantization_arms",
                                    return_value={"rocmi4-control": {}}),
                  mock.patch.object(candidate_request.quant,
                                    "validate_rocmi4_sweep_authorization",
                                    return_value={"configuration_id":
                                                  configuration["id"]})):
                request_value, actual = candidate_request.derive(
                    intent_path, intent["sha256"], "a" * 40)
            self.assertEqual(actual, output)
            parameters = request_value["parameters"]
            self.assertEqual(parameters["intervention_configuration_id"],
                             configuration["id"])
            self.assertEqual(parameters["quantization_arm"], "rocmi4-control")
            self.assertEqual(parameters["mtp_matrix_quant_contract"], "Q4_0_ROCMI4")
            self.assertEqual(parameters["intervention_manifest"]["sha256"],
                             intervention_sha)

    def test_candidate_plan_workflow_is_create_only_and_chains_to_constructor(self) -> None:
        body = (ROOT / ".github/workflows/qwen-candidate-plan.yml").read_text(
            encoding="utf-8")
        self.assertIn("workflow_call:", body)
        self.assertIn("runs-on: [self-hosted, linux, x64, gfx1151]", body)
        self.assertIn("qwen-candidate-plan-${{ github.run_id }}", body)
        self.assertIn("O_EXCL", body)
        self.assertIn("decoded candidate intent must contain 1..32768 bytes", body)
        self.assertIn("scripts/qwen_candidate_request.py", body)
        self.assertNotIn("ember-gpu-lock acquire", body)
        for block in workflow_run_blocks(body):
            neutral = re.sub(r"\$\{\{.*?\}\}", "github-expression", block)
            shell = subprocess.run(["bash", "-n"], input=neutral, text=True,
                                   capture_output=True)
            self.assertEqual(shell.returncode, 0, shell.stderr)
            for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", neutral, re.S):
                compile(script, "candidate-plan-heredoc.py", "exec")

    def test_candidate_planner_format_row_is_bound_to_attested_sweep_winner(self) -> None:
        configuration = {"id": "lambda-0.50-upper-12"}
        arm = {"id": "rocmi4-q6k-main-rocmi4-mtp-d3",
               "quantization_arm": "rocmi4-q6k-embedding-head",
               "mtp_matrix_quant_contract": "Q4_0_ROCMI4"}
        plan = {"sweep_configurations": [configuration], "format_arms": [arm]}
        intent = {"stage": "format", "row_id": arm["id"],
                  "prior_ledger": {"attested": True}}
        prior = {"selected_configuration_id": configuration["id"],
                 "selected_configuration": configuration}
        with mock.patch.object(candidate_request.bakeoff, "read_prior_ledger",
                               return_value=(prior, "1" * 64)) as verifier:
            selected, selected_arm, quant_arm, mtp = (
                candidate_request.selected_configuration(intent, plan))
        self.assertEqual(selected, configuration)
        self.assertEqual(selected_arm, arm)
        self.assertEqual(quant_arm, arm["quantization_arm"])
        self.assertEqual(mtp, arm["mtp_matrix_quant_contract"])
        verifier.assert_called_once_with(intent["prior_ledger"], "sweep", plan)

    def selection_fixture(self, root: Path) -> tuple[Path, Path, str]:
        source = root / "protected"
        staged = root / "staged"
        source.mkdir()
        staged.mkdir(mode=0o700)
        artifacts = []
        for index, name in enumerate(selection_stage.ARTIFACT_NAMES):
            path = source / name
            path.write_text(json.dumps({"id": index}) + "\n", encoding="utf-8")
            artifacts.append({"filename": name,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "record_count": 1})
        revision = "a" * 40
        manifest = {"source": {"revision": revision},
                    "partition": {"pairwise_request_overlap_count": 0},
                    "artifacts": artifacts}
        (source / selection_stage.MANIFEST_NAME).write_text(
            json.dumps(manifest), encoding="utf-8")
        contract = root / "contract.json"
        contract.write_text(json.dumps({
            "source": {"revision": revision},
            "pairwise_request_overlap_count": 0,
            "derived_artifacts": {
                row["filename"]: {"sha256": row["sha256"],
                                  "record_count": row["record_count"]}
                for row in artifacts
            },
        }), encoding="utf-8")
        return source, staged, revision

    def test_selection_corpus_stage_is_exact_resumable_and_final_blind(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, staged, revision = self.selection_fixture(root)
            contract = root / "contract.json"
            contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
            # The Docker daemon's protected /srv view exposes the artifacts but
            # not the Actions-host phase manifest. The helper must not depend
            # on that namespace-sensitive source file.
            (source / selection_stage.MANIFEST_NAME).unlink()
            first = selection_stage.stage(
                source, staged, os.getuid(), os.getgid(), contract, contract_sha, revision)
            self.assertFalse(first["reused"])
            self.assertEqual(set(path.name for path in staged.iterdir()),
                             set(selection_stage.STAGED_NAMES))
            self.assertEqual(stat.S_IMODE(staged.stat().st_mode), 0o500)
            for path in staged.iterdir():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)
            second = selection_stage.stage(
                source, staged, os.getuid(), os.getgid(), contract, contract_sha, revision)
            self.assertTrue(second["reused"])
            verified = selection_stage.verify(
                contract, contract_sha, staged, revision)
            self.assertEqual(verified["status"], "complete")

            staged.chmod(0o700)
            (staged / "unexpected-heldout.jsonl").write_text("hidden\n", encoding="utf-8")
            with self.assertRaisesRegex(selection_stage.SelectionCorpusError,
                                        "unexpected corpus entry"):
                selection_stage.verify(
                    contract, contract_sha, staged, revision)

    def test_selection_corpus_stage_rejects_partial_prior_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, staged, revision = self.selection_fixture(root)
            contract = root / "contract.json"
            (staged / selection_stage.MANIFEST_NAME).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(selection_stage.SelectionCorpusError,
                                        "partial or unexpected"):
                selection_stage.stage(
                    source, staged, os.getuid(), os.getgid(), contract,
                    hashlib.sha256(contract.read_bytes()).hexdigest(), revision)

    def test_dispatcher_disk_reclaim_is_exact_and_keeps_tooling_margin(self) -> None:
        body = DISPATCHER.read_text(encoding="utf-8")
        reclaim = body[
            body.index("  qwen-reclaim-stale-builds:"):
            body.index("\n  certify:", body.index("  qwen-reclaim-stale-builds:"))
        ]
        self.assertIn("qwen-reclaim-stale-builds-v7-20260828", body)
        self.assertIn('CALLER_SHA: ${{ github.sha }}', reclaim)
        self.assertIn('test "$CALLER_SHA" = "$TARGET_SHA"', reclaim)
        self.assertNotIn("actions/checkout", reclaim)
        self.assertIn("construction_floor=$((1152 * 1024 * 1024 * 1024))", body)
        self.assertIn("required=$((1160 * 1024 * 1024 * 1024))", body)
        for tag, digest in (
            ("10236b8489a3", "7ab721ddf095ecf0526a5cc2059c308e992e26f748f33de4b1b883cc46226ab3"),
            ("2f99a12f8afd", "1d6c71627ac657b5b36defddbde3b99d2099185ec4bbed9f69ee39e09b905cdb"),
            ("34f832863317", "f6661ec650a2957a1041b738b5326f296e610be6c802526b454c0fb6350ba19b"),
            ("35eca745378a", "9251ec47dd52f54be017a4c551ccdf7300727bd3ef52b70cd5829b27f05f4bd8"),
            ("3afa5b3d8a86", "fec837c44a735faccec88ba773d14208a47c9440548dce931f9a02756c49a424"),
            ("3f4d1ee72b71", "500f0a6829d9848a0f81e4fa828c25dbd6853d9ed69ad0ebc010673269bc57e6"),
            ("4af543cc1ad0", "cdd549e22658b0a2bca0f125cf8c7a2a1cf9f85b3f867d11bd71fe5ce30cfae4"),
            ("6bc799ba9f22", "18e184ffa6075e6aecf701d683224508634e6d8049937f481bd5ed03bddb6ec6"),
            ("78261de805a5", "b9936deb22bb4d7dff6ef4e6c61064987e433755557a51e080847dfa1cb07a54"),
            ("90bf8f9f9096", "b752d459c07d24a1af925c1e95ab0ee323fe017ec7195fc7298ec47aa0c0bdeb"),
            ("936ac6f1ec95", "8088484329554efff07ca3f68624d9b23bd7e20e18717f56e0e0875c07b0bbaa"),
            ("9e44f17e8189", "f4e095c5d8de0aa0ad339ecb928bd1a6357714a354ae604bc0b1320a294f7fcb"),
        ):
            self.assertIn(f"dev-sha-{tag} sha256:{digest}", reclaim)
        self.assertNotIn("dev-sha-051b43846756", reclaim)
        self.assertNotIn("dev-sha-bc3b0999999b", reclaim)
        self.assertIn('docker ps -aq --filter ancestor="$image"', body)
        self.assertIn('docker ps -aq --filter ancestor="$expected_id"', body)
        self.assertIn('Already absent stale image:', body)
        self.assertIn('Already absent stale tooling:', body)
        self.assertIn('test "$revision" != "$TARGET_SHA"', body)
        for revision in (
            "126c4ff8284ef99e399d366aec6275a5df5b7c0f",
            "d3990664c3ba8ef2cb368cf4e7aff8baf1c06e20",
            "13e272f1bcab35b2e133cb2d0c040fdc7f7bc9b1",
            "4b038f692cd83e405da4b9834b88f4e113158e43",
        ):
            self.assertIn(revision, reclaim)
        self.assertIn('^ghcr\\.io/otheru-ai/ember:dev-sha-[0-9a-f]{12}$', body)
        self.assertIn('^sha256:[0-9a-f]{64}$', body)
        self.assertIn('dev-sha-${TARGET_SHA:0:12}', body)
        self.assertIn("Preserving in-use stale image", body)
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
        # bounded. Two capability-minimal 256-MiB root helpers cover only the
        # completed fetch-lock inode and the four-file selection-corpus staging
        # boundary; neither can read model weights or candidate outputs.
        self.assertEqual(body.count("docker run"), 11)
        self.assertEqual(body.count("--memory 125g"), 9)
        self.assertEqual(body.count("--memory-swap 125g"), 9)
        self.assertEqual(body.count('--user "$uid:$gid"'), 9)
        self.assertEqual(
            body.count("--env USER=ember-qwen --env LOGNAME=ember-qwen"), 9
        )
        self.assertEqual(body.count("--user 0:0"), 2)
        self.assertEqual(body.count("--memory 256m --memory-swap 256m"), 2)
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
            "options: [prepare-cache, prepare-companions, build-candidate, reuse-candidate, normalize-candidate]",
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
        self.assertIn('"publishes":False,"deletes":False', body)
        self.assertIn("QWEN_COMPANION_CONSTRUCTION_SHA256", body)
        self.assertIn("QWEN_COMPANION_ROCMI4_OUTPUT_SHA256", body)
        self.assertIn("QWEN_COMPANION_FAST_OUTPUT_SHA256", body)
        self.assertIn("QWEN_SELECTION_PLAN_OUTPUT_SHA256", body)
        self.assertIn("Stage only the digest-pinned selection corpus", body)
        self.assertIn("scripts/qwen_selection_corpus_stage.py", body)
        self.assertNotIn('"final-heldout.jsonl")', body[
            body.index("Stage only the digest-pinned selection corpus"):
            body.index("Prepare exact pinned conversion tools")
        ])
        stage_source = (ROOT / "scripts/qwen_selection_corpus_stage.py").read_text(
            encoding="utf-8")
        self.assertIn('"sweep-validation.jsonl"', stage_source)
        self.assertNotIn('"final-heldout.jsonl"', stage_source)
        self.assertIn("Reusing completed immutable companion export", body)
        self.assertIn("Validated completed immutable companion inventory", body)
        self.assertIn("Companion construction descriptor SHA-256", body)
        self.assertIn("ROCMI4 inventory SHA-256", body)
        self.assertIn("ROCmFP4 FAST inventory SHA-256", body)
        self.assertIn("Selection plan SHA-256", body)
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
                                     (ROOT / ".github/workflows/qwen-gfx1151-vision.yml", 15),
                                     (ROOT / ".github/workflows/qwen-quality-capture.yml", 3),
                                     (ROOT / ".github/workflows/qwen-quality-plan.yml", 4),
                                     (ROOT / ".github/workflows/qwen-candidate-plan.yml", 4),
                                     (ROOT / ".github/workflows/qwen-gfx1151-bakeoff.yml", 6),
                                     (Q3_FIRST_TOKEN_PLAN, 1), (Q3_FIRST_TOKEN, 4)):
            called = re.search(r"workflow_call:\n    inputs:\n(.*?)(?:\n    outputs:|\n  workflow_dispatch:|\npermissions:)",
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
            "qwen-quality-capture.yml",
            "qwen-quality-plan.yml",
            "qwen-candidate-plan.yml",
            "qwen-gfx1151-bakeoff.yml",
            "qwen-q3-first-token-plan.yml",
            "qwen-q3-first-token.yml",
        ):
            self.assertIn(f"uses: ./.github/workflows/{workflow}", body)
        self.assertNotRegex(body, r"uses:.*\$\{\{")
        self.assertEqual(body.count("uses: ./.github/workflows/qwen-gfx1151-"), 7)
        self.assertEqual(body.count("uses: ./.github/workflows/qwen-quality-capture.yml"), 2)
        self.assertEqual(body.count("uses: ./.github/workflows/qwen-quality-plan.yml"), 1)
        self.assertEqual(body.count("uses: ./.github/workflows/qwen-candidate-plan.yml"), 1)
        self.assertIn("qwen-call-planned-quality:", body)
        self.assertIn("needs: [qwen-dispatch-envelope, qwen-call-quality-plan]", body)
        self.assertIn("needs.qwen-call-quality-plan.outputs.phase_descriptor", body)
        self.assertIn("needs.qwen-call-quality-plan.outputs.phase_descriptor_sha256", body)
        self.assertIn("qwen-call-planned-candidate:", body)
        self.assertIn("needs: [qwen-dispatch-envelope, qwen-call-candidate-plan]", body)
        self.assertIn("needs.qwen-call-candidate-plan.outputs.operation_request", body)
        self.assertIn(
            "inputs.release_version == 'qwen-q3-first-token-v1-20260828'", body)
        self.assertIn("qwen-plan-q3-first-token:", body)
        self.assertIn("qwen-build-q3-first-token:", body)
        self.assertIn("qwen-prove-q3-first-token:", body)
        q3_proof_job = body[body.index("  qwen-prove-q3-first-token:"):
                            body.index("  qwen-inspect-control-residue:")]
        for permission in ("contents: read", "packages: read", "id-token: write",
                           "attestations: write", "artifact-metadata: write"):
            self.assertIn(permission, q3_proof_job)
        self.assertIn("contains(github.workflow_ref, '/.github/workflows/gfx1151-certify.yml@')",
                      WORKFLOW.read_text(encoding="utf-8"))
        self.assertIn("ember.qwen3.8.branch-dispatch-envelope.v1", body)
        self.assertIn("decoded dispatch envelope must contain 1..32768 bytes", body)
        self.assertIn("len(nested_payload) > 16384", body)
        self.assertIn("RETIRE_CAPTURED_STOCK_SHARDS", body)
        self.assertIn('destructive = operation in {"retire", "bakeoff"}', body)
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
            quality = {
                "schema": "ember.qwen3.8.branch-dispatch-envelope.v1",
                "ember_revision": revision,
                "operation": "quality",
                "inputs": {
                    "phase_descriptor": "/var/tmp/ember-qwen3.8-flash-next/quality.json",
                    "phase_descriptor_sha256": "9" * 64,
                },
                "publishes": False, "deletes": False,
            }
            accepted_quality = invoke(quality)
            self.assertEqual(accepted_quality.returncode, 0, accepted_quality.stderr)
            emitted = output.read_text(encoding="utf-8")
            self.assertIn("operation=quality\n", emitted)
            self.assertIn("phase_descriptor_sha256=" + "9" * 64 + "\n", emitted)

            output.unlink()
            bakeoff = {
                "schema": "ember.qwen3.8.branch-dispatch-envelope.v1",
                "ember_revision": revision,
                "operation": "bakeoff",
                "inputs": {
                    "candidate_manifest": "/var/tmp/ember-qwen3.8-flash-next/candidate.json",
                    "candidate_manifest_sha256": "a" * 64,
                    "phase": "sweep",
                    "phase_request": "/var/tmp/ember-qwen3.8-flash-next/phase.json",
                    "phase_request_sha256": "b" * 64,
                },
                "publishes": False, "deletes": True,
            }
            accepted_bakeoff = invoke(bakeoff)
            self.assertEqual(accepted_bakeoff.returncode, 0, accepted_bakeoff.stderr)
            emitted = output.read_text(encoding="utf-8")
            self.assertIn("operation=bakeoff\n", emitted)
            self.assertIn("phase=sweep\n", emitted)

            output.unlink()
            quality_request = b'{"schema":"fixture"}\n'
            quality_plan = {
                "schema": "ember.qwen3.8.branch-dispatch-envelope.v1",
                "ember_revision": revision,
                "operation": "quality-plan",
                "inputs": {
                    "request_payload_base64": base64.b64encode(quality_request).decode(),
                    "request_payload_sha256": hashlib.sha256(quality_request).hexdigest(),
                    "request_output": "/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/operation-requests/quality.json",
                },
                "publishes": False, "deletes": False,
            }
            accepted_quality_plan = invoke(quality_plan)
            self.assertEqual(accepted_quality_plan.returncode, 0,
                             accepted_quality_plan.stderr)
            emitted = output.read_text(encoding="utf-8")
            self.assertIn("operation=quality-plan\n", emitted)
            self.assertIn("quality_request_output=/var/tmp/ember-", emitted)

            output.unlink()
            candidate_intent = b'{"schema":"fixture"}\n'
            candidate_plan = {
                "schema": "ember.qwen3.8.branch-dispatch-envelope.v1",
                "ember_revision": revision,
                "operation": "candidate-plan",
                "inputs": {
                    "intent_payload_base64": base64.b64encode(candidate_intent).decode(),
                    "intent_payload_sha256": hashlib.sha256(candidate_intent).hexdigest(),
                    "intent_output": "/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/operation-requests/candidate-intent-first.json",
                },
                "publishes": False, "deletes": False,
            }
            accepted_candidate_plan = invoke(candidate_plan)
            self.assertEqual(accepted_candidate_plan.returncode, 0,
                             accepted_candidate_plan.stderr)
            emitted = output.read_text(encoding="utf-8")
            self.assertIn("operation=candidate-plan\n", emitted)
            self.assertIn("candidate_intent_output=/var/tmp/ember-", emitted)

            output.unlink()
            invalid = dict(valid)
            invalid["deletes"] = True
            rejected = invoke(invalid)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(output.exists())

    def test_q3_first_token_lane_is_create_only_exclusive_and_scope_bounded(self) -> None:
        ruby = shutil.which("ruby")
        for path in (Q3_FIRST_TOKEN_PLAN, Q3_FIRST_TOKEN):
            workflow_body = path.read_text(encoding="utf-8")
            if ruby:
                subprocess.run(
                    [ruby, "-e", "require 'yaml'; YAML.parse_file(ARGV[0])", str(path)],
                    check=True,
                )
            for index, block in enumerate(workflow_run_blocks(workflow_body)):
                neutral = re.sub(r"\$\{\{.*?\}\}", "github-expression", block)
                shell = subprocess.run(
                    ["bash", "-n"], input=neutral, text=True, capture_output=True,
                )
                self.assertEqual(shell.returncode, 0,
                                 f"{path.name} run block {index}: {shell.stderr}")
                for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", neutral, re.S):
                    compile(script, f"{path.name}-{index}-heredoc.py", "exec")

        plan = Q3_FIRST_TOKEN_PLAN.read_text(encoding="utf-8")
        self.assertIn("rocmfp4-fast-matrix-q3-ple-q6k-embedding-head", plan)
        self.assertIn("lambda-0.25-band-10-42", plan)
        self.assertIn("os.O_EXCL", plan)
        self.assertIn('"publishes": False', plan)
        self.assertIn('"deletes": False', plan)
        self.assertIn("scripts/qwen_bakeoff.py --corpus-dir", plan)
        self.assertNotIn("ember-gpu-lock acquire", plan)
        self.assertNotIn("ember-cert-production stop", plan)

        proof = Q3_FIRST_TOKEN.read_text(encoding="utf-8")
        self.assertIn("iflag=direct", proof)
        self.assertIn("EMBER_TRACE_TOKENS=1", proof)
        self.assertIn("--validate-tokens 2", proof)
        self.assertIn('required = {"baseline", "restored", "fresh", "disk"}', proof)
        self.assertIn('"first_token_id": first', proof)
        self.assertIn("ember-gpu-lock acquire", proof)
        self.assertIn("ember-cert-production stop", proof)
        self.assertIn("ember-cert-production mask", proof)
        self.assertIn("if: ${{ always() && steps.safety.outputs.armed == 'yes' }}", proof)
        self.assertIn("ember-cert-production unmask", proof)
        self.assertIn("ember-gpu-lock release", proof)
        self.assertIn("actions/attest@", proof)
        self.assertIn("scripts/qwen_real_weight_gate.sh", proof)
        self.assertIn("--measurement-only --out-dir", proof)
        self.assertIn("--mtp-depth 3", proof)
        self.assertIn("hardware-measured.json", proof)
        self.assertIn("Measured peak UMA", proof)
        self.assertIn("Attest the complete Q3 benchmark evidence", proof)
        self.assertIn("no quality or performance claim", proof)

        self.assertIn("construction_mode:", plan)
        self.assertIn("ARTIFACT_BUILDER_SHA:", plan)
        self.assertIn("git diff --quiet", plan)
        self.assertIn("construction_mode=reuse-candidate", plan)
        self.assertIn("mode: ${{ needs.qwen-plan-q3-first-token.outputs.construction_mode }}",
                      DISPATCHER.read_text(encoding="utf-8"))
        construct_workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("scripts/qwen_reuse_candidate.py", construct_workflow)
        self.assertIn("inputs.mode != 'reuse-candidate'", construct_workflow)

        construct = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("construction_descriptor:", construct)
        self.assertIn("steps.construction_descriptor.outputs.path", construct)
        self.assertIn("steps.construction_descriptor.outputs.sha256", construct)

    def test_candidate_reuse_rebinds_runtime_without_changing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)

            def write(name: str, value: object) -> tuple[Path, str]:
                path = root / name
                path.write_text(json.dumps(value, sort_keys=True) + "\n",
                                encoding="utf-8")
                return path, hashlib.sha256(path.read_bytes()).hexdigest()

            def desc(pair: tuple[Path, str]) -> dict[str, str]:
                return {"path": str(pair[0].resolve()), "sha256": pair[1]}

            capture = write("capture.json", {"capture": True})
            cache = write("cache.json", {"cache": True})
            selection = write("selection.json", {"selection": True})
            intervention = write("intervention.json", {"intervention": True})
            rocmi4 = write("rocmi4.json", {"format": "ROCMI4"})
            fast = write("fast.json", {"format": "FAST"})
            shard = root / "model.gguf"
            shard.write_bytes(b"immutable-q3-artifact")
            shard_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
            builder_revision = "b" * 40
            candidate_id = "q3-stable-artifact"
            format_sha = "f" * 64
            rows = [{"path": str(shard.resolve()), "size_bytes": shard.stat().st_size,
                     "sha256": shard_sha}]
            build = write("build.json", {
                "status": "complete", "mode": "execute",
                "tools": {"ember_revision": builder_revision},
                "quantization_recipe": {
                    "id": "q3-recipe",
                    "selected_mtp_matrix_quant_contract": "Q4_0_ROCMFP4_FAST",
                    "ple_override_preserved": True,
                },
                "output": {"shards": rows},
            })
            attestation = write("attestation.json", {
                "schema": "ember.qwen3.8.candidate-workset-attestation.v1",
                "candidate_id": candidate_id,
                "build_record_sha256": build[1],
                "builder_identity": {
                    "ember_revision": builder_revision,
                    "tensor_format_contract_sha256": format_sha,
                },
                "tensor_format_compatibility_sha256": format_sha,
            })
            old_image = "ghcr.io/otheru-ai/ember@sha256:" + "1" * 64
            construction = write("construction.json", {
                "schema": "ember.qwen3.8.candidate-construction.v1",
                "status": "complete", "publishes": False, "deletes": False,
                "candidate_id": candidate_id, "kind": "intervention",
                "intended_stage": "format", "row_id": "q3-row",
                "intervention_configuration_id": "lambda",
                "quantization_arm": "q3-recipe",
                "mtp_matrix_quant_contract": "Q4_0_ROCMFP4_FAST",
                "runtime_mode": "exact_dequant",
                "builder_revision": builder_revision,
                "runtime_revision": builder_revision,
                "images": {
                    "builder": {"ref": old_image, "digest": "sha256:" + "1" * 64},
                    "runtime": {"release_ref": old_image,
                                "release_digest": "sha256:" + "1" * 64,
                                "dev_ref": old_image,
                                "dev_digest": "sha256:" + "1" * 64,
                                "tensor_format_contract_sha256": format_sha},
                },
                "capture": desc(capture), "stock_capture": None,
                "bf16_cache": desc(cache),
                "shared_companions": {"Q4_0_ROCMI4": desc(rocmi4),
                                      "Q4_0_ROCMFP4_FAST": desc(fast)},
                "selection_plan": desc(selection), "build_record": desc(build),
                "builder_attestation": desc(attestation),
                "intervention_manifest": desc(intervention),
                "artifacts": {"shards": rows,
                              "total_bytes": shard.stat().st_size},
                "v3_candidate_manifest": {"ready": False, "blocked_on": []},
            })
            runtime_revision = "c" * 40
            release = "ghcr.io/otheru-ai/ember@sha256:" + "2" * 64
            dev = "ghcr.io/otheru-ai/ember@sha256:" + "3" * 64
            args = reuse_candidate.parse_args([
                "--construction", str(construction[0]),
                "--construction-sha256", construction[1],
                "--runtime-revision", runtime_revision,
                "--runtime-release-ref", release,
                "--runtime-release-digest", "sha256:" + "2" * 64,
                "--runtime-dev-ref", dev,
                "--runtime-dev-digest", "sha256:" + "3" * 64,
                "--tensor-format-contract-sha256", format_sha,
                "--output", str(root / "rebound.json"),
            ])
            rebound = reuse_candidate.rebind(args)
            self.assertEqual(rebound["builder_revision"], builder_revision)
            self.assertEqual(rebound["runtime_revision"], runtime_revision)
            self.assertEqual(rebound["artifacts"]["shards"], rows)
            self.assertEqual(rebound["images"]["runtime"]["release_ref"], release)
            shard.write_bytes(b"mutated")
            with self.assertRaisesRegex(reuse_candidate.ReuseError,
                                        "candidate shard 0 differs"):
                reuse_candidate.rebind(args)

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
            self.assertEqual(len(result.stdout.splitlines()), 23)
            value["parameters"]["unexpected"] = None
            request.write_text(json.dumps(value) + "\n", encoding="utf-8")
            malformed = subprocess.run(
                [sys.executable, "-", str(request), "prepare-cache"], input=parser,
                text=True, capture_output=True,
            )
            self.assertNotEqual(malformed.returncode, 0)

    def test_companion_handoff_rehashes_every_exposed_child(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        scripts = [script for block in workflow_run_blocks(body)
                   for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", block, re.S)]
        validator = next(script for script in scripts
                         if "companion construction descriptor lifecycle/schema differs" in script)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            subjects = {}
            for name in ("rocmi4", "fast", "plan"):
                path = root / f"{name}.json"
                path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
                subjects[name] = {"path": str(path),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            descriptor = root / "companions.json"
            value = {
                "schema": "ember.qwen3.8.companion-construction.v1",
                "status": "complete", "publishes": False, "deletes": False,
                "cache": {"path": "/cache/manifest.json", "sha256": "0" * 64},
                "selection_plan": subjects["plan"],
                "inventories": {"ROCMI4": subjects["rocmi4"],
                                "ROCMFP4-FAST": subjects["fast"]},
            }
            descriptor.write_text(json.dumps(value) + "\n", encoding="utf-8")
            descriptor_sha = hashlib.sha256(descriptor.read_bytes()).hexdigest()
            valid = subprocess.run(
                [sys.executable, "-", str(descriptor), descriptor_sha], input=validator,
                text=True, capture_output=True,
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(valid.stdout.splitlines(), [
                subjects["rocmi4"]["path"], subjects["rocmi4"]["sha256"],
                subjects["fast"]["path"], subjects["fast"]["sha256"],
                subjects["plan"]["path"], subjects["plan"]["sha256"],
            ])

            Path(subjects["fast"]["path"]).write_text("tampered\n", encoding="utf-8")
            tampered = subprocess.run(
                [sys.executable, "-", str(descriptor), descriptor_sha], input=validator,
                text=True, capture_output=True,
            )
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("ROCmFP4 FAST inventory digest differs", tampered.stderr)

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
