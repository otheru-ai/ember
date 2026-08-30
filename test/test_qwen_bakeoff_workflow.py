#!/usr/bin/env python3
"""Static/adversarial contracts for the dedicated gfx1151 Qwen bakeoff."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/qwen-gfx1151-bakeoff.yml"
TARGET_GATE = ROOT / "scripts/qwen_target_only_gate.sh"
BALANCED_RUNNER = ROOT / "scripts/qwen_balanced_confirmation.sh"
BALANCED_HELPER = ROOT / "scripts/qwen_balanced_confirmation.py"
BALANCED_SLOT = ROOT / "scripts/qwen_balanced_slot.py"
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
    def test_balanced_runner_dry_run_and_syntax_are_side_effect_free(self) -> None:
        subprocess.run(["bash", "-n", str(BALANCED_RUNNER)], check=True)
        with tempfile.TemporaryDirectory() as temporary:
            for tool in ("docker", "curl", "sudo", "dd", "python3", "stat"):
                path = Path(temporary) / tool
                path.write_text(f"#!/bin/sh\necho FORBIDDEN:{tool} >&2\nexit 97\n")
                path.chmod(0o755)
            env = os.environ | {"PATH": temporary + os.pathsep + os.environ["PATH"]}
            result = subprocess.run([
                "bash", str(BALANCED_RUNNER), "--dry-run",
                "--plan", "/evidence/plan.json", "--plan-sha256", HEX,
                "--accumulator", "/evidence/format.json",
                "--accumulator-sha256", HEX,
                "--evidence-root", "/evidence", "--image", "candidate:exact",
                "--image-digest", f"sha256:{HEX}",
                "--runtime-revision", "1" * 40,
                "--engine-binary-sha256", HEX,
                "--tensor-format-contract-sha256", HEX,
                "--out-dir", "/tmp/qwen-balanced-never-created",
            ], cwd=ROOT, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("FORBIDDEN", result.stderr)
        self.assertIn("ABBAAB", result.stdout)
        self.assertIn("one exact 2074-token prefill + one 256-token decode", result.stdout)
        self.assertIn("clean; profiling/counters forbidden", result.stdout)

    def test_balanced_slot_and_assembler_emit_the_selector_schema(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import qwen_balanced_slot as slot
        import qwen_bakeoff as qb
        prefill = {"ok": True, "group": "prefill-2048",
                   "evaluated_prefill_tokens": 2074,
                   "restored_prefix": 0,
                   "prefill_tps_rounding_consistent": True,
                   "prefill_tokens_per_second": 413.25}
        decode = {"ok": True, "group": "decode-256", "completion_tokens": 256,
                  "decode_tps_rounding_consistent": True,
                  "decode_tokens_per_second": 40.25, "spec_ran": True,
                  "accept_rate": 0.981}
        process = {"schema": qb.BALANCED_PROCESS_SCHEMA, "run_index": 0,
                   "arm_id": "arm-a"}
        recipe = {"workload_id": "pair-0", "marker": "QBC0",
                  "prefill_generator": "benchmark.make_prefill_prompt.v1",
                  "decode_generator": "benchmark.make_decode_prompt.v1",
                  "evaluated_prefill_tokens": 2074, "completion_tokens": 256}
        workload = {**recipe, "recipe_sha256": qb.canonical_sha256(recipe)}
        row = slot.make_run(0, "arm-a", process, workload, 2040,
                            "paired prefill", "paired decode", prefill, decode)
        self.assertEqual(row["evaluated_prefill_tokens"], 2074)
        self.assertEqual(row["completion_tokens"], 256)
        with self.assertRaisesRegex(ValueError, "exactly 256"):
            slot.make_run(0, "arm-a", process, workload, 2040,
                          "paired prefill", "paired decode", prefill,
                          decode | {"completion_tokens": 255})

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runner = root / "runner.json"
            order = ["arm-a", "arm-b", "arm-b", "arm-a", "arm-a", "arm-b"]
            workload_order = ["pair-0", "pair-0", "pair-1", "pair-1",
                              "pair-2", "pair-2"]
            workloads = []
            for index in range(3):
                value = {"workload_id": f"pair-{index}", "marker": f"QBC{index}",
                         "prefill_generator": "benchmark.make_prefill_prompt.v1",
                         "decode_generator": "benchmark.make_decode_prompt.v1",
                         "evaluated_prefill_tokens": 2074, "completion_tokens": 256}
                workloads.append({**value, "recipe_sha256": qb.canonical_sha256(value)})
            runtime = {"ember_revision": "1" * 40, "container_digest": f"sha256:{HEX}",
                       "engine_binary_sha256": "2" * 64,
                       "tensor_format_contract_sha256": "3" * 64}
            bindings = []
            for arm, digit in (("arm-a", "a"), ("arm-b", "b")):
                capability = ("rocmi4_dense_and_routed" if arm == "arm-a" else
                              "no_eligible_rocmi4_mmq")
                bindings.append({"arm_id": arm, "candidate_id": f"candidate-{arm}",
                                 "candidate_kernel_capability": capability,
                                 "model_sha256": digit * 64,
                                 "model_inventory_sha256": "4" * 64,
                                 "companion_inventory_sha256": "5" * 64,
                                 "mtp_sha256": "6" * 64, "mtp_depth": 3,
                                 "candidate_binding": {"sha256": digit * 64}})
            runner.write_text(json.dumps({
                "schema": "ember.qwen3.8.balanced-confirmation-runner-plan.v1",
                "confirmation_plan": {"run_order": order,
                                      "workload_order": workload_order,
                                      "workloads": workloads,
                                      "finalists": [{
                                          "arm_id": row["arm_id"],
                                          "candidate_kernel_capability": row[
                                              "candidate_kernel_capability"],
                                      } for row in bindings]},
                "runtime_identity": runtime, "bindings": bindings,
            }), encoding="utf-8")
            slots = []
            for index, arm in enumerate(order):
                path = root / f"slot-{index}.json"
                binding = next(item for item in bindings if item["arm_id"] == arm)
                w4a8 = (binding["candidate_kernel_capability"] !=
                        "no_eligible_rocmi4_mmq")
                process = {"schema": qb.BALANCED_PROCESS_SCHEMA, "run_index": index,
                           "arm_id": arm, "candidate_id": binding["candidate_id"],
                           "container_id": f"{index + 10:064x}", "host_pid": 100 + index,
                           "proc_start_ticks": 1000 + index,
                           **runtime,
                           "candidate_kernel_capability": binding[
                               "candidate_kernel_capability"],
                           "rocmi4_w4a8_iu4_requested": w4a8,
                           "candidate_binding_sha256": binding[
                               "candidate_binding"]["sha256"],
                           "model_first_shard_sha256": binding["model_sha256"],
                           "model_inventory_sha256": binding["model_inventory_sha256"],
                           "companion_inventory_sha256": binding[
                               "companion_inventory_sha256"],
                           "mtp_sha256": binding["mtp_sha256"], "mtp_depth": 3}
                workload = next(item for item in workloads
                                if item["workload_id"] == workload_order[index])
                pair = index // 2
                path.write_text(json.dumps({
                    "run_index": index, "arm_id": arm,
                    "process_instance": process,
                    "process_instance_sha256": qb.canonical_sha256(process),
                    "workload_id": workload["workload_id"],
                    "workload_recipe_sha256": workload["recipe_sha256"],
                    "prefill_prompt_sha256": f"{pair + 20:064x}",
                    "decode_prompt_sha256": f"{pair + 30:064x}",
                    "calibrated_prefill_words": 2040 + pair,
                    "evaluated_prefill_tokens": 2074, "completion_tokens": 256,
                    "prefill_tps": 413.0 + index, "decode_tps": 40.0 + index,
                    "spec_ran": True, "accept_rate": 0.981,
                    "startup_kernel_mode": (
                        "w4a8_iu4_register_pack" if w4a8 else
                        "not_applicable_no_eligible_rocmi4_mmq"),
                    "startup_log_sha256": f"{index + 40:064x}",
                }), encoding="utf-8")
                slots.extend(["--slot", str(path)])
            output = root / "confirmation.json"
            result = subprocess.run([
                sys.executable, str(BALANCED_HELPER), "assemble",
                "--runner-plan", str(runner), *slots, "--output", str(output),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["schema"], "ember.qwen3.8.balanced-confirmation.v2")
            self.assertEqual([item["arm_id"] for item in value["runs"]], order)

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
        self.assertIn('row["stage"] in {"format", "mtp-depth", "final"}',
                      TARGET_GATE.read_text(encoding="utf-8"))
        gate = TARGET_GATE.read_text(encoding="utf-8")
        self.assertIn(
            "--prefill-words 2048 --decode-tokens 256 --gap-secs 3", gate)
        self.assertIn('gid="$(stat -c %g -- "$node")"', gate)
        self.assertNotIn("--group-add video", gate)
        self.assertNotIn("--group-add render", gate)
        self.assertIn(
            '--counter-calibration "$OUT_DIR/profile/counter-calibration.json"',
            gate,
        )
        self.assertIn('"profile_report": {"path": "profile/report.json"', gate)
        self.assertIn("from qwen_integrity_cache import IntegrityCache", gate)
        self.assertIn('"mtp", "vision_mmproj", "vision_vocab"', gate)
        self.assertGreaterEqual(gate.count("defer_content=True"), 3)
        self.assertLess(gate.index('sudo -n "$GPU_LOCK" acquire'),
                        gate.index("from qwen_integrity_cache import IntegrityCache"))

    def test_workflow_yaml_shell_and_python_heredocs_parse(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        ruby = shutil.which("ruby")
        if ruby:
            subprocess.run([ruby, "-e", "require 'yaml'; YAML.parse_file(ARGV[0])",
                            str(WORKFLOW)], check=True)
        blocks = workflow_run_blocks(body)
        self.assertGreaterEqual(len(blocks), 10)
        for index, block in enumerate(blocks):
            self.assertLessEqual(len(block.encode("utf-8")), 21_000,
                                 f"run block {index} exceeds GitHub's expression limit")
            neutral = re.sub(r"\$\{\{.*?\}\}", "github-expression", block)
            result = subprocess.run(["bash", "-n"], input=neutral, text=True,
                                    capture_output=True)
            self.assertEqual(result.returncode, 0, f"run block {index}: {result.stderr}")
            for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", neutral, re.S):
                compile(script, f"workflow-run-{index}-heredoc.py", "exec")

    def test_workflow_uses_v3_serial_immutable_workset_contract(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", body)
        self.assertIn("workflow_call:", body)
        called = re.search(
            r"workflow_call:\n    inputs:\n(.*?)\n  workflow_dispatch:", body, re.S)
        self.assertIsNotNone(called)
        self.assertEqual(re.findall(r"^      ([a-z0-9_]+):$", called.group(1), re.M),
                         ["commit_sha", "candidate_manifest",
                          "candidate_manifest_sha256", "phase", "phase_request",
                          "phase_request_sha256"])
        dispatch = re.search(r"workflow_dispatch:\n    inputs:\n(.*?)\npermissions:", body, re.S)
        self.assertIsNotNone(dispatch)
        inputs = re.findall(r"^      ([a-z0-9_]+):$", dispatch.group(1), re.M)
        self.assertEqual(inputs, ["commit_sha", "candidate_manifest",
                                  "candidate_manifest_sha256", "phase",
                                  "phase_request", "phase_request_sha256"])
        self.assertLessEqual(len(inputs), 10)
        self.assertIn(
            "contains(github.workflow_ref, '/.github/workflows/gfx1151-certify.yml@')",
            body)
        self.assertIn("qwen-bakeoff-called-{0}", body)
        self.assertIn("ember.qwen3.8.sequential-bakeoff-phase-request.v1", body)
        self.assertIn("runs-on: [self-hosted, linux, x64, gfx1151]", body)
        self.assertIn("one-candidate-per-dispatch is required", body)
        self.assertIn("ember.qwen3.8.sequential-bakeoff-candidate.v3", body)
        self.assertIn("BF16 cache is not content-addressed beneath the workset", body)
        self.assertIn('"Q4_0_ROCMI4", "Q4_0_ROCMFP4_FAST"', body)
        self.assertIn("shared BF16 mmproj inventory differs", body)
        self.assertIn("candidate is not the next row in the canonical serial phase order", body)
        self.assertIn("candidate-workset-attestation.v1", body)
        self.assertIn("tensor_format_compatibility_sha256", body)
        self.assertIn("quantizer_tool_sha256", body)
        self.assertIn("engine_binary_sha256", body)
        self.assertIn('c.get("intervention_manifest_sha256") != "0"*64', body)
        self.assertIn('declared_capture = c.get("stock_capture")', body)
        self.assertIn('row["intervention_manifest_sha256"] = "0" * 64', body)
        for field in ("candidate_id", "model_inventory_sha256",
                      "tensor_format_compatibility_sha256", "artifact_bytes"):
            self.assertIn(f'"{field}"', body)

    def test_embedded_phase_request_parser_is_exact_and_phase_scoped(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        scripts = [script for block in workflow_run_blocks(body)
                   for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", block, re.S)]
        parser = next(script for script in scripts
                      if "sequential-bakeoff-phase-request.v1" in script)
        with tempfile.TemporaryDirectory() as raw:
            request = Path(raw) / "phase-request.json"
            sweep = {"schema": "ember.qwen3.8.sequential-bakeoff-phase-request.v1",
                     "phase": "sweep", "results_accumulator": None,
                     "prior_ledger": None, "publishes": False, "deletes": False}
            request.write_text(json.dumps(sweep) + "\n", encoding="utf-8")
            valid = subprocess.run(
                [sys.executable, "-", str(request), "sweep"], input=parser,
                text=True, capture_output=True,
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(len(valid.stdout.splitlines()), 8)
            sweep["prior_ledger"] = {"subject": {
                "path": "/tmp/ledger.json", "sha256": HEX,
                "schema": "ember.qwen3.8.sequential-bakeoff-ledger.v3"},
                "bundle": {"path": "/tmp/ledger.sigstore.json", "sha256": HEX}}
            request.write_text(json.dumps(sweep) + "\n", encoding="utf-8")
            forbidden = subprocess.run(
                [sys.executable, "-", str(request), "sweep"], input=parser,
                text=True, capture_output=True,
            )
            self.assertNotEqual(forbidden.returncode, 0)

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
        self.assertIn("final-confirmation is not the exact attested MTP-depth winner", body)
        self.assertIn('c.get("id") != "final-confirmation"', body)
        self.assertIn("selection-only corpus", body)
        self.assertIn("options: [sweep, format, mtp-depth, final]", body)
        self.assertIn('plan["mtp_depth_configurations"]', body)

    def test_every_retained_decision_is_externally_attested_and_verified(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("id-token: write", body)
        self.assertIn("attestations: write", body)
        self.assertIn("artifact-metadata: write", body)
        self.assertGreaterEqual(body.count(f"actions/attest@{ATTEST_SHA}"), 3)
        self.assertGreaterEqual(body.count("gh attestation verify"), 4)
        self.assertIn("ember.qwen3.8.sequential-bakeoff-result.v4", body)
        self.assertIn("ember.qwen3.8.candidate-assessment.v2", body)
        self.assertIn("ember.qwen3.8.sequential-bakeoff-accumulator.v2", body)
        self.assertIn('"contains_raw_measurements":False', body)
        self.assertIn("ember.qwen3.8.sequential-bakeoff-ledger.v3", body)
        self.assertIn('"mtp_depth":c["mtp_depth"]', body)
        self.assertIn("Retain and verify completed phase ledger", body)
        self.assertIn("subject-path: ${{ env.QWEN_ASSESSMENT }}", body)
        self.assertIn("QWEN_CURRENT_ACCUMULATOR_SHA256", body)
        self.assertIn("QWEN_LEDGER_SHA256", body)
        self.assertIn("Attested compact accumulator SHA-256", body)
        self.assertIn("Accumulator attestation bundle SHA-256", body)
        self.assertIn("Completed phase ledger SHA-256", body)
        self.assertIn("Ledger attestation bundle SHA-256", body)

    def test_measure_attest_authorize_then_retire_order_is_fail_closed(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        target = body.index("scripts/qwen_target_only_gate.sh")
        mtp = body.index("scripts/qwen_real_weight_gate.sh")
        assess = body.index("--stage assess")
        attest = body.index("GitHub-attest candidate assessment")
        accumulator_attest = body.index("GitHub-attest compact accumulator")
        authority = body.index("authorize-rolling-retention")
        retire = body.index("retire-reconstructable")
        self.assertLess(target, mtp)
        self.assertLess(mtp, assess)
        self.assertLess(assess, attest)
        self.assertLess(attest, accumulator_attest)
        self.assertLess(accumulator_attest, authority)
        self.assertLess(authority, retire)
        self.assertNotIn("scripts/qwen_candidate_builder.py delete-loser", body)
        self.assertIn("QWEN_RETENTION_AUTHORITY_SHA256", body)
        self.assertIn("every eviction is reconstructable", body)
        self.assertNotIn("mapfile -t retirement < <(", body)
        self.assertGreaterEqual(body.count('retirement_output="$(python3'), 2)
        self.assertIn('2 * expected_retirements', body)
        self.assertIn('$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT.json', body)
        self.assertIn("--measurement-only", body)

    def test_format_boundary_runs_balanced_confirmation_before_selection(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        runner = body.index("scripts/qwen_balanced_confirmation.sh")
        merge = body.index('value["balanced_confirmation"]')
        select = body.index('args=(--plan "$EFFECTIVE_PLAN" --results "$selection_results"')
        ledger_verify = body.index('gh attestation verify "$QWEN_LEDGER"')
        sealed = body.index("authorize-sealed-retention")
        sealed_retire = body.index("retire-reconstructable", sealed)
        self.assertLess(runner, merge)
        self.assertLess(merge, select)
        self.assertLess(select, ledger_verify)
        self.assertLess(ledger_verify, sealed)
        self.assertLess(sealed, sealed_retire)
        self.assertIn('if [[ "$QWEN_BAKEOFF_PHASE" = format ]]', body)
        self.assertIn('--accumulator-sha256 "$(sha256sum', body)
        self.assertIn('--engine-binary-sha256 "$RUNTIME_ENGINE_SHA256"', body)
        runner_body = BALANCED_RUNNER.read_text(encoding="utf-8")
        self.assertIn("for index in $(seq 0 5)", runner_body)
        self.assertIn('remove_container', runner_body)
        self.assertIn('"ember.qwen3.8.fresh-server-process.v2"', runner_body)
        self.assertIn('-e DFLASH_QWEN_MTP=/gate/mtp.gguf', runner_body)
        self.assertIn('DFLASH_QWEN_MTP_DEPTH=$mtp_depth', runner_body)
        self.assertIn("from qwen_integrity_cache import IntegrityCache", runner_body)
        self.assertIn("if path in seen: continue", runner_body)
        self.assertIn("--integrity-cache", runner_body)
        self.assertLess(runner_body.index('sudo -n "$GPU_LOCK" acquire'),
                        runner_body.index("from qwen_integrity_cache import IntegrityCache"))
        self.assertIn('--prefix-cache-slots 1', runner_body)
        self.assertIn('process_instance_sha256', BALANCED_SLOT.read_text(encoding="utf-8"))
        slot_body = BALANCED_SLOT.read_text(encoding="utf-8")
        self.assertIn('marker=args.workload_marker', slot_body)
        self.assertIn('prefix-cache isolation request failed', slot_body)
        self.assertIn('0 < accept_rate < 1', slot_body)
        self.assertNotIn("profile_gpu.sh", runner_body)
        self.assertNotIn("rocprof", runner_body)
        self.assertLess(runner_body.index('sudo -n "$GPU_LOCK" acquire'),
                        runner_body.index("for index in $(seq 0 5)"))
        self.assertIn('restore_exclusive || die', runner_body)

    def test_nested_gates_exclusively_own_gpu_and_production_lifecycle(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("Restore production and release exclusive GPU ownership", body)
        self.assertNotIn("sudo -n /usr/local/sbin/ember-gpu-lock acquire", body)
        self.assertNotIn("sudo -n /usr/local/sbin/ember-gpu-lock release", body)
        self.assertNotIn("sudo -n /usr/local/sbin/ember-cert-production stop", body)
        self.assertNotIn("sudo -n /usr/local/sbin/ember-cert-production mask", body)
        self.assertNotIn("sudo -n /usr/local/sbin/ember-cert-production unmask", body)
        self.assertNotIn("sudo -n /usr/local/sbin/ember-cert-production start", body)
        self.assertNotIn("production_was_active", body)
        self.assertNotIn("armed=yes", body)
        self.assertIn("ember-cert-production is-active", body)
        self.assertIn("http://127.0.0.1:8000/health", body)
        for forbidden in ("docker push", "huggingface-cli", "hf upload",
                          "actions/upload-artifact", "gh release"):
            self.assertNotIn(forbidden, body)

    def test_mtp_depth_disposition_cannot_delete_shared_selected_bytes(self) -> None:
        body = WORKFLOW.read_text(encoding="utf-8")
        scripts = [script for block in workflow_run_blocks(body)
                   for script in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", block, re.S)]
        disposition = next(script for script in scripts
                           if "without_depth" in script and "MTP-depth artifact" in script)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            assessment = root / "assessment.json"
            prior = root / "format-ledger.json"
            shared = {"candidate_id": "format-winner", "model_inventory_sha256": HEX,
                      "build_record_sha256": "2" * 64}
            assessment.write_text(json.dumps({
                "artifact_identity": shared | {"mtp_depth": 4},
                "observed_decision": {"passes": False},
            }), encoding="utf-8")
            prior.write_text(json.dumps({
                "selected_artifact_identity": shared | {"mtp_depth": 3},
            }), encoding="utf-8")
            retained = subprocess.run(
                [sys.executable, "-", str(assessment), str(prior)],
                input=disposition, text=True, capture_output=True,
            )
            self.assertEqual(retained.returncode, 0, retained.stderr)
            self.assertEqual(retained.stdout.strip(), "")

            prior.write_text(json.dumps({
                "selected_artifact_identity": shared | {
                    "mtp_depth": 3, "model_inventory_sha256": "3" * 64},
            }), encoding="utf-8")
            rejected = subprocess.run(
                [sys.executable, "-", str(assessment), str(prior)],
                input=disposition, text=True, capture_output=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("not the shared format-selected artifact", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
