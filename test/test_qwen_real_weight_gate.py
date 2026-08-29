#!/usr/bin/env python3
"""GPU-free safety and methodology tests for the Qwen real-weight gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts/qwen_real_weight_gate.sh"
PROFILE = ROOT / "scripts/profile_gpu.sh"
DISPATCH = ROOT / "scripts/qwen_w4a8_dispatch_evidence.py"
HEX = "1" * 64


def dry_args(out: str = "/tmp/qwen-real-gate-never-created") -> list[str]:
    return [
        "--dry-run", "--image", "candidate:exact",
        "--image-digest", f"sha256:{HEX}",
        "--profile-image", "candidate-dev:exact",
        "--profile-image-digest", f"sha256:{HEX}",
        "--model", "/models/qwen.gguf", "--model-sha256", HEX,
        "--model-build-record", "/models/qwen-quant-build-record.json",
        "--model-build-record-sha256", HEX,
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
        self.assertIn("/models/qwen-quant-build-record.json", result.stdout)
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

    def test_exclusive_restore_and_success_only_hardware_certification(self) -> None:
        body = GATE.read_text(encoding="utf-8")
        self.assertIn("trap cleanup EXIT INT TERM", body)
        self.assertIn('sudo -n "$GPU_LOCK" acquire', body)
        self.assertIn('sudo -n "$GPU_LOCK" release', body)
        self.assertIn('sudo -n "$PRODUCTION" stop', body)
        self.assertIn('sudo -n "$PRODUCTION" mask', body)
        self.assertIn('sudo -n "$PRODUCTION" unmask', body)
        self.assertIn('sudo -n "$PRODUCTION" start', body)
        self.assertIn('sudo -n "$PRODUCTION" is-active', body)
        self.assertIn('curl --fail --silent --max-time 2 "$PRODUCTION_HEALTH"', body)
        restore = body.index("restore_exclusive || die")
        approval = body.index(".hardware-certified.")
        self.assertLess(restore, approval)
        self.assertIn('"hardware_certified": passed', body)
        self.assertIn('"publish_approved": False', body)
        self.assertIn('"text_model_plus_mtp_only" if passed', body)
        self.assertIn('"measurement_only_not_certified"', body)
        self.assertIn('os.link(measured, os.path.join(out, "hardware-certified.json"))', body)
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
        self.assertIn('--server-pid "$TIMING_HOST_PID"', body)
        self.assertIn("--health-endpoint", body)
        self.assertIn("--require-memory-gate", body)
        self.assertIn("runner_rss_gtt_sampler_v1", body)
        self.assertIn('gid="$(stat -c %g -- "$node")"', body)
        self.assertNotIn("--group-add video", body)
        self.assertNotIn("--group-add render", body)
        self.assertIn('docker logs "$CONTAINER" >"$OUT_DIR/timing-server.log"', body)
        self.assertIn("DFLASH_ROCMI4_W4A8_DISPATCH_EVIDENCE=1", body)
        self.assertIn("qwen_w4a8_dispatch_evidence.py", body)
        self.assertIn("--rocmi4-w4a8-iu4", body)
        self.assertIn("check_rocmi4_w4a8_isa.py", body)
        self.assertIn("llvm-objdump --disassemble --mcpu=gfx1151", body)
        self.assertIn('build["build_mode"] != runtime["configured_mmq_mode"]', body)
        self.assertIn('"kernel_runtime": kernel_runtime', body)
        self.assertIn('"kernel_build": kernel_build', body)
        self.assertIn('"timing_kernel_mode": timing_kernel_mode', body)
        self.assertIn("clean timing used {mode}, expected candidate mode", body)
        self.assertIn("candidate_kernel_capability", body)
        self.assertIn('"profile-default-rocmi4":"rocmi4_dense_and_routed"', body)
        self.assertIn('"rocmi4-control":"rocmi4_dense_and_routed"', body)
        self.assertIn('"timing_server_log":', body)
        self.assertIn('"dispatch_server_log":', body)
        frontier = (ROOT / "engine/dflash/qwen4exp/qwen4exp_frontier.cpp").read_text(
            encoding="utf-8")
        self.assertIn("qwen4exp_frontier_run_rocmi4_dispatch_controls", frontier)
        self.assertIn('capability=%s dense_q=1,4,5,16 routed_expert_q=%s', frontier)
        self.assertIn("target_weight=%s", frontier)
        self.assertIn("event=post_compute", frontier)
        dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG EMBER_ROCMI4_W4A8_IU4=OFF", dockerfile)
        self.assertIn("ARG EMBER_ROCMI4_W4A8_IU4_PREPACK=OFF", dockerfile)
        self.assertIn("ARG EMBER_HIP_EXPORT_METRICS=OFF", dockerfile)

    def test_kernel_runtime_evidence_requires_actual_dispatch_controls(self) -> None:
        startup = (
            "ROCmI4 W4A8 IU4: exact experimental MMQ enabled for device 0; "
            "activation_prepack=off\n")
        route = ("[rocmi4-w4a8-dispatch] event=route op={} physical_q={} "
                 "type=Q4_0_ROCMI4 path={} weight={} dst=d\n")
        kernel = ("[rocmi4-w4a8-dispatch] event=kernel "
                  "variant=w4a8_iu4_register_pack op={} physical_q={} "
                  "type=Q4_0_ROCMI4 weight={} device=0 arch=gfx1151\n")
        logical = ("[rocmi4-w4a8-dispatch] event=logical_scope op=dense "
                   "logical_q={} physical_q={} type=Q4_0_ROCMI4 "
                   "execution=completed weight=w\n")
        control = ("[rocmi4-w4a8-dispatch] event=control control_id={} "
                   "op={} logical_q={} target_weight={} phase={}\n")
        post = ("[rocmi4-w4a8-dispatch] event=post_compute control_id={} "
                "op={} logical_q={} physical_q={} target_weight={} "
                "execution=completed\n")
        def bounded(control_id: str, op: str, logical_q: int,
                    physical_q: int, weight: str, *body: str) -> str:
            return "".join((control.format(
                                control_id, op, logical_q, weight, "begin"),
                            *body,
                            post.format(control_id, op, logical_q, physical_q,
                                        weight),
                            control.format(control_id, op, logical_q, weight,
                                           "completed")))
        extra_graph = "".join((
            route.format("dense", 16, "mmq", "shared.weight"),
            kernel.format("dense", 16, "shared.weight"),
            logical.format(16, 16).replace("weight=w", "weight=shared.weight"),
        ))
        complete = startup + "".join((
            bounded("dense-q1", "dense", 1, 1, "dense.weight",
                    route.format("dense", 1, "mmvq", "dense.weight"),
                    logical.format(1, 1).replace("weight=w", "weight=dense.weight")),
            bounded("dense-q4", "dense", 4, 5, "dense.weight",
                    route.format("dense", 5, "mmq", "dense.weight"),
                    kernel.format("dense", 5, "dense.weight"),
                    logical.format(4, 5)),
            bounded("dense-q5", "dense", 5, 5, "dense.weight",
                    route.format("dense", 5, "mmq", "dense.weight"),
                    kernel.format("dense", 5, "dense.weight"),
                    logical.format(5, 5)),
            bounded("dense-q16", "dense", 16, 16, "dense.weight",
                    route.format("dense", 16, "mmq", "dense.weight"),
                    kernel.format("dense", 16, "dense.weight"),
                    logical.format(16, 16)),
            bounded("routed-expert-q1", "routed_expert", 1, 1,
                    "expert.weight", extra_graph,
                    route.format("routed_expert", 1, "mmvq", "expert.weight")),
            bounded("routed-expert-q5", "routed_expert", 5, 5,
                    "expert.weight", extra_graph,
                    route.format("routed_expert", 5, "mmvq", "expert.weight")),
            bounded("routed-expert-q16", "routed_expert", 16, 16,
                    "expert.weight", extra_graph,
                    route.format("routed_expert", 16, "mmq", "expert.weight"),
                    kernel.format("routed_expert", 16, "expert.weight")),
            ("[rocmi4-w4a8-dispatch] event=control_suite "
             "capability=rocmi4_dense_and_routed dense_q=1,4,5,16 "
             "routed_expert_q=1,5,16 "
             "execution=completed\n"),
        ))
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "dispatch.log"
            out_path = Path(temporary) / "evidence.json"
            log_path.write_text(complete, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(DISPATCH), "--log", str(log_path),
                 "--output", str(out_path)],
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["dispatch_confirmation"],
                             "actual_real_weight_dense_and_routed_launches")
            self.assertEqual(evidence["candidate_kernel_capability"],
                             "rocmi4_dense_and_routed")
            self.assertTrue(all(evidence["positive_controls"].values()))
            self.assertTrue(all(evidence["negative_controls"].values()))
            self.assertEqual(evidence["isa_contract"]["opcode"], 69)
            self.assertEqual(len(evidence["ordered_control_ids"]), 7)

            missing = Path(temporary) / "missing.log"
            missing.write_text(complete.replace(
                kernel.format("dense", 16, "dense.weight"), "", 1),
                               encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(DISPATCH), "--log", str(missing),
                 "--output", str(Path(temporary) / "missing.json")],
                text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ordered dispatch evidence mismatch", result.stderr)

            out_of_order = Path(temporary) / "out-of-order.log"
            dense_q4_ordered = "".join((
                route.format("dense", 5, "mmq", "dense.weight"),
                kernel.format("dense", 5, "dense.weight")))
            dense_q4_reversed = "".join((
                kernel.format("dense", 5, "dense.weight"),
                route.format("dense", 5, "mmq", "dense.weight")))
            out_of_order.write_text(
                complete.replace(dense_q4_ordered, dense_q4_reversed, 1),
                encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(DISPATCH), "--log", str(out_of_order),
                 "--output", str(Path(temporary) / "out-of-order.json")],
                text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ordered dispatch evidence mismatch", result.stderr)

            mixed = Path(temporary) / "mixed-control.log"
            mixed.write_text(complete.replace(
                kernel.format("dense", 5, "dense.weight"),
                kernel.format("dense", 16, "dense.weight"), 1),
                encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(DISPATCH), "--log", str(mixed),
                 "--output", str(Path(temporary) / "mixed-control.json")],
                text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ordered dispatch evidence mismatch", result.stderr)

            wrong_weight = Path(temporary) / "wrong-weight.log"
            wrong_weight.write_text(complete.replace(
                kernel.format("dense", 16, "dense.weight"),
                kernel.format("dense", 16, "shared.weight"), 1), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(DISPATCH), "--log", str(wrong_weight),
                 "--output", str(Path(temporary) / "wrong-weight.json")],
                text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("target kernel", result.stderr)

            dense_suite = complete[:complete.index(
                control.format("routed-expert-q1", "routed_expert", 1,
                               "expert.weight", "begin"))]
            dense_suite += ("[rocmi4-w4a8-dispatch] event=control_suite "
                            "capability=rocmi4_dense_only dense_q=1,4,5,16 "
                            "routed_expert_q=none execution=completed\n")
            dense_path = Path(temporary) / "dense-only.log"
            dense_out = Path(temporary) / "dense-only.json"
            dense_path.write_text(dense_suite, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(DISPATCH), "--log", str(dense_path),
                 "--output", str(dense_out)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            dense_value = json.loads(dense_out.read_text(encoding="utf-8"))
            self.assertEqual(dense_value["candidate_kernel_capability"],
                             "rocmi4_dense_only")
            self.assertEqual(len(dense_value["ordered_control_ids"]), 4)

            na_path = Path(temporary) / "not-applicable.log"
            na_out = Path(temporary) / "not-applicable.json"
            na_path.write_text(
                "[rocmi4-w4a8-dispatch] event=control_suite "
                "capability=no_eligible_rocmi4_mmq dense_q=none "
                "routed_expert_q=none execution=completed\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(DISPATCH), "--log", str(na_path),
                 "--output", str(na_out)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            na_value = json.loads(na_out.read_text(encoding="utf-8"))
            self.assertEqual(na_value["candidate_timing_kernel_mode"],
                             "not_applicable_no_eligible_rocmi4_mmq")
            self.assertEqual(na_value["ordered_control_ids"], [])

    def test_kernel_runtime_control_mode_does_not_claim_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "control.log"
            out_path = Path(temporary) / "control.json"
            log_path.write_text("ordinary exact int8 startup\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(DISPATCH), "--log", str(log_path),
                 "--output", str(out_path)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["dispatch_confirmation"],
                             "not_applicable_w4a8_not_configured")

    def test_no_eligible_recipe_binds_control_mode_without_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "control.log"
            out_path = Path(temporary) / "control.json"
            log_path.write_text("ordinary exact int8 startup\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(DISPATCH), "--log", str(log_path),
                 "--output", str(out_path), "--expected-capability",
                 "no_eligible_rocmi4_mmq"], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["candidate_kernel_capability"],
                             "no_eligible_rocmi4_mmq")
            self.assertEqual(evidence["candidate_timing_kernel_mode"],
                             "not_applicable_no_eligible_rocmi4_mmq")
            self.assertEqual(evidence["ordered_control_ids"], [])

            rejected = subprocess.run(
                [sys.executable, str(DISPATCH), "--log", str(log_path),
                 "--output", str(Path(temporary) / "eligible.json"),
                 "--expected-capability", "rocmi4_dense_only"],
                text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("differs from expected", rejected.stderr)

    def test_compile_evidence_triggers_cover_all_production_tu_inputs(self) -> None:
        workflow = (ROOT / ".github/workflows/rocmi4-w4a8-compile-evidence.yml").read_text(
            encoding="utf-8")
        self.assertEqual(workflow.count("'engine/ggml/src/ggml-cuda/**'"), 2)
        self.assertEqual(workflow.count("'engine/ggml/rocmfpx/**'"), 2)
        for path in ("'engine/CMakeLists.txt'", "'engine/ggml/src/CMakeLists.txt'",
                     "'engine/ggml/cmake/**'"):
            self.assertEqual(workflow.count(path), 2)
        ci_doc = (ROOT / "docs/ci.md").read_text(encoding="utf-8")
        self.assertIn(
            "saved-ISA ROCMI4 W4A8 compile-evidence gate is intentionally GitHub-only",
            ci_doc)

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
        integrity = (ROOT / "scripts/qwen_integrity_cache.py").read_text(
            encoding="utf-8")
        self.assertIn('"iflag=direct"', integrity)
        self.assertIn("IntegrityCache", gate)
        self.assertIn('inventory["shards"]', gate)
        self.assertIn("model-inventory.json", gate)
        self.assertIn("qwen-quant-build-record.json", gate)
        self.assertIn("DFLASH_QWEN_MTP=/gate/mtp.gguf", gate)
        self.assertIn("--kv-cache-dir /gate/cache", gate)
        self.assertIn("--mtp PATH", profile)
        self.assertIn("DFLASH_QWEN_MTP=/pmtp/", profile)
        self.assertIn("--draft and --mtp are mutually exclusive", profile)
        self.assertIn('--pmc "$counter"', profile)

    def test_approval_embeds_measured_memory_and_hard_fit(self) -> None:
        body = GATE.read_text(encoding="utf-8")
        self.assertIn('"memory": memory["hard_fit"]', body)
        self.assertIn('"resources": memory["resources"]', body)
        self.assertIn('"memory-evidence.json"', body)
        self.assertIn('"model_inventory":', body)
        self.assertIn('"quant_build_record":', body)

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
