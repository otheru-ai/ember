#!/usr/bin/env python3
"""GPU-free safety and methodology tests for the Qwen real-weight gate."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts/qwen_real_weight_gate.sh"
PROFILE = ROOT / "scripts/profile_gpu.sh"
DISPATCH = ROOT / "scripts/qwen_w4a8_dispatch_evidence.py"
HEX = "1" * 64
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_quant_comparison as quant_comparison  # noqa: E402
import qwen_builder_change_scope as builder_scope  # noqa: E402


def timing_rows(prefill: tuple[float, float, float] = (411.0, 412.0, 413.0),
                decode: tuple[float, float, float] = (39.5, 40.0, 40.5)) -> list[dict]:
    requests = []
    for repeat, rate in enumerate(prefill, 1):
        milliseconds = 2074 * 1000.0 / rate
        requests.append({
            "kind": "request", "group": "prefill-2048", "repeat": repeat,
            "ok": True, "evaluated_prefill_tokens": 2074,
            "prefill_ms": milliseconds,
            "declared_prefill_tokens_per_second": round(rate, 1),
            "prefill_tokens_per_second": 2074 * 1000.0 / milliseconds,
            "prefill_tps_rounding_consistent": True,
        })
    decode_rows = []
    for repeat, rate in enumerate(decode, 1):
        milliseconds = 256 * 1000.0 / rate
        decode_rows.append({
            "kind": "request", "group": "decode-256", "repeat": repeat,
            "ok": True, "completion_tokens": 256, "decode_ms": milliseconds,
            "declared_decode_tokens_per_second": round(rate, 2),
            "decode_tokens_per_second": 256 * 1000.0 / milliseconds,
            "decode_tps_rounding_consistent": True, "spec_ran": True,
            "accept_rate": 0.75, "spec_cycles": 10,
            "spec_provider_age_ms": 1.0, "spec_provider_block_ms": 2.0,
            "spec_head_ms": 3.0, "spec_verify_ms": 4.0,
        })
    requests.extend(decode_rows)
    speculation = {
        "samples": 3, "timing_complete": True, "cycles": 30,
        "accept_rate_mean": 0.75,
        "spec_provider_age_ms_total": 3.0,
        "spec_provider_age_ms_per_cycle": 0.1,
        "spec_provider_block_ms_total": 6.0,
        "spec_provider_block_ms_per_cycle": 0.2,
        "spec_head_ms_total": 9.0, "spec_head_ms_per_cycle": 0.3,
        "spec_verify_ms_total": 12.0, "spec_verify_ms_per_cycle": 0.4,
    }
    resources = {
        "server_host_pid": 123, "measured_peak_rss_bytes": 70_000_000_000,
        "measured_peak_gtt_bytes": 5_000_000_000,
        "measured_peak_uma_bytes": 75_000_000_000,
    }
    return [{
        "kind": "metadata", "server_pid_source": "explicit", "container_pid": 123,
        "protocol": quant_comparison.bakeoff.BENCHMARK_MODULE.QWEN_HARD_GATE_PROTOCOL,
    }, *requests, {
        "kind": "summary", "resources": resources, "memory_gate": {"passed": True},
        "hard_gate": {"passed": True}, "groups": {"decode-256": {
            "speculation": speculation}},
        "prefill_calibration": {"target_prompt_tokens": 2074,
                                "selected_words": 2040, "attempts": []},
        "decode_calibration": {"target_completion_tokens": 256,
                               "selected_marker": "B", "attempts": []},
    }]


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

    def run_differential_timing_parser(
            self, report: dict, directory: Path) -> subprocess.CompletedProcess[str]:
        body = GATE.read_text(encoding="utf-8")
        marker = '"$OUT_DIR/differential-decode-comparison.json" <<\'PY\''
        start = body.index("\n", body.index(marker)) + 1
        end = body.index("\nPY\n", start)
        source = directory / "differential.json"
        output = directory / "differential-decode-comparison.json"
        source.write_text(json.dumps(report), encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-", str(source), str(output)],
            input=body[start:end], cwd=ROOT, text=True, capture_output=True)

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
        self.assertIn("--calibrate-qwen-shapes", body)
        self.assertIn("--require-gate", body)
        self.assertNotIn("--prefix-cache-slots 0", body)
        self.assertIn('row.get("spec_ran") is True', body)
        self.assertIn('prefill.get("checked") and prefill.get("accepted")', body)
        self.assertIn('prefill.get("tv_checked") and prefill.get("tv_within_bound")', body)
        self.assertIn("0.0 <= rate < 1.0", body)
        self.assertIn('report.get("baseline_tokens") != 64', body)
        self.assertIn('positive_finite(ar, "decode_seconds")', body)
        self.assertIn('positive_finite(spec, "fresh_decode_seconds")', body)
        self.assertIn('"warm_speedup_vs_ar": fresh_spec_tps / ar_tps', body)
        self.assertIn('"purpose": "same_process_diagnostic_not_hard_gate_timing"', body)
        self.assertIn('"differential_decode": differential_decode', body)
        self.assertIn('"path": "differential-decode-comparison.json"', body)
        self.assertIn('"spec_verify_ms"', body)
        self.assertIn('"speculation": speculation', body)
        self.assertIn('"speculation": memory["speculation"]', body)
        differential = body.index('log "running q=1/native-batch snapshot differential"')
        trace = body.index("-e EMBER_TRACE_TOKENS=1")
        timing = body.index('python3 "$BENCHMARK"')
        timing_start = body.index('log "starting exact candidate for clean hard-gate timing"')
        profile = body.index('"$PROFILE_SCRIPT" --no-quiesce')
        self.assertLess(differential, trace)
        self.assertLess(trace, timing_start)
        self.assertEqual(body.count("-e EMBER_TRACE_TOKENS=1"), 1)
        self.assertIn("--validate-tokens 64", body[differential:timing_start])
        self.assertLess(timing, profile)
        self.assertIn("never timing evidence", body)
        self.assertIn('--image "$PROFILE_IMAGE"', body)
        self.assertIn(
            "--prefill-words 2048 --decode-tokens 256 --gap-secs 3", body)
        self.assertIn('--counter-calibration "$OUT_DIR/profile/counter-calibration.json"', body)
        self.assertIn('"profile_report": {"path": "profile/report.json"', body)
        self.assertIn('"counter_calibration": {"path": "profile/counter-calibration.json"', body)
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

    def test_differential_timing_parser_records_warm_ar_mtp_comparison(self) -> None:
        report = {
            "ok": True, "requested_tokens": 64, "snapshot_ok": True,
            "baseline_tokens": 64,
            "ar": {"tokens": 64, "decode_seconds": 4.0,
                   "decode_tokens_per_second": 16.0},
            "prefill": {
                "checked": True, "exact": True, "accepted": True,
                "tv_checked": True, "tv_within_bound": True,
            },
            "spec": {"checked": True, "exact": True, "tokens": 64,
                     "accept_rate": 0.75,
                     "restored_decode_seconds": 3.0,
                     "fresh_decode_seconds": 2.0,
                     "fresh_decode_tokens_per_second": 32.0},
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = self.run_differential_timing_parser(report, directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            comparison = json.loads((
                directory / "differential-decode-comparison.json").read_text(
                    encoding="utf-8"))
        self.assertEqual(
            comparison["schema"],
            "ember.qwen3.8.differential-decode-comparison.v1")
        self.assertEqual(comparison["mtp"]["warm_speedup_vs_ar"], 2.0)
        self.assertEqual(
            comparison["purpose"],
            "same_process_diagnostic_not_hard_gate_timing")

    def test_differential_timing_parser_rejects_missing_duration(self) -> None:
        report = {
            "ok": True, "requested_tokens": 64, "snapshot_ok": True,
            "baseline_tokens": 64,
            "ar": {"tokens": 64, "decode_seconds": 4.0,
                   "decode_tokens_per_second": 16.0},
            "prefill": {
                "checked": True, "exact": True, "accepted": True,
                "tv_checked": True, "tv_within_bound": True,
            },
            "spec": {"checked": True, "exact": True, "tokens": 64,
                     "accept_rate": 0.75,
                     "restored_decode_seconds": 3.0,
                     "fresh_decode_tokens_per_second": 32.0},
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_differential_timing_parser(
                report, Path(temporary))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fresh_decode_seconds is not positive", result.stderr)

    def test_hardware_gate_seals_a_matched_comparison_contract(self) -> None:
        body = GATE.read_text(encoding="utf-8")
        derive = body.index('"$QUANT_COMPARISON" derive-contract')
        profile = body.index('"$PROFILE_SCRIPT" --no-quiesce')
        self.assertLess(derive, profile)
        self.assertIn('"benchmark_contract": benchmark_contract', body)
        self.assertIn('"benchmark_contract": {"path": "benchmark-contract.json"', body)

    def test_q3_reuse_classifier_ignores_only_retention_bookkeeping(self) -> None:
        plan = (ROOT / ".github/workflows/qwen-q3-first-token-plan.yml").read_text(
            encoding="utf-8")
        self.assertIn("scripts/qwen_builder_change_scope.py", plan)
        self.assertNotIn(
            "scripts/qwen_candidate_builder.py scripts/qwen_quantize.py", plan)
        baseline = '''
RECONSTRUCTABLE_RETIREMENT_SCHEMA = "v1"
RECONSTRUCTABLE_RETIREMENT_COMPLETE_SCHEMA = "v1"
BUILD_FORMAT = "Q3"
def build_candidate():
    return BUILD_FORMAT
def _validate_reconstruction_intervention():
    return None
def _record_reconstruction_contract():
    return None
def retire_reconstructable():
    return RECONSTRUCTABLE_RETIREMENT_SCHEMA
def restore_reconstructable():
    return RECONSTRUCTABLE_RETIREMENT_COMPLETE_SCHEMA
'''
        retention_only = baseline.replace('"v1"', '"v2"').replace(
            "def retire_reconstructable():\n",
            "def added_retention_helper():\n"
            "    return None\n"
            "def retire_reconstructable():\n")
        # A new helper is conservative unless it is explicitly classified.
        self.assertNotEqual(builder_scope.content_fingerprint(baseline),
                            builder_scope.content_fingerprint(retention_only))
        retention_only = baseline.replace('"v1"', '"v2"').replace(
            "    return None\ndef _record_reconstruction_contract():",
            "    return {'retained': True}\ndef _record_reconstruction_contract():")
        self.assertEqual(builder_scope.content_fingerprint(baseline),
                         builder_scope.content_fingerprint(retention_only))
        build_change = baseline.replace('BUILD_FORMAT = "Q3"',
                                        'BUILD_FORMAT = "IU4"')
        self.assertNotEqual(builder_scope.content_fingerprint(baseline),
                            builder_scope.content_fingerprint(build_change))
        retention_used_by_build = baseline.replace(
            "    return BUILD_FORMAT\ndef _validate_reconstruction_intervention():",
            "    return RECONSTRUCTABLE_RETIREMENT_SCHEMA\n"
            "def _validate_reconstruction_intervention():")
        changed_retention_used_by_build = retention_used_by_build.replace(
            'RECONSTRUCTABLE_RETIREMENT_SCHEMA = "v1"',
            'RECONSTRUCTABLE_RETIREMENT_SCHEMA = "v2"')
        self.assertNotEqual(
            builder_scope.content_fingerprint(retention_used_by_build),
            builder_scope.content_fingerprint(changed_retention_used_by_build))

    def test_benchmark_contract_binds_the_exact_workload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            timing = Path(temporary) / "timing.jsonl"
            timing.write_text("".join(json.dumps(row) + "\n" for row in timing_rows()),
                              encoding="utf-8")
            contract = quant_comparison.derive_contract(
                timing, ROOT / "scripts/bench/benchmark.py", GATE)
            self.assertEqual(contract["schema"], quant_comparison.CONTRACT_SCHEMA)
            identity = contract["identity"]
            self.assertEqual(identity["evaluated_prefill_tokens"], [2074] * 3)
            self.assertEqual(identity["completion_tokens"], [256] * 3)
            self.assertEqual(identity["samples_per_group"], 3)
            self.assertEqual(len(identity["prefill_prompt_sha256"]), 3)
            expected_decode = quant_comparison.bakeoff.BENCHMARK_MODULE.make_decode_prompt("B")
            self.assertEqual(
                identity["decode_prompt_sha256"],
                hashlib.sha256(expected_decode.encode("utf-8")).hexdigest())
            self.assertEqual(contract["identity_sha256"],
                             quant_comparison.canonical_sha256(identity))
            with self.assertRaisesRegex(
                    quant_comparison.ComparisonError, "must be absolute paths"):
                quant_comparison.derive_contract(
                    Path("timing.jsonl"), ROOT / "scripts/bench/benchmark.py", GATE)

            broken = timing_rows()
            broken[-1]["decode_calibration"]["selected_marker"] = "two"
            timing.write_text("".join(json.dumps(row) + "\n" for row in broken),
                              encoding="utf-8")
            with self.assertRaisesRegex(
                    quant_comparison.ComparisonError, "calibration is incomplete"):
                quant_comparison.derive_contract(
                    timing, ROOT / "scripts/bench/benchmark.py", GATE)

    def test_q3_iu4_comparison_rejects_every_confounder(self) -> None:
        contract = {"schema": quant_comparison.CONTRACT_SCHEMA,
                    "status": "complete", "identity": {"workload": "exact"},
                    "identity_sha256": "c" * 64, "publishes": False,
                    "selection_allowed": False}
        runtime = {"runtime_revision": "1" * 40,
                   "release_image": "image@sha256:" + "2" * 64,
                   "release_digest": "sha256:" + "2" * 64,
                   "dev_image": "dev@sha256:" + "3" * 64,
                   "dev_digest": "sha256:" + "3" * 64,
                   "engine_binary_sha256": "4" * 64,
                   "tensor_format_contract_sha256": "5" * 64}
        source = {"cache_id": "6" * 64, "manifest": {
            "sha256": "7" * 64}}
        profile = {"sha256": "8" * 64}
        mtp = {"sha256": "9" * 64, "size_bytes": 1024,
               "matrix_quant_contract": quant_comparison.MATCHED_MTP_CONTRACT}

        def arm(name: str, quant_arm: str, kernel: str, mode: str,
                prefill: list[float], decode: list[float]) -> tuple[dict, dict]:
            descriptor = {
                "candidate_id": name, "quantization_arm": quant_arm,
                "selection_plan": {"sha256": "a" * 64},
                "capture": {"sha256": "b" * 64},
                "intervention_configuration_id": "lambda-0.25-band-10-42",
                "intervention_manifest": {"sha256": "d" * 64},
            }
            construction = {
                "descriptor": descriptor, "descriptor_path": f"/{name}.json",
                "descriptor_sha256": "e" * 64,
                "build": {"bf16_cache": copy.deepcopy(source), "profile": profile},
                "mtp": copy.deepcopy(mtp),
            }
            facts = {
                "prefill_tps_samples": prefill, "decode_tps_samples": decode,
                "mtp_speculation": {
                    "accept_rate_mean": 0.75 if name == "q3" else 0.80},
                "resources": {"measured_peak_rss_bytes": 70_000_000_000,
                              "measured_peak_gtt_bytes": 5_000_000_000,
                              "measured_peak_uma_bytes": 75_000_000_000},
            }
            ar_tps = 20.0 if name == "q3" else 22.0
            mtp_tps = 25.0 if name == "q3" else 27.0
            differential = {
                "schema": quant_comparison.DIFFERENTIAL_DECODE_SCHEMA,
                "purpose": "same_process_diagnostic_not_hard_gate_timing",
                "tokens_per_path": 64,
                "ar": {"decode_seconds": 64.0 / ar_tps,
                       "tokens_per_second": ar_tps},
                "mtp": {"accept_rate": 0.70 if name == "q3" else 0.78,
                        "restored_decode_seconds": 64.0 / mtp_tps + 0.1,
                        "warm_fresh_decode_seconds": 64.0 / mtp_tps,
                        "warm_fresh_tokens_per_second": mtp_tps,
                        "warm_speedup_vs_ar": mtp_tps / ar_tps},
            }
            hardware = {
                "value": {"hard_gates": {"performance": {"passed": True},
                                           "memory": {"passed": True}},
                          "kernel_runtime": {"candidate_kernel_capability": kernel},
                          "timing_kernel_mode": {"configured_mmq_mode": mode}},
                "path": f"/{name}-hardware.json", "sha256": "f" * 64,
                "contract": copy.deepcopy(contract), "facts": facts,
                "differential_decode": differential,
                "artifact_bytes": 80_000_000_000,
                "runtime_identity": copy.deepcopy(runtime),
            }
            return construction, hardware

        q3c, q3h = arm("q3", quant_comparison.Q3_ARM,
                       "no_eligible_rocmi4_mmq",
                       "not_applicable_no_eligible_rocmi4_mmq",
                       [410.0, 412.0, 414.0], [39.0, 40.0, 41.0])
        iu4c, iu4h = arm("iu4", quant_comparison.IU4_ARM,
                         "rocmi4_dense_and_routed",
                         "w4a8_iu4_prepack",
                         [420.0, 422.0, 424.0], [41.0, 42.0, 43.0])
        comparison = quant_comparison.make_comparison(q3c, q3h, iu4c, iu4h)
        self.assertFalse(comparison["selection_allowed"])
        self.assertEqual(comparison["deltas"]["iu4_minus_q3_decode_median_tps"], 2.0)
        self.assertEqual(
            comparison["deltas"]["iu4_minus_q3_differential_ar_tps"], 2.0)
        self.assertAlmostEqual(
            comparison["deltas"]["iu4_minus_q3_clean_mtp_accept_rate"], 0.05)
        self.assertEqual(comparison["matched_identity"]["mtp"]["depth"], 3)

        mutations = (
            ("source", lambda c, h: c["build"]["bf16_cache"].update(
                {"cache_id": "0" * 64}), "BF16/intervention"),
            ("mtp", lambda c, h: c["mtp"].update({"sha256": "0" * 64}),
             "MTP companion"),
            ("runtime", lambda c, h: h["runtime_identity"].update(
                {"engine_binary_sha256": "0" * 64}), "runtime engine"),
            ("workload", lambda c, h: h["contract"].update(
                {"identity_sha256": "0" * 64}), "workload contract"),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label):
                changed_c = copy.deepcopy(iu4c)
                changed_h = copy.deepcopy(iu4h)
                mutate(changed_c, changed_h)
                with self.assertRaisesRegex(quant_comparison.ComparisonError, message):
                    quant_comparison.make_comparison(q3c, q3h, changed_c, changed_h)

        invalid_differential = copy.deepcopy(q3h["differential_decode"])
        invalid_differential["mtp"]["warm_speedup_vs_ar"] = 99.0
        with self.assertRaisesRegex(
                quant_comparison.ComparisonError,
                "differential decode derivation differs"):
            quant_comparison.validate_differential_decode(
                invalid_differential, "Q3")

    def test_q3_iu4_comparison_cli_revalidates_pinned_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            def write_json(path: Path, value: dict) -> Path:
                path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
                return path

            def desc(path: Path) -> dict[str, str]:
                return {"path": str(path.resolve()), "sha256": digest(path)}

            intervention = write_json(root / "intervention.json", {"direction": "same"})
            capture = write_json(root / "capture.json", {"capture": "same"})
            selection = write_json(root / "selection.json", {"selection": "same"})
            bf16 = write_json(root / "bf16.json", {"cache": "same"})
            mtp = root / "mtp.gguf"
            mtp.write_bytes(b"matching-fast-mtp")
            companion = write_json(root / "companion.json", {
                "schema": quant_comparison.COMPANION_SCHEMA,
                "companions": [{"role": "mtp", "enabled": True,
                                "path": str(mtp), "size_bytes": mtp.stat().st_size,
                                "sha256": digest(mtp),
                                "matrix_quant_contract":
                                    quant_comparison.MATCHED_MTP_CONTRACT}],
            })
            runtime = {
                "release_ref": "ghcr.io/otheru-ai/ember@sha256:" + "1" * 64,
                "release_digest": "sha256:" + "1" * 64,
                "dev_ref": "ghcr.io/otheru-ai/ember@sha256:" + "2" * 64,
                "dev_digest": "sha256:" + "2" * 64,
                "tensor_format_contract_sha256": "3" * 64,
            }
            revision = "4" * 40
            binary_sha = "5" * 64

            def make_arm(label: str, quant_arm: str, prefill: tuple[float, float, float],
                         decode: tuple[float, float, float], kernel: str) -> tuple[Path, Path]:
                directory = root / label
                directory.mkdir()
                timing = directory / "timing.jsonl"
                timing.write_text("".join(json.dumps(row) + "\n"
                                          for row in timing_rows(prefill, decode)),
                                  encoding="utf-8")
                contract = quant_comparison.derive_contract(
                    timing, ROOT / "scripts/bench/benchmark.py", GATE)
                contract_path = write_json(directory / "benchmark-contract.json", contract)
                model = directory / "model.gguf"
                model.write_bytes((label + "-main-model").encode("utf-8"))
                shard = {"path": str(model), "size_bytes": model.stat().st_size,
                         "sha256": digest(model)}
                record = write_json(directory / "qwen-quant-build-record.json", {
                    "status": "complete", "mode": "execute",
                    "compute_mode": "exact_dequant", "w4a4_enabled": False,
                    "tools": {"ember_revision": revision},
                    "profile": {"sha256": "6" * 64},
                    "bf16_cache": {"cache_id": "7" * 64,
                                   "manifest": {"sha256": digest(bf16)}},
                    "intervention": {"manifest_sha256": digest(intervention)},
                    "quantization_recipe": {
                        "id": quant_arm,
                        "formats": (["Q3_0_ROCMFPX", "Q4_0_ROCMFP4_FAST", "Q6_K"]
                                    if label == "q3" else ["Q4_0_ROCMI4", "Q6_K"]),
                        "selected_mtp_matrix_quant_contract":
                            quant_comparison.MATCHED_MTP_CONTRACT,
                        "ple_override_preserved": True,
                    },
                    "output": {"shards": [shard]},
                })
                attestation = write_json(directory / "attestation.json", {
                    "schema": "ember.qwen3.8.candidate-workset-attestation.v1",
                    "candidate_id": f"{label}-candidate",
                    "build_record_sha256": digest(record),
                    "builder_identity": {"ember_revision": revision,
                                         "tensor_format_contract_sha256": "3" * 64},
                    "tensor_format_compatibility_sha256": "3" * 64,
                })
                construction = write_json(directory / "construction.json", {
                    "schema": quant_comparison.CONSTRUCTION_SCHEMA,
                    "status": "complete", "publishes": False, "deletes": False,
                    "candidate_id": f"{label}-candidate", "kind": "intervention",
                    "intended_stage": "format",
                    "row_id": (quant_comparison.Q3_ROW if label == "q3"
                               else quant_comparison.IU4_ROW),
                    "intervention_configuration_id": "lambda-0.25-band-10-42",
                    "quantization_arm": quant_arm,
                    "mtp_matrix_quant_contract": quant_comparison.MATCHED_MTP_CONTRACT,
                    "runtime_mode": "exact_dequant", "builder_revision": revision,
                    "runtime_revision": revision,
                    "images": {"builder": {"ref": runtime["dev_ref"],
                                             "digest": runtime["dev_digest"]},
                               "runtime": runtime},
                    "capture": desc(capture), "stock_capture": None,
                    "bf16_cache": desc(bf16),
                    "shared_companions": {
                        "Q4_0_ROCMI4": desc(companion),
                        "Q4_0_ROCMFP4_FAST": desc(companion)},
                    "selection_plan": desc(selection), "build_record": desc(record),
                    "builder_attestation": desc(attestation),
                    "intervention_manifest": desc(intervention),
                    "artifacts": {"shards": [shard],
                                  "total_bytes": model.stat().st_size},
                    "v3_candidate_manifest": {"ready": False, "blocked_on": []},
                })
                facts = quant_comparison.bakeoff.timing_facts(timing, True)
                mode = ("not_applicable_no_eligible_rocmi4_mmq"
                        if label == "q3" else "w4a8_iu4_prepack")
                kernel_runtime = write_json(directory / "kernel-runtime-evidence.json", {
                    "schema": quant_comparison.KERNEL_RUNTIME_SCHEMA,
                    "candidate_kernel_capability": kernel,
                    "candidate_timing_kernel_mode": mode,
                    "passed": True,
                })
                kernel_build = write_json(directory / "w4a8-build-evidence.json", {
                    "schema": quant_comparison.KERNEL_BUILD_SCHEMA,
                    "build_mode": "w4a8_iu4_prepack",
                    "candidate_and_profiler_binary_sha256": binary_sha,
                    "saved_isa_gate": {"passed": True},
                })
                timing_mode = write_json(directory / "timing-kernel-mode.json", {
                    "configured_mmq_mode": mode,
                    "confirmation": "clean_timing_startup_marker",
                    "passed": True,
                })
                ar_tps = 16.0 if label == "q3" else 20.0
                warm_mtp_tps = 32.0 if label == "q3" else 40.0
                differential_decode_value = {
                    "schema": quant_comparison.DIFFERENTIAL_DECODE_SCHEMA,
                    "purpose": "same_process_diagnostic_not_hard_gate_timing",
                    "tokens_per_path": 64,
                    "ar": {"decode_seconds": 64.0 / ar_tps,
                           "tokens_per_second": ar_tps},
                    "mtp": {
                        "accept_rate": 0.70 if label == "q3" else 0.80,
                        "restored_decode_seconds": 64.0 / warm_mtp_tps + 0.1,
                        "warm_fresh_decode_seconds": 64.0 / warm_mtp_tps,
                        "warm_fresh_tokens_per_second": warm_mtp_tps,
                        "warm_speedup_vs_ar": warm_mtp_tps / ar_tps,
                    },
                }
                differential_decode = write_json(
                    directory / "differential-decode-comparison.json",
                    differential_decode_value)
                hardware = write_json(directory / "hardware-measured.json", {
                    "schema": quant_comparison.HARDWARE_SCHEMA,
                    "publish_approved": False,
                    "methodology": "clean timing and profiler/counter passes are separate",
                    "image": {"ref": runtime["release_ref"],
                              "digest": runtime["release_digest"]},
                    "profile_image": {"ref": runtime["dev_ref"],
                                      "digest": runtime["dev_digest"],
                                      "ember_revision": revision,
                                      "candidate_binary_sha256": binary_sha,
                                      "candidate_binary_byte_identical": True},
                    "model": {"ordered_inventory": {
                        "build_record": {"sha256": digest(record)},
                        "shards": [shard]}},
                    "mtp": {"path": str(mtp), "sha256": digest(mtp), "depth": 3},
                    "hard_gates": {"performance": facts["hard_gate"],
                                   "memory": facts["memory_gate"]},
                    "resources": facts["resources"],
                    "speculation": facts["mtp_speculation"],
                    "kernel_runtime": json.loads(kernel_runtime.read_text()),
                    "kernel_build": json.loads(kernel_build.read_text()),
                    "timing_kernel_mode": json.loads(timing_mode.read_text()),
                    "benchmark_contract": contract,
                    "differential_decode": differential_decode_value,
                    "evidence": {"timing": {"path": timing.name,
                                               "sha256": digest(timing)},
                                 "benchmark_contract": {
                                     "path": contract_path.name,
                                     "sha256": digest(contract_path)},
                                 "kernel_runtime": {
                                     "path": kernel_runtime.name,
                                     "sha256": digest(kernel_runtime)},
                                 "kernel_build": {
                                     "path": kernel_build.name,
                                     "sha256": digest(kernel_build)},
                                 "timing_kernel_mode": {
                                     "path": timing_mode.name,
                                     "sha256": digest(timing_mode)},
                                 "differential_decode": {
                                     "path": differential_decode.name,
                                     "sha256": digest(differential_decode)}},
                })
                return construction, hardware

            q3_construction, q3_hardware = make_arm(
                "q3", quant_comparison.Q3_ARM, (411.0, 412.0, 413.0),
                (39.0, 40.0, 41.0), "no_eligible_rocmi4_mmq")
            iu4_construction, iu4_hardware = make_arm(
                "iu4", quant_comparison.IU4_ARM, (421.0, 422.0, 423.0),
                (41.0, 42.0, 43.0), "rocmi4_dense_and_routed")
            q3_evidence_path = root / "q3-arm-evidence.json"
            validated = subprocess.run([
                sys.executable, str(ROOT / "scripts/qwen_quant_comparison.py"),
                "validate-arm", "--arm", "q3",
                "--construction", str(q3_construction),
                "--construction-sha256", digest(q3_construction),
                "--hardware", str(q3_hardware),
                "--hardware-sha256", digest(q3_hardware),
                "--output", str(q3_evidence_path),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(validated.returncode, 0, validated.stderr)
            q3_evidence = json.loads(q3_evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(q3_evidence["schema"],
                             quant_comparison.ARM_EVIDENCE_SCHEMA)
            self.assertEqual(q3_evidence["construction"]["quantization_arm"],
                             quant_comparison.Q3_ARM)
            self.assertEqual(
                q3_evidence["observations"]["differential_decode"]["mtp"]
                ["warm_speedup_vs_ar"], 2.0)
            q3_construction_value = quant_comparison._construction(
                q3_construction, digest(q3_construction),
                quant_comparison.Q3_ARM, "Q3")
            q3_hardware_value = quant_comparison.validate_hardware(
                q3_hardware, digest(q3_hardware), q3_construction_value, "Q3")
            verified_plan = {"format_arms": [{
                "id": quant_comparison.IU4_ROW,
                "quantization_arm": quant_comparison.IU4_ARM,
                "mtp_matrix_quant_contract": quant_comparison.MATCHED_MTP_CONTRACT,
                "mtp_depth": 3,
            }]}
            request_output = quant_comparison.REQUEST_ROOT / "construction-iu4-test.json"
            with mock.patch.object(
                    quant_comparison.bakeoff, "verify_plan",
                    return_value=verified_plan):
                request = quant_comparison.make_matched_iu4_request(
                    q3_construction_value, q3_hardware_value,
                    "iu4-matched-test", revision, request_output)
            self.assertEqual(request["mode"], "build-candidate")
            self.assertEqual(request["parameters"]["row_id"],
                             quant_comparison.IU4_ROW)
            self.assertEqual(request["parameters"]["quantization_arm"],
                             quant_comparison.IU4_ARM)
            self.assertFalse(request["publishes"])
            with self.assertRaisesRegex(
                    quant_comparison.ComparisonError, "complete current Q3 proof"):
                quant_comparison.make_matched_iu4_request(
                    q3_construction_value, q3_hardware_value,
                    "iu4-matched-test", "0" * 40, request_output)
            binding_path = root / "iu4-binding.json"
            inspected = subprocess.run([
                sys.executable, str(ROOT / "scripts/qwen_quant_comparison.py"),
                "inspect-construction", "--arm", "iu4",
                "--construction", str(iu4_construction),
                "--construction-sha256", digest(iu4_construction),
                "--output", str(binding_path),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(inspected.returncode, 0, inspected.stderr)
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            self.assertEqual(binding["schema"], quant_comparison.BINDING_SCHEMA)
            self.assertEqual(binding["quantization_arm"], quant_comparison.IU4_ARM)
            self.assertEqual(binding["mtp"]["depth"], 3)
            self.assertFalse(binding["w4a4_enabled"])
            pair_path = root / "construction-pair.json"
            paired = subprocess.run([
                sys.executable, str(ROOT / "scripts/qwen_quant_comparison.py"),
                "validate-pair",
                "--q3-construction", str(q3_construction),
                "--q3-construction-sha256", digest(q3_construction),
                "--iu4-construction", str(iu4_construction),
                "--iu4-construction-sha256", digest(iu4_construction),
                "--output", str(pair_path),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(paired.returncode, 0, paired.stderr)
            pair = json.loads(pair_path.read_text(encoding="utf-8"))
            self.assertEqual(pair["schema"], quant_comparison.PAIR_BINDING_SCHEMA)
            self.assertFalse(pair["selection_allowed"])
            output = root / "comparison.json"
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/qwen_quant_comparison.py"), "compare",
                "--q3-construction", str(q3_construction),
                "--q3-construction-sha256", digest(q3_construction),
                "--q3-hardware", str(q3_hardware),
                "--q3-hardware-sha256", digest(q3_hardware),
                "--iu4-construction", str(iu4_construction),
                "--iu4-construction-sha256", digest(iu4_construction),
                "--iu4-hardware", str(iu4_hardware),
                "--iu4-hardware-sha256", digest(iu4_hardware),
                "--output", str(output),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["schema"], quant_comparison.COMPARISON_SCHEMA)
            self.assertEqual(value["deltas"]["iu4_minus_q3_decode_median_tps"], 2.0)
            self.assertEqual(
                value["deltas"]["iu4_minus_q3_differential_ar_tps"], 4.0)
            self.assertEqual(
                value["deltas"]["iu4_minus_q3_differential_warm_mtp_tps"],
                8.0)
            self.assertEqual(value["interpretation"],
                             "descriptive_sequential_comparison_not_counterbalanced_selection")

            repeated = subprocess.run([
                sys.executable, str(ROOT / "scripts/qwen_quant_comparison.py"), "compare",
                "--q3-construction", str(q3_construction),
                "--q3-construction-sha256", digest(q3_construction),
                "--q3-hardware", str(q3_hardware),
                "--q3-hardware-sha256", digest(q3_hardware),
                "--iu4-construction", str(iu4_construction),
                "--iu4-construction-sha256", digest(iu4_construction),
                "--iu4-hardware", str(iu4_hardware),
                "--iu4-hardware-sha256", digest(iu4_hardware),
                "--output", str(output),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("output must be one new absolute path", repeated.stderr)

            differential_path = root / "q3" / "differential-decode-comparison.json"
            differential_value = json.loads(
                differential_path.read_text(encoding="utf-8"))
            differential_value["mtp"]["warm_speedup_vs_ar"] = 99.0
            write_json(differential_path, differential_value)
            tampered = subprocess.run([
                sys.executable, str(ROOT / "scripts/qwen_quant_comparison.py"),
                "validate-arm", "--arm", "q3",
                "--construction", str(q3_construction),
                "--construction-sha256", digest(q3_construction),
                "--hardware", str(q3_hardware),
                "--hardware-sha256", digest(q3_hardware),
                "--output", str(root / "tampered-arm.json"),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("differential decode evidence SHA-256 differs",
                          tampered.stderr)

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
