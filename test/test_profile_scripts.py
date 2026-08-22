#!/usr/bin/env python3
"""GPU-free contract tests for the profiling harness.

scripts/profile_gpu.sh only ever runs on the gfx1151 host, where it takes
exclusive GPU access and stops production. None of that is reachable from CI, so
what is tested here is everything that does not need a GPU: the argument
contract, the dry-run plan, and -- most importantly -- that production restore
is unconditional. A harness that can strand production is worse than no harness.

scripts/profile_report.py is pure analysis and is tested end to end against
synthetic rocprofv3 CSVs, including the failure modes that would otherwise
produce a confident wrong number: an unseparated warmup, and trace/pmc passes
that did not execute the same work.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROFILE_SH = ROOT / "scripts" / "profile_gpu.sh"
PROFILE_PY = ROOT / "scripts" / "profile_report.py"

sys.path.insert(0, str(ROOT / "scripts"))
import profile_report  # noqa: E402


def run_sh(args, env=None, expect_ok=True):
    result = subprocess.run(
        ["bash", str(PROFILE_SH), *args],
        text=True,
        capture_output=True,
        env=env,
        cwd=str(ROOT),
    )
    if expect_ok:
        assert result.returncode == 0, f"exit {result.returncode}\n{result.stderr}"
    return result


def sabotaged_path(directory):
    """A PATH whose docker/curl/sudo abort loudly, proving --dry-run touches nothing."""
    for tool in ("docker", "curl", "sudo", "rocprofv3"):
        stub = pathlib.Path(directory) / tool
        stub.write_text(
            "#!/bin/sh\necho \"FORBIDDEN: %s invoked during dry run\" >&2\nexit 97\n" % tool
        )
        stub.chmod(0o755)
    return os.environ | {"PATH": directory + os.pathsep + os.environ["PATH"]}


class HarnessContractTests(unittest.TestCase):
    def test_script_is_syntactically_valid(self):
        subprocess.run(["bash", "-n", str(PROFILE_SH)], check=True, capture_output=True)

    def test_dry_run_touches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_sh(
                ["--dry-run", "--model", "/models/model.gguf"],
                env=sabotaged_path(tmp),
            )
        self.assertIn("plan:", result.stdout)
        self.assertIn("no GPU touched", result.stdout)
        self.assertNotIn("FORBIDDEN", result.stderr)

    def test_dry_run_without_model_still_prints_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_sh(["--dry-run"], env=sabotaged_path(tmp))
        self.assertIn("MISSING --model", result.stdout)
        self.assertIn("plan:", result.stdout)

    def test_relative_model_path_is_rejected(self):
        # The model is bind-mounted by dirname; a relative path would mount the
        # wrong directory and load a different file than the operator named.
        with tempfile.TemporaryDirectory() as tmp:
            result = run_sh(
                ["--dry-run", "--model", "models/model.gguf"],
                env=sabotaged_path(tmp),
                expect_ok=False,
            )
        self.assertIn("must be absolute", result.stdout)

    def test_invalid_arguments_are_rejected(self):
        for args in (
            ["--port", "80"],
            ["--port", "notaport"],
            ["--prefill-words", "0"],
            ["--decode-tokens", "-1"],
            ["--gap-secs", "0"],
            ["--nonsense"],
        ):
            with self.subTest(args=args):
                result = run_sh([*args, "--dry-run"], expect_ok=False)
                self.assertNotEqual(result.returncode, 0)

    def test_production_restore_is_unconditional(self):
        body = PROFILE_SH.read_text()
        # Restore must be wired to abnormal exits, not just the happy path.
        self.assertIn("trap cleanup EXIT INT TERM", body)
        self.assertIn("restore_production", body)
        # Both deployment shapes must be restorable.
        self.assertIn("sudo -n \"$PRODUCTION_WRAPPER\" start", body)
        self.assertIn('docker start "$PRODUCTION_CONTAINER"', body)
        # Restore must not be able to abort the trap before it finishes.
        for line in body.splitlines():
            if "PRODUCTION_WRAPPER\" start" in line or 'start "$PRODUCTION_CONTAINER"' in line:
                self.assertIn("|| true", line, f"restore must not abort: {line.strip()}")

    def test_quiesce_uses_the_fixed_purpose_wrapper(self):
        # Stopping the container alone lets systemd recreate the ~90 GiB process
        # mid-measurement; the wrapper is the only supported quiesce path.
        body = PROFILE_SH.read_text()
        self.assertIn("/usr/local/sbin/ember-cert-production", body)
        self.assertIn("is-active", body)

    def test_dev_image_guarantees_the_profiler(self):
        # rocprofv3 ships with the pinned rocm/dev-ubuntu-24.04:*-full base, not
        # from apt, so without an explicit assertion nothing records the
        # dependency and a base-image bump could silently remove it.
        dockerfile = (ROOT / "docker" / "Dockerfile").read_text()
        dev = dockerfile.split("FROM toolchain AS dev", 1)
        self.assertEqual(len(dev), 2, "dev stage not found in docker/Dockerfile")
        dev_stage = dev[1].split("\nFROM ", 1)[0]
        self.assertIn("rocprofv3", dev_stage)
        self.assertRegex(dev_stage, r"command -v rocprofv3")

    def test_release_image_does_not_carry_the_profiler(self):
        # The release image is a stripped runtime closure; collect-runtime.sh
        # exists precisely to keep compilers and profilers out of it.
        collect = (ROOT / "docker" / "collect-runtime.sh").read_text()
        self.assertIn("profiler", collect)

    def test_profiled_binary_default_is_unstripped(self):
        # The dev image installs a stripped ember-dflash into /usr/local/bin and
        # leaves the unstripped one in the build tree. Profile the latter.
        body = PROFILE_SH.read_text()
        self.assertIn("/ember/build-rocm/ember-dflash", body)
        self.assertNotIn('BINARY:-/usr/local/bin/ember-dflash', body)

    def test_one_counter_per_pmc_pass(self):
        # MEASURED on gfx1151: two counters in one --pmc pass fail with
        # "error code 38: Request exceeds the capabilities of the hardware to
        # collect" and rocprofv3 SIGSEGVs (exit 139) with no CSV and no
        # diagnostic. Collapsing these back into one invocation silently
        # destroys every bandwidth measurement.
        body = PROFILE_SH.read_text()
        self.assertNotIn('--pmc "${PMC_COUNTERS[@]}"', body)
        self.assertIn('--pmc "$counter"', body)
        self.assertIn("error code 38", body)
        self.assertIn('for counter in "${PMC_COUNTERS[@]}"', body)

    def test_counter_pass_is_separate_from_timing_pass(self):
        # Durations under --pmc are serialized and must never be a bandwidth
        # denominator. The two passes existing separately is the whole guard.
        body = PROFILE_SH.read_text()
        self.assertIn("run_pass trace", body)
        self.assertIn("run_pass pmc", body)


def write_trace(path, rows):
    with open(path, "w") as fh:
        fh.write("Kind,Kernel_Name,Correlation_Id,Start_Timestamp,End_Timestamp\n")
        for name, start, end in rows:
            fh.write(f"KERNEL_DISPATCH,{name},0,{start},{end}\n")


def write_pmc(path, per_dispatch):
    with open(path, "w") as fh:
        fh.write("Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value\n")
        for i, (name, counter, value) in enumerate(per_dispatch):
            fh.write(f"{i},{name},{counter},{value}\n")


class ReportTests(unittest.TestCase):
    def make_outdir(self, tmp, decode_kb, decode_busy_ms=100.0, decode_pmc_rows=None):
        ms = 1_000_000
        # Warmup burst, a 2s idle gap, then the measured request.
        warm = [("warmup_kernel", 0, 5 * ms)]
        gap_end = 2_000 * ms
        measured = [
            ("gemm_kernel", gap_end, gap_end + int(decode_busy_ms * 0.7 * ms)),
            (
                "dequant_kernel",
                gap_end + int(decode_busy_ms * 0.7 * ms),
                gap_end + int(decode_busy_ms * ms),
            ),
        ]
        write_trace(os.path.join(tmp, "trace-decode_kernel_trace.csv"), warm + measured)
        write_trace(os.path.join(tmp, "trace-prefill_kernel_trace.csv"), warm + measured)
        rows = decode_pmc_rows or [
            ("gemm_kernel", "FETCH_SIZE", decode_kb),
            ("dequant_kernel", "WRITE_SIZE", 0),
        ]
        write_pmc(os.path.join(tmp, "pmc-decode_counter_collection.csv"), rows)
        write_pmc(os.path.join(tmp, "pmc-prefill_counter_collection.csv"), rows)

    def analyse(self, tmp, *extra):
        result = subprocess.run(
            [sys.executable, str(PROFILE_PY), tmp, "--json", *extra],
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    def test_segments_split_on_the_idle_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_outdir(tmp, decode_kb=1000)
            report = self.analyse(tmp)
        decode = report["phases"]["decode"]
        self.assertEqual(len(decode["segments"]), 2)
        self.assertEqual(decode["measured_segment"], 1)
        self.assertEqual(decode["dispatches"], 2)
        self.assertNotIn("warning", decode)

    def test_single_segment_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            ms = 1_000_000
            write_trace(
                os.path.join(tmp, "trace-decode_kernel_trace.csv"),
                [("k", 0, ms), ("k", ms, 2 * ms)],
            )
            write_trace(os.path.join(tmp, "trace-prefill_kernel_trace.csv"), [("k", 0, ms)])
            report = self.analyse(tmp)
        self.assertIn("warning", report["phases"]["decode"])

    def test_bandwidth_uses_trace_time_and_kb_counters(self):
        # 21_200_000 KB (KB=1024) over 100 ms == 217 GB/s, just over the roofline.
        with tempfile.TemporaryDirectory() as tmp:
            self.make_outdir(tmp, decode_kb=21_200_000, decode_busy_ms=100.0)
            report = self.analyse(tmp)
        decode = report["phases"]["decode"]
        self.assertAlmostEqual(decode["achieved_gbps"], 217.1, delta=1.0)
        self.assertIn("BANDWIDTH-BOUND", report["verdict"])

    def test_low_bandwidth_reports_compute_headroom(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_outdir(tmp, decode_kb=1_000_000, decode_busy_ms=100.0)
            report = self.analyse(tmp)
        self.assertIn("COMPUTE HEADROOM", report["verdict"])
        # The RDNA 3.5 layout correction must ride along with any advice to go
        # write kernels, or the next reader repeats the gfx12 mistake.
        self.assertIn("RDNA 3.5", report["verdict"])

    def test_missing_counters_are_inconclusive_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_outdir(tmp, decode_kb=0, decode_pmc_rows=[("k", "GRBM_COUNT", 5)])
            report = self.analyse(tmp)
        self.assertIn("INCONCLUSIVE", report["verdict"])
        self.assertNotIn("achieved_gbps", report["phases"]["decode"])

    def test_counter_unit_is_overridable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_outdir(tmp, decode_kb=21_200_000, decode_busy_ms=100.0)
            as_bytes = self.analyse(tmp, "--counter-unit", "b")
        self.assertLess(as_bytes["phases"]["decode"]["achieved_gbps"], 1.0)

    def test_overlapping_kernels_do_not_inflate_busy_time(self):
        # Concurrent dispatches overlap; summing durations would double-count and
        # halve the apparent bandwidth.
        ms = 1_000_000
        seg = [(0, 10 * ms, "a"), (2 * ms, 8 * ms, "b"), (5 * ms, 12 * ms, "c")]
        self.assertEqual(profile_report.busy_ns(seg), 12 * ms)

    def test_pmc_trace_dispatch_mismatch_is_warned(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_outdir(
                tmp,
                decode_kb=0,
                decode_pmc_rows=[(f"k{i}", "FETCH_SIZE", 10) for i in range(50)],
            )
            result = subprocess.run(
                [sys.executable, str(PROFILE_PY), tmp],
                text=True,
                capture_output=True,
                check=True,
            )
        self.assertIn("did not execute the same work", result.stdout)

    def test_per_counter_passes_are_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_outdir(tmp, decode_kb=0)
            os.remove(os.path.join(tmp, "pmc-decode_counter_collection.csv"))
            # One file per counter, as the fixed harness emits.
            write_pmc(
                os.path.join(tmp, "pmc-decode-FETCH_SIZE_counter_collection.csv"),
                [("gemm_kernel", "FETCH_SIZE", 10_600_000)],
            )
            write_pmc(
                os.path.join(tmp, "pmc-decode-WRITE_SIZE_counter_collection.csv"),
                [("gemm_kernel", "WRITE_SIZE", 10_600_000)],
            )
            report = self.analyse(tmp)
        decode = report["phases"]["decode"]
        self.assertEqual(len(decode["counter_files"]), 2)
        self.assertEqual(decode["counters_raw"]["FETCH_SIZE"], 10_600_000)
        self.assertEqual(decode["counters_raw"]["WRITE_SIZE"], 10_600_000)
        # Both directions summed: 21.2M KB over 100 ms.
        self.assertAlmostEqual(decode["achieved_gbps"], 217.1, delta=1.0)
        self.assertNotIn("partial_counters", decode)

    def test_single_direction_is_marked_a_lower_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_outdir(
                tmp, decode_kb=0,
                decode_pmc_rows=[("gemm_kernel", "FETCH_SIZE", 10_600_000)],
            )
            report = self.analyse(tmp)
        self.assertIn("LOWER BOUND", report["phases"]["decode"]["partial_counters"])

    def test_missing_directory_fails_cleanly(self):
        result = subprocess.run(
            [sys.executable, str(PROFILE_PY), "/nonexistent-profile-dir"],
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a directory", result.stderr)


if __name__ == "__main__":
    unittest.main()
