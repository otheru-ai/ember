#!/usr/bin/env python3
"""Integration seam for the continuous-batch coordinator loop.

test_batch_wake.c pins the wake predicate's truth table, but nothing in it makes
the worker USE the predicate: deleting it from the real wait leaves that test
green. This test closes that gap by compiling and RUNNING the actual worker.

The worker lives in src/backend/backend_dflash.cc, which includes HIP and is
therefore excluded from the host build, so it cannot simply be linked. Instead
the production functions are extracted VERBATIM from that file by name and
compiled against the real scheduler, executor and coordinator with the fake
backend the coordinator tests already use. Nothing here is a paraphrase of the
worker; if the extraction anchors stop matching, the test fails closed rather
than silently testing nothing.

Approach adopted from codex-rejoin-01's review harness.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "src" / "backend" / "backend_dflash.cc"
COORD_TEST = ROOT / "test" / "test_resident_batch_coordinator.cpp"
ENGINE = ROOT / "engine" / "dflash" / "common"


def die(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def block(text, start_anchor, what):
    """Return start_anchor plus its brace-balanced body, verbatim."""
    i = text.find(start_anchor)
    if i < 0:
        die(f"anchor for {what} not found in backend_dflash.cc: {start_anchor!r}. "
            f"If the code was renamed, update this test -- do not delete it, "
            f"it is the only thing exercising the real worker loop.")
    if text.find(start_anchor, i + 1) >= 0:
        die(f"anchor for {what} is ambiguous: {start_anchor!r}")
    j = text.index("{", i)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[i:k + 1]
    die(f"unbalanced braces extracting {what}")


def main():
    src = BACKEND.read_text()

    pieces = [
        # Struct blocks need their trailing semicolon, which the brace matcher
        # stops before.
        block(src, "struct ember_batch_call {", "ember_batch_call") + ";",
        block(src, "struct ember_batch_control {", "ember_batch_control") + ";",
        block(src, "static void batch_fail_call(", "batch_fail_call"),
        block(src, "static void ember_batch_discard_controls_locked(",
              "discard_controls"),
        block(src, "static void ember_batch_thread_main(", "thread_main"),
        block(src, "static bool ember_batch_control_run(", "control_run"),
    ]
    worker = "\n\n".join(pieces)

    # The template header line sits above the extracted control_run body.
    if "template <typename Fn>" not in src:
        die("control_run is no longer a template; update the harness")
    worker = worker.replace("static bool ember_batch_control_run(",
                            "template <typename Fn>\nstatic bool "
                            "ember_batch_control_run(")

    # Guard the specific regressions this seam exists for, so a silent removal
    # cannot pass by merely still compiling.
    for needed, why in (
        ("ember_batch_should_wake(", "the wait must use the shared predicate"),
        ("if (!b->batch_running || b->batch_stop) return false;",
         "controls must be refused once the coordinator is stopping"),
        ("ember_batch_discard_controls_locked(b);",
         "queued controls must be discarded when the worker exits"),
    ):
        if needed not in worker:
            die(f"extracted worker no longer contains {needed!r}: {why}")

    fake = COORD_TEST.read_text()
    fake = fake[fake.index("struct FakeResidentBackend"):
                fake.index("\nstatic GenerateRequest request")]

    harness = (ROOT / "test" / "batch_worker_harness.inc").read_text()

    prog = harness.replace("// @@WORKER@@", worker).replace("// @@FAKE@@", fake)

    tmp = Path(tempfile.mkdtemp(prefix="ember-batch-seam-"))
    try:
        cpp = tmp / "worker_seam.cpp"
        cpp.write_text(prog)
        binary = tmp / "worker_seam"
        cxx = os.environ.get("CXX", "g++")
        cmd = [cxx, "-std=c++17", "-O1", "-g", "-pthread",
               f"-I{ROOT / 'src'}", f"-I{ENGINE}",
               f"-I{ROOT / 'engine' / 'ggml' / 'include'}",
               str(cpp),
               str(ENGINE / "resident_batch_coordinator.cpp"),
               str(ENGINE / "continuous_batch_scheduler.cpp"),
               str(ENGINE / "continuous_batch_executor.cpp"),
               "-o", str(binary)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            die(f"compiling the extracted worker failed:\n{r.stderr[:4000]}")
        try:
            r = subprocess.run([str(binary)], capture_output=True, text=True,
                               timeout=60)
        except subprocess.TimeoutExpired:
            die("the extracted worker DEADLOCKED. That is the failure this "
                "test exists to catch: a coordinator wait that can miss "
                "queued work, or a control waiting on a worker that has "
                "exited.")
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            die("extracted worker scenarios failed")
        print("batch worker seam: ok")
    finally:
        if os.environ.get("EMBER_SEAM_KEEP"):
            print(f"seam workspace kept at {tmp}", file=sys.stderr)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
