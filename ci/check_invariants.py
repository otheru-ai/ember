#!/usr/bin/env python3
"""Repo invariants that a compiler cannot catch.

Ember keeps two hand-maintained CMake source lists — `ember_core` (stub build,
drives every GPU-free test) and the `ember-dflash` executable (ROCm build). A
new src/ file added to only one list builds fine on the host and fails, or
silently misses code, in the container. That has already cost one fix commit
(d8ace73), and CI is the only place it can be caught cheaply: the ROCm build
needs a GPU toolchain no runner has.

Also checks that every test source is actually registered with ctest — an
add_executable without a matching add_test compiles and is never run, which
reads as passing coverage that does not exist.

Exit status is 0 when every invariant holds, 1 otherwise; each violation is
printed as one actionable line.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CMAKE = ROOT / "CMakeLists.txt"

# Files deliberately present in only one list. Each needs a reason, because the
# default assumption is that a src/ file belongs in both.
SINGLE_LIST_EXCEPTIONS = {
    "src/backend/backend_stub.c":
        "GPU-free stub backend; ember_core only. Linking it into ember-dflash "
        "would collide with backend_dflash.cc's ABI symbols.",
    "src/backend/backend_dflash.cc":
        "ROCm bridge; ember-dflash only. Requires the HIP toolchain.",
    "src/server/main.c":
        "ember-dflash lists it directly; the stub build compiles it into the "
        "separate ember-server executable target instead of ember_core.",
    "src/model/gguf.c":
        "Standalone GGUF metadata reader, ember_core only. No src/server/ or "
        "src/backend/ file includes gguf.h, so ember-dflash has no use for it "
        "-- the server reads GGUF through the vendored engine's loader. It is "
        "NOT dead code and must not be retired: providers/xdna2/"
        "{q8,rocmfp4}_model_weights.cpp include it, which pulls it into six "
        "targets besides ember_core -- the GPU-free test_xdna_q8_model_weights "
        "and test_xdna_rocmfp4_model_weights ctest binaries, the two "
        "ember-dspark-*-bench targets, and the two ember-xdna-dspark-*-validate "
        "targets. This exception is permanent; keep it.",
}


def read_cmake() -> str:
    return CMAKE.read_text()


def extract_block(text: str, header: str) -> set[str]:
    """Source paths inside a single add_library/add_executable(...) call."""
    start = text.index(header)
    depth, i = 0, start
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = text[start:i]
    return {tok for tok in body.split() if tok.startswith("src/")}


def check_source_lists(text: str, fail) -> None:
    core = extract_block(text, "add_library(ember_core STATIC")
    dflash = extract_block(text, "add_executable(ember-dflash")

    on_disk = {
        str(p.relative_to(ROOT))
        for p in list(ROOT.glob("src/**/*.c")) + list(ROOT.glob("src/**/*.cc"))
    }

    for path in sorted(on_disk):
        in_core, in_dflash = path in core, path in dflash
        if in_core and in_dflash:
            continue
        if path in SINGLE_LIST_EXCEPTIONS:
            continue
        if not in_core and not in_dflash:
            fail(f"{path}: compiled by neither ember_core nor ember-dflash; "
                 f"add it to both CMake source lists")
        else:
            present = "ember_core" if in_core else "ember-dflash"
            missing = "ember-dflash" if in_core else "ember_core"
            fail(f"{path}: listed in {present} but not {missing}. Add it to "
                 f"both, or document it in SINGLE_LIST_EXCEPTIONS with a "
                 f"reason (see d8ace73).")

    for path in sorted(core | dflash):
        if path not in on_disk:
            fail(f"{path}: referenced by CMakeLists.txt but not on disk")

    for path in sorted(SINGLE_LIST_EXCEPTIONS):
        if path in core and path in dflash:
            fail(f"{path}: now in both source lists; remove its stale "
                 f"SINGLE_LIST_EXCEPTIONS entry")
        elif path not in on_disk:
            fail(f"{path}: SINGLE_LIST_EXCEPTIONS names a file that no longer "
                 f"exists; remove the entry")


def check_tests_registered(text: str, fail) -> None:
    registered = set(re.findall(r"add_test\(\s*NAME\s+(\w+)", text))
    commands = " ".join(re.findall(r"add_test\((.*?)\)", text, re.S))

    for src in sorted(ROOT.glob("test/test_*")):
        if src.suffix not in {".c", ".cpp", ".py"}:
            continue
        rel = str(src.relative_to(ROOT))
        if src.suffix == ".py":
            if src.name not in commands:
                fail(f"{rel}: no add_test() references this script")
            continue
        stem = src.stem                      # test_sse
        # The source may sit on the next line, so match any whitespace after
        # the target name rather than a single space.
        if not re.search(rf"add_executable\(\s*{re.escape(stem)}\s", text):
            fail(f"{rel}: no add_executable({stem} ...) in CMakeLists.txt")
            continue
        name = stem[len("test_"):]           # ctest name drops the prefix
        if name not in registered:
            fail(f"{rel}: builds as {stem} but has no "
                 f"add_test(NAME {name} ...) — it never runs under ctest")


def check_test_timeouts(text: str, fail) -> None:
    """Every ctest test must carry a TIMEOUT.

    Without one, a hung test blocks the whole run instead of failing: several
    tests here join threads or drive real sockets, and a stale build has already
    been seen sitting in futex_wait for hours. CI passing --timeout on the
    command line is not a substitute — it applies only when someone remembers to
    pass it, and it does not protect a plain `ctest` locally.
    """
    # Expand set(VAR a b c) so a set_tests_properties(${VAR} ...) resolves.
    variables: dict[str, list[str]] = {}
    for m in re.finditer(r"set\(\s*(\w+)([^)]*)\)", text):
        variables[m.group(1)] = m.group(2).split()

    def expand(tokens: list[str]) -> list[str]:
        out: list[str] = []
        for tok in tokens:
            ref = re.fullmatch(r"\$\{(\w+)\}", tok)
            out.extend(variables.get(ref.group(1), []) if ref else [tok])
        return out

    timed: set[str] = set()
    for m in re.finditer(r"set_tests_properties\(([^)]*)\)", text, re.S):
        body = m.group(1)
        if "PROPERTIES" not in body:
            continue
        names, _, props = body.partition("PROPERTIES")
        if "TIMEOUT" not in props:
            continue
        timed.update(expand(names.split()))

    registered = set(re.findall(r"add_test\(\s*NAME\s+(\w+)", text))
    for name in sorted(registered - timed):
        fail(f"test '{name}' has no TIMEOUT — a hang would block ctest "
             f"indefinitely. Add it to EMBER_C_TESTS or give it its own "
             f"set_tests_properties(... PROPERTIES TIMEOUT n).")
    for name in sorted(timed - registered):
        fail(f"set_tests_properties names '{name}', which is not a registered "
             f"test; remove it")


def check_strict_targets(text: str, fail) -> None:
    """Every ember-owned target must be held to EMBER_STRICT.

    A target that links ember_core compiles only Ember code, so it belongs in
    EMBER_STRICT_TARGETS. Targets that build vendored engine/ sources are
    deliberately absent — upstream keeps its own warning standard — and they
    are identifiable because they never link ember_core.
    """
    m = re.search(r"set\(EMBER_STRICT_TARGETS(.*?)\)", text, re.S)
    if not m:
        fail("CMakeLists.txt: EMBER_STRICT_TARGETS is missing")
        return
    listed = set(m.group(1).split())

    owned = {"ember_core", "ember-server"}
    for tm in re.finditer(r"target_link_libraries\(\s*(\S+)([^)]*)\)", text):
        target, libs = tm.group(1), tm.group(2)
        if "ember_core" in libs.split():
            owned.add(target)

    for target in sorted(owned - listed):
        fail(f"target '{target}' links ember_core but is not in "
             f"EMBER_STRICT_TARGETS — it escapes -Werror in CI")
    for target in sorted(listed - owned):
        fail(f"EMBER_STRICT_TARGETS names '{target}', which no longer exists "
             f"or no longer links ember_core; remove it")


def main() -> int:
    text = read_cmake()
    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)

    check_source_lists(text, fail)
    check_tests_registered(text, fail)
    check_test_timeouts(text, fail)
    check_strict_targets(text, fail)

    if failures:
        print(f"{len(failures)} invariant violation(s):\n", file=sys.stderr)
        for msg in failures:
            print(f"  ✗ {msg}", file=sys.stderr)
        return 1
    print("repo invariants OK (CMake source lists in sync, all tests registered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
