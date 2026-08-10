#!/usr/bin/env python3
"""Per-file line-coverage report with a ratchet.

Whole-repo coverage percentages hide exactly the regressions worth catching: a
new untested 400-line module barely moves an 80% aggregate. So the floor is
per-file, checked in as ci/coverage_floors.json, and CI fails when any file
drops below its recorded floor.

  ci/coverage.py --build <dir>            # report + enforce floors
  ci/coverage.py --build <dir> --update   # re-baseline (raises floors only)

The ratchet only ever moves up: --update refuses to lower a floor, so a commit
that loses coverage cannot quietly rewrite the baseline. Deliberately lowering
one is an explicit edit of the JSON, which shows up in review.

Requires a build configured with --coverage (see .forgejo/workflows/ci.yml).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLOORS = Path(__file__).resolve().parent / "coverage_floors.json"

# Coverage is only meaningful for code the GPU-free suite can reach. The ROCm
# bridge and the vendored engine are excluded: no runner has the toolchain.
INCLUDE_PREFIX = "src/"
EXCLUDE = {"src/backend/backend_dflash.cc"}


def run_gcov(build: Path) -> dict[str, tuple[float, int]]:
    """Map repo-relative source path -> (line coverage %, executable lines)."""
    gcda = sorted(build.rglob("*.gcda"))
    if not gcda:
        sys.exit(f"no .gcda under {build}: configure with --coverage and run ctest first")

    results: dict[str, tuple[float, int]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for path in gcda:
            proc = subprocess.run(
                ["gcov", "-b", str(path)],
                cwd=tmp, capture_output=True, text=True,
            )
            for block in re.split(r"\n(?=File ')", proc.stdout):
                head = re.match(r"File '([^']+)'", block)
                lines = re.search(r"Lines executed:([\d.]+)% of (\d+)", block)
                if not head or not lines:
                    continue
                src = Path(head.group(1))
                try:
                    rel = str(src.resolve().relative_to(ROOT))
                except ValueError:
                    continue          # system or engine header
                if not rel.startswith(INCLUDE_PREFIX) or rel in EXCLUDE:
                    continue
                pct, total = float(lines.group(1)), int(lines.group(2))
                # A file compiled into several TUs is reported once per TU, and
                # a header's inline functions instantiate differently in each.
                # Keep the most complete instantiation (largest executable-line
                # count) — picking the highest percentage instead would report
                # a header as 100% covered off the one TU that inlined 5 of its
                # 58 lines.
                prev = results.get(rel)
                if prev is None or (total, pct) > (prev[1], prev[0]):
                    results[rel] = (pct, total)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--update", action="store_true",
                    help="raise floors to current coverage (never lowers)")
    args = ap.parse_args()

    cov = run_gcov(args.build)
    floors = json.loads(FLOORS.read_text()) if FLOORS.exists() else {}

    width = max(len(p) for p in cov)
    print(f"{'line%':>7} {'lines':>6} {'floor':>7}  file")
    total_lines = covered = 0
    failures: list[str] = []

    for rel in sorted(cov, key=lambda r: cov[r][0]):
        pct, total = cov[rel]
        total_lines += total
        covered += total * pct / 100
        floor = floors.get(rel)
        mark = ""
        if floor is not None and pct + 1e-9 < floor:
            mark = "  <-- BELOW FLOOR"
            failures.append(
                f"{rel}: {pct:.1f}% is below its {floor:.1f}% floor "
                f"({total} executable lines)")
        shown = f"{floor:7.1f}" if floor is not None else "      -"
        print(f"{pct:7.1f} {total:6d} {shown}  {rel.ljust(width)}{mark}")

    print(f"\nweighted total: {100 * covered / total_lines:.1f}% "
          f"({int(covered)}/{total_lines} executable lines)")

    if args.update:
        merged = dict(floors)
        raised = 0
        for rel, (pct, _) in cov.items():
            new = round(pct - 0.5, 1)         # small slack for gcov jitter
            if new > merged.get(rel, -1.0):
                merged[rel] = new
                raised += 1
        FLOORS.write_text(json.dumps(dict(sorted(merged.items())), indent=2) + "\n")
        print(f"\nwrote {FLOORS.relative_to(ROOT)} ({raised} floor(s) raised)")
        return 0

    untracked = sorted(set(cov) - set(floors))
    if untracked:
        print("\nnew files without a floor (run --update to record one):")
        for rel in untracked:
            print(f"  {rel}")

    if failures:
        print(f"\n{len(failures)} coverage regression(s):", file=sys.stderr)
        for msg in failures:
            print(f"  ✗ {msg}", file=sys.stderr)
    if failures or untracked:
        return 1
    print("\ncoverage floors OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
