#!/usr/bin/env python3
"""Collect per-release benchmark bundles into one JSON the perf page renders.

    build_perf_site_data.py --bundle <dir> [--bundle <dir> ...] --out docs/perf/data.json

Each bundle is a benchmarks/ember-<date>/ directory produced by
scripts/benchmark_bundle.sh. Re-running with more bundles is how the site gains
a release axis: nothing here is hand-maintained.
"""

import argparse
import json
from pathlib import Path


def load(bundle: Path, certified: bool):
    summary = json.loads((bundle / "summary.json").read_text())
    env = json.loads((bundle / "environment.json").read_text())
    rt = env.get("runtime", {})
    return {
        "id": rt.get("release") or bundle.name,
        "measured": env.get("measured_utc"),
        "host": env.get("host", {}),
        "model": env.get("model", {}),
        "image": rt.get("container_image"),
        "throughput": summary.get("decode", {}),
        "prefill_groups": summary.get("prefill", {}),
        "workloads": summary.get("by_workload", {}),
        "depths": summary.get("by_context_depth", []),
        "provenance": {
            "bundle": bundle.name,
            "certified": certified,
            # A bundle run with --skip-context-sweep has no depth series at all.
            # Say that, rather than let the page imply the sweep was run.
            "has_depths": bool(summary.get("by_context_depth")),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", action="append", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--certified", action="store_true",
                    help="these bundles came from release certification, not a manual run")
    args = ap.parse_args()

    releases = []
    for b in args.bundle:
        if not (b / "summary.json").exists():
            raise SystemExit(f"{b} has no summary.json")
        releases.append(load(b, args.certified))
    releases.sort(key=lambda r: (r.get("measured") or "", r["id"]))

    workloads = sorted({w for r in releases for w in r["workloads"]})
    depths = sorted({d["depth"] for r in releases for d in r["depths"]})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": 1,
        "workloads": workloads,
        "depths": depths,
        "releases": releases,
    }, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {len(releases)} release(s), "
          f"{len(workloads)} workloads, {len(depths)} depths")


if __name__ == "__main__":
    main()
