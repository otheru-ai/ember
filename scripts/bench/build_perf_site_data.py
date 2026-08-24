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

from assemble_bundle import summarise_context


def norm_decode(d):
    """Aggregate decode, from either bundle vintage.

    The 2026-08-02 bundle names these decode_tps_median/min/max and
    accept_rate_median; everything since uses median_tps/min_tps/max_tps and
    median_accept_rate. Both are the same measurement -- median of N samples of
    a fixed-length greedy generation -- so they belong on one axis.
    """
    if not d:
        return {}
    if "median_tps" in d:
        return d
    return {
        "median_tps": d.get("decode_tps_median"),
        "min_tps": d.get("decode_tps_min"),
        "max_tps": d.get("decode_tps_max"),
        "median_accept_rate": d.get("accept_rate_median"),
        "completion_tokens": d.get("tokens_per_sample"),
        "samples": d.get("samples"),
    }


def norm_prefill(p):
    """Prefill groups as a mapping. The oldest bundle emits a list of records
    carrying their own "group" key; later ones are already keyed by it."""
    if isinstance(p, list):
        return {r["group"]: {k: v for k, v in r.items() if k != "group"} for r in p}
    return p or {}


def norm_host(env):
    """The oldest bundle describes the machine under "hardware" with different
    field names. Map it onto the current shape so the header reads the same."""
    if env.get("host"):
        return env["host"]
    h = env.get("hardware") or {}
    if not h:
        return {}
    return {
        "cpu": h.get("system") or h.get("cpu"),
        "gpu": h.get("gpu"),
        "kernel": h.get("kernel"),
        "os": h.get("os"),
        "memory_gib": h.get("os_visible_memory_gib") or h.get("unified_memory_gb"),
    }


def context_depths(bundle: Path, summary: dict):
    """Prefer the summary. Fall back to the bundle's raw context-sweep.jsonl,
    which is what a bundle assembled before summarise_context existed still has
    on disk -- deriving it here beats leaving the depth chart empty."""
    if summary.get("by_context_depth"):
        return summary["by_context_depth"]
    raw = bundle / "context-sweep.jsonl"
    if not raw.exists():
        return []
    rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
    if not rows or "target" not in rows[0]:
        return []
    return summarise_context(rows)


def load(bundle: Path, certified: bool):
    summary = json.loads((bundle / "summary.json").read_text())
    env = json.loads((bundle / "environment.json").read_text())
    rt = env.get("runtime", {})
    depths = context_depths(bundle, summary)
    release = rt.get("release") or (rt.get("git_commit") or "")[:12] or bundle.name
    return {
        "id": release,
        "measured": env.get("measured_utc") or (env.get("benchmark") or {}).get("date"),
        "host": norm_host(env),
        "model": env.get("model") or env.get("target_model") or {},
        "image": rt.get("container_image"),
        "throughput": norm_decode(summary.get("decode")),
        "prefill_groups": norm_prefill(summary.get("prefill")),
        "workloads": summary.get("by_workload") or {},
        "depths": depths,
        "provenance": {
            "bundle": bundle.name,
            "certified": certified,
            "has_depths": bool(depths),
            "depths_from": ("summary" if summary.get("by_context_depth")
                            else "context-sweep.jsonl" if depths else None),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", action="append", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--certified", action="store_true",
                    help="these bundles came from release certification, not a manual run")
    ap.add_argument("--id", action="append", default=[], metavar="OLD=NEW",
                    help="rename a release, e.g. --id a1b2c3d4e5f6=2026.8.24. CI labels "
                         "bundles with the commit SHA; the release gives them its version.")
    args = ap.parse_args()

    releases = []
    for b in args.bundle:
        if not (b / "summary.json").exists():
            raise SystemExit(f"{b} has no summary.json")
        releases.append(load(b, args.certified))

    renames = dict(kv.split("=", 1) for kv in args.id)
    for r in releases:
        if r["id"] in renames:
            r["id"] = renames[r["id"]]
    # Order by release, not by when it was measured. Re-benchmarking an old
    # release against the current model gives it today's date, and 2026.8.10
    # measured after 2026.8.23 is still the older release.
    def key(r):
        parts = r["id"].split(".")
        if all(x.isdigit() for x in parts):
            return (0, tuple(int(x) for x in parts), "")
        return (1, (), r.get("measured") or "")
    releases.sort(key=key)

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
