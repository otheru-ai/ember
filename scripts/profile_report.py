#!/usr/bin/env python3
"""Turn a scripts/profile_gpu.sh output directory into a roofline verdict.

The question this answers: is decode bandwidth-bound or compute-bound?

If decode sits near the memory roofline, instruction-level work in the matmul
kernels cannot move end-to-end throughput -- Amdahl caps it no matter how good
the assembly is -- and the useful direction is fusion (fewer passes over the
weights) or quantization. If decode sits well below the roofline, kernel work
has real headroom and the hotspot table says where.

Methodology notes that matter for reading the output:

  * Time comes from the trace-* passes and bytes come from the pmc-* passes.
    They are different runs of the same workload, because counter collection
    serializes dispatches: durations measured under --pmc understate real
    throughput badly and must never be the denominator of a bandwidth figure.
    This is sound only because the two runs execute the same probe; the report
    prints the dispatch counts of both so a mismatch is visible rather than
    silently averaged away.

  * Phases are separated by idle gaps, not by clock alignment between the
    client and the profiler. Every segment found is printed with its duration
    and dispatch count, so a bad split is obvious instead of load-bearing.

  * FETCH_SIZE/WRITE_SIZE were measured in KiB with Ember's ROCm 7.14 setup.
    ROCm 10 does not document those derived-counter units, so a new bundle's
    manifest can withhold the roofline verdict until a known-traffic gfx1151
    calibration is supplied explicitly with --counter-unit.

Stdlib only, matching every other script in this repo.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

# Measured, not theoretical. Strix Halo's 256-bit LPDDR5X-8000 bus is 256 GB/s
# on paper; ~212 GB/s is what this platform actually sustains, and a roofline
# drawn at the paper number would flatter every result by ~20%.
DEFAULT_PEAK_GBPS = 212.0

UNIT_SCALE = {"b": 1, "kb": 1024, "mb": 1024 * 1024}


def load_manifest(outdir):
    path = os.path.join(outdir, "manifest.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{path}: invalid profiling manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: profiling manifest must be a JSON object")
    return value


def _find(header, *candidates):
    """Locate a column by fuzzy name; rocprofv3 has renamed these across releases."""
    norm = {h.lower().replace("_", "").replace(" ", ""): h for h in header}
    for cand in candidates:
        key = cand.lower().replace("_", "").replace(" ", "")
        if key in norm:
            return norm[key]
    for cand in candidates:
        key = cand.lower().replace("_", "").replace(" ", "")
        for k, original in norm.items():
            if key in k:
                return original
    return None


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_dispatches(path):
    """Return [(start_ns, end_ns, kernel_name)] sorted by start."""
    rows = read_csv(path)
    if not rows:
        return []
    header = rows[0].keys()
    c_start = _find(header, "Start_Timestamp", "BeginNs", "start")
    c_end = _find(header, "End_Timestamp", "EndNs", "end")
    c_name = _find(header, "Kernel_Name", "KernelName", "name")
    if not (c_start and c_end and c_name):
        raise SystemExit(
            f"{path}: cannot find start/end/kernel columns in {list(header)}"
        )
    out = []
    for r in rows:
        try:
            start, end = int(float(r[c_start])), int(float(r[c_end]))
        except (TypeError, ValueError):
            continue
        if end >= start:
            out.append((start, end, (r[c_name] or "").strip()))
    out.sort()
    return out


def segment(dispatches, gap_ns):
    """Split a dispatch stream wherever the GPU went idle longer than gap_ns."""
    segments, current, last_end = [], [], None
    for d in dispatches:
        if last_end is not None and d[0] - last_end > gap_ns:
            segments.append(current)
            current = []
        current.append(d)
        last_end = max(last_end or d[1], d[1])
    if current:
        segments.append(current)
    return segments


def busy_ns(seg):
    """Union of dispatch intervals -- kernels overlap, so summing durations lies."""
    if not seg:
        return 0
    total, cur_start, cur_end = 0, seg[0][0], seg[0][1]
    for start, end, _ in seg[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    return total + (cur_end - cur_start)


def span_ns(seg):
    return (max(e for _, e, _ in seg) - min(s for s, _, _ in seg)) if seg else 0


def hotspots(seg, top):
    per = defaultdict(lambda: [0, 0])
    for start, end, name in seg:
        per[name][0] += end - start
        per[name][1] += 1
    ranked = sorted(per.items(), key=lambda kv: -kv[1][0])
    total = sum(v[0] for v in per.values()) or 1
    return [
        {
            "kernel": k,
            "total_ns": v[0],
            "calls": v[1],
            "share": v[0] / total,
        }
        for k, v in ranked[:top]
    ]


def load_counters(path):
    """Sum each counter over the last dispatch segment of a --pmc pass."""
    rows = read_csv(path)
    if not rows:
        return {}, 0
    header = rows[0].keys()
    c_name = _find(header, "Counter_Name", "counter")
    c_val = _find(header, "Counter_Value", "value")
    c_disp = _find(header, "Dispatch_Id", "Correlation_Id", "dispatch")
    if not (c_name and c_val):
        return {}, 0
    totals = defaultdict(float)
    dispatches = set()
    for r in rows:
        try:
            totals[(r[c_name] or "").strip()] += float(r[c_val])
        except (TypeError, ValueError):
            continue
        if c_disp:
            dispatches.add(r[c_disp])
    return dict(totals), len(dispatches) or len(rows)


def find_one(outdir, *patterns):
    hits = find_all(outdir, *patterns)
    return hits[0] if hits else None


def find_all(outdir, *patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(outdir, pat)))
        if hits:
            return hits
    return []


def analyse_phase(outdir, phase, gap_ns, top, unit_scale):
    trace = find_one(outdir, f"*trace-{phase}*kernel_trace.csv", f"*trace-{phase}*.csv")
    if not trace:
        return {"phase": phase, "error": f"no trace CSV for {phase}"}

    dispatches = load_dispatches(trace)
    if not dispatches:
        return {"phase": phase, "error": f"{os.path.basename(trace)} has no dispatches"}

    segs = segment(dispatches, gap_ns)
    measured = segs[-1]
    result = {
        "phase": phase,
        "trace_file": os.path.basename(trace),
        "segments": [
            {"index": i, "dispatches": len(s), "span_ms": span_ns(s) / 1e6}
            for i, s in enumerate(segs)
        ],
        "measured_segment": len(segs) - 1,
        "dispatches": len(measured),
        "busy_ms": busy_ns(measured) / 1e6,
        "span_ms": span_ns(measured) / 1e6,
        "hotspots": hotspots(measured, top),
    }
    if len(segs) < 2:
        result["warning"] = (
            "only one segment found -- warmup and the measured request were not "
            "separated; raise --gap-secs in profile_gpu.sh or lower --gap-ms here"
        )

    # gfx1151 cannot collect FETCH_SIZE and WRITE_SIZE in one pass (it SIGSEGVs
    # with "error code 38"), so the harness emits one pmc pass per counter and
    # the totals are merged here.
    pmc_files = find_all(
        outdir, f"*pmc-{phase}*counter_collection.csv", f"*pmc-{phase}*.csv"
    )
    if pmc_files:
        counters, pmc_dispatches = {}, 0
        for f in pmc_files:
            part, disp = load_counters(f)
            for k, v in part.items():
                counters[k] = counters.get(k, 0.0) + v
            pmc_dispatches = max(pmc_dispatches, disp)
        result["counter_files"] = [os.path.basename(f) for f in pmc_files]
        result["counters_raw"] = counters
        result["pmc_dispatches"] = pmc_dispatches
        moved = sum(
            v for k, v in counters.items() if k in ("FETCH_SIZE", "WRITE_SIZE")
        )
        present = [k for k in ("FETCH_SIZE", "WRITE_SIZE") if k in counters]
        if len(present) == 1:
            result["partial_counters"] = (
                f"only {present[0]} was collected; the bandwidth figure counts "
                "traffic in one direction only and is a LOWER BOUND"
            )
        if moved and result["busy_ms"] > 0:
            gb = moved * unit_scale / 1e9
            result["bytes_moved_gb"] = gb
            result["achieved_gbps"] = gb / (result["busy_ms"] / 1e3)
    return result


def verdict(decode, peak_gbps):
    got = decode.get("achieved_gbps")
    if got is None:
        return (
            "INCONCLUSIVE: no bandwidth counters were collected for decode. "
            "Rerun with FETCH_SIZE/WRITE_SIZE available, or the roofline question "
            "stays open."
        )
    frac = got / peak_gbps
    if frac >= 0.80:
        return (
            f"DECODE IS BANDWIDTH-BOUND ({got:.0f} GB/s = {frac:.0%} of {peak_gbps:.0f} "
            "GB/s). Instruction-level rewrites of the matmul kernels cannot move "
            "end-to-end decode throughput. Spend the effort on reducing bytes moved: "
            "fuse dequant into the GEMM to eliminate a pass over the weights, or "
            "revisit quantization. See the hotspot table for which kernels move them."
        )
    if frac >= 0.50:
        return (
            f"DECODE IS MIXED ({got:.0f} GB/s = {frac:.0%} of {peak_gbps:.0f} GB/s). "
            "Neither bound dominates. Attack the top hotspot first and re-measure; "
            "expect sublinear end-to-end return from kernel work alone."
        )
    return (
        f"DECODE HAS COMPUTE HEADROOM ({got:.0f} GB/s = {frac:.0%} of {peak_gbps:.0f} "
        "GB/s). Memory is not the limit, so kernel-level work can pay off. Target the "
        "top hotspot below, and confirm the fragment layout against "
        "tools/bench_wmma_decode.hip -- gfx1151 is RDNA 3.5 (16 elements/lane, A/B "
        "replicated across lanes 0-15/16-31), NOT the RDNA 4 gfx12 layout."
    )


def human(report, peak_gbps):
    out = []
    if not report.get("counter_unit_certified", True):
        out.append(
            "WARNING: counter unit is an uncalibrated manifest assumption; "
            "bandwidth values below are exploratory"
        )
    for phase in ("prefill", "decode"):
        p = report["phases"].get(phase, {})
        out.append(f"\n=== {phase.upper()} ===")
        if "error" in p:
            out.append(f"  error: {p['error']}")
            continue
        out.append(
            f"  segments found: {len(p['segments'])}"
            f" (measured #{p['measured_segment']})"
        )
        for s in p["segments"]:
            mark = " <- measured" if s["index"] == p["measured_segment"] else ""
            out.append(
                f"    [{s['index']}] {s['dispatches']:>7} dispatches"
                f"  {s['span_ms']:>10.1f} ms{mark}"
            )
        if "warning" in p:
            out.append(f"  WARNING: {p['warning']}")
        out.append(f"  GPU busy: {p['busy_ms']:.1f} ms over {p['span_ms']:.1f} ms wall")
        if "achieved_gbps" in p:
            out.append(
                f"  bytes moved: {p['bytes_moved_gb']:.2f} GB"
                f"  ->  {p['achieved_gbps']:.1f} GB/s"
                f"  ({p['achieved_gbps'] / peak_gbps:.0%} of {peak_gbps:.0f} GB/s)"
            )
            if p.get("partial_counters"):
                out.append(f"  NOTE: {p['partial_counters']}")
            if p.get("pmc_dispatches") and p["dispatches"]:
                ratio = p["pmc_dispatches"] / p["dispatches"]
                if not 0.5 <= ratio <= 2.0:
                    out.append(
                        f"  WARNING: pmc pass saw {p['pmc_dispatches']} dispatches vs "
                        f"{p['dispatches']} in the trace pass -- the two runs did not "
                        "execute the same work, so the bandwidth figure is unreliable"
                    )
        else:
            out.append("  bytes moved: unavailable (no FETCH_SIZE/WRITE_SIZE counters)")
        out.append("  top kernels:")
        for h in p["hotspots"]:
            out.append(
                f"    {h['share']:>6.1%}  {h['total_ns'] / 1e6:>9.1f} ms"
                f"  x{h['calls']:<6} {h['kernel'][:78]}"
            )
    out.append("\n=== VERDICT ===")
    out.append("  " + report["verdict"].replace(". ", ".\n  "))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", help="directory produced by scripts/profile_gpu.sh")
    ap.add_argument("--gap-ms", type=float, default=500.0,
                    help="idle gap that separates phases (default 500)")
    ap.add_argument("--top", type=int, default=12, help="hotspots to list (default 12)")
    ap.add_argument("--peak-gbps", type=float, default=DEFAULT_PEAK_GBPS,
                    help=f"measured memory roofline (default {DEFAULT_PEAK_GBPS})")
    ap.add_argument("--counter-unit", choices=sorted(UNIT_SCALE), default=None,
                    help=("calibrated unit of FETCH_SIZE/WRITE_SIZE; required to "
                          "certify a bundle whose manifest marks units uncertified"))
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.outdir):
        raise SystemExit(f"not a directory: {args.outdir}")

    manifest = load_manifest(args.outdir)
    unit_metadata = manifest.get("counter_unit", {}) if manifest else {}
    manifest_unit = unit_metadata.get("assumed", "kb")
    if manifest_unit not in UNIT_SCALE:
        raise SystemExit(f"manifest has unsupported counter unit: {manifest_unit!r}")
    counter_unit = args.counter_unit or manifest_unit
    explicit_counter_unit = args.counter_unit is not None
    unit_certified = bool(
        unit_metadata.get("release_bandwidth_verdict_certified", True)
    ) or explicit_counter_unit
    scale = UNIT_SCALE[counter_unit]
    gap_ns = int(args.gap_ms * 1e6)
    report = {
        "outdir": args.outdir,
        "peak_gbps": args.peak_gbps,
        "counter_unit": counter_unit,
        "counter_unit_source": "explicit" if explicit_counter_unit else (
            "manifest" if manifest else "legacy_default"
        ),
        "counter_unit_certified": unit_certified,
        "phases": {
            phase: analyse_phase(args.outdir, phase, gap_ns, args.top, scale)
            for phase in ("prefill", "decode")
        },
    }
    if unit_certified:
        report["verdict"] = verdict(
            report["phases"].get("decode", {}), args.peak_gbps
        )
    else:
        report["verdict"] = (
            "INCONCLUSIVE: this bundle marks FETCH_SIZE/WRITE_SIZE units "
            "uncertified for ROCm 10 on gfx1151. Calibrate with known traffic "
            "and pass the verified unit explicitly via --counter-unit before "
            "publishing a bandwidth or roofline verdict."
        )

    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(human(report, args.peak_gbps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
