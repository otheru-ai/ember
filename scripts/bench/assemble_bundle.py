#!/usr/bin/env python3
"""Turn a directory of raw benchmark output into a publishable bundle.

Reads whatever the harnesses left in the bundle directory and writes
environment.json, summary.json and README.md. Kept separate from the
orchestrator so the assembly can be re-run, and unit-tested, without a GPU.

    assemble_bundle.py <bundle-dir>

Configuration arrives through the environment (RELEASE, TARGET_SHA, ...) rather
than a long argv, because the orchestrator already holds those values.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def host_facts(bundle):
    """Prefer facts captured at collection time.

    Detecting the host at assembly time is only correct if assembly happens on
    the machine that ran the benchmark. Re-assembling a bundle anywhere else
    would silently attribute the numbers to the wrong hardware, which for a
    published artifact is worse than failing. The orchestrator writes host.json
    on the benchmark host; this falls back to live detection only when it is
    absent.
    """
    captured = bundle / "host.json"
    if captured.exists():
        return json.loads(captured.read_text())

    def run(cmd, default=""):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=10).stdout.strip() or default
        except Exception:
            return default

    cpu = run("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-").strip()
    mem_kb = run("awk '/MemTotal/{print $2}' /proc/meminfo", "0")
    os_name = run(". /etc/os-release 2>/dev/null && echo $PRETTY_NAME")
    return {
        "cpu": cpu or platform.processor(),
        "gpu": "AMD Radeon 8060S (gfx1151), integrated",
        "kernel": platform.release(),
        "memory_gib": round(int(mem_kb) / 1024 / 1024) if mem_kb.isdigit() else None,
        "os": os_name,
    }


def summarise_groups(rows):
    """Median/min/max per benchmark.py group, matching earlier bundles."""
    out = {"prefill": {}, "decode": {}}
    dec = [r for r in rows if r.get("group") == "decode-256" and r.get("ok")]
    tps = [r["decode_tokens_per_second"] for r in dec if r.get("decode_tokens_per_second")]
    acc = [r["accept_rate"] for r in dec if r.get("accept_rate") is not None]
    if tps:
        out["decode"] = {
            "samples": len(tps),
            "median_tps": round(statistics.median(tps), 2),
            "min_tps": round(min(tps), 2),
            "max_tps": round(max(tps), 2),
            "completion_tokens": dec[0].get("completion_tokens"),
            # benchmark.py flattens usage.backend, so spec_ran is top level.
            "spec_ran": sum(1 for r in dec if r.get("spec_ran")),
        }
        if acc:
            out["decode"]["median_accept_rate"] = round(statistics.median(acc), 3)
            out["decode"]["min_accept_rate"] = round(min(acc), 3)
    groups = {}
    for r in rows:
        g = r.get("group", "")
        if g.startswith("prefill-") and r.get("prefill_tokens_per_second") \
                and (r.get("evaluated_prefill_tokens") or 0) > 0:
            groups.setdefault(g, []).append(r)
    for g, rs in groups.items():
        v = [r["prefill_tokens_per_second"] for r in rs]
        out["prefill"][g] = {
            "samples": len(v),
            "evaluated_prompt_tokens": rs[0].get("evaluated_prefill_tokens"),
            "median_tps": round(statistics.median(v), 1),
            "min_tps": round(min(v), 1),
            "max_tps": round(max(v), 1),
        }
    return out


def summarise_workloads(on_rows, off_rows):
    off = {r["label"]: r for r in off_rows if r.get("decode_tps")}
    out = {}
    for r in on_rows:
        if not r.get("decode_tps"):
            continue
        base = off.get(r["label"], {}).get("decode_tps")
        out[r["label"]] = {
            "tok_s": round(r["decode_tps"], 2),
            "autoregressive_tok_s": round(base, 2) if base else None,
            "speedup": round(r["decode_tps"] / base, 3) if base else None,
        }
    return out


def summarise_context(rows):
    """Depth series keyed on the requested depth, not the measured prompt length.

    sweep_probe.py sizes its filler with a words-per-token estimate that drifts
    badly at depth -- a target of 65536 measures 77068 tokens -- so keying on the
    measurement puts unrepeatable numbers like 77068 on the axis instead of the
    64k that was asked for, and makes two releases impossible to line up. It also
    silently drops a point whenever the two configs happen to tokenise
    differently, since the spec-off lookup would miss.

    The measured count is kept as prompt_tokens, and every rate that depends on
    how much text was really prefilled still uses it.
    """
    on = {r["target"]: r for r in rows if r["config"] == "spec-on"}
    off = {r["target"]: r for r in rows if r["config"] == "spec-off"}
    out = []
    for t in sorted(on):
        r = on[t]
        n = r["prompt_tokens"]
        p, c = r["prefill_tps"], r["decode_tps"]
        b = off.get(t, {}).get("decode_tps")
        ttft = n / p if p else None
        out.append({
            "depth": t,
            "prompt_tokens": n,
            "prefill_tok_s": round(p, 1),
            "decode_tok_s": round(c, 2),
            "autoregressive_tok_s": round(b, 2) if b else None,
            "speedup": round(c / b, 3) if b else None,
            "ttft_ms": round(ttft * 1000) if ttft else None,
            "total_tok_s": round((n + 256) / (ttft + 256 / c), 1) if ttft else None,
            "accept_rate": r.get("accept"),
        })
    return out


def main():
    bundle = Path(sys.argv[1])
    env = os.environ
    rows = jsonl(bundle / "raw-results.jsonl")
    wl_on = jsonl(bundle / "workloads-spec-on.jsonl")
    wl_off = jsonl(bundle / "workloads-spec-off.jsonl")
    ctx = jsonl(bundle / "context-sweep.jsonl")

    summary = summarise_groups(rows)
    benchmark_summaries = [r for r in rows if r.get("kind") == "summary"]
    if benchmark_summaries and benchmark_summaries[-1].get("hard_gate"):
        summary["hard_gate"] = benchmark_summaries[-1]["hard_gate"]
        summary["prefill_calibration"] = benchmark_summaries[-1].get(
            "prefill_calibration")
    if wl_on:
        summary["by_workload"] = summarise_workloads(wl_on, wl_off)
        # A file of connection errors still parses as JSONL and used to assemble
        # into a bundle with an empty by_workload and no complaint. Refuse it:
        # a bundle missing its headline table is worse than a failed run.
        if not summary["by_workload"]:
            errs = [r.get("error") for r in wl_on if r.get("error")]
            raise SystemExit(
                f"workload sweep produced no usable rows "
                f"({len(errs)} of {len(wl_on)} errored; first: {errs[0] if errs else 'n/a'})")
    if ctx:
        summary["by_context_depth"] = summarise_context(ctx)

    environment = {
        "measured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "host": host_facts(bundle),
        "model": {
            "target": Path(env.get("TARGET", "")).name,
            "target_sha256": env.get("TARGET_SHA"),
            "drafter": Path(env.get("DRAFT", "")).name,
            "drafter_sha256": env.get("DRAFT_SHA"),
        },
        "runtime": {
            "engine": "ember (ember-dflash)",
            "release": env.get("RELEASE"),
            "container_image": env.get("IMAGE"),
            "binary": env.get("BIN"),
        },
        "server_args": (f"ember-dflash -m <target> --host 127.0.0.1 "
                        f"--port {env.get('PORT')} --max-ctx 65536 "
                        f"--ds4-expert-top-k 4 --default-temperature 0.6"),
        "notes": [
            "Per-token instrumentation DISABLED (DFLASH_DS4_TIMING=0, EMBER_GTT_TRACE=0).",
            "Single request at a time; no concurrency.",
            "Prefill mode sparse (the production serving configuration), not "
            "exact-prefill reference.",
            "Decode is workload-dependent. See by_workload in summary.json: the "
            "throughput groups use a highly predictable generation task and sit "
            "at the top of the range.",
        ],
    }

    (bundle / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    (bundle / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    rel = env.get("RELEASE")
    date = environment["measured_utc"]
    lines = [f"# Ember {rel} performance bundle", "",
             f"Measured {date} on {environment['host']['cpu']}, "
             f"{environment['host']['memory_gib']} GiB unified memory, "
             f"{environment['host']['os']}.", "",
             "## Contents", "",
             "| File | What it is |", "| --- | --- |"]
    # Describe only what is actually here. A run with --skip-context-sweep used
    # to advertise a chart and a sweep it had not produced.
    catalogue = [
        ("raw-results.jsonl", "Every throughput request, unaggregated"),
        ("workloads-spec-on.jsonl", "Decode across ten generation tasks, speculation on"),
        ("workloads-spec-off.jsonl", "The same ten tasks, autoregressive baseline"),
        ("context-sweep.jsonl", "Prefill and decode against prompt depth, both configurations"),
        ("ember-context-scaling.svg", "Chart generated from `context-sweep.jsonl`"),
        ("summary.json", "Aggregates: medians, ranges, speedups"),
        ("environment.json", "Host, model SHA-256, runtime release, flags"),
        ("host.json", "Host facts captured on the machine that ran the benchmark"),
        ("benchmark.py", "Throughput harness"),
        ("accept_sweep.py", "Workload harness"),
        ("sweep_probe.py", "Context-depth harness"),
    ]
    for name, what in catalogue:
        if (bundle / name).exists():
            lines.append(f"| `{name}` | {what} |")
    lines.append("")
    if summary.get("decode"):
        d = summary["decode"]
        lines += [f"Decode on the throughput group: **{d['median_tps']} tok/s** median "
                  f"of {d['samples']} samples ({d['min_tps']}-{d['max_tps']}).", ""]
    if summary.get("by_workload"):
        w = summary["by_workload"]
        best = max(w.values(), key=lambda r: r["tok_s"])
        worst = min(w.values(), key=lambda r: r["tok_s"])
        lines += [f"Across ten workloads decode spans **{worst['tok_s']}-{best['tok_s']} tok/s**. "
                  "That spread is the honest headline; a single figure taken from the "
                  "throughput group above describes only the most predictable case.", ""]
    lines += ["## Reproducing", "",
              "```bash", f"scripts/benchmark_bundle.sh --out ./out --release {rel}", "```", "",
              "Needs the GPU, the model pair whose SHA-256 values are recorded in "
              "`environment.json`, and exclusive use of the machine.", ""]
    (bundle / "README.md").write_text("\n".join(lines))
    print(f"  wrote environment.json, summary.json, README.md")


if __name__ == "__main__":
    main()
