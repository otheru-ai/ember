#!/usr/bin/env python3
"""Export docs/perf/data.json as Prometheus textfile gauges.

    perf_textfile.py --data docs/perf/data.json --out ember-bench.prom

Benchmark bundles are per-release measurements, not a time series, so they
never reach the scrape path on their own. Written through node_exporter's
textfile collector they become labelled gauges -- one sample per release,
workload, prefill group and context depth -- that Grafana can put beside the
live ember_* series, which is where a "did the release get slower" question is
actually asked. Every value is copied from the bundle summary as published;
nothing here measures anything.

Series carry the `release` label verbatim (e.g. 2026.9.11) plus `seq`, a
zero-padded ordinal of the release's position in data.json, because CalVer
sorts wrong lexically (2026.9.11 < 2026.9.8) and Prometheus label sorting is
lexical. `ember_bench_release_info` carries the provenance; the `certified`
label is the bundle's own claim and must not be read as more than that.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys


def esc(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def labels(**kv: object) -> str:
    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in kv.items() if v is not None) + "}"


def num(value: object) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return repr(float(value)) if isinstance(value, float) else str(value)
    return None


def export(data: dict) -> str:
    out: list[str] = []
    helped: set[str] = set()

    def sample(name: str, help_text: str, value: object, **lbl: object) -> None:
        v = num(value)
        if v is None:
            return
        if name not in helped:
            out.append(f"# HELP {name} {help_text}")
            out.append(f"# TYPE {name} gauge")
            helped.add(name)
        out.append(f"{name}{labels(**lbl)} {v}")

    releases = data.get("releases") or []
    for seq, r in enumerate(releases):
        rel = r.get("id")
        if not rel:
            continue
        base = {"release": rel, "seq": f"{seq:03d}"}
        prov = r.get("provenance") or {}
        model = r.get("model") or {}
        sample("ember_bench_release_info",
               "One per benchmarked release; value is always 1. Labels are the bundle's provenance.",
               1, **base, image=r.get("image"), measured=r.get("measured"),
               certified=str(bool(prov.get("certified"))).lower(),
               bundle=prov.get("bundle"), target=model.get("target"), drafter=model.get("drafter"))
        measured = r.get("measured")
        if isinstance(measured, str):
            try:
                ts = dt.datetime.strptime(measured, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
                sample("ember_bench_measured_timestamp_seconds",
                       "Unix time of the bundle's measurement date (UTC midnight).",
                       int(ts.timestamp()), **base)
            except ValueError:
                pass

        t = r.get("throughput") or {}
        for stat in ("median", "min", "max"):
            sample("ember_bench_throughput_tokens_per_second",
                   "Decode throughput of the fixed-length greedy generation, by statistic over samples.",
                   t.get(f"{stat}_tps"), **base, stat=stat)
        for stat in ("median", "min"):
            sample("ember_bench_accept_rate",
                   "DSpark draft acceptance rate over the throughput samples.",
                   t.get(f"{stat}_accept_rate"), **base, stat=stat)
        sample("ember_bench_throughput_samples", "Samples behind the throughput statistics.",
               t.get("samples"), **base)
        sample("ember_bench_throughput_spec_ran", "Throughput samples in which speculation ran.",
               t.get("spec_ran"), **base)

        for wl, w in sorted((r.get("workloads") or {}).items()):
            sample("ember_bench_workload_tokens_per_second",
                   "Decode tok/s per workload; mode=spec is the shipped configuration, mode=autoregressive the same prompt with the drafter off.",
                   w.get("tok_s"), **base, workload=wl, mode="spec")
            sample("ember_bench_workload_tokens_per_second", "", w.get("autoregressive_tok_s"),
                   **base, workload=wl, mode="autoregressive")
            sample("ember_bench_workload_prefill_tokens_per_second",
                   "Prefill tok/s per workload, by mode.",
                   w.get("prefill_tok_s"), **base, workload=wl, mode="spec")
            sample("ember_bench_workload_prefill_tokens_per_second", "", w.get("autoregressive_prefill_tok_s"),
                   **base, workload=wl, mode="autoregressive")
            sample("ember_bench_workload_speedup",
                   "Speculative over autoregressive decode ratio per workload; below 1 means the drafter cost more than it saved.",
                   w.get("speedup"), **base, workload=wl)

        for grp, g in sorted((r.get("prefill_groups") or {}).items()):
            for stat in ("median", "min", "max"):
                sample("ember_bench_prefill_group_tokens_per_second",
                       "Prefill tok/s by prompt-size group and statistic.",
                       g.get(f"{stat}_tps"), **base, group=grp, stat=stat,
                       prompt_tokens=g.get("evaluated_prompt_tokens"))

        for d in r.get("depths") or []:
            depth = d.get("depth")
            if depth is None:
                continue
            dl = dict(base, depth=str(depth))
            for kind, key in (("decode", "decode_tok_s"), ("prefill", "prefill_tok_s"),
                              ("autoregressive", "autoregressive_tok_s"), ("total", "total_tok_s")):
                sample("ember_bench_depth_tokens_per_second",
                       "Context-depth sweep: tok/s at a given resident KV depth, by kind.",
                       d.get(key), **dl, kind=kind)
            sample("ember_bench_depth_accept_rate", "Draft acceptance rate at each context depth.",
                   d.get("accept_rate"), **dl)
            sample("ember_bench_depth_speedup", "Speculative speedup at each context depth.",
                   d.get("speedup"), **dl)
            sample("ember_bench_depth_prompt_tokens", "Prompt tokens used to reach each depth.",
                   d.get("prompt_tokens"), **dl)

        v = r.get("vision") or {}
        for key, help_text in (("warm_decode_tps", "Vision probe: warm decode tok/s."),
                               ("cold_wall_seconds", "Vision probe: cold end-to-end wall seconds."),
                               ("warm_wall_seconds", "Vision probe: warm end-to-end wall seconds."),
                               ("prompt_tokens", "Vision probe: prompt tokens including image tokens."),
                               ("image_bytes", "Vision probe: image size in bytes."),
                               ("samples", "Vision probe: samples.")):
            sample(f"ember_bench_vision_{key}", help_text, v.get(key), **base)
        for key, name in (("cold_prefill_ms", "cold_prefill_seconds"),
                          ("warm_prefill_ms", "warm_prefill_seconds")):
            if isinstance(v.get(key), (int, float)) and not isinstance(v.get(key), bool):
                sample(f"ember_bench_vision_{name}",
                       f"Vision probe: {name.replace('_', ' ')} (bundle records ms; converted).",
                       v[key] / 1000.0, **base)
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path,
                    help="written atomically (temp file + rename), as the textfile collector expects")
    args = ap.parse_args(argv)
    data = json.loads(args.data.read_text())
    text = export(data)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(args.out)
    n = sum(1 for line in text.splitlines() if line and not line.startswith("#"))
    print(f"{args.out}: {n} samples across {len(data.get('releases') or [])} releases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
