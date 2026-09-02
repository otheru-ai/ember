#!/usr/bin/env python3
"""Release-over-release delta for a benchmark bundle.

The ship/no-ship question is whether a candidate regresses the metrics an
existing release already published, so this compares like for like and refuses
to compare anything else.

It does not decide ship/no-ship. It reports every metric with its delta and
flags regressions beyond --tolerance; the call is the reader's.

Fails closed on: an unknown baseline id, a workload or depth set that differs
between the two, and a missing metric on either side. A quietly dropped
workload is exactly how a regression disappears from a comparison.
"""
import argparse, json, sys
from pathlib import Path

W_METRICS = ("tok_s", "autoregressive_tok_s", "speedup")
D_METRICS = ("decode_tok_s", "prefill_tok_s", "total_tok_s", "accept_rate",
             "speedup", "ttft_ms")
LOWER_IS_BETTER = {"ttft_ms"}


def pct(new, old):
    if old in (None, 0): return None
    return (new - old) / old * 100.0


def regressed(metric, new, old, tol):
    d = pct(new, old)
    if d is None: return False
    return (d > tol) if metric in LOWER_IS_BETTER else (d < -tol)


def load_release(data, rid):
    for r in data["releases"]:
        if r["id"] == rid: return r
    raise SystemExit(f"baseline id {rid!r} not in data.json "
                     f"(have: {', '.join(r['id'] for r in data['releases'])})")


def table(title, rows, tol):
    if not rows: return []
    print(f"\n== {title}")
    print(f"{'key':22s} {'metric':22s} {'baseline':>11s} {'candidate':>11s} {'delta':>9s}")
    bad = []
    for key, metric, old, new in rows:
        d = pct(new, old)
        flag = ""
        if regressed(metric, new, old, tol):
            flag = "  REGRESSION"; bad.append((key, metric, old, new, d))
        ds = "n/a" if d is None else f"{d:+7.2f}%"
        print(f"{key:22s} {metric:22s} {old:11.4g} {new:11.4g} {ds:>9s}{flag}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="docs/perf/data.json")
    ap.add_argument("--baseline", required=True, help="release id to compare against")
    ap.add_argument("--candidate", required=True,
                    help="a release id in --data, or a bundle summary .json")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="percent a metric may move against us before it is flagged")
    a = ap.parse_args()

    data = json.loads(Path(a.data).read_text())
    base = load_release(data, a.baseline)
    p = Path(a.candidate)
    if p.exists() and p.suffix == ".json":
        cand = json.loads(p.read_text())
        cand = cand.get("release", cand)
    else:
        cand = load_release(data, a.candidate)

    print(f"baseline  {base['id']}  measured {base.get('measured')}")
    print(f"candidate {cand.get('id', a.candidate)}  measured {cand.get('measured')}")

    bw, cw = base.get("workloads", {}), cand.get("workloads", {})
    if set(bw) != set(cw):
        raise SystemExit(
            f"workload sets differ; refusing to compare.\n"
            f"  only in baseline:  {sorted(set(bw) - set(cw))}\n"
            f"  only in candidate: {sorted(set(cw) - set(bw))}")
    rows = []
    for k in sorted(bw):
        for m in W_METRICS:
            if m not in bw[k] or m not in cw[k]:
                raise SystemExit(f"workload {k}: metric {m} missing on one side")
            rows.append((k, m, bw[k][m], cw[k][m]))
    bad = table("workloads", rows, a.tolerance)

    bd = {d["depth"]: d for d in base.get("depths", [])}
    cd = {d["depth"]: d for d in cand.get("depths", [])}
    if set(bd) != set(cd):
        raise SystemExit(f"depth sets differ; refusing to compare. "
                         f"baseline={sorted(bd)} candidate={sorted(cd)}")
    rows = []
    for k in sorted(bd):
        for m in D_METRICS:
            if m in bd[k] and m in cd[k]:
                rows.append((f"depth {k}", m, bd[k][m], cd[k][m]))
    bad += table("depths", rows, a.tolerance)

    bt, ct = base.get("throughput", {}), cand.get("throughput", {})
    rows = [("throughput", m, bt[m], ct[m])
            for m in ("median_tps", "min_tps", "max_tps",
                      "median_accept_rate", "min_accept_rate")
            if m in bt and m in ct]
    bad += table("throughput", rows, a.tolerance)

    print(f"\n{len(bad)} regression(s) beyond {a.tolerance}%")
    for key, metric, old, new, d in sorted(bad, key=lambda x: x[4]):
        print(f"  {key:22s} {metric:22s} {old:.4g} -> {new:.4g}  {d:+.2f}%")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
