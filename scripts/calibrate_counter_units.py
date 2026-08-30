#!/usr/bin/env python3
"""Fit ROCm FETCH_SIZE/WRITE_SIZE counter values to known GPU traffic.

The collector is intentionally separate from this analyzer: counter collection
is GPU- and driver-specific, while the regression is deterministic and easy to
review offline.  Input is JSONL with one record per traffic/baseline pair:
{"counter":"FETCH_SIZE", "expected_bytes":..., "traffic":..., "baseline":...}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


# ROCm counters may report bytes, fixed-size memory transactions, or aggregate
# KiB/MiB values depending on the counter definition and release.  gfx1151's
# ROCm 10 FETCH_SIZE/WRITE_SIZE measurements use 64-byte and 128-byte
# transactions respectively; keep the legacy aggregate candidates so the same
# harness remains useful across releases.
SCALES = {"b": 1, "64b": 64, "128b": 128, "kb": 1024, "mb": 1024 * 1024}


def fit(rows: list[dict[str, float]], counter: str) -> dict[str, object]:
    points = []
    for row in rows:
        if row.get("counter") != counter:
            continue
        expected = float(row["expected_bytes"])
        value = float(row["traffic"]) - float(row.get("baseline", 0.0))
        if expected <= 0 or value <= 0 or not math.isfinite(expected + value):
            raise ValueError(f"{counter}: expected_bytes and counter delta must be positive")
        points.append((value, expected))
    if len(points) < 3:
        raise ValueError(f"{counter}: need at least three traffic points")
    denom = sum(value * value for value, _ in points)
    slope = sum(value * expected for value, expected in points) / denom
    residuals = [expected - slope * value for value, expected in points]
    rmse = math.sqrt(sum(error * error for error in residuals) / len(points))
    mean = statistics.fmean(expected for _, expected in points)
    relative_rmse = rmse / mean
    candidates = {
        unit: abs(slope - scale) / scale for unit, scale in SCALES.items()
    }
    unit = min(candidates, key=candidates.get)
    return {
        "counter": counter,
        "points": len(points),
        "bytes_per_counter": slope,
        "candidate_unit": unit,
        "candidate_error": candidates[unit],
        "relative_rmse": relative_rmse,
        "certified": candidates[unit] <= 0.02 and relative_rmse <= 0.02,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path, help="JSONL calibration samples")
    parser.add_argument("--output", type=Path, help="optional JSON output")
    args = parser.parse_args(argv)
    try:
        rows = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
        result = {counter: fit(rows, counter) for counter in ("FETCH_SIZE", "WRITE_SIZE")}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"counter calibration: {error}", file=sys.stderr)
        return 1
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        sys.stdout.write(payload)
    if not all(item["certified"] for item in result.values()):
        print("counter calibration: no unit passed the 2% fit gates", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
