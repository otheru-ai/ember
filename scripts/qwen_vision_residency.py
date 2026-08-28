#!/usr/bin/env python3
"""Phase-aware RSS/GTT/UMA sampler for the Qwen real-weight vision gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time


PHASES = ("idle", "cold", "warm", "settled")
METHOD = "runner_phase_rss_gtt_uma_sampler_v1"


def meminfo(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        fields = raw.split()
        if fields:
            result[key] = int(fields[0]) * (1024 if fields[1:] == ["kB"] else 1)
    return result


def sample(pid: int, phase: str, gtt_paths: list[Path]) -> dict[str, int | float | str]:
    result: dict[str, int | float | str] = {
        "monotonic": time.monotonic(), "phase": phase}
    values = meminfo(Path("/proc/meminfo"))
    result["memtotal_bytes"] = values["MemTotal"]
    result["mem_available_bytes"] = values["MemAvailable"]
    result["uma_used_bytes"] = max(0, values["MemTotal"] - values["MemAvailable"])
    for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            result["rss_bytes"] = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            result["hwm_bytes"] = int(line.split()[1]) * 1024
    gtt = []
    for path in gtt_paths:
        try:
            gtt.append(int(path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            pass
    if gtt:
        result["gtt_bytes"] = max(gtt)
    return result


def summarize(rows: list[dict[str, int | float | str]], pid: int) -> dict:
    phases = {}
    for phase in PHASES:
        selected = [row for row in rows if row["phase"] == phase]
        def numbers(key: str) -> list[int]:
            return [int(row[key]) for row in selected if key in row]
        phases[phase] = {
            "samples": len(selected),
            "rss_peak_bytes": max(numbers("rss_bytes") + numbers("hwm_bytes"),
                                  default=None),
            "gtt_peak_bytes": max(numbers("gtt_bytes"), default=None),
            "uma_used_peak_bytes": max(numbers("uma_used_bytes"), default=None),
            "mem_available_floor_bytes": min(numbers("mem_available_bytes"), default=None),
        }
    try:
        pages_limit = int(Path("/sys/module/ttm/parameters/pages_limit").read_text().strip())
    except (OSError, ValueError):
        pages_limit = None
    memtotal = [int(row["memtotal_bytes"]) for row in rows if "memtotal_bytes" in row]
    return {
        "schema": "ember.qwen3.8.vision-residency.v1",
        "method": METHOD,
        "server_host_pid": pid,
        "sample_interval_seconds": None,
        "runner_memtotal_bytes": int(statistics.median(memtotal)) if memtotal else None,
        "runner_gtt_pages_limit": pages_limit,
        "phases": phases,
        "raw_samples": rows,
    }


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--phase-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.05)
    args = parser.parse_args()
    if args.pid <= 1 or args.interval <= 0 or not Path(f"/proc/{args.pid}/status").is_file():
        parser.error("--pid must be one live host process and interval must be positive")
    gtt_paths = sorted(Path("/sys/class/drm").glob("card*/device/mem_info_gtt_used"))
    if not gtt_paths:
        parser.error("no amdgpu GTT telemetry path exists")
    rows = []
    while not args.stop_file.exists():
        try:
            phase = args.phase_file.read_text(encoding="utf-8").strip()
            if phase not in PHASES:
                raise ValueError("unknown phase")
            rows.append(sample(args.pid, phase, gtt_paths))
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"resource sampling failed: {exc}")
        time.sleep(args.interval)
    value = summarize(rows, args.pid)
    value["sample_interval_seconds"] = args.interval
    write_new(args.output.absolute(), value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
