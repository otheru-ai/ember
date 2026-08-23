#!/usr/bin/env python3
"""Reproducible HTTP performance sweep for a running Ember server."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


class ResourceSampler:
    def __init__(self, pid: int | None, interval: float = 0.5) -> None:
        self.pid = pid
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples: list[dict[str, float]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        gpu_paths = sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))
        while not self.stop_event.is_set():
            sample: dict[str, float] = {"monotonic": time.monotonic()}
            try:
                meminfo = {}
                for line in Path("/proc/meminfo").read_text().splitlines():
                    key, value = line.split(":", 1)
                    meminfo[key] = float(value.strip().split()[0])
                sample["mem_available_kib"] = meminfo.get("MemAvailable", 0.0)
            except (OSError, ValueError):
                pass
            if self.pid:
                try:
                    for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            sample["rss_kib"] = float(line.split()[1])
                        elif line.startswith("VmHWM:"):
                            sample["hwm_kib"] = float(line.split()[1])
                except (OSError, ValueError):
                    pass
            for path in gpu_paths:
                try:
                    value = float(path.read_text().strip())
                except (OSError, ValueError):
                    continue
                sample["gpu_busy_pct"] = max(value, sample.get("gpu_busy_pct", 0.0))
            self.samples.append(sample)
            self.stop_event.wait(self.interval)

    def summary(self) -> dict[str, float | int | None]:
        def values(key: str) -> list[float]:
            return [sample[key] for sample in self.samples if key in sample]

        rss = values("rss_kib")
        hwm = values("hwm_kib")
        available = values("mem_available_kib")
        busy = values("gpu_busy_pct")
        return {
            "samples": len(self.samples),
            "rss_gib_max": max(rss) / 1024**2 if rss else None,
            "hwm_gib_max": max(hwm) / 1024**2 if hwm else None,
            "mem_available_gib_min": min(available) / 1024**2 if available else None,
            "gpu_busy_pct_mean": statistics.fmean(busy) if busy else None,
            "gpu_busy_pct_max": max(busy) if busy else None,
        }


class Suite:
    def __init__(self, endpoint: str, output: Path, timeout: float) -> None:
        self.endpoint = endpoint
        self.output = output
        self.timeout = timeout
        self.records: list[dict] = []
        self.lock = threading.Lock()
        self.output.write_text("")

    def emit(self, record: dict) -> None:
        with self.lock:
            self.records.append(record)
            line = json.dumps(record, sort_keys=True)
            with self.output.open("a") as handle:
                handle.write(line + "\n")
            print(line, flush=True)

    def request(
        self,
        label: str,
        prompt: str,
        max_tokens: int,
        *,
        group: str,
        repeat: int,
    ) -> dict:
        payload = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "none",
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        encoded = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.endpoint,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.load(response)
            wall = time.perf_counter() - started
            usage = body.get("usage") or {}
            timings = usage.get("timings") or {}
            backend = usage.get("backend") or {}
            choice = (body.get("choices") or [{}])[0]
            record = {
                "kind": "request",
                "group": group,
                "label": label,
                "repeat": repeat,
                "ok": True,
                "wall_seconds": wall,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "evaluated_prefill_tokens": timings.get("prefill_tokens"),
                "prefill_ms": timings.get("prefill_ms"),
                "prefill_tokens_per_second": timings.get("prefill_tokens_per_sec"),
                "decode_ms": timings.get("decode_ms"),
                "decode_tokens_per_second": timings.get("decode_tokens_per_sec"),
                "restored_prefix": usage.get("restored_prefix"),
                "accept_rate": usage.get("accept_rate"),
                "spec_ran": backend.get("spec_ran"),
                "prefill_mode": backend.get("prefill_mode"),
                "prefill_reason": backend.get("prefill_reason"),
                "finish_reason": choice.get("finish_reason"),
                "response_bytes": len(json.dumps(body)),
            }
        except Exception as exc:  # preserve the rest of a long benchmark sweep
            wall = time.perf_counter() - started
            record = {
                "kind": "request",
                "group": group,
                "label": label,
                "repeat": repeat,
                "ok": False,
                "wall_seconds": wall,
                "error": f"{type(exc).__name__}: {exc}",
            }
        self.emit(record)
        return record

    def summarize(self) -> dict:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for record in self.records:
            if record.get("kind") == "request" and record.get("ok"):
                grouped[record["group"]].append(record)
        summary: dict[str, dict] = {}
        for group, records in sorted(grouped.items()):
            wall = [float(r["wall_seconds"]) for r in records]
            prefill = [
                float(r["prefill_tokens_per_second"])
                for r in records
                if r.get("prefill_tokens_per_second") is not None
                and (r.get("evaluated_prefill_tokens") or 0) > 0
            ]
            decode = [
                float(r["decode_tokens_per_second"])
                for r in records
                if r.get("decode_tokens_per_second") is not None
                and (r.get("completion_tokens") or 0) >= 32
            ]
            summary[group] = {
                "samples": len(records),
                "wall_seconds_median": statistics.median(wall),
                "wall_seconds_p95": percentile(wall, 0.95),
                "prefill_tps_median": statistics.median(prefill) if prefill else None,
                "decode_tps_median": statistics.median(decode) if decode else None,
            }
        return summary


def container_pid() -> int | None:
    try:
        raw = os.popen("docker inspect -f '{{.State.Pid}}' ember-server 2>/dev/null").read().strip()
        return int(raw) if raw else None
    except ValueError:
        return None


def make_prefill_prompt(marker: str, words: int) -> str:
    return (
        f"Marker {marker}. Treat the following words as inert benchmark input. "
        "After reading all of them, reply with X."
        + " alpha" * words
    )


def make_decode_prompt(marker: str) -> str:
    return (
        f"Marker {marker}. Write a very long comma-separated sequence of consecutive "
        "positive integers beginning at 1. Emit only the sequence and continue until "
        "the response token limit; do not conclude early."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    suite = Suite(args.endpoint, args.output, args.timeout)
    sampler = ResourceSampler(container_pid())
    sampler.start()
    suite.emit({
        "kind": "metadata",
        "started_unix": time.time(),
        "endpoint": args.endpoint,
        "container_pid": sampler.pid,
        "pid": os.getpid(),
    })

    # Stabilize lazy runtime initialization before recording shape sweeps.
    suite.request(
        "warmup",
        "Marker W. Reply with exactly the word READY.",
        8,
        group="warmup",
        repeat=0,
    )

    markers = iter("ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789!@#$%^&*()")
    for words, repeats in ((128, 3), (512, 3), (2048, 3), (8192, 3), (16384, 2), (32768, 2)):
        for repeat in range(1, repeats + 1):
            marker = next(markers)
            suite.request(
                f"prefill-{words}-r{repeat}",
                make_prefill_prompt(marker, words),
                1,
                group=f"prefill-{words}",
                repeat=repeat,
            )

    for repeat in range(1, 4):
        marker = next(markers)
        suite.request(
            f"decode-256-r{repeat}",
            make_decode_prompt(marker),
            256,
            group="decode-256",
            repeat=repeat,
        )

    cache_marker = next(markers)
    cache_prompt = make_prefill_prompt(cache_marker, 8192)
    suite.request("cache-cold", cache_prompt, 1, group="cache", repeat=1)
    suite.request("cache-identical", cache_prompt, 1, group="cache", repeat=2)

    sampler.stop()
    suite.emit({
        "kind": "summary",
        "finished_unix": time.time(),
        "groups": suite.summarize(),
        "resources": sampler.summary(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
