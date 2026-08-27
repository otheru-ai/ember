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


HARD_GATE_PROTOCOL = "ember-2026.8.24-prefill2048-decode256-v1"
HARD_GATE_PREFILL_TPS = 412.0
HARD_GATE_DECODE_TPS = 39.49
HARD_GATE_PREFILL_TOKENS = 2074
HARD_GATE_SAMPLES = 3


class Suite:
    def __init__(self, endpoint: str, output: Path, timeout: float,
                 model: str) -> None:
        self.endpoint = endpoint
        self.output = output
        self.timeout = timeout
        self.model = model
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
            "model": self.model,
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


def evaluate_hard_gate(records: list[dict], *, prefill_target: float,
                       decode_target: float) -> dict:
    prefill_rows = [
        row for row in records
        if row.get("kind") == "request" and row.get("group") == "prefill-2048"
        and row.get("ok") and row.get("prefill_tokens_per_second") is not None
    ]
    decode_rows = [
        row for row in records
        if row.get("kind") == "request" and row.get("group") == "decode-256"
        and row.get("ok") and row.get("decode_tokens_per_second") is not None
    ]
    prefill_values = [float(row["prefill_tokens_per_second"])
                      for row in prefill_rows]
    decode_values = [float(row["decode_tokens_per_second"])
                     for row in decode_rows]
    prefill_median = (statistics.median(prefill_values)
                      if len(prefill_values) == HARD_GATE_SAMPLES else None)
    decode_median = (statistics.median(decode_values)
                     if len(decode_values) == HARD_GATE_SAMPLES else None)
    prefill_shape_match = (
        len(prefill_rows) == HARD_GATE_SAMPLES and
        all(row.get("evaluated_prefill_tokens") == HARD_GATE_PREFILL_TOKENS
            for row in prefill_rows)
    )
    decode_shape_match = (
        len(decode_rows) == HARD_GATE_SAMPLES and
        all(row.get("completion_tokens") == 256 for row in decode_rows)
    )
    passed = bool(
        prefill_shape_match and decode_shape_match and
        prefill_median is not None and prefill_median >= prefill_target and
        decode_median is not None and decode_median >= decode_target
    )
    return {
        "protocol": HARD_GATE_PROTOCOL,
        "passed": passed,
        "required_samples_per_group": HARD_GATE_SAMPLES,
        "prefill_2048": {
            "samples": len(prefill_values),
            "expected_evaluated_tokens": HARD_GATE_PREFILL_TOKENS,
            "evaluated_tokens": [row.get("evaluated_prefill_tokens")
                                 for row in prefill_rows],
            "shape_match": prefill_shape_match,
            "median_tps": prefill_median,
            "target_tps": prefill_target,
            "passed": bool(prefill_shape_match and prefill_median is not None
                           and prefill_median >= prefill_target),
        },
        "decode_256_counting": {
            "samples": len(decode_values),
            "completion_tokens": [row.get("completion_tokens")
                                  for row in decode_rows],
            "shape_match": decode_shape_match,
            "median_tps": decode_median,
            "target_tps": decode_target,
            "passed": bool(decode_shape_match and decode_median is not None
                           and decode_median >= decode_target),
        },
    }


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


def calibrate_prefill_words(suite: Suite, target_tokens: int,
                            max_attempts: int = 8) -> tuple[int, list[dict]]:
    """Find the alpha-word count that this server tokenizes to target_tokens.

    The published 2K shape happened to be 2,074 tokens with the DeepSeek
    template. Qwen has a different template/tokenizer, so its hard gate first
    calibrates the inert filler rather than failing on tokenizer overhead. Use
    usage.prompt_tokens: prefix-cache hits can reduce evaluated_prefill_tokens
    during the probes even though the complete prompt shape is unchanged.
    """
    words = 2048
    seen: set[int] = set()
    attempts: list[dict] = []
    low: tuple[int, int] | None = None
    high: tuple[int, int] | None = None
    for attempt in range(1, max_attempts + 1):
        if words in seen:
            break
        seen.add(words)
        record = suite.request(
            f"prefill-calibration-r{attempt}",
            make_prefill_prompt("0", words), 1,
            group="calibration", repeat=attempt)
        observed = record.get("prompt_tokens") if record.get("ok") else None
        attempts.append({
            "attempt": attempt,
            "words": words,
            "prompt_tokens": observed,
            "ok": bool(record.get("ok")),
        })
        if not isinstance(observed, int):
            break
        if observed == target_tokens:
            return words, attempts
        if observed < target_tokens:
            low = (words, observed)
        else:
            high = (words, observed)
        proposed = max(1, words + target_tokens - observed)
        if proposed in seen and low and high:
            proposed = (low[0] + high[0]) // 2
        words = proposed
    return words, attempts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--protocol", choices=("full", "hard-gate"),
                        default="full")
    parser.add_argument("--prefill-target", type=float,
                        default=HARD_GATE_PREFILL_TPS)
    parser.add_argument("--decode-target", type=float,
                        default=HARD_GATE_DECODE_TPS)
    parser.add_argument("--require-gate", action="store_true",
                        help="exit nonzero unless the matched hard gate passes")
    args = parser.parse_args()
    if args.require_gate and args.protocol != "hard-gate":
        parser.error("--require-gate requires --protocol hard-gate")
    if args.prefill_target <= 0 or args.decode_target <= 0:
        parser.error("gate targets must be positive")

    suite = Suite(args.endpoint, args.output, args.timeout, args.model)
    sampler = ResourceSampler(container_pid())
    sampler.start()
    suite.emit({
        "kind": "metadata",
        "started_unix": time.time(),
        "endpoint": args.endpoint,
        "model": args.model,
        "protocol": (HARD_GATE_PROTOCOL if args.protocol == "hard-gate"
                     else "full"),
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
    calibration: list[dict] = []
    hard_gate_words = 2048
    if args.protocol == "hard-gate":
        hard_gate_words, calibration = calibrate_prefill_words(
            suite, HARD_GATE_PREFILL_TOKENS)
    shapes = ((hard_gate_words, 3),) if args.protocol == "hard-gate" else (
        (128, 3), (512, 3), (2048, 3), (8192, 3), (16384, 2), (32768, 2))
    for words, repeats in shapes:
        for repeat in range(1, repeats + 1):
            marker = next(markers)
            suite.request(
                f"prefill-{words}-r{repeat}",
                make_prefill_prompt(marker, words),
                1,
                group=("prefill-2048" if args.protocol == "hard-gate"
                       else f"prefill-{words}"),
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

    if args.protocol == "full":
        cache_marker = next(markers)
        cache_prompt = make_prefill_prompt(cache_marker, 8192)
        suite.request("cache-cold", cache_prompt, 1, group="cache", repeat=1)
        suite.request("cache-identical", cache_prompt, 1, group="cache", repeat=2)

    sampler.stop()
    groups = suite.summarize()
    gate = evaluate_hard_gate(
        suite.records, prefill_target=args.prefill_target,
        decode_target=args.decode_target) if args.protocol == "hard-gate" else None
    suite.emit({
        "kind": "summary",
        "finished_unix": time.time(),
        "groups": groups,
        "hard_gate": gate,
        "prefill_calibration": ({
            "target_prompt_tokens": HARD_GATE_PREFILL_TOKENS,
            "selected_words": hard_gate_words,
            "attempts": calibration,
        } if args.protocol == "hard-gate" else None),
        "resources": sampler.summary(),
    })
    return 1 if args.require_gate and gate and not gate["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
