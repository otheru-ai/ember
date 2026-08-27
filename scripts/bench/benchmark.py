#!/usr/bin/env python3
"""Reproducible HTTP performance sweep for a running Ember server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
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


RUNNER_GTT_CAP_BYTES = 124 * 1024**3
MEMORY_MEASUREMENT_METHOD = "runner_rss_gtt_sampler_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def model_inventory_from_build_record(
    record_path: Path,
    record_sha256: str,
    model: Path,
    model_sha256: str,
) -> tuple[dict, bytes]:
    """Bind a first-shard path to every ordered shard in its quant record.

    The record can contain container-absolute output paths. Only their safe
    basenames are rebound into the explicitly supplied host model directory.
    Content digests are verified later by the gate with direct I/O, after it
    owns the GPU and production has released UMA.
    """
    raw = record_path.read_bytes()
    if sha256_bytes(raw) != record_sha256:
        raise ValueError("quant build-record digest mismatch")
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid quant build record: {exc}") from exc
    if (not isinstance(record, dict) or record.get("status") != "complete" or
            record.get("mode") != "execute"):
        raise ValueError("quant build record is not a completed execute record")
    rows = (record.get("output") or {}).get("shards")
    preflight = record.get("memory_preflight") or {}
    if not isinstance(rows, list) or not rows:
        raise ValueError("quant build record has no ordered output shards")
    normalized = []
    names: set[str] = set()
    recorded_parent: Path | None = None
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"quant build-record shard {index} is not an object")
        recorded = Path(str(row.get("path", "")))
        name = recorded.name
        digest = row.get("sha256")
        size = row.get("size_bytes")
        actual = model.parent / name
        if (not recorded.is_absolute() or not name or name in names or
                not name.endswith(".gguf") or actual.is_symlink() or
                not actual.is_file() or not isinstance(size, int) or
                isinstance(size, bool) or size < 1 or actual.stat().st_size != size or
                not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None):
            raise ValueError(
                f"quant build-record shard {index} is missing or malformed")
        if recorded_parent is None:
            recorded_parent = recorded.parent
        elif recorded.parent != recorded_parent:
            raise ValueError(
                "quant build-record shards do not share one recorded directory")
        names.add(name)
        normalized.append({
            "index": index, "filename": name, "path": str(actual),
            "recorded_path": str(recorded), "size_bytes": size,
            "sha256": digest,
        })
    if len(normalized) > 1:
        pattern = re.compile(r"^.+-([0-9]{5})-of-([0-9]{5})\.gguf$")
        matches = [pattern.fullmatch(row["filename"]) for row in normalized]
        numbers = [int(match.group(1)) for match in matches if match]
        totals = {int(match.group(2)) for match in matches if match}
        if (any(match is None for match in matches) or
                numbers != list(range(1, len(normalized) + 1)) or
                totals != {len(normalized)}):
            raise ValueError(
                "quant build record is not a complete ordered GGUF shard set")
    if model.resolve() != Path(normalized[0]["path"]).resolve():
        raise ValueError("--model must name the first ordered quantized shard")
    if model_sha256 != normalized[0]["sha256"]:
        raise ValueError("--model-sha256 differs from build-record shard 1")
    sizes = [row["size_bytes"] for row in normalized]
    if (preflight.get("shard_count") != len(normalized) or
            preflight.get("shard_bytes") != sizes or
            preflight.get("artifact_bytes") != sum(sizes)):
        raise ValueError(
            "ordered output shards differ from quantizer size preflight")
    return ({
        "schema": "ember.qwen3.8.ordered-model-inventory.v1",
        "source": "required_quant_build_record",
        "build_record": {"path": str(record_path), "sha256": record_sha256},
        "first_shard": str(model),
        "shard_count": len(normalized),
        "aggregate_bytes": sum(sizes),
        "shards": normalized,
    }, raw)


def _meminfo_bytes(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        fields = raw.strip().split()
        if not fields:
            continue
        scale = 1024 if len(fields) > 1 and fields[1] == "kB" else 1
        values[key] = int(fields[0]) * scale
    return values


class ResourceSampler:
    """Sample the server's host PID and the APU's system-wide UMA/GTT use.

    Docker PIDs are resolved by the ownership wrapper and passed explicitly.
    Reading a process named ``ember-server`` would silently sample production
    instead of the short-lived certification container.
    """

    def __init__(
        self,
        pid: int | None,
        interval: float = 0.5,
        *,
        proc_root: Path = Path("/proc"),
        meminfo_path: Path = Path("/proc/meminfo"),
        pages_limit_path: Path = Path("/sys/module/ttm/parameters/pages_limit"),
        gtt_paths: list[Path] | None = None,
    ) -> None:
        self.pid = pid
        self.interval = interval
        self.status_path = proc_root / str(pid) / "status" if pid else None
        self.meminfo_path = meminfo_path
        self.pages_limit_path = pages_limit_path
        self.gtt_paths = (gtt_paths if gtt_paths is not None else sorted(
            Path("/sys/class/drm").glob("card*/device/mem_info_gtt_used")))
        self.stop_event = threading.Event()
        self.samples: list[dict[str, float | int]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.samples.append(self._sample())

    def _sample(self) -> dict[str, float | int]:
        sample: dict[str, float | int] = {"monotonic": time.monotonic()}
        try:
            meminfo = _meminfo_bytes(self.meminfo_path)
            sample["memtotal_bytes"] = meminfo["MemTotal"]
            sample["mem_available_bytes"] = meminfo["MemAvailable"]
            sample["uma_used_bytes"] = max(
                0, meminfo["MemTotal"] - meminfo["MemAvailable"])
        except (KeyError, OSError, ValueError):
            pass
        if self.status_path is not None:
            try:
                for line in self.status_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("VmRSS:"):
                        sample["rss_bytes"] = int(line.split()[1]) * 1024
                    elif line.startswith("VmHWM:"):
                        sample["hwm_bytes"] = int(line.split()[1]) * 1024
            except (IndexError, OSError, ValueError):
                pass
        gtt = []
        for path in self.gtt_paths:
            try:
                gtt.append(int(path.read_text(encoding="utf-8").strip()))
            except (OSError, ValueError):
                continue
        if gtt:
            sample["gtt_bytes"] = max(gtt)
        gpu_paths = sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))
        for path in gpu_paths:
            try:
                value = float(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            sample["gpu_busy_pct"] = max(value, sample.get("gpu_busy_pct", 0.0))
        return sample

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.samples.append(self._sample())
            self.stop_event.wait(self.interval)

    def summary(self) -> dict[str, float | int | None]:
        def values(key: str) -> list[float | int]:
            return [sample[key] for sample in self.samples if key in sample]

        rss = values("rss_bytes")
        hwm = values("hwm_bytes")
        available = values("mem_available_bytes")
        memtotal = values("memtotal_bytes")
        uma = values("uma_used_bytes")
        gtt = values("gtt_bytes")
        busy = values("gpu_busy_pct")
        try:
            pages_limit = int(self.pages_limit_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pages_limit = None
        return {
            "peak_memory_measurement_method": MEMORY_MEASUREMENT_METHOD,
            "server_host_pid": self.pid,
            "samples": len(self.samples),
            "runner_memtotal_bytes": int(memtotal[0]) if memtotal else None,
            "runner_gtt_pages_limit": pages_limit,
            "runner_gtt_cap_bytes": pages_limit * 4096 if pages_limit is not None else None,
            "measured_peak_rss_bytes": (
                int(max([*rss, *hwm])) if rss or hwm else None),
            "process_hwm_bytes": int(max(hwm)) if hwm else None,
            "measured_peak_gtt_bytes": int(max(gtt)) if gtt else None,
            "measured_peak_uma_bytes": int(max(uma)) if uma else None,
            "mem_available_bytes_min": int(min(available)) if available else None,
            # Preserve the historical presentation fields for bundle readers.
            "rss_gib_max": max(rss) / 1024**3 if rss else None,
            "hwm_gib_max": max(hwm) / 1024**3 if hwm else None,
            "mem_available_gib_min": min(available) / 1024**3 if available else None,
            "gpu_busy_pct_mean": statistics.fmean(busy) if busy else None,
            "gpu_busy_pct_max": max(busy) if busy else None,
        }


def evaluate_memory_gate(resources: dict, *, gtt_cap_bytes: int) -> dict:
    method = resources.get("peak_memory_measurement_method")
    samples = resources.get("samples")
    server_pid = resources.get("server_host_pid")
    memtotal = resources.get("runner_memtotal_bytes")
    pages_limit = resources.get("runner_gtt_pages_limit")
    live_gtt_cap = resources.get("runner_gtt_cap_bytes")
    peak_rss = resources.get("measured_peak_rss_bytes")
    peak_gtt = resources.get("measured_peak_gtt_bytes")
    peak_uma = resources.get("measured_peak_uma_bytes")
    values = (memtotal, pages_limit, live_gtt_cap, peak_rss, peak_gtt, peak_uma)
    complete = all(isinstance(value, int) and not isinstance(value, bool)
                   and value >= 0 for value in values)
    measurement_valid = bool(
        method == MEMORY_MEASUREMENT_METHOD and
        isinstance(samples, int) and not isinstance(samples, bool) and samples > 0 and
        isinstance(server_pid, int) and not isinstance(server_pid, bool) and
        server_pid > 1)
    live_cap_match = bool(
        complete and pages_limit > 0 and live_gtt_cap == pages_limit * 4096
        and live_gtt_cap == gtt_cap_bytes)
    coherent = bool(complete and peak_uma >= max(peak_rss, peak_gtt))
    rss_fits = bool(complete and peak_rss <= memtotal)
    gtt_fits = bool(complete and peak_gtt <= gtt_cap_bytes)
    uma_fits = bool(complete and peak_uma <= memtotal)
    return {
        "protocol": MEMORY_MEASUREMENT_METHOD,
        "passed": bool(complete and measurement_valid and live_cap_match and coherent and
                       rss_fits and gtt_fits and uma_fits),
        "complete": complete,
        "measurement_contract_valid": measurement_valid,
        "server_host_pid": server_pid,
        "samples": samples,
        "live_gtt_cap_matches_required": live_cap_match,
        "measurements_coherent": coherent,
        "runner_memtotal_bytes": memtotal,
        "runner_gtt_pages_limit": pages_limit,
        "runner_gtt_cap_bytes": live_gtt_cap,
        "required_gtt_cap_bytes": gtt_cap_bytes,
        "measured_peak_rss_bytes": peak_rss,
        "measured_peak_gtt_bytes": peak_gtt,
        "measured_peak_uma_bytes": peak_uma,
        "rss_fits_host_memtotal": rss_fits,
        "gtt_fits_required_cap": gtt_fits,
        "uma_fits_host_memtotal": uma_fits,
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


def container_pid(name: str = "ember-server") -> int | None:
    try:
        raw = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Pid}}", name],
            check=False, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return int(raw) if raw else None
    except (OSError, subprocess.SubprocessError, ValueError):
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


def wait_for_health(endpoint: str, timeout: float, status_path: Path) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not status_path.is_file():
            return False
        try:
            with urllib.request.urlopen(endpoint, timeout=2) as response:
                if 200 <= response.status < 300:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.5)
    return False


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
    parser.add_argument("--server-pid", type=int,
                        help="explicit host PID of the timed server container")
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--meminfo-path", type=Path, default=Path("/proc/meminfo"))
    parser.add_argument("--ttm-pages-limit-path", type=Path,
                        default=Path("/sys/module/ttm/parameters/pages_limit"))
    parser.add_argument("--gtt-path", action="append", type=Path, default=[])
    parser.add_argument("--resource-sample-interval", type=float, default=0.5)
    parser.add_argument("--require-memory-gate", action="store_true",
                        help="fail closed unless exact host RSS/GTT/UMA evidence fits")
    parser.add_argument("--gtt-cap-bytes", type=int, default=RUNNER_GTT_CAP_BYTES)
    parser.add_argument("--health-endpoint",
                        help="wait here while sampling the server's model-load phase")
    parser.add_argument("--health-timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.require_gate and args.protocol != "hard-gate":
        parser.error("--require-gate requires --protocol hard-gate")
    if args.prefill_target <= 0 or args.decode_target <= 0:
        parser.error("gate targets must be positive")
    if args.server_pid is not None and args.server_pid <= 1:
        parser.error("--server-pid must be a host PID greater than 1")
    if args.require_memory_gate and args.server_pid is None:
        parser.error("--require-memory-gate requires explicit --server-pid")
    if args.resource_sample_interval <= 0:
        parser.error("--resource-sample-interval must be positive")
    if args.gtt_cap_bytes <= 0 or args.gtt_cap_bytes % 4096:
        parser.error("--gtt-cap-bytes must be a positive multiple of 4096")
    if args.health_timeout <= 0:
        parser.error("--health-timeout must be positive")
    if args.health_endpoint and args.server_pid is None:
        parser.error("--health-endpoint requires explicit --server-pid")

    suite = Suite(args.endpoint, args.output, args.timeout, args.model)
    selected_pid = args.server_pid if args.server_pid is not None else container_pid()
    if args.require_memory_gate and not (args.proc_root / str(selected_pid) / "status").is_file():
        parser.error("explicit --server-pid is not visible in --proc-root")
    sampler = ResourceSampler(
        selected_pid, args.resource_sample_interval,
        proc_root=args.proc_root, meminfo_path=args.meminfo_path,
        pages_limit_path=args.ttm_pages_limit_path,
        gtt_paths=args.gtt_path or None,
    )
    sampler.start()
    if args.health_endpoint and not wait_for_health(
            args.health_endpoint, args.health_timeout,
            args.proc_root / str(selected_pid) / "status"):
        sampler.stop()
        parser.error("timed server exited or did not become healthy")
    suite.emit({
        "kind": "metadata",
        "started_unix": time.time(),
        "endpoint": args.endpoint,
        "model": args.model,
        "protocol": (HARD_GATE_PROTOCOL if args.protocol == "hard-gate"
                     else "full"),
        "container_pid": sampler.pid,
        "server_pid_source": "explicit" if args.server_pid is not None else "ember-server-fallback",
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
    resources = sampler.summary()
    memory_gate = evaluate_memory_gate(
        resources, gtt_cap_bytes=args.gtt_cap_bytes)
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
        "resources": resources,
        "memory_gate": memory_gate,
    })
    performance_failed = bool(args.require_gate and gate and not gate["passed"])
    memory_failed = bool(args.require_memory_gate and not memory_gate["passed"])
    return 1 if performance_failed or memory_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
