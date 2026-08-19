#!/usr/bin/env python3
"""Deterministic HTTP A/B harness for resident CPU/GPU/NPU decode.

The server is launched separately so the same client can measure a target-only
baseline and an XDNA-enabled candidate.  Each round releases simultaneous
requests, records both aggregate wall throughput and Ember's backend timings,
and hashes visible output for cross-run correctness comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def response_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    choice = choices[0]
    if isinstance(choice.get("text"), str):
        return choice["text"]
    message = choice.get("message") or {}
    if isinstance(message.get("content"), str):
        return message["content"]
    raise ValueError("response has no text content")


def response_metrics(payload: dict, elapsed_s: float,
                     prompt_hash: str) -> dict:
    usage = payload.get("usage") or {}
    timings = usage.get("timings") or {}
    backend = usage.get("backend") or {}
    text = response_text(payload)
    return {
        "prompt_sha256": prompt_hash,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "request_wall_ms": elapsed_s * 1000.0,
        "decode_ms": float(timings.get("decode_ms", 0.0)),
        "decode_tokens_per_second": float(
            timings.get("decode_tokens_per_sec", 0.0)),
        "accept_rate": float(usage.get("accept_rate", 0.0)),
        "spec_ran": bool(backend.get("spec_ran", False)),
        "prefill_mode": str(backend.get("prefill_mode", "unknown")),
    }


def request_json(url: str, body: dict, timeout: float) -> tuple[dict, float]:
    data = json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {payload}")
    return payload, time.monotonic() - started


def get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def run_round(url: str, body: dict, prompt_hash: str, concurrency: int,
              timeout: float) -> dict:
    barrier = threading.Barrier(concurrency + 1)
    rows: list[dict] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            barrier.wait()
            payload, elapsed = request_json(url, body, timeout)
            row = response_metrics(payload, elapsed, prompt_hash)
            with lock:
                rows.append(row)
        except BaseException as exc:  # retain worker failures for main thread
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    started = time.monotonic()
    barrier.wait()
    for thread in threads:
        thread.join(timeout + 5.0)
    wall_s = time.monotonic() - started
    if any(thread.is_alive() for thread in threads):
        raise TimeoutError("resident benchmark worker did not finish")
    if errors:
        raise RuntimeError(f"resident request failed: {errors[0]}")
    if len(rows) != concurrency:
        raise RuntimeError("resident benchmark returned the wrong row count")
    rows.sort(key=lambda row: row["output_sha256"])
    tokens = sum(row["completion_tokens"] for row in rows)
    return {
        "wall_ms": wall_s * 1000.0,
        "completion_tokens": tokens,
        "aggregate_tokens_per_second": tokens / max(wall_s, 1e-9),
        "rows": rows,
    }


def summarize(rounds: list[dict], status_before: dict,
              status_after: dict) -> dict:
    rows = [row for result in rounds for row in result["rows"]]
    total_tokens = sum(result["completion_tokens"] for result in rounds)
    total_wall_s = sum(result["wall_ms"] for result in rounds) / 1000.0
    latency = [row["request_wall_ms"] for row in rows]
    decode_tps = [row["decode_tokens_per_second"] for row in rows]
    accept = [row["accept_rate"] for row in rows if row["spec_ran"]]
    before_batch = status_before.get("continuous_batching") or {}
    after_batch = status_after.get("continuous_batching") or {}
    decode_batches_delta = int(after_batch.get("decode_batches", 0)) - int(
        before_batch.get("decode_batches", 0))
    decode_rows_delta = int(after_batch.get("decode_rows", 0)) - int(
        before_batch.get("decode_rows", 0))
    return {
        "rounds": len(rounds),
        "requests": len(rows),
        "total_completion_tokens": total_tokens,
        "aggregate_tokens_per_second":
            total_tokens / max(total_wall_s, 1e-9),
        "round_tokens_per_second_mean": statistics.fmean(
            result["aggregate_tokens_per_second"] for result in rounds),
        "request_latency_ms_mean": statistics.fmean(latency),
        "request_latency_ms_p50": percentile(latency, 0.50),
        "request_latency_ms_p95": percentile(latency, 0.95),
        "backend_decode_tokens_per_second_mean": statistics.fmean(decode_tps),
        "spec_rows": sum(1 for row in rows if row["spec_ran"]),
        "accept_rate_mean": statistics.fmean(accept) if accept else 0.0,
        "max_decode_batch_before": int(
            before_batch.get("max_decode_batch", 0)),
        "max_decode_batch_after": int(after_batch.get("max_decode_batch", 0)),
        "decode_batches_delta": decode_batches_delta,
        "decode_rows_delta": decode_rows_delta,
        "mean_decode_batch": (
            decode_rows_delta / decode_batches_delta
            if decode_batches_delta > 0 else 0.0),
    }


def output_sets(rounds: list[dict]) -> dict[str, list[str]]:
    outputs: dict[str, set[str]] = {}
    for result in rounds:
        for row in result["rows"]:
            outputs.setdefault(row["prompt_sha256"], set()).add(
                row["output_sha256"])
    return {key: sorted(values) for key, values in sorted(outputs.items())}


def compare_reference(outputs: dict[str, list[str]], reference: dict) -> None:
    expected = reference.get("outputs")
    if outputs != expected:
        raise RuntimeError(
            f"candidate output hashes differ from reference: "
            f"expected={expected} actual={outputs}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--require-spec", action="store_true")
    parser.add_argument("--allow-unbatched", action="store_true",
                        help="permit a run that never forms a decode batch")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency < 1 or args.rounds < 1 or args.warmup_rounds < 0:
        raise SystemExit("concurrency/rounds must be positive")
    prompt = args.prompt_file.read_text()
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    status_url = args.url.rstrip("/") + "/status"
    initial_status = get_json(status_url, args.timeout)
    batch = initial_status.get("continuous_batching") or {}
    if not batch.get("enabled") or int(batch.get("capacity", 0)) < args.concurrency:
        raise RuntimeError(
            "server continuous-batching capacity is below requested concurrency")

    for _ in range(args.warmup_rounds):
        run_round(endpoint, body, prompt_hash, args.concurrency, args.timeout)
    status_before = get_json(status_url, args.timeout)
    rounds = [
        run_round(endpoint, body, prompt_hash, args.concurrency, args.timeout)
        for _ in range(args.rounds)
    ]
    status_after = get_json(status_url, args.timeout)
    outputs = output_sets(rounds)
    if any(len(values) != 1 for values in outputs.values()):
        raise RuntimeError(f"greedy output changed within one run: {outputs}")
    summary = summarize(rounds, status_before, status_after)
    if (not args.allow_unbatched and args.concurrency > 1 and
            (summary["decode_batches_delta"] <= 0 or
             summary["max_decode_batch_after"] < args.concurrency)):
        raise RuntimeError(
            "measured requests never formed the requested resident decode "
            "batch; result does not test cross-session overlap")
    if args.require_spec and summary["spec_rows"] != args.rounds * args.concurrency:
        raise RuntimeError(
            f"speculation required but ran for {summary['spec_rows']}/"
            f"{args.rounds * args.concurrency} measured rows")

    report = {
        "schema_version": 1,
        "config": {
            "url": args.url,
            "model": args.model,
            "prompt_sha256": prompt_hash,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "warmup_rounds": args.warmup_rounds,
            "rounds": args.rounds,
            "require_spec": args.require_spec,
            "allow_unbatched": args.allow_unbatched,
        },
        "summary": summary,
        "outputs": outputs,
        "round_results": rounds,
    }
    if args.reference:
        compare_reference(outputs, json.loads(args.reference.read_text()))
        report["reference"] = str(args.reference)
        report["reference_exact"] = True
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
