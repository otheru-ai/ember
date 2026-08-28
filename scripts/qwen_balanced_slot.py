#!/usr/bin/env python3
"""Measure one exact prefill/decode slot against one fresh Qwen server."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts" / "bench" / "benchmark.py"


def load_benchmark():
    spec = importlib.util.spec_from_file_location("ember_balanced_benchmark", BENCHMARK)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load benchmark implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_metric(row: dict, key: str, label: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is absent")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} is not a positive finite value")
    return result


def make_run(index: int, arm_id: str, process_identity: dict,
             workload: dict, calibrated_words: int,
             prefill_prompt: str, decode_prompt: str,
             prefill: dict, decode: dict) -> dict:
    if (prefill.get("ok") is not True or prefill.get("group") != "prefill-2048"
            or prefill.get("evaluated_prefill_tokens") != 2074
            or prefill.get("restored_prefix") not in (0, False)
            or prefill.get("prefill_tps_rounding_consistent") is not True):
        raise ValueError(
            "balanced slot prefill was not exactly 2074 uncached evaluated tokens")
    if (decode.get("ok") is not True or decode.get("group") != "decode-256"
            or decode.get("completion_tokens") != 256
            or decode.get("decode_tps_rounding_consistent") is not True):
        raise ValueError("balanced slot decode was not exactly 256 generated tokens")
    accept_rate = exact_metric(decode, "accept_rate", "balanced slot MTP accept rate")
    if decode.get("spec_ran") is not True or not 0.0 < accept_rate < 1.0:
        raise ValueError(
            "balanced slot decode must run native MTP with 0 < accept_rate < 1")
    if (not isinstance(calibrated_words, int) or isinstance(calibrated_words, bool)
            or calibrated_words < 1):
        raise ValueError("balanced slot calibration word count is invalid")
    canonical_process = json.dumps(
        process_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    digest = hashlib.sha256(canonical_process).hexdigest()
    return {
        "run_index": index,
        "arm_id": arm_id,
        "process_instance": process_identity,
        "process_instance_sha256": digest,
        "workload_id": workload["workload_id"],
        "workload_recipe_sha256": workload["recipe_sha256"],
        "prefill_prompt_sha256": hashlib.sha256(prefill_prompt.encode()).hexdigest(),
        "decode_prompt_sha256": hashlib.sha256(decode_prompt.encode()).hexdigest(),
        "calibrated_prefill_words": calibrated_words,
        "evaluated_prefill_tokens": 2074,
        "completion_tokens": 256,
        "prefill_tps": exact_metric(
            prefill, "prefill_tokens_per_second", "balanced slot prefill throughput"),
        "decode_tps": exact_metric(
            decode, "decode_tokens_per_second", "balanced slot decode throughput"),
        "spec_ran": True,
        "accept_rate": accept_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--health-endpoint", required=True)
    parser.add_argument("--health-timeout", type=float, default=1800)
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--process-identity", type=Path, required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--workload-marker", required=True)
    parser.add_argument("--workload-recipe-sha256", required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--timing-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.run_index < 6:
        parser.error("--run-index must be 0..5")
    if (not args.output.is_absolute() or args.output.exists() or args.output.is_symlink()
            or not args.timing_output.is_absolute()
            or args.timing_output.exists() or args.timing_output.is_symlink()
            or not args.process_identity.is_absolute()
            or args.process_identity.is_symlink() or not args.process_identity.is_file()):
        parser.error("outputs must be new absolute paths")
    try:
        process_identity = json.loads(args.process_identity.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(f"fresh process identity is invalid: {exc}")
    if not isinstance(process_identity, dict):
        parser.error("fresh process identity must be an object")
    workload = {"workload_id": args.workload_id, "marker": args.workload_marker,
                "prefill_generator": "benchmark.make_prefill_prompt.v1",
                "decode_generator": "benchmark.make_decode_prompt.v1",
                "evaluated_prefill_tokens": 2074, "completion_tokens": 256}
    canonical_workload = json.dumps(
        workload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    if (hashlib.sha256(canonical_workload).hexdigest() != args.workload_recipe_sha256
            or process_identity.get("run_index") != args.run_index
            or process_identity.get("arm_id") != args.arm_id):
        parser.error("sealed workload or process identity differs from slot arguments")
    workload["recipe_sha256"] = args.workload_recipe_sha256

    benchmark = load_benchmark()
    if args.server_pid <= 1:
        parser.error("--server-pid must identify the fresh server")
    if not benchmark.wait_for_health(
            args.health_endpoint, args.health_timeout,
            Path("/proc") / str(args.server_pid) / "status"):
        parser.error("fresh finalist server did not become healthy")
    suite = benchmark.Suite(args.endpoint, args.timing_output, 1800, args.model)
    suite.emit({"kind": "metadata", "protocol": "balanced-confirmation-slot-v1",
                "run_index": args.run_index, "arm_id": args.arm_id})
    warmup = suite.request(
        "warmup", "Marker W. Reply with exactly the word READY.", 8,
        group="warmup", repeat=0)
    if warmup.get("ok") is not True:
        parser.error("fresh finalist server warmup failed")
    words, calibration = benchmark.calibrate_prefill_words(
        suite, benchmark.HARD_GATE_PREFILL_TOKENS,
        marker=args.workload_marker)
    cache_isolation = suite.request(
        f"cache-isolation-{args.run_index}",
        f"Cache isolation {args.workload_id}. Reply with EVICT.", 1,
        group="cache-isolation", repeat=1)
    if cache_isolation.get("ok") is not True:
        parser.error("prefix-cache isolation request failed")
    prefill_prompt = benchmark.make_prefill_prompt(args.workload_marker, words)
    decode_prompt = benchmark.make_decode_prompt(args.workload_marker)
    prefill = suite.request(
        f"balanced-prefill-{args.run_index}",
        prefill_prompt, 1,
        group="prefill-2048", repeat=1)
    decode = suite.request(
        f"balanced-decode-{args.run_index}",
        decode_prompt, 256,
        group="decode-256", repeat=1)
    try:
        run = make_run(args.run_index, args.arm_id, process_identity, workload,
                       words, prefill_prompt, decode_prompt, prefill, decode)
    except ValueError as exc:
        parser.error(str(exc))
    suite.emit({"kind": "summary", "protocol": "balanced-confirmation-slot-v1",
                "selected_words": words, "calibration_attempts": calibration,
                "run": run})
    args.output.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
