#!/usr/bin/env python3
"""Expand and adjudicate the Qwen 128 GiB quant/intervention bakeoff.

Planning is deterministic and offline. Decision mode accepts only measured,
complete rows and uses the sweep corpus for selection. The final-heldout row
can confirm the already selected winner but can never change it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = ROOT / "share" / "quant_eval" / "qwen3.8-flash-next-bakeoff.json"


class BakeoffError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BakeoffError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise BakeoffError(f"{label} must be a JSON object")
    return value


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise BakeoffError(f"refusing to overwrite output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def artifact_by_name(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise BakeoffError("corpus manifest has no artifact inventory")
    matches = [row for row in rows if isinstance(row, dict) and row.get("filename") == name]
    if len(matches) != 1:
        raise BakeoffError(f"corpus manifest must name {name} exactly once")
    return matches[0]


def layers(policy: str) -> list[int]:
    if policy == "all-48":
        return list(range(48))
    if policy == "upper-24":
        return list(range(24, 48))
    if policy == "upper-12":
        return list(range(36, 48))
    if policy == "non-qsa":
        return [layer for layer in range(48) if layer % 4 != 3]
    raise BakeoffError(f"unknown layer policy: {policy}")


def make_plan(recipe_path: Path, corpus_dir: Path) -> dict[str, Any]:
    recipe = read_object(recipe_path, "bakeoff recipe")
    if recipe.get("schema_version") != 1:
        raise BakeoffError("unsupported bakeoff recipe")
    corpus_manifest_path = corpus_dir / "qwen-corpora-manifest.json"
    corpus_manifest = read_object(corpus_manifest_path, "corpus manifest")
    if corpus_manifest.get("partition", {}).get("pairwise_request_overlap_count") != 0:
        raise BakeoffError("bakeoff corpora are not pairwise disjoint")
    corpus_evidence: dict[str, Any] = {}
    for name in ("extraction-good.jsonl", "extraction-bad.jsonl",
                 "sweep-validation.jsonl", "final-heldout.jsonl"):
        row = artifact_by_name(corpus_manifest, name)
        path = corpus_dir / name
        if not path.is_file() or sha256(path) != row.get("sha256"):
            raise BakeoffError(f"corpus artifact differs from its manifest: {name}")
        corpus_evidence[name] = {
            "path": str(path.resolve()), "sha256": row["sha256"],
            "record_count": row["record_count"],
        }
    if corpus_evidence["sweep-validation.jsonl"]["sha256"] == corpus_evidence["final-heldout.jsonl"]["sha256"]:
        raise BakeoffError("sweep and final corpora must differ")

    configurations = []
    for scale in recipe["intervention_sweep"]["lambdas"]:
        for policy in recipe["intervention_sweep"]["layer_policies"]:
            identifier = f"lambda-{scale:.2f}-{policy}"
            selected = set(layers(policy))
            configurations.append({
                "id": identifier,
                "scale": scale,
                "layer_policy": policy,
                "layer_scales": {str(layer): scale if layer in selected else 0.0
                                 for layer in range(48)},
                "intervention_command": [
                    "python3", "scripts/qwen_intervention.py",
                    "--activation-backend", "dump",
                    "--good-corpus", corpus_evidence["extraction-good.jsonl"]["path"],
                    "--bad-corpus", corpus_evidence["extraction-bad.jsonl"]["path"],
                    "--held-out-corpus", corpus_evidence["sweep-validation.jsonl"]["path"],
                    "--good-activations", "<GOOD_ACTIVATIONS_48x2560_F32>",
                    "--bad-activations", "<BAD_ACTIVATIONS_48x2560_F32>",
                    "--activation-artifact-sha256", "<STOCK_ROCMI4_SHA256>",
                    "--layer-scales", f"<PLAN_DIR>/layer-scales/{identifier}.json",
                    "--output", f"<PLAN_DIR>/interventions/{identifier}.json",
                ],
            })
    return {
        "schema_version": 1,
        "status": "planned_unmeasured",
        "recipe": {"path": str(recipe_path.resolve()), "sha256": sha256(recipe_path),
                   "value": recipe},
        "corpus_manifest": {"path": str(corpus_manifest_path.resolve()),
                            "sha256": sha256(corpus_manifest_path)},
        "corpora": corpus_evidence,
        "stock_control": recipe["stock_control"],
        "sweep_configurations": configurations,
        "format_arms": recipe["format_arms"],
        "required_result_stages": ["stock", "sweep", "format", "final"],
        "publication_allowed": False,
    }


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise BakeoffError(f"{label} must be a finite measured number")
    return float(value)


def assess(row: dict[str, Any], gates: dict[str, Any], corpus_sha: str) -> dict[str, Any]:
    if row.get("measurement_kind") != "measured" or row.get("status") != "complete":
        raise BakeoffError("every bakeoff row must be a complete measured result")
    if row.get("corpus_sha256") != corpus_sha:
        raise BakeoffError("result used the wrong corpus partition")
    if row.get("differential_correctness_pass") is not True or row.get("audited_quality_pass") is not True:
        return {"passes": False}
    artifact_bytes = row.get("artifact_bytes")
    if not isinstance(artifact_bytes, int) or isinstance(artifact_bytes, bool) or artifact_bytes < 1:
        raise BakeoffError("artifact_bytes must be a measured positive integer")
    prefill = row.get("prefill_tps_samples")
    decode = row.get("decode_tps_samples")
    count = gates["minimum_samples_per_performance_gate"]
    if not isinstance(prefill, list) or not isinstance(decode, list) or len(prefill) < count or len(decode) < count:
        raise BakeoffError("performance rows require at least three measured samples per gate")
    prefill_median = statistics.median(finite_number(v, "prefill sample") for v in prefill)
    decode_median = statistics.median(finite_number(v, "decode sample") for v in decode)
    quality = finite_number(row.get("quality_score"), "quality_score")
    companion = row.get("companion_artifact_bytes")
    required_companions = gates["required_companion_inventory_keys"]
    if not isinstance(companion, dict) or set(companion) != set(required_companions):
        raise BakeoffError("measurement must inventory MTP and vision-mmproj companion bytes")
    enabled = row.get("enabled_companions")
    if (not isinstance(enabled, list) or len(enabled) != len(set(enabled))
            or any(name not in required_companions for name in enabled)):
        raise BakeoffError("enabled_companions must be a unique subset of the pinned inventory")
    for name in required_companions:
        value = companion[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise BakeoffError("companion artifact sizes must be measured non-negative integers")
    enabled_companion_bytes = sum(companion[name] for name in enabled)
    host_memtotal = row.get("runner_memtotal_bytes")
    if host_memtotal != gates["certification_host_memtotal_bytes"]:
        raise BakeoffError("measurement did not use the pinned OtherU MemTotal")
    host_gtt_pages = row.get("runner_gtt_pages_limit")
    if host_gtt_pages != gates["certification_host_gtt_pages_limit"]:
        raise BakeoffError("measurement did not use the pinned OtherU TTM pages_limit")
    host_gtt_cap = gates["certification_host_gtt_cap_bytes"]
    if host_gtt_cap != host_gtt_pages * 4096:
        raise BakeoffError("pinned OtherU GTT byte cap is inconsistent")
    if row.get("peak_memory_measurement_method") != gates["peak_memory_measurement_method"]:
        raise BakeoffError("measurement lacks the pinned RSS/GTT sampling method")
    peak_rss = row.get("measured_peak_rss_bytes")
    peak_gtt = row.get("measured_peak_gtt_bytes")
    peak_uma = row.get("measured_peak_uma_bytes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in (peak_rss, peak_gtt, peak_uma)):
        raise BakeoffError("measured peak RSS, GTT, and accounted UMA bytes are required")
    if peak_uma < max(peak_rss, peak_gtt):
        raise BakeoffError("accounted UMA peak cannot be smaller than RSS or GTT peak")
    static_total = artifact_bytes + gates["runtime_reserve_bytes"] + enabled_companion_bytes
    fits = (static_total <= gates["device_budget_bytes"]
            and static_total <= host_memtotal and peak_uma <= host_memtotal
            and peak_gtt <= host_gtt_cap)
    passes = (fits and prefill_median >= gates["minimum_prefill_median_tps"]
              and decode_median >= gates["minimum_decode_median_tps"])
    result = {"passes": passes, "quality_score": quality,
              "prefill_median_tps": prefill_median,
              "decode_median_tps": decode_median,
              "artifact_bytes": artifact_bytes,
              "enabled_companion_bytes": enabled_companion_bytes,
              "static_accounted_bytes": static_total,
              "runner_memtotal_bytes": host_memtotal,
              "runner_gtt_pages_limit": host_gtt_pages,
              "runner_gtt_cap_bytes": host_gtt_cap,
              "measured_peak_rss_bytes": peak_rss,
              "measured_peak_gtt_bytes": peak_gtt,
              "measured_peak_uma_bytes": peak_uma}
    return result


def decide(plan: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    recipe = plan["recipe"]["value"]
    gates = recipe["hard_gates"]
    sweep_sha = plan["corpora"]["sweep-validation.jsonl"]["sha256"]
    final_sha = plan["corpora"]["final-heldout.jsonl"]["sha256"]
    rows = results.get("results")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise BakeoffError("results must be an array of objects")
    by_stage: dict[str, list[dict[str, Any]]] = {
        stage: [row for row in rows if row.get("stage") == stage]
        for stage in ("stock", "sweep", "format", "final")
    }
    if len(by_stage["stock"]) != 1 or by_stage["stock"][0].get("id") != recipe["stock_control"]["id"]:
        raise BakeoffError("exactly one pinned stock control measurement is required")
    stock_assessment = assess(by_stage["stock"][0], gates, sweep_sha)
    if by_stage["stock"][0].get("final_release_eligible") is not False:
        raise BakeoffError("stock control must remain final-ineligible")
    if not stock_assessment["passes"]:
        raise BakeoffError("stock control failed a measured correctness/quality/performance/memory gate")

    expected_sweep = {item["id"] for item in plan["sweep_configurations"]}
    if {row.get("id") for row in by_stage["sweep"]} != expected_sweep or len(by_stage["sweep"]) != len(expected_sweep):
        raise BakeoffError("sweep results must cover every lambda/layer configuration exactly once")
    assessed_sweep = [(row, assess(row, gates, sweep_sha)) for row in by_stage["sweep"]]
    passing_sweep = [(row, value) for row, value in assessed_sweep if value["passes"]]
    if not passing_sweep:
        raise BakeoffError("no measured intervention sweep configuration passed all gates")
    sweep_winner, sweep_metrics = sorted(
        passing_sweep,
        key=lambda pair: (-pair[1]["quality_score"], -pair[1]["decode_median_tps"],
                          -pair[1]["prefill_median_tps"], pair[0]["id"]),
    )[0]

    arms = {item["id"]: item for item in recipe["format_arms"]}
    if {row.get("arm_id") for row in by_stage["format"]} != set(arms) or len(by_stage["format"]) != len(arms):
        raise BakeoffError("format results must cover ROCMI4, ROCmFP4 FAST, and W4A4 exactly once")
    assessed_formats = []
    for row in by_stage["format"]:
        if row.get("configuration_id") != sweep_winner["id"]:
            raise BakeoffError("format crosscheck must use the selected sweep configuration")
        arm = arms[row["arm_id"]]
        if row.get("final_release_eligible") is not arm["final_release_eligible"]:
            raise BakeoffError("result eligibility differs from the pinned arm")
        assessed_formats.append((row, arm, assess(row, gates, sweep_sha)))
    eligible_formats = [(row, value) for row, arm, value in assessed_formats
                        if arm["final_release_eligible"] and value["passes"]]
    if not eligible_formats:
        raise BakeoffError("no final-eligible format passed all measured gates")
    format_winner, format_metrics = sorted(
        eligible_formats,
        key=lambda pair: (-pair[1]["quality_score"], -pair[1]["decode_median_tps"],
                          -pair[1]["prefill_median_tps"], pair[0]["arm_id"]),
    )[0]

    if len(by_stage["final"]) != 1:
        raise BakeoffError("final-heldout must contain exactly one preselected winner result")
    final_row = by_stage["final"][0]
    if (final_row.get("configuration_id") != sweep_winner["id"]
            or final_row.get("arm_id") != format_winner["arm_id"]):
        raise BakeoffError("final-heldout result attempted to change the selected recipe")
    final_metrics = assess(final_row, gates, final_sha)
    if not final_metrics["passes"]:
        raise BakeoffError("preselected winner failed final-heldout confirmation")
    return {
        "schema_version": 1, "status": "confirmed_from_measured_results",
        "stock_control": {"id": by_stage["stock"][0]["id"], "metrics": stock_assessment,
                          "final_release_eligible": False},
        "selected_configuration_id": sweep_winner["id"],
        "selected_arm_id": format_winner["arm_id"],
        "sweep_metrics": sweep_metrics, "format_metrics": format_metrics,
        "final_metrics": final_metrics,
        "final_heldout_used_for_selection": False,
        "publication_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.corpus_dir is not None and args.plan is None and args.results is None:
            output = make_plan(args.recipe.resolve(), args.corpus_dir.resolve())
        elif args.corpus_dir is None and args.plan is not None and args.results is not None:
            output = decide(read_object(args.plan, "bakeoff plan"),
                            read_object(args.results, "bakeoff results"))
        else:
            raise BakeoffError("use --corpus-dir to plan, or --plan plus --results to decide")
        write_new(args.output.absolute(), output)
    except (BakeoffError, OSError) as exc:
        print(f"qwen_bakeoff.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.absolute()), "status": output["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
