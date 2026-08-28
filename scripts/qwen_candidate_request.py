#!/usr/bin/env python3
"""Derive one intervention construction request from pinned runner evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import qwen_bakeoff as bakeoff
import qwen_quality_descriptor as evidence
import qwen_quantize as quant


INTENT_SCHEMA = "ember.qwen3.8.candidate-construction-intent.v1"
REQUEST_SCHEMA = "ember.qwen3.8.candidate-construction-request.v1"
CAPTURE_SCHEMA = "ember.qwen3.8.stock-control-activation-capture.v1"
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
HEX40 = re.compile(r"[0-9a-f]{40}")
REQUEST_ROOT = Path(
    "/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/"
    "evidence/operation-requests")

# Capture run 33122633860 used the checked-in recipe at 35eca745.  The only
# later recipe change before the current selection plan added ``vision_vocab``
# to the release companion inventory (fe2ad51); it did not change the input
# read by qwen_capture_control.py:generate_manifests.  Bind that predecessor's
# complete file digest to the exact capture-relevant projection so unrelated
# release-policy enrichment cannot invalidate already-measured activations,
# while any intervention-grid drift still fails closed.
CAPTURE_RECIPE_PROJECTIONS = {
    "4f9ec83bdd22bf0213179c657b3408edbcf9075958d4e9037ac5c0b6f0f5c634":
        "cb2e37e0460d0b5bcb62b61458d52372209b4bb429c66959b50799370b7c30de",
}


class CandidateRequestError(ValueError):
    pass


def fail(message: str) -> None:
    raise CandidateRequestError(message)


def pinned(value: Any, label: str) -> tuple[dict[str, str], dict[str, Any], Path]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        fail(f"{label} descriptor differs")
    path = Path(str(value.get("path", "")))
    parsed, exact = evidence.exact_json(path, value.get("sha256"), label)
    return {"path": str(exact), "sha256": value["sha256"]}, parsed, exact


def capture_recipe_compatible(capture_sha256: Any,
                              plan_recipe: Any) -> bool:
    if not isinstance(plan_recipe, dict):
        return False
    if capture_sha256 == plan_recipe.get("sha256"):
        return True
    expected_projection = CAPTURE_RECIPE_PROJECTIONS.get(capture_sha256)
    value = plan_recipe.get("value")
    intervention = value.get("intervention_sweep") if isinstance(value, dict) else None
    return (expected_projection is not None and isinstance(intervention, dict)
            and bakeoff.canonical_sha256(intervention) == expected_projection)


def selected_configuration(
    intent: dict[str, Any], plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    stage = intent.get("stage")
    row_id = intent.get("row_id")
    configurations = {row.get("id"): row for row in plan["sweep_configurations"]}
    if stage == "sweep":
        if intent.get("prior_ledger") is not None or row_id not in configurations:
            fail("sweep intent is not one canonical configuration without a prior ledger")
        configuration = configurations[row_id]
        return configuration, configuration, "rocmi4-control", "Q4_0_ROCMI4"
    if stage != "format":
        fail("candidate construction intent stage must be sweep or format")
    prior_descriptor = intent.get("prior_ledger")
    if not isinstance(prior_descriptor, dict):
        fail("format intent requires the attested sweep ledger")
    try:
        prior, _digest = bakeoff.read_prior_ledger(
            prior_descriptor, "sweep", plan)
    except (KeyError, OSError, TypeError, ValueError, bakeoff.BakeoffError) as exc:
        fail(f"format prior ledger does not reproduce: {exc}")
    configuration_id = prior.get("selected_configuration_id")
    if (configuration_id not in configurations
            or prior.get("selected_configuration") != configurations[configuration_id]):
        fail("format prior ledger does not select one canonical intervention")
    arms = {row.get("id"): row for row in plan["format_arms"]}
    if row_id not in arms:
        fail("format intent row is not one canonical format arm")
    arm = arms[row_id]
    return configurations[configuration_id], arm, arm["quantization_arm"], arm[
        "mtp_matrix_quant_contract"]


def derive(intent_path: Path, intent_sha256: str,
           ember_revision: str) -> tuple[dict[str, Any], Path]:
    intent, _ = evidence.exact_json(
        intent_path, intent_sha256, "candidate construction intent")
    expected = {
        "schema", "ember_revision", "stage", "row_id", "candidate_id",
        "capture_manifest", "cache_manifest", "companion_rocmi4",
        "companion_fast", "selection_plan", "prior_ledger",
        "construction_request_output", "publishes", "deletes",
    }
    if (set(intent) != expected or intent.get("schema") != INTENT_SCHEMA
            or intent.get("publishes") is not False
            or intent.get("deletes") is not False):
        fail("candidate construction intent schema/lifecycle differs")
    if (HEX40.fullmatch(ember_revision) is None
            or intent.get("ember_revision") != ember_revision):
        fail("candidate construction intent Ember revision differs")
    for name in ("row_id", "candidate_id"):
        if SAFE_ID.fullmatch(str(intent.get(name, ""))) is None:
            fail(f"candidate construction intent {name} is malformed")

    plan_desc, plan_raw, _plan_path = pinned(
        intent.get("selection_plan"), "selection plan")
    try:
        plan = bakeoff.verify_plan(plan_raw)
    except (KeyError, OSError, TypeError, ValueError, bakeoff.BakeoffError) as exc:
        fail(f"selection plan does not reproduce: {exc}")
    configuration, _row, arm, mtp_contract = selected_configuration(intent, plan)
    capture_desc, capture, capture_path = pinned(
        intent.get("capture_manifest"), "stock activation capture")
    if (capture.get("schema") != CAPTURE_SCHEMA
            or capture.get("status") != "complete"
            or capture.get("publishes") is not False
            or not capture_recipe_compatible(
                capture.get("recipe_sha256"), plan.get("recipe"))):
        fail("stock activation capture lifecycle/recipe differs from the plan")
    rows = capture.get("interventions")
    matches = [row for row in rows if isinstance(row, dict)
               and row.get("id") == configuration["id"]] if isinstance(rows, list) else []
    if len(matches) != 1:
        fail("capture does not contain exactly one selected intervention")
    selected = matches[0]
    filename = f"interventions/{configuration['id']}.json"
    if (selected.get("filename") != filename
            or selected.get("lambda") != configuration.get("scale")
            or selected.get("layer_policy") != configuration.get("layer_policy")):
        fail("captured intervention identity differs from the canonical configuration")
    intervention_path = capture_path.parent / filename
    exact_intervention = evidence.exact_file(
        str(intervention_path), selected.get("sha256"), "selected intervention manifest")
    profile_path = Path(plan["release_profile"]["path"]).resolve()
    profile_value, _inventory, _inventory_path = quant.validate_profile(
        profile_path)
    if (quant.sha256_file(profile_path) != plan["release_profile"]["sha256"]):
        fail("selection plan release profile digest differs")
    manifest, validated = quant.validate_intervention_manifest(
        exact_intervention, profile_value)
    if (validated.get("target_count") != selected.get("target_count")):
        fail("captured intervention target count differs after validation")
    # The quantizer repeats the complete plan-to-layer-scale authorization for
    # ROCMI4 sweep rows; retain the parsed object here only to ensure validation
    # cannot be optimized away as a digest-only check.
    if manifest.get("kind") != "directional_ablation":
        fail("selected intervention is not directional ablation")
    if arm == "rocmi4-control":
        quant_arm = quant.validated_quantization_arms(profile_value).get(arm)
        if quant_arm is None:
            fail("selection profile omits the ROCMI4 control arm")
        authorization = quant.validate_rocmi4_sweep_authorization(
            _plan_path, plan_desc["sha256"], profile_path,
            plan["release_profile"]["sha256"], quant_arm, manifest)
        if authorization.get("configuration_id") != configuration["id"]:
            fail("ROCMI4 sweep authorization selected a different configuration")

    cache_desc, _cache, _ = pinned(intent.get("cache_manifest"), "BF16 cache manifest")
    rocmi4_desc, _rocmi4, _ = pinned(
        intent.get("companion_rocmi4"), "ROCMI4 companion inventory")
    fast_desc, _fast, _ = pinned(
        intent.get("companion_fast"), "ROCmFP4 FAST companion inventory")
    output = Path(str(intent.get("construction_request_output", "")))
    if (not output.is_absolute() or output.parent != REQUEST_ROOT
            or re.fullmatch(r"construction-[A-Za-z0-9._-]{1,80}\.json", output.name) is None):
        fail("construction request output is not one safe child of the fixed request root")
    request = {
        "schema": REQUEST_SCHEMA,
        "mode": "build-candidate",
        "parameters": {
            "capture_manifest": capture_desc,
            "cache_manifest": cache_desc,
            "companion_rocmi4": rocmi4_desc,
            "companion_fast": fast_desc,
            "selection_plan": plan_desc,
            "candidate_kind": "intervention",
            "candidate_stage": intent["stage"],
            "row_id": intent["row_id"],
            "candidate_id": intent["candidate_id"],
            "intervention_configuration_id": configuration["id"],
            "intervention_manifest": {
                "path": str(exact_intervention), "sha256": selected["sha256"]},
            "quantization_arm": arm,
            "mtp_matrix_quant_contract": mtp_contract,
        },
        "publishes": False,
        "deletes": False,
    }
    evidence.write_new(output, request)
    return request, output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--intent", type=Path, required=True)
    result.add_argument("--intent-sha256", required=True)
    result.add_argument("--ember-revision", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        request, output = derive(
            args.intent.absolute(), args.intent_sha256, args.ember_revision)
    except (CandidateRequestError, evidence.DescriptorError, quant.PipelineError,
            OSError, ValueError) as exc:
        print(f"qwen-candidate-request: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "complete", "mode": request["mode"],
        "operation_request": {"path": str(output),
                              "sha256": evidence.sha256_file(output)},
        "publishes": False, "deletes": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
