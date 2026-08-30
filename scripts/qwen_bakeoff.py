#!/usr/bin/env python3
"""Expand and adjudicate the Qwen 128 GiB quant/intervention bakeoff.

Planning is deterministic and offline. Decision mode accepts only measured,
complete rows and uses the sweep corpus for selection. The final-heldout row
can confirm the already selected winner but can never change it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = ROOT / "share" / "quant_eval" / "qwen3.8-flash-next-bakeoff.json"
DEFAULT_PROFILE = (ROOT / "share" / "release_profiles" /
                   "qwen3.8-flash-next-rocmi4-strix-halo.json")
RECIPE_SCHEMA_VERSION = 3
PLAN_SCHEMA_VERSION = 2
RESULT_SCHEMA = "ember.qwen3.8.sequential-bakeoff-result.v4"
ASSESSMENT_SCHEMA = "ember.qwen3.8.candidate-assessment.v2"
LEDGER_SCHEMA = "ember.qwen3.8.sequential-bakeoff-ledger.v3"
BALANCED_CONFIRMATION_SCHEMA = "ember.qwen3.8.balanced-confirmation.v2"
BALANCED_PROCESS_SCHEMA = "ember.qwen3.8.fresh-server-process.v2"
BALANCED_CONFIRMATION_SAMPLES = 3
# A sub-1 tok/s decode lead is under 2.6% at the 39.49 tok/s release target and
# does not justify selecting a different 128 GiB quant artifact. Persist this
# practical-effect threshold in the pre-measurement plan rather than deriving
# it from the six observed samples.
BALANCED_MIN_DECODE_EFFECT_TPS = 1.0
KERNEL_CAPABILITY_BY_QUANTIZATION_ARM = {
    "profile-default-rocmi4": "rocmi4_dense_and_routed",
    "rocmi4-control": "rocmi4_dense_and_routed",
    "rocmi4-q6k-embedding-head": "rocmi4_dense_and_routed",
    "rocmfp4-fast-routed-experts-q6k-embedding-head": "rocmi4_dense_only",
    "rocmfp4-fast-matrix-q6k-embedding-head": "no_eligible_rocmi4_mmq",
    "rocmfp4-fast-matrix-q3-ple-q6k-embedding-head": "no_eligible_rocmi4_mmq",
}
SUPPORTED_MTP_MATRIX_CONTRACTS = {
    "Q4_0_ROCMI4", "Q4_0_ROCMFP4_FAST",
}
WINNER_ORDER = [
    "decode_median_tps_desc",
    "prefill_median_tps_desc",
    "quality_score_desc",
    "id_asc",
]
QUALITY_SPEC = importlib.util.spec_from_file_location(
    "ember_qwen_quality_judge", ROOT / "scripts" / "qwen_quality_judge.py")
if QUALITY_SPEC is None or QUALITY_SPEC.loader is None:
    raise RuntimeError("cannot load qwen_quality_judge.py")
QUALITY_MODULE = importlib.util.module_from_spec(QUALITY_SPEC)
QUALITY_SPEC.loader.exec_module(QUALITY_MODULE)
QUALITY_EVALUATOR = QUALITY_MODULE.evaluate_contract
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "ember_qwen_benchmark", ROOT / "scripts" / "bench" / "benchmark.py")
if BENCHMARK_SPEC is None or BENCHMARK_SPEC.loader is None:
    raise RuntimeError("cannot load benchmark.py")
BENCHMARK_MODULE = importlib.util.module_from_spec(BENCHMARK_SPEC)
BENCHMARK_SPEC.loader.exec_module(BENCHMARK_MODULE)


class BakeoffError(ValueError):
    pass


def candidate_kernel_capability(quantization_arm: Any) -> str:
    capability = KERNEL_CAPABILITY_BY_QUANTIZATION_ARM.get(quantization_arm)
    if capability is None:
        raise BakeoffError(
            "quantization arm has no declared timing-kernel capability: "
            f"{quantization_arm!r}")
    return capability


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


def canonical_sha256(value: Any) -> str:
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    if policy == "band-10-42":
        return list(range(10, 43))
    if policy == "upper-24":
        return list(range(24, 48))
    if policy == "upper-12":
        return list(range(36, 48))
    if policy == "non-qsa-band-10-42":
        return [layer for layer in range(10, 43) if layer % 4 != 3]
    raise BakeoffError(f"unknown layer policy: {policy}")


def _manifest_artifacts(
    corpus_dir: Path, manifest_name: str, expected_names: tuple[str, ...], label: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Read one phase-scoped manifest and only the artifacts in that phase.

    The selection manifest deliberately cannot name final-heldout.  This is a
    capability boundary, not merely an instruction to ignore one member of a
    combined manifest: a sweep plan never opens a file containing the final
    partition's pathname or digest.
    """
    manifest_path = corpus_dir / manifest_name
    manifest = read_object(manifest_path, label)
    rows = manifest.get("artifacts")
    if (not isinstance(rows, list) or len(rows) != len(expected_names)
            or {row.get("filename") for row in rows if isinstance(row, dict)}
            != set(expected_names)):
        raise BakeoffError(f"{label} must inventory exactly {', '.join(expected_names)}")
    if manifest.get("partition", {}).get("pairwise_request_overlap_count") != 0:
        raise BakeoffError(f"{label} corpora are not pairwise disjoint")
    evidence: dict[str, Any] = {}
    for name in expected_names:
        row = artifact_by_name(manifest, name)
        path = corpus_dir / name
        if not path.is_file() or sha256(path) != row.get("sha256"):
            raise BakeoffError(f"corpus artifact differs from its manifest: {name}")
        evidence[name] = {
            "path": str(path.resolve()), "sha256": row["sha256"],
            "record_count": row["record_count"],
        }
    return manifest, manifest_path, evidence


def make_plan(
    recipe_path: Path, corpus_dir: Path, profile_path: Path = DEFAULT_PROFILE,
) -> dict[str, Any]:
    recipe = read_object(recipe_path, "bakeoff recipe")
    if recipe.get("schema_version") != RECIPE_SCHEMA_VERSION:
        raise BakeoffError("unsupported bakeoff recipe")
    profile = read_object(profile_path, "release profile")
    profile_bakeoff = profile.get("quantization", {}).get("performance_bakeoff", {})
    profile_rows = profile_bakeoff.get("arms")
    if not isinstance(profile_rows, list):
        raise BakeoffError("release profile has no quantization arm inventory")
    measurement_policy = recipe.get("measurement_policy")
    if (not isinstance(measurement_policy, dict)
            or measurement_policy.get("winner_order") != WINNER_ORDER
            or profile_bakeoff.get("winner_order") != WINNER_ORDER):
        raise BakeoffError(
            "recipe and release profile must pin the performance-first winner order")
    profile_arms = {
        row.get("id"): row for row in profile_rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    profile_sha = sha256(profile_path)
    pairing = recipe.get("mtp_depth_sweep")
    if (not isinstance(pairing, dict)
            or pairing.get("pairing_reference_depth") != 3
            or pairing.get("depths") != [1, 2, 3, 4]
            or pairing.get("selected_exact_runtime_pair_only") is not True
            or pairing.get("reuse_exact_main_and_companion_artifacts") is not True
            or pairing.get("runtime_mode") != "exact_dequant"
            or pairing.get("final_release_eligible") is not True):
        raise BakeoffError("MTP depth sweep must pin exact-runtime depths 1..4 after depth-3 pairing")
    supported_mtp = set(profile.get("quantization", {}).get(
        "performance_bakeoff", {}).get("supported_mtp_matrix_contracts", []))
    if supported_mtp != SUPPORTED_MTP_MATRIX_CONTRACTS:
        raise BakeoffError("release profile does not declare both supported MTP contracts")

    def normalized_arm(recipe_arm: Any, *, auxiliary: bool) -> dict[str, Any]:
        if not isinstance(recipe_arm, dict):
            raise BakeoffError("bakeoff arm must be an object")
        quantization_arm = recipe_arm.get("quantization_arm")
        profile_arm = profile_arms.get(quantization_arm)
        overrides = profile_arm.get("per_tensor_overrides") if profile_arm else None
        mtp_contract = recipe_arm.get("mtp_matrix_quant_contract")
        mtp_depth = recipe_arm.get("mtp_depth")
        if (not isinstance(overrides, list)
                or mtp_contract not in SUPPORTED_MTP_MATRIX_CONTRACTS
                or isinstance(mtp_depth, bool) or not isinstance(mtp_depth, int)
                or not 1 <= mtp_depth <= 4):
            raise BakeoffError(
                f"recipe arm {recipe_arm.get('id')} is not bound to one supported main/MTP pair")
        if auxiliary:
            if (recipe_arm.get("id") != "rocmi4-w4a4"
                    or recipe_arm.get("runtime_mode") != "w4a4_opt_in"
                    or recipe_arm.get("classification") != "auxiliary_performance_control_only"
                    or recipe_arm.get("included_in_exact_runtime_winner_ledger") is not False
                    or recipe_arm.get("final_release_eligible") is not False):
                raise BakeoffError("auxiliary control could enter the exact-runtime winner ledger")
        elif (recipe_arm.get("runtime_mode") != "exact_dequant"
              or recipe_arm.get("final_release_eligible") is not True
              or mtp_depth != pairing["pairing_reference_depth"]):
            raise BakeoffError("selectable format arms must be exact-runtime depth-3 pairs")
        serialized = json.dumps(overrides, separators=(",", ":"), ensure_ascii=True)
        return {
            **recipe_arm,
            "profile_sha256": profile_sha,
            "quantization_overrides_sha256": hashlib.sha256(
                serialized.encode("utf-8")).hexdigest(),
        }

    format_rows = recipe.get("format_arms")
    if not isinstance(format_rows, list) or not format_rows:
        raise BakeoffError("bakeoff recipe has no exact-runtime format arms")
    format_arms = [normalized_arm(row, auxiliary=False) for row in format_rows]
    if len({row.get("id") for row in format_arms}) != len(format_arms):
        raise BakeoffError("exact-runtime format arm ids must be unique")
    expected_pairs = {
        (main, mtp) for main in {
            "rocmi4-q6k-embedding-head",
            "rocmfp4-fast-routed-experts-q6k-embedding-head",
            "rocmfp4-fast-matrix-q6k-embedding-head",
            "rocmfp4-fast-matrix-q3-ple-q6k-embedding-head",
        } for mtp in SUPPORTED_MTP_MATRIX_CONTRACTS
    }
    actual_pairs = {(row["quantization_arm"], row["mtp_matrix_quant_contract"])
                    for row in format_arms}
    if actual_pairs != expected_pairs or len(format_arms) != len(expected_pairs):
        raise BakeoffError("exact-runtime format arms must cover the full main/MTP cross-pair")
    auxiliary_rows = recipe.get("auxiliary_controls")
    if not isinstance(auxiliary_rows, list) or len(auxiliary_rows) != 1:
        raise BakeoffError("bakeoff recipe must declare exactly one auxiliary W4A4 control")
    auxiliary_controls = [normalized_arm(auxiliary_rows[0], auxiliary=True)]
    manifest_names = recipe.get("corpus_manifests") or {}
    selection_name = manifest_names.get("selection")
    if not isinstance(selection_name, str):
        raise BakeoffError("recipe lacks a phase-scoped selection corpus manifest")
    _corpus_manifest, corpus_manifest_path, corpus_evidence = _manifest_artifacts(
        corpus_dir, selection_name,
        ("extraction-good.jsonl", "extraction-bad.jsonl", "sweep-validation.jsonl"),
        "selection corpus manifest")

    configurations = []
    source = profile.get("source") or {}
    intervention = profile.get("intervention") or {}
    direction_basis = {
        "source": {key: source.get(key) for key in (
            "repo_id", "revision", "snapshot_inventory_sha256")},
        "tooling": {
            "otheru_quant_pipeline": intervention.get("otheru_pipeline"),
            "upstream_heretic": intervention.get("upstream_heretic"),
            "extractor": {
                "implementation": "ember-qwen-residual-writer-activation-extractor",
                "schema_version": 2,
            },
        },
        "extraction": {
            "semantic_capture_point": "decoder_layer.residual_writer.output",
            "transformers_hook_module": (
                "model.language_model.layers.N.{linear_attn.out_proj|self_attn.o_proj}"
            ),
            "transformers_hook_value": "forward_output[:,-1,:]",
            "hidden_states_api_used": False,
            "policy_evidence": {
                "source_revision": "a3c6a728510f91394e991504951ac316cd3a89af",
                "deepseek_reference_band": "10-42",
                "qwen_status": "exploratory_transfer_hypothesis",
            },
        },
        "corpora": [
            {"class": "good_control", "role": "direction_extraction",
             "sha256": corpus_evidence["extraction-good.jsonl"]["sha256"],
             "record_count": corpus_evidence["extraction-good.jsonl"]["record_count"]},
            {"class": "bad_target", "role": "direction_extraction",
             "sha256": corpus_evidence["extraction-bad.jsonl"]["sha256"],
             "record_count": corpus_evidence["extraction-bad.jsonl"]["record_count"]},
        ],
    }
    sweep_arm = profile_arms.get("rocmi4-control")
    sweep_overrides = sweep_arm.get("per_tensor_overrides") if sweep_arm else None
    if not isinstance(sweep_overrides, list):
        raise BakeoffError("release profile lacks the ROCMI4 sweep control arm")
    sweep_override_sha = hashlib.sha256(json.dumps(
        sweep_overrides, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    for scale in recipe["intervention_sweep"]["lambdas"]:
        for policy in recipe["intervention_sweep"]["layer_policies"]:
            identifier = f"lambda-{scale:.2f}-{policy}"
            selected = set(layers(policy))
            configurations.append({
                "id": identifier,
                "scale": scale,
                "layer_policy": policy,
                "quantization_arm": "rocmi4-control",
                "profile_sha256": profile_sha,
                "quantization_overrides_sha256": sweep_override_sha,
                "runtime_mode": recipe["intervention_sweep"]["runtime_mode"],
                "final_release_eligible": False,
                "direction_basis": direction_basis,
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
        "schema_version": PLAN_SCHEMA_VERSION,
        "phase_scope": "selection",
        "status": "planned_unmeasured",
        "recipe": {"path": str(recipe_path.resolve()), "sha256": sha256(recipe_path),
                   "value": recipe},
        "release_profile": {"path": str(profile_path.resolve()),
                            "sha256": profile_sha},
        "corpus_manifest": {"path": str(corpus_manifest_path.resolve()),
                            "sha256": sha256(corpus_manifest_path)},
        "corpora": corpus_evidence,
        "stock_control": recipe["stock_control"],
        "direction_basis": direction_basis,
        "sweep_configurations": configurations,
        "format_arms": format_arms,
        "mtp_depth_configurations": [
            {
                "id": f"mtp-depth-{depth}",
                "mtp_depth": depth,
                "selected_pair_source": "sealed_format_ledger",
                "runtime_mode": pairing["runtime_mode"],
                "final_release_eligible": pairing["final_release_eligible"],
            }
            for depth in pairing["depths"]
        ],
        "auxiliary_controls": auxiliary_controls,
        "required_result_stages": ["stock", "sweep", "format", "mtp-depth"],
        "publication_allowed": False,
    }


def make_final_plan(selection: dict[str, Any], corpus_dir: Path,
                    depth_descriptor: dict[str, Any]) -> dict[str, Any]:
    """Unlock final-heldout only after the main/MTP pair and depth are sealed.

    ``selection`` contains no final corpus pathname or digest.  The caller must
    first supply a pinned, externally-attested MTP-depth ledger; only then is
    the final-only manifest opened and incorporated into a new plan.
    """
    selection = verify_plan(selection)
    prior, prior_digest = read_prior_ledger(depth_descriptor, "mtp-depth", selection)
    recipe = selection["recipe"]["value"]
    final_name = (recipe.get("corpus_manifests") or {}).get("final")
    if not isinstance(final_name, str):
        raise BakeoffError("recipe lacks a phase-scoped final corpus manifest")
    _manifest, manifest_path, corpus = _manifest_artifacts(
        corpus_dir, final_name, ("final-heldout.jsonl",), "final corpus manifest")
    final = corpus["final-heldout.jsonl"]
    sweep = selection["corpora"]["sweep-validation.jsonl"]
    if final["sha256"] == sweep["sha256"]:
        raise BakeoffError("sweep and final corpora must differ")
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "phase_scope": "final_confirmation",
        "status": "final_heldout_unlocked_after_mtp_depth_selection",
        "selection_plan": selection,
        "selection_plan_sha256": canonical_sha256(selection),
        "sealed_recipe_ledger": {
            "descriptor": depth_descriptor,
            "sha256": prior_digest,
            "selected_runtime_identity_sha256": canonical_sha256({
                "artifact_identity": prior["selected_artifact_identity"],
                "mtp_matrix_quant_contract": prior["selected_mtp_matrix_quant_contract"],
                "mtp_depth": prior["selected_mtp_depth"],
            }),
        },
        "final_corpus_manifest": {
            "path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
        "corpora": {"final-heldout.jsonl": final},
        "publication_allowed": False,
    }


def verify_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Recreate a plan from checked policy files and compare every semantic byte."""
    if plan.get("phase_scope") == "final_confirmation":
        selection = plan.get("selection_plan")
        if not isinstance(selection, dict):
            raise BakeoffError("final plan does not embed its selection plan")
        selection = verify_plan(selection)
        if plan.get("selection_plan_sha256") != canonical_sha256(selection):
            raise BakeoffError("final plan selection-plan commitment differs")
        sealed = plan.get("sealed_recipe_ledger") or {}
        final_manifest = plan.get("final_corpus_manifest") or {}
        if (set(sealed) != {"descriptor", "sha256", "selected_runtime_identity_sha256"}
                or set(final_manifest) != {"path", "sha256"}):
            raise BakeoffError("final plan descriptors are malformed")
        if sha256(Path(final_manifest["path"])) != final_manifest["sha256"]:
            raise BakeoffError("final corpus manifest digest differs")
        expected = make_final_plan(selection, Path(final_manifest["path"]).parent,
                                   sealed["descriptor"])
        if plan != expected:
            raise BakeoffError("final plan semantics differ from its sealed selection")
        return expected
    if plan.get("phase_scope") != "selection":
        raise BakeoffError("plan lacks an explicit selection/final phase scope")
    recipe = plan.get("recipe") or {}
    profile = plan.get("release_profile") or {}
    corpus = plan.get("corpus_manifest") or {}
    if Path(str(recipe.get("path", ""))).resolve() != DEFAULT_RECIPE.resolve():
        raise BakeoffError("plan recipe is not the checked-in bakeoff recipe")
    if Path(str(profile.get("path", ""))).resolve() != DEFAULT_PROFILE.resolve():
        raise BakeoffError("plan profile is not the checked-in release profile")
    corpus_path = Path(str(corpus.get("path", ""))).resolve()
    expected_name = ((plan.get("recipe") or {}).get("value") or {}).get(
        "corpus_manifests", {}).get("selection")
    if corpus_path.name != expected_name:
        raise BakeoffError("plan corpus manifest path is malformed")
    expected = make_plan(DEFAULT_RECIPE.resolve(), corpus_path.parent,
                         DEFAULT_PROFILE.resolve())
    if plan != expected:
        raise BakeoffError(
            "bakeoff plan semantics differ from the plan derived from checked policy/corpora")
    return expected


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise BakeoffError(f"{label} must be a finite measured number")
    return float(value)


def audited_quality(row: dict[str, Any], corpus_sha: str) -> dict[str, Any]:
    descriptor = row.get("quality_contract")
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise BakeoffError("candidate row must carry exactly one quality contract path and SHA-256")
    path = descriptor["path"]
    digest = descriptor["sha256"]
    if not isinstance(path, str) or not path or not isinstance(digest, str):
        raise BakeoffError("quality contract descriptor is malformed")
    try:
        result = QUALITY_EVALUATOR(Path(path), digest)
    except (QUALITY_MODULE.EvidenceError, OSError, ValueError) as exc:
        raise BakeoffError(f"quality contract verification failed: {exc}") from exc
    candidate = result.get("models", {}).get("candidate", {})
    expected = {
        "candidate_id": row.get("candidate_id"),
        "build_record_sha256": row.get("build_record_sha256"),
        "intervention_manifest_sha256": row.get("intervention_manifest_sha256"),
        "profile_sha256": row.get("profile_sha256"),
        "quantization_overrides_sha256": row.get("quantization_overrides_sha256"),
        "inventory_sha256": row.get("model_inventory_sha256"),
        "artifact_bytes": row.get("artifact_bytes"),
    }
    if any(candidate.get(key) != value for key, value in expected.items()):
        raise BakeoffError("quality evidence candidate differs from the measured candidate provenance")
    if result.get("corpus", {}).get("sha256") != corpus_sha:
        raise BakeoffError("quality evidence used the wrong pinned corpus partition")
    if (result.get("release_scope") != {
            "modality": "text_only", "multimodal_release_claim": False,
            "vision_mmproj_differential_pass": False}
            or result.get("multimodal_release_approved") is not False):
        raise BakeoffError("quality evidence overstates the untested multimodal release scope")
    if result.get("audited_quality_pass") is not True:
        return {"passes": False, "quality_score": result.get("quality_score"),
                "evidence": result}
    return {"passes": True,
            "quality_score": finite_number(result.get("quality_score"), "audited quality score"),
            "evidence": result}


def pinned_json(descriptor: Any, label: str) -> tuple[dict[str, Any], str]:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise BakeoffError(f"{label} descriptor must contain only path and sha256")
    try:
        value, digest = QUALITY_MODULE.read_pinned_json(
            Path("/"), descriptor, label)
    except (QUALITY_MODULE.EvidenceError, OSError, ValueError) as exc:
        raise BakeoffError(f"cannot verify {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise BakeoffError(f"{label} must be a JSON object")
    return value, digest


def pinned_evidence_path(base: Path, descriptor: Any, label: str) -> tuple[str, Path]:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise BakeoffError(f"{label} descriptor must contain only path and sha256")
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise BakeoffError(f"{label} path is malformed")
    path = Path(raw_path)
    path = (path if path.is_absolute() else base / path).resolve()
    if path.is_symlink() or not path.is_file():
        raise BakeoffError(f"{label} is not a regular evidence file")
    digest = sha256(path)
    if digest != descriptor.get("sha256"):
        raise BakeoffError(f"{label} digest mismatch")
    return digest, path


def pinned_evidence(base: Path, descriptor: Any,
                    label: str) -> tuple[dict[str, Any], str, Path]:
    digest, path = pinned_evidence_path(base, descriptor, label)
    return read_object(path, label), digest, path


def timing_facts(path: Path, expected_spec: bool) -> dict[str, Any]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise BakeoffError(f"cannot parse timing evidence: {exc}") from exc
    metadata = [item for item in rows if item.get("kind") == "metadata"]
    summaries = [item for item in rows if item.get("kind") == "summary"]
    requests = [item for item in rows if item.get("kind") == "request" and item.get("ok")]
    prefill = [item for item in requests if item.get("group") == "prefill-2048"]
    decode = [item for item in requests if item.get("group") == "decode-256"]
    if len(metadata) != 1 or len(summaries) != 1 or len(prefill) != 3 or len(decode) != 3:
        raise BakeoffError("timing evidence lacks one metadata/summary row and three samples per shape")
    resources = summaries[0].get("resources") or {}
    if (metadata[0].get("server_pid_source") != "explicit"
            or metadata[0].get("container_pid") != resources.get("server_host_pid")):
        raise BakeoffError("timing evidence is not bound to one explicit server PID")
    evaluated = [item.get("evaluated_prefill_tokens") for item in prefill]
    completed = [item.get("completion_tokens") for item in decode]
    spec_ran = [item.get("spec_ran") for item in decode]
    accept_rates = [item.get("accept_rate") for item in decode]
    if evaluated != [2074, 2074, 2074] or completed != [256, 256, 256]:
        raise BakeoffError("timing evidence does not use exact 2074/256 shapes")
    if spec_ran != [expected_spec, expected_spec, expected_spec]:
        raise BakeoffError("timing evidence has the wrong speculative execution mode")
    if expected_spec and any(
            isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not 0.0 < float(rate) < 1.0 for rate in accept_rates):
        raise BakeoffError("every matching-MTP sample must prove 0 < accept_rate < 1")
    speculation = None
    if expected_spec:
        timing_keys = ("spec_provider_age_ms", "spec_provider_block_ms",
                       "spec_head_ms", "spec_verify_ms")
        if any(not isinstance(item.get("spec_cycles"), int)
               or isinstance(item.get("spec_cycles"), bool)
               or item["spec_cycles"] <= 0
               or any(isinstance(item.get(key), bool)
                      or not isinstance(item.get(key), (int, float))
                      or not math.isfinite(float(item[key]))
                      or float(item[key]) < 0.0 for key in timing_keys)
               for item in decode):
            raise BakeoffError(
                "matching-MTP samples lack complete speculative timing evidence")
        cycles = sum(item["spec_cycles"] for item in decode)
        speculation = {
            "samples": len(decode), "timing_complete": True,
            "cycles": cycles,
            "accept_rate_mean": statistics.fmean(
                float(rate) for rate in accept_rates),
        }
        for key in timing_keys:
            total = sum(float(item[key]) for item in decode)
            speculation[f"{key}_total"] = total
            speculation[f"{key}_per_cycle"] = total / cycles
        if (((summaries[0].get("groups") or {}).get("decode-256") or {}).get(
                "speculation") != speculation):
            raise BakeoffError(
                "matching-MTP timing summary differs from its request rows")
    prefill_rates = []
    for item in prefill:
        rate, consistent = BENCHMARK_MODULE.derived_tps(
            item.get("evaluated_prefill_tokens"), item.get("prefill_ms"),
            item.get("declared_prefill_tokens_per_second"), declared_decimals=1)
        if (rate is None or not consistent
                or item.get("prefill_tps_rounding_consistent") is not True
                or not math.isclose(
                    finite_number(item.get("prefill_tokens_per_second"), "prefill sample"),
                    rate, rel_tol=1e-12, abs_tol=1e-12)):
            raise BakeoffError("prefill throughput is not independently derived from count/time")
        prefill_rates.append(rate)
    decode_rates = []
    for item in decode:
        rate, consistent = BENCHMARK_MODULE.derived_tps(
            item.get("completion_tokens"), item.get("decode_ms"),
            item.get("declared_decode_tokens_per_second"), declared_decimals=2)
        if (rate is None or not consistent
                or item.get("decode_tps_rounding_consistent") is not True
                or not math.isclose(
                    finite_number(item.get("decode_tokens_per_second"), "decode sample"),
                    rate, rel_tol=1e-12, abs_tol=1e-12)):
            raise BakeoffError("decode throughput is not independently derived from count/time")
        decode_rates.append(rate)
    return {
        "prefill_tps_samples": prefill_rates,
        "decode_tps_samples": decode_rates,
        "evaluated_prefill_tokens": evaluated,
        "completion_tokens": completed,
        "mtp_spec_ran": spec_ran,
        "mtp_accept_rates": accept_rates,
        "mtp_speculation": speculation,
        "resources": resources,
        "hard_gate": summaries[0].get("hard_gate"),
        "memory_gate": summaries[0].get("memory_gate"),
    }


def validate_measurement_evidence(row: dict[str, Any]) -> dict[str, Any]:
    descriptor = row.get("evidence_manifest")
    if (not isinstance(descriptor, dict)
            or set(descriptor) != {"path", "sha256", "schema"}
            or descriptor.get("schema") != RESULT_SCHEMA):
        raise BakeoffError("measurement lacks a v4 evidence-manifest descriptor")
    manifest, digest = pinned_json(
        {"path": descriptor["path"], "sha256": descriptor["sha256"]},
        "measurement evidence manifest")
    if (manifest.get("schema") != descriptor["schema"]
            or manifest.get("candidate_id") != row.get("candidate_id")
            or manifest.get("status") != "complete" or manifest.get("publishes") is not False):
        raise BakeoffError("measurement evidence identity/status differs from its row")
    expected_provenance = {
        "quantization_arm": row.get("quantization_arm"),
        "override_sha256": row.get("quantization_overrides_sha256"),
        "intervention_configuration_id": row.get("intervention_configuration_id"),
        "intervention_manifest_sha256": row.get("intervention_manifest_sha256"),
        "build_record_sha256": row.get("build_record_sha256"),
        "companion_inventory_sha256": row.get("companion_inventory_sha256"),
        "mtp_matrix_quant_contract": row.get("mtp_matrix_quant_contract"),
        "mtp_depth": row.get("mtp_depth"),
        "profile_sha256": row.get("profile_sha256"),
        "builder_identity": row.get("builder_identity"),
        "runtime_identity": row.get("runtime_identity"),
        "tensor_format_compatibility_sha256": row.get(
            "tensor_format_compatibility_sha256"),
    }
    if manifest.get("provenance") != expected_provenance:
        raise BakeoffError("measurement evidence provenance differs from its row")
    artifacts = manifest.get("artifacts") or {}
    if artifacts.get("combined_fits") is not True:
        raise BakeoffError("measurement evidence does not record an exact combined fit")
    evidence = manifest.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
            "target_complete", "matching_mtp_hardware_measurement",
            "matching_mtp_timing", "quality_contract"}:
        raise BakeoffError("measurement evidence inventory is incomplete")
    target, _, target_path = pinned_evidence(
        Path("/"), evidence["target_complete"], "target-only completion")
    hardware, _, hardware_path = pinned_evidence(
        Path("/"), evidence["matching_mtp_hardware_measurement"],
        "matching-MTP hardware measurement")
    if (target.get("schema") != "ember.qwen3.8.target-only-gate.v1"
            or target.get("passed") is not True or target.get("release_approval") is not False
            or target.get("publishes") is not False
            or (target.get("model") or {}).get("build_record_sha256") != row.get("build_record_sha256")):
        raise BakeoffError("target-only evidence is incomplete or bound to another build")
    if (hardware.get("schema") != "ember.qwen3.8.real-weight-gate.v2"
            or hardware.get("publish_approved") is not False
            or hardware.get("certification_scope") not in {
                "text_model_plus_mtp_only", "measurement_only_not_certified"}
            or (hardware.get("evidence", {}).get("quant_build_record") or {}).get("sha256") !=
            row.get("build_record_sha256")):
        raise BakeoffError("matching-MTP hardware evidence is incomplete or bound to another build")

    target_files = target.get("evidence") or {}
    target_timing_sha, target_timing_path = pinned_evidence_path(
        target_path.parent, target_files.get("timing"), "target-only timing")
    target_facts = timing_facts(target_timing_path, False)
    target_summary, _, _ = pinned_evidence(
        target_path.parent, target_files.get("summary"), "target-only summary")
    if (target_summary.get("hard_gate_observation") != target_facts["hard_gate"]
            or target_summary.get("resources") != target_facts["resources"]
            or target_summary.get("memory_gate") != target_facts["memory_gate"]):
        raise BakeoffError("target-only completion does not match its pinned timing evidence")

    hardware_files = hardware.get("evidence") or {}
    differential, _, _ = pinned_evidence(
        hardware_path.parent, hardware_files.get("differential"),
        "matching-MTP differential")
    prefill = differential.get("prefill") or {}
    spec = differential.get("spec") or {}
    rate = spec.get("accept_rate")
    if (differential.get("ok") is not True or
            differential.get("snapshot_ok") is not True or
            prefill.get("checked") is not True or
            prefill.get("exact") is not True
            or spec.get("checked") is not True or spec.get("exact") is not True
            or isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not 0.0 < float(rate) < 1.0):
        raise BakeoffError("matching-MTP differential is not exact or did not exercise accept/reject")
    differential_decode_record, _, _ = pinned_evidence(
        hardware_path.parent, hardware_files.get("differential_decode"),
        "matching-MTP differential decode")
    try:
        differential_decode = BENCHMARK_MODULE.validate_differential_decode(
            differential_decode_record)
    except ValueError as exc:
        raise BakeoffError(f"matching-MTP {exc}") from exc
    if hardware.get("differential_decode") != differential_decode:
        raise BakeoffError(
            "matching-MTP differential decode summary differs from pinned evidence")
    hardware_timing_sha, hardware_timing_path = pinned_evidence_path(
        hardware_path.parent, hardware_files.get("timing"), "matching-MTP timing")
    if (hardware_timing_sha != (evidence["matching_mtp_timing"] or {}).get("sha256")
            or Path(str((evidence["matching_mtp_timing"] or {}).get("path"))).resolve()
            != hardware_timing_path):
        raise BakeoffError("outer evidence manifest substitutes different matching-MTP timing")
    facts = timing_facts(hardware_timing_path, True)
    memory, _, _ = pinned_evidence(
        hardware_path.parent, hardware_files.get("memory"), "matching-MTP memory")
    if (memory.get("resources") != facts["resources"]
            or memory.get("performance") != facts["hard_gate"]
            or memory.get("hard_fit") != facts["memory_gate"]
            or memory.get("speculation") != facts["mtp_speculation"]
            or hardware.get("resources") != facts["resources"]
            or hardware.get("speculation") != facts["mtp_speculation"]
            or (hardware.get("hard_gates") or {}).get("performance") != facts["hard_gate"]
            or (hardware.get("hard_gates") or {}).get("memory") != facts["memory_gate"]):
        raise BakeoffError("hardware/memory summaries differ from pinned timing evidence")

    inventory = (hardware.get("model") or {}).get("ordered_inventory") or {}
    shards = inventory.get("shards")
    if not isinstance(shards, list) or not shards:
        raise BakeoffError("hardware evidence lacks an ordered model inventory")
    identities = []
    for index, shard in enumerate(shards, 1):
        if not isinstance(shard, dict):
            raise BakeoffError("hardware model inventory contains a malformed shard")
        identities.append({"index": index, "sha256": shard.get("sha256"),
                           "bytes": shard.get("size_bytes")})
    inventory_sha = canonical_sha256(identities)
    if inventory_sha != row.get("model_inventory_sha256"):
        raise BakeoffError("hardware evidence ordered model inventory differs from its row")
    if any(not isinstance(item.get("bytes"), int) or isinstance(item.get("bytes"), bool)
           or item["bytes"] < 1 for item in identities):
        raise BakeoffError("hardware model inventory has an invalid shard size")
    artifact_bytes = sum(item["bytes"] for item in identities)
    if artifact_bytes != artifacts.get("model_artifact_bytes"):
        raise BakeoffError("main artifact bytes do not equal the ordered hardware shard sum")
    companion_bytes: dict[str, int] = {}
    for role in ("mtp", "vision_mmproj", "vision_vocab"):
        item = artifacts.get(role) or {}
        size = item.get("bytes")
        path_value = item.get("path")
        if (not isinstance(size, int) or isinstance(size, bool) or size < 1
                or not isinstance(path_value, str)):
            raise BakeoffError(f"{role} companion is absent or empty")
        companion_path = Path(path_value)
        if (companion_path.is_symlink() or not companion_path.is_file()
                or companion_path.stat().st_size != size):
            raise BakeoffError(f"{role} companion byte count differs from the exact artifact")
        companion_bytes[role] = size
    hardware_mtp = hardware.get("mtp") or {}
    if (hardware_mtp.get("path") != (artifacts.get("mtp") or {}).get("path")
            or hardware_mtp.get("sha256") != (artifacts.get("mtp") or {}).get("sha256")
            or hardware_mtp.get("depth") != row.get("mtp_depth")):
        raise BakeoffError("hardware run used a different MTP companion")

    derived = {**facts, "differential_correctness_pass": True,
               "differential_decode": differential_decode,
               "artifact_bytes": artifact_bytes,
               "companion_artifact_bytes": companion_bytes}
    expected_row = {
        "prefill_tps_samples": facts["prefill_tps_samples"],
        "decode_tps_samples": facts["decode_tps_samples"],
        "evaluated_prefill_tokens": facts["evaluated_prefill_tokens"],
        "completion_tokens": facts["completion_tokens"],
        "mtp_spec_ran": facts["mtp_spec_ran"],
        "artifact_bytes": artifact_bytes,
        "companion_artifact_bytes": companion_bytes,
    }
    for key in ("runner_memtotal_bytes", "runner_gtt_pages_limit",
                "peak_memory_measurement_method", "measured_peak_rss_bytes",
                "measured_peak_gtt_bytes", "measured_peak_uma_bytes"):
        expected_row[key] = facts["resources"].get(key)
    if any(row.get(key) != value for key, value in expected_row.items()):
        raise BakeoffError("measurement row differs from independently derived evidence")
    if manifest.get("measurement_contract") != {
        "prefill_statistic": "peak", "decode_statistic": "median",
        "evaluated_prefill_tokens": facts["evaluated_prefill_tokens"],
        "completion_tokens": facts["completion_tokens"],
        "mtp_spec_ran": facts["mtp_spec_ran"],
        "mtp_depth": row.get("mtp_depth"),
    }:
        raise BakeoffError("measurement manifest contract differs from pinned timing")
    if row.get("stage") == "stock":
        if evidence["quality_contract"] is not None:
            raise BakeoffError("stock evidence must not claim a candidate quality contract")
    elif evidence["quality_contract"] != row.get("quality_contract"):
        raise BakeoffError("measurement and row quality-contract descriptors differ")
    return {"manifest_sha256": digest, "target": target, "hardware": hardware,
            "target_timing_sha256": target_timing_sha, "derived": derived}


EVIDENCE_VALIDATOR = validate_measurement_evidence


def validate_revision_identities(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    builder = row.get("builder_identity")
    runtime = row.get("runtime_identity")
    builder_keys = {"ember_revision", "quantizer_tool_sha256", "container_digest",
                    "tensor_format_contract_sha256"}
    runtime_keys = {"ember_revision", "engine_binary_sha256", "container_digest",
                    "tensor_format_contract_sha256"}
    if (not isinstance(builder, dict) or set(builder) != builder_keys
            or not isinstance(runtime, dict) or set(runtime) != runtime_keys):
        raise BakeoffError("measurement must separate exact builder and runtime identities")
    if (re.fullmatch(r"[0-9a-f]{40}", str(builder["ember_revision"])) is None
            or re.fullmatch(r"[0-9a-f]{40}", str(runtime["ember_revision"])) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(builder["quantizer_tool_sha256"])) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(runtime["engine_binary_sha256"])) is None
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(builder["container_digest"])) is None
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(runtime["container_digest"])) is None):
        raise BakeoffError("builder/runtime revisions, tools, or images are not exact digests")
    contract = row.get("tensor_format_compatibility_sha256")
    if (re.fullmatch(r"[0-9a-f]{64}", str(contract)) is None
            or builder["tensor_format_contract_sha256"] != contract
            or runtime["tensor_format_contract_sha256"] != contract):
        raise BakeoffError("builder and runtime do not prove one tensor-format compatibility contract")
    return builder, runtime


def direction_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the scale-independent identity shared by every intervention.

    Targets and held-out scores intentionally are not part of this identity:
    targets encode the arm's lambda/layer policy and final-heldout is not
    available during extraction.  Everything that can change the direction
    itself is included, including all 48 packed-F32 direction digests.
    """
    direction_rows = manifest.get("directions")
    if not isinstance(direction_rows, list) or len(direction_rows) != 48:
        raise BakeoffError("intervention manifest must carry exactly 48 directions")
    compact = []
    seen: set[int] = set()
    for item in direction_rows:
        if not isinstance(item, dict):
            raise BakeoffError("intervention direction row is malformed")
        layer = item.get("layer")
        digest = item.get("sha256")
        if (not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < 48
                or layer in seen or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise BakeoffError("intervention direction layer/digest inventory is malformed")
        seen.add(layer)
        compact.append({
            "layer": layer, "id": item.get("id"), "dtype": item.get("dtype"),
            "activation": item.get("activation"), "sha256": digest,
        })
    compact.sort(key=lambda item: item["layer"])
    if [item["layer"] for item in compact] != list(range(48)):
        raise BakeoffError("intervention directions do not cover layers 0-47")
    extraction = manifest.get("extraction")
    if not isinstance(extraction, dict) or not isinstance(
            extraction.get("activation_evidence"), dict):
        raise BakeoffError("intervention extraction/activation evidence is absent")
    identity = {
        "schema": "ember.qwen3.8.direction-identity.v1",
        "source": manifest.get("source"),
        "tooling": manifest.get("tooling"),
        "corpora": manifest.get("corpora"),
        "extraction": extraction,
        "directions": compact,
    }
    for key in ("source", "tooling", "corpora"):
        if not identity[key]:
            raise BakeoffError(f"intervention direction identity lacks {key}")
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def validate_intervention_binding(row: dict[str, Any],
                                  configuration: dict[str, Any],
                                  corpus_sha: str) -> dict[str, Any]:
    descriptor = row.get("intervention_manifest")
    manifest, digest = pinned_json(descriptor, "intervention manifest")
    if digest != row.get("intervention_manifest_sha256"):
        raise BakeoffError("intervention manifest digest differs from candidate provenance")
    if ((manifest.get("held_out_evaluation") or {}).get("sha256") != corpus_sha
            or manifest.get("kind") != "directional_ablation"
            or manifest.get("status") != "complete"):
        raise BakeoffError("intervention manifest used the wrong held-out corpus or is incomplete")
    actual = {str(index): 0.0 for index in range(48)}
    targets = manifest.get("targets")
    if not isinstance(targets, list):
        raise BakeoffError("intervention manifest has no target list")
    for target in targets:
        if not isinstance(target, dict):
            raise BakeoffError("intervention target is malformed")
        match = re.fullmatch(r"blk\.([0-9]+)\.(?:attn_output|ssm_out)\.weight",
                             str(target.get("tensor_name")))
        scale = target.get("scale")
        if (match is None or isinstance(scale, bool) or not isinstance(scale, (int, float))):
            raise BakeoffError("intervention target tensor/scale is malformed")
        layer = int(match.group(1))
        if not 0 <= layer < 48 or actual[str(layer)] != 0.0:
            raise BakeoffError("intervention target layer is duplicated or out of range")
        actual[str(layer)] = float(scale)
    expected = {key: float(value) for key, value in configuration["layer_scales"].items()}
    if actual != expected:
        raise BakeoffError("intervention manifest lambda/layer scales differ from the selected plan")
    if (row.get("intervention_configuration_id") != configuration["id"]
            or row.get("configuration_id", configuration["id"]) != configuration["id"]):
        raise BakeoffError("candidate intervention configuration identity differs from the plan")
    identity = direction_identity(manifest)
    basis = configuration.get("direction_basis") or {}
    compact_corpora = [{key: item.get(key) for key in (
        "class", "role", "sha256", "record_count")}
        for item in identity["corpora"] if isinstance(item, dict)]
    extraction = identity.get("extraction") or {}
    expected_extraction = basis.get("extraction") or {}
    if (identity.get("source") != basis.get("source")
            or identity.get("tooling") != basis.get("tooling")
            or compact_corpora != basis.get("corpora")
            or any(extraction.get(key) != value
                   for key, value in expected_extraction.items())
            or extraction.get("layer_policy") != configuration.get("layer_policy")):
        raise BakeoffError(
            "intervention direction source/tool/extraction semantics differ from the plan"
        )
    return identity


INTERVENTION_VALIDATOR = validate_intervention_binding


def assess(row: dict[str, Any], gates: dict[str, Any], corpus_sha: str) -> dict[str, Any]:
    if row.get("measurement_kind") != "measured" or row.get("status") != "complete":
        raise BakeoffError("every bakeoff row must be a complete measured result")
    if row.get("corpus_sha256") != corpus_sha:
        raise BakeoffError("result used the wrong corpus partition")
    measured_evidence = EVIDENCE_VALIDATOR(row)
    derived = measured_evidence.get("derived")
    if not isinstance(derived, dict):
        raise BakeoffError("evidence validator did not return independently derived measurements")
    if derived.get("differential_correctness_pass") is not True:
        return {"passes": False}
    artifact_bytes = derived.get("artifact_bytes")
    if not isinstance(artifact_bytes, int) or isinstance(artifact_bytes, bool) or artifact_bytes < 1:
        raise BakeoffError("artifact_bytes must be a measured positive integer")
    prefill = derived.get("prefill_tps_samples")
    decode = derived.get("decode_tps_samples")
    count = gates["minimum_samples_per_performance_gate"]
    if (not isinstance(prefill, list) or not isinstance(decode, list)
            or len(prefill) != count or len(decode) != count):
        raise BakeoffError("performance rows require exactly three measured samples per gate")
    if derived.get("evaluated_prefill_tokens") != [2074] * count:
        raise BakeoffError("prefill measurements must each evaluate exactly 2074 tokens")
    if derived.get("completion_tokens") != [256] * count:
        raise BakeoffError("decode measurements must each complete exactly 256 tokens")
    if derived.get("mtp_spec_ran") != [True] * count:
        raise BakeoffError("matching-MTP measurements must prove speculative decode ran")
    accept_rates = derived.get("mtp_accept_rates")
    if (not isinstance(accept_rates, list) or len(accept_rates) != count
            or any(isinstance(rate, bool) or not isinstance(rate, (int, float))
                   or not 0.0 < float(rate) < 1.0 for rate in accept_rates)):
        raise BakeoffError("matching-MTP measurements must prove 0 < accept_rate < 1")
    prefill_values = [finite_number(v, "prefill sample") for v in prefill]
    prefill_peak = max(prefill_values)
    prefill_median = statistics.median(prefill_values)
    decode_median = statistics.median(finite_number(v, "decode sample") for v in decode)
    stage = row.get("stage")
    if stage == "stock":
        quality = 0.0
        quality_result: dict[str, Any] | None = None
    else:
        quality_result = audited_quality(row, corpus_sha)
        if not quality_result["passes"]:
            return {"passes": False, "correctness_quality_pass": False,
                    "quality_score": quality_result.get("quality_score")}
        quality = quality_result["quality_score"]
    companion = derived.get("companion_artifact_bytes")
    required_companions = gates["required_companion_inventory_keys"]
    if not isinstance(companion, dict) or set(companion) != set(required_companions):
        raise BakeoffError(
            "measurement must inventory MTP, vision-mmproj, and vision-vocab companion bytes")
    enabled = row.get("enabled_companions")
    if enabled != required_companions:
        raise BakeoffError(
            "every exact MTP/mmproj/vocab companion must be enabled and accounted")
    for name in required_companions:
        value = companion[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise BakeoffError("companion artifact sizes must be measured positive integers")
    enabled_companion_bytes = sum(companion[name] for name in enabled)
    resources = derived.get("resources") or {}
    host_memtotal = resources.get("runner_memtotal_bytes")
    if host_memtotal != gates["certification_host_memtotal_bytes"]:
        raise BakeoffError("measurement did not use the pinned OtherU MemTotal")
    host_gtt_pages = resources.get("runner_gtt_pages_limit")
    if host_gtt_pages != gates["certification_host_gtt_pages_limit"]:
        raise BakeoffError("measurement did not use the pinned OtherU TTM pages_limit")
    host_gtt_cap = gates["certification_host_gtt_cap_bytes"]
    if host_gtt_cap != host_gtt_pages * 4096:
        raise BakeoffError("pinned OtherU GTT byte cap is inconsistent")
    if resources.get("peak_memory_measurement_method") != gates["peak_memory_measurement_method"]:
        raise BakeoffError("measurement lacks the pinned RSS/GTT sampling method")
    peak_rss = resources.get("measured_peak_rss_bytes")
    peak_gtt = resources.get("measured_peak_gtt_bytes")
    peak_uma = resources.get("measured_peak_uma_bytes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in (peak_rss, peak_gtt, peak_uma)):
        raise BakeoffError("measured peak RSS, GTT, and accounted UMA bytes are required")
    if peak_uma < max(peak_rss, peak_gtt):
        raise BakeoffError("accounted UMA peak cannot be smaller than RSS or GTT peak")
    static_total = artifact_bytes + gates["runtime_reserve_bytes"] + enabled_companion_bytes
    fits = (static_total <= gates["device_budget_bytes"]
            and static_total <= host_memtotal and static_total <= host_gtt_cap
            and peak_uma <= host_memtotal
            and peak_gtt <= host_gtt_cap)
    performance_passes = (
        prefill_peak >= gates["minimum_prefill_peak_tps"]
        and decode_median >= gates["minimum_decode_median_tps"]
    )
    passes = fits and performance_passes
    result = {"passes": passes, "correctness_quality_pass": True,
              "memory_fits": fits, "performance_passes": performance_passes,
              "quality_score": quality,
              "prefill_peak_tps": prefill_peak,
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
              "measured_peak_uma_bytes": peak_uma,
              "quality_evidence": quality_result,
              "measurement_evidence": measured_evidence}
    return result


def candidate_artifact_identity(row: dict[str, Any],
                                metrics: dict[str, Any]) -> dict[str, Any]:
    """Identity of the bytes plus the exact MTP runtime pairing measured."""
    builder, _runtime = validate_revision_identities(row)
    identity = {
        "candidate_id": row.get("candidate_id"),
        "build_record_sha256": row.get("build_record_sha256"),
        "intervention_manifest_sha256": row.get("intervention_manifest_sha256"),
        "profile_sha256": row.get("profile_sha256"),
        "quantization_overrides_sha256": row.get("quantization_overrides_sha256"),
        "model_inventory_sha256": row.get("model_inventory_sha256"),
        "companion_inventory_sha256": row.get("companion_inventory_sha256"),
        "mtp_matrix_quant_contract": row.get("mtp_matrix_quant_contract"),
        "mtp_depth": row.get("mtp_depth"),
        "artifact_bytes": metrics.get("artifact_bytes"),
        "companion_artifact_bytes": (metrics.get("measurement_evidence") or {}).get(
            "derived", {}).get("companion_artifact_bytes"),
        "quantization_arm": row.get("quantization_arm"),
        "intervention_configuration_id": row.get("intervention_configuration_id"),
        "builder_identity": builder,
        "tensor_format_compatibility_sha256": row.get(
            "tensor_format_compatibility_sha256"),
    }
    hex_keys = {
        "build_record_sha256", "intervention_manifest_sha256", "profile_sha256",
        "quantization_overrides_sha256", "model_inventory_sha256",
        "companion_inventory_sha256",
    }
    if (not isinstance(identity["candidate_id"], str) or not identity["candidate_id"]
            or any(not isinstance(identity[key], str)
                   or re.fullmatch(r"[0-9a-f]{64}", identity[key]) is None
                   for key in hex_keys)
            or not isinstance(identity["artifact_bytes"], int)
            or not isinstance(identity["companion_artifact_bytes"], dict)
            or not isinstance(identity["quantization_arm"], str)
            or identity["mtp_matrix_quant_contract"] not in SUPPORTED_MTP_MATRIX_CONTRACTS
            or isinstance(identity["mtp_depth"], bool)
            or not isinstance(identity["mtp_depth"], int)
            or not 1 <= identity["mtp_depth"] <= 4):
        raise BakeoffError("candidate lacks exact artifact/build/inventory provenance")
    return identity


def decision_inputs(row: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    derived = (metrics.get("measurement_evidence") or {}).get("derived") or {}
    quality = metrics.get("quality_evidence")
    return {
        "stage": row.get("stage"),
        "differential_correctness_pass": derived.get("differential_correctness_pass"),
        "audited_quality_pass": True if row.get("stage") == "stock" else
            bool((quality or {}).get("passes")),
        "quality_score": 0.0 if row.get("stage") == "stock" else
            (quality or {}).get("quality_score"),
        "prefill_tps_samples": derived.get("prefill_tps_samples"),
        "decode_tps_samples": derived.get("decode_tps_samples"),
        "evaluated_prefill_tokens": derived.get("evaluated_prefill_tokens"),
        "completion_tokens": derived.get("completion_tokens"),
        "mtp_spec_ran": derived.get("mtp_spec_ran"),
        "mtp_accept_rates": derived.get("mtp_accept_rates"),
        "mtp_matrix_quant_contract": row.get("mtp_matrix_quant_contract"),
        "mtp_depth": row.get("mtp_depth"),
        "artifact_bytes": derived.get("artifact_bytes"),
        "companion_artifact_bytes": derived.get("companion_artifact_bytes"),
        "enabled_companions": row.get("enabled_companions"),
        "resources": derived.get("resources"),
    }


def derive_assessment(inputs: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """Recompute every scalar selection decision from an attested compact row."""
    stage = inputs.get("stage")
    count = gates["minimum_samples_per_performance_gate"]
    prefill = inputs.get("prefill_tps_samples")
    decode = inputs.get("decode_tps_samples")
    accepts = inputs.get("mtp_accept_rates")
    if (stage not in {"stock", "sweep", "format", "mtp-depth", "final"}
            or inputs.get("differential_correctness_pass") is not True
            or inputs.get("audited_quality_pass") is not True
            or not isinstance(prefill, list) or len(prefill) != count
            or not isinstance(decode, list) or len(decode) != count
            or inputs.get("evaluated_prefill_tokens") != [2074] * count
            or inputs.get("completion_tokens") != [256] * count
            or inputs.get("mtp_spec_ran") != [True] * count
            or not isinstance(accepts, list) or len(accepts) != count
            or any(isinstance(rate, bool) or not isinstance(rate, (int, float))
                   or not 0.0 < float(rate) < 1.0 for rate in accepts)):
        raise BakeoffError("assessment decision inputs violate exact correctness/shape/spec gates")
    if (inputs.get("mtp_matrix_quant_contract") not in SUPPORTED_MTP_MATRIX_CONTRACTS
            or isinstance(inputs.get("mtp_depth"), bool)
            or not isinstance(inputs.get("mtp_depth"), int)
            or not 1 <= inputs["mtp_depth"] <= 4):
        raise BakeoffError("assessment decision inputs lack an exact supported MTP pairing")
    prefill_values = [finite_number(value, "attested prefill sample") for value in prefill]
    decode_values = [finite_number(value, "attested decode sample") for value in decode]
    quality = finite_number(inputs.get("quality_score"), "attested quality score")
    artifact_bytes = inputs.get("artifact_bytes")
    companion = inputs.get("companion_artifact_bytes")
    enabled = inputs.get("enabled_companions")
    required = gates["required_companion_inventory_keys"]
    resources = inputs.get("resources")
    if (not isinstance(artifact_bytes, int) or isinstance(artifact_bytes, bool)
            or artifact_bytes < 1 or not isinstance(companion, dict)
            or set(companion) != set(required) or enabled != required
            or any(not isinstance(companion[key], int) or isinstance(companion[key], bool)
                   or companion[key] < 1 for key in required)
            or not isinstance(resources, dict)):
        raise BakeoffError("assessment decision inputs lack exact artifact/companion facts")
    host_memtotal = resources.get("runner_memtotal_bytes")
    host_gtt_pages = resources.get("runner_gtt_pages_limit")
    peak_rss = resources.get("measured_peak_rss_bytes")
    peak_gtt = resources.get("measured_peak_gtt_bytes")
    peak_uma = resources.get("measured_peak_uma_bytes")
    if (host_memtotal != gates["certification_host_memtotal_bytes"]
            or host_gtt_pages != gates["certification_host_gtt_pages_limit"]
            or resources.get("peak_memory_measurement_method") !=
            gates["peak_memory_measurement_method"]
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 1
                   for value in (peak_rss, peak_gtt, peak_uma))
            or peak_uma < max(peak_rss, peak_gtt)):
        raise BakeoffError("assessment decision inputs violate pinned host/memory facts")
    static = artifact_bytes + gates["runtime_reserve_bytes"] + sum(
        companion[key] for key in required)
    gtt_cap = gates["certification_host_gtt_cap_bytes"]
    if gtt_cap != host_gtt_pages * 4096:
        raise BakeoffError("pinned certification GTT cap is inconsistent")
    fits = (static <= gates["device_budget_bytes"] and static <= host_memtotal
            and static <= gtt_cap and peak_uma <= host_memtotal and peak_gtt <= gtt_cap)
    prefill_peak = max(prefill_values)
    decode_median = statistics.median(decode_values)
    performance = (prefill_peak >= gates["minimum_prefill_peak_tps"]
                   and decode_median >= gates["minimum_decode_median_tps"])
    return {
        "passes": fits and performance,
        "correctness_quality_pass": True,
        "memory_fits": fits,
        "performance_passes": performance,
        "quality_score": quality,
        "prefill_peak_tps": prefill_peak,
        "prefill_median_tps": statistics.median(prefill_values),
        "decode_median_tps": decode_median,
        "artifact_bytes": artifact_bytes,
        "enabled_companion_bytes": sum(companion[key] for key in required),
        "static_accounted_bytes": static,
        "runner_memtotal_bytes": host_memtotal,
        "runner_gtt_pages_limit": host_gtt_pages,
        "runner_gtt_cap_bytes": gtt_cap,
        "measured_peak_rss_bytes": peak_rss,
        "measured_peak_gtt_bytes": peak_gtt,
        "measured_peak_uma_bytes": peak_uma,
    }


def make_candidate_assessment(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Validate a live candidate and emit the small record safe to retain.

    This function is run before deleting the model.  Its output is merely
    content-bound by SHA-256; the selector additionally requires a GitHub
    artifact attestation over this exact JSON to authenticate who produced it.
    """
    verified = verify_plan(plan)
    selection = verified.get("selection_plan", verified)
    stage = row.get("stage")
    corpus = (verified.get("corpora", {}).get("final-heldout.jsonl") if stage == "final"
              else selection["corpora"].get("sweep-validation.jsonl"))
    if not isinstance(corpus, dict):
        raise BakeoffError("assessment stage is not authorized by this phase plan")
    _builder, runtime = validate_revision_identities(row)
    metrics = assess(row, selection["recipe"]["value"]["hard_gates"], corpus["sha256"])
    direction = None
    if stage != "stock":
        configurations = {item["id"]: item for item in selection["sweep_configurations"]}
        configuration_id = row.get("configuration_id", row.get("id"))
        configuration = configurations.get(configuration_id)
        if configuration is None:
            raise BakeoffError("assessment candidate is not a planned intervention")
        direction = INTERVENTION_VALIDATOR(row, configuration,
                                           selection["corpora"]["sweep-validation.jsonl"]["sha256"])
        if not isinstance(direction, dict):
            raise BakeoffError("intervention validator did not return a direction identity")
    inputs = decision_inputs(row, metrics)
    recomputed = derive_assessment(inputs, selection["recipe"]["value"]["hard_gates"])
    quality_stock = None
    if stage != "stock":
        quality_stock = (metrics.get("quality_evidence") or {}).get(
            "evidence", {}).get("models", {}).get("stock")
        if not isinstance(quality_stock, dict):
            raise BakeoffError("quality contract lacks its stock-control identity")
    return {
        "schema": ASSESSMENT_SCHEMA,
        "status": "complete",
        "phase_plan_sha256": canonical_sha256(verified),
        "selection_plan_sha256": canonical_sha256(selection),
        "row_id": row.get("id"),
        "stage": stage,
        "configuration_id": row.get("configuration_id", row.get("id")),
        "arm_id": row.get("arm_id"),
        "final_release_eligible": row.get("final_release_eligible"),
        "runtime_mode": row.get("runtime_mode"),
        "mtp_matrix_quant_contract": row.get("mtp_matrix_quant_contract"),
        "mtp_depth": row.get("mtp_depth"),
        "corpus_sha256": corpus["sha256"],
        "measurement_manifest_sha256": row["evidence_manifest"]["sha256"],
        "quality_contract_sha256": None if stage == "stock" else
            row["quality_contract"]["sha256"],
        "artifact_identity": candidate_artifact_identity(row, metrics),
        "runtime_identity": runtime,
        "direction_identity": direction,
        "quality_stock_identity": quality_stock,
        "decision_inputs": inputs,
        "observed_decision": recomputed,
        "artifact_may_be_deleted_after_external_attestation": True,
        "local_sha256_authenticates_authorship": False,
        "external_attestation_required": True,
        "publication_allowed": False,
    }


def verify_external_attestation(subject: Path, bundle: Path, repository: str,
                                signer_workflow: str) -> None:
    """Authenticate a retained assessment/ledger with GitHub's verifier."""
    command = ["gh", "attestation", "verify", str(subject), "--bundle", str(bundle),
               "--repo", repository, "--signer-workflow", signer_workflow]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, check=False)
    except OSError as exc:
        raise BakeoffError(f"cannot run external attestation verifier: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BakeoffError(f"external attestation verification failed: {detail}")


ATTESTATION_VERIFIER = verify_external_attestation


def read_attested_json(descriptor: Any, label: str, schema: str,
                       recipe: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if not isinstance(descriptor, dict) or set(descriptor) != {
            "subject", "bundle", "repository", "signer_workflow"}:
        raise BakeoffError(f"{label} requires an exact subject/bundle/signer descriptor")
    subject = descriptor["subject"]
    if (not isinstance(subject, dict) or set(subject) != {"path", "sha256", "schema"}
            or subject.get("schema") != schema):
        raise BakeoffError(f"{label} subject descriptor is malformed")
    value, digest = pinned_json(
        {"path": subject.get("path"), "sha256": subject.get("sha256")}, label)
    bundle_digest, bundle_path = pinned_evidence_path(
        Path("/"), descriptor["bundle"], f"{label} attestation bundle")
    del bundle_digest
    contract = recipe.get("assessment_attestation") or {}
    if (descriptor.get("repository") != contract.get("repository")
            or descriptor.get("signer_workflow") != contract.get("signer_workflow")
            or contract.get("required") is not True):
        raise BakeoffError(f"{label} attestation signer differs from checked policy")
    ATTESTATION_VERIFIER(Path(subject["path"]).resolve(), bundle_path,
                         descriptor["repository"], descriptor["signer_workflow"])
    if value.get("schema") != schema:
        raise BakeoffError(f"{label} content schema differs from its descriptor")
    return value, digest


def assessment_rows(results: dict[str, Any], stages: set[str], plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    descriptors = results.get("assessments")
    if not isinstance(descriptors, list) or not descriptors:
        raise BakeoffError("phase input must contain externally-attested candidate assessments")
    recipe = plan["recipe"]["value"]
    rows, digests = [], []
    for descriptor in descriptors:
        row, digest = read_attested_json(
            descriptor, "candidate assessment", ASSESSMENT_SCHEMA,
            recipe)
        if row.get("stage") not in stages:
            raise BakeoffError("phase assessments contain a candidate from another stage")
        if row.get("selection_plan_sha256") != canonical_sha256(plan):
            raise BakeoffError("candidate assessment is not bound to this selection plan")
        if (row.get("stage") != "final"
                and row.get("phase_plan_sha256") != canonical_sha256(plan)):
            raise BakeoffError("selection assessment was produced after a different phase plan")
        metrics = derive_assessment(row.get("decision_inputs") or {}, recipe["hard_gates"])
        if row.get("observed_decision") != metrics:
            raise BakeoffError("candidate assessment scalar decisions do not rederive")
        rows.append(row); digests.append(digest)
    return rows, digests


def ledger_base(phase: str, plan: dict[str, Any], rows: list[dict[str, Any]],
                digests: list[str]) -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "phase": phase,
        "status": f"{phase.replace('-', '_')}_selection_complete",
        "plan_sha256": canonical_sha256(plan),
        "assessments": rows,
        "assessment_sha256": digests,
        "external_attestation_verified": True,
        "local_sha256_authenticates_authorship": False,
        "publication_allowed": False,
    }


def read_prior_ledger(descriptor: dict[str, Any], phase: str,
                      plan: dict[str, Any]) -> tuple[dict[str, Any], str]:
    ledger, digest = read_attested_json(
        descriptor, f"{phase} prior ledger", LEDGER_SCHEMA,
        plan["recipe"]["value"])
    if (ledger.get("phase") != phase or ledger.get("plan_sha256") != canonical_sha256(plan)
            or ledger.get("publication_allowed") is not False
            or ledger.get("external_attestation_verified") is not True):
        raise BakeoffError(f"{phase} prior ledger is not bound to this canonical plan")
    verify_ledger_semantics(plan, ledger)
    return ledger, digest


def _metrics(row: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    metrics = derive_assessment(row.get("decision_inputs") or {},
                                plan["recipe"]["value"]["hard_gates"])
    if row.get("observed_decision") != metrics:
        raise BakeoffError("ledger assessment scalar decisions do not reproduce")
    return metrics


def _same_direction(rows: list[dict[str, Any]], expected: dict[str, Any] | None = None) -> dict[str, Any]:
    directions = [row.get("direction_identity") for row in rows]
    if not directions or any(not isinstance(item, dict) for item in directions):
        raise BakeoffError("every intervention assessment must bind its direction identity")
    first = directions[0]
    if any(item != first for item in directions[1:]) or (expected is not None and first != expected):
        raise BakeoffError("intervention candidates do not share one common direction identity")
    return first


def _winner_key(metrics: dict[str, Any], identifier: str) -> tuple[float, float, float, str]:
    """Rank candidates that have already passed every hard gate.

    The checked recipe/profile pin this exact order. Keep the implementation
    explicit so an unknown policy token cannot silently change a release
    decision.
    """
    if not isinstance(identifier, str) or not identifier:
        raise BakeoffError("winner candidate lacks a stable non-empty id")
    return (
        -finite_number(metrics.get("decode_median_tps"), "winner decode median"),
        -finite_number(metrics.get("prefill_median_tps"), "winner prefill median"),
        -finite_number(metrics.get("quality_score"), "winner quality score"),
        identifier,
    )


def rolling_retention_transition(
    plan: dict[str, Any], rows: list[dict[str, Any]], phase: str,
) -> dict[str, Any]:
    """Derive the one-time artifact eviction after appending one assessment.

    A serial accumulator retains only the candidates that can still affect its
    eventual selector.  Sweep keeps its stock control plus current top one,
    while the format screen keeps the exact top two needed by balanced
    confirmation.
    The transition is intentionally derived from the previous prefix survivors
    plus the newly appended row: candidates retired by an earlier prefix never
    reappear in a later deletion authorization.
    """
    verified = verify_plan(plan)
    if phase == "sweep":
        expected = [verified["stock_control"]["id"], *[
            item["id"] for item in verified["sweep_configurations"]]]
        retain_count = 1
        policy = "rolling_stock_plus_exact_winner_top1"
    elif phase == "format":
        expected = [item["id"] for item in verified["format_arms"]]
        retain_count = 2
        policy = "rolling_exact_finalists_top2"
    else:
        raise BakeoffError("rolling artifact retirement applies only to sweep or format")
    if not rows or len(rows) > len(expected):
        raise BakeoffError("rolling retention requires one nonempty canonical phase prefix")
    if [row.get("row_id") for row in rows] != expected[:len(rows)]:
        raise BakeoffError("rolling retention assessments are not the canonical phase prefix")
    plan_sha256 = canonical_sha256(verified)
    configurations = {
        item["id"]: item for item in verified["sweep_configurations"]
    }
    runtime_identity = None
    direction_identity = None
    stock_identity = None
    format_configuration = None
    candidate_ids = []
    for row in rows:
        artifact = row.get("artifact_identity")
        if not isinstance(artifact, dict):
            raise BakeoffError("rolling retention assessment lacks its artifact identity")
        candidate_id = artifact.get("candidate_id")
        if (not isinstance(candidate_id, str) or not candidate_id
                or row.get("selection_plan_sha256") != plan_sha256
                or row.get("phase_plan_sha256") != plan_sha256):
            raise BakeoffError("rolling retention assessment lacks its plan/candidate identity")
        if phase == "sweep" and row.get("stage") == "stock":
            stock = verified["stock_control"]
            if (row.get("row_id") != stock["id"]
                    or row.get("final_release_eligible") is not False
                    or not isinstance(row.get("runtime_identity"), dict)):
                raise BakeoffError("rolling stock assessment differs from the canonical plan")
            stock_identity = artifact
            runtime_identity = row.get("runtime_identity")
        elif phase == "sweep":
            configuration = next(
                item for item in verified["sweep_configurations"]
                if item["id"] == row.get("row_id"))
            if (row.get("stage") != "sweep"
                    or row.get("configuration_id") != configuration["id"]
                    or artifact.get("intervention_configuration_id") != configuration["id"]
                    or artifact.get("quantization_arm") != configuration["quantization_arm"]
                    or artifact.get("quantization_overrides_sha256") !=
                    configuration["quantization_overrides_sha256"]
                    or artifact.get("profile_sha256") != configuration["profile_sha256"]
                    or row.get("runtime_mode") != configuration["runtime_mode"]
                    or row.get("final_release_eligible") !=
                    configuration["final_release_eligible"]
                    or row.get("runtime_identity") != runtime_identity
                    or row.get("quality_stock_identity") != stock_identity
                    or not isinstance(row.get("direction_identity"), dict)):
                raise BakeoffError("rolling sweep assessment provenance differs from its plan")
            if direction_identity is None:
                direction_identity = row.get("direction_identity")
            elif row.get("direction_identity") != direction_identity:
                raise BakeoffError("rolling sweep assessments use different directions")
        else:
            arm = next(item for item in verified["format_arms"]
                       if item["id"] == row.get("row_id"))
            configuration_id = row.get("configuration_id")
            if format_configuration is None:
                format_configuration = configuration_id
                runtime_identity = row.get("runtime_identity")
                direction_identity = row.get("direction_identity")
                stock_identity = row.get("quality_stock_identity")
            if (row.get("stage") != "format" or row.get("arm_id") != arm["id"]
                    or not isinstance(configuration_id, str) or not configuration_id
                    or configuration_id not in configurations
                    or configuration_id != format_configuration
                    or artifact.get("intervention_configuration_id") != configuration_id
                    or artifact.get("quantization_arm") != arm["quantization_arm"]
                    or artifact.get("quantization_overrides_sha256") !=
                    arm["quantization_overrides_sha256"]
                    or artifact.get("profile_sha256") != arm["profile_sha256"]
                    or row.get("mtp_matrix_quant_contract") != arm["mtp_matrix_quant_contract"]
                    or row.get("mtp_depth") != arm["mtp_depth"]
                    or row.get("final_release_eligible") != arm["final_release_eligible"]
                    or row.get("runtime_identity") != runtime_identity
                    or row.get("direction_identity") != direction_identity
                    or row.get("quality_stock_identity") != stock_identity
                    or not isinstance(runtime_identity, dict)
                    or not isinstance(direction_identity, dict)
                    or not isinstance(stock_identity, dict)):
                raise BakeoffError("rolling format assessment provenance differs from its plan")
        candidate_ids.append(candidate_id)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise BakeoffError("rolling retention candidate identities are not unique")

    def survivors(prefix: list[dict[str, Any]]) -> list[str]:
        stock = []
        ranked = []
        for row in prefix:
            if phase == "sweep" and row.get("stage") == "stock":
                stock.append((row.get("artifact_identity") or {})["candidate_id"])
                continue
            if row.get("stage") != phase:
                continue
            metrics = _metrics(row, verified)
            if not metrics["passes"]:
                continue
            if phase == "format" and row.get("final_release_eligible") is not True:
                continue
            identifier = row.get("row_id") if phase == "sweep" else row.get("arm_id")
            artifact = row.get("artifact_identity") or {}
            ranked.append((row, metrics, identifier, artifact["candidate_id"]))
        ranked.sort(key=lambda item: _winner_key(item[1], item[2]))
        return [*stock, *[item[3] for item in ranked[:retain_count]]]

    previous = survivors(rows[:-1])
    retained = survivors(rows)
    live_before_decision = [*previous, candidate_ids[-1]]
    retire = []
    for candidate_id in live_before_decision:
        if candidate_id not in retained and candidate_id not in retire:
            retire.append(candidate_id)
    return {
        "phase": phase,
        "selection_policy": policy,
        "retained_candidate_ids": retained,
        "retire_candidate_ids": retire,
    }


def sealed_format_retention(
    plan: dict[str, Any], ledger: dict[str, Any],
) -> dict[str, Any]:
    """Collapse the two live format finalists to the attested ledger winner."""
    verified = verify_plan(plan)
    if ledger.get("phase") != "format":
        raise BakeoffError("sealed format retention requires the completed format ledger")
    verify_ledger_semantics(verified, ledger)
    rows = ledger.get("assessments")
    if not isinstance(rows, list):
        raise BakeoffError("sealed format ledger has no compact assessments")
    rolling = rolling_retention_transition(verified, rows, "format")
    selected = ledger.get("selected_artifact_identity") or {}
    selected_id = selected.get("candidate_id")
    if (not isinstance(selected_id, str)
            or selected_id not in rolling["retained_candidate_ids"]):
        raise BakeoffError("sealed format winner was not one of the live finalists")
    return {
        "phase": "format",
        "selection_policy": "sealed_balanced_format_winner_top1",
        "retained_candidate_ids": [selected_id],
        "retire_candidate_ids": [candidate_id for candidate_id in
                                 rolling["retained_candidate_ids"]
                                 if candidate_id != selected_id],
    }


def balanced_confirmation_plan(
    plan: dict[str, Any], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Freeze the top-two format order before balanced measurements begin.

    The format-arm sweep remains the screening stage. Only its top two
    gate-passing arms enter confirmation. The counterbalanced A/B, B/A, A/B
    pair order and three sealed workload recipes are persisted before any
    confirmation result exists.
    """
    assessed = [(row, _metrics(row, plan)) for row in rows]
    passing = [(row, metrics) for row, metrics in assessed
               if metrics["passes"] and row.get("final_release_eligible") is True]
    if len(passing) < 2:
        raise BakeoffError("balanced confirmation requires two gate-passing format finalists")
    finalists = sorted(
        passing, key=lambda pair: _winner_key(pair[1], pair[0].get("arm_id"))
    )[:2]
    identities = []
    for row, metrics in finalists:
        arm_id = row.get("arm_id")
        artifact = row.get("artifact_identity")
        if not isinstance(arm_id, str) or not isinstance(artifact, dict):
            raise BakeoffError("format finalist lacks its stable arm/artifact identity")
        identities.append({
            "arm_id": arm_id,
            "candidate_id": artifact.get("candidate_id"),
            "candidate_kernel_capability": candidate_kernel_capability(
                artifact.get("quantization_arm")),
            "artifact_identity": artifact,
            "first_stage_metrics": metrics,
        })
    arm_a, arm_b = (item["arm_id"] for item in identities)
    order = [arm_a, arm_b, arm_b, arm_a, arm_a, arm_b]
    workloads = []
    for index in range(BALANCED_CONFIRMATION_SAMPLES):
        workload_id = f"counterbalanced-pair-{index}"
        marker = f"QBC{index}"
        recipe = {
            "workload_id": workload_id,
            "marker": marker,
            "prefill_generator": "benchmark.make_prefill_prompt.v1",
            "decode_generator": "benchmark.make_decode_prompt.v1",
            "evaluated_prefill_tokens": 2074,
            "completion_tokens": 256,
        }
        workloads.append({**recipe, "recipe_sha256": canonical_sha256(recipe)})
    workload_order = [workloads[index // 2]["workload_id"] for index in range(6)]
    value = {
        "schema": BALANCED_CONFIRMATION_SCHEMA,
        "selection_plan_sha256": canonical_sha256(plan),
        "first_stage": "complete_format_arm_sweep",
        "finalists": identities,
        "run_order": order,
        "workload_order": workload_order,
        "counterbalance_design": {
            "sequence": "ABBAAB",
            "adjacent_pair_orientations": ["A_then_B", "B_then_A", "A_then_B"],
            "signed_pair_metric": "decode_tps_A_minus_B",
            "required_consistency": "all_three_signed_pair_differences_have_same_nonzero_sign",
        },
        "workloads": workloads,
        "fresh_process_per_run": True,
        "native_mtp_required_per_decode": True,
        "workload": {
            "evaluated_prefill_tokens": 2074,
            "completion_tokens": 256,
            "samples_per_arm": BALANCED_CONFIRMATION_SAMPLES,
        },
        "noise_rule": {
            "metric": "decode_median_tps",
            "observed_noise": "max_within_arm_decode_range_tps",
            "predeclared_practical_effect_floor_tps": BALANCED_MIN_DECODE_EFFECT_TPS,
            "required_separation": "absolute_median_difference_strictly_greater_than_max_observed_noise_and_practical_floor",
            "sequence_consistency": "same_nonzero_A_minus_B_sign_in_all_adjacent_pairs",
            "indistinguishable_action": "reject_selection",
        },
    }
    value["ordering_sha256"] = canonical_sha256({
        "finalists": [item["arm_id"] for item in identities],
        "run_order": order,
        "workload_order": workload_order,
        "counterbalance_design": value["counterbalance_design"],
        "workloads": workloads,
        "workload": value["workload"],
    })
    return value


def adjudicate_balanced_confirmation(
    plan: dict[str, Any], rows: list[dict[str, Any]], evidence: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate fresh-process finalist evidence and return winner + metrics."""
    expected = balanced_confirmation_plan(plan, rows)
    if not isinstance(evidence, dict):
        raise BakeoffError("format selection requires balanced finalist confirmation")
    if set(evidence) != {"schema", "confirmation_plan", "runs"}:
        raise BakeoffError("balanced confirmation evidence keys differ from the contract")
    if evidence.get("schema") != BALANCED_CONFIRMATION_SCHEMA:
        raise BakeoffError("balanced confirmation evidence schema differs")
    if evidence.get("confirmation_plan") != expected:
        raise BakeoffError("balanced confirmation used a different finalist order or workload")
    runs = evidence.get("runs")
    if not isinstance(runs, list) or len(runs) != BALANCED_CONFIRMATION_SAMPLES * 2:
        raise BakeoffError("balanced confirmation requires exactly six counterbalanced runs")
    process_ids: set[str] = set()
    container_ids: set[str] = set()
    host_process_ids: set[tuple[int, int]] = set()
    process_by_arm: dict[str, dict[str, Any]] = {}
    workload_runs: dict[str, list[dict[str, Any]]] = {
        item["workload_id"]: [] for item in expected["workloads"]
    }
    eligible_kernel_modes: set[str] = set()
    samples: dict[str, dict[str, list[float]]] = {
        item["arm_id"]: {"prefill": [], "decode": []}
        for item in expected["finalists"]
    }
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or set(run) != {
            "run_index", "arm_id", "process_instance", "process_instance_sha256",
            "workload_id", "workload_recipe_sha256", "prefill_prompt_sha256",
            "decode_prompt_sha256", "calibrated_prefill_words",
            "evaluated_prefill_tokens", "completion_tokens",
            "prefill_tps", "decode_tps", "spec_ran", "accept_rate",
            "startup_kernel_mode", "startup_log_sha256",
        }:
            raise BakeoffError("balanced confirmation run keys differ from the contract")
        arm_id = run.get("arm_id")
        process_id = run.get("process_instance_sha256")
        if (run.get("run_index") != index
                or arm_id != expected["run_order"][index]):
            raise BakeoffError("balanced confirmation did not follow its persisted counterbalanced order")
        if arm_id not in samples:
            raise BakeoffError("balanced confirmation contains an unknown finalist")
        process = run.get("process_instance")
        finalist = next(item for item in expected["finalists"] if item["arm_id"] == arm_id)
        artifact = finalist["artifact_identity"]
        capability = finalist["candidate_kernel_capability"]
        w4a8_required = capability != "no_eligible_rocmi4_mmq"
        row = next(item for item in rows if item.get("arm_id") == arm_id)
        runtime = row.get("runtime_identity") or {}
        if (not isinstance(process, dict) or set(process) != {
                "schema", "run_index", "arm_id", "candidate_id", "container_id",
                "host_pid", "proc_start_ticks", "ember_revision", "container_digest",
                "engine_binary_sha256", "tensor_format_contract_sha256",
                "candidate_kernel_capability", "rocmi4_w4a8_iu4_requested",
                "candidate_binding_sha256", "model_first_shard_sha256",
                "model_inventory_sha256", "companion_inventory_sha256",
                "mtp_sha256", "mtp_depth"}):
            raise BakeoffError("fresh process identity descriptor keys differ")
        stable = {key: process.get(key) for key in (
            "candidate_id", "container_digest", "engine_binary_sha256",
            "tensor_format_contract_sha256", "candidate_binding_sha256",
            "model_first_shard_sha256", "model_inventory_sha256",
            "companion_inventory_sha256", "mtp_sha256", "mtp_depth")}
        prior_stable = process_by_arm.setdefault(arm_id, stable)
        if prior_stable != stable:
            raise BakeoffError("one finalist changed its model/MTP/runtime binding between slots")
        if (process.get("schema") != BALANCED_PROCESS_SCHEMA
                or process.get("run_index") != index or process.get("arm_id") != arm_id
                or process.get("candidate_id") != finalist["candidate_id"]
                or not isinstance(process.get("container_id"), str)
                or re.fullmatch(r"[0-9a-f]{64}", process["container_id"]) is None
                or isinstance(process.get("host_pid"), bool)
                or not isinstance(process.get("host_pid"), int) or process["host_pid"] <= 1
                or isinstance(process.get("proc_start_ticks"), bool)
                or not isinstance(process.get("proc_start_ticks"), int)
                or process["proc_start_ticks"] <= 0
                or process.get("ember_revision") != runtime.get("ember_revision")
                or process.get("container_digest") != runtime.get("container_digest")
                or process.get("engine_binary_sha256") != runtime.get("engine_binary_sha256")
                or process.get("tensor_format_contract_sha256") !=
                   runtime.get("tensor_format_contract_sha256")
                or process.get("candidate_kernel_capability") != capability
                or process.get("rocmi4_w4a8_iu4_requested") is not w4a8_required
                or process.get("model_inventory_sha256") != artifact.get("model_inventory_sha256")
                or process.get("companion_inventory_sha256") !=
                   artifact.get("companion_inventory_sha256")
                or process.get("mtp_depth") != artifact.get("mtp_depth")
                or any(re.fullmatch(r"[0-9a-f]{64}", str(process.get(key, ""))) is None
                       for key in ("candidate_binding_sha256", "model_first_shard_sha256",
                                   "mtp_sha256"))):
            raise BakeoffError("fresh process identity is not bound to its candidate/image/binary/MTP")
        if (not isinstance(process_id, str)
                or re.fullmatch(r"[0-9a-f]{64}", process_id) is None
                or process_id != canonical_sha256(process)
                or process_id in process_ids):
            raise BakeoffError("every confirmation sample must use one unique fresh process")
        host_process_id = (process["host_pid"], process["proc_start_ticks"])
        if (process["container_id"] in container_ids or host_process_id in host_process_ids):
            raise BakeoffError("every confirmation sample must use one unique fresh process")
        process_ids.add(process_id)
        container_ids.add(process["container_id"])
        host_process_ids.add(host_process_id)
        if (run.get("evaluated_prefill_tokens") != 2074
                or run.get("completion_tokens") != 256):
            raise BakeoffError("confirmation workload must be exactly 2074 prefill and 256 decode tokens")
        accept_rate = finite_number(run.get("accept_rate"), "confirmation MTP accept rate")
        if run.get("spec_ran") is not True or not 0.0 < accept_rate < 1.0:
            raise BakeoffError("every confirmation decode must run native MTP with 0 < accept_rate < 1")
        workload_id = run.get("workload_id")
        workload = next((item for item in expected["workloads"]
                         if item["workload_id"] == workload_id), None)
        if (workload is None or workload_id != expected["workload_order"][index]
                or run.get("workload_recipe_sha256") != workload["recipe_sha256"]
                or re.fullmatch(r"[0-9a-f]{64}", str(run.get("prefill_prompt_sha256", ""))) is None
                or re.fullmatch(r"[0-9a-f]{64}", str(run.get("decode_prompt_sha256", ""))) is None
                or isinstance(run.get("calibrated_prefill_words"), bool)
                or not isinstance(run.get("calibrated_prefill_words"), int)
                or run["calibrated_prefill_words"] < 1):
            raise BakeoffError("confirmation run is not bound to its sealed paired workload")
        workload_runs[workload_id].append(run)
        kernel_mode = run.get("startup_kernel_mode")
        eligible_modes = {"w4a8_iu4_register_pack", "w4a8_iu4_prepack"}
        if ((w4a8_required and kernel_mode not in eligible_modes)
                or (not w4a8_required and kernel_mode !=
                    "not_applicable_no_eligible_rocmi4_mmq")
                or re.fullmatch(r"[0-9a-f]{64}", str(run.get("startup_log_sha256", "")))
                   is None):
            raise BakeoffError(
                "confirmation timing kernel mode differs from candidate capability")
        if w4a8_required:
            eligible_kernel_modes.add(kernel_mode)
        samples[arm_id]["prefill"].append(
            finite_number(run.get("prefill_tps"), "confirmation prefill sample"))
        samples[arm_id]["decode"].append(
            finite_number(run.get("decode_tps"), "confirmation decode sample"))
    for workload_id, paired in workload_runs.items():
        if (len(paired) != 2 or {item["arm_id"] for item in paired} != set(samples)
                or paired[0]["prefill_prompt_sha256"] != paired[1]["prefill_prompt_sha256"]
                or paired[0]["decode_prompt_sha256"] != paired[1]["decode_prompt_sha256"]
                or paired[0]["calibrated_prefill_words"] !=
                   paired[1]["calibrated_prefill_words"]):
            raise BakeoffError(
                f"paired workload {workload_id} did not use identical prompts once per arm")
    if len(eligible_kernel_modes) > 1:
        raise BakeoffError("confirmation slots changed W4A8 startup kernel mode")
    gates = plan["recipe"]["value"]["hard_gates"]
    metrics: dict[str, dict[str, Any]] = {}
    for arm_id, values in samples.items():
        if (len(values["prefill"]) != BALANCED_CONFIRMATION_SAMPLES
                or len(values["decode"]) != BALANCED_CONFIRMATION_SAMPLES):
            raise BakeoffError("confirmation must contain exactly three samples per arm")
        prefill_peak = max(values["prefill"])
        decode_median = statistics.median(values["decode"])
        metrics[arm_id] = {
            "prefill_tps_samples": values["prefill"],
            "decode_tps_samples": values["decode"],
            "prefill_peak_tps": prefill_peak,
            "decode_median_tps": decode_median,
            "decode_observed_range_tps": max(values["decode"]) - min(values["decode"]),
            "hard_gate_passes": (
                prefill_peak >= gates["minimum_prefill_peak_tps"]
                and decode_median >= gates["minimum_decode_median_tps"]
            ),
        }
    if not all(item["hard_gate_passes"] for item in metrics.values()):
        raise BakeoffError("a balanced-confirmation finalist failed the unchanged hard gates")
    arm_a, arm_b = [item["arm_id"] for item in expected["finalists"]]
    median_delta = abs(
        metrics[arm_a]["decode_median_tps"] - metrics[arm_b]["decode_median_tps"])
    observed_noise = max(
        metrics[arm_a]["decode_observed_range_tps"],
        metrics[arm_b]["decode_observed_range_tps"],
    )
    required_separation = max(observed_noise, BALANCED_MIN_DECODE_EFFECT_TPS)
    if median_delta <= required_separation:
        raise BakeoffError(
            "balanced finalist decode difference is within observed run noise or the "
            "predeclared practical-effect floor; refusing selection")
    pair_deltas = []
    pair_details = []
    for pair_index in range(BALANCED_CONFIRMATION_SAMPLES):
        pair = runs[pair_index * 2:pair_index * 2 + 2]
        values = {item["arm_id"]: finite_number(
            item["decode_tps"], "paired confirmation decode sample") for item in pair}
        delta = values[arm_a] - values[arm_b]
        pair_deltas.append(delta)
        pair_details.append({
            "workload_id": pair[0]["workload_id"],
            "run_indices": [item["run_index"] for item in pair],
            "run_order": [item["arm_id"] for item in pair],
            "decode_tps_A_minus_B": delta,
        })
    signs = {1 if value > 0 else -1 if value < 0 else 0 for value in pair_deltas}
    median_sign = (1 if metrics[arm_a]["decode_median_tps"] >
                   metrics[arm_b]["decode_median_tps"] else -1)
    if signs != {median_sign}:
        raise BakeoffError(
            "counterbalanced adjacent-pair results are sequence/order inconsistent")
    winner_id = max(
        (arm_a, arm_b), key=lambda arm_id: metrics[arm_id]["decode_median_tps"])
    winner = next(row for row in rows if row.get("arm_id") == winner_id)
    confirmation = {
        "schema": BALANCED_CONFIRMATION_SCHEMA,
        "confirmation_plan": expected,
        "runs": runs,
        "metrics_by_arm": metrics,
        "decode_median_difference_tps": median_delta,
        "observed_run_noise_tps": observed_noise,
        "predeclared_practical_effect_floor_tps": BALANCED_MIN_DECODE_EFFECT_TPS,
        "required_decode_separation_tps": required_separation,
        "counterbalanced_pair_results": pair_details,
        "signed_pair_decode_differences_A_minus_B": pair_deltas,
        "sequence_consistency_pass": True,
        "selected_arm_id": winner_id,
    }
    return winner, confirmation


def verify_ledger_semantics(plan: dict[str, Any], ledger: dict[str, Any]) -> None:
    phase = ledger.get("phase")
    rows = ledger.get("assessments")
    digests = ledger.get("assessment_sha256")
    if (not isinstance(rows, list) or not isinstance(digests, list)
            or len(rows) != len(digests)):
        raise BakeoffError("prior ledger lacks compact attested assessment rows")
    if phase == "sweep":
        expected = select_sweep_from_assessments(plan, rows, digests)
    elif phase == "format":
        prior = ledger.get("prior_sweep_ledger")
        if not isinstance(prior, dict):
            raise BakeoffError("format ledger does not embed its compact sweep ledger")
        verify_ledger_semantics(plan, prior)
        retained = ledger.get("balanced_confirmation")
        confirmation = ({key: retained.get(key) for key in (
            "schema", "confirmation_plan", "runs")}
            if isinstance(retained, dict) else retained)
        expected = select_format_from_assessments(
            plan, rows, digests, prior, confirmation)
    elif phase == "mtp-depth":
        prior = ledger.get("prior_format_ledger")
        if not isinstance(prior, dict):
            raise BakeoffError("MTP-depth ledger does not embed its compact format ledger")
        verify_ledger_semantics(plan, prior)
        expected = select_mtp_depth_from_assessments(plan, rows, digests, prior)
    else:
        raise BakeoffError("only completed sweep/format/MTP-depth ledgers may select a later phase")
    if ledger != expected:
        raise BakeoffError("prior ledger semantics do not reproduce from compact assessments")


def select_sweep_from_assessments(plan: dict[str, Any], rows: list[dict[str, Any]],
                                  digests: list[str]) -> dict[str, Any]:
    recipe = plan["recipe"]["value"]
    stock_rows = [row for row in rows if row.get("stage") == "stock"]
    sweep_rows = [row for row in rows if row.get("stage") == "sweep"]
    if (len(stock_rows) != 1 or stock_rows[0].get("row_id") != recipe["stock_control"]["id"]
            or stock_rows[0].get("final_release_eligible") is not False):
        raise BakeoffError("sweep requires exactly the final-ineligible stock control assessment")
    stock_metrics = _metrics(stock_rows[0], plan)
    if (not stock_metrics["correctness_quality_pass"] or not stock_metrics["memory_fits"]):
        raise BakeoffError("stock control failed correctness/quality or memory")
    stock_identity = stock_rows[0].get("artifact_identity")
    runtime_identity = stock_rows[0].get("runtime_identity")
    if (not isinstance(runtime_identity, dict)
            or any(row.get("runtime_identity") != runtime_identity for row in sweep_rows)):
        raise BakeoffError("comparative sweep assessments did not use one exact runtime engine")
    configurations = {item["id"]: item for item in plan["sweep_configurations"]}
    if ({row.get("row_id") for row in sweep_rows} != set(configurations)
            or len(sweep_rows) != len(configurations)):
        raise BakeoffError("sweep requires every planned configuration exactly once")
    common_direction = _same_direction(sweep_rows)
    assessed = []
    for row in sweep_rows:
        configuration = configurations[row["row_id"]]
        artifact = row.get("artifact_identity") or {}
        if (row.get("configuration_id") != configuration["id"]
                or artifact.get("intervention_configuration_id") != configuration["id"]
                or artifact.get("quantization_arm") != configuration["quantization_arm"]
                or artifact.get("quantization_overrides_sha256") != configuration["quantization_overrides_sha256"]
                or artifact.get("profile_sha256") != configuration["profile_sha256"]
                or row.get("runtime_mode") != configuration["runtime_mode"]
                or row.get("final_release_eligible") != configuration["final_release_eligible"]):
            raise BakeoffError("sweep assessment provenance differs from its canonical configuration")
        if row.get("quality_stock_identity") != stock_identity:
            raise BakeoffError("candidate quality evidence used a different stock control")
        assessed.append((row, _metrics(row, plan)))
    passing = [(row, metrics) for row, metrics in assessed if metrics["passes"]]
    if not passing:
        raise BakeoffError("no intervention sweep configuration passed all gates")
    winner, metrics = sorted(
        passing, key=lambda pair: _winner_key(pair[1], pair[0]["row_id"]))[0]
    return {**ledger_base("sweep", plan, rows, digests),
            "stock_identity": stock_identity,
            "stock_metrics": stock_metrics,
            "runtime_identity": runtime_identity,
            "direction_identity": common_direction,
            "selected_configuration_id": winner["row_id"],
            "selected_configuration": configurations[winner["row_id"]],
            "selected_artifact_identity": winner["artifact_identity"],
            "selected_metrics": metrics}


def select_sweep(plan: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    plan = verify_plan(plan)
    if plan["phase_scope"] != "selection":
        raise BakeoffError("sweep selection requires the final-blind selection plan")
    rows, digests = assessment_rows(results, {"stock", "sweep"}, plan)
    return select_sweep_from_assessments(plan, rows, digests)


def select_format_from_assessments(plan: dict[str, Any], rows: list[dict[str, Any]],
                                   digests: list[str], prior: dict[str, Any],
                                   confirmation: Any) -> dict[str, Any]:
    arms = {item["id"]: item for item in plan["format_arms"]}
    if ({row.get("arm_id") for row in rows} != set(arms) or len(rows) != len(arms)):
        raise BakeoffError("format requires every planned arm exactly once")
    configurations = {item["id"]: item for item in plan["sweep_configurations"]}
    selected_id = prior.get("selected_configuration_id")
    if selected_id not in configurations or prior.get("selected_configuration") != configurations[selected_id]:
        raise BakeoffError("sweep ledger selected configuration semantics are inconsistent")
    common_direction = _same_direction(rows, prior.get("direction_identity"))
    if any(row.get("runtime_identity") != prior.get("runtime_identity") for row in rows):
        raise BakeoffError("format assessments did not use the sweep-selected runtime engine")
    assessed = []
    for row in rows:
        arm = arms[row["arm_id"]]
        artifact = row.get("artifact_identity") or {}
        if (row.get("configuration_id") != selected_id
                or artifact.get("intervention_configuration_id") != selected_id
                or artifact.get("quantization_arm") != arm["quantization_arm"]
                or artifact.get("quantization_overrides_sha256") != arm["quantization_overrides_sha256"]
                or artifact.get("profile_sha256") != arm["profile_sha256"]
                or row.get("mtp_matrix_quant_contract") != arm["mtp_matrix_quant_contract"]
                or row.get("mtp_depth") != arm["mtp_depth"]
                or row.get("final_release_eligible") != arm["final_release_eligible"]
                or row.get("quality_stock_identity") != prior["stock_identity"]):
            raise BakeoffError("format assessment provenance differs from the selected intervention/arm")
        assessed.append((row, arm, _metrics(row, plan)))
    eligible = [(row, metrics) for row, arm, metrics in assessed
                if arm["final_release_eligible"] and metrics["passes"]]
    if not eligible:
        raise BakeoffError("no final-eligible format passed all measured gates")
    eligible_rows = [row for row, _metrics_value in eligible]
    winner, balanced = adjudicate_balanced_confirmation(
        plan, eligible_rows, confirmation)
    first_stage_metrics = next(
        metrics for row, metrics in eligible if row is winner)
    metrics = {
        **first_stage_metrics,
        "prefill_peak_tps": balanced["metrics_by_arm"][winner["arm_id"]][
            "prefill_peak_tps"],
        "prefill_median_tps": statistics.median(
            balanced["metrics_by_arm"][winner["arm_id"]]["prefill_tps_samples"]),
        "decode_median_tps": balanced["metrics_by_arm"][winner["arm_id"]][
            "decode_median_tps"],
        "balanced_confirmation_pass": True,
    }
    return {**ledger_base("format", plan, rows, digests),
            "prior_sweep_ledger": prior,
            "prior_sweep_ledger_sha256": canonical_sha256(prior),
            "stock_identity": prior["stock_identity"],
            "direction_identity": common_direction,
            "runtime_identity": prior["runtime_identity"],
            "selected_configuration_id": selected_id,
            "selected_configuration": configurations[selected_id],
            "balanced_confirmation": balanced,
            "selected_arm_id": winner["arm_id"],
            "selected_arm": arms[winner["arm_id"]],
            "selected_artifact_identity": winner["artifact_identity"],
            "selected_metrics": metrics}


def select_format(plan: dict[str, Any], results: dict[str, Any],
                  sweep_descriptor: dict[str, Any]) -> dict[str, Any]:
    plan = verify_plan(plan)
    prior, _ = read_prior_ledger(sweep_descriptor, "sweep", plan)
    rows, digests = assessment_rows(results, {"format"}, plan)
    return select_format_from_assessments(
        plan, rows, digests, prior, results.get("balanced_confirmation"))


def _artifact_without_depth(identity: dict[str, Any]) -> dict[str, Any]:
    """Return reusable byte identity, excluding measurement label and runtime depth."""
    return {key: value for key, value in identity.items()
            if key not in {"candidate_id", "mtp_depth"}}


def select_mtp_depth_from_assessments(
    plan: dict[str, Any], rows: list[dict[str, Any]], digests: list[str],
    prior: dict[str, Any],
) -> dict[str, Any]:
    configurations = {item["id"]: item for item in plan["mtp_depth_configurations"]}
    if ({row.get("row_id") for row in rows} != set(configurations)
            or len(rows) != len(configurations)):
        raise BakeoffError("MTP-depth selection requires depths 1 through 4 exactly once")
    selected_arm = prior.get("selected_arm")
    selected_artifact = prior.get("selected_artifact_identity")
    if not isinstance(selected_arm, dict) or not isinstance(selected_artifact, dict):
        raise BakeoffError("format ledger lacks its selected exact-runtime pair")
    common_direction = _same_direction(rows, prior.get("direction_identity"))
    if any(row.get("runtime_identity") != prior.get("runtime_identity") for row in rows):
        raise BakeoffError("MTP-depth assessments changed the format-selected runtime engine")
    assessed = []
    for row in rows:
        configuration = configurations[row["row_id"]]
        artifact = row.get("artifact_identity") or {}
        if (row.get("stage") != "mtp-depth"
                or row.get("configuration_id") != prior.get("selected_configuration_id")
                or row.get("arm_id") != prior.get("selected_arm_id")
                or row.get("runtime_mode") != configuration["runtime_mode"]
                or row.get("final_release_eligible") is not
                configuration["final_release_eligible"]
                or row.get("mtp_matrix_quant_contract") !=
                selected_arm.get("mtp_matrix_quant_contract")
                or row.get("mtp_depth") != configuration["mtp_depth"]
                or artifact.get("mtp_matrix_quant_contract") !=
                selected_arm.get("mtp_matrix_quant_contract")
                or artifact.get("mtp_depth") != configuration["mtp_depth"]
                or _artifact_without_depth(artifact) !=
                _artifact_without_depth(selected_artifact)
                or row.get("quality_stock_identity") != prior.get("stock_identity")):
            raise BakeoffError(
                "MTP-depth assessment changed the selected main/companion artifacts or pairing")
        assessed.append((row, configuration, _metrics(row, plan)))
    eligible = [(row, configuration, metrics)
                for row, configuration, metrics in assessed
                if configuration["final_release_eligible"] and metrics["passes"]]
    if not eligible:
        raise BakeoffError("no MTP depth passed all measured gates")
    winner, configuration, metrics = sorted(
        eligible, key=lambda item: _winner_key(item[2], item[0]["row_id"]))[0]
    return {**ledger_base("mtp-depth", plan, rows, digests),
            "prior_format_ledger": prior,
            "prior_format_ledger_sha256": canonical_sha256(prior),
            "stock_identity": prior["stock_identity"],
            "direction_identity": common_direction,
            "runtime_identity": prior["runtime_identity"],
            "selected_configuration_id": prior["selected_configuration_id"],
            "selected_configuration": prior["selected_configuration"],
            "selected_arm_id": prior["selected_arm_id"],
            "selected_arm": selected_arm,
            "selected_depth_id": winner["row_id"],
            "selected_mtp_matrix_quant_contract": selected_arm[
                "mtp_matrix_quant_contract"],
            "selected_mtp_depth": configuration["mtp_depth"],
            "selected_artifact_identity": winner["artifact_identity"],
            "selected_metrics": metrics}


def select_mtp_depth(plan: dict[str, Any], results: dict[str, Any],
                     format_descriptor: dict[str, Any]) -> dict[str, Any]:
    plan = verify_plan(plan)
    prior, _ = read_prior_ledger(format_descriptor, "format", plan)
    rows, digests = assessment_rows(results, {"mtp-depth"}, plan)
    return select_mtp_depth_from_assessments(plan, rows, digests, prior)


def confirm_final(plan: dict[str, Any], results: dict[str, Any],
                  depth_descriptor: dict[str, Any]) -> dict[str, Any]:
    final_plan = verify_plan(plan)
    if final_plan.get("phase_scope") != "final_confirmation":
        raise BakeoffError("final confirmation requires a separately unlocked final plan")
    selection = final_plan["selection_plan"]
    sealed = final_plan["sealed_recipe_ledger"]
    supplied = depth_descriptor.get("subject", {}) if isinstance(depth_descriptor, dict) else {}
    if (supplied.get("sha256") != sealed["sha256"]
            or depth_descriptor != sealed["descriptor"]):
        raise BakeoffError("final confirmation did not use the ledger that unlocked final-heldout")
    prior, _ = read_prior_ledger(depth_descriptor, "mtp-depth", selection)
    rows, digests = assessment_rows(results, {"final"}, selection)
    if len(rows) != 1:
        raise BakeoffError("final requires exactly one preselected assessment")
    row = rows[0]
    if row.get("phase_plan_sha256") != canonical_sha256(final_plan):
        raise BakeoffError("final assessment is not bound to the unlocked final plan")
    if (row.get("row_id") != "final-confirmation"
            or row.get("configuration_id") != prior.get("selected_configuration_id")
            or row.get("arm_id") != prior.get("selected_arm_id")
            or row.get("mtp_matrix_quant_contract") !=
            prior.get("selected_mtp_matrix_quant_contract")
            or row.get("mtp_depth") != prior.get("selected_mtp_depth")):
        raise BakeoffError("final assessment attempted to change the preselected recipe")
    if row.get("artifact_identity") != prior.get("selected_artifact_identity"):
        raise BakeoffError("final assessment is not the exact pair/depth-winning runtime identity")
    if (row.get("direction_identity") != prior.get("direction_identity")
            or row.get("quality_stock_identity") != prior.get("stock_identity")
            or row.get("runtime_identity") != prior.get("runtime_identity")):
        raise BakeoffError("final assessment changed direction or stock-control identity")
    metrics = _metrics(row, selection)
    if not metrics["passes"]:
        raise BakeoffError("preselected winner failed final-heldout confirmation")
    return {**ledger_base("final", final_plan, rows, digests),
            "prior_mtp_depth_ledger_sha256": canonical_sha256(prior),
            "selected_configuration_id": prior["selected_configuration_id"],
            "selected_arm_id": prior["selected_arm_id"],
            "selected_mtp_matrix_quant_contract": prior[
                "selected_mtp_matrix_quant_contract"],
            "selected_mtp_depth": prior["selected_mtp_depth"],
            "selected_artifact_identity": prior["selected_artifact_identity"],
            "final_metrics": metrics,
            "final_heldout_used_for_selection": False}


def decide(plan: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """Reject the removed monolithic selector.

    Schema v2 has two externally attested selection boundaries after format:
    MTP depth selection and final-data unlock. A one-shot decision could bypass
    those capabilities, so callers must use the staged selectors.
    """
    del plan, results
    raise BakeoffError(
        "schema-v2 bakeoff requires staged sweep, format, MTP-depth, and final ledgers")

def prior_descriptor(args: argparse.Namespace, schema: str) -> dict[str, Any]:
    if (args.prior_ledger is None or not isinstance(args.prior_ledger_sha256, str)
            or args.prior_attestation_bundle is None
            or not isinstance(args.prior_attestation_bundle_sha256, str)):
        raise BakeoffError(
            "prior ledger requires exact subject and external-attestation bundle digests")
    recipe = read_object(args.recipe.resolve(), "bakeoff recipe")
    contract = recipe.get("assessment_attestation") or {}
    return {
        "subject": {"path": str(args.prior_ledger.resolve()),
                    "sha256": args.prior_ledger_sha256, "schema": schema},
        "bundle": {"path": str(args.prior_attestation_bundle.resolve()),
                   "sha256": args.prior_attestation_bundle_sha256},
        "repository": contract.get("repository"),
        "signer_workflow": contract.get("signer_workflow"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--stage", choices=(
        "verify", "assess", "sweep", "format", "mtp-depth", "unlock-final", "final"),
        default="verify")
    parser.add_argument("--prior-ledger", type=Path)
    parser.add_argument("--prior-ledger-sha256")
    parser.add_argument("--prior-attestation-bundle", type=Path)
    parser.add_argument("--prior-attestation-bundle-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if (args.corpus_dir is not None and args.plan is None and args.results is None
                and args.stage == "verify"):
            output = make_plan(
                args.recipe.resolve(), args.corpus_dir.resolve(), args.profile.resolve())
        elif args.plan is not None:
            plan = read_object(args.plan, "bakeoff plan")
            if args.stage == "verify" and args.results is None and args.prior_ledger is None:
                verify_plan(plan)
                output = {"schema_version": PLAN_SCHEMA_VERSION,
                          "status": "canonical_plan_verified",
                          "plan_sha256": canonical_sha256(plan), "publication_allowed": False}
            elif args.stage == "assess" and args.results is not None:
                raw = read_object(args.results, "candidate measurement")
                row = raw.get("measurement", raw)
                if not isinstance(row, dict):
                    raise BakeoffError("candidate measurement must be one object")
                output = make_candidate_assessment(plan, row)
            elif args.stage == "unlock-final" and args.corpus_dir is not None:
                if args.prior_ledger is None:
                    raise BakeoffError("unlock-final requires the sealed MTP-depth ledger")
                prior = prior_descriptor(args, LEDGER_SCHEMA)
                output = make_final_plan(plan, args.corpus_dir.resolve(), prior)
            elif args.results is not None:
                results = read_object(args.results, "bakeoff results")
                if args.stage == "sweep" and args.prior_ledger is None:
                    output = select_sweep(plan, results)
                elif args.stage in {"format", "mtp-depth", "final"} and args.prior_ledger is not None:
                    prior = prior_descriptor(args, LEDGER_SCHEMA)
                    if args.stage == "format":
                        output = select_format(plan, results, prior)
                    elif args.stage == "mtp-depth":
                        output = select_mtp_depth(plan, results, prior)
                    else:
                        output = confirm_final(plan, results, prior)
                else:
                    raise BakeoffError("invalid stage/prior-ledger combination")
            else:
                raise BakeoffError("selection stage requires --results")
        else:
            raise BakeoffError("use --corpus-dir to plan, or a --plan stage")
        write_new(args.output.absolute(), output)
    except (BakeoffError, OSError) as exc:
        print(f"qwen_bakeoff.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.absolute()), "status": output["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
