#!/usr/bin/env python3
"""Offline tests for measured-only Qwen bakeoff selection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qwen_bakeoff", ROOT / "scripts" / "qwen_bakeoff.py")
assert SPEC is not None and SPEC.loader is not None
qb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qb)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BakeoffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.quality_results: dict[tuple[str, str], dict] = {}
        self.original_quality_evaluator = qb.QUALITY_EVALUATOR
        self.original_evidence_validator = qb.EVIDENCE_VALIDATOR
        self.original_intervention_validator = qb.INTERVENTION_VALIDATOR
        self.original_attestation_validator = qb.ATTESTATION_VERIFIER

        def fake_evaluator(path: Path, expected_sha: str) -> dict:
            return self.quality_results[(str(path), expected_sha)]

        qb.QUALITY_EVALUATOR = fake_evaluator
        def fake_evidence(row: dict) -> dict:
            return {"candidate_id": row["candidate_id"], "derived": {
                "differential_correctness_pass": row["differential_correctness_pass"],
                "artifact_bytes": row["artifact_bytes"],
                "prefill_tps_samples": row["prefill_tps_samples"],
                "decode_tps_samples": row["decode_tps_samples"],
                "evaluated_prefill_tokens": row["evaluated_prefill_tokens"],
                "completion_tokens": row["completion_tokens"],
                "mtp_spec_ran": row["mtp_spec_ran"],
                "mtp_accept_rates": [0.5, 0.5, 0.5],
                "companion_artifact_bytes": row["companion_artifact_bytes"],
                "resources": {key: row[key] for key in (
                    "runner_memtotal_bytes", "runner_gtt_pages_limit",
                    "peak_memory_measurement_method", "measured_peak_rss_bytes",
                    "measured_peak_gtt_bytes", "measured_peak_uma_bytes")},
            }}
        qb.EVIDENCE_VALIDATOR = fake_evidence
        self.direction_identity = {
            "schema": "ember.qwen3.8.direction-identity.v1",
            "source": {"revision": "source"}, "tooling": {"extractor": "tool"},
            "corpora": [{"sha256": "a" * 64}],
            "extraction": {"activation_evidence": {"sha256": "b" * 64}},
            "directions": [{"layer": layer, "sha256": f"{layer:064x}"}
                           for layer in range(48)],
            "identity_sha256": "c" * 64,
        }
        qb.INTERVENTION_VALIDATOR = lambda row, configuration, corpus_sha: self.direction_identity
        qb.ATTESTATION_VERIFIER = lambda subject, bundle, repository, workflow: None
        self.stock_identity: dict[str, object] | None = None

    def tearDown(self) -> None:
        qb.QUALITY_EVALUATOR = self.original_quality_evaluator
        qb.EVIDENCE_VALIDATOR = self.original_evidence_validator
        qb.INTERVENTION_VALIDATOR = self.original_intervention_validator
        qb.ATTESTATION_VERIFIER = self.original_attestation_validator

    def make_corpora(self, root: Path) -> Path:
        directory = root / "corpora"
        directory.mkdir()
        names = ("extraction-good.jsonl", "extraction-bad.jsonl",
                 "sweep-validation.jsonl", "final-heldout.jsonl")
        artifacts = []
        for index, name in enumerate(names):
            path = directory / name
            path.write_text(json.dumps({"id": f"case-{index}"}) + "\n", encoding="utf-8")
            artifacts.append({"filename": name, "sha256": digest(path), "record_count": 1})
        for manifest, selected in (
            ("qwen-selection-corpora-manifest.json", artifacts[:3]),
            ("qwen-final-corpus-manifest.json", artifacts[3:]),
        ):
            (directory / manifest).write_text(json.dumps({
                "partition": {"pairwise_request_overlap_count": 0},
                "artifacts": selected,
            }), encoding="utf-8")
        return directory

    def measured(self, identifier: str, stage: str, corpus_sha: str, **extra: object) -> dict:
        seed = hashlib.sha256(f"{stage}:{identifier}".encode()).hexdigest()
        row = {
            "id": identifier, "stage": stage, "measurement_kind": "measured",
            "status": "complete", "corpus_sha256": corpus_sha,
            "differential_correctness_pass": True,
            "artifact_bytes": 90_000_000_000,
            "prefill_tps_samples": [412.0, 413.0, 414.0],
            "decode_tps_samples": [39.49, 40.0, 40.5], **extra,
            "evaluated_prefill_tokens": [2074, 2074, 2074],
            "completion_tokens": [256, 256, 256],
            "mtp_spec_ran": [True, True, True],
            "companion_artifact_bytes": {"mtp": 2_000_000_000,
                                         "vision_mmproj": 901_943_132,
                                         "vision_vocab": 12_000_000},
            "enabled_companions": ["mtp", "vision_mmproj", "vision_vocab"],
            "runner_memtotal_bytes": 134_297_894_912,
            "runner_gtt_pages_limit": 32_505_856,
            "peak_memory_measurement_method": "runner_rss_gtt_sampler_v1",
            "measured_peak_rss_bytes": 120_000_000_000,
            "measured_peak_gtt_bytes": 100_000_000_000,
            "measured_peak_uma_bytes": 125_000_000_000,
        }
        row.setdefault("candidate_id", identifier)
        row.setdefault("quantization_arm", "rocmi4-control" if stage == "stock" else "test-arm")
        row.setdefault("intervention_configuration_id", None if stage == "stock" else identifier)
        row.setdefault("build_record_sha256", seed)
        row.setdefault("intervention_manifest_sha256", "2" * 64)
        row.setdefault("companion_inventory_sha256", "6" * 64)
        row.setdefault("mtp_matrix_quant_contract", "Q4_0_ROCMI4")
        row.setdefault("mtp_depth", 3)
        row.setdefault("profile_sha256", "3" * 64)
        row.setdefault("quantization_overrides_sha256", "4" * 64)
        row.setdefault("model_inventory_sha256", hashlib.sha256(
            f"inventory:{stage}:{identifier}".encode()).hexdigest())
        row.setdefault("evidence_manifest", {"path": f"/evidence/{seed}.json",
                                               "sha256": seed,
                                               "schema": qb.RESULT_SCHEMA})
        contract = "9" * 64
        row.setdefault("builder_identity", {
            "ember_revision": "1" * 40, "quantizer_tool_sha256": "2" * 64,
            "container_digest": "sha256:" + "3" * 64,
            "tensor_format_contract_sha256": contract,
        })
        row.setdefault("runtime_identity", {
            "ember_revision": "4" * 40, "engine_binary_sha256": "5" * 64,
            "container_digest": "sha256:" + "6" * 64,
            "tensor_format_contract_sha256": contract,
        })
        row.setdefault("tensor_format_compatibility_sha256", contract)
        quality_identity = {
            "candidate_id": row["candidate_id"],
            "build_record_sha256": row["build_record_sha256"],
            "intervention_manifest_sha256": row["intervention_manifest_sha256"],
            "profile_sha256": row["profile_sha256"],
            "quantization_overrides_sha256": row["quantization_overrides_sha256"],
            "inventory_sha256": row["model_inventory_sha256"],
            "artifact_bytes": row["artifact_bytes"],
        }
        artifact_identity = {
            "candidate_id": row["candidate_id"],
            "build_record_sha256": row["build_record_sha256"],
            "intervention_manifest_sha256": row["intervention_manifest_sha256"],
            "profile_sha256": row["profile_sha256"],
            "quantization_overrides_sha256": row["quantization_overrides_sha256"],
            "model_inventory_sha256": row["model_inventory_sha256"],
            "companion_inventory_sha256": row["companion_inventory_sha256"],
            "mtp_matrix_quant_contract": row["mtp_matrix_quant_contract"],
            "mtp_depth": row["mtp_depth"],
            "artifact_bytes": row["artifact_bytes"],
            "companion_artifact_bytes": row["companion_artifact_bytes"],
            "quantization_arm": row["quantization_arm"],
            "intervention_configuration_id": row["intervention_configuration_id"],
            "builder_identity": row["builder_identity"],
            "tensor_format_compatibility_sha256": contract,
        }
        if stage == "stock":
            self.stock_identity = artifact_identity
        else:
            assert self.stock_identity is not None
            contract_path = f"/quality/{stage}-{seed}.json"
            contract_sha = hashlib.sha256(contract_path.encode()).hexdigest()
            row["quality_contract"] = {"path": contract_path, "sha256": contract_sha}
            self.quality_results[(contract_path, contract_sha)] = {
                "audited_quality_pass": True,
                "quality_score": float(extra.get("quality_score", 0.9)),
                "corpus": {"sha256": corpus_sha},
                "models": {"stock": self.stock_identity, "candidate": quality_identity},
                "release_scope": {"modality": "text_only",
                                  "multimodal_release_claim": False,
                                  "vision_mmproj_differential_pass": False},
                "multimodal_release_approved": False,
            }
        row.pop("quality_score", None)
        return row

    def complete_results(self, plan: dict) -> dict:
        sweep_sha = plan["corpora"]["sweep-validation.jsonl"]["sha256"]
        rows = [self.measured(
            "stock-rocmi4-exact", "stock", sweep_sha, final_release_eligible=False,
        )]
        for index, configuration in enumerate(plan["sweep_configurations"]):
            rows.append(self.measured(
                configuration["id"], "sweep", sweep_sha,
                quality_score=1.0 if index == 0 else 0.8,
                quantization_arm=configuration["quantization_arm"],
                quantization_overrides_sha256=configuration["quantization_overrides_sha256"],
                profile_sha256=configuration["profile_sha256"],
                runtime_mode=configuration["runtime_mode"],
                final_release_eligible=configuration["final_release_eligible"],
            ))
        winner = plan["sweep_configurations"][0]["id"]
        for arm in plan["format_arms"]:
            rows.append(self.measured(
                f"{winner}:{arm['id']}", "format", sweep_sha,
                configuration_id=winner, arm_id=arm["id"],
                intervention_configuration_id=winner,
                runtime_mode=arm["runtime_mode"],
                final_release_eligible=arm["final_release_eligible"],
                quantization_arm=arm["quantization_arm"],
                quantization_overrides_sha256=arm["quantization_overrides_sha256"],
                profile_sha256=arm["profile_sha256"],
                mtp_matrix_quant_contract=arm["mtp_matrix_quant_contract"],
                mtp_depth=arm["mtp_depth"],
                quality_score=(1.0 if arm["id"] ==
                               "rocmi4-q6k-main-rocmi4-mtp-d3" else 0.8),
            ))
        return {"results": rows}

    def persist_attested(self, root: Path, name: str, value: dict,
                         schema: str) -> dict:
        subject = root / f"{name}.json"
        subject.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        bundle = root / f"{name}.bundle.json"
        bundle.write_text(json.dumps({"subject_sha256": digest(subject)}) + "\n",
                          encoding="utf-8")
        return {
            "subject": {"path": str(subject), "sha256": digest(subject), "schema": schema},
            "bundle": {"path": str(bundle), "sha256": digest(bundle)},
            "repository": "OtherU-AI/ember",
            "signer_workflow": ".github/workflows/qwen-gfx1151-bakeoff.yml",
        }

    def assessment_input(self, root: Path, plan: dict, rows: list[dict], prefix: str) -> dict:
        descriptors = []
        for index, row in enumerate(rows):
            assessment = qb.make_candidate_assessment(plan, row)
            descriptors.append(self.persist_attested(
                root, f"{prefix}-{index}", assessment,
                qb.ASSESSMENT_SCHEMA))
        return {"assessments": descriptors}

    def with_balanced_confirmation(self, plan: dict, results: dict) -> dict:
        assessments = [json.loads(Path(item["subject"]["path"]).read_text(
            encoding="utf-8")) for item in results["assessments"]]
        confirmation_plan = qb.balanced_confirmation_plan(plan, assessments)
        first, second = [item["arm_id"] for item in confirmation_plan["finalists"]]
        samples = {
            first: {"prefill": [413.0, 414.0, 415.0],
                    "decode": [40.9, 41.0, 41.1]},
            second: {"prefill": [412.0, 413.0, 414.0],
                     "decode": [39.8, 39.9, 40.0]},
        }
        counts = {first: 0, second: 0}
        by_arm = {row["arm_id"]: row for row in assessments}
        workloads = {item["workload_id"]: item
                     for item in confirmation_plan["workloads"]}
        runs = []
        for index, arm_id in enumerate(confirmation_plan["run_order"]):
            sample = counts[arm_id]
            counts[arm_id] += 1
            assessment = by_arm[arm_id]
            artifact = assessment["artifact_identity"]
            runtime = assessment["runtime_identity"]
            capability = qb.candidate_kernel_capability(
                artifact["quantization_arm"])
            w4a8_required = capability != "no_eligible_rocmi4_mmq"
            workload_id = confirmation_plan["workload_order"][index]
            workload = workloads[workload_id]
            process = {
                "schema": qb.BALANCED_PROCESS_SCHEMA,
                "run_index": index, "arm_id": arm_id,
                "candidate_id": artifact["candidate_id"],
                "container_id": hashlib.sha256(
                    f"container:{index}".encode()).hexdigest(),
                "host_pid": 1000 + index, "proc_start_ticks": 5000 + index,
                "ember_revision": runtime["ember_revision"],
                "container_digest": runtime["container_digest"],
                "engine_binary_sha256": runtime["engine_binary_sha256"],
                "tensor_format_contract_sha256": runtime[
                    "tensor_format_contract_sha256"],
                "candidate_kernel_capability": capability,
                "rocmi4_w4a8_iu4_requested": w4a8_required,
                "candidate_binding_sha256": hashlib.sha256(
                    f"binding:{arm_id}".encode()).hexdigest(),
                "model_first_shard_sha256": hashlib.sha256(
                    f"model:{arm_id}".encode()).hexdigest(),
                "model_inventory_sha256": artifact["model_inventory_sha256"],
                "companion_inventory_sha256": artifact[
                    "companion_inventory_sha256"],
                "mtp_sha256": hashlib.sha256(f"mtp:{arm_id}".encode()).hexdigest(),
                "mtp_depth": artifact["mtp_depth"],
            }
            runs.append({
                "run_index": index, "arm_id": arm_id,
                "process_instance": process,
                "process_instance_sha256": qb.canonical_sha256(process),
                "workload_id": workload_id,
                "workload_recipe_sha256": workload["recipe_sha256"],
                "prefill_prompt_sha256": hashlib.sha256(
                    f"prefill:{workload_id}".encode()).hexdigest(),
                "decode_prompt_sha256": hashlib.sha256(
                    f"decode:{workload_id}".encode()).hexdigest(),
                "calibrated_prefill_words": 2040 + index // 2,
                "evaluated_prefill_tokens": 2074, "completion_tokens": 256,
                "prefill_tps": samples[arm_id]["prefill"][sample],
                "decode_tps": samples[arm_id]["decode"][sample],
                "spec_ran": True, "accept_rate": 0.981,
                "startup_kernel_mode": (
                    "w4a8_iu4_register_pack" if w4a8_required else
                    "not_applicable_no_eligible_rocmi4_mmq"),
                "startup_log_sha256": hashlib.sha256(
                    f"startup:{index}".encode()).hexdigest(),
            })
        results["balanced_confirmation"] = {
            "schema": qb.BALANCED_CONFIRMATION_SCHEMA,
            "confirmation_plan": confirmation_plan,
            "runs": runs,
        }
        return results

    def staged_selection(self, root: Path, plan: dict, corpora: Path,
                         tamper_final_build: bool = False,
                         tamper_final_depth: bool = False) -> tuple[dict, dict, dict, dict]:
        raw = self.complete_results(plan)["results"]
        sweep_rows = [row for row in raw if row["stage"] in {"stock", "sweep"}]
        format_rows = [row for row in raw if row["stage"] == "format"]
        sweep = qb.select_sweep(plan, self.assessment_input(root, plan, sweep_rows, "sweep"))
        sweep_desc = self.persist_attested(
            root, "sweep-ledger", sweep, qb.LEDGER_SCHEMA)
        format_input = self.with_balanced_confirmation(
            plan, self.assessment_input(root, plan, format_rows, "format"))
        selected_format = qb.select_format(plan, format_input, sweep_desc)
        format_desc = self.persist_attested(
            root, "format-ledger", selected_format, qb.LEDGER_SCHEMA)
        winner_raw = next(row for row in format_rows
                          if row["arm_id"] == selected_format["selected_arm_id"])
        depth_rows = []
        for configuration in plan["mtp_depth_configurations"]:
            depth_rows.append(self.measured(
                configuration["id"], "mtp-depth",
                plan["corpora"]["sweep-validation.jsonl"]["sha256"],
                candidate_id=winner_raw["candidate_id"],
                configuration_id=winner_raw["configuration_id"],
                arm_id=winner_raw["arm_id"],
                quantization_arm=winner_raw["quantization_arm"],
                quantization_overrides_sha256=winner_raw[
                    "quantization_overrides_sha256"],
                profile_sha256=winner_raw["profile_sha256"],
                mtp_matrix_quant_contract=winner_raw["mtp_matrix_quant_contract"],
                mtp_depth=configuration["mtp_depth"],
                final_release_eligible=configuration["final_release_eligible"],
                runtime_mode=configuration["runtime_mode"],
                intervention_configuration_id=winner_raw[
                    "intervention_configuration_id"],
                intervention_manifest_sha256=winner_raw[
                    "intervention_manifest_sha256"],
                build_record_sha256=winner_raw["build_record_sha256"],
                companion_inventory_sha256=winner_raw[
                    "companion_inventory_sha256"],
                model_inventory_sha256=winner_raw["model_inventory_sha256"],
                builder_identity=winner_raw["builder_identity"],
                runtime_identity=winner_raw["runtime_identity"],
                tensor_format_compatibility_sha256=winner_raw[
                    "tensor_format_compatibility_sha256"],
                quality_score=1.0 if configuration["mtp_depth"] == 3 else 0.9,
            ))
        selected_depth = qb.select_mtp_depth(
            plan, self.assessment_input(root, plan, depth_rows, "mtp-depth"),
            format_desc)
        depth_desc = self.persist_attested(
            root, "mtp-depth-ledger", selected_depth, qb.LEDGER_SCHEMA)
        final_plan = qb.make_final_plan(plan, corpora, depth_desc)
        winner_raw = next(row for row in depth_rows
                          if row["id"] == selected_depth["selected_depth_id"])
        final_sha = final_plan["corpora"]["final-heldout.jsonl"]["sha256"]
        final = self.measured(
            "final-confirmation", "final", final_sha,
            candidate_id=winner_raw["candidate_id"],
            configuration_id=winner_raw["configuration_id"],
            arm_id=winner_raw["arm_id"],
            quantization_arm=winner_raw["quantization_arm"],
            quantization_overrides_sha256=winner_raw["quantization_overrides_sha256"],
            profile_sha256=winner_raw["profile_sha256"],
            mtp_matrix_quant_contract=winner_raw["mtp_matrix_quant_contract"],
            mtp_depth=(4 if tamper_final_depth and winner_raw["mtp_depth"] != 4
                       else 1 if tamper_final_depth else winner_raw["mtp_depth"]),
            final_release_eligible=winner_raw["final_release_eligible"],
            runtime_mode=winner_raw["runtime_mode"],
            intervention_configuration_id=winner_raw["intervention_configuration_id"],
            intervention_manifest_sha256=winner_raw["intervention_manifest_sha256"],
            build_record_sha256=("f" * 64 if tamper_final_build else
                                 winner_raw["build_record_sha256"]),
            companion_inventory_sha256=winner_raw["companion_inventory_sha256"],
            model_inventory_sha256=winner_raw["model_inventory_sha256"],
            builder_identity=winner_raw["builder_identity"],
            runtime_identity=winner_raw["runtime_identity"],
            tensor_format_compatibility_sha256=winner_raw["tensor_format_compatibility_sha256"],
        )
        final_result = qb.confirm_final(
            final_plan, self.assessment_input(root, final_plan, [final], "final"), depth_desc)
        return sweep, selected_format, selected_depth, final_result

    def test_plan_is_complete_and_final_partition_cannot_select(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpora = self.make_corpora(Path(temporary))
            plan = qb.make_plan(qb.DEFAULT_RECIPE, corpora)
            profile = json.loads((ROOT / "share" / "release_profiles" /
                                  "qwen3.8-flash-next-rocmi4-strix-halo.json")
                                 .read_text(encoding="utf-8"))
            profile_arms = {
                arm["id"] for arm in
                profile["quantization"]["performance_bakeoff"]["arms"]
            }
            self.assertEqual(len(plan["sweep_configurations"]), 16)
            self.assertEqual({arm["id"] for arm in plan["format_arms"]}, {
                "rocmi4-q6k-main-rocmi4-mtp-d3",
                "rocmi4-q6k-main-rocmfp4-fast-mtp-d3",
                "rocmfp4-fast-routed-main-rocmi4-mtp-d3",
                "rocmfp4-fast-routed-main-rocmfp4-fast-mtp-d3",
                "rocmfp4-fast-matrix-main-rocmi4-mtp-d3",
                "rocmfp4-fast-matrix-main-rocmfp4-fast-mtp-d3",
            })
            self.assertEqual(
                {arm["quantization_arm"] for arm in plan["format_arms"]},
                {
                    "rocmi4-q6k-embedding-head",
                    "rocmfp4-fast-routed-experts-q6k-embedding-head",
                    "rocmfp4-fast-matrix-q6k-embedding-head",
                },
            )
            self.assertTrue(all(
                arm["quantization_arm"] in profile_arms
                for arm in plan["format_arms"]
            ))
            self.assertEqual(
                {(arm["quantization_arm"], arm["mtp_matrix_quant_contract"])
                 for arm in plan["format_arms"]},
                {(main, mtp) for main in {
                    "rocmi4-q6k-embedding-head",
                    "rocmfp4-fast-routed-experts-q6k-embedding-head",
                    "rocmfp4-fast-matrix-q6k-embedding-head",
                } for mtp in qb.SUPPORTED_MTP_MATRIX_CONTRACTS},
            )
            self.assertEqual(
                [row["mtp_depth"] for row in plan["mtp_depth_configurations"]],
                [1, 2, 3, 4])
            self.assertEqual([row["id"] for row in plan["auxiliary_controls"]],
                             ["rocmi4-w4a4"])
            self.assertFalse(plan["auxiliary_controls"][0][
                "included_in_exact_runtime_winner_ledger"])
            self.assertFalse(plan["publication_allowed"])
            self.assertEqual(plan["phase_scope"], "selection")
            expected_winner_order = [
                "decode_median_tps_desc",
                "prefill_median_tps_desc",
                "quality_score_desc",
                "id_asc",
            ]
            self.assertEqual(
                plan["recipe"]["value"]["measurement_policy"]["winner_order"],
                expected_winner_order,
            )
            self.assertEqual(
                profile["quantization"]["performance_bakeoff"]["winner_order"],
                expected_winner_order,
            )
            self.assertNotIn("final-heldout.jsonl", plan["corpora"])
            for configuration in plan["sweep_configurations"]:
                self.assertEqual(len(configuration["layer_scales"]), 48)
            sweep, selected_format, selected_depth, decision = self.staged_selection(
                Path(temporary), plan, corpora)
            self.assertEqual(sweep["selected_configuration_id"],
                             plan["sweep_configurations"][0]["id"])
            self.assertEqual(selected_format["selected_arm_id"],
                             "rocmi4-q6k-main-rocmi4-mtp-d3")
            self.assertEqual(selected_depth["selected_mtp_depth"], 3)
            self.assertFalse(decision["final_heldout_used_for_selection"])
            self.assertFalse(decision["publication_allowed"])

    def test_cross_pair_and_auxiliary_control_inventory_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpora = self.make_corpora(root)
            recipe = json.loads(qb.DEFAULT_RECIPE.read_text(encoding="utf-8"))
            recipe["format_arms"][1]["mtp_matrix_quant_contract"] = "Q4_0_ROCMI4"
            duplicate = root / "duplicate-pair.json"
            duplicate.write_text(json.dumps(recipe), encoding="utf-8")
            with self.assertRaisesRegex(qb.BakeoffError, "full main/MTP cross-pair"):
                qb.make_plan(duplicate, corpora)

            recipe = json.loads(qb.DEFAULT_RECIPE.read_text(encoding="utf-8"))
            recipe["format_arms"].append(recipe["auxiliary_controls"][0])
            injected = root / "w4a4-injected.json"
            injected.write_text(json.dumps(recipe), encoding="utf-8")
            with self.assertRaisesRegex(qb.BakeoffError,
                                        "selectable format arms must be exact-runtime"):
                qb.make_plan(injected, corpora)

            recipe = json.loads(qb.DEFAULT_RECIPE.read_text(encoding="utf-8"))
            recipe["measurement_policy"]["winner_order"] = [
                "quality_score_desc", "decode_median_tps_desc",
                "prefill_median_tps_desc", "id_asc",
            ]
            quality_first = root / "quality-first-policy.json"
            quality_first.write_text(json.dumps(recipe), encoding="utf-8")
            with self.assertRaisesRegex(qb.BakeoffError, "performance-first winner order"):
                qb.make_plan(quality_first, corpora)

    def test_performance_first_winner_key_and_hard_gate_filter(self) -> None:
        def metrics(decode: float, prefill: float, quality: float) -> dict:
            return {
                "decode_median_tps": decode,
                "prefill_median_tps": prefill,
                "quality_score": quality,
            }

        cases = [
            ("decode", [
                ("quality", metrics(40.0, 500.0, 1.0)),
                ("decode", metrics(41.0, 412.0, 0.5)),
            ]),
            ("prefill", [
                ("quality", metrics(40.0, 413.0, 1.0)),
                ("prefill", metrics(40.0, 414.0, 0.5)),
            ]),
            ("quality", [
                ("lower-quality", metrics(40.0, 414.0, 0.5)),
                ("quality", metrics(40.0, 414.0, 1.0)),
            ]),
            ("a", [
                ("z", metrics(40.0, 414.0, 1.0)),
                ("a", metrics(40.0, 414.0, 1.0)),
            ]),
        ]
        for expected, rows in cases:
            with self.subTest(expected=expected):
                selected = sorted(rows, key=lambda item: qb._winner_key(item[1], item[0]))[0]
                self.assertEqual(selected[0], expected)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = qb.make_plan(qb.DEFAULT_RECIPE, self.make_corpora(root))
            rows = self.complete_results(plan)["results"]
            sweep_rows = [row for row in rows if row["stage"] in {"stock", "sweep"}]
            candidates = [row for row in sweep_rows if row["stage"] == "sweep"]
            faster = candidates[1]
            faster["decode_tps_samples"] = [41.0, 42.0, 43.0]
            # This row has the highest raw decode result, but misses the hard
            # prefill-peak gate and therefore cannot enter the winner ranking.
            ineligible = candidates[2]
            ineligible["decode_tps_samples"] = [90.0, 100.0, 110.0]
            ineligible["prefill_tps_samples"] = [300.0, 301.0, 302.0]
            selected = qb.select_sweep(
                plan, self.assessment_input(root, plan, sweep_rows, "performance-first"))
            self.assertEqual(selected["selected_configuration_id"], faster["id"])
            self.assertEqual(selected["selected_metrics"]["decode_median_tps"], 42.0)

    def test_rolling_sweep_retains_stock_and_evicts_each_candidate_once(self) -> None:
        override_sha = "1" * 64
        profile_sha = "2" * 64
        plan = {
            "stock_control": {"id": "stock"},
            "sweep_configurations": [{
                "id": item, "quantization_arm": "test-arm",
                "quantization_overrides_sha256": override_sha,
                "profile_sha256": profile_sha, "runtime_mode": "target",
                "final_release_eligible": True,
            } for item in ("s1", "s2", "s3")],
            "format_arms": [],
        }
        plan_sha = qb.canonical_sha256(plan)
        runtime = {"identity": "common-runtime"}
        direction = {"identity": "common-direction"}
        stock_artifact = {"candidate_id": "stock-candidate"}
        rows = [
            {"row_id": "stock", "stage": "stock",
             "selection_plan_sha256": plan_sha, "phase_plan_sha256": plan_sha,
             "final_release_eligible": False, "runtime_identity": runtime,
             "artifact_identity": stock_artifact},
            {"row_id": "s1", "stage": "sweep", "configuration_id": "s1",
             "selection_plan_sha256": plan_sha, "phase_plan_sha256": plan_sha,
             "runtime_mode": "target", "final_release_eligible": True,
             "runtime_identity": runtime, "direction_identity": direction,
             "quality_stock_identity": stock_artifact,
             "artifact_identity": {
                 "candidate_id": "candidate-1", "intervention_configuration_id": "s1",
                 "quantization_arm": "test-arm",
                 "quantization_overrides_sha256": override_sha,
                 "profile_sha256": profile_sha},
             "metrics": {"passes": True, "decode_median_tps": 40.0,
                         "prefill_median_tps": 420.0, "quality_score": 0.8}},
            {"row_id": "s2", "stage": "sweep", "configuration_id": "s2",
             "selection_plan_sha256": plan_sha, "phase_plan_sha256": plan_sha,
             "runtime_mode": "target", "final_release_eligible": True,
             "runtime_identity": runtime, "direction_identity": direction,
             "quality_stock_identity": stock_artifact,
             "artifact_identity": {
                 "candidate_id": "candidate-2", "intervention_configuration_id": "s2",
                 "quantization_arm": "test-arm",
                 "quantization_overrides_sha256": override_sha,
                 "profile_sha256": profile_sha},
             "metrics": {"passes": True, "decode_median_tps": 41.0,
                         "prefill_median_tps": 415.0, "quality_score": 0.7}},
            {"row_id": "s3", "stage": "sweep", "configuration_id": "s3",
             "selection_plan_sha256": plan_sha, "phase_plan_sha256": plan_sha,
             "runtime_mode": "target", "final_release_eligible": True,
             "runtime_identity": runtime, "direction_identity": direction,
             "quality_stock_identity": stock_artifact,
             "artifact_identity": {
                 "candidate_id": "candidate-3", "intervention_configuration_id": "s3",
                 "quantization_arm": "test-arm",
                 "quantization_overrides_sha256": override_sha,
                 "profile_sha256": profile_sha},
             "metrics": {"passes": False, "decode_median_tps": 45.0,
                         "prefill_median_tps": 500.0, "quality_score": 1.0}},
        ]
        with (mock.patch.object(qb, "verify_plan", return_value=plan),
              mock.patch.object(qb, "_metrics", side_effect=lambda row, _plan: row["metrics"])):
            first = qb.rolling_retention_transition(plan, rows[:2], "sweep")
            displaced = qb.rolling_retention_transition(plan, rows[:3], "sweep")
            failed = qb.rolling_retention_transition(plan, rows, "sweep")
        self.assertEqual(first["retained_candidate_ids"],
                         ["stock-candidate", "candidate-1"])
        self.assertEqual(first["retire_candidate_ids"], [])
        self.assertEqual(displaced["retained_candidate_ids"],
                         ["stock-candidate", "candidate-2"])
        self.assertEqual(displaced["retire_candidate_ids"], ["candidate-1"])
        self.assertEqual(failed["retire_candidate_ids"], ["candidate-3"])

    def test_rolling_format_keeps_exact_two_eligible_finalists(self) -> None:
        override_sha = "3" * 64
        profile_sha = "4" * 64
        plan = {
            "stock_control": {"id": "stock"},
            "sweep_configurations": [{"id": "winner"}],
            "format_arms": [{
                "id": item, "quantization_arm": "format-arm",
                "quantization_overrides_sha256": override_sha,
                "profile_sha256": profile_sha,
                "mtp_matrix_quant_contract": "Q4_0_ROCMI4", "mtp_depth": 3,
                "final_release_eligible": True,
            } for item in ("a", "b", "c")],
        }
        plan_sha = qb.canonical_sha256(plan)
        runtime = {"identity": "common-runtime"}
        direction = {"identity": "common-direction"}
        stock = {"candidate_id": "stock-candidate"}
        rows = [
            {"row_id": "a", "arm_id": "a", "stage": "format",
             "selection_plan_sha256": plan_sha, "phase_plan_sha256": plan_sha,
             "configuration_id": "winner", "runtime_identity": runtime,
             "direction_identity": direction, "quality_stock_identity": stock,
             "mtp_matrix_quant_contract": "Q4_0_ROCMI4", "mtp_depth": 3,
             "final_release_eligible": True,
             "artifact_identity": {
                 "candidate_id": "candidate-a", "intervention_configuration_id": "winner",
                 "quantization_arm": "format-arm",
                 "quantization_overrides_sha256": override_sha,
                 "profile_sha256": profile_sha},
             "metrics": {"passes": True, "decode_median_tps": 40.0,
                         "prefill_median_tps": 420.0, "quality_score": 0.8}},
            {"row_id": "b", "arm_id": "b", "stage": "format",
             "selection_plan_sha256": plan_sha, "phase_plan_sha256": plan_sha,
             "configuration_id": "winner", "runtime_identity": runtime,
             "direction_identity": direction, "quality_stock_identity": stock,
             "mtp_matrix_quant_contract": "Q4_0_ROCMI4", "mtp_depth": 3,
             "final_release_eligible": True,
             "artifact_identity": {
                 "candidate_id": "candidate-b", "intervention_configuration_id": "winner",
                 "quantization_arm": "format-arm",
                 "quantization_overrides_sha256": override_sha,
                 "profile_sha256": profile_sha},
             "metrics": {"passes": True, "decode_median_tps": 41.0,
                         "prefill_median_tps": 415.0, "quality_score": 0.7}},
            {"row_id": "c", "arm_id": "c", "stage": "format",
             "selection_plan_sha256": plan_sha, "phase_plan_sha256": plan_sha,
             "configuration_id": "winner", "runtime_identity": runtime,
             "direction_identity": direction, "quality_stock_identity": stock,
             "mtp_matrix_quant_contract": "Q4_0_ROCMI4", "mtp_depth": 3,
             "final_release_eligible": True,
             "artifact_identity": {
                 "candidate_id": "candidate-c", "intervention_configuration_id": "winner",
                 "quantization_arm": "format-arm",
                 "quantization_overrides_sha256": override_sha,
                 "profile_sha256": profile_sha},
             "metrics": {"passes": True, "decode_median_tps": 42.0,
                         "prefill_median_tps": 414.0, "quality_score": 0.6}},
        ]
        with (mock.patch.object(qb, "verify_plan", return_value=plan),
              mock.patch.object(qb, "_metrics", side_effect=lambda row, _plan: row["metrics"])):
            transition = qb.rolling_retention_transition(plan, rows, "format")
        self.assertEqual(transition["retained_candidate_ids"],
                         ["candidate-c", "candidate-b"])
        self.assertEqual(transition["retire_candidate_ids"], ["candidate-a"])

        mismatched = json.loads(json.dumps(rows))
        mismatched[-1]["artifact_identity"]["quantization_arm"] = "other-arm"
        with (mock.patch.object(qb, "verify_plan", return_value=plan),
              mock.patch.object(qb, "_metrics",
                                side_effect=lambda row, _plan: row["metrics"]),
              self.assertRaisesRegex(qb.BakeoffError, "provenance differs")):
            qb.rolling_retention_transition(plan, mismatched, "format")

    def test_sealed_format_retention_collapses_only_to_ledger_winner(self) -> None:
        plan = {"stock_control": {"id": "stock"}, "sweep_configurations": [],
                "format_arms": []}
        ledger = {
            "phase": "format", "assessments": [],
            "selected_artifact_identity": {"candidate_id": "candidate-b"},
        }
        rolling = {"retained_candidate_ids": ["candidate-a", "candidate-b"]}
        with (mock.patch.object(qb, "verify_plan", return_value=plan),
              mock.patch.object(qb, "verify_ledger_semantics"),
              mock.patch.object(qb, "rolling_retention_transition",
                                return_value=rolling)):
            transition = qb.sealed_format_retention(plan, ledger)
        self.assertEqual(transition["retained_candidate_ids"], ["candidate-b"])
        self.assertEqual(transition["retire_candidate_ids"], ["candidate-a"])

    def test_missing_estimated_or_final_reselection_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = qb.make_plan(qb.DEFAULT_RECIPE, self.make_corpora(Path(temporary)))
            results = self.complete_results(plan)["results"]
            results[1]["measurement_kind"] = "estimated"
            with self.assertRaisesRegex(qb.BakeoffError, "complete measured"):
                qb.make_candidate_assessment(plan, results[1])
            with self.assertRaisesRegex(qb.BakeoffError, "staged sweep, format, MTP-depth"):
                qb.decide(plan, {"results": results})

    def test_staged_ledgers_bind_prior_selection_and_canonical_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = qb.make_plan(qb.DEFAULT_RECIPE, self.make_corpora(root))
            complete = self.complete_results(plan)["results"]
            sweep_results = self.assessment_input(
                root, plan, [row for row in complete if row["stage"] in {"stock", "sweep"}],
                "staged-sweep")
            format_results = self.with_balanced_confirmation(plan, self.assessment_input(
                root, plan, [row for row in complete if row["stage"] == "format"],
                "staged-format"))
            sweep = qb.select_sweep(plan, sweep_results)
            self.assertEqual(sweep["phase"], "sweep")
            sweep_descriptor = self.persist_attested(
                root, "staged-sweep-ledger", sweep,
                qb.LEDGER_SCHEMA)
            selected_format = qb.select_format(plan, format_results, sweep_descriptor)
            self.assertEqual(selected_format["prior_sweep_ledger_sha256"],
                             qb.canonical_sha256(sweep))
            format_descriptor = self.persist_attested(
                root, "staged-format-ledger", selected_format,
                qb.LEDGER_SCHEMA)
            with self.assertRaisesRegex(qb.BakeoffError, "mtp-depth prior ledger"):
                qb.make_final_plan(plan, root / "corpora", format_descriptor)

            forged = json.loads(json.dumps(sweep))
            forged["selected_configuration_id"] = plan["sweep_configurations"][1]["id"]
            forged_descriptor = self.persist_attested(
                root, "forged-sweep-ledger", forged,
                qb.LEDGER_SCHEMA)
            with self.assertRaisesRegex(qb.BakeoffError, "do not reproduce|semantics"):
                qb.select_format(plan, format_results, forged_descriptor)

            tampered = json.loads(json.dumps(plan))
            tampered["sweep_configurations"][0]["scale"] = 9.0
            with self.assertRaisesRegex(qb.BakeoffError, "semantics differ"):
                qb.verify_plan(tampered)

    def test_balanced_confirmation_is_counterbalanced_fresh_and_noise_aware(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = qb.make_plan(qb.DEFAULT_RECIPE, self.make_corpora(root))
            raw = self.complete_results(plan)["results"]
            sweep_rows = [row for row in raw if row["stage"] in {"stock", "sweep"}]
            format_rows = [row for row in raw if row["stage"] == "format"]
            sweep = qb.select_sweep(
                plan, self.assessment_input(root, plan, sweep_rows, "balanced-sweep"))
            sweep_descriptor = self.persist_attested(
                root, "balanced-sweep-ledger", sweep, qb.LEDGER_SCHEMA)
            base = self.assessment_input(root, plan, format_rows, "balanced-format")

            with self.assertRaisesRegex(qb.BakeoffError, "requires balanced finalist"):
                qb.select_format(plan, base, sweep_descriptor)

            confirmed = self.with_balanced_confirmation(
                plan, json.loads(json.dumps(base)))
            selected = qb.select_format(plan, confirmed, sweep_descriptor)
            evidence = selected["balanced_confirmation"]
            arms = [item["arm_id"] for item in
                    evidence["confirmation_plan"]["finalists"]]
            self.assertEqual(evidence["confirmation_plan"]["run_order"],
                             [arms[0], arms[1], arms[1], arms[0], arms[0], arms[1]])
            self.assertEqual(evidence["confirmation_plan"]["workload_order"],
                             ["counterbalanced-pair-0"] * 2 +
                             ["counterbalanced-pair-1"] * 2 +
                             ["counterbalanced-pair-2"] * 2)
            self.assertEqual(len({row["process_instance_sha256"]
                                  for row in evidence["runs"]}), 6)
            self.assertTrue(evidence["sequence_consistency_pass"])
            self.assertEqual(selected["selected_metrics"]["decode_median_tps"], 41.0)
            self.assertGreater(evidence["decode_median_difference_tps"],
                               evidence["observed_run_noise_tps"])
            self.assertEqual(
                evidence["predeclared_practical_effect_floor_tps"], 1.0)
            self.assertGreater(evidence["decode_median_difference_tps"],
                               evidence["required_decode_separation_tps"])

            wrong_order = self.with_balanced_confirmation(
                plan, json.loads(json.dumps(base)))
            wrong_order["balanced_confirmation"]["runs"][0]["arm_id"] = (
                wrong_order["balanced_confirmation"]["runs"][1]["arm_id"])
            with self.assertRaisesRegex(qb.BakeoffError, "persisted counterbalanced order"):
                qb.select_format(plan, wrong_order, sweep_descriptor)

            reused = self.with_balanced_confirmation(
                plan, json.loads(json.dumps(base)))
            reused_process = reused["balanced_confirmation"]["runs"][1]["process_instance"]
            reused_process["container_id"] = reused["balanced_confirmation"]["runs"][0][
                "process_instance"]["container_id"]
            reused["balanced_confirmation"]["runs"][1]["process_instance_sha256"] = (
                qb.canonical_sha256(reused_process))
            with self.assertRaisesRegex(qb.BakeoffError, "unique fresh process"):
                qb.select_format(plan, reused, sweep_descriptor)

            wrong_mtp = self.with_balanced_confirmation(
                plan, json.loads(json.dumps(base)))
            wrong_mtp["balanced_confirmation"]["runs"][0]["spec_ran"] = False
            with self.assertRaisesRegex(qb.BakeoffError, "run native MTP"):
                qb.select_format(plan, wrong_mtp, sweep_descriptor)

            full_accept = self.with_balanced_confirmation(
                plan, json.loads(json.dumps(base)))
            full_accept["balanced_confirmation"]["runs"][0]["accept_rate"] = 1.0
            with self.assertRaisesRegex(qb.BakeoffError, "0 < accept_rate < 1"):
                qb.select_format(plan, full_accept, sweep_descriptor)

            wrong_binary = self.with_balanced_confirmation(
                plan, json.loads(json.dumps(base)))
            process = wrong_binary["balanced_confirmation"]["runs"][0][
                "process_instance"]
            process["engine_binary_sha256"] = "f" * 64
            wrong_binary["balanced_confirmation"]["runs"][0][
                "process_instance_sha256"] = qb.canonical_sha256(process)
            with self.assertRaisesRegex(qb.BakeoffError, "candidate/image/binary/MTP"):
                qb.select_format(plan, wrong_binary, sweep_descriptor)

            wrong_prompt = self.with_balanced_confirmation(
                plan, json.loads(json.dumps(base)))
            wrong_prompt["balanced_confirmation"]["runs"][1][
                "prefill_prompt_sha256"] = "f" * 64
            with self.assertRaisesRegex(qb.BakeoffError, "identical prompts once per arm"):
                qb.select_format(plan, wrong_prompt, sweep_descriptor)

            wrong_shape = self.with_balanced_confirmation(
                plan, json.loads(json.dumps(base)))
            wrong_shape["balanced_confirmation"]["runs"][0][
                "evaluated_prefill_tokens"] = 2073
            with self.assertRaisesRegex(qb.BakeoffError, "exactly 2074 prefill"):
                qb.select_format(plan, wrong_shape, sweep_descriptor)

            below_gate = self.with_balanced_confirmation(
                plan, json.loads(json.dumps(base)))
            loser = below_gate["balanced_confirmation"]["confirmation_plan"][
                "run_order"][1]
            for run in below_gate["balanced_confirmation"]["runs"]:
                if run["arm_id"] == loser:
                    run["prefill_tps"] = 411.99
            with self.assertRaisesRegex(qb.BakeoffError, "unchanged hard gates"):
                qb.select_format(plan, below_gate, sweep_descriptor)

            noisy = self.with_balanced_confirmation(
                plan, json.loads(json.dumps(base)))
            arms = noisy["balanced_confirmation"]["confirmation_plan"]["run_order"][:2]
            noisy_values = {arms[0]: [39.8, 40.0, 40.2],
                            arms[1]: [39.7, 39.9, 40.1]}
            counts = {arms[0]: 0, arms[1]: 0}
            for run in noisy["balanced_confirmation"]["runs"]:
                arm_id = run["arm_id"]
                run["decode_tps"] = noisy_values[arm_id][counts[arm_id]]
                counts[arm_id] += 1
            with self.assertRaisesRegex(qb.BakeoffError, "within observed run noise"):
                qb.select_format(plan, noisy, sweep_descriptor)

            immaterial = self.with_balanced_confirmation(
                plan, json.loads(json.dumps(base)))
            arms = immaterial["balanced_confirmation"]["confirmation_plan"][
                "run_order"][:2]
            for run in immaterial["balanced_confirmation"]["runs"]:
                run["decode_tps"] = 40.5 if run["arm_id"] == arms[0] else 40.0
            with self.assertRaisesRegex(qb.BakeoffError, "practical-effect floor"):
                qb.select_format(plan, immaterial, sweep_descriptor)

            sequence = self.with_balanced_confirmation(
                plan, json.loads(json.dumps(base)))
            arms = [item["arm_id"] for item in sequence["balanced_confirmation"][
                "confirmation_plan"]["finalists"]]
            sequence_values = {arms[0]: [42.0, 40.5, 42.0],
                               arms[1]: [40.0, 41.2, 40.0]}
            counts = {arms[0]: 0, arms[1]: 0}
            for run in sequence["balanced_confirmation"]["runs"]:
                arm_id = run["arm_id"]
                run["decode_tps"] = sequence_values[arm_id][counts[arm_id]]
                counts[arm_id] += 1
            with self.assertRaisesRegex(qb.BakeoffError, "sequence/order inconsistent"):
                qb.select_format(plan, sequence, sweep_descriptor)

    def test_intervention_layer_scales_and_accept_rate_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = qb.make_plan(qb.DEFAULT_RECIPE, self.make_corpora(root))
            configuration = plan["sweep_configurations"][0]
            targets = []
            for layer, scale in configuration["layer_scales"].items():
                if scale:
                    suffix = "attn_output" if int(layer) % 4 == 3 else "ssm_out"
                    targets.append({"tensor_name": f"blk.{layer}.{suffix}.weight",
                                    "scale": scale})
            manifest_path = root / "intervention.json"
            base_manifest = {
                "kind": "directional_ablation", "status": "complete",
                "held_out_evaluation": {"sha256": plan["corpora"]["sweep-validation.jsonl"]["sha256"]},
                "source": configuration["direction_basis"]["source"],
                "tooling": configuration["direction_basis"]["tooling"],
                "corpora": configuration["direction_basis"]["corpora"],
                "extraction": {
                    **configuration["direction_basis"]["extraction"],
                    "layer_policy": configuration["layer_policy"],
                    "activation_evidence": {"sha256": "a" * 64},
                },
                "directions": [{"id": f"layer-{layer:02d}", "dtype": "F32",
                                "layer": layer,
                                "activation": "residual_writer.output",
                                "sha256": f"{layer:064x}"} for layer in range(48)],
                "targets": targets,
            }
            manifest_path.write_text(json.dumps(base_manifest), encoding="utf-8")
            row = {"intervention_manifest": {"path": str(manifest_path),
                                               "sha256": digest(manifest_path)},
                   "intervention_manifest_sha256": digest(manifest_path),
                   "intervention_configuration_id": configuration["id"],
                   "configuration_id": configuration["id"]}
            qb.validate_intervention_binding(
                row, configuration, plan["corpora"]["sweep-validation.jsonl"]["sha256"])
            base_manifest["extraction"]["semantic_capture_point"] = (
                "decoder_layer.attn_hyper_connection.mixed_input"
            )
            manifest_path.write_text(json.dumps(base_manifest), encoding="utf-8")
            row["intervention_manifest"]["sha256"] = digest(manifest_path)
            row["intervention_manifest_sha256"] = digest(manifest_path)
            with self.assertRaisesRegex(qb.BakeoffError, "extraction semantics"):
                qb.validate_intervention_binding(
                    row, configuration,
                    plan["corpora"]["sweep-validation.jsonl"]["sha256"],
                )
            base_manifest["extraction"] = {
                **configuration["direction_basis"]["extraction"],
                "layer_policy": "upper-24",
                "activation_evidence": {"sha256": "a" * 64},
            }
            manifest_path.write_text(json.dumps(base_manifest), encoding="utf-8")
            row["intervention_manifest"]["sha256"] = digest(manifest_path)
            row["intervention_manifest_sha256"] = digest(manifest_path)
            with self.assertRaisesRegex(qb.BakeoffError, "extraction semantics"):
                qb.validate_intervention_binding(
                    row, configuration,
                    plan["corpora"]["sweep-validation.jsonl"]["sha256"],
                )
            base_manifest["extraction"] = {
                **configuration["direction_basis"]["extraction"],
                "layer_policy": configuration["layer_policy"],
                "activation_evidence": {"sha256": "a" * 64},
            }
            targets[0]["scale"] = 9.0
            base_manifest["targets"] = targets
            manifest_path.write_text(json.dumps(base_manifest), encoding="utf-8")
            row["intervention_manifest"]["sha256"] = digest(manifest_path)
            row["intervention_manifest_sha256"] = digest(manifest_path)
            with self.assertRaisesRegex(qb.BakeoffError, "lambda/layer scales"):
                qb.validate_intervention_binding(
                    row, configuration,
                    plan["corpora"]["sweep-validation.jsonl"]["sha256"])

            recipe = json.loads(qb.DEFAULT_RECIPE.read_text(encoding="utf-8"))
            self.measured("stock-rocmi4-exact", "stock", "a" * 64)
            measured = self.measured("zero-rate", "sweep", "a" * 64)
            original = qb.EVIDENCE_VALIDATOR
            good = original(measured)
            good["derived"]["mtp_accept_rates"] = [0.0, 0.5, 0.5]
            qb.EVIDENCE_VALIDATOR = lambda unused: good
            try:
                with self.assertRaisesRegex(qb.BakeoffError, "accept_rate"):
                    qb.assess(measured, recipe["hard_gates"], "a" * 64)
            finally:
                qb.EVIDENCE_VALIDATOR = original
            changed = json.loads(json.dumps(base_manifest))
            changed["directions"][0]["sha256"] = "f" * 64
            changed_identity = qb.direction_identity(changed)
            self.assertNotEqual(
                changed_identity["identity_sha256"],
                qb.direction_identity(base_manifest)["identity_sha256"])

    def test_stock_baseline_may_miss_final_performance_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = qb.make_plan(qb.DEFAULT_RECIPE, self.make_corpora(Path(temporary)))
            results = self.complete_results(plan)
            stock = results["results"][0]
            stock["prefill_tps_samples"] = [300.0, 301.0, 302.0]
            stock["decode_tps_samples"] = [30.0, 31.0, 32.0]
            stock_assessment = qb.make_candidate_assessment(plan, stock)
            metrics = stock_assessment["observed_decision"]
            self.assertFalse(metrics["passes"])
            self.assertFalse(metrics["performance_passes"])
            self.assertTrue(metrics["correctness_quality_pass"])
            self.assertTrue(metrics["memory_fits"])

    def test_final_is_inaccessible_until_format_seal_and_exact_artifact_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpora = self.make_corpora(root)
            final_path = corpora / "final-heldout.jsonl"
            hidden = corpora / "final-heldout.hidden"
            final_path.rename(hidden)
            # Selection succeeds while the final file is literally unavailable.
            plan = qb.make_plan(qb.DEFAULT_RECIPE, corpora)
            self.assertNotIn(str(hidden), json.dumps(plan))
            hidden.rename(final_path)
            with self.assertRaisesRegex(qb.BakeoffError, "exact pair/depth-winning runtime"):
                self.staged_selection(root, plan, corpora, tamper_final_build=True)
            with self.assertRaisesRegex(qb.BakeoffError, "preselected recipe"):
                self.staged_selection(root, plan, corpora, tamper_final_depth=True)

    def test_compact_assessments_survive_artifact_deletion_and_rederive_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = qb.make_plan(qb.DEFAULT_RECIPE, self.make_corpora(root))
            raw = self.complete_results(plan)["results"]
            sweep_rows = [row for row in raw if row["stage"] in {"stock", "sweep"}]
            inputs = self.assessment_input(root, plan, sweep_rows, "deletion")
            enormous_artifact = root / "candidate.gguf"
            enormous_artifact.write_bytes(b"measured-then-deleted")
            enormous_artifact.unlink()
            selected = qb.select_sweep(plan, inputs)
            self.assertEqual(selected["selected_configuration_id"],
                             plan["sweep_configurations"][0]["id"])
            first = inputs["assessments"][1]["subject"]
            assessment = json.loads(Path(first["path"]).read_text(encoding="utf-8"))
            assessment["observed_decision"]["decode_median_tps"] += 1.0
            forged = self.persist_attested(
                root, "scalar-forgery", assessment,
                qb.ASSESSMENT_SCHEMA)
            tampered = json.loads(json.dumps(inputs))
            tampered["assessments"][1] = forged
            with self.assertRaisesRegex(qb.BakeoffError, "do not rederive"):
                qb.select_sweep(plan, tampered)

    def test_direction_and_runtime_must_be_common_but_builder_is_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = qb.make_plan(qb.DEFAULT_RECIPE, self.make_corpora(root))
            raw = self.complete_results(plan)["results"]
            sweep_rows = [row for row in raw if row["stage"] in {"stock", "sweep"}]
            inputs = self.assessment_input(root, plan, sweep_rows, "identity")
            descriptor = inputs["assessments"][2]["subject"]
            changed = json.loads(Path(descriptor["path"]).read_text(encoding="utf-8"))
            changed["direction_identity"]["directions"][0]["sha256"] = "f" * 64
            inputs["assessments"][2] = self.persist_attested(
                root, "direction-forgery", changed,
                qb.ASSESSMENT_SCHEMA)
            with self.assertRaisesRegex(qb.BakeoffError, "common direction"):
                qb.select_sweep(plan, inputs)

            candidate = raw[1]
            candidate["runtime_identity"] = dict(candidate["runtime_identity"])
            candidate["runtime_identity"]["ember_revision"] = "a" * 40
            # A runtime-only revision is valid and leaves artifact identity tied
            # to its original builder. Comparative selection later requires all
            # candidates to use this same runtime.
            assessment = qb.make_candidate_assessment(plan, candidate)
            self.assertEqual(assessment["artifact_identity"]["builder_identity"]["ember_revision"],
                             "1" * 40)
            self.assertEqual(assessment["runtime_identity"]["ember_revision"], "a" * 40)
            candidate["runtime_identity"]["tensor_format_contract_sha256"] = "0" * 64
            with self.assertRaisesRegex(qb.BakeoffError, "tensor-format compatibility"):
                qb.make_candidate_assessment(plan, candidate)

    def test_hard_perf_and_128g_fit_gates_are_inclusive(self) -> None:
        recipe = json.loads(qb.DEFAULT_RECIPE.read_text(encoding="utf-8"))
        self.measured("stock-rocmi4-exact", "stock", "a" * 64)
        row = self.measured("x", "sweep", "a" * 64)
        passed = qb.assess(row, recipe["hard_gates"], "a" * 64)
        self.assertTrue(passed["passes"])
        row["prefill_tps_samples"] = [390.0, 400.0, 412.0]
        peak_passed = qb.assess(row, recipe["hard_gates"], "a" * 64)
        self.assertTrue(peak_passed["passes"])
        self.assertEqual(peak_passed["prefill_peak_tps"], 412.0)
        row = self.measured(
            "too-large", "sweep", "a" * 64,
            artifact_bytes=(recipe["hard_gates"]["device_budget_bytes"]
                            - recipe["hard_gates"]["runtime_reserve_bytes"] + 1),
        )
        self.assertFalse(qb.assess(row, recipe["hard_gates"], "a" * 64)["passes"])
        row = self.measured("x", "sweep", "a" * 64)
        row["measured_peak_gtt_bytes"] = recipe["hard_gates"]["certification_host_gtt_cap_bytes"] + 1
        row["measured_peak_uma_bytes"] = row["measured_peak_gtt_bytes"]
        self.assertFalse(qb.assess(row, recipe["hard_gates"], "a" * 64)["passes"])

    def test_measurement_manifest_binds_hardware_samples_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self.measured("stock-rocmi4-exact", "stock", "a" * 64)
            shard = {"path": "/models/stock.gguf", "sha256": "5" * 64,
                     "size_bytes": row["artifact_bytes"]}
            identity = [{"index": 1, "sha256": shard["sha256"],
                         "bytes": shard["size_bytes"]}]
            row["model_inventory_sha256"] = hashlib.sha256(
                (json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest()

            def put(name: str, value: dict) -> dict[str, str]:
                path = root / name
                raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
                path.write_bytes(raw)
                return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

            def put_jsonl(name: str, values: list[dict]) -> dict[str, str]:
                path = root / name
                raw = b"".join((json.dumps(value, sort_keys=True) + "\n").encode()
                               for value in values)
                path.write_bytes(raw)
                return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

            row["companion_artifact_bytes"] = {
                "mtp": 3, "vision_mmproj": 4, "vision_vocab": 5}
            mtp_path = root / "mtp.gguf"; mtp_path.write_bytes(b"mtp")
            mmproj_path = root / "mmproj.gguf"; mmproj_path.write_bytes(b"mmpr")
            vocab_path = root / "vocab.gguf"; vocab_path.write_bytes(b"vocab")
            resources = {key: row[key] for key in (
                "runner_memtotal_bytes", "runner_gtt_pages_limit",
                "peak_memory_measurement_method", "measured_peak_rss_bytes",
                "measured_peak_gtt_bytes", "measured_peak_uma_bytes",
            )}
            resources["server_host_pid"] = 1234
            hard_gate = {"passed": True}
            memory_gate = {"passed": True}
            def timing_rows(spec_ran: bool) -> list[dict]:
                values = [{"kind": "metadata", "server_pid_source": "explicit",
                           "container_pid": 1234}]
                for index, value in enumerate(row["prefill_tps_samples"]):
                    values.append({"kind": "request", "ok": True,
                                   "group": "prefill-2048", "repeat": index,
                                   "evaluated_prefill_tokens": 2074,
                                   "prefill_ms": 2074 * 1000.0 / value,
                                   "prefill_tokens_per_second": value,
                                   "declared_prefill_tokens_per_second": round(value, 1),
                                   "prefill_tps_rounding_consistent": True})
                for index, value in enumerate(row["decode_tps_samples"]):
                    values.append({"kind": "request", "ok": True,
                                   "group": "decode-256", "repeat": index,
                                   "completion_tokens": 256,
                                   "decode_ms": 256 * 1000.0 / value,
                                   "decode_tokens_per_second": value,
                                   "declared_decode_tokens_per_second": round(value, 2),
                                   "decode_tps_rounding_consistent": True,
                                   "spec_ran": spec_ran,
                                   "accept_rate": 0.5 if spec_ran else None})
                values.append({"kind": "summary", "resources": resources,
                               "hard_gate": hard_gate, "memory_gate": memory_gate})
                return values

            target_timing = put_jsonl("target-timing.jsonl", timing_rows(False))
            target_summary = put("target-summary.json", {
                "hard_gate_observation": hard_gate, "resources": resources,
                "memory_gate": memory_gate,
            })
            target = put("target.json", {
                "schema": "ember.qwen3.8.target-only-gate.v1", "passed": True,
                "release_approval": False, "publishes": False,
                "model": {"build_record_sha256": row["build_record_sha256"]},
                "evidence": {"timing": target_timing, "summary": target_summary},
            })
            hardware_timing = put_jsonl("hardware-timing.jsonl", timing_rows(True))
            differential = put("differential.json", {
                "ok": True, "snapshot_ok": True,
                "spec": {"checked": True, "exact": True, "accept_rate": 0.5},
            })
            memory = put("memory.json", {"resources": resources,
                                          "performance": hard_gate,
                                          "hard_fit": memory_gate})
            hardware = put("hardware.json", {
                "schema": "ember.qwen3.8.real-weight-gate.v2", "publish_approved": False,
                "certification_scope": "measurement_only_not_certified",
                "model": {"ordered_inventory": {"shards": [shard]}},
                "mtp": {"path": str(mtp_path), "sha256": "7" * 64,
                        "depth": row["mtp_depth"]},
                "resources": resources,
                "hard_gates": {"performance": hard_gate, "memory": memory_gate},
                "evidence": {"quant_build_record": {
                    "path": "build.json", "sha256": row["build_record_sha256"]},
                    "timing": hardware_timing, "differential": differential,
                    "memory": memory},
            })
            manifest = {
                "schema": qb.RESULT_SCHEMA,
                "candidate_id": row["candidate_id"], "status": "complete", "publishes": False,
                "provenance": {
                    "quantization_arm": row["quantization_arm"],
                    "override_sha256": row["quantization_overrides_sha256"],
                    "intervention_configuration_id": row["intervention_configuration_id"],
                    "intervention_manifest_sha256": row["intervention_manifest_sha256"],
                    "build_record_sha256": row["build_record_sha256"],
                    "companion_inventory_sha256": row["companion_inventory_sha256"],
                    "mtp_matrix_quant_contract": row["mtp_matrix_quant_contract"],
                    "mtp_depth": row["mtp_depth"],
                    "profile_sha256": row["profile_sha256"],
                    "builder_identity": row["builder_identity"],
                    "runtime_identity": row["runtime_identity"],
                    "tensor_format_compatibility_sha256": row[
                        "tensor_format_compatibility_sha256"],
                },
                "artifacts": {
                    "model_artifact_bytes": row["artifact_bytes"],
                    "mtp": {"path": str(mtp_path), "sha256": "7" * 64,
                            "bytes": row["companion_artifact_bytes"]["mtp"]},
                    "vision_mmproj": {"path": str(mmproj_path), "sha256": "8" * 64,
                                      "bytes": row["companion_artifact_bytes"]["vision_mmproj"]},
                    "vision_vocab": {"path": str(vocab_path), "sha256": "9" * 64,
                                     "bytes": row["companion_artifact_bytes"]["vision_vocab"]},
                    "combined_fits": True,
                },
                "measurement_contract": {
                    "prefill_statistic": "peak", "decode_statistic": "median",
                    "evaluated_prefill_tokens": row["evaluated_prefill_tokens"],
                    "completion_tokens": row["completion_tokens"], "mtp_spec_ran": row["mtp_spec_ran"],
                    "mtp_depth": row["mtp_depth"],
                },
                "performance": {"prefill_tps_samples": row["prefill_tps_samples"],
                                "decode_tps_samples": row["decode_tps_samples"],
                                "resources": resources},
                "evidence": {"target_complete": target,
                             "matching_mtp_hardware_measurement": hardware,
                             "matching_mtp_timing": hardware_timing, "quality_contract": None},
            }
            manifest_desc = put("result.json", manifest)
            row["evidence_manifest"] = {**manifest_desc, "schema": manifest["schema"]}
            actual = qb.validate_measurement_evidence(row)
            self.assertEqual(actual["manifest_sha256"], manifest_desc["sha256"])
            row["mtp_depth"] = 4
            with self.assertRaisesRegex(qb.BakeoffError, "provenance differs|different MTP"):
                qb.validate_measurement_evidence(row)
            row["mtp_depth"] = 3
            row["prefill_tps_samples"][0] += 1.0
            with self.assertRaisesRegex(qb.BakeoffError, "independently derived"):
                qb.validate_measurement_evidence(row)


if __name__ == "__main__":
    unittest.main()
