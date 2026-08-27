#!/usr/bin/env python3
"""Offline tests for measured-only Qwen bakeoff selection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


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
            "companion_artifact_bytes": {"mtp": 2_000_000_000, "vision_mmproj": 901_943_132},
            "enabled_companions": ["mtp", "vision_mmproj"],
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

    def staged_selection(self, root: Path, plan: dict, corpora: Path,
                         tamper_final_build: bool = False,
                         tamper_final_depth: bool = False) -> tuple[dict, dict, dict, dict]:
        raw = self.complete_results(plan)["results"]
        sweep_rows = [row for row in raw if row["stage"] in {"stock", "sweep"}]
        format_rows = [row for row in raw if row["stage"] == "format"]
        sweep = qb.select_sweep(plan, self.assessment_input(root, plan, sweep_rows, "sweep"))
        sweep_desc = self.persist_attested(
            root, "sweep-ledger", sweep, qb.LEDGER_SCHEMA)
        selected_format = qb.select_format(
            plan, self.assessment_input(root, plan, format_rows, "format"), sweep_desc)
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
            format_results = self.assessment_input(
                root, plan, [row for row in complete if row["stage"] == "format"],
                "staged-format")
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

            row["companion_artifact_bytes"] = {"mtp": 3, "vision_mmproj": 4}
            mtp_path = root / "mtp.gguf"; mtp_path.write_bytes(b"mtp")
            mmproj_path = root / "mmproj.gguf"; mmproj_path.write_bytes(b"mmpr")
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
