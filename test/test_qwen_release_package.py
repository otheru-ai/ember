#!/usr/bin/env python3
"""Offline tests for Qwen Hugging Face release-package generation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_release_package.py"
PROFILE = ROOT / "share" / "release_profiles" / "qwen3.8-flash-next-rocmi4-strix-halo.json"
SPEC = importlib.util.spec_from_file_location("qwen_release_package", SCRIPT)
assert SPEC and SPEC.loader
qwen_release_package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qwen_release_package)
ENGINE_REVISION = "1" * 40
CONTAINER_IMAGE = "ghcr.io/otheru/ember@sha256:" + "2" * 64
MTP_NAME = "Qwen3.8-Flash-Next-MTP-ROCmI4-Strix-Halo.gguf"


def write_mmproj_fixture(path: Path, mutation: str | None = None) -> None:
    metadata = [
        ("general.architecture", 8, "clip"),
        ("general.file_type", 4, 32),
        ("clip.projector_type", 8, "qwen3vl_merger"),
        ("clip.has_vision_encoder", 7, True),
        ("clip.vision.projection_dim", 4, 2560),
        ("clip.vision.spatial_merge_size", 4, 2),
    ]
    tensors = [dict(name=row["name"], shape=list(row["shape"]))
               for row in qwen_release_package.vision_inventory.load_contract()["tensors"]]
    if mutation == "missing":
        tensors.pop()
    elif mutation == "duplicate":
        tensors[-1]["name"] = tensors[0]["name"]
    elif mutation == "wrong_shape":
        tensors[0]["shape"][0] += 1
    elif mutation is not None:
        raise ValueError(f"unknown mmproj fixture mutation: {mutation}")
    payload = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))

    def add_string(value: str) -> None:
        encoded = value.encode()
        payload.extend(struct.pack("<Q", len(encoded)))
        payload.extend(encoded)

    for key, kind, value in metadata:
        add_string(key)
        payload.extend(struct.pack("<I", kind))
        if kind == 8:
            add_string(str(value))
        elif kind == 4:
            payload.extend(struct.pack("<I", value))
        elif kind == 7:
            payload.extend(struct.pack("<?", value))
    for tensor in tensors:
        add_string(tensor["name"])
        payload.extend(struct.pack("<I", len(tensor["shape"])))
        payload.extend(struct.pack("<" + "Q" * len(tensor["shape"]),
                                   *tensor["shape"]))
        payload.extend(struct.pack("<IQ", 30, 0))
    payload.extend(b"\0" * ((-len(payload)) % 32))
    payload.extend(b"\0\0")
    path.write_bytes(payload)


class QwenReleasePackageTests(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        arguments = list(args)
        if "--mtp" not in arguments and "--build-record" in arguments:
            build_record = Path(arguments[arguments.index("--build-record") + 1])
            mtp = build_record.parent / MTP_NAME
            arguments.extend(("--mtp", str(mtp), "--mtp-sha256",
                              hashlib.sha256(mtp.read_bytes()).hexdigest()))
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def synthetic_inputs(self, directory: Path) -> tuple[Path, Path, Path, Path]:
        artifact = directory / "model.gguf"
        artifact.write_bytes(b"small deterministic GGUF fixture\n")
        license_path = directory / "upstream-license.txt"
        license_path.write_text("Qwen test license fixture\n", encoding="utf-8")
        profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        profile["source"]["license"]["sha256"] = hashlib.sha256(license_path.read_bytes()).hexdigest()
        profile_path = directory / "profile.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        mmproj = directory / profile["artifact"]["required_companion_artifacts"][0]["filename"]
        write_mmproj_fixture(mmproj)
        (directory / MTP_NAME).write_bytes(b"selected deterministic MTP fixture\n")
        target_names = ["blk.0.ssm_out.weight"]
        target_names_sha = hashlib.sha256("\n".join(target_names).encode()).hexdigest()
        direction_values = [1.0]
        direction_sha = hashlib.sha256(struct.pack("<f", *direction_values)).hexdigest()
        intervention_manifest = directory / profile["intervention"]["manifest_filename"]
        intervention_manifest.write_text(json.dumps({
            "schema_version": 1, "kind": "directional_ablation", "status": "complete",
            "weight_intervention": True, "prompt_only": False,
            "application_stage": "pre_quantization_encoding",
            "source": {
                "repo_id": profile["source"]["repo_id"],
                "revision": profile["source"]["revision"],
                "snapshot_inventory_sha256": profile["source"]["snapshot_inventory_sha256"],
            },
            "tooling": {
                "otheru_quant_pipeline": profile["intervention"]["otheru_pipeline"],
                "upstream_heretic": profile["intervention"]["upstream_heretic"],
            },
            "corpora": [{
                "id": "fixture-refusal-pairs", "role": "direction_extraction",
                "sha256": "7" * 64, "record_count": 2,
                "held_out_evaluation_overlap_count": 0,
            }],
            "directions": [{
                "id": "refusal-r1", "dtype": "F32", "values": direction_values,
                "sha256": direction_sha,
            }],
            "targets": [{
                "tensor_name": target_names[0], "direction_id": "refusal-r1",
                "scale": 1.0, "normalization": "row_norm_preserve",
                "expected_shape": [160, 1],
            }],
            "tensor_map": {
                "kind": "exact_tensor_names", "target_count": 1,
                "target_names_sha256": target_names_sha,
            },
        }), encoding="utf-8")
        intervention_sha = hashlib.sha256(intervention_manifest.read_bytes()).hexdigest()
        intervention_report = {
            "manifest_sha256": intervention_sha,
            "target_names_sha256": target_names_sha,
            "target_count": 1,
            "targets": target_names,
            "validated": True,
        }
        intervention_metrics = [{
            "tensor_name": target_names[0],
            "source_projection_l2": 1.0,
            "stored_projection_l2": 0.2,
            "stored_projection_ratio": 0.2,
            "signed_projection_coefficient": -0.8,
            "relative_frobenius_delta": 0.05,
            "row_norm_relative_rmse": 0.001,
            "row_norm_relative_max": 0.002,
        }]
        build_record = directory / "build-record.json"
        artifact_size = artifact.stat().st_size
        memory_gate = profile["quantization"]["native_262k_memory_gate"]
        reserve = memory_gate["runtime_reserve_bytes"]
        budget = memory_gate["device_budget_bytes"]
        build_record.write_text(json.dumps({
            "schema_version": 1, "status": "complete", "mode": "execute",
            "publishes": False, "credentials_accessed": False,
            "compute_mode": "exact_dequant", "w4a4_enabled": False,
            "intervention": {
                "manifest_filename": profile["intervention"]["manifest_filename"],
                "manifest_source_path": str(intervention_manifest),
                "manifest_sha256": intervention_sha,
                "kind": "directional_ablation",
                "application_stage": "pre_quantization_encoding",
                "weight_intervention": True,
                "prompt_only": False,
                "corpus_count": 1,
                "corpus_record_count": 2,
                "direction_count": 1,
                "target_count": 1,
                "targets": target_names,
                "target_names_sha256": target_names_sha,
                "tooling": {
                    "otheru_quant_pipeline": profile["intervention"]["otheru_pipeline"],
                    "upstream_heretic": profile["intervention"]["upstream_heretic"],
                },
                "quantizer_preflight": {**intervention_report, "applied": False},
                "quantizer_application": {
                    **intervention_report, "applied": True,
                    "metrics": intervention_metrics,
                },
            },
            "profile": {"sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest()},
            "snapshot_inventory": {"sha256": profile["source"]["snapshot_inventory_sha256"]},
            "snapshot": {"revision": profile["source"]["revision"]},
            "tools": {
                "ember_revision": ENGINE_REVISION,
                "llama_cpp_revision": profile["conversion"]["revision"],
                "llama_cpp_base_revision": profile["conversion"]["base_revision"],
                "rocmfpx_revision": profile["quantizer"]["revision"],
                "quantizer_build_info": {
                    "tool": profile["quantization"]["tool"],
                    "ember_revision": ENGINE_REVISION,
                    "rocmfpx_revision": profile["quantizer"]["revision"],
                    "format": profile["quantization"]["format"],
                    "ggml_tensor_type": profile["quantization"]["ggml_tensor_type"],
                    "intervention_manifest_schema": 1,
                },
            },
            "native_262k_memory_gate": memory_gate,
            "resources": {
                "minimum_free_gib": 1152,
                "minimum_ram_gib": 256,
                "free_disk_bytes": 1152 * 1024**3,
                "physical_ram_bytes": 256 * 1024**3,
            },
            "memory_preflight": {
                "artifact_bytes": artifact_size,
                "shard_count": 1,
                "shard_bytes": [artifact_size],
                "runtime_reserve_bytes": reserve,
                "budget_bytes": budget,
                "total_bytes": artifact_size + reserve,
                "headroom_bytes": budget - artifact_size - reserve,
                "fits": True,
            },
            "output": {"shards": [{
                "path": str(directory / profile["artifact"]["filename"]),
                "size_bytes": artifact_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }], "tensor_count": 1, "tensor_names_sha256": "a" * 64,
                "tensor_type_counts": {"108": 1}},
            "staging_transaction": {
                "boundary": "atomic_directory",
                "commit_method": "renameat2(RENAME_NOREPLACE)",
                "committed_directory": str(directory.resolve()),
                "same_filesystem": True,
                "verified_before_promotion": True,
                "publication_state": "committed_on_visibility",
                "promoted": [str(directory.resolve() / profile["artifact"]["filename"])],
                "evidence_promoted": [str(directory.resolve() / profile["intervention"]["manifest_filename"])],
            },
        }), encoding="utf-8")
        return profile_path, artifact, license_path, build_record

    def test_checked_in_profile_pins_audited_inputs(self) -> None:
        profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        self.assertEqual(profile["source"]["repo_id"], "Qwen/Qwen3.8-Flash-Next")
        self.assertEqual(profile["source"]["revision"], "f5d08274bafd880402bd16f5e3e6c514136ec06c")
        self.assertEqual(profile["source"]["architecture"], "Qwen4ExpForConditionalGeneration")
        self.assertEqual(profile["quantizer"]["revision"], "c49ebdbd5c9f01ec242369f9e7f7967855f80cba")
        self.assertEqual(profile["quantizer"]["format"], "Q4_0_ROCMI4")
        self.assertFalse(profile["quantizer"]["w4a4_default"])
        intervention = profile["intervention"]
        self.assertTrue(intervention["required"])
        self.assertEqual(intervention["kind"], "directional_ablation")
        self.assertFalse(intervention["prompt_only_allowed"])
        self.assertEqual(
            intervention["otheru_pipeline"]["revision"],
            "a3c6a728510f91394e991504951ac316cd3a89af",
        )
        self.assertEqual(
            intervention["upstream_heretic"]["revision"],
            "bedb94ef117a271532ac2058447fbc165d5051bd",
        )
        self.assertIn("Heretic", profile["artifact"]["repo_id"])
        memory_gate = profile["quantization"]["native_262k_memory_gate"]
        self.assertEqual(memory_gate["native_context_tokens"], 262144)
        self.assertEqual(memory_gate["device_budget_bytes"], 128 * 1024**3)
        self.assertEqual(memory_gate["runtime_reserve_bytes"], 32 * 1024**3)
        self.assertEqual(
            memory_gate["rule"],
            "artifact_bytes + runtime_reserve_bytes <= device_budget_bytes",
        )
        self.assertTrue(memory_gate["yarn_1m_math_oracle_passed"])
        self.assertFalse(memory_gate["yarn_1m_runtime_certified"])
        self.assertFalse(memory_gate["yarn_1m_fit_claim"])
        vision = profile["quantization"]["vision_artifact"]
        self.assertEqual(vision["layout"], "separate_mmproj_gguf")
        self.assertEqual(vision["storage_format"], "BF16")
        self.assertTrue(vision["runtime_provider"]["lazy_load"])
        self.assertEqual(vision["runtime_provider"]["text_model_view"], "vocab_only")
        self.assertFalse(vision["runtime_provider"]["duplicates_text_tensor_weights"])
        self.assertFalse(vision["runtime_provider"]["real_weight_differential_certified"])
        baselines = {item["id"]: item for item in profile["external_comparison_baselines"]}
        official = baselines["official-qwen-serving-recipe"]
        self.assertEqual(official["settings"]["max_model_len"], 262144)
        self.assertEqual(official["settings"]["num_speculative_tokens"], 3)
        self.assertFalse(official["ember_claim"])
        tokenspeed = baselines["tokenspeed-optional-hf-overrides"]
        self.assertTrue(tokenspeed["approximation"])
        self.assertFalse(tokenspeed["release_default"])
        unsloth = baselines["unsloth-qwen3.8-next-quant-comparison"]
        self.assertEqual(unsloth["source"], "https://unsloth.ai/docs/models/qwen3.8-next.md")
        self.assertEqual(
            unsloth["results"][0],
            {"quant": "UD-Q4_K_XL", "size_gb": 111.3, "kld": 0.044715,
             "same_top_percent": 93.481},
        )
        self.assertEqual(unsloth["results"][-1]["same_top_percent"], 80.239)
        self.assertFalse(unsloth["ember_claim"])
        lean = baselines["kingjones777-rocmfp4-strix-lean"]
        self.assertEqual(lean["revision"], "dec9c5c1053ef814cfaa39b342efd4cdd721ef0b")
        self.assertEqual(lean["aggregate_artifact_bytes"], 105753530752)
        self.assertEqual(len(lean["artifacts"]), 3)
        self.assertEqual(
            lean["artifacts"][0]["sha256"],
            "50830ad914a60eaf42e1722449c14b572866a934d3f6e2e44fb56f8c01b6adf0",
        )
        self.assertFalse(lean["native_262k_gate"]["fits"])
        self.assertFalse(lean["metadata_audit"]["weights_downloaded"])
        strix = baselines["kingjones777-rocmfp4-strix"]
        self.assertEqual(strix["revision"], "976378158e6005da4152e98eb672f71f8bd5265c")
        self.assertEqual(strix["aggregate_artifact_bytes"], 121838036032)
        self.assertEqual(
            strix["artifacts"][0]["sha256"],
            "c6770d7442a06bf1d78edf28cec83e1ec93afdd34664c23ff898807b6b9349fa",
        )
        self.assertFalse(strix["native_262k_gate"]["fits"])
        self.assertFalse(strix["ember_compatibility"]["rocm_i4_or_w4a4_evidence"])
        q1_memory = profile["release"]["q1_correctness_memory"]
        self.assertEqual(q1_memory["native_cache_bytes"], 14495514624)
        self.assertEqual(q1_memory["gdn_state_bytes"], 117669888)
        self.assertEqual(q1_memory["accounted_total_bytes"], 23203119104)
        self.assertTrue(q1_memory["copy_on_write_accounting"])
        self.assertFalse(q1_memory["performance_claim"])
        runner = profile["release"]["conversion_runner_requirements"]
        self.assertEqual(runner["minimum_free_disk_gib"], 1152)
        self.assertEqual(runner["minimum_physical_ram_gib"], 256)
        layout = profile["release"]["artifact_layout_gate"]
        self.assertTrue(layout["current_quantizer_multi_shard_supported"])
        self.assertEqual(layout["default_split_max_size"], "48G")
        self.assertTrue(layout["aggregate_preflight_before_output"])
        self.assertTrue(layout["transactional_no_clobber_shards"])
        self.assertTrue(layout["private_same_filesystem_verification_staging"])
        self.assertTrue(layout["atomic_committed_work_directory"])
        self.assertEqual(layout["directory_commit_method"], "renameat2(RENAME_NOREPLACE)")
        self.assertFalse(layout["conditional_rollback_unlink"])
        self.assertTrue(layout["build_record_atomic_no_clobber"])
        self.assertTrue(layout["package_atomic_no_clobber_directory"])
        self.assertFalse(layout["single_large_file_roundtrip_hash_tested"])
        self.assertFalse(layout["publication_blocked"])
        provisional = {item["id"]: item for item in profile["provisional_runtime_provenance"]}
        self.assertEqual(
            provisional["halospeckv-accepted-prefix-replay"]["revision"],
            "60ff854bdc25e27ee211ac0c4df896e9379edd3f",
        )
        self.assertFalse(provisional["llama-cpp-qwen4exp-base"]["release_evidence"])
        self.assertEqual(
            profile["source"]["license"]["sha256"],
            "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08",
        )

    def test_generates_metadata_checksums_and_nonpublishing_plan(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
            out = tmp / "candidate"
            result = self.run_script(
                "--profile", str(profile),
                "--artifact", str(artifact),
                "--license", str(license_path),
                "--build-record", str(build_record),
                "--engine-revision", ENGINE_REVISION,
                "--container-image", CONTAINER_IMAGE,
                "--created-at", "2026-08-26T12:00:00Z",
                "--out-dir", str(out),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            artifact_name = json.loads(profile.read_text())["artifact"]["filename"]
            mmproj_name = "Qwen3.8-Flash-Next-BF16-mmproj.gguf"
            for name in ("README.md", "LICENSE", "artifact-manifest.json", "SHA256SUMS", "upload-plan.json", "qwen-quant-build-record.json", "qwen-intervention-manifest.json", "release-profile.json", artifact_name, MTP_NAME, mmproj_name):
                self.assertTrue((out / name).is_file(), name)
            manifest = json.loads((out / "artifact-manifest.json").read_text(encoding="utf-8"))
            plan = json.loads((out / "upload-plan.json").read_text(encoding="utf-8"))
            card = (out / "README.md").read_text(encoding="utf-8")
            self.assertFalse(plan["publishes"])
            self.assertFalse(plan["authentication"]["token_embedded"])
            self.assertEqual(manifest["certification"]["status"], "pending")
            self.assertEqual(manifest["artifact"]["sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest())
            self.assertTrue(manifest["native_262k_memory_preflight"]["fits"])
            self.assertEqual(manifest["q1_correctness_memory"]["native_cache_bytes"], 14495514624)
            self.assertFalse(manifest["q1_correctness_memory"]["performance_claim"])
            self.assertEqual(manifest["conversion_runner_requirements"]["minimum_physical_ram_gib"], 256)
            self.assertFalse(manifest["artifact_layout_gate"]["publication_blocked"])
            self.assertTrue(manifest["intervention"]["weight_intervention"])
            self.assertFalse(manifest["intervention"]["prompt_only"])
            self.assertTrue(manifest["intervention"]["quantizer_applied"])
            companions = {row["role"]: row for row in manifest["companion_artifacts"]}
            self.assertEqual(set(companions), {"mtp", "vision_mmproj"})
            self.assertEqual(companions["mtp"]["filename"], MTP_NAME)
            self.assertEqual(
                companions["vision_mmproj"]["inspection"]["metadata"]["general.file_type"],
                32,
            )
            self.assertEqual(
                companions["vision_mmproj"]["inspection"][
                    "tensor_inventory_sha256"],
                qwen_release_package.vision_inventory.load_contract()[
                    "tensor_inventory_sha256"],
            )
            self.assertFalse(
                manifest["certification"]["vision"]["real_weight_differential_certified"]
            )
            self.assertEqual(plan["publication_blockers"], [])
            self.assertIn("license: other", card)
            self.assertIn("base_model: Qwen/Qwen3.8-Flash-Next", card)
            self.assertIn("exact-dequant", card)
            self.assertIn("W4A4", card)
            self.assertIn("128 GiB UMA", card)
            self.assertIn("real-weight target runtime is not certified", card)
            self.assertIn("ordered 48G split layout", card)
            self.assertIn("not** a prompt-only jailbreak", card)
            self.assertIn("weights were changed", card)
            self.assertEqual((out / "LICENSE").read_bytes(), license_path.read_bytes())
            destinations = {entry["path_in_repo"] for entry in plan["files"]}
            self.assertIn("Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo.gguf", destinations)
            self.assertIn("qwen-intervention-manifest.json", destinations)
            self.assertIn(MTP_NAME, destinations)
            self.assertIn(mmproj_name, destinations)
            self.assertNotIn("upload-plan.json", destinations)
            self.assertFalse(plan["authentication"]["token_embedded"])
            checksum_names = [line.split("  ", 1)[1] for line in
                              (out / "SHA256SUMS").read_text().splitlines()]
            self.assertEqual(checksum_names, [artifact_name, MTP_NAME, mmproj_name])
            self.assertTrue(all(Path(name).name == name for name in checksum_names))
            integrity = manifest["model_artifact_integrity"]
            self.assertEqual(integrity["ordered_filenames"], checksum_names)
            self.assertEqual(integrity["entry_count"], 3)
            self.assertTrue(integrity["basenames_only"])
            self.assertEqual(integrity["sha256"], hashlib.sha256(
                (out / "SHA256SUMS").read_bytes()).hexdigest())

    def test_sha256sums_rejects_omissions_duplicates_and_unsafe_names(self) -> None:
        digest = "a" * 64
        cases = (
            ([(digest, "main.gguf"), (digest, "mtp.gguf")],
             ["main.gguf", "mtp.gguf", "mmproj.gguf"], "every selected"),
            ([(digest, "main.gguf"), (digest, "main.gguf"),
              (digest, "mmproj.gguf")],
             ["main.gguf", "main.gguf", "mmproj.gguf"], "unique"),
            ([(digest, "main.gguf"), (digest, "../mtp.gguf"),
              (digest, "mmproj.gguf")],
             ["main.gguf", "../mtp.gguf", "mmproj.gguf"], "safe basename"),
        )
        for entries, required, expected in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(
                qwen_release_package.PackageError, expected
            ):
                qwen_release_package.render_selected_sha256sums(entries, required)

    def test_rejects_selected_mtp_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
            result = self.run_script(
                "--profile", str(profile), "--artifact", str(artifact),
                "--license", str(license_path), "--build-record", str(build_record),
                "--mtp", str(tmp / MTP_NAME), "--mtp-sha256", "0" * 64,
                "--engine-revision", ENGINE_REVISION,
                "--container-image", CONTAINER_IMAGE, "--out-dir", str(tmp / "out"),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("selected MTP SHA-256 differs", result.stderr)

    def test_rejects_selected_mtp_with_undeployable_name(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
            source = tmp / MTP_NAME
            renamed = tmp / "mtp.gguf"
            source.rename(renamed)
            result = self.run_script(
                "--profile", str(profile), "--artifact", str(artifact),
                "--license", str(license_path), "--build-record", str(build_record),
                "--mtp", str(renamed), "--mtp-sha256",
                hashlib.sha256(renamed.read_bytes()).hexdigest(),
                "--engine-revision", ENGINE_REVISION,
                "--container-image", CONTAINER_IMAGE, "--out-dir", str(tmp / "out"),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Qwen3.8-Flash-Next deployment contract", result.stderr)

    def test_rejects_missing_or_non_qwen_bf16_mmproj(self) -> None:
        for mutation, expected in (("missing", "cannot inspect package input"),
                                   ("wrong_type", "pinned Qwen3.8 BF16 tower")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw_tmp:
                tmp = Path(raw_tmp)
                profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
                mmproj = tmp / "Qwen3.8-Flash-Next-BF16-mmproj.gguf"
                if mutation == "missing":
                    mmproj.unlink()
                else:
                    raw = bytearray(mmproj.read_bytes())
                    marker = b"general.file_type"
                    at = raw.index(marker) + len(marker) + 4
                    raw[at:at + 4] = struct.pack("<I", 1)
                    mmproj.write_bytes(raw)
                result = self.run_script(
                    "--profile", str(profile), "--artifact", str(artifact),
                    "--license", str(license_path), "--build-record", str(build_record),
                    "--engine-revision", ENGINE_REVISION, "--container-image", CONTAINER_IMAGE,
                    "--out-dir", str(tmp / "out"),
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)

    def test_rejects_inexact_mmproj_tensor_inventory(self) -> None:
        for mutation, expected in (("missing", "count mismatch"),
                                   ("duplicate", "duplicate"),
                                   ("wrong_shape", "shape mismatch")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw_tmp:
                tmp = Path(raw_tmp)
                profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
                mmproj = tmp / "Qwen3.8-Flash-Next-BF16-mmproj.gguf"
                write_mmproj_fixture(mmproj, mutation)
                result = self.run_script(
                    "--profile", str(profile), "--artifact", str(artifact),
                    "--license", str(license_path), "--build-record", str(build_record),
                    "--engine-revision", ENGINE_REVISION,
                    "--container-image", CONTAINER_IMAGE,
                    "--out-dir", str(tmp / "out"),
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)

    def test_rejects_wrong_license_copy(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
            license_path.write_text("modified after pinning\n", encoding="utf-8")
            result = self.run_script(
                "--profile", str(profile),
                "--artifact", str(artifact),
                "--license", str(license_path),
                "--build-record", str(build_record),
                "--engine-revision", ENGINE_REVISION,
                "--container-image", CONTAINER_IMAGE,
                "--out-dir", str(tmp / "out"),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("license SHA-256 does not match", result.stderr)

    def test_rejects_mutable_container_tag(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
            result = self.run_script(
                "--profile", str(profile),
                "--artifact", str(artifact),
                "--license", str(license_path),
                "--build-record", str(build_record),
                "--engine-revision", ENGINE_REVISION,
                "--container-image", "ghcr.io/otheru/ember:latest",
                "--out-dir", str(tmp / "out"),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("immutable", result.stderr)

    def test_rejects_build_record_without_passing_memory_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
            record = json.loads(build_record.read_text(encoding="utf-8"))
            record["memory_preflight"]["fits"] = False
            build_record.write_text(json.dumps(record), encoding="utf-8")
            result = self.run_script(
                "--profile", str(profile), "--artifact", str(artifact),
                "--license", str(license_path), "--build-record", str(build_record),
                "--engine-revision", ENGINE_REVISION, "--container-image", CONTAINER_IMAGE,
                "--out-dir", str(tmp / "out"),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("native-262K memory gate", result.stderr)

    def test_rejects_stock_quant_or_prompt_only_record_under_heretic_name(self) -> None:
        for mutation, expected in (("missing", "build_record.intervention"), ("prompt", "Heretic intervention evidence")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw_tmp:
                tmp = Path(raw_tmp)
                profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
                record = json.loads(build_record.read_text())
                if mutation == "missing":
                    del record["intervention"]
                else:
                    record["intervention"]["weight_intervention"] = False
                    record["intervention"]["prompt_only"] = True
                build_record.write_text(json.dumps(record))
                result = self.run_script(
                    "--profile", str(profile), "--artifact", str(artifact),
                    "--license", str(license_path), "--build-record", str(build_record),
                    "--engine-revision", ENGINE_REVISION, "--container-image", CONTAINER_IMAGE,
                    "--out-dir", str(tmp / "out"),
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)

    def test_rejects_changed_applied_intervention_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
            intervention_path = tmp / "qwen-intervention-manifest.json"
            manifest = json.loads(intervention_path.read_text())
            manifest["targets"][0]["scale"] = 2.0
            intervention_path.write_text(json.dumps(manifest))
            result = self.run_script(
                "--profile", str(profile), "--artifact", str(artifact),
                "--license", str(license_path), "--build-record", str(build_record),
                "--engine-revision", ENGINE_REVISION, "--container-image", CONTAINER_IMAGE,
                "--out-dir", str(tmp / "out"),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("manifest SHA-256 differs", result.stderr)

    def test_rejects_missing_or_nonfinite_intervention_metrics(self) -> None:
        for mutation, expected in (("missing", "per-target intervention metrics"), ("nonfinite", "must be finite")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw_tmp:
                tmp = Path(raw_tmp)
                profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
                record = json.loads(build_record.read_text())
                if mutation == "missing":
                    del record["intervention"]["quantizer_application"]["metrics"]
                else:
                    record["intervention"]["quantizer_application"]["metrics"][0][
                        "stored_projection_ratio"
                    ] = "NaN"
                build_record.write_text(json.dumps(record))
                result = self.run_script(
                    "--profile", str(profile), "--artifact", str(artifact),
                    "--license", str(license_path), "--build-record", str(build_record),
                    "--engine-revision", ENGINE_REVISION, "--container-image", CONTAINER_IMAGE,
                    "--out-dir", str(tmp / "out"),
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)

    def test_rejects_build_record_from_underprovisioned_conversion_runner(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
            record = json.loads(build_record.read_text(encoding="utf-8"))
            record["resources"]["minimum_ram_gib"] = 128
            build_record.write_text(json.dumps(record), encoding="utf-8")
            result = self.run_script(
                "--profile", str(profile), "--artifact", str(artifact),
                "--license", str(license_path), "--build-record", str(build_record),
                "--engine-revision", ENGINE_REVISION, "--container-image", CONTAINER_IMAGE,
                "--out-dir", str(tmp / "out"),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("release runner resource floors", result.stderr)

    def test_accepts_complete_ordered_gguf_shards(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            profile, _artifact, license_path, single_build_record = self.synthetic_inputs(tmp)
            first = tmp / "Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-00001-of-00002.gguf"
            second = tmp / "Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-00002-of-00002.gguf"
            first.write_bytes(b"shard one")
            second.write_bytes(b"shard two")
            artifact_size = first.stat().st_size + second.stat().st_size
            profile_data = json.loads(profile.read_text(encoding="utf-8"))
            memory_gate = profile_data["quantization"]["native_262k_memory_gate"]
            reserve = memory_gate["runtime_reserve_bytes"]
            budget = memory_gate["device_budget_bytes"]
            build_record = tmp / "qwen-quant-build-record.json"
            record = json.loads(single_build_record.read_text())
            record["memory_preflight"].update({
                "artifact_bytes": artifact_size,
                "shard_count": 2,
                "shard_bytes": [first.stat().st_size, second.stat().st_size],
                "total_bytes": artifact_size + reserve,
                "headroom_bytes": budget - artifact_size - reserve,
            })
            record["output"] = {
                "shards": [
                    {"path": str(first), "size_bytes": first.stat().st_size,
                     "sha256": hashlib.sha256(first.read_bytes()).hexdigest()},
                    {"path": str(second), "size_bytes": second.stat().st_size,
                     "sha256": hashlib.sha256(second.read_bytes()).hexdigest()},
                ],
                "tensor_count": 2, "tensor_names_sha256": "b" * 64,
                "tensor_type_counts": {"108": 2},
            }
            record["staging_transaction"]["promoted"] = [str(first.resolve()), str(second.resolve())]
            build_record.write_text(json.dumps(record), encoding="utf-8")
            out = tmp / "candidate"
            result = self.run_script(
                "--profile", str(profile), "--artifact", str(first), "--artifact", str(second),
                "--license", str(license_path), "--engine-revision", ENGINE_REVISION,
                "--container-image", CONTAINER_IMAGE, "--build-record", str(build_record),
                "--out-dir", str(out),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "artifact-manifest.json").read_text())
            plan = json.loads((out / "upload-plan.json").read_text())
            self.assertEqual(len(manifest["artifacts"]), 2)
            self.assertEqual(manifest["artifact"]["kind"], "ordered_shard_set")
            self.assertIsNone(manifest["artifact"]["filename"])
            self.assertIsNone(manifest["artifact"]["sha256"])
            self.assertEqual(manifest["artifact"]["size_bytes"], artifact_size)
            self.assertEqual(manifest["artifact"]["shard_count"], 2)
            destinations = [entry["path_in_repo"] for entry in plan["files"][:2]]
            self.assertEqual(destinations, [first.name, second.name])
            checksum_names = [line.split("  ", 1)[1] for line in
                              (out / "SHA256SUMS").read_text().splitlines()]
            self.assertEqual(checksum_names, [
                first.name, second.name, MTP_NAME,
                "Qwen3.8-Flash-Next-BF16-mmproj.gguf",
            ])
            self.assertEqual(
                manifest["model_artifact_integrity"]["ordered_filenames"],
                checksum_names,
            )
            self.assertTrue((out / "qwen-quant-build-record.json").is_file())
            self.assertIn("qwen-quant-build-record.json", {entry["path_in_repo"] for entry in plan["files"]})

    def test_rejects_build_record_profile_or_engine_revision_mismatch(self) -> None:
        for mismatch in ("profile", "engine"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as raw_tmp:
                tmp = Path(raw_tmp)
                profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
                record = json.loads(build_record.read_text(encoding="utf-8"))
                if mismatch == "profile":
                    record["profile"]["sha256"] = "0" * 64
                else:
                    record["tools"]["ember_revision"] = "3" * 40
                build_record.write_text(json.dumps(record), encoding="utf-8")
                result = self.run_script(
                    "--profile", str(profile), "--artifact", str(artifact),
                    "--license", str(license_path), "--build-record", str(build_record),
                    "--engine-revision", ENGINE_REVISION,
                    "--container-image", CONTAINER_IMAGE, "--out-dir", str(tmp / "out"),
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "profile SHA-256" if mismatch == "profile" else "tool revisions",
                    result.stderr,
                )

    def test_rejects_recorded_output_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
            record = json.loads(build_record.read_text(encoding="utf-8"))
            record["output"]["shards"][0]["size_bytes"] += 1
            build_record.write_text(json.dumps(record), encoding="utf-8")
            result = self.run_script(
                "--profile", str(profile), "--artifact", str(artifact),
                "--license", str(license_path), "--build-record", str(build_record),
                "--engine-revision", ENGINE_REVISION,
                "--container-image", CONTAINER_IMAGE, "--out-dir", str(tmp / "out"),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("files/sizes/hashes", result.stderr)

    def test_rejects_missing_transaction_or_tensor_evidence(self) -> None:
        for missing in ("staging_transaction", "tensor_type_counts"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as raw_tmp:
                tmp = Path(raw_tmp)
                profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
                record = json.loads(build_record.read_text())
                if missing == "staging_transaction":
                    del record["staging_transaction"]
                else:
                    del record["output"]["tensor_type_counts"]
                build_record.write_text(json.dumps(record))
                result = self.run_script(
                    "--profile", str(profile), "--artifact", str(artifact),
                    "--license", str(license_path), "--build-record", str(build_record),
                    "--engine-revision", ENGINE_REVISION, "--container-image", CONTAINER_IMAGE,
                    "--out-dir", str(tmp / "out"),
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "staging_transaction" if missing == "staging_transaction" else "tensor inventory/type",
                    result.stderr,
                )

    def test_existing_or_dangling_package_directory_is_never_replaced(self) -> None:
        for dangling in (False, True):
            with self.subTest(dangling=dangling), tempfile.TemporaryDirectory() as raw_tmp:
                tmp = Path(raw_tmp)
                profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
                out = tmp / "candidate"
                if dangling:
                    out.symlink_to(tmp / "missing", target_is_directory=True)
                else:
                    out.mkdir()
                    (out / "sentinel").write_bytes(b"foreign")
                result = self.run_script(
                    "--profile", str(profile), "--artifact", str(artifact),
                    "--license", str(license_path), "--build-record", str(build_record),
                    "--engine-revision", ENGINE_REVISION, "--container-image", CONTAINER_IMAGE,
                    "--out-dir", str(out),
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("existing package directory", result.stderr)
                if dangling:
                    self.assertTrue(out.is_symlink())
                else:
                    self.assertEqual((out / "sentinel").read_bytes(), b"foreign")

    def test_profile_is_validated_from_stable_snapshot_not_mutable_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            profile, artifact, license_path, build_record = self.synthetic_inputs(tmp)
            args = qwen_release_package.parse_args([
                "--profile", str(profile), "--artifact", str(artifact),
                "--license", str(license_path), "--build-record", str(build_record),
                "--mtp", str(tmp / MTP_NAME), "--mtp-sha256",
                hashlib.sha256((tmp / MTP_NAME).read_bytes()).hexdigest(),
                "--engine-revision", ENGINE_REVISION, "--container-image", CONTAINER_IMAGE,
                "--out-dir", str(tmp / "out"),
            ])
            real_copy = qwen_release_package.copy_stable_file

            def copy_then_mutate(source: Path, destination: Path) -> str:
                digest = real_copy(source, destination)
                if source == profile.resolve():
                    profile.write_text("{}", encoding="utf-8")
                return digest

            with mock.patch.object(
                qwen_release_package, "copy_stable_file", side_effect=copy_then_mutate
            ):
                plan = qwen_release_package.build_package(args)
            self.assertFalse(plan["publishes"])
            self.assertTrue((tmp / "out" / "release-profile.json").is_file())

    def test_stable_copy_rejects_source_changed_during_copy(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            source = tmp / "source"
            destination = tmp / "destination"
            source.write_bytes(b"stable input")
            real_fstat = qwen_release_package.os.fstat
            calls = 0

            def changed_fstat(descriptor: int):
                nonlocal calls
                calls += 1
                status = real_fstat(descriptor)
                if calls == 2:
                    source.write_bytes(b"changed input")
                    return real_fstat(descriptor)
                return status

            with mock.patch.object(qwen_release_package.os, "fstat", side_effect=changed_fstat):
                with self.assertRaisesRegex(qwen_release_package.PackageError, "changed while"):
                    qwen_release_package.copy_stable_file(source, destination)

    def test_script_has_no_uploader_or_network_client(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden in ("huggingface_hub", "HfApi", "urllib", "requests", "os.environ", "os.getenv"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
