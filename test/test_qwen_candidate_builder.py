#!/usr/bin/env python3
"""GPU-free lifecycle contracts for serial Qwen candidate construction."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "test"))
import qwen_candidate_builder as builder  # noqa: E402
import qwen_quantize as quant  # noqa: E402
from test_qwen_quantize import Fixture, make_gguf, make_mmproj_companion  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class CandidateBuilderTests(unittest.TestCase):
    def test_build_source_is_exactly_intervention_or_stock_capture(self) -> None:
        parser = builder.parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["build-candidate"])
        help_text = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "qwen_candidate_builder.py"),
             "build-candidate", "--help"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        ).stdout
        self.assertIn("--intervention-manifest", help_text)
        self.assertIn("--stock-control", help_text)
        self.assertIn("--stock-capture-manifest", help_text)

    def test_cache_built_stock_must_match_captured_shard_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            capture = root / "capture-manifest.json"
            rows = [{"filename": "captured-00001.gguf", "size_bytes": 11,
                     "sha256": "1" * 64}]
            write_json(capture, {
                "schema": "ember.qwen3.8.stock-control-activation-capture.v1",
                "status": "complete", "stock_rocmi4_only": True,
                "model": {"build_record_sha256": "2" * 64, "shards": rows},
            })
            record = {"output": {"shards": [
                {"path": "/different/name.gguf", "size_bytes": 11,
                 "sha256": "1" * 64},
            ]}}
            evidence = builder.validate_stock_capture_match(capture, digest(capture), record)
            self.assertTrue(evidence["byte_identical"])
            record["output"]["shards"][0]["sha256"] = "3" * 64
            with self.assertRaisesRegex(builder.BuilderError, "differ"):
                builder.validate_stock_capture_match(capture, digest(capture), record)

    def test_captured_stock_retirement_is_durable_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stock = root / "stock"
            evidence = root / "evidence"
            workset = root / "workset"
            stock.mkdir()
            shard = stock / "stock-00001.gguf"
            shard.write_bytes(b"captured-stock")
            record = stock / "qwen-quant-build-record.json"
            write_json(record, {
                "status": "complete",
                "experiment": {"kind": "stock_control", "stock_weights_unchanged": True},
                "output": {"shards": [{"path": str(shard),
                                          "size_bytes": shard.stat().st_size,
                                          "sha256": digest(shard)}]},
            })
            capture = root / "capture.json"
            write_json(capture, {
                "schema": "ember.qwen3.8.stock-control-activation-capture.v1",
                "status": "complete", "stock_rocmi4_only": True,
                "model": {"build_record_sha256": digest(record), "shards": [{
                    "filename": shard.name, "size_bytes": shard.stat().st_size,
                    "sha256": digest(shard),
                }]},
            })
            authorization = evidence / "retire-stock.json"
            result = builder.retire_captured_stock(argparse.Namespace(
                stock_dir=stock, build_record_sha256=digest(record),
                stock_capture_manifest=capture,
                stock_capture_manifest_sha256=digest(capture),
                workset_root=workset, output=authorization,
            ))
            self.assertEqual(result["deleted_bytes"], len(b"captured-stock"))
            self.assertTrue(result["recoverable"])
            self.assertFalse(shard.exists())
            self.assertTrue(record.exists())
            self.assertTrue(authorization.exists())
            self.assertTrue(Path(result["completion"]).exists())

    def test_cache_address_binds_main_and_mmproj(self) -> None:
        main = [{"name": "m-00001-of-00002.gguf", "size_bytes": 11,
                 "sha256": "1" * 64}]
        mmproj = {"name": builder.MMPROJ_BASENAME, "size_bytes": 7,
                  "sha256": "2" * 64}
        main_sha, cache_id = builder.cache_content_address(main, mmproj)
        self.assertEqual(main_sha, builder.canonical_sha256(main))
        changed = dict(mmproj, sha256="3" * 64)
        self.assertNotEqual(cache_id, builder.cache_content_address(main, changed)[1])

    def test_format_compatibility_excludes_builder_binary_and_arm_default(self) -> None:
        base = {"tool": "ember-gguf-quantize", "format": "Q4_0_ROCMI4",
                "ggml_tensor_type": 108,
                "per_tensor_formats": ["Q6_K", "Q4_0_ROCMI4"]}
        changed = dict(base, tool="renamed-builder", format="Q6_K", ggml_tensor_type=14)
        self.assertEqual(
            builder.tensor_format_contract_sha256(base, "a" * 40),
            builder.tensor_format_contract_sha256(changed, "a" * 40),
        )
        self.assertNotEqual(
            builder.tensor_format_contract_sha256(base, "a" * 40),
            builder.tensor_format_contract_sha256(base, "b" * 40),
        )

    def test_cgroup_contract_is_exact_and_no_swap(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            memory = root / "memory.max"
            swap = root / "memory.swap.max"
            peak = root / "memory.peak"
            memory.write_text(str(builder.MEMORY_LIMIT_BYTES), encoding="ascii")
            swap.write_text("0", encoding="ascii")
            peak.write_text("1234", encoding="ascii")
            args = argparse.Namespace(
                memory_limit_bytes=builder.MEMORY_LIMIT_BYTES,
                cgroup_memory_max_path=memory,
                cgroup_memory_swap_max_path=swap,
                cgroup_memory_peak_path=peak,
            )
            self.assertEqual(builder.validate_cgroup(args)["peak_before_bytes"], 1234)
            swap.write_text("1", encoding="ascii")
            with self.assertRaisesRegex(builder.BuilderError, "no-swap"):
                builder.validate_cgroup(args)

    def test_companion_inventory_binds_mtp_arm_and_cached_mmproj(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture_root = root / "fixture"
            fixture_root.mkdir()
            fixture = Fixture(fixture_root)
            fixture.companion_args(enable_mmproj=True)
            mmproj = fixture_root / builder.MMPROJ_BASENAME
            cache_manifest = fixture_root / "bf16-cache-manifest.json"
            write_json(cache_manifest, {
                "schema": quant.BF16_CACHE_SCHEMA,
                "vision_mmproj": {"name": mmproj.name,
                                  "size_bytes": mmproj.stat().st_size,
                                  "sha256": digest(mmproj)},
            })
            mtp = fixture_root / "Qwen3.8-Flash-Next-MTP.gguf"
            mtp_export = fixture_root / "Qwen3.8-Flash-Next-MTP.export.json"
            output = root / "companions.json"
            result = builder.make_companion_inventory(argparse.Namespace(
                profile=fixture.profile,
                quantization_arm=quant.DEFAULT_QUANTIZATION_ARM,
                bf16_cache_manifest=cache_manifest,
                bf16_cache_manifest_sha256=digest(cache_manifest),
                mtp=mtp, mtp_bytes=mtp.stat().st_size, mtp_sha256=digest(mtp),
                mtp_matrix_quant_contract="Q4_0_ROCMI4",
                mtp_export_manifest=mtp_export,
                mtp_export_manifest_sha256=digest(mtp_export), output=output,
            ))
            self.assertEqual(result["inventory_sha256"], digest(output))
            roles = json.loads(output.read_text(encoding="utf-8"))["companions"]
            self.assertEqual([row["role"] for row in roles], ["mtp", "vision_mmproj"])
            self.assertEqual(roles[0]["matrix_quant_contract"], "Q4_0_ROCMI4")
            self.assertEqual(roles[1]["sha256"], digest(mmproj))

    def test_bf16_cache_reuse_revalidates_content_address_and_toolchain(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture_root = root / "fixture"
            fixture_root.mkdir()
            fixture = Fixture(fixture_root)
            profile = json.loads(fixture.profile.read_text(encoding="utf-8"))
            staging = root / "staging"
            staging.mkdir()
            main = staging / builder.CACHE_BASENAME
            make_gguf(main, split_no=0, split_count=1,
                      tensor_name="per_layer_token_embd.weight", tensor_type=0)
            mmproj = staging / builder.MMPROJ_BASENAME
            make_mmproj_companion(mmproj)
            ple = quant.safetensor_ple_constants(fixture.snapshot)
            main_gguf = quant.verify_gguf_set(
                [main], ple, quantized=False, profile=profile)
            main_rows = [{"name": main.name, "size_bytes": main.stat().st_size,
                          "sha256": digest(main)}]
            mmproj_row = {"name": mmproj.name, "size_bytes": mmproj.stat().st_size,
                          "sha256": digest(mmproj), "format": "BF16",
                          "gguf": quant.validate_bf16_qwen_mmproj_gguf(mmproj)}
            main_sha, cache_id = builder.cache_content_address(main_rows, mmproj_row)
            tools = {
                "llama_cpp_revision": fixture.llama_head,
                "llama_cpp_base_revision": fixture.llama_base,
                "converter_sha256": digest(fixture.llama / "convert_hf_to_gguf.py"),
                "qwen4exp_converter_sha256": digest(
                    fixture.llama / "conversion" / "qwen4exp.py"),
                "ple_cgroup_writeback_patch_sha256": digest(fixture.ple_patch),
                "gguf_splitter_sha256": "4" * 64,
                "converter_environment_lock_sha256": "6" * 64,
                "converter_environment_lock_bytes": 123,
                "builder_container_digest": "sha256:" + "7" * 64,
            }
            manifest = {
                "schema": quant.BF16_CACHE_SCHEMA, "cache_id": cache_id,
                "source": {"repo_id": profile["source"]["repo_id"],
                           "revision": profile["source"]["revision"],
                           "snapshot_inventory_sha256": profile["source"][
                               "snapshot_inventory_sha256"]},
                "profile": {"profile_id": profile["profile_id"],
                            "sha256": digest(fixture.profile)},
                "toolchain": dict(tools),
                "conversion": {"outtype": "bf16", "split_max_size": "48G",
                               "use_temp_file": True,
                               "main_storage_policy": "mostly_bf16_with_f32_ple",
                               "ple_intermediate_storage":
                                   "F32_streamed_to_temp_file_then_release_quant_override",
                               "ple_ggml_tensor_type": 0,
                               "mmproj": {"outtype": "bf16",
                                          "converter_option": "--mmproj"},
                               "gguf_writer_temp_cleanup": {
                                   "policy": "exact_converter_private_tmp_residue_v2",
                                   "main_removed": [], "mmproj_removed": []}},
                "resources": {"free_bytes": 1152 * quant.GIB,
                              "physical_ram_bytes": 120 * quant.GIB},
                "measurement": {"status": "measured_target_cgroup_v2",
                                "memory_limit_bytes": builder.MEMORY_LIMIT_BYTES,
                                "swap_limit_bytes": 0, "cgroup_peak_bytes": 1234},
                "main": {"base_path": main.name, "content_sha256": main_sha,
                         "shards": main_rows, "ple": ple,
                         "gguf": {key: main_gguf[key] for key in (
                             "tensor_count", "tensor_names_sha256", "tensor_type_counts")}},
                "vision_mmproj": mmproj_row,
            }
            manifest_path = staging / "bf16-cache-manifest.json"
            write_json(manifest_path, manifest)
            cache = root / f"bf16-{cache_id}"
            staging.rename(cache)
            manifest_path = cache / manifest_path.name
            validated = quant.validate_bf16_cache_manifest(
                manifest_path, digest(manifest_path), profile, digest(fixture.profile), tools)
            self.assertEqual(validated["cache_id"], cache_id)
            manifest["toolchain"]["converter_sha256"] = "5" * 64
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(quant.PipelineError, "toolchain"):
                quant.validate_bf16_cache_manifest(
                    manifest_path, digest(manifest_path), profile,
                    digest(fixture.profile), tools)

    def make_candidate(self, root: Path) -> tuple[Path, Path, str, Path, str]:
        candidate = root / "candidate"
        evidence = root / "evidence"
        candidate.mkdir()
        evidence.mkdir()
        cache_dir = root / "cache" / ("bf16-" + "a" * 64)
        cache_dir.mkdir(parents=True)
        cache_manifest = cache_dir / "bf16-cache-manifest.json"
        cache_manifest.write_text("{}\n", encoding="utf-8")
        shard = candidate / "candidate-00001-of-00001.gguf"
        shard.write_bytes(b"quantized-candidate")
        record = candidate / "qwen-quant-build-record.json"
        write_json(record, {
            "status": "complete",
            "bf16_cache": {"manifest": {"path": str(cache_manifest),
                                         "sha256": digest(cache_manifest)}},
            "output": {"shards": [{"path": str(shard),
                                    "size_bytes": shard.stat().st_size,
                                    "sha256": digest(shard)}]},
        })
        assessment = evidence / "assessment.json"
        write_json(assessment, {"candidate_id": "candidate-1", "passed": True})
        return candidate, shard, digest(record), assessment, digest(assessment)

    def test_loser_deletion_requires_durable_external_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, shard, record_sha, assessment, assessment_sha = self.make_candidate(root)
            bundle_path = root / "evidence" / "candidate-1.bundle.json"
            recorded = builder.record_assessment(argparse.Namespace(
                candidate_id="candidate-1", candidate_dir=candidate,
                build_record_sha256=record_sha, assessment=assessment,
                assessment_sha256=assessment_sha, selected=False, output=bundle_path,
            ))
            planned = builder.delete_loser(argparse.Namespace(
                assessment_bundle=bundle_path,
                assessment_bundle_sha256=recorded["bundle_sha256"], execute=False,
            ))
            self.assertEqual(planned["status"], "planned")
            self.assertTrue(shard.exists())
            deleted = builder.delete_loser(argparse.Namespace(
                assessment_bundle=bundle_path,
                assessment_bundle_sha256=recorded["bundle_sha256"], execute=True,
            ))
            self.assertEqual(deleted["deleted_bytes"], len(b"quantized-candidate"))
            self.assertFalse(shard.exists())
            self.assertTrue(Path(deleted["tombstone"]).is_file())
            self.assertTrue((candidate / "qwen-quant-build-record.json").is_file())

    def test_selected_candidate_cannot_be_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, shard, record_sha, assessment, assessment_sha = self.make_candidate(root)
            bundle_path = root / "selected.bundle.json"
            recorded = builder.record_assessment(argparse.Namespace(
                candidate_id="candidate-1", candidate_dir=candidate,
                build_record_sha256=record_sha, assessment=assessment,
                assessment_sha256=assessment_sha, selected=True, output=bundle_path,
            ))
            with self.assertRaisesRegex(builder.BuilderError, "does not authorize"):
                builder.delete_loser(argparse.Namespace(
                    assessment_bundle=bundle_path,
                    assessment_bundle_sha256=recorded["bundle_sha256"], execute=True,
                ))
            self.assertTrue(shard.exists())

    def test_cli_is_dry_by_default_for_deletion(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "qwen_candidate_builder.py"),
             "delete-loser", "--help"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--execute", completed.stdout)


if __name__ == "__main__":
    unittest.main()
