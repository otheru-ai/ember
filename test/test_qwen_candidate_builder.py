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
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "test"))
import qwen_candidate_builder as builder  # noqa: E402
import qwen_quantize as quant  # noqa: E402
from test_qwen_quantize import (Fixture, make_gguf, make_mmproj_companion,
                                make_vision_vocab_companion)  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class CandidateBuilderTests(unittest.TestCase):
    def test_converter_temp_environment_is_private_and_cleanup_compatible(self) -> None:
        temp_dir = Path("/work/.converter-tmp")
        self.assertEqual(builder.converter_temp_env(temp_dir), {
            "TMPDIR": str(temp_dir),
            "TORCHINDUCTOR_CACHE_DIR": str(temp_dir / "torchinductor_root"),
        })

    def test_cache_split_prefix_produces_discoverable_shard_names(self) -> None:
        base = Path("/cache") / builder.CACHE_BASENAME
        prefix = quant.gguf_split_output_prefix(base)
        self.assertEqual(prefix, base.with_suffix(""))
        self.assertEqual(
            prefix.with_name(f"{prefix.name}-00001-of-00002.gguf"),
            base.with_name(f"{base.stem}-00001-of-00002.gguf"),
        )
        with self.assertRaisesRegex(quant.PipelineError, "must end in .gguf"):
            quant.gguf_split_output_prefix(Path("/cache/Qwen3.8-BF16"))

    def test_cpp_inventory_include_is_generated_from_shared_descriptor(self) -> None:
        contract = quant.vision_inventory.load_contract()
        self.assertEqual(contract["tensor_count"], 334)
        self.assertEqual(
            quant.vision_inventory.generate_cpp_include(contract),
            quant.vision_inventory.CPP_INCLUDE_PATH.read_text(encoding="utf-8"),
        )

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
                "output": {"shards": [{
                    "path": f"/qwen-work/artifacts/{stock.name}/{shard.name}",
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

    def test_cache_address_binds_main_mmproj_and_vision_vocab(self) -> None:
        main = [{"name": "m-00001-of-00002.gguf", "size_bytes": 11,
                 "sha256": "1" * 64}]
        mmproj = {"name": builder.MMPROJ_BASENAME, "size_bytes": 7,
                  "sha256": "2" * 64}
        vocab = {"name": builder.VISION_VOCAB_BASENAME, "size_bytes": 5,
                 "sha256": "4" * 64}
        main_sha, cache_id = builder.cache_content_address(main, mmproj, vocab)
        self.assertEqual(main_sha, builder.canonical_sha256(main))
        changed = dict(mmproj, sha256="3" * 64)
        self.assertNotEqual(cache_id, builder.cache_content_address(main, changed, vocab)[1])
        self.assertNotEqual(cache_id, builder.cache_content_address(
            main, mmproj, dict(vocab, sha256="5" * 64))[1])

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
            vocab = fixture_root / builder.VISION_VOCAB_BASENAME
            make_vision_vocab_companion(vocab)
            vocab_gguf = quant.validate_qwen_vocab_only_gguf(vocab)
            cache_manifest = fixture_root / "bf16-cache-manifest.json"
            write_json(cache_manifest, {
                "schema": quant.BF16_CACHE_SCHEMA,
                "vision_mmproj": {"name": mmproj.name,
                                  "size_bytes": mmproj.stat().st_size,
                                  "sha256": digest(mmproj)},
                "vision_vocab": {"name": vocab.name,
                                  "size_bytes": vocab.stat().st_size,
                                  "sha256": digest(vocab),
                                  "gguf": vocab_gguf},
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
            self.assertEqual(roles[1]["text_model"]["sha256"], digest(vocab))
            self.assertEqual(
                roles[1]["tensor_inventory_sha256"],
                quant.vision_inventory.load_contract()["tensor_inventory_sha256"],
            )

    def test_companion_inventory_rejects_inexact_mmproj_inventory(self) -> None:
        for mutation, message in (("missing", "count mismatch"),
                                  ("duplicate", "duplicate"),
                                  ("wrong_shape", "shape mismatch")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                fixture = Fixture(root)
                fixture.companion_args(enable_mmproj=True)
                mmproj = root / builder.MMPROJ_BASENAME
                make_mmproj_companion(mmproj, mutation)
                vocab = root / builder.VISION_VOCAB_BASENAME
                make_vision_vocab_companion(vocab)
                cache_manifest = root / "bf16-cache-manifest.json"
                write_json(cache_manifest, {
                    "schema": quant.BF16_CACHE_SCHEMA,
                    "vision_mmproj": {"name": mmproj.name,
                                      "size_bytes": mmproj.stat().st_size,
                                      "sha256": digest(mmproj)},
                    "vision_vocab": {"name": vocab.name,
                                      "size_bytes": vocab.stat().st_size,
                                      "sha256": digest(vocab)},
                })
                mtp = root / "Qwen3.8-Flash-Next-MTP.gguf"
                mtp_export = root / "Qwen3.8-Flash-Next-MTP.export.json"
                with self.assertRaisesRegex(ValueError, message):
                    builder.make_companion_inventory(argparse.Namespace(
                        profile=fixture.profile,
                        quantization_arm=quant.DEFAULT_QUANTIZATION_ARM,
                        bf16_cache_manifest=cache_manifest,
                        bf16_cache_manifest_sha256=digest(cache_manifest),
                        mtp=mtp, mtp_bytes=mtp.stat().st_size,
                        mtp_sha256=digest(mtp),
                        mtp_matrix_quant_contract="Q4_0_ROCMI4",
                        mtp_export_manifest=mtp_export,
                        mtp_export_manifest_sha256=digest(mtp_export),
                        output=root / "companions.json",
                    ))

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
            vocab = staging / builder.VISION_VOCAB_BASENAME
            make_vision_vocab_companion(vocab)
            ple = quant.safetensor_ple_constants(fixture.snapshot)
            main_gguf = quant.verify_gguf_set(
                [main], ple, quantized=False, profile=profile)
            main_rows = [{"name": main.name, "size_bytes": main.stat().st_size,
                          "sha256": digest(main)}]
            mmproj_row = {"name": mmproj.name, "size_bytes": mmproj.stat().st_size,
                          "sha256": digest(mmproj), "format": "BF16",
                          "gguf": quant.validate_bf16_qwen_mmproj_gguf(mmproj)}
            vocab_row = {"name": vocab.name, "size_bytes": vocab.stat().st_size,
                         "sha256": digest(vocab), "format": "GGUF_VOCAB_ONLY",
                         "gguf": quant.validate_qwen_vocab_only_gguf(vocab)}
            main_sha, cache_id = builder.cache_content_address(
                main_rows, mmproj_row, vocab_row)
            tools = {
                "llama_cpp_revision": fixture.llama_head,
                "llama_cpp_base_revision": fixture.llama_base,
                "converter_sha256": digest(fixture.llama / "convert_hf_to_gguf.py"),
                "qwen4exp_converter_sha256": digest(
                    fixture.llama / "conversion" / "qwen4exp.py"),
                "ple_cgroup_writeback_patch_sha256": digest(fixture.ple_patch),
                "gguf_split_bounded_copy_patch_sha256":
                    digest(fixture.splitter_patch),
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
                               "split_copy_buffer_bytes": 16 * 1024 * 1024,
                               "main_storage_policy": "mostly_bf16_with_f32_ple",
                               "ple_intermediate_storage":
                                   "F32_streamed_to_temp_file_then_release_quant_override",
                               "ple_ggml_tensor_type": 0,
                               "mmproj": {"outtype": "bf16",
                                          "converter_option": "--mmproj"},
                               "vision_vocab": {"converter_option": "--vocab-only",
                                                "tensor_count": 0},
                               "gguf_writer_temp_cleanup": {
                                   "policy": "exact_converter_private_tmp_residue_v3",
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
                "vision_vocab": vocab_row,
            }
            manifest_path = staging / "bf16-cache-manifest.json"
            write_json(manifest_path, manifest)
            cache = root / f"bf16-{cache_id}"
            staging.rename(cache)
            manifest_path = cache / manifest_path.name
            validated = quant.validate_bf16_cache_manifest(
                manifest_path, digest(manifest_path), profile, digest(fixture.profile), tools)
            self.assertEqual(validated["cache_id"], cache_id)
            tree_cleanup = {
                "name": "torchinductor_root",
                "kind": "bounded_torchinductor_cache_tree",
                "entries": 1,
                "size_bytes": 0,
                "inventory_sha256": hashlib.sha256(
                    b'[{"kind":"directory","mode":448,"path":"torchinductor_root"}]'
                ).hexdigest(),
            }
            manifest["conversion"]["gguf_writer_temp_cleanup"]["main_removed"] = [
                tree_cleanup]
            write_json(manifest_path, manifest)
            validated = quant.validate_bf16_cache_manifest(
                manifest_path, digest(manifest_path), profile, digest(fixture.profile), tools)
            self.assertEqual(
                validated["conversion"]["gguf_writer_temp_cleanup"]["main_removed"],
                [tree_cleanup],
            )
            manifest["conversion"]["gguf_writer_temp_cleanup"]["main_removed"] = [
                tree_cleanup, tree_cleanup]
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(quant.PipelineError, "cleanup rows are malformed"):
                quant.validate_bf16_cache_manifest(
                    manifest_path, digest(manifest_path), profile,
                    digest(fixture.profile), tools)
            manifest["conversion"]["gguf_writer_temp_cleanup"]["main_removed"] = [{
                **tree_cleanup,
                "entries": quant.TORCHINDUCTOR_CACHE_MAX_ENTRIES + 1,
            }]
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(quant.PipelineError, "cleanup rows are malformed"):
                quant.validate_bf16_cache_manifest(
                    manifest_path, digest(manifest_path), profile,
                    digest(fixture.profile), tools)
            manifest["conversion"]["gguf_writer_temp_cleanup"]["main_removed"] = [
                tree_cleanup]
            write_json(manifest_path, manifest)
            # The splitter embeds its absolute build directory.  Rebuilding
            # pinned source in a new Ember revision may change only this binary
            # digest; both construction and consumption identities remain
            # recorded, while the cache shards are independently rehashed.
            tools["gguf_splitter_sha256"] = "8" * 64
            validated = quant.validate_bf16_cache_manifest(
                manifest_path, digest(manifest_path), profile, digest(fixture.profile), tools)
            self.assertEqual(validated["cache_id"], cache_id)
            manifest["toolchain"]["converter_sha256"] = "5" * 64
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(quant.PipelineError, "converter_sha256"):
                quant.validate_bf16_cache_manifest(
                    manifest_path, digest(manifest_path), profile,
                    digest(fixture.profile), tools)
            manifest["toolchain"]["converter_sha256"] = tools["converter_sha256"]
            manifest["toolchain"]["gguf_splitter_sha256"] = "malformed"
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(quant.PipelineError, "provenance is malformed"):
                quant.validate_bf16_cache_manifest(
                    manifest_path, digest(manifest_path), profile,
                    digest(fixture.profile), tools)

    def make_candidate(self, root: Path) -> tuple[Path, Path, str, Path, str]:
        workset = root / "workset"
        candidate = workset / "candidates" / "candidate-1"
        evidence = root / "evidence"
        candidate.mkdir(parents=True)
        evidence.mkdir()
        main_bytes = b"immutable-main"
        mmproj_bytes = b"immutable-mmproj"
        vocab_bytes = b"immutable-vocab"
        main_row = {"name": "main.gguf", "size_bytes": len(main_bytes),
                    "sha256": hashlib.sha256(main_bytes).hexdigest()}
        mmproj_row = {"name": "mmproj.gguf", "size_bytes": len(mmproj_bytes),
                      "sha256": hashlib.sha256(mmproj_bytes).hexdigest()}
        vocab_row = {"name": "vocab.gguf", "size_bytes": len(vocab_bytes),
                     "sha256": hashlib.sha256(vocab_bytes).hexdigest()}
        main_sha, cache_id = builder.cache_content_address(
            [main_row], mmproj_row, vocab_row)
        cache_dir = workset / "bf16-cache" / f"bf16-{cache_id}"
        cache_dir.mkdir(parents=True)
        (cache_dir / main_row["name"]).write_bytes(main_bytes)
        (cache_dir / mmproj_row["name"]).write_bytes(mmproj_bytes)
        (cache_dir / vocab_row["name"]).write_bytes(vocab_bytes)
        cache_manifest = cache_dir / "bf16-cache-manifest.json"
        write_json(cache_manifest, {
            "schema": quant.BF16_CACHE_SCHEMA, "cache_id": cache_id,
            "main": {"content_sha256": main_sha, "shards": [main_row]},
            "vision_mmproj": mmproj_row, "vision_vocab": vocab_row,
        })
        companion_dir = workset / "companions"
        companion_dir.mkdir()
        mtp = companion_dir / "mtp.gguf"
        mtp.write_bytes(b"immutable-mtp")
        mtp_export = companion_dir / "mtp-export.json"
        write_json(mtp_export, {"status": "complete"})
        companion_manifest = companion_dir / "companion-inventory.json"
        write_json(companion_manifest, {
            "schema": quant.COMPANION_INVENTORY_SCHEMA,
            "companions": [
                {"role": "mtp", "enabled": True, "path": str(mtp),
                 "size_bytes": mtp.stat().st_size, "sha256": digest(mtp),
                 "export_manifest_path": str(mtp_export),
                 "export_manifest_sha256": digest(mtp_export)},
                {"role": "vision_mmproj", "enabled": True,
                 "path": str(cache_dir / mmproj_row["name"]),
                 "size_bytes": mmproj_row["size_bytes"],
                 "sha256": mmproj_row["sha256"],
                 "text_model": {"path": str(cache_dir / vocab_row["name"]),
                                "size_bytes": vocab_row["size_bytes"],
                                "sha256": vocab_row["sha256"]}},
            ],
        })
        shard = candidate / "candidate-00001-of-00001.gguf"
        shard.write_bytes(b"quantized-candidate")
        record = candidate / "qwen-quant-build-record.json"
        write_json(record, {
            "status": "complete",
            "bf16_cache": {"cache_id": cache_id,
                            "manifest": {"path": str(cache_manifest),
                                         "sha256": digest(cache_manifest)}},
            "companion_inventory": {
                "status": "verified_exact",
                "manifest": {"path": str(companion_manifest),
                             "sha256": digest(companion_manifest)}},
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

    def test_external_attestation_descriptor_is_verified_with_fixed_signer(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            subject = root / "assessment.json"
            write_json(subject, {"schema": "ember.qwen3.8.candidate-assessment.v2"})
            bundle = root / "assessment.sigstore.json"
            bundle.write_bytes(b"signed")
            descriptor = {
                "subject": {"path": str(subject), "sha256": digest(subject),
                            "schema": "ember.qwen3.8.candidate-assessment.v2"},
                "bundle": {"path": str(bundle), "sha256": digest(bundle)},
                "repository": builder.ATTEST_REPOSITORY,
                "signer_workflow": builder.ATTEST_WORKFLOW,
            }
            calls = []
            with mock.patch.object(
                    builder, "ATTESTATION_VERIFIER",
                    side_effect=lambda *args: calls.append(args)):
                value, _evidence = builder._read_attested_json(
                    descriptor, "candidate assessment",
                    "ember.qwen3.8.candidate-assessment.v2")
            self.assertEqual(value["schema"], "ember.qwen3.8.candidate-assessment.v2")
            self.assertEqual(calls, [(subject, bundle, builder.ATTEST_REPOSITORY,
                                      builder.ATTEST_WORKFLOW)])
            descriptor["repository"] = "attacker/example"
            with self.assertRaisesRegex(builder.BuilderError, "signer/subject"):
                builder._read_attested_json(
                    descriptor, "candidate assessment",
                    "ember.qwen3.8.candidate-assessment.v2")

    def make_rolling_authority(
        self, root: Path, candidate_id: str, build_record_sha256: str,
    ) -> tuple[Path, str]:
        assessment = root / "evidence" / "rolling-assessment.json"
        write_json(assessment, {
            "schema": "ember.qwen3.8.candidate-assessment.v2",
            "artifact_identity": {"candidate_id": candidate_id,
                                  "build_record_sha256": build_record_sha256},
            "artifact_may_be_deleted_after_external_attestation": True,
        })
        assessment_bundle = root / "evidence" / "rolling-assessment.sigstore.json"
        assessment_bundle.write_bytes(b"signed-assessment")
        accumulator = root / "evidence" / "rolling-accumulator.json"
        write_json(accumulator, {
            "schema": "ember.qwen3.8.sequential-bakeoff-accumulator.v2",
            "assessments": [{
                "subject": {"path": str(assessment), "sha256": digest(assessment),
                            "schema": "ember.qwen3.8.candidate-assessment.v2"},
                "bundle": {"path": str(assessment_bundle),
                           "sha256": digest(assessment_bundle)},
                "repository": "OtherU-AI/ember",
                "signer_workflow": ".github/workflows/qwen-gfx1151-bakeoff.yml",
            }],
        })
        accumulator_bundle = root / "evidence" / "rolling-accumulator.sigstore.json"
        accumulator_bundle.write_bytes(b"signed-accumulator")
        authority = root / "evidence" / "rolling-authority.json"
        write_json(authority, {
            "schema": builder.RETENTION_AUTHORITY_SCHEMA,
            "status": "externally_attested_accumulator_verified",
            "phase": "sweep",
            "selection_plan": {"path": str(root / "evidence" / "plan.json"),
                               "sha256": "9" * 64},
            "accumulator": {
                "subject": {"path": str(accumulator), "sha256": digest(accumulator),
                            "schema": "ember.qwen3.8.sequential-bakeoff-accumulator.v2"},
                "bundle": {"path": str(accumulator_bundle),
                           "sha256": digest(accumulator_bundle)},
                "repository": "OtherU-AI/ember",
                "signer_workflow": ".github/workflows/qwen-gfx1151-bakeoff.yml",
            },
            "retained_candidate_ids": [],
            "retire_candidate_ids": [candidate_id],
            "selection_policy": "rolling_stock_plus_exact_winner_top1",
            "reconstruction_required": True,
            "publishes": False,
        })
        return authority, digest(authority)

    def mocked_rolling_loader(self, build_record_sha256: str):
        transition = {
            "phase": "sweep",
            "selection_policy": "rolling_stock_plus_exact_winner_top1",
            "retained_candidate_ids": [],
            "retire_candidate_ids": ["candidate-1"],
        }
        assessment = {
            "artifact_identity": {"candidate_id": "candidate-1",
                                  "build_record_sha256": build_record_sha256},
            "artifact_may_be_deleted_after_external_attestation": True,
        }
        evidence = {"path": "/evidence/assessment.json", "sha256": "8" * 64}
        return mock.patch.object(builder, "_load_rolling_retention", return_value=(
            transition,
            {"selection_plan": {"path": "/evidence/plan.json", "sha256": "9" * 64},
             "accumulator": {"subject": evidence}},
            {"candidate-1": {"assessment": assessment, "evidence": evidence}},
        ))

    def test_rolling_retirement_is_reconstructable_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, shard, record_sha, _assessment, _assessment_sha = self.make_candidate(root)
            authority, authority_sha = self.make_rolling_authority(
                root, "candidate-1", record_sha)
            retirement = root / "evidence" / "candidate-1-retirement.json"
            with self.mocked_rolling_loader(record_sha):
                planned = builder.retire_reconstructable(argparse.Namespace(
                    retention_authority=authority,
                    retention_authority_sha256=authority_sha,
                    candidate_id="candidate-1", candidate_dir=candidate,
                    build_record_sha256=record_sha, output=retirement, execute=False,
                ))
                self.assertEqual(planned["status"], "planned")
                self.assertTrue(shard.exists())
                result = builder.retire_reconstructable(argparse.Namespace(
                    retention_authority=authority,
                    retention_authority_sha256=authority_sha,
                    candidate_id="candidate-1", candidate_dir=candidate,
                    build_record_sha256=record_sha, output=retirement, execute=True,
                ))
            self.assertTrue(result["recoverable"])
            self.assertFalse(shard.exists())
            self.assertTrue((candidate / "qwen-quant-build-record.json").is_file())

            rebuilt = root / "rebuilt"
            rebuilt.mkdir()
            rebuilt_shard = rebuilt / shard.name
            rebuilt_shard.write_bytes(b"quantized-candidate")
            original_record = json.loads(
                (candidate / "qwen-quant-build-record.json").read_text(encoding="utf-8"))
            rebuilt_record = rebuilt / "qwen-quant-build-record.json"
            original_record["output"]["shards"][0]["path"] = str(rebuilt_shard)
            write_json(rebuilt_record, original_record)
            receipt = root / "evidence" / "candidate-1-reconstruction.json"
            # Simulate a process interruption after the shard rename but
            # before the durable reconstruction receipt was published.
            quant.rename_directory_noreplace(rebuilt_shard, shard)
            restored = builder.restore_reconstructable(argparse.Namespace(
                retirement_completion=Path(result["completion"]),
                retirement_completion_sha256=result["completion_sha256"],
                rebuilt_candidate_dir=rebuilt,
                rebuilt_build_record_sha256=digest(rebuilt_record), output=receipt,
            ))
            self.assertEqual(restored["restored_bytes"], len(b"quantized-candidate"))
            self.assertEqual(shard.read_bytes(), b"quantized-candidate")
            self.assertFalse(rebuilt_shard.exists())
            self.assertTrue(receipt.is_file())

    def test_reconstruction_rejects_nonidentical_shards(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, shard, record_sha, _assessment, _assessment_sha = self.make_candidate(root)
            authority, authority_sha = self.make_rolling_authority(
                root, "candidate-1", record_sha)
            retirement = root / "evidence" / "candidate-1-retirement.json"
            with self.mocked_rolling_loader(record_sha):
                result = builder.retire_reconstructable(argparse.Namespace(
                    retention_authority=authority,
                    retention_authority_sha256=authority_sha,
                    candidate_id="candidate-1", candidate_dir=candidate,
                    build_record_sha256=record_sha, output=retirement, execute=True,
                ))
            rebuilt = root / "rebuilt"
            rebuilt.mkdir()
            rebuilt_shard = rebuilt / shard.name
            rebuilt_shard.write_bytes(b"different-quantized-candidate")
            rebuilt_record = rebuilt / "qwen-quant-build-record.json"
            original_record = json.loads(
                (candidate / "qwen-quant-build-record.json").read_text(encoding="utf-8"))
            original_record["output"]["shards"][0] = {
                "path": str(rebuilt_shard), "size_bytes": rebuilt_shard.stat().st_size,
                "sha256": digest(rebuilt_shard),
            }
            write_json(rebuilt_record, original_record)
            with self.assertRaisesRegex(builder.BuilderError, "does not reproduce"):
                builder.restore_reconstructable(argparse.Namespace(
                    retirement_completion=Path(result["completion"]),
                    retirement_completion_sha256=result["completion_sha256"],
                    rebuilt_candidate_dir=rebuilt,
                    rebuilt_build_record_sha256=digest(rebuilt_record),
                    output=root / "evidence" / "must-not-exist.json",
                ))
            self.assertFalse(shard.exists())

    def test_retirement_rejects_missing_reconstruction_cache_content(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, shard, record_sha, _assessment, _assessment_sha = self.make_candidate(root)
            record = json.loads(
                (candidate / "qwen-quant-build-record.json").read_text(encoding="utf-8"))
            manifest_path = Path(record["bf16_cache"]["manifest"]["path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            (manifest_path.parent / manifest["main"]["shards"][0]["name"]).unlink()
            authority, authority_sha = self.make_rolling_authority(
                root, "candidate-1", record_sha)
            with self.mocked_rolling_loader(record_sha), self.assertRaisesRegex(
                    quant.PipelineError, "immutable BF16 cache shard"):
                builder.retire_reconstructable(argparse.Namespace(
                    retention_authority=authority,
                    retention_authority_sha256=authority_sha,
                    candidate_id="candidate-1", candidate_dir=candidate,
                    build_record_sha256=record_sha,
                    output=root / "evidence" / "must-not-exist.json", execute=True,
                ))
            self.assertTrue(shard.exists())

    def test_retirement_resumes_after_one_authorized_shard_was_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, first, _record_sha, _assessment, _assessment_sha = self.make_candidate(root)
            second = candidate / "candidate-00002-of-00002.gguf"
            second.write_bytes(b"second-quantized-shard")
            record_path = candidate / "qwen-quant-build-record.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["output"]["shards"].append({
                "path": str(second), "size_bytes": second.stat().st_size,
                "sha256": digest(second),
            })
            write_json(record_path, record)
            record_sha = digest(record_path)
            authority, authority_sha = self.make_rolling_authority(
                root, "candidate-1", record_sha)
            retirement = root / "evidence" / "partial-retirement.json"
            arguments = argparse.Namespace(
                retention_authority=authority,
                retention_authority_sha256=authority_sha,
                candidate_id="candidate-1", candidate_dir=candidate,
                build_record_sha256=record_sha, output=retirement, execute=False,
            )
            with self.mocked_rolling_loader(record_sha):
                builder.retire_reconstructable(arguments)
                authorization = json.loads(retirement.read_text(encoding="utf-8"))
                quarantine = Path(authorization["quarantine"][0]["path"])
                # Simulate termination after the no-clobber quarantine rename
                # but before the exact shard was unlinked.
                quant.rename_directory_noreplace(first, quarantine)
                quant.fsync_directory(candidate)
                arguments.execute = True
                result = builder.retire_reconstructable(arguments)
            self.assertEqual(result["status"], "complete")
            self.assertFalse(first.exists())
            self.assertFalse(quarantine.exists())
            self.assertFalse(second.exists())
            self.assertTrue(Path(result["completion"]).is_file())


if __name__ == "__main__":
    unittest.main()
