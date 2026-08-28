#!/usr/bin/env python3
"""Build Qwen3.8 intervention candidates serially from one immutable BF16 cache.

This is a local construction tool, not a publisher.  The expensive HF -> BF16
conversion and the BF16 vision projector are committed once under a content
address.  Each candidate then streams those unchanged BF16 shards through the
audited quantizer, which applies the common intervention immediately before the
selected mixed-format encoding.  Losing quant shards may be removed only after
``record-assessment`` has fsync'd an exact external evidence bundle.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import resource
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import qwen_quantize as quant


MEMORY_LIMIT_BYTES = 134217728000  # docker --memory 125g
SWAP_LIMIT_BYTES = 0
CACHE_BASENAME = "Qwen3.8-Flash-Next-BF16.gguf"
MMPROJ_BASENAME = "Qwen3.8-Flash-Next-BF16-mmproj.gguf"
VISION_VOCAB_BASENAME = "Qwen3.8-Flash-Next-vocab-only.gguf"
ASSESSMENT_SCHEMA = "ember.qwen3.8.candidate-assessment-bundle.v1"
TOMBSTONE_SCHEMA = "ember.qwen3.8.deleted-loser.v1"
RECONSTRUCTABLE_RETIREMENT_SCHEMA = (
    "ember.qwen3.8.reconstructable-candidate-retirement.v1")
RECONSTRUCTABLE_RETIREMENT_COMPLETE_SCHEMA = (
    "ember.qwen3.8.reconstructable-candidate-retirement-complete.v1")
RECONSTRUCTION_RECEIPT_SCHEMA = "ember.qwen3.8.candidate-reconstruction.v1"
RETENTION_AUTHORITY_SCHEMA = "ember.qwen3.8.attested-rolling-retention.v1"
SEALED_RETENTION_AUTHORITY_SCHEMA = "ember.qwen3.8.attested-sealed-retention.v1"
STOCK_RETIRE_AUTH_SCHEMA = "ember.qwen3.8.stock-retirement-authorization.v1"
STOCK_RETIRE_COMPLETE_SCHEMA = "ember.qwen3.8.stock-retirement-complete.v1"
CONTAINER_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ATTEST_REPOSITORY = "OtherU-AI/ember"
ATTEST_WORKFLOW = ".github/workflows/qwen-gfx1151-bakeoff.yml"


class BuilderError(ValueError):
    pass


def verify_external_attestation(
    subject: Path, bundle: Path, repository: str, signer_workflow: str,
) -> None:
    command = ["gh", "attestation", "verify", str(subject), "--bundle", str(bundle),
               "--repo", repository, "--signer-workflow", signer_workflow]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False)
    except OSError as exc:
        raise BuilderError(f"cannot run external attestation verifier: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BuilderError(f"external attestation verification failed: {detail}")


ATTESTATION_VERIFIER = verify_external_attestation


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()


def safe_id(value: str, label: str) -> str:
    if (not value or len(value) > 96 or value in {".", ".."}
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                   for character in value)):
        raise BuilderError(f"{label} is not a safe identifier")
    return value


def write_json_fsync(path: Path, value: dict[str, Any], *, create: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if create else os.O_TRUNC)
    descriptor = os.open(path, flags, 0o600)
    try:
        data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    quant.fsync_directory(path.parent)


def write_json_atomic_noreplace(path: Path, value: dict[str, Any]) -> None:
    """Publish critical lifecycle evidence without exposing a partial final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_stage = tempfile.mkstemp(prefix=f".{path.name}.stage-", dir=path.parent)
    os.close(descriptor)
    stage = Path(raw_stage)
    try:
        write_json_fsync(stage, value, create=False)
        quant.rename_directory_noreplace(stage, path)
        quant.fsync_directory(path.parent)
    finally:
        if stage.exists() and not stage.is_symlink():
            stage.unlink()


class WorksetLease:
    """Serialize cache conversion, intervention encoding, and loser deletion."""

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        self.descriptor = -1

    def __enter__(self) -> "WorksetLease":
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".qwen-candidate-workset.lock"
        self.descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.descriptor)
            self.descriptor = -1
            raise BuilderError("another Qwen candidate workset is active") from exc
        return self

    def __exit__(self, *_ignored: Any) -> None:
        if self.descriptor >= 0:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = -1


def validate_cgroup(args: argparse.Namespace) -> dict[str, int]:
    if args.memory_limit_bytes != MEMORY_LIMIT_BYTES:
        raise BuilderError(f"--memory-limit-bytes must be exactly {MEMORY_LIMIT_BYTES}")
    try:
        memory_max = quant.read_cgroup_counter(args.cgroup_memory_max_path, "memory.max")
        swap_max = quant.read_cgroup_counter(args.cgroup_memory_swap_max_path, "memory.swap.max")
        peak = quant.read_cgroup_counter(args.cgroup_memory_peak_path, "memory.peak")
    except quant.PipelineError as exc:
        raise BuilderError(str(exc)) from exc
    if memory_max != MEMORY_LIMIT_BYTES or swap_max != SWAP_LIMIT_BYTES:
        raise BuilderError("candidate construction requires an exact 125 GiB no-swap cgroup")
    if peak > memory_max:
        raise BuilderError("pre-existing cgroup peak already exceeds the construction limit")
    return {"memory_limit_bytes": memory_max, "swap_limit_bytes": swap_max,
            "peak_before_bytes": peak}


def finish_measurement(args: argparse.Namespace, start: float, before: dict[str, int]) -> dict[str, Any]:
    peak = quant.read_cgroup_counter(args.cgroup_memory_peak_path, "memory.peak")
    child_rss_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if peak > before["memory_limit_bytes"]:
        raise BuilderError("measured cgroup peak exceeded the 125 GiB construction limit")
    if child_rss_kib * 1024 > before["memory_limit_bytes"]:
        raise BuilderError("child maximum RSS exceeded the 125 GiB construction limit")
    return {
        "status": "measured_target_cgroup_v2",
        "memory_limit_bytes": before["memory_limit_bytes"],
        "swap_limit_bytes": before["swap_limit_bytes"],
        "cgroup_peak_before_bytes": before["peak_before_bytes"],
        "cgroup_peak_bytes": peak,
        "child_maximum_resident_set_kib": child_rss_kib,
        "wall_seconds": round(time.monotonic() - start, 3),
    }


def cache_content_address(
    main_rows: list[dict[str, Any]], mmproj: dict[str, Any],
    vision_vocab: dict[str, Any],
) -> tuple[str, str]:
    main_digest = canonical_sha256(main_rows)
    cache_id = canonical_sha256({
        "main_content_sha256": main_digest,
        "vision_mmproj": {key: mmproj[key] for key in ("name", "size_bytes", "sha256")},
        "vision_vocab": {
            key: vision_vocab[key] for key in ("name", "size_bytes", "sha256")},
    })
    return main_digest, cache_id


def tensor_format_contract_sha256(
    build_info: dict[str, Any], rocmfpx_revision: str,
) -> str:
    formats = build_info.get("per_tensor_formats")
    if (not isinstance(formats, list) or not formats
            or any(not isinstance(value, str) or not value for value in formats)):
        raise BuilderError("quantizer build info omits supported per-tensor formats")
    return canonical_sha256({
        "schema": "ember.qwen3.8.tensor-format-contract.v1",
        "architecture": "qwen4exp",
        "gguf_version": 3,
        "quant_type_ids_and_block_layout_revision": canonical_sha256({
            "rocmfpx_revision": rocmfpx_revision,
            "type_ids": dict(sorted(quant.SUPPORTED_TENSOR_FORMATS.items())),
        }),
        "supported_per_tensor_formats": sorted(formats),
        "metadata_contract_revision": "ember-qwen4exp-ple-intervention-metadata-v1",
        "decoder_semantics_revision": "ember-gfx1151-qwen4exp-quant-decode-v1",
    })


def converter_temp_env(temp_dir: Path) -> dict[str, str]:
    """Keep converter and torch compiler scratch inside its private transaction."""
    return {
        "TMPDIR": str(temp_dir),
        "TORCHINDUCTOR_CACHE_DIR": str(temp_dir / "torchinductor_root"),
    }


def prepare_cache(args: argparse.Namespace) -> dict[str, Any]:
    root = args.cache_root.absolute()
    if CONTAINER_DIGEST_RE.fullmatch(args.builder_container_digest) is None:
        raise BuilderError("--builder-container-digest must be one exact sha256: digest")
    try:
        environment_lock = quant.inspect_exact_file(
            args.converter_environment_lock.absolute(),
            args.converter_environment_lock_sha256,
            args.converter_environment_lock.absolute().stat().st_size,
            "converter environment lock")
    except quant.PipelineError as exc:
        raise BuilderError(str(exc)) from exc
    with WorksetLease(root):
        existing = sorted(path.name for path in root.iterdir()
                          if path.is_dir() and path.name.startswith("bf16-"))
        transactions = sorted(path.name for path in root.iterdir()
                              if path.name.startswith(".bf16-cache.transaction-"))
        if existing:
            raise BuilderError(
                f"one immutable BF16 cache already exists; refusing a second: {existing}")
        if transactions:
            raise BuilderError(
                f"stale BF16 transaction requires operator review: {transactions}")
        profile_path = args.profile.resolve()
        profile, inventory, inventory_path = quant.validate_profile(profile_path)
        resources = quant.validate_resources(root / "bf16-cache-planned", 1152, 120)
        cgroup = validate_cgroup(args)
        tools = quant.validate_tools(
            args.llama_cpp_dir.resolve(), args.rocmfpx_dir.resolve(),
            args.ember_dir.resolve(), args.ember_revision,
            args.quantizer.resolve(), profile, args.gguf_splitter.resolve())
        with quant.SnapshotReadLease(args.snapshot_dir.resolve()):
            snapshot = quant.validate_snapshot(
                args.snapshot_dir.resolve(), args.snapshot_revision, profile, inventory)
            ple = quant.safetensor_ple_constants(args.snapshot_dir.resolve())
            transaction = Path(tempfile.mkdtemp(prefix=".bf16-cache.transaction-", dir=root))
            started = time.monotonic()
            try:
                unsplit = transaction / "Qwen3.8-Flash-Next-BF16.unsplit.gguf"
                main = transaction / CACHE_BASENAME
                temp_dir = transaction / ".converter-tmp"
                temp_dir.mkdir(mode=0o700)
                converter = args.llama_cpp_dir.resolve() / "convert_hf_to_gguf.py"
                quant.run_checked([
                    sys.executable, str(converter), str(args.snapshot_dir.resolve()),
                    "--outfile", str(unsplit), "--outtype", "bf16", "--use-temp-file",
                ], env_overrides=converter_temp_env(temp_dir))
                quant.run_checked([
                    str(args.gguf_splitter.resolve()), "--split-max-size", "48G",
                    str(unsplit), str(main),
                ])
                unsplit.unlink()
                main_temp_cleanup = quant.cleanup_gguf_writer_temp(temp_dir)
                main_paths = quant.discover_gguf(main)
                main_gguf = quant.verify_gguf_set(
                    main_paths, ple, quantized=False, profile=profile)
                ple_tensors = [
                    tensor for path in main_paths
                    for tensor in quant.inspect_gguf(path)["tensors"]
                    if tensor["name"] == profile["quantization"]["ple_tensor_name"]
                ]
                if len(ple_tensors) != 1 or ple_tensors[0]["type"] != 0:
                    raise BuilderError(
                        "patched conversion must keep the transient PLE table as F32")

                mmproj = transaction / MMPROJ_BASENAME
                mmproj_temp_dir = transaction / ".mmproj-converter-tmp"
                mmproj_temp_dir.mkdir(mode=0o700)
                quant.run_checked([
                    sys.executable, str(converter), str(args.snapshot_dir.resolve()),
                    "--outfile", str(mmproj), "--outtype", "bf16", "--mmproj",
                    "--use-temp-file",
                ], env_overrides=converter_temp_env(mmproj_temp_dir))
                mmproj_temp_cleanup = quant.cleanup_gguf_writer_temp(mmproj_temp_dir)
                if not mmproj.is_file():
                    raise BuilderError("pinned converter did not produce the named BF16 mmproj")
                mmproj_gguf = quant.validate_bf16_qwen_mmproj_gguf(mmproj)
                vision_vocab = transaction / VISION_VOCAB_BASENAME
                quant.run_checked([
                    sys.executable, str(converter), str(args.snapshot_dir.resolve()),
                    "--outfile", str(vision_vocab), "--vocab-only",
                ])
                if not vision_vocab.is_file():
                    raise BuilderError(
                        "pinned converter did not produce the named vision vocab companion")
                vision_vocab_gguf = quant.validate_qwen_vocab_only_gguf(vision_vocab)
                measurement = finish_measurement(args, started, cgroup)
                quant.sync_verified_outputs([*main_paths, mmproj, vision_vocab])
                main_rows = [{"name": Path(row["path"]).name,
                              "size_bytes": row["size_bytes"],
                              "sha256": row["sha256"]}
                             for row in main_gguf["shards"]]
                mmproj_row = {"name": mmproj.name, "size_bytes": mmproj.stat().st_size,
                              "sha256": quant.sha256_file(mmproj), "format": "BF16",
                              "gguf": mmproj_gguf}
                vision_vocab_row = {
                    "name": vision_vocab.name,
                    "size_bytes": vision_vocab.stat().st_size,
                    "sha256": quant.sha256_file(vision_vocab),
                    "format": "GGUF_VOCAB_ONLY",
                    "gguf": vision_vocab_gguf,
                }
                main_digest, cache_id = cache_content_address(
                    main_rows, mmproj_row, vision_vocab_row)
                manifest = {
                    "schema": quant.BF16_CACHE_SCHEMA,
                    "cache_id": cache_id,
                    "source": {
                        "repo_id": profile["source"]["repo_id"],
                        "revision": profile["source"]["revision"],
                        "snapshot_inventory_sha256": profile["source"]["snapshot_inventory_sha256"],
                    },
                    "profile": {"profile_id": profile["profile_id"],
                                "sha256": quant.sha256_file(profile_path)},
                    "toolchain": {
                        key: tools.get(key) for key in (
                            "llama_cpp_revision", "llama_cpp_base_revision",
                            "converter_sha256", "qwen4exp_converter_sha256",
                            "ple_cgroup_writeback_patch_sha256",
                            "gguf_split_bounded_copy_patch_sha256",
                            "gguf_splitter_sha256")
                    } | {
                        "converter_environment_lock_sha256": environment_lock["sha256"],
                        "converter_environment_lock_bytes": environment_lock["size_bytes"],
                        "builder_container_digest": args.builder_container_digest,
                    },
                    "conversion": {
                        "outtype": "bf16", "split_max_size": "48G", "use_temp_file": True,
                        "split_copy_buffer_bytes": tools["gguf_split_copy_buffer_bytes"],
                        "main_storage_policy": "mostly_bf16_with_f32_ple",
                        "ple_intermediate_storage":
                            "F32_streamed_to_temp_file_then_release_quant_override",
                        "ple_ggml_tensor_type": 0,
                        "mmproj": {"outtype": "bf16", "converter_option": "--mmproj"},
                        "vision_vocab": {
                            "converter_option": "--vocab-only", "tensor_count": 0},
                        "gguf_writer_temp_cleanup": {
                            "policy": "exact_converter_private_tmp_residue_v3",
                            "main_removed": main_temp_cleanup,
                            "mmproj_removed": mmproj_temp_cleanup,
                        },
                    },
                    "resources": resources,
                    "measurement": measurement,
                    "main": {
                        "base_path": CACHE_BASENAME, "content_sha256": main_digest,
                        "shards": main_rows, "ple": ple,
                        "gguf": {key: main_gguf[key] for key in (
                            "tensor_count", "tensor_names_sha256", "tensor_type_counts")},
                    },
                    "vision_mmproj": mmproj_row,
                    "vision_vocab": vision_vocab_row,
                }
                manifest_path = transaction / "bf16-cache-manifest.json"
                write_json_fsync(manifest_path, manifest)
                for path in [*main_paths, mmproj, vision_vocab, manifest_path]:
                    path.chmod(0o444)
                final = root / f"bf16-{cache_id}"
                quant.rename_directory_noreplace(transaction, final)
                transaction = None
                final.chmod(0o555)
                quant.fsync_directory(root)
                return {
                    "status": "complete", "cache_id": cache_id,
                    "cache_dir": str(final),
                    "manifest": str(final / manifest_path.name),
                    "manifest_sha256": quant.sha256_file(final / manifest_path.name),
                    "snapshot_inventory": {"path": str(inventory_path),
                                           "sha256": quant.sha256_file(inventory_path)},
                    "snapshot": snapshot,
                }
            finally:
                if transaction is not None:
                    shutil.rmtree(transaction, ignore_errors=True)


def make_companion_inventory(args: argparse.Namespace) -> dict[str, Any]:
    profile, _inventory, _inventory_path = quant.validate_profile(args.profile.resolve())
    if args.quantization_arm == quant.DEFAULT_QUANTIZATION_ARM:
        arm = {"mtp_matrix_quant_contract": "Q4_0_ROCMI4"}
    else:
        arm = quant.validated_quantization_arms(profile).get(args.quantization_arm)
    if arm is None:
        raise BuilderError("--quantization-arm must name the default or a declared bakeoff arm")
    matrix = quant.selected_mtp_matrix_contract(
        profile, arm, args.mtp_matrix_quant_contract)
    # Cache validation itself needs full tool provenance and is performed by the
    # build command.  Here the out-of-band manifest digest binds the exact cache
    # and the companion files are independently hashed and structurally checked.
    manifest, evidence = quant.read_exact_json_file(
        args.bf16_cache_manifest.absolute(), args.bf16_cache_manifest_sha256,
        "BF16 cache manifest")
    if manifest.get("schema") != quant.BF16_CACHE_SCHEMA:
        raise BuilderError("unsupported BF16 cache manifest")
    mmproj = manifest.get("vision_mmproj")
    if not isinstance(mmproj, dict):
        raise BuilderError("BF16 cache omits vision_mmproj")
    cache_dir = args.bf16_cache_manifest.absolute().parent
    mmproj_path = cache_dir / str(mmproj.get("name", ""))
    mmproj_exact = quant.inspect_exact_file(
        mmproj_path, mmproj.get("sha256"), mmproj.get("size_bytes"), "vision mmproj")
    mmproj_gguf = quant.validate_bf16_qwen_mmproj_gguf(mmproj_path)
    recorded_gguf = mmproj.get("gguf")
    if recorded_gguf is not None and recorded_gguf != mmproj_gguf:
        raise BuilderError("cached vision mmproj inventory differs from its creation record")
    vision_vocab = manifest.get("vision_vocab")
    if not isinstance(vision_vocab, dict):
        raise BuilderError("BF16 cache omits vision_vocab")
    vision_vocab_path = cache_dir / str(vision_vocab.get("name", ""))
    vision_vocab_exact = quant.inspect_exact_file(
        vision_vocab_path, vision_vocab.get("sha256"),
        vision_vocab.get("size_bytes"), "vision vocab companion")
    vision_vocab_gguf = quant.validate_qwen_vocab_only_gguf(vision_vocab_path)
    if vision_vocab.get("gguf") is not None and vision_vocab.get("gguf") != vision_vocab_gguf:
        raise BuilderError(
            "cached vision vocab inventory differs from its creation record")
    mtp_exact = quant.inspect_exact_file(
        args.mtp.absolute(), args.mtp_sha256, args.mtp_bytes, "MTP companion")
    quant.validate_mtp_companion_gguf(
        args.mtp.absolute(), matrix, profile["source"]["revision"])
    export_evidence = quant.validate_mtp_export_manifest(
        args.mtp_export_manifest.absolute(), args.mtp_export_manifest_sha256,
        mtp_exact, matrix, profile)
    output = args.output.absolute()
    payload = {
        "schema": quant.COMPANION_INVENTORY_SCHEMA,
        "source": {
            "repo_id": profile["source"]["repo_id"],
            "revision": profile["source"]["revision"],
            "snapshot_inventory_sha256": profile["source"]["snapshot_inventory_sha256"],
        },
        "companions": [
            {"role": "mtp", "enabled": True, "path": mtp_exact["path"],
             "size_bytes": mtp_exact["size_bytes"], "sha256": mtp_exact["sha256"],
             "matrix_quant_contract": matrix,
             "export_manifest_path": export_evidence["path"],
             "export_manifest_sha256": export_evidence["sha256"]},
            {"role": "vision_mmproj", "enabled": True,
             "path": mmproj_exact["path"], "size_bytes": mmproj_exact["size_bytes"],
             "sha256": mmproj_exact["sha256"], "format": "BF16",
             "tensor_inventory_sha256": mmproj_gguf["tensor_inventory_sha256"],
             "text_model": {
                 "path": vision_vocab_exact["path"],
                 "size_bytes": vision_vocab_exact["size_bytes"],
                 "sha256": vision_vocab_exact["sha256"],
                 "format": "GGUF_VOCAB_ONLY",
                 "metadata_sha256": vision_vocab_gguf["metadata_sha256"],
             }},
        ],
    }
    write_json_fsync(output, payload)
    return {"status": "complete", "inventory": str(output),
            "inventory_sha256": quant.sha256_file(output),
            "bf16_cache_manifest": evidence}


def validate_stock_capture_match(
    capture_path: Path,
    capture_sha256: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Prove a cache-built stock control reproduces the captured control bytes."""
    capture, evidence = quant.read_exact_json_file(
        capture_path.absolute(), capture_sha256, "stock activation capture manifest")
    if (capture.get("schema") != "ember.qwen3.8.stock-control-activation-capture.v1"
            or capture.get("status") != "complete"
            or capture.get("stock_rocmi4_only") is not True):
        raise BuilderError("stock activation capture manifest is not a completed stock control")
    captured_model = capture.get("model")
    captured_rows = captured_model.get("shards") if isinstance(captured_model, dict) else None
    output = record.get("output")
    built_rows = output.get("shards") if isinstance(output, dict) else None
    if not isinstance(captured_rows, list) or not captured_rows:
        raise BuilderError("stock activation capture manifest has no shard inventory")
    if not isinstance(built_rows, list) or len(built_rows) != len(captured_rows):
        raise BuilderError("cache-built stock shard count differs from the captured control")
    expected = [(row.get("size_bytes"), row.get("sha256")) for row in captured_rows
                if isinstance(row, dict)]
    actual = [(row.get("size_bytes"), row.get("sha256")) for row in built_rows
              if isinstance(row, dict)]
    if len(expected) != len(captured_rows) or len(actual) != len(built_rows):
        raise BuilderError("stock shard inventory row is malformed")
    if actual != expected:
        raise BuilderError("cache-built stock bytes differ from the activation-captured control")
    return {
        **evidence,
        "captured_build_record_sha256": captured_model.get("build_record_sha256"),
        "captured_shards": len(expected),
        "byte_identical": True,
    }


def build_candidate(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise BuilderError(f"refusing to overwrite candidate output: {output}")
    if CONTAINER_DIGEST_RE.fullmatch(args.builder_container_digest) is None:
        raise BuilderError("--builder-container-digest must be one exact sha256: digest")
    capture_pair = (args.stock_capture_manifest is not None,
                    args.stock_capture_manifest_sha256 is not None)
    if capture_pair[0] != capture_pair[1]:
        raise BuilderError(
            "--stock-capture-manifest and --stock-capture-manifest-sha256 are required together")
    if args.stock_control and not capture_pair[0]:
        raise BuilderError("cache-built stock control requires its activation capture manifest")
    if not args.stock_control and capture_pair[0]:
        raise BuilderError("stock capture evidence applies only to --stock-control")
    if args.stock_control and args.bakeoff_plan is not None:
        raise BuilderError("stock control cannot consume directional sweep authorization")
    cache_root = args.bf16_cache_manifest.absolute().parent.parent
    if args.workset_root.absolute() != cache_root:
        raise BuilderError("--workset-root must be the BF16 cache root owning the shared lock")
    with WorksetLease(cache_root):
        before = validate_cgroup(args)
        started = time.monotonic()
        qargs = quant.parse_args([
            "--profile", str(args.profile),
            "--snapshot-dir", str(args.snapshot_dir),
            "--snapshot-revision", args.snapshot_revision,
            "--llama-cpp-dir", str(args.llama_cpp_dir),
            "--rocmfpx-dir", str(args.rocmfpx_dir),
            "--ember-dir", str(args.ember_dir), "--ember-revision", args.ember_revision,
            "--quantizer", str(args.quantizer), "--gguf-splitter", str(args.gguf_splitter),
            "--bf16-cache-manifest", str(args.bf16_cache_manifest),
            "--bf16-cache-manifest-sha256", args.bf16_cache_manifest_sha256,
            "--companion-inventory", str(args.companion_inventory),
            "--companion-inventory-sha256", args.companion_inventory_sha256,
            "--mtp-matrix-quant-contract", args.mtp_matrix_quant_contract,
            "--ttm-pages-limit-path", str(args.ttm_pages_limit_path),
            "--quantization-arm", args.quantization_arm,
            "--threads", str(args.threads), "--min-free-gib", str(args.min_free_gib),
            "--work-dir", str(output), "--execute",
            *(["--stock-control"] if args.stock_control else [
                "--intervention-manifest", str(args.intervention_manifest),
            ]),
            *([] if args.bakeoff_plan is None else [
                "--bakeoff-plan", str(args.bakeoff_plan),
                "--bakeoff-plan-sha256", str(args.bakeoff_plan_sha256),
            ]),
        ])
        try:
            record = quant.orchestrate(qargs)
            measurement = finish_measurement(args, started, before)
            record_path = output / "qwen-quant-build-record.json"
            build_info = record["tools"]["quantizer_build_info"]
            compatibility_sha256 = tensor_format_contract_sha256(
                build_info, record["tools"]["rocmfpx_revision"])
            emitted_formats = sorted(record["quantization_recipe"]["formats"])
            if not set(emitted_formats).issubset(set(build_info["per_tensor_formats"])):
                raise BuilderError("candidate emitted formats exceed the attested format contract")
            versions = {quant.inspect_gguf(Path(row["path"]))["version"]
                        for row in record["output"]["shards"]}
            if versions != {3}:
                raise BuilderError("candidate artifacts must use GGUF version 3")
            builder_identity = {
                "ember_revision": record["tools"]["ember_revision"],
                "quantizer_tool_sha256": record["tools"]["quantizer_sha256"],
                "container_digest": args.builder_container_digest,
                "tensor_format_contract_sha256": compatibility_sha256,
            }
            output_rows = [{key: row[key] for key in ("path", "size_bytes", "sha256")}
                           for row in record["output"]["shards"]]
            stock_capture = None
            if args.stock_control:
                stock_capture = validate_stock_capture_match(
                    args.stock_capture_manifest,
                    args.stock_capture_manifest_sha256,
                    record,
                )
            attestation = {
                "schema": "ember.qwen3.8.candidate-workset-attestation.v1",
                "candidate_id": safe_id(args.candidate_id, "candidate id"),
                "build_record_sha256": quant.sha256_file(record_path),
                "bf16_cache_id": record["bf16_cache"]["cache_id"],
                "bf16_cache_manifest_sha256": args.bf16_cache_manifest_sha256,
                "intervention_manifest_sha256": (
                    stock_capture["sha256"]
                    if args.stock_control else record["intervention"]["manifest_sha256"]),
                "stock_capture": stock_capture,
                "quantization_arm": record["quantization_recipe"]["id"],
                "builder_identity": builder_identity,
                "artifact_builder_detail": {
                    "candidate_builder_sha256": quant.sha256_file(Path(__file__).resolve()),
                    "qwen_quantize_sha256": quant.sha256_file(Path(quant.__file__).resolve()),
                    "quantizer_build_info": build_info,
                    "rocmfpx_revision": record["tools"]["rocmfpx_revision"],
                    "tensor_formats": record["quantization_recipe"]["formats"],
                    "tensor_overrides_sha256": record["quantization_recipe"][
                        "per_tensor_overrides_sha256"],
                },
                "tensor_format_compatibility_sha256": compatibility_sha256,
                "artifact_tensor_format_subset": emitted_formats,
                "artifact_identity": {
                    "builder_identity": builder_identity,
                    "tensor_format_compatibility_sha256": compatibility_sha256,
                    "artifact_tensor_format_subset": emitted_formats,
                    "quantized_shards_sha256": canonical_sha256(output_rows),
                    "quantized_shards": output_rows,
                },
                "runtime_engine_compatibility": {
                    "binding": "format_contract_not_runtime_engine_revision",
                    "architecture": "qwen4exp",
                    "required_tensor_formats": record["quantization_recipe"]["formats"],
                    "requires_exact_artifact_hashes": True,
                    "runtime_engine_revision": None,
                    "reason": (
                        "graph_and_kernel_only engine revisions do not alter encoded weights"
                    ),
                },
                "measurement": measurement,
                "publishes": False,
            }
            attestation_path = output / "qwen-candidate-workset-attestation.json"
            write_json_fsync(attestation_path, attestation)
            return {"status": "complete", "candidate_id": args.candidate_id,
                    "output": str(output), "build_record": str(record_path),
                    "build_record_sha256": quant.sha256_file(record_path),
                    "workset_attestation": str(attestation_path),
                    "workset_attestation_sha256": quant.sha256_file(attestation_path)}
        except (quant.PipelineError, BuilderError):
            # This command owns only a previously absent output.  A cgroup or
            # provenance failure must not leave a seemingly usable candidate.
            if output.is_dir() and not output.is_symlink():
                shutil.rmtree(output)
            raise


def declared_candidate_shards(
    candidate_dir: Path, record: dict[str, Any],
) -> list[dict[str, Any]]:
    output = record.get("output")
    if (not isinstance(output, dict) or not isinstance(output.get("shards"), list)
            or not output["shards"]):
        raise BuilderError("candidate build record omits output shards")
    rows: list[dict[str, Any]] = []
    paths: set[Path] = set()
    for index, row in enumerate(output["shards"]):
        if not isinstance(row, dict):
            raise BuilderError("candidate shard row is malformed")
        path = Path(str(row.get("path", ""))).absolute()
        size_bytes = row.get("size_bytes")
        sha256 = row.get("sha256")
        if (path.parent != candidate_dir or PurePosixPath(path.name).name != path.name
                or path in paths or not isinstance(size_bytes, int)
                or isinstance(size_bytes, bool) or size_bytes < 1
                or not isinstance(sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", sha256) is None):
            raise BuilderError("candidate shard is duplicated, malformed, or escapes its directory")
        paths.add(path)
        rows.append({"path": str(path), "size_bytes": size_bytes, "sha256": sha256})
    return rows


def exact_candidate_shards(candidate_dir: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(declared_candidate_shards(candidate_dir, record)):
        path = Path(row["path"])
        try:
            evidence = quant.inspect_exact_file(
                path, row.get("sha256"), row.get("size_bytes"), f"candidate shard {index}")
        except quant.PipelineError as exc:
            raise BuilderError(str(exc)) from exc
        rows.append({"path": evidence["path"], "size_bytes": evidence["size_bytes"],
                     "sha256": evidence["sha256"]})
    return rows


def exact_retirement_stock_shards(
    stock_dir: Path,
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Map the converter namespace to the one mounted retirement directory.

    The stock converter runs with its workspace mounted at ``/qwen-work`` and
    records paths below ``/qwen-work/artifacts/<stock-dir-name>``.  Retirement
    runs later with that artifact directory mounted at its fixed host-absolute
    path.  Accept only that exact, documented namespace substitution; trusting
    an arbitrary recorded parent here would broaden the deletion boundary.
    """
    output = record.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("shards"), list):
        raise BuilderError("captured stock build record omits output shards")
    expected_parent = PurePosixPath("/qwen-work/artifacts") / stock_dir.name
    rows: list[dict[str, Any]] = []
    filenames: set[str] = set()
    for index, row in enumerate(output["shards"]):
        if not isinstance(row, dict):
            raise BuilderError("captured stock shard row is malformed")
        raw = row.get("path")
        if not isinstance(raw, str) or not raw or posixpath.normpath(raw) != raw:
            raise BuilderError("captured stock shard path is not normalized")
        recorded_path = PurePosixPath(raw)
        if (not recorded_path.is_absolute()
                or recorded_path.parent != expected_parent
                or recorded_path.name in filenames
                or PurePosixPath(recorded_path.name).name != recorded_path.name):
            raise BuilderError(
                "captured stock shard is duplicated or outside the exact converter namespace")
        filenames.add(recorded_path.name)
        path = stock_dir / recorded_path.name
        if path.parent != stock_dir:
            raise BuilderError("mapped stock shard escapes the retirement directory")
        try:
            evidence = quant.inspect_exact_file(
                path, row.get("sha256"), row.get("size_bytes"),
                f"captured stock shard {index}")
        except quant.PipelineError as exc:
            raise BuilderError(str(exc)) from exc
        rows.append({"path": evidence["path"], "size_bytes": evidence["size_bytes"],
                     "sha256": evidence["sha256"]})
    return rows


def retire_captured_stock(args: argparse.Namespace) -> dict[str, Any]:
    """Free stock shard space only after durable activation-capture evidence."""
    stock_dir = args.stock_dir.absolute()
    if stock_dir.is_symlink() or not stock_dir.is_dir():
        raise BuilderError("stock directory must be one existing non-symlink directory")
    record_path = stock_dir / "qwen-quant-build-record.json"
    record, record_evidence = quant.read_exact_json_file(
        record_path, args.build_record_sha256, "captured stock build record")
    experiment = record.get("experiment")
    if (record.get("status") != "complete" or not isinstance(experiment, dict)
            or experiment.get("kind") != "stock_control"
            or experiment.get("stock_weights_unchanged") is not True):
        raise BuilderError("only a completed unchanged stock control can be retired")
    shards = exact_retirement_stock_shards(stock_dir, record)
    capture = validate_stock_capture_match(
        args.stock_capture_manifest, args.stock_capture_manifest_sha256, record)
    manifest, _ = quant.read_exact_json_file(
        args.stock_capture_manifest.absolute(), args.stock_capture_manifest_sha256,
        "stock activation capture manifest")
    captured_rows = manifest["model"]["shards"]
    if [Path(row["path"]).name for row in shards] != [row.get("filename") for row in captured_rows]:
        raise BuilderError("captured stock filenames differ from the retirement target")
    if manifest["model"].get("build_record_sha256") != args.build_record_sha256:
        raise BuilderError("activation capture used a different stock build record")
    authorization = args.output.absolute()
    completion = authorization.with_name(authorization.name + ".complete.json")
    if completion.exists() or completion.is_symlink():
        raise BuilderError(f"refusing to overwrite stock retirement completion: {completion}")
    workset_root = args.workset_root.absolute()
    with WorksetLease(workset_root):
        payload = {
            "schema": STOCK_RETIRE_AUTH_SCHEMA,
            "status": "authorized_before_deletion",
            "stock_dir": str(stock_dir),
            "workset_root": str(workset_root),
            "build_record": record_evidence,
            "stock_capture": capture,
            "shards": shards,
            "total_bytes": sum(row["size_bytes"] for row in shards),
            "recovery": "pinned_snapshot_then_content_addressed_bf16_cache",
            "publishes": False,
        }
        write_json_fsync(authorization, payload)
        authorization_sha256 = quant.sha256_file(authorization)
        for shard in shards:
            Path(shard["path"]).unlink()
        quant.fsync_directory(stock_dir)
        result = {
            "schema": STOCK_RETIRE_COMPLETE_SCHEMA,
            "status": "complete",
            "authorization": {"path": str(authorization),
                              "sha256": authorization_sha256},
            "deleted_shards": shards,
            "deleted_bytes": payload["total_bytes"],
            "build_record_retained": record_evidence,
            "stock_capture": capture,
            "recoverable": True,
        }
        write_json_fsync(completion, result)
        return {"status": "complete", "completion": str(completion),
                "completion_sha256": quant.sha256_file(completion),
                "deleted_bytes": result["deleted_bytes"], "recoverable": True}


def record_assessment(args: argparse.Namespace) -> dict[str, Any]:
    candidate_dir = args.candidate_dir.absolute()
    bundle = args.output.absolute()
    if bundle.is_relative_to(candidate_dir):
        raise BuilderError("assessment bundle must live outside the deletable candidate directory")
    record_path = candidate_dir / "qwen-quant-build-record.json"
    record, record_evidence = quant.read_exact_json_file(
        record_path, args.build_record_sha256, "candidate build record")
    if record.get("status") != "complete":
        raise BuilderError("only a complete candidate can be assessed")
    shards = exact_candidate_shards(candidate_dir, record)
    cache = record.get("bf16_cache")
    if not isinstance(cache, dict) or not isinstance(cache.get("manifest"), dict):
        raise BuilderError("candidate build record omits its BF16 cache binding")
    cache_manifest_path = Path(str(cache["manifest"].get("path", ""))).absolute()
    workset_root = cache_manifest_path.parent.parent
    assessment, assessment_evidence = quant.read_exact_json_file(
        args.assessment.absolute(), args.assessment_sha256, "candidate assessment")
    if assessment.get("candidate_id") != args.candidate_id:
        raise BuilderError("assessment candidate id differs from the requested candidate")
    payload = {
        "schema": ASSESSMENT_SCHEMA,
        "candidate_id": safe_id(args.candidate_id, "candidate id"),
        "selected": args.selected,
        "artifact_state": "present_at_bundle_commit",
        "candidate_dir": str(candidate_dir),
        "workset_root": str(workset_root),
        "build_record": record_evidence,
        "assessment": assessment_evidence,
        "shards": shards,
        "total_artifact_bytes": sum(row["size_bytes"] for row in shards),
        "deletion_eligible": not args.selected,
    }
    write_json_fsync(bundle, payload)
    return {"status": "complete", "bundle": str(bundle),
            "bundle_sha256": quant.sha256_file(bundle),
            "deletion_eligible": not args.selected}


def delete_loser(args: argparse.Namespace) -> dict[str, Any]:
    bundle, bundle_evidence = quant.read_exact_json_file(
        args.assessment_bundle.absolute(), args.assessment_bundle_sha256,
        "candidate assessment bundle")
    if (bundle.get("schema") != ASSESSMENT_SCHEMA or bundle.get("selected") is not False
            or bundle.get("deletion_eligible") is not True
            or bundle.get("artifact_state") != "present_at_bundle_commit"):
        raise BuilderError("assessment bundle does not authorize loser deletion")
    candidate_dir = Path(str(bundle.get("candidate_dir", ""))).absolute()
    workset_root = Path(str(bundle.get("workset_root", ""))).absolute()
    if not workset_root.is_dir() or workset_root == candidate_dir:
        raise BuilderError("assessment bundle workset root is malformed")
    if args.assessment_bundle.absolute().is_relative_to(candidate_dir):
        raise BuilderError("assessment bundle is not durable outside the candidate directory")
    rows = bundle.get("shards")
    if not isinstance(rows, list) or not rows:
        raise BuilderError("assessment bundle has no exact shard inventory")
    with WorksetLease(workset_root):
        checked: list[Path] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise BuilderError("assessment bundle shard row is malformed")
            path = Path(str(row.get("path", ""))).absolute()
            if path.parent != candidate_dir:
                raise BuilderError("assessment bundle shard escapes candidate directory")
            quant.inspect_exact_file(path, row.get("sha256"), row.get("size_bytes"),
                                     f"loser shard {index}")
            checked.append(path)
        if not args.execute:
            return {"status": "planned", "candidate_id": bundle["candidate_id"],
                    "would_delete": [str(path) for path in checked],
                    "recoverable": False}
        tombstone = args.assessment_bundle.absolute().with_name(
            args.assessment_bundle.name + ".deleted.json")
        if tombstone.exists() or tombstone.is_symlink():
            raise BuilderError(f"refusing to overwrite deletion tombstone: {tombstone}")
        # The authorization bundle is already fsync'd.  Remove only the exact
        # regular shard inodes it inventories; retain build/intervention evidence.
        for path in checked:
            path.unlink()
        quant.fsync_directory(candidate_dir)
        payload = {
            "schema": TOMBSTONE_SCHEMA, "candidate_id": bundle["candidate_id"],
            "assessment_bundle": bundle_evidence,
            "deleted_shards": rows,
            "deleted_bytes": sum(row["size_bytes"] for row in rows),
            "recoverable": False,
        }
        write_json_fsync(tombstone, payload)
        return {"status": "complete", "candidate_id": bundle["candidate_id"],
                "deleted_bytes": payload["deleted_bytes"], "tombstone": str(tombstone),
                "tombstone_sha256": quant.sha256_file(tombstone), "recoverable": False}


def _read_attested_json(
    descriptor: Any, label: str, schema: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(descriptor, dict) or set(descriptor) != {
            "subject", "bundle", "repository", "signer_workflow"}:
        raise BuilderError(f"{label} attestation descriptor is malformed")
    subject = descriptor.get("subject")
    bundle = descriptor.get("bundle")
    if (not isinstance(subject, dict) or set(subject) != {"path", "sha256", "schema"}
            or subject.get("schema") != schema or not isinstance(bundle, dict)
            or set(bundle) != {"path", "sha256"}
            or descriptor.get("repository") != ATTEST_REPOSITORY
            or descriptor.get("signer_workflow") != ATTEST_WORKFLOW):
        raise BuilderError(f"{label} attestation signer/subject differs")
    subject_path = Path(str(subject.get("path", ""))).absolute()
    value, subject_evidence = quant.read_exact_json_file(
        subject_path, subject.get("sha256"), label)
    if value.get("schema") != schema:
        raise BuilderError(f"{label} content schema differs")
    bundle_path = Path(str(bundle.get("path", ""))).absolute()
    try:
        bundle_size = bundle_path.lstat().st_size
    except OSError as exc:
        raise BuilderError(f"cannot stat {label} attestation bundle: {exc}") from exc
    bundle_evidence = quant.inspect_exact_file(
        bundle_path, bundle.get("sha256"), bundle_size, f"{label} attestation bundle")
    ATTESTATION_VERIFIER(subject_path, bundle_path, ATTEST_REPOSITORY, ATTEST_WORKFLOW)
    return value, {
        "subject": {**subject_evidence, "schema": schema},
        "bundle": bundle_evidence,
        "repository": ATTEST_REPOSITORY,
        "signer_workflow": ATTEST_WORKFLOW,
    }


def _load_rolling_retention(
    plan_path: Path, plan_sha256: str, accumulator_descriptor: Any, phase: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    if phase not in {"sweep", "format"}:
        raise BuilderError("rolling retention phase must be sweep or format")
    plan, plan_evidence = quant.read_exact_json_file(
        plan_path.absolute(), plan_sha256, "rolling retention selection plan")
    accumulator, accumulator_evidence = _read_attested_json(
        accumulator_descriptor, "rolling accumulator",
        "ember.qwen3.8.sequential-bakeoff-accumulator.v2")
    if (accumulator.get("phase") != phase
            or accumulator.get("plan_sha256") != plan_evidence["sha256"]
            or accumulator.get("contains_raw_measurements") is not False
            or accumulator.get("external_attestation_required") is not True
            or accumulator.get("publication_allowed") is not False):
        raise BuilderError("rolling accumulator lifecycle/plan differs")
    descriptors = accumulator.get("assessments")
    if not isinstance(descriptors, list) or not descriptors:
        raise BuilderError("rolling accumulator has no assessment descriptors")
    rows = []
    by_candidate: dict[str, dict[str, Any]] = {}
    for descriptor in descriptors:
        assessment, evidence = _read_attested_json(
            descriptor, "candidate assessment", "ember.qwen3.8.candidate-assessment.v2")
        artifact = assessment.get("artifact_identity")
        candidate_id = artifact.get("candidate_id") if isinstance(artifact, dict) else None
        if (not isinstance(candidate_id, str) or not candidate_id
                or candidate_id in by_candidate
                or assessment.get("artifact_may_be_deleted_after_external_attestation") is not True):
            raise BuilderError("rolling assessment candidate identity/lifecycle differs")
        rows.append(assessment)
        by_candidate[candidate_id] = {"assessment": assessment, "evidence": evidence}
    try:
        import qwen_bakeoff as bakeoff
        transition = bakeoff.rolling_retention_transition(plan, rows, phase)
    except (ImportError, ValueError) as exc:
        raise BuilderError(f"cannot derive rolling retention transition: {exc}") from exc
    return transition, {
        "selection_plan": plan_evidence,
        "accumulator": accumulator_evidence,
    }, by_candidate


def authorize_rolling_retention(args: argparse.Namespace) -> dict[str, Any]:
    accumulator_descriptor = {
        "subject": {"path": str(args.accumulator.absolute()),
                    "sha256": args.accumulator_sha256,
                    "schema": "ember.qwen3.8.sequential-bakeoff-accumulator.v2"},
        "bundle": {"path": str(args.accumulator_bundle.absolute()),
                   "sha256": args.accumulator_bundle_sha256},
        "repository": ATTEST_REPOSITORY,
        "signer_workflow": ATTEST_WORKFLOW,
    }
    transition, evidence, _by_candidate = _load_rolling_retention(
        args.plan, args.plan_sha256, accumulator_descriptor, args.phase)
    payload = {
        "schema": RETENTION_AUTHORITY_SCHEMA,
        "status": "externally_attested_accumulator_verified",
        **transition,
        **evidence,
        "reconstruction_required": True,
        "publishes": False,
    }
    write_json_fsync(args.output.absolute(), payload)
    return {"status": "complete", "authority": str(args.output.absolute()),
            "authority_sha256": quant.sha256_file(args.output.absolute()),
            **transition, "publishes": False}


def _load_sealed_retention(
    plan_path: Path, plan_sha256: str, ledger_descriptor: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    plan, plan_evidence = quant.read_exact_json_file(
        plan_path.absolute(), plan_sha256, "sealed retention selection plan")
    ledger, ledger_evidence = _read_attested_json(
        ledger_descriptor, "sealed format ledger",
        "ember.qwen3.8.sequential-bakeoff-ledger.v3")
    try:
        import qwen_bakeoff as bakeoff
        transition = bakeoff.sealed_format_retention(plan, ledger)
    except (ImportError, ValueError) as exc:
        raise BuilderError(f"cannot derive sealed format retention: {exc}") from exc
    rows = ledger.get("assessments")
    digests = ledger.get("assessment_sha256")
    if (not isinstance(rows, list) or not isinstance(digests, list)
            or len(rows) != len(digests)):
        raise BuilderError("sealed format ledger assessment inventory differs")
    by_candidate: dict[str, dict[str, Any]] = {}
    for row, digest in zip(rows, digests, strict=True):
        artifact = row.get("artifact_identity") if isinstance(row, dict) else None
        candidate_id = artifact.get("candidate_id") if isinstance(artifact, dict) else None
        if (not isinstance(candidate_id, str) or candidate_id in by_candidate
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise BuilderError("sealed format ledger candidate identity differs")
        by_candidate[candidate_id] = {
            "assessment": row,
            "evidence": {"embedded_in_ledger": ledger_evidence["subject"],
                         "assessment_sha256": digest},
        }
    return transition, {
        "selection_plan": plan_evidence,
        "selection_ledger": ledger_evidence,
    }, by_candidate


def authorize_sealed_retention(args: argparse.Namespace) -> dict[str, Any]:
    ledger_descriptor = {
        "subject": {"path": str(args.ledger.absolute()), "sha256": args.ledger_sha256,
                    "schema": "ember.qwen3.8.sequential-bakeoff-ledger.v3"},
        "bundle": {"path": str(args.ledger_bundle.absolute()),
                   "sha256": args.ledger_bundle_sha256},
        "repository": ATTEST_REPOSITORY,
        "signer_workflow": ATTEST_WORKFLOW,
    }
    transition, evidence, _by_candidate = _load_sealed_retention(
        args.plan, args.plan_sha256, ledger_descriptor)
    payload = {
        "schema": SEALED_RETENTION_AUTHORITY_SCHEMA,
        "status": "externally_attested_ledger_verified",
        **transition,
        **evidence,
        "reconstruction_required": True,
        "publishes": False,
    }
    write_json_fsync(args.output.absolute(), payload)
    return {"status": "complete", "authority": str(args.output.absolute()),
            "authority_sha256": quant.sha256_file(args.output.absolute()),
            **transition, "publishes": False}


def _validate_reconstruction_cache(
    record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    cache = record.get("bf16_cache")
    manifest_desc = cache.get("manifest") if isinstance(cache, dict) else None
    if (not isinstance(manifest_desc, dict)
            or not isinstance(cache.get("cache_id"), str)):
        raise BuilderError("candidate build record omits its reconstructable BF16 cache")
    manifest_path = Path(str(manifest_desc.get("path", ""))).absolute()
    manifest, manifest_evidence = quant.read_exact_json_file(
        manifest_path, manifest_desc.get("sha256"), "immutable BF16 cache manifest")
    if (manifest.get("schema") != quant.BF16_CACHE_SCHEMA
            or manifest.get("cache_id") != cache.get("cache_id")):
        raise BuilderError("immutable BF16 cache schema/content address differs")
    main = manifest.get("main")
    main_rows = main.get("shards") if isinstance(main, dict) else None
    mmproj = manifest.get("vision_mmproj")
    vocab = manifest.get("vision_vocab")
    if (not isinstance(main_rows, list) or not main_rows
            or not isinstance(mmproj, dict) or not isinstance(vocab, dict)):
        raise BuilderError("immutable BF16 cache inventory is incomplete")
    cache_dir = manifest_path.parent
    normalized = []
    names: set[str] = set()
    for index, row in enumerate(main_rows):
        if not isinstance(row, dict):
            raise BuilderError("immutable BF16 cache shard row is malformed")
        name = row.get("name")
        if (not isinstance(name, str) or not name or PurePosixPath(name).name != name
                or name in names):
            raise BuilderError("immutable BF16 cache shard name is unsafe or duplicated")
        names.add(name)
        exact = quant.inspect_exact_file(
            cache_dir / name, row.get("sha256"), row.get("size_bytes"),
            f"immutable BF16 cache shard {index}")
        normalized.append({"name": name, "size_bytes": exact["size_bytes"],
                           "sha256": exact["sha256"]})
    companions = []
    for label, row in (("vision mmproj", mmproj), ("vision vocab", vocab)):
        name = row.get("name")
        if (not isinstance(name, str) or not name or PurePosixPath(name).name != name
                or name in names):
            raise BuilderError(f"immutable BF16 cache {label} name is unsafe or duplicated")
        names.add(name)
        exact = quant.inspect_exact_file(
            cache_dir / name, row.get("sha256"), row.get("size_bytes"),
            f"immutable BF16 cache {label}")
        companions.append({"name": name, "size_bytes": exact["size_bytes"],
                           "sha256": exact["sha256"]})
    main_sha, cache_id = cache_content_address(normalized, companions[0], companions[1])
    if (main.get("content_sha256") != main_sha or manifest.get("cache_id") != cache_id
            or cache_dir.name != f"bf16-{cache_id}"):
        raise BuilderError("immutable BF16 cache content address does not reproduce")
    companion = record.get("companion_inventory")
    companion_desc = companion.get("manifest") if isinstance(companion, dict) else None
    if (not isinstance(companion_desc, dict)
            or companion.get("status") != "verified_exact"):
        raise BuilderError("candidate build record omits its exact companion inventory")
    companion_path = Path(str(companion_desc.get("path", ""))).absolute()
    companion_value, companion_evidence = quant.read_exact_json_file(
        companion_path, companion_desc.get("sha256"), "immutable companion inventory")
    rows = companion_value.get("companions")
    if (companion_value.get("schema") != quant.COMPANION_INVENTORY_SCHEMA
            or not isinstance(rows, list) or not rows):
        raise BuilderError("immutable companion inventory schema/content differs")
    roles: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("enabled") is not True:
            raise BuilderError("reconstruction requires every declared companion enabled")
        role = row.get("role")
        if not isinstance(role, str) or role in roles:
            raise BuilderError("immutable companion inventory role is malformed or duplicated")
        roles.add(role)
        path = Path(str(row.get("path", ""))).absolute()
        quant.inspect_exact_file(path, row.get("sha256"), row.get("size_bytes"),
                                 f"immutable {role} companion {index}")
        if role == "mtp":
            export_path = Path(str(row.get("export_manifest_path", ""))).absolute()
            try:
                export_size = export_path.lstat().st_size
            except OSError as exc:
                raise BuilderError(f"cannot stat immutable MTP export manifest: {exc}") from exc
            quant.inspect_exact_file(
                export_path, row.get("export_manifest_sha256"), export_size,
                "immutable MTP export manifest")
        text_model = row.get("text_model")
        if text_model is not None:
            if not isinstance(text_model, dict):
                raise BuilderError("immutable vision text companion is malformed")
            quant.inspect_exact_file(
                Path(str(text_model.get("path", ""))).absolute(),
                text_model.get("sha256"), text_model.get("size_bytes"),
                "immutable vision vocab companion")
    if roles != {"mtp", "vision_mmproj"}:
        raise BuilderError("reconstruction companion roles are incomplete")
    return manifest_evidence, companion_evidence, cache_dir.parent


def _record_reconstruction_contract(record: dict[str, Any]) -> dict[str, Any]:
    """Select immutable construction inputs while excluding paths and timings."""
    cache = record.get("bf16_cache") or {}
    companion = record.get("companion_inventory") or {}
    tools = record.get("tools") or {}
    recipe = record.get("quantization_recipe") or {}
    intervention = record.get("intervention") or {}
    profile = record.get("profile") or {}
    snapshot = record.get("snapshot") or {}
    return {
        "bf16_cache": {"cache_id": cache.get("cache_id"),
                       "manifest": cache.get("manifest")},
        "companion_inventory_manifest": companion.get("manifest"),
        "intervention": {key: intervention.get(key) for key in (
            "manifest_sha256", "kind", "application_stage", "weight_intervention",
            "target_names_sha256")},
        "profile": {"sha256": profile.get("sha256")},
        "snapshot": {key: snapshot.get(key) for key in ("repo_id", "revision")},
        "quantization_recipe": {key: recipe.get(key) for key in (
            "id", "formats", "per_tensor_overrides_sha256")},
        "tools": {key: tools.get(key) for key in (
            "ember_revision", "quantizer_sha256", "rocmfpx_revision")},
    }


def retire_reconstructable(args: argparse.Namespace) -> dict[str, Any]:
    """Retire exact shards under an attested rolling-selection authority."""
    authority, authority_evidence = quant.read_exact_json_file(
        args.retention_authority.absolute(), args.retention_authority_sha256,
        "rolling retention authority")
    schema = authority.get("schema")
    common_keys = {"schema", "status", "phase", "selection_policy",
                   "retained_candidate_ids", "retire_candidate_ids", "selection_plan",
                   "reconstruction_required", "publishes"}
    if (authority.get("publishes") is not False
            or authority.get("reconstruction_required") is not True):
        raise BuilderError("rolling retention authority lifecycle/schema differs")
    retained = authority.get("retained_candidate_ids")
    retired = authority.get("retire_candidate_ids")
    if (not isinstance(retained, list) or not isinstance(retired, list)
            or any(not isinstance(item, str) for item in [*retained, *retired])
            or len(set(retained)) != len(retained) or len(set(retired)) != len(retired)
            or set(retained) & set(retired) or args.candidate_id not in retired):
        raise BuilderError("rolling retention authority does not retire this candidate")
    plan_desc = authority.get("selection_plan")
    if not isinstance(plan_desc, dict):
        raise BuilderError("rolling retention authority omits its selection plan")
    if (schema == RETENTION_AUTHORITY_SCHEMA
            and set(authority) == common_keys | {"accumulator"}
            and authority.get("status") == "externally_attested_accumulator_verified"):
        transition, retention_evidence, candidates = _load_rolling_retention(
            Path(str(plan_desc.get("path", ""))), plan_desc.get("sha256"),
            authority.get("accumulator"), authority.get("phase"))
        selection_evidence = retention_evidence["accumulator"]
    elif (schema == SEALED_RETENTION_AUTHORITY_SCHEMA
          and set(authority) == common_keys | {"selection_ledger"}
          and authority.get("status") == "externally_attested_ledger_verified"):
        transition, retention_evidence, candidates = _load_sealed_retention(
            Path(str(plan_desc.get("path", ""))), plan_desc.get("sha256"),
            authority.get("selection_ledger"))
        selection_evidence = retention_evidence["selection_ledger"]
    else:
        raise BuilderError("retention authority lifecycle/schema differs")
    if any(authority.get(key) != transition[key] for key in transition):
        raise BuilderError("rolling retention authority does not rederive from attested evidence")

    candidate_dir = args.candidate_dir.absolute()
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise BuilderError("candidate directory must be one existing non-symlink directory")
    record_path = candidate_dir / "qwen-quant-build-record.json"
    record, record_evidence = quant.read_exact_json_file(
        record_path, args.build_record_sha256, "candidate build record")
    if record.get("status") != "complete":
        raise BuilderError("only a completed candidate can be retired reconstructably")
    candidate = candidates.get(args.candidate_id)
    if candidate is None:
        raise BuilderError("rolling accumulator does not identify the retired candidate")
    assessment = candidate["assessment"]
    assessment_evidence = candidate["evidence"]
    artifact = assessment.get("artifact_identity") or {}
    if artifact.get("build_record_sha256") != args.build_record_sha256:
        raise BuilderError("attested candidate assessment does not bind this build record")
    cache_evidence, companion_evidence, workset_root = _validate_reconstruction_cache(record)
    if candidate_dir != workset_root.parent / "candidates" / safe_id(
            args.candidate_id, "candidate id"):
        raise BuilderError("candidate directory is not its canonical workset location")
    output = args.output.absolute()
    completion = output.with_name(output.name + ".complete.json")
    if output.is_relative_to(candidate_dir) or completion.is_relative_to(candidate_dir):
        raise BuilderError("candidate retirement evidence must live outside its candidate directory")
    with WorksetLease(workset_root):
        record, record_evidence = quant.read_exact_json_file(
            record_path, args.build_record_sha256, "candidate build record")
        cache_evidence, companion_evidence, current_workset_root = (
            _validate_reconstruction_cache(record))
        if current_workset_root != workset_root:
            raise BuilderError("candidate reconstruction workset changed under its lease")
        declared = declared_candidate_shards(candidate_dir, record)
        quarantine = [{
            "original_path": row["path"],
            "path": str(candidate_dir / (
                f".retiring-{args.retention_authority_sha256}-{index:04d}")),
        } for index, row in enumerate(declared)]

        def shard_present(path: Path, row: dict[str, Any], label: str) -> bool:
            if path.is_symlink():
                raise BuilderError(f"{label} was replaced by a symlink")
            if not path.exists():
                return False
            quant.inspect_exact_file(path, row["sha256"], row["size_bytes"], label)
            return True

        states = []
        for index, (row, quarantine_row) in enumerate(
                zip(declared, quarantine, strict=True)):
            original = shard_present(
                Path(row["path"]), row, f"retirable candidate shard {index}")
            staged = shard_present(
                Path(quarantine_row["path"]), row,
                f"quarantined candidate shard {index}")
            if original and staged:
                raise BuilderError("candidate shard exists at both original and quarantine paths")
            states.append("original" if original else "quarantine" if staged else "deleted")
        payload = {
            "schema": RECONSTRUCTABLE_RETIREMENT_SCHEMA,
            "status": "authorized_before_deletion",
            "candidate_id": safe_id(args.candidate_id, "candidate id"),
            "candidate_dir": str(candidate_dir), "workset_root": str(workset_root),
            "retention_authority": authority_evidence,
            "selection_evidence": selection_evidence,
            "assessment": assessment_evidence,
            "build_record": record_evidence,
            "bf16_cache_manifest": cache_evidence,
            "companion_inventory_manifest": companion_evidence,
            "reconstruction_contract": _record_reconstruction_contract(record),
            "shards": declared,
            "quarantine": quarantine,
            "total_artifact_bytes": sum(row["size_bytes"] for row in declared),
            "recovery": "rebuild_from_immutable_bf16_cache_then_match_every_original_shard",
            "publishes": False,
        }
        authorization_exists = output.exists() and not output.is_symlink()
        if not authorization_exists and any(state != "original" for state in states):
            raise BuilderError("cannot create retirement authorization after shards are missing")
        if authorization_exists:
            existing, _ = quant.read_exact_json_file(
                output, quant.sha256_file(output), "existing reconstruction authorization")
            if existing != payload:
                raise BuilderError("existing reconstruction authorization differs")
        elif output.is_symlink():
            raise BuilderError("reconstruction authorization must not be a symlink")
        else:
            write_json_atomic_noreplace(output, payload)
        output_sha256 = quant.sha256_file(output)
        if not args.execute:
            return {"status": "planned", "candidate_id": args.candidate_id,
                    "would_delete": [row["path"] for row, state in
                                     zip(declared, states, strict=True)
                                     if state != "deleted"],
                    "recoverable": True}
        for index, (row, quarantine_row) in enumerate(
                zip(declared, quarantine, strict=True)):
            original_path = Path(row["path"])
            quarantine_path = Path(quarantine_row["path"])
            original = shard_present(
                original_path, row, f"retirable candidate shard {index}")
            staged = shard_present(
                quarantine_path, row, f"quarantined candidate shard {index}")
            if original and staged:
                raise BuilderError("candidate shard exists at both original and quarantine paths")
            if original:
                # Move the directory entry first, then hash the entry actually
                # moved. A concurrent pathname replacement is quarantined and
                # rejected rather than being unlinked as the authorized shard.
                quant.rename_directory_noreplace(original_path, quarantine_path)
                quant.fsync_directory(candidate_dir)
                shard_present(
                    quarantine_path, row, f"newly quarantined candidate shard {index}")
                staged = True
            if staged:
                # Revalidate immediately before unlink. The deterministic
                # quarantine path makes a crash after rename resumable.
                quant.inspect_exact_file(
                    quarantine_path, row["sha256"], row["size_bytes"],
                    f"deletable quarantined candidate shard {index}")
                quarantine_path.unlink()
                quant.fsync_directory(candidate_dir)
        quant.fsync_directory(candidate_dir)
        for row, quarantine_row in zip(declared, quarantine, strict=True):
            for path in (Path(row["path"]), Path(quarantine_row["path"])):
                if path.exists() or path.is_symlink():
                    raise BuilderError(
                        "candidate shard reappeared before retirement completion")
        result = {
            "schema": RECONSTRUCTABLE_RETIREMENT_COMPLETE_SCHEMA,
            "status": "complete", "candidate_id": args.candidate_id,
            "authorization": {"path": str(output), "sha256": output_sha256},
            "deleted_shards": declared, "deleted_bytes": payload["total_artifact_bytes"],
            "build_record_retained": record_evidence,
            "bf16_cache_manifest": cache_evidence,
            "companion_inventory_manifest": companion_evidence,
            "recoverable": True, "publishes": False,
        }
        if completion.exists() and not completion.is_symlink():
            existing, _ = quant.read_exact_json_file(
                completion, quant.sha256_file(completion), "existing retirement completion")
            if existing != result:
                raise BuilderError("existing retirement completion differs")
        elif completion.is_symlink():
            raise BuilderError("retirement completion must not be a symlink")
        else:
            write_json_atomic_noreplace(completion, result)
        return {"status": "complete", "candidate_id": args.candidate_id,
                "completion": str(completion),
                "completion_sha256": quant.sha256_file(completion),
                "deleted_bytes": result["deleted_bytes"], "recoverable": True}


def restore_reconstructable(args: argparse.Namespace) -> dict[str, Any]:
    """Restore retired shard paths only from a byte-identical fresh rebuild."""
    completion, completion_evidence = quant.read_exact_json_file(
        args.retirement_completion.absolute(), args.retirement_completion_sha256,
        "reconstructable retirement completion")
    if (completion.get("schema") != RECONSTRUCTABLE_RETIREMENT_COMPLETE_SCHEMA
            or completion.get("status") != "complete"
            or completion.get("recoverable") is not True
            or completion.get("publishes") is not False):
        raise BuilderError("retirement completion is not reconstructable")
    authorization_desc = completion.get("authorization") or {}
    authorization, authorization_evidence = quant.read_exact_json_file(
        Path(str(authorization_desc.get("path", ""))).absolute(),
        authorization_desc.get("sha256"), "reconstructable retirement authorization")
    if (authorization.get("schema") != RECONSTRUCTABLE_RETIREMENT_SCHEMA
            or authorization.get("candidate_id") != completion.get("candidate_id")):
        raise BuilderError("retirement authorization and completion differ")
    candidate_dir = Path(str(authorization.get("candidate_dir", ""))).absolute()
    record_desc = authorization.get("build_record") or {}
    original, original_evidence = quant.read_exact_json_file(
        candidate_dir / "qwen-quant-build-record.json", record_desc.get("sha256"),
        "retained original build record")
    expected = declared_candidate_shards(candidate_dir, original)
    if expected != authorization.get("shards"):
        raise BuilderError("retirement authorization shard inventory differs from its build record")

    rebuilt_dir = args.rebuilt_candidate_dir.absolute()
    if rebuilt_dir.is_symlink() or not rebuilt_dir.is_dir() or rebuilt_dir == candidate_dir:
        raise BuilderError("rebuilt candidate must be a distinct non-symlink directory")
    rebuilt_record_path = rebuilt_dir / "qwen-quant-build-record.json"
    rebuilt, rebuilt_evidence = quant.read_exact_json_file(
        rebuilt_record_path, args.rebuilt_build_record_sha256,
        "fresh reconstruction build record")
    if rebuilt.get("status") != "complete":
        raise BuilderError("fresh reconstruction build record is not complete")
    contract = authorization.get("reconstruction_contract")
    if (_record_reconstruction_contract(rebuilt) != contract
            or _record_reconstruction_contract(original) != contract):
        raise BuilderError("fresh rebuild construction inputs differ from the retired candidate")
    rebuilt_rows = declared_candidate_shards(rebuilt_dir, rebuilt)
    expected_bytes = [(row.get("size_bytes"), row.get("sha256")) for row in expected]
    actual_bytes = [(row.get("size_bytes"), row.get("sha256")) for row in rebuilt_rows]
    if actual_bytes != expected_bytes:
        raise BuilderError("fresh rebuild does not reproduce every original shard byte hash")
    if rebuilt_dir.stat().st_dev != candidate_dir.stat().st_dev:
        raise BuilderError("reconstruction requires same-filesystem atomic shard moves")

    output = args.output.absolute()
    if output.is_relative_to(candidate_dir) or output.is_relative_to(rebuilt_dir):
        raise BuilderError("reconstruction receipt must live outside both candidate directories")
    workset_root = Path(str(authorization.get("workset_root", ""))).absolute()
    with WorksetLease(workset_root):
        _validate_reconstruction_cache(original)
        for index, (rebuilt_row, expected_row) in enumerate(
                zip(rebuilt_rows, expected, strict=True)):
            source = Path(rebuilt_row["path"])
            destination = Path(expected_row["path"])
            source_present = source.exists() and not source.is_symlink()
            destination_present = destination.exists() and not destination.is_symlink()
            if source.is_symlink() or destination.is_symlink():
                raise BuilderError("reconstruction shard path was replaced by a symlink")
            if source_present and not destination_present:
                quant.inspect_exact_file(source, rebuilt_row["sha256"],
                                         rebuilt_row["size_bytes"],
                                         f"fresh reconstruction shard {index}")
                quant.rename_directory_noreplace(source, destination)
            elif destination_present and not source_present:
                quant.inspect_exact_file(destination, expected_row["sha256"],
                                         expected_row["size_bytes"],
                                         f"already restored candidate shard {index}")
            else:
                raise BuilderError("reconstruction requires exactly one source or restored shard")
        quant.fsync_directory(candidate_dir)
        quant.fsync_directory(rebuilt_dir)
        for index, row in enumerate(expected):
            quant.inspect_exact_file(Path(row["path"]), row["sha256"], row["size_bytes"],
                                     f"restored candidate shard {index}")
        receipt = {
            "schema": RECONSTRUCTION_RECEIPT_SCHEMA,
            "status": "byte_identical_restore_complete",
            "candidate_id": completion["candidate_id"],
            "retirement_completion": completion_evidence,
            "retirement_authorization": authorization_evidence,
            "original_build_record": original_evidence,
            "reconstruction_build_record": rebuilt_evidence,
            "restored_shards": expected,
            "all_original_hashes_reproduced": True,
            "publishes": False,
        }
        if output.exists() and not output.is_symlink():
            existing, _ = quant.read_exact_json_file(
                output, quant.sha256_file(output), "existing reconstruction receipt")
            if existing != receipt:
                raise BuilderError("existing reconstruction receipt differs")
        elif output.is_symlink():
            raise BuilderError("reconstruction receipt must not be a symlink")
        else:
            write_json_atomic_noreplace(output, receipt)
        return {"status": "complete", "candidate_id": completion["candidate_id"],
                "receipt": str(output), "receipt_sha256": quant.sha256_file(output),
                "restored_bytes": sum(row["size_bytes"] for row in expected)}


def add_cgroup_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--memory-limit-bytes", type=int, required=True)
    parser.add_argument("--cgroup-memory-max-path", type=Path,
                        default=Path("/sys/fs/cgroup/memory.max"))
    parser.add_argument("--cgroup-memory-swap-max-path", type=Path,
                        default=Path("/sys/fs/cgroup/memory.swap.max"))
    parser.add_argument("--cgroup-memory-peak-path", type=Path,
                        default=Path("/sys/fs/cgroup/memory.peak"))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-cache")
    prepare.add_argument("--profile", type=Path, required=True)
    prepare.add_argument("--builder-container-digest", required=True)
    prepare.add_argument("--converter-environment-lock", type=Path, required=True)
    prepare.add_argument("--converter-environment-lock-sha256", required=True)
    prepare.add_argument("--snapshot-dir", type=Path, required=True)
    prepare.add_argument("--snapshot-revision", required=True)
    prepare.add_argument("--llama-cpp-dir", type=Path, required=True)
    prepare.add_argument("--rocmfpx-dir", type=Path, required=True)
    prepare.add_argument("--ember-dir", type=Path, required=True)
    prepare.add_argument("--ember-revision", required=True)
    prepare.add_argument("--quantizer", type=Path, required=True)
    prepare.add_argument("--gguf-splitter", type=Path, required=True)
    prepare.add_argument("--cache-root", type=Path, required=True)
    add_cgroup_args(prepare)

    companion = commands.add_parser("make-companion-inventory")
    companion.add_argument("--profile", type=Path, required=True)
    companion.add_argument("--quantization-arm", required=True)
    companion.add_argument("--bf16-cache-manifest", type=Path, required=True)
    companion.add_argument("--bf16-cache-manifest-sha256", required=True)
    companion.add_argument("--mtp", type=Path, required=True)
    companion.add_argument("--mtp-bytes", type=int, required=True)
    companion.add_argument("--mtp-sha256", required=True)
    companion.add_argument("--mtp-matrix-quant-contract", required=True)
    companion.add_argument("--mtp-export-manifest", type=Path, required=True)
    companion.add_argument("--mtp-export-manifest-sha256", required=True)
    companion.add_argument("--output", type=Path, required=True)

    build = commands.add_parser("build-candidate")
    build.add_argument("--candidate-id", required=True)
    build.add_argument("--builder-container-digest", required=True)
    build.add_argument("--profile", type=Path, required=True)
    build.add_argument("--snapshot-dir", type=Path, required=True)
    build.add_argument("--snapshot-revision", required=True)
    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument("--intervention-manifest", type=Path)
    source.add_argument("--stock-control", action="store_true")
    build.add_argument("--stock-capture-manifest", type=Path)
    build.add_argument("--stock-capture-manifest-sha256")
    build.add_argument("--llama-cpp-dir", type=Path, required=True)
    build.add_argument("--rocmfpx-dir", type=Path, required=True)
    build.add_argument("--ember-dir", type=Path, required=True)
    build.add_argument("--ember-revision", required=True)
    build.add_argument("--quantizer", type=Path, required=True)
    build.add_argument("--gguf-splitter", type=Path, required=True)
    build.add_argument("--bf16-cache-manifest", type=Path, required=True)
    build.add_argument("--bf16-cache-manifest-sha256", required=True)
    build.add_argument("--companion-inventory", type=Path, required=True)
    build.add_argument("--companion-inventory-sha256", required=True)
    build.add_argument("--mtp-matrix-quant-contract", required=True,
                       choices=("Q4_0_ROCMI4", "Q4_0_ROCMFP4_FAST"))
    build.add_argument("--ttm-pages-limit-path", type=Path,
                       default=Path("/sys/module/ttm/parameters/pages_limit"))
    build.add_argument("--quantization-arm", required=True)
    build.add_argument("--bakeoff-plan", type=Path)
    build.add_argument("--bakeoff-plan-sha256")
    build.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    build.add_argument("--min-free-gib", type=int, default=100)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--workset-root", type=Path, required=True)
    add_cgroup_args(build)

    assessment = commands.add_parser("record-assessment")
    assessment.add_argument("--candidate-id", required=True)
    assessment.add_argument("--candidate-dir", type=Path, required=True)
    assessment.add_argument("--build-record-sha256", required=True)
    assessment.add_argument("--assessment", type=Path, required=True)
    assessment.add_argument("--assessment-sha256", required=True)
    assessment.add_argument("--selected", action=argparse.BooleanOptionalAction,
                            required=True)
    assessment.add_argument("--output", type=Path, required=True)

    retire = commands.add_parser("retire-captured-stock")
    retire.add_argument("--stock-dir", type=Path, required=True)
    retire.add_argument("--build-record-sha256", required=True)
    retire.add_argument("--stock-capture-manifest", type=Path, required=True)
    retire.add_argument("--stock-capture-manifest-sha256", required=True)
    retire.add_argument("--workset-root", type=Path, required=True)
    retire.add_argument("--output", type=Path, required=True)

    delete = commands.add_parser("delete-loser")
    delete.add_argument("--assessment-bundle", type=Path, required=True)
    delete.add_argument("--assessment-bundle-sha256", required=True)
    delete.add_argument("--execute", action="store_true")

    authority = commands.add_parser("authorize-rolling-retention")
    authority.add_argument("--plan", type=Path, required=True)
    authority.add_argument("--plan-sha256", required=True)
    authority.add_argument("--phase", choices=("sweep", "format"), required=True)
    authority.add_argument("--accumulator", type=Path, required=True)
    authority.add_argument("--accumulator-sha256", required=True)
    authority.add_argument("--accumulator-bundle", type=Path, required=True)
    authority.add_argument("--accumulator-bundle-sha256", required=True)
    authority.add_argument("--output", type=Path, required=True)

    sealed = commands.add_parser("authorize-sealed-retention")
    sealed.add_argument("--plan", type=Path, required=True)
    sealed.add_argument("--plan-sha256", required=True)
    sealed.add_argument("--ledger", type=Path, required=True)
    sealed.add_argument("--ledger-sha256", required=True)
    sealed.add_argument("--ledger-bundle", type=Path, required=True)
    sealed.add_argument("--ledger-bundle-sha256", required=True)
    sealed.add_argument("--output", type=Path, required=True)

    reconstructable = commands.add_parser("retire-reconstructable")
    reconstructable.add_argument("--retention-authority", type=Path, required=True)
    reconstructable.add_argument("--retention-authority-sha256", required=True)
    reconstructable.add_argument("--candidate-id", required=True)
    reconstructable.add_argument("--candidate-dir", type=Path, required=True)
    reconstructable.add_argument("--build-record-sha256", required=True)
    reconstructable.add_argument("--output", type=Path, required=True)
    reconstructable.add_argument("--execute", action="store_true")

    restore = commands.add_parser("restore-reconstructable")
    restore.add_argument("--retirement-completion", type=Path, required=True)
    restore.add_argument("--retirement-completion-sha256", required=True)
    restore.add_argument("--rebuilt-candidate-dir", type=Path, required=True)
    restore.add_argument("--rebuilt-build-record-sha256", required=True)
    restore.add_argument("--output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare-cache":
            value = prepare_cache(args)
        elif args.command == "make-companion-inventory":
            value = make_companion_inventory(args)
        elif args.command == "build-candidate":
            value = build_candidate(args)
        elif args.command == "record-assessment":
            value = record_assessment(args)
        elif args.command == "retire-captured-stock":
            value = retire_captured_stock(args)
        elif args.command == "delete-loser":
            value = delete_loser(args)
        elif args.command == "authorize-rolling-retention":
            value = authorize_rolling_retention(args)
        elif args.command == "authorize-sealed-retention":
            value = authorize_sealed_retention(args)
        elif args.command == "retire-reconstructable":
            value = retire_reconstructable(args)
        elif args.command == "restore-reconstructable":
            value = restore_reconstructable(args)
        else:
            raise BuilderError(f"unsupported command: {args.command}")
    except (BuilderError, quant.PipelineError, OSError) as exc:
        print(f"qwen_candidate_builder.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
