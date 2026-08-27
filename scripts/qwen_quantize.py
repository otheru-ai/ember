#!/usr/bin/env python3
"""Plan or execute the pinned Qwen3.8 Flash Next ROCmI4 conversion.

Dry-run planning is the default.  ``--execute`` is the only path that invokes
the converter and quantizer.  The program uses no network client, constructs a
credential-free subprocess environment, and never publishes artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import datetime as dt
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import resource
import shlex
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, BinaryIO


GIB = 1024 ** 3
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<number>[0-9]{5})-of-(?P<count>[0-9]{5})\.gguf$")
GGUF_TYPES = {
    0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32",
    5: "INT32", 6: "FLOAT32", 7: "BOOL", 8: "STRING", 9: "ARRAY",
    10: "UINT64", 11: "INT64", 12: "FLOAT64",
}
FIXED_FORMATS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
    7: "?", 10: "Q", 11: "q", 12: "d",
}
PLE_SUFFIXES = {
    "ple_embedding.layer_multipliers": "qwen4exp.ple.layer_multipliers",
    "ple_embedding.ngram_heads_offsets": "qwen4exp.ple.head_offsets",
    "ple_embedding.ngram_heads_vocab_sizes": "qwen4exp.ple.head_vocab_sizes",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QWEN_QSA_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47}
INTERVENTION_TARGET_RE = re.compile(r"^blk\.([0-9]+)\.(attn_output|ssm_out)\.weight$")
DEFAULT_QUANTIZATION_ARM = "profile-default-rocmi4"
COMPANION_INVENTORY_SCHEMA = "ember.qwen3.8-flash-next.companion-inventory.v1"
COMPANION_ROLES = ("mtp", "vision_mmproj")
BF16_CACHE_SCHEMA = "ember.qwen3.8-flash-next.bf16-cache.v1"
GGUF_WRITER_TEMP_NAME_RE = re.compile(r"^tmp[a-z0-9_]{8}$")
TTM_PAGE_BYTES = 4096
CANONICAL_TTM_PAGES_LIMIT = Path("/sys/module/ttm/parameters/pages_limit")
DIRECT_IO_MIN_BYTES = 512 * 1024 * 1024
DIRECT_IO_BLOCK_BYTES = 8 * 1024 * 1024
SUPPORTED_TENSOR_FORMATS = {
    "Q4_0_ROCMI4": 108,
    "Q6_K": 14,
    "Q4_0_ROCMFP4_FAST": 101,
}
MTP_QUANTIZED_MATRIX_NAMES = frozenset({
    "mtp_hc_down.weight", "mtp_hc_up.weight",
    "mtp.hc_attn_inject.weight", "mtp.hc_attn_down.weight", "mtp.hc_attn_up.weight",
    "mtp.ffn_gate_up_exps.weight", "mtp.ffn_down_exps.weight",
    "mtp.ffn_gate_shexp.weight", "mtp.ffn_up_shexp.weight", "mtp.ffn_down_shexp.weight",
    "mtp.hc_ffn_inject.weight", "mtp.hc_ffn_down.weight", "mtp.hc_ffn_up.weight",
    "mtp.indexer.q_proj.weight", "mtp.indexer.k_proj.weight",
    "mtp.attn_output.weight", "mtp.attn_k.weight", "mtp.attn_q.weight", "mtp.attn_v.weight",
    "mtp_fc_emb.weight", "mtp_fc_hc.weight",
})
MTP_BF16_TENSOR_NAMES = frozenset({
    "mtp_hc_norm.weight", "mtp.hc_attn_norm.weight",
    "mtp.ffn_gate_inp.weight", "mtp.ffn_gate_inp_shexp.weight",
    "mtp.hc_ffn_norm.weight", "mtp.indexer.k_norm.weight", "mtp.indexer.q_norm.weight",
    "mtp.attn_k_norm.weight", "mtp.attn_q_norm.weight",
    "mtp_pre_emb_norm.weight", "mtp_pre_hc_norm.weight",
})
ROCMFP4_FAST_MATRIX_PATTERNS = [
    (r"^blk\.[0-9]+\.(hc_(attn|ffn)_(down|up|inject)|"
     r"ffn_(gate_up|down)_exps|ffn_(gate|up|down)_shexp|"
     r"attn_(q|k|v|output|qkv|gate)|indexer\.(q_proj|k_proj)|"
     r"ssm_(alpha|beta|out)|ple_(key|value))\.weight$"),
    r"^output_hc_(down|up)\.weight$",
]
ROCMFP4_FAST_ROUTED_EXPERT_PATTERN = (
    r"^blk\.[0-9]+\.ffn_(gate_up|down)_exps\.weight$"
)


class PipelineError(ValueError):
    pass


def read_cgroup_counter(path: Path, label: str) -> int:
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise PipelineError(f"cannot read {label} cgroup evidence at {path}: {exc}") from exc
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise PipelineError(f"{label} cgroup evidence is not a finite byte count: {value!r}")
    return int(value)


def validate_conversion_cgroup(args: argparse.Namespace) -> dict[str, int] | None:
    """Bind the certification container's real cgroup-v2 memory boundary."""
    if args.conversion_memory_limit_bytes is None:
        return None
    memory_limit = read_cgroup_counter(args.cgroup_memory_max_path, "memory.max")
    swap_limit = read_cgroup_counter(args.cgroup_memory_swap_max_path, "memory.swap.max")
    if memory_limit != args.conversion_memory_limit_bytes:
        raise PipelineError(
            "conversion cgroup memory.max differs from --conversion-memory-limit-bytes")
    if swap_limit != 0:
        raise PipelineError("bounded conversion certification requires memory.swap.max=0")
    return {"memory_limit_bytes": memory_limit, "swap_limit_bytes": swap_limit}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_digest_buffered(path: Path, size: int, algorithm: str) -> str:
    if algorithm == "sha256":
        digest = hashlib.sha256()
    elif algorithm == "git_blob_sha1":
        digest = hashlib.sha1(f"blob {size}\0".encode("ascii"), usedforsecurity=False)
    else:
        raise PipelineError(f"unsupported snapshot digest algorithm: {algorithm}")
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(DIRECT_IO_BLOCK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_digest_direct(path: Path, size: int, algorithm: str) -> str:
    """Hash a large snapshot artifact through O_DIRECT, outside page cache."""
    dd = shutil.which("dd", path=os.defpath)
    if dd is None:
        raise PipelineError("dd is required for large snapshot O_DIRECT integrity reads")
    if algorithm == "sha256":
        digest = hashlib.sha256()
    elif algorithm == "git_blob_sha1":
        digest = hashlib.sha1(f"blob {size}\0".encode("ascii"), usedforsecurity=False)
    else:
        raise PipelineError(f"unsupported snapshot digest algorithm: {algorithm}")
    try:
        process = subprocess.Popen(
            [dd, f"if={path}", "iflag=direct", "bs=8M", "status=none"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise PipelineError(f"cannot start O_DIRECT integrity read for {path}: {exc}") from exc
    assert process.stdout is not None
    total = 0
    for chunk in iter(lambda: process.stdout.read(DIRECT_IO_BLOCK_BYTES), b""):
        total += len(chunk)
        digest.update(chunk)
    process.stdout.close()
    assert process.stderr is not None
    stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    process.stderr.close()
    returncode = process.wait()
    if returncode != 0 or total != size:
        detail = f": {stderr}" if stderr else ""
        raise PipelineError(f"O_DIRECT integrity read failed for {path}{detail}")
    return digest.hexdigest()


def snapshot_artifact_digest(path: Path, size: int, algorithm: str) -> tuple[str, str]:
    if size >= DIRECT_IO_MIN_BYTES:
        return snapshot_digest_direct(path, size, algorithm), "o_direct_dd_v1"
    return snapshot_digest_buffered(path, size, algorithm), "buffered_small_file_v1"


def sha256_open_file(stream: Any, size: int, path: Path) -> tuple[str, str]:
    """Hash an already identity-bound file, using O_DIRECT when it is large."""
    if size < DIRECT_IO_MIN_BYTES:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(DIRECT_IO_BLOCK_BYTES), b""):
            digest.update(chunk)
        return digest.hexdigest(), "buffered_small_file_v1"
    dd = shutil.which("dd", path=os.defpath)
    if dd is None:
        raise PipelineError("dd is required for large companion O_DIRECT integrity reads")
    descriptor = stream.fileno()
    try:
        process = subprocess.Popen(
            [dd, f"if=/proc/self/fd/{descriptor}", "iflag=direct", "bs=8M", "status=none"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(descriptor,),
        )
    except OSError as exc:
        raise PipelineError(f"cannot start O_DIRECT integrity read for {path}: {exc}") from exc
    assert process.stdout is not None
    digest = hashlib.sha256()
    total = 0
    for chunk in iter(lambda: process.stdout.read(DIRECT_IO_BLOCK_BYTES), b""):
        total += len(chunk)
        digest.update(chunk)
    process.stdout.close()
    assert process.stderr is not None
    stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    process.stderr.close()
    returncode = process.wait()
    if returncode != 0 or total != size:
        detail = f": {stderr}" if stderr else ""
        raise PipelineError(f"O_DIRECT integrity read failed for {path}{detail}")
    return digest.hexdigest(), "o_direct_bound_fd_v1"


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"{label} must be a JSON object")
    return value


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineError(f"{field} must be a JSON object")
    return value


def require_exact_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise PipelineError(
            f"{field} keys differ from the exact contract; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


class SnapshotReadLease:
    """Hold the fetcher's coordination inode shared for the full transaction."""

    def __init__(self, snapshot: Path) -> None:
        self.path = snapshot / ".ember-fetch.lock"
        self.descriptor = -1

    def __enter__(self) -> "SnapshotReadLease":
        if not self.path.parent.is_dir():
            raise PipelineError(f"snapshot directory does not exist: {self.path.parent}")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            self.descriptor = os.open(self.path, flags, 0o600)
            status = os.fstat(self.descriptor)
            if not stat.S_ISREG(status.st_mode):
                raise PipelineError("snapshot coordination inode must be a regular file")
            fcntl.flock(self.descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            if self.descriptor >= 0:
                os.close(self.descriptor)
                self.descriptor = -1
            raise PipelineError("the resumable source download still owns the snapshot") from exc
        except PipelineError:
            if self.descriptor >= 0:
                os.close(self.descriptor)
                self.descriptor = -1
            raise
        except OSError as exc:
            if self.descriptor >= 0:
                os.close(self.descriptor)
                self.descriptor = -1
            if exc.errno == errno.ELOOP:
                raise PipelineError(
                    "snapshot coordination file .ember-fetch.lock must be a regular non-symlink file"
                ) from exc
            raise PipelineError(
                f"cannot acquire snapshot read lease {self.path}: {exc}") from exc
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        if self.descriptor >= 0:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = -1


def inspect_exact_file(
    path: Path, expected_sha256: str, expected_bytes: int, label: str,
) -> dict[str, Any]:
    """Hash one named regular file without following symlinks or path swaps."""
    if not path.is_absolute():
        raise PipelineError(f"{label} path must be absolute")
    if not isinstance(expected_sha256, str) or SHA256_RE.fullmatch(expected_sha256) is None:
        raise PipelineError(f"{label} sha256 must be a lowercase 64-character digest")
    if (not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool)
            or expected_bytes < 1):
        raise PipelineError(f"{label} size_bytes must be a positive integer")
    try:
        named = os.lstat(path)
    except OSError as exc:
        raise PipelineError(f"cannot inspect {label} {path}: {exc}") from exc
    if not stat.S_ISREG(named.st_mode):
        raise PipelineError(f"{label} must be a regular non-symlink file: {path}")
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
                raise PipelineError(f"{label} identity changed before hashing")
            actual_sha256, read_method = sha256_open_file(
                stream, before.st_size, path)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise PipelineError(f"cannot hash {label} {path}: {exc}") from exc
    if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
        raise PipelineError(f"{label} changed while it was hashed")
    if after.st_size != expected_bytes:
        raise PipelineError(
            f"{label} size mismatch: expected {expected_bytes}, got {after.st_size}")
    if actual_sha256 != expected_sha256:
        raise PipelineError(f"{label} SHA-256 mismatch")
    return {
        "path": str(path),
        "size_bytes": after.st_size,
        "sha256": actual_sha256,
        "integrity_read_method": read_method,
        "regular_file": True,
        "symlink": False,
    }


def read_exact_json_file(
    path: Path, expected_sha256: str, label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash and parse one bounded JSON manifest from the same open inode."""
    if not path.is_absolute():
        raise PipelineError(f"{label} path must be absolute")
    if SHA256_RE.fullmatch(expected_sha256 or "") is None:
        raise PipelineError(f"{label} sha256 must be a lowercase 64-character digest")
    try:
        named = os.lstat(path)
        if not stat.S_ISREG(named.st_mode):
            raise PipelineError(f"{label} must be a regular non-symlink file: {path}")
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
                raise PipelineError(f"{label} identity changed before reading")
            raw = stream.read(16 * 1024 * 1024 + 1)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise PipelineError(f"cannot read {label} {path}: {exc}") from exc
    if len(raw) > 16 * 1024 * 1024:
        raise PipelineError(f"{label} exceeds the 16 MiB manifest bound")
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise PipelineError(f"{label} changed while it was read")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise PipelineError(f"{label} SHA-256 mismatch")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"{label} must be a JSON object")
    evidence = {
        "path": str(path), "size_bytes": len(raw), "sha256": actual_sha256,
        "integrity_read_method": "buffered_bound_fd_v1",
        "regular_file": True, "symlink": False,
    }
    return value, evidence


def validate_bf16_cache_manifest(
    path: Path | None,
    expected_sha256: str | None,
    profile: dict[str, Any],
    profile_sha256: str,
    tools: dict[str, Any],
) -> dict[str, Any] | None:
    """Verify a content-addressed, read-only BF16 conversion cache.

    The cache removes the 360 GB HF -> BF16 conversion from every intervention
    arm.  It does not cache modified weights: the quantizer still applies the
    selected direction while encoding each candidate.  Every reuse rehashes the
    shards so a stale or replaced cache cannot silently enter a bakeoff.
    """
    if path is None and expected_sha256 is None:
        return None
    if path is None or expected_sha256 is None:
        raise PipelineError(
            "--bf16-cache-manifest and --bf16-cache-manifest-sha256 must be supplied together")
    manifest, manifest_file = read_exact_json_file(
        path.absolute(), expected_sha256, "BF16 cache manifest")
    require_exact_keys(
        manifest,
        {"schema", "cache_id", "source", "profile", "toolchain", "conversion",
         "resources", "measurement", "main", "vision_mmproj"},
        "BF16 cache manifest",
    )
    if manifest.get("schema") != BF16_CACHE_SCHEMA:
        raise PipelineError("unsupported BF16 cache manifest schema")
    cache_id = manifest.get("cache_id")
    if not isinstance(cache_id, str) or SHA256_RE.fullmatch(cache_id) is None:
        raise PipelineError("BF16 cache id must be a lowercase SHA-256 digest")
    source = require_mapping(manifest.get("source"), "BF16 cache source")
    expected_source = profile["source"]
    if source != {
        "repo_id": expected_source["repo_id"],
        "revision": expected_source["revision"],
        "snapshot_inventory_sha256": expected_source["snapshot_inventory_sha256"],
    }:
        raise PipelineError("BF16 cache source differs from the pinned release profile")
    profile_row = require_mapping(manifest.get("profile"), "BF16 cache profile")
    if profile_row != {"profile_id": profile.get("profile_id"),
                       "sha256": profile_sha256}:
        raise PipelineError("BF16 cache profile differs from the current release profile")
    toolchain = require_mapping(manifest.get("toolchain"), "BF16 cache toolchain")
    require_exact_keys(toolchain, {
        "llama_cpp_revision", "llama_cpp_base_revision", "converter_sha256",
        "qwen4exp_converter_sha256", "ple_cgroup_writeback_patch_sha256",
        "gguf_splitter_sha256", "converter_environment_lock_sha256",
        "converter_environment_lock_bytes", "builder_container_digest",
    }, "BF16 cache toolchain")
    expected_toolchain = {
        "llama_cpp_revision": tools["llama_cpp_revision"],
        "llama_cpp_base_revision": tools["llama_cpp_base_revision"],
        "converter_sha256": tools["converter_sha256"],
        "qwen4exp_converter_sha256": tools["qwen4exp_converter_sha256"],
        "ple_cgroup_writeback_patch_sha256": tools["ple_cgroup_writeback_patch_sha256"],
        "gguf_splitter_sha256": tools.get("gguf_splitter_sha256"),
    }
    if any(toolchain.get(key) != value for key, value in expected_toolchain.items()):
        raise PipelineError("BF16 cache toolchain differs from the current pinned tools")
    if (SHA256_RE.fullmatch(str(toolchain.get("converter_environment_lock_sha256", ""))) is None
            or not isinstance(toolchain.get("converter_environment_lock_bytes"), int)
            or toolchain["converter_environment_lock_bytes"] < 1
            or re.fullmatch(r"sha256:[0-9a-f]{64}",
                            str(toolchain.get("builder_container_digest", ""))) is None):
        raise PipelineError("BF16 cache converter environment/image provenance is malformed")
    conversion = require_mapping(manifest.get("conversion"), "BF16 cache conversion")
    cleanup = conversion.get("gguf_writer_temp_cleanup")
    conversion_recipe = {key: value for key, value in conversion.items()
                         if key != "gguf_writer_temp_cleanup"}
    if conversion_recipe != {
        "outtype": "bf16", "split_max_size": "48G", "use_temp_file": True,
        "main_storage_policy": "mostly_bf16_with_f32_ple",
        "ple_intermediate_storage":
            "F32_streamed_to_temp_file_then_release_quant_override",
        "ple_ggml_tensor_type": 0,
        "mmproj": {"outtype": "bf16", "converter_option": "--mmproj"},
    }:
        raise PipelineError("BF16 cache conversion recipe is not canonical")
    if (not isinstance(cleanup, dict)
            or set(cleanup) != {"policy", "main_removed", "mmproj_removed"}
            or cleanup.get("policy") != "exact_converter_private_tmp_residue_v2"):
        raise PipelineError("BF16 cache lacks its exact GGUFWriter temp cleanup evidence")
    for label in ("main_removed", "mmproj_removed"):
        rows = cleanup.get(label)
        if (not isinstance(rows, list)
                or any(not isinstance(row, dict)
                       or set(row) != {"name", "size_bytes", "mode"}
                       or GGUF_WRITER_TEMP_NAME_RE.fullmatch(str(row.get("name", ""))) is None
                       or not isinstance(row.get("size_bytes"), int)
                       or isinstance(row.get("size_bytes"), bool)
                       or row["size_bytes"] < 0 or row.get("mode") != 0o600
                       for row in rows)):
            raise PipelineError("BF16 cache GGUFWriter temp cleanup rows are malformed")
    resources = require_mapping(manifest.get("resources"), "BF16 cache resources")
    if (not isinstance(resources.get("free_bytes"), int)
            or resources.get("free_bytes", 0) < 1152 * GIB
            or not isinstance(resources.get("physical_ram_bytes"), int)
            or resources.get("physical_ram_bytes", 0) < 120 * GIB):
        raise PipelineError("BF16 cache lacks the pinned disk/RAM construction preflight")
    measurement = require_mapping(manifest.get("measurement"), "BF16 cache measurement")
    if (measurement.get("status") != "measured_target_cgroup_v2"
            or measurement.get("memory_limit_bytes") != 134217728000
            or measurement.get("swap_limit_bytes") != 0
            or not isinstance(measurement.get("cgroup_peak_bytes"), int)
            or measurement["cgroup_peak_bytes"] > measurement["memory_limit_bytes"]):
        raise PipelineError("BF16 cache lacks a passing 125 GiB no-swap cgroup measurement")
    main = require_mapping(manifest.get("main"), "BF16 cache main")
    require_exact_keys(main, {"base_path", "content_sha256", "shards", "gguf", "ple"},
                       "BF16 cache main")
    ple = require_mapping(main.get("ple"), "BF16 cache PLE metadata")
    expected_ple: dict[str, list[int]] = {}
    if set(ple) != set(PLE_SUFFIXES):
        raise PipelineError("BF16 cache PLE metadata keys are incomplete")
    for key, value in ple.items():
        if (not isinstance(value, list) or not value
                or any(not isinstance(item, int) or isinstance(item, bool) for item in value)):
            raise PipelineError(f"BF16 cache PLE metadata {key} is malformed")
        expected_ple[key] = value
    base_path = main.get("base_path")
    if not isinstance(base_path, str) or PurePosixPath(base_path).name != base_path:
        raise PipelineError("BF16 cache main base_path must be one safe filename")
    cache_dir = path.absolute().parent
    shards = main.get("shards")
    if not isinstance(shards, list) or not shards:
        raise PipelineError("BF16 cache main shard inventory is empty")
    normalized_shards: list[dict[str, Any]] = []
    content_rows: list[dict[str, Any]] = []
    shard_identities: list[tuple[int, int, int, int]] = []
    for index, row_value in enumerate(shards):
        row = require_mapping(row_value, f"BF16 cache shard {index}")
        require_exact_keys(row, {"name", "size_bytes", "sha256"},
                           f"BF16 cache shard {index}")
        name = row.get("name")
        if not isinstance(name, str) or PurePosixPath(name).name != name:
            raise PipelineError("BF16 cache shard name must be one safe filename")
        if (not isinstance(row.get("sha256"), str)
                or SHA256_RE.fullmatch(row["sha256"]) is None
                or not isinstance(row.get("size_bytes"), int)
                or isinstance(row["size_bytes"], bool) or row["size_bytes"] < 1):
            raise PipelineError("BF16 cache shard digest/size is malformed")
        shard_path = cache_dir / name
        try:
            status = os.lstat(shard_path)
        except OSError as exc:
            raise PipelineError(f"cannot inspect BF16 cache shard {index}: {exc}") from exc
        if not stat.S_ISREG(status.st_mode) or status.st_size != row["size_bytes"]:
            raise PipelineError("BF16 cache shard must be an exact regular file")
        shard_identities.append(
            (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns))
        normalized_shards.append({
            "path": str(shard_path), "size_bytes": row["size_bytes"],
            "sha256": row["sha256"], "regular_file": True, "symlink": False,
            "integrity_read_method": "verify_gguf_set_sha256_v1",
        })
        content_rows.append({"name": name, "size_bytes": row["size_bytes"],
                             "sha256": row["sha256"]})
    content_sha256 = hashlib.sha256(json.dumps(
        content_rows, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
    if main.get("content_sha256") != content_sha256:
        raise PipelineError("BF16 cache main content digest differs from its shard inventory")
    paths = discover_gguf(cache_dir / base_path)
    if paths != [Path(row["path"]) for row in normalized_shards]:
        raise PipelineError("BF16 cache discovered shard set differs from its manifest")
    gguf = verify_gguf_set(paths, expected_ple, quantized=False, profile=profile)
    ple_tensors = [tensor for shard in paths for tensor in inspect_gguf(shard)["tensors"]
                   if tensor["name"] == profile["quantization"]["ple_tensor_name"]]
    if len(ple_tensors) != 1 or ple_tensors[0]["type"] != 0:
        raise PipelineError(
            "BF16 cache must store the one 204.8 GB PLE tensor as streaming F32")
    for index, (path_value, identity) in enumerate(zip(paths, shard_identities, strict=True)):
        status = os.lstat(path_value)
        current = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
        if current != identity:
            raise PipelineError(f"BF16 cache shard {index} changed during GGUF verification")
        actual = gguf["shards"][index]
        if (actual["size_bytes"] != shards[index]["size_bytes"]
                or actual["sha256"] != shards[index]["sha256"]):
            raise PipelineError(f"BF16 cache shard {index} differs from its manifest")
    recorded_gguf = main.get("gguf")
    if not isinstance(recorded_gguf, dict) or any(
            gguf.get(key) != recorded_gguf.get(key)
            for key in ("tensor_count", "tensor_names_sha256", "tensor_type_counts")):
        raise PipelineError("BF16 cache GGUF inventory differs from its creation record")
    mmproj = require_mapping(manifest.get("vision_mmproj"), "BF16 cache vision_mmproj")
    require_exact_keys(mmproj, {"name", "size_bytes", "sha256", "format", "gguf"},
                       "BF16 cache vision_mmproj")
    name = mmproj.get("name")
    if (not isinstance(name, str) or PurePosixPath(name).name != name
            or name != profile["artifact"]["required_companion_artifacts"][0]["filename"]
            or mmproj.get("format") != "BF16"):
        raise PipelineError("BF16 cache mmproj filename/format differs from the profile")
    mmproj_evidence = inspect_exact_file(
        cache_dir / name, mmproj.get("sha256"), mmproj.get("size_bytes"),
        "BF16 cache vision_mmproj")
    mmproj_gguf = validate_bf16_qwen_mmproj_gguf(cache_dir / name)
    if mmproj.get("gguf") != mmproj_gguf:
        raise PipelineError("BF16 cache mmproj GGUF inventory differs from its creation record")
    cache_address = hashlib.sha256(json.dumps({
        "main_content_sha256": content_sha256,
        "vision_mmproj": {"name": name, "size_bytes": mmproj["size_bytes"],
                           "sha256": mmproj["sha256"]},
    }, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
    if cache_id != cache_address or path.absolute().parent.name != f"bf16-{cache_id}":
        raise PipelineError("BF16 cache directory is not its exact content address")
    return {
        "schema": BF16_CACHE_SCHEMA, "cache_id": cache_id,
        "manifest": manifest_file, "source": source, "profile": profile_row,
        "toolchain": toolchain, "conversion": conversion,
        "resources": resources, "measurement": measurement,
        "main": {**main, "shards": normalized_shards, "gguf": gguf},
        "vision_mmproj": {**mmproj, **mmproj_evidence, "gguf": mmproj_gguf},
    }


def expected_mtp_matrix_contract(quantization_arm: dict[str, Any]) -> str:
    contract = quantization_arm.get("mtp_matrix_quant_contract")
    if contract not in ("Q4_0_ROCMI4", "Q4_0_ROCMFP4_FAST"):
        raise PipelineError("quantization arm lacks an explicit supported MTP matrix contract")
    return contract


def validate_companion_inventory(
    path: Path | None,
    expected_inventory_sha256: str | None,
    pages_limit_path: Path,
    profile: dict[str, Any],
    quantization_arm: dict[str, Any],
) -> dict[str, Any]:
    """Bind exact enabled companion bytes; absence is explicitly non-final."""
    if path is None and expected_inventory_sha256 is None:
        return {
            "status": "not_supplied",
            "schema": COMPANION_INVENTORY_SCHEMA,
            "required_roles": list(COMPANION_ROLES),
            "roles": [],
            "enabled_roles": [],
            "disabled_roles": [],
            "enabled_artifact_bytes": None,
            "fit_status": "pending_exact_companion_inventory",
            "final_release_eligibility": "pending_exact_companion_inventory",
            "estimated_bytes_used": False,
        }
    if path is None or expected_inventory_sha256 is None:
        raise PipelineError(
            "--companion-inventory and --companion-inventory-sha256 must be supplied together")
    inventory, inventory_file = read_exact_json_file(
        path.absolute(), expected_inventory_sha256, "companion inventory")
    require_exact_keys(inventory, {"schema", "source", "companions"},
                       "companion inventory")
    if inventory.get("schema") != COMPANION_INVENTORY_SCHEMA:
        raise PipelineError("unsupported companion inventory schema")
    source = require_mapping(inventory.get("source"), "companion inventory source")
    require_exact_keys(
        source, {"repo_id", "revision", "snapshot_inventory_sha256"},
        "companion inventory source",
    )
    expected_source = profile["source"]
    if source != {
        "repo_id": expected_source["repo_id"],
        "revision": expected_source["revision"],
        "snapshot_inventory_sha256": expected_source["snapshot_inventory_sha256"],
    }:
        raise PipelineError("companion inventory source does not match the pinned main snapshot")
    rows = inventory.get("companions")
    if not isinstance(rows, list) or len(rows) != len(COMPANION_ROLES):
        raise PipelineError("companion inventory must contain exactly MTP and vision_mmproj roles")
    by_role: dict[str, dict[str, Any]] = {}
    for index, row_value in enumerate(rows):
        row = require_mapping(row_value, f"companion inventory companions[{index}]")
        role = row.get("role")
        if role not in COMPANION_ROLES or role in by_role:
            raise PipelineError("companion inventory roles must be unique MTP and vision_mmproj")
        by_role[role] = row
    if set(by_role) != set(COMPANION_ROLES):
        raise PipelineError("companion inventory must contain exactly MTP and vision_mmproj roles")

    normalized_roles: list[dict[str, Any]] = []
    mtp = by_role["mtp"]
    require_exact_keys(
        mtp,
        {"role", "enabled", "path", "size_bytes", "sha256",
         "matrix_quant_contract", "export_manifest_path", "export_manifest_sha256"},
        "MTP companion",
    )
    if mtp.get("enabled") is not True:
        raise PipelineError("matching MTP companion must be enabled")
    expected_matrix = expected_mtp_matrix_contract(quantization_arm)
    if mtp.get("matrix_quant_contract") != expected_matrix:
        raise PipelineError(
            f"MTP matrix quant contract must match the selected arm ({expected_matrix})")
    mtp_path = mtp.get("path")
    if not isinstance(mtp_path, str) or not mtp_path:
        raise PipelineError("MTP companion path must be a non-empty absolute string")
    mtp_evidence = inspect_exact_file(
        Path(mtp_path), mtp.get("sha256"), mtp.get("size_bytes"), "MTP companion")
    mtp_gguf = validate_mtp_companion_gguf(
        Path(mtp_path), expected_matrix, profile["source"]["revision"])
    manifest_path = mtp.get("export_manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise PipelineError("MTP export manifest path must be a non-empty absolute string")
    mtp_manifest = validate_mtp_export_manifest(
        Path(manifest_path), mtp.get("export_manifest_sha256"),
        mtp_evidence, expected_matrix, profile)
    normalized_roles.append({
        "role": "mtp", "enabled": True, "artifact_present": True,
        "matrix_quant_contract": expected_matrix, "gguf_contract": mtp_gguf,
        "export_manifest": mtp_manifest,
        **mtp_evidence,
    })

    mmproj = by_role["vision_mmproj"]
    enabled = mmproj.get("enabled")
    if not isinstance(enabled, bool):
        raise PipelineError("vision_mmproj enabled must be boolean")
    if not enabled:
        require_exact_keys(mmproj, {"role", "enabled"}, "disabled vision_mmproj companion")
        normalized_roles.append({
            "role": "vision_mmproj", "enabled": False,
            "artifact_present": False, "counted_bytes": 0,
        })
    else:
        require_exact_keys(
            mmproj, {"role", "enabled", "path", "size_bytes", "sha256", "format"},
            "enabled vision_mmproj companion",
        )
        required_companions = profile["artifact"].get("required_companion_artifacts")
        if (not isinstance(required_companions, list) or len(required_companions) != 1
                or required_companions[0].get("role") != "vision_mmproj"):
            raise PipelineError("profile vision_mmproj artifact contract is malformed")
        expected_mmproj = required_companions[0]
        mmproj_path = mmproj.get("path")
        if not isinstance(mmproj_path, str) or not mmproj_path:
            raise PipelineError("vision_mmproj path must be a non-empty absolute string")
        if (Path(mmproj_path).name != expected_mmproj.get("filename")
                or mmproj.get("format") != expected_mmproj.get("format")):
            raise PipelineError("vision_mmproj filename/format differs from the release profile")
        mmproj_evidence = inspect_exact_file(
            Path(mmproj_path), mmproj.get("sha256"), mmproj.get("size_bytes"),
            "vision_mmproj companion",
        )
        mmproj_gguf = validate_bf16_qwen_mmproj_gguf(Path(mmproj_path))
        normalized_roles.append({
            "role": "vision_mmproj", "enabled": True, "artifact_present": True,
            "format": mmproj["format"], "gguf_contract": mmproj_gguf,
            **mmproj_evidence,
        })

    try:
        pages_limit_text = pages_limit_path.read_text(encoding="utf-8").strip()
        live_pages_limit = int(pages_limit_text)
    except (OSError, ValueError) as exc:
        raise PipelineError(
            f"cannot read live TTM pages_limit evidence {pages_limit_path}: {exc}") from exc
    gate = profile["quantization"]["native_262k_memory_gate"]
    host_page_bytes = int(os.sysconf("SC_PAGE_SIZE"))
    if host_page_bytes != TTM_PAGE_BYTES:
        raise PipelineError(
            f"host page size {host_page_bytes} differs from the pinned TTM page contract")
    live_gtt_cap = live_pages_limit * host_page_bytes
    if (live_pages_limit != gate["certification_host_gtt_pages_limit"]
            or live_gtt_cap != gate["certification_host_gtt_cap_bytes"]):
        raise PipelineError("live TTM pages_limit does not match the pinned 124 GiB certification cap")
    enabled_roles = [row["role"] for row in normalized_roles if row["enabled"]]
    disabled_roles = [row["role"] for row in normalized_roles if not row["enabled"]]
    enabled_bytes = sum(row["size_bytes"] for row in normalized_roles if row["enabled"])
    pages_path_absolute = pages_limit_path.absolute()
    authoritative_live_gtt = pages_path_absolute == CANONICAL_TTM_PAGES_LIMIT
    pending_release_evidence: list[str] = []
    if not authoritative_live_gtt:
        pending_release_evidence.append("canonical_live_gtt_evidence")
    if "vision_mmproj" in disabled_roles:
        pending_release_evidence.append("vision_mmproj")
    return {
        "status": "verified_exact",
        "schema": COMPANION_INVENTORY_SCHEMA,
        "manifest": inventory_file,
        "source": dict(source),
        "required_roles": list(COMPANION_ROLES),
        "roles": normalized_roles,
        "enabled_roles": enabled_roles,
        "disabled_roles": disabled_roles,
        "enabled_artifact_bytes": enabled_bytes,
        "fit_status": "pending_main_artifact_preflight",
        "final_release_eligibility": (
            "pending_" + "_and_".join(pending_release_evidence)
            if pending_release_evidence else "pending_combined_memory_preflight"
        ),
        "pending_release_evidence": pending_release_evidence,
        "release_companions_complete": "vision_mmproj" not in disabled_roles,
        "estimated_bytes_used": False,
        "live_gtt_evidence": {
            "pages_limit_path": str(pages_path_absolute),
            "runner_gtt_pages_limit": live_pages_limit,
            "runner_gtt_cap_bytes": live_gtt_cap,
            "page_bytes": host_page_bytes,
            "matches_required_cap": True,
            "authoritative_sysfs": authoritative_live_gtt,
        },
    }


def parse_profile_tensor_override(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, str) or value.count("=") != 1:
        raise PipelineError(f"{field} must be one REGEX=FORMAT string")
    pattern, tensor_format = value.rsplit("=", 1)
    if (not pattern or pattern[0] != "^" or pattern[-1] != "$" or
            tensor_format not in SUPPORTED_TENSOR_FORMATS):
        raise PipelineError(
            f"{field} must be an anchored regex using a kernel-backed format")
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise PipelineError(f"{field} contains an invalid regex: {exc}") from exc
    return {
        "value": value,
        "pattern": pattern,
        "format": tensor_format,
        "ggml_tensor_type": SUPPORTED_TENSOR_FORMATS[tensor_format],
        "compiled": compiled,
    }


def validated_quantization_arms(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    quantization = require_mapping(profile.get("quantization"), "profile.quantization")
    bakeoff = require_mapping(
        quantization.get("performance_bakeoff"),
        "profile.quantization.performance_bakeoff",
    )
    if (bakeoff.get("status") != "experimental_unpromoted" or
            bakeoff.get("control_unchanged") is not True or
            bakeoff.get("override_precedence") !=
            "exactly_one_matching_regex; overlap_is_an_error"):
        raise PipelineError("performance bakeoff lacks the pinned override contract")
    runtime_support = bakeoff.get("runtime_support")
    if (not isinstance(runtime_support, dict) or
            set(runtime_support) != set(SUPPORTED_TENSOR_FORMATS)):
        raise PipelineError("performance bakeoff runtime support is incomplete")
    rows = bakeoff.get("arms")
    if not isinstance(rows, list):
        raise PipelineError("performance bakeoff arms must be an array")
    expected_overrides = {
        "rocmi4-control": [quantization.get("ple_tensor_override")],
        "rocmi4-q6k-embedding-head": [
            quantization.get("ple_tensor_override"),
            r"^token_embd\.weight$=Q6_K",
            r"^output\.weight$=Q6_K",
        ],
        "rocmfp4-fast-routed-experts-q6k-embedding-head": [
            quantization.get("ple_tensor_override"),
            ROCMFP4_FAST_ROUTED_EXPERT_PATTERN + "=Q4_0_ROCMFP4_FAST",
            r"^token_embd\.weight$=Q6_K",
            r"^output\.weight$=Q6_K",
        ],
        "rocmfp4-fast-matrix": [
            quantization.get("ple_tensor_override"),
            *[pattern + "=Q4_0_ROCMFP4_FAST"
              for pattern in ROCMFP4_FAST_MATRIX_PATTERNS],
        ],
        "rocmfp4-fast-matrix-q6k-embedding-head": [
            quantization.get("ple_tensor_override"),
            *[pattern + "=Q4_0_ROCMFP4_FAST"
              for pattern in ROCMFP4_FAST_MATRIX_PATTERNS],
            r"^token_embd\.weight$=Q6_K",
            r"^output\.weight$=Q6_K",
        ],
    }
    expected_mtp_contracts = {
        "rocmi4-control": "Q4_0_ROCMI4",
        "rocmi4-q6k-embedding-head": "Q4_0_ROCMI4",
        "rocmfp4-fast-routed-experts-q6k-embedding-head": "Q4_0_ROCMFP4_FAST",
        "rocmfp4-fast-matrix": "Q4_0_ROCMFP4_FAST",
        "rocmfp4-fast-matrix-q6k-embedding-head": "Q4_0_ROCMFP4_FAST",
    }
    if [row.get("id") for row in rows if isinstance(row, dict)] != list(expected_overrides):
        raise PipelineError("performance bakeoff arms are missing, duplicated, or reordered")
    result: dict[str, dict[str, Any]] = {}
    ple_override = quantization.get("ple_tensor_override")
    for index, row_value in enumerate(rows):
        arm = require_mapping(row_value, f"performance_bakeoff.arms[{index}]")
        arm_id = arm.get("id")
        overrides = arm.get("per_tensor_overrides")
        if arm.get("default_matrix_format") != "Q4_0_ROCMI4":
            raise PipelineError(f"quantization arm {arm_id} changed the base matrix format")
        if arm.get("mtp_matrix_quant_contract") != expected_mtp_contracts[arm_id]:
            raise PipelineError(f"quantization arm {arm_id} has the wrong MTP matrix contract")
        if not isinstance(overrides, list) or overrides != expected_overrides[arm_id]:
            raise PipelineError(
                f"quantization arm {arm_id} has malformed or unaudited per_tensor_overrides")
        parsed = [
            parse_profile_tensor_override(
                value, f"performance_bakeoff.arms[{index}].per_tensor_overrides[{item}]")
            for item, value in enumerate(overrides)
        ]
        if len({item["pattern"] for item in parsed}) != len(parsed):
            raise PipelineError(f"quantization arm {arm_id} repeats a tensor regex")
        ple_matches = [item for item in parsed
                       if item["compiled"].search("per_layer_token_embd.weight")]
        if (len(ple_matches) != 1 or ple_matches[0]["value"] != ple_override or
                parsed[0]["value"] != ple_override):
            raise PipelineError(
                f"quantization arm {arm_id} does not preserve the pinned PLE override")
        serialized = json.dumps(overrides, separators=(",", ":"), ensure_ascii=True)
        result[arm_id] = {
            "id": arm_id,
            "default_matrix_format": "Q4_0_ROCMI4",
            "per_tensor_overrides": list(overrides),
            "per_tensor_overrides_sha256": hashlib.sha256(
                serialized.encode("utf-8")).hexdigest(),
            "formats": list(dict.fromkeys(item["format"] for item in parsed)),
            "mtp_matrix_quant_contract": expected_mtp_contracts[arm_id],
            "override_precedence": bakeoff["override_precedence"],
        }
    return result


def validate_intervention_manifest(
    path: Path, profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate evidence for an actual pre-quantization weight intervention.

    The directions are inline so the exact bytes consumed by the quantizer are
    covered by the manifest digest.  Corpus rows need not be embedded, but a
    non-empty count and content digest make the extraction set identifiable.
    Tensor targets are exact names: allowing a pattern here would make the
    claimed intervention depend on a tool's regex dialect.
    """
    manifest = read_json(path, "intervention manifest")
    contract = require_mapping(profile.get("intervention"), "profile.intervention")
    if (
        contract.get("required") is not True
        or contract.get("kind") != "directional_ablation"
        or contract.get("manifest_schema_version") != 1
        or contract.get("application_stage") != "pre_quantization_encoding"
        or contract.get("weight_intervention_required") is not True
        or contract.get("prompt_only_allowed") is not False
        or contract.get("manifest_filename") != "qwen-intervention-manifest.json"
        or contract.get("quantizer_application_report_required") is not True
    ):
        raise PipelineError("release profile does not require the audited intervention contract")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "directional_ablation"
        or manifest.get("status") != "complete"
        or manifest.get("weight_intervention") is not True
        or manifest.get("prompt_only") is not False
        or manifest.get("application_stage") != "pre_quantization_encoding"
    ):
        raise PipelineError(
            "intervention manifest must describe a complete pre-quantization weight intervention, not prompt-only behavior"
        )

    source = require_mapping(manifest.get("source"), "intervention.source")
    expected_source = profile["source"]
    if any(source.get(field) != expected_source.get(field) for field in (
        "repo_id", "revision", "snapshot_inventory_sha256"
    )):
        raise PipelineError("intervention source evidence does not match the pinned base model")

    tooling = require_mapping(manifest.get("tooling"), "intervention.tooling")
    for manifest_key, profile_key in (
        ("otheru_quant_pipeline", "otheru_pipeline"),
        ("upstream_heretic", "upstream_heretic"),
    ):
        actual = require_mapping(tooling.get(manifest_key), f"intervention.tooling.{manifest_key}")
        expected = require_mapping(contract.get(profile_key), f"profile.intervention.{profile_key}")
        if actual != expected:
            raise PipelineError(f"intervention tooling evidence for {manifest_key} is not pinned to the release profile")

    corpora = manifest.get("corpora")
    if not isinstance(corpora, list) or not corpora:
        raise PipelineError("intervention.corpora must contain direction-extraction corpus evidence")
    corpus_ids: set[str] = set()
    corpus_records = 0
    for index, item in enumerate(corpora):
        corpus = require_mapping(item, f"intervention.corpora[{index}]")
        corpus_id = corpus.get("id")
        if not isinstance(corpus_id, str) or not corpus_id or corpus_id in corpus_ids:
            raise PipelineError("intervention corpus ids must be unique non-empty strings")
        corpus_ids.add(corpus_id)
        if corpus.get("role") != "direction_extraction":
            raise PipelineError("intervention corpus role must be direction_extraction")
        if not isinstance(corpus.get("sha256"), str) or not SHA256_RE.fullmatch(corpus["sha256"]):
            raise PipelineError("intervention corpus evidence requires a lowercase SHA-256")
        count = corpus.get("record_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise PipelineError("intervention corpus record_count must be positive")
        if corpus.get("held_out_evaluation_overlap_count") != 0:
            raise PipelineError("direction-extraction corpora must be disjoint from held-out evaluation")
        corpus_records += count

    directions = manifest.get("directions")
    if not isinstance(directions, list) or not directions:
        raise PipelineError("intervention.directions must contain inline F32 direction evidence")
    direction_ids: set[str] = set()
    direction_dimensions: dict[str, int] = {}
    for index, item in enumerate(directions):
        direction = require_mapping(item, f"intervention.directions[{index}]")
        direction_id = direction.get("id")
        if not isinstance(direction_id, str) or not direction_id or direction_id in direction_ids:
            raise PipelineError("intervention direction ids must be unique non-empty strings")
        direction_ids.add(direction_id)
        values = direction.get("values")
        if direction.get("dtype") != "F32" or not isinstance(values, list) or not values:
            raise PipelineError("intervention directions must contain non-empty inline F32 values")
        packed = bytearray()
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise PipelineError("intervention direction values must be finite numbers")
            try:
                packed.extend(struct.pack("<f", float(value)))
            except (OverflowError, struct.error) as exc:
                raise PipelineError("intervention direction value is outside F32 range") from exc
        digest = hashlib.sha256(packed).hexdigest()
        if direction.get("sha256") != digest:
            raise PipelineError("intervention direction SHA-256 does not match its packed little-endian F32 values")
        direction_dimensions[direction_id] = len(values)

    # The release model's ordinary Transformers hidden-state API mixes the
    # 10240-wide HC state with the final 2560-wide state and cannot be used to
    # construct these left-projection directions.  Require evidence from the
    # Qwen-specific mixed-input extractor.  Fixture profiles use another repo
    # id and intentionally retain small synthetic direction dimensions.
    if expected_source.get("repo_id") == "Qwen/Qwen3.8-Flash-Next":
        extraction = require_mapping(manifest.get("extraction"), "intervention.extraction")
        if (
            extraction.get("direction_scope") != "per_layer"
            or extraction.get("layer_count") != 48
            or extraction.get("activation_width") != 2560
            or extraction.get("semantic_capture_point")
            != "decoder_layer.attn_hyper_connection.mixed_input"
            or extraction.get("hidden_states_api_used") is not False
            or extraction.get("streaming") is not True
            or extraction.get("efficacy_evaluated") is not False
        ):
            raise PipelineError(
                "Qwen intervention extraction must use streamed 2560-wide per-layer HC mixed inputs"
            )
        expected_direction_ids = {
            f"layer-{layer:02d}-attn-mixed-input-r1" for layer in range(48)
        }
        if direction_ids != expected_direction_ids or any(
            direction_dimensions[direction_id] != 2560
            for direction_id in expected_direction_ids
        ):
            raise PipelineError("Qwen intervention must carry all 48 per-layer 2560-wide directions")
        classes = {corpus.get("class"): corpus for corpus in corpora}
        if set(classes) != {"good_control", "bad_target"}:
            raise PipelineError("Qwen intervention requires distinct good and bad extraction corpora")
        if (
            extraction.get("good_records_processed") != classes["good_control"].get("record_count")
            or extraction.get("bad_records_processed") != classes["bad_target"].get("record_count")
        ):
            raise PipelineError("Qwen intervention activation counts differ from corpus evidence")
        held_out = require_mapping(
            manifest.get("held_out_evaluation"), "intervention.held_out_evaluation"
        )
        if (
            not isinstance(held_out.get("sha256"), str)
            or SHA256_RE.fullmatch(held_out["sha256"]) is None
            or not isinstance(held_out.get("record_count"), int)
            or isinstance(held_out.get("record_count"), bool)
            or held_out["record_count"] < 1
            or held_out.get("overlap_count") != 0
            or held_out.get("comparison") != "canonical_text_chat_messages_sha256"
        ):
            raise PipelineError("Qwen intervention lacks disjoint held-out evaluation evidence")
        activation = require_mapping(
            extraction.get("activation_evidence"),
            "intervention.extraction.activation_evidence",
        )
        backend = activation.get("backend")
        if backend == "ember_qwen_runtime_f32_dump":
            if (
                activation.get("format") != "48x2560-little-endian-f32-records-v1"
                or activation.get("record_order") != "corpus_jsonl_order"
                or activation.get("artifact_sha256_verification")
                != "supplied_not_locally_rehashed"
                or any(
                    not isinstance(activation.get(field), str)
                    or SHA256_RE.fullmatch(activation[field]) is None
                    for field in (
                        "stock_rocmi4_artifact_sha256",
                        "good_dump_sha256",
                        "bad_dump_sha256",
                    )
                )
                or activation.get("good_dump_bytes")
                != extraction["good_records_processed"] * 48 * 2560 * 4
                or activation.get("bad_dump_bytes")
                != extraction["bad_records_processed"] * 48 * 2560 * 4
            ):
                raise PipelineError("Qwen runtime activation-dump evidence is incomplete")
        elif backend == "transformers_optional_reference":
            if activation.get("snapshot_revision") != expected_source.get("revision"):
                raise PipelineError("Qwen Transformers extraction did not use the pinned snapshot")
        else:
            raise PipelineError("Qwen intervention names an unsupported activation backend")

    targets = manifest.get("targets")
    if not isinstance(targets, list) or not targets:
        raise PipelineError("intervention.targets must contain at least one exact weight tensor")
    target_names: list[str] = []
    for index, item in enumerate(targets):
        target = require_mapping(item, f"intervention.targets[{index}]")
        name = target.get("tensor_name")
        match = INTERVENTION_TARGET_RE.fullmatch(name) if isinstance(name, str) else None
        if match is None or name in target_names:
            raise PipelineError(
                "intervention targets must be unique exact Qwen residual-writer tensor names"
            )
        layer = int(match.group(1))
        writer = match.group(2)
        if layer > 47 or (
            (layer in QWEN_QSA_LAYERS and writer != "attn_output")
            or (layer not in QWEN_QSA_LAYERS and writer != "ssm_out")
        ):
            raise PipelineError(
                f"intervention target {name} does not match the Qwen layer's residual writer"
            )
        target_names.append(name)
        direction_id = target.get("direction_id")
        if direction_id not in direction_ids:
            raise PipelineError(f"intervention target {name} refers to an unknown direction")
        scale = target.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) == 0.0:
            raise PipelineError(f"intervention target {name} requires a finite non-zero scale")
        if target.get("normalization") != "row_norm_preserve":
            raise PipelineError(f"intervention target {name} must use row_norm_preserve")
        shape = target.get("expected_shape")
        if (
            not isinstance(shape, list) or len(shape) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in shape)
            or direction_dimensions[str(direction_id)] != shape[1]
        ):
            raise PipelineError(
                f"intervention target {name} has invalid [ne0 columns, ne1 rows] direction/shape evidence"
            )
        if expected_source.get("repo_id") == "Qwen/Qwen3.8-Flash-Next" and (
            shape != [6144, 2560]
            or direction_id != f"layer-{layer:02d}-attn-mixed-input-r1"
        ):
            raise PipelineError(
                f"intervention target {name} must use its same-layer 2560-wide direction and [6144, 2560] GGUF shape"
            )

    tensor_map = require_mapping(manifest.get("tensor_map"), "intervention.tensor_map")
    if target_names != sorted(target_names):
        raise PipelineError("intervention targets must be lexicographically ordered")
    names_digest = hashlib.sha256("\n".join(sorted(target_names)).encode("utf-8")).hexdigest()
    if (
        tensor_map.get("kind") != "exact_tensor_names"
        or tensor_map.get("target_count") != len(target_names)
        or tensor_map.get("target_names_sha256") != names_digest
    ):
        raise PipelineError("intervention tensor-map evidence does not match the exact target list")

    manifest_hash = sha256_file(path)
    return manifest, {
        "manifest_filename": contract["manifest_filename"],
        "manifest_source_path": str(path),
        "manifest_sha256": manifest_hash,
        "kind": manifest["kind"],
        "application_stage": manifest["application_stage"],
        "weight_intervention": True,
        "prompt_only": False,
        "corpus_count": len(corpora),
        "corpus_record_count": corpus_records,
        "direction_count": len(directions),
        "target_count": len(target_names),
        "targets": target_names,
        "target_names_sha256": names_digest,
        "tooling": tooling,
    }


def validate_rocmi4_sweep_authorization(
    plan_path: Path | None,
    plan_sha256: str | None,
    profile_path: Path,
    profile_sha256: str,
    quantization_arm: dict[str, Any],
    intervention_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Authorize ROCMI4 control encoding only for one canonical planned sweep row."""
    if plan_path is None or plan_sha256 is None:
        raise PipelineError(
            "non-stock rocmi4-control requires an exact canonical bakeoff plan descriptor")
    plan, evidence = read_exact_json_file(
        plan_path.resolve(), plan_sha256, "ROCMI4 sweep bakeoff plan")
    if (plan.get("schema_version") != 1 or plan.get("phase_scope") != "selection"
            or plan.get("status") != "planned_unmeasured"
            or plan.get("publication_allowed") is not False):
        raise PipelineError("rocmi4-control authorization requires the selection-only canonical plan")
    profile = require_mapping(plan.get("release_profile"), "bakeoff plan release_profile")
    if profile != {"path": str(profile_path), "sha256": profile_sha256}:
        raise PipelineError("rocmi4-control bakeoff plan differs from the active release profile")
    corpora = require_mapping(plan.get("corpora"), "bakeoff plan corpora")
    held_out = require_mapping(
        intervention_manifest.get("held_out_evaluation"),
        "intervention.held_out_evaluation",
    )
    sweep_corpus = require_mapping(
        corpora.get("sweep-validation.jsonl"), "bakeoff sweep-validation corpus")
    if held_out.get("sha256") != sweep_corpus.get("sha256"):
        raise PipelineError("rocmi4-control intervention used a corpus outside the selection plan")

    actual_scales = {str(layer): 0.0 for layer in range(48)}
    for target in intervention_manifest.get("targets", []):
        match = INTERVENTION_TARGET_RE.fullmatch(str(target.get("tensor_name", "")))
        if match is None:
            raise PipelineError("rocmi4-control intervention contains an unvalidated target")
        actual_scales[str(int(match.group(1)))] = float(target["scale"])
    expected_basis = {
        "source": intervention_manifest.get("source"),
        "tooling": intervention_manifest.get("tooling"),
        "corpora": [{key: corpus.get(key) for key in (
            "class", "role", "sha256", "record_count")}
                    for corpus in intervention_manifest.get("corpora", [])],
    }
    rows = plan.get("sweep_configurations")
    if not isinstance(rows, list):
        raise PipelineError("rocmi4-control bakeoff plan lacks sweep configurations")
    matches = []
    for row in rows:
        if (isinstance(row, dict)
                and row.get("quantization_arm") == "rocmi4-control"
                and row.get("profile_sha256") == profile_sha256
                and row.get("quantization_overrides_sha256") ==
                    quantization_arm["per_tensor_overrides_sha256"]
                and row.get("runtime_mode") == "exact_dequant"
                and row.get("final_release_eligible") is False
                and row.get("direction_basis") == expected_basis
                and row.get("layer_scales") == actual_scales):
            matches.append(row)
    if len(matches) != 1 or not isinstance(matches[0].get("id"), str):
        raise PipelineError(
            "rocmi4-control intervention does not match exactly one canonical sweep configuration")
    return {
        "status": "authorized_selection_sweep_control_encoding",
        "configuration_id": matches[0]["id"],
        "plan": evidence,
        "profile_sha256": profile_sha256,
        "quantization_overrides_sha256": quantization_arm[
            "per_tensor_overrides_sha256"],
        "held_out_corpus_sha256": sweep_corpus["sha256"],
        "final_release_eligible": False,
    }


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def path_entry_exists(path: Path) -> bool:
    """Return true for every directory entry, including dangling symlinks."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def cleanup_gguf_writer_temp(directory: Path) -> list[dict[str, Any]]:
    """Remove only regular Python/GGUFWriter spool residues from a private TMPDIR.

    llama.cpp's pinned ``GGUFWriter`` uses ``SpooledTemporaryFile``.  Once its
    256 MiB threshold is crossed, Python creates a mode-0600 ``tmpXXXXXXXX``
    file in TMPDIR.  A successful converter can leave that already-consumed
    spool inode behind at process teardown.  Importing the pinned converter's
    torch stack can also create one empty ``torchinductor_root`` cache directory
    even though conversion never compiles an Inductor kernel.  Treating every
    nonempty TMPDIR as a conversion failure discarded completed control
    conversions.

    The directory is private and the converter process has exited, but deletion
    still fails closed: names must match Python's exact eight-character random
    tempfile convention, entries must be owner-only regular single-link files,
    and lstat/fstat identities must agree.  The one allowed directory must have
    the exact torch name, be empty, owner-controlled, and identity-bound through
    a no-follow directory descriptor before a non-recursive ``rmdir``.  No
    symlink, hardlink, nonempty directory, or conveniently named foreign
    artifact is removed.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(directory, flags)
    except OSError as exc:
        raise PipelineError(f"cannot open private converter TMPDIR {directory}: {exc}") from exc
    removed: list[dict[str, Any]] = []
    try:
        names = sorted(os.listdir(directory_fd))
        for name in names:
            if name == "torchinductor_root":
                before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (not stat.S_ISDIR(before.st_mode) or before.st_uid != os.geteuid()
                        or before.st_nlink != 2
                        or stat.S_IMODE(before.st_mode) not in {0o700, 0o755}):
                    raise PipelineError(
                        "converter torchinductor cache is not an empty owner-controlled directory")
                cache_flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                               | getattr(os, "O_DIRECTORY", 0)
                               | getattr(os, "O_NOFOLLOW", 0))
                try:
                    cache_fd = os.open(name, cache_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise PipelineError(
                        f"cannot identity-bind converter torchinductor cache: {exc}") from exc
                try:
                    opened = os.fstat(cache_fd)
                    if ((opened.st_dev, opened.st_ino, opened.st_mode, opened.st_uid,
                         opened.st_nlink) !=
                            (before.st_dev, before.st_ino, before.st_mode, before.st_uid,
                             before.st_nlink)):
                        raise PipelineError(
                            "converter torchinductor cache changed during cleanup validation")
                    if os.listdir(cache_fd):
                        raise PipelineError("converter torchinductor cache is not empty")
                    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                        raise PipelineError(
                            "converter torchinductor cache pathname changed before removal")
                    os.rmdir(name, dir_fd=directory_fd)
                    removed.append({"name": name,
                                    "kind": "empty_torchinductor_cache_directory",
                                    "mode": stat.S_IMODE(opened.st_mode)})
                finally:
                    os.close(cache_fd)
                continue
            if GGUF_WRITER_TEMP_NAME_RE.fullmatch(name) is None:
                raise PipelineError(
                    f"converter TMPDIR contains an unexpected entry: {name!r}")
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                    or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o600):
                raise PipelineError(
                    f"converter TMPDIR entry is not an owner-only regular single-link spool: {name!r}")
            file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(name, file_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise PipelineError(
                    f"cannot identity-bind converter spool {name!r}: {exc}") from exc
            try:
                opened = os.fstat(descriptor)
                if ((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
                    raise PipelineError(
                        f"converter spool changed during cleanup validation: {name!r}")
                row = {"name": name, "size_bytes": opened.st_size,
                       "mode": stat.S_IMODE(opened.st_mode)}
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                    raise PipelineError(
                        f"converter spool pathname changed before removal: {name!r}")
                os.unlink(name, dir_fd=directory_fd)
                removed.append(row)
            finally:
                os.close(descriptor)
        if os.listdir(directory_fd):
            raise PipelineError("converter TMPDIR changed while expected spools were removed")
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    try:
        directory.rmdir()
    except OSError as exc:
        raise PipelineError(f"cannot remove validated empty converter TMPDIR: {exc}") from exc
    for row in removed:
        print(
            "qwen_quantize.py: removed expected converter TMPDIR residue "
            f"{row['name']}"
            + (f" ({row['size_bytes']} bytes)" if "size_bytes" in row else ""),
            file=sys.stderr,
        )
    return removed


def write_json_atomic(
    path: Path, value: dict[str, Any], *, create: bool,
) -> None:
    """Durably create or replace JSON inside an unpublished private directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if create:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise PipelineError(f"refusing to overwrite existing build record: {path}") from exc
        else:
            os.replace(temporary, path)
        fsync_directory(path.parent)
    except OSError as exc:
        raise PipelineError(f"cannot write build record {path}: {exc}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one same-filesystem directory without replacement."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise PipelineError("Linux renameat2 is required for atomic transaction publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    if renameat2(at_fdcwd, os.fsencode(source), at_fdcwd,
                 os.fsencode(destination), rename_noreplace) != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise PipelineError(f"refusing to overwrite existing transaction directory: {destination}")
        raise PipelineError(
            f"cannot atomically publish transaction directory {destination}: {os.strerror(error)}"
        )
    fsync_directory(destination.parent)


def sync_verified_outputs(paths: list[Path]) -> None:
    for path in paths:
        status = os.lstat(path)
        if not stat.S_ISREG(status.st_mode):
            raise PipelineError(f"staged GGUF is not a regular file: {path}")
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (status.st_dev, status.st_ino):
                raise PipelineError(f"staged GGUF identity changed before sync: {path}")
            os.fsync(stream.fileno())


def copy_verified_input(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy immutable evidence into the private transaction via a pinned fd."""
    try:
        named = os.lstat(source)
    except OSError as exc:
        raise PipelineError(f"cannot inspect intervention manifest {source}: {exc}") from exc
    if not stat.S_ISREG(named.st_mode):
        raise PipelineError(f"intervention manifest is not a regular file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            before = os.fstat(input_stream.fileno())
            if (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
                raise PipelineError("intervention manifest identity changed before transaction copy")
            for chunk in iter(lambda: input_stream.read(8 * 1024 * 1024), b""):
                output_stream.write(chunk)
                digest.update(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
            after = os.fstat(input_stream.fileno())
    except OSError as exc:
        raise PipelineError(f"cannot copy intervention manifest: {exc}") from exc
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise PipelineError("intervention manifest changed while copied into transaction")
    if digest.hexdigest() != expected_sha256:
        raise PipelineError("intervention manifest changed after validation")


def credential_free_env() -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
    }


def run_checked(
    command: list[str], cwd: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> str:
    environment = credential_free_env()
    if env_overrides:
        environment.update(env_overrides)
    result = subprocess.run(
        command, cwd=cwd, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise PipelineError(f"command failed ({result.returncode}): {shlex.join(command)}\n{detail}")
    return result.stdout.strip()


def run_checked_combined(command: list[str]) -> str:
    """Return stdout+stderr for tools such as llama.cpp that version on stderr."""
    result = subprocess.run(
        command, env=credential_free_env(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    combined = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise PipelineError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n{combined}"
        )
    return combined


def validate_profile(profile_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    profile = read_json(profile_path, "release profile")
    if profile.get("schema_version") != 1:
        raise PipelineError("unsupported release profile schema_version")
    for section in ("source", "conversion", "intervention", "quantizer", "quantization", "artifact", "release"):
        if not isinstance(profile.get(section), dict):
            raise PipelineError(f"profile section {section} is missing")
    source = profile["source"]
    conversion = profile["conversion"]
    quantizer = profile["quantizer"]
    quantization = profile["quantization"]
    intervention = profile["intervention"]
    for label, revision in (
        ("source", source.get("revision")),
        ("conversion base", conversion.get("base_revision")),
        ("conversion", conversion.get("revision")),
        ("quantizer", quantizer.get("revision")),
    ):
        if not isinstance(revision, str) or not HEX40.fullmatch(revision):
            raise PipelineError(f"{label} revision is not a pinned 40-character commit")
    if quantization.get("format") != "Q4_0_ROCMI4" or quantization.get("ggml_tensor_type") != 108:
        raise PipelineError("release profile must select Q4_0_ROCMI4 / GGML tensor type 108")
    validated_quantization_arms(profile)
    if (
        intervention.get("required") is not True
        or intervention.get("kind") != "directional_ablation"
        or intervention.get("manifest_schema_version") != 1
        or intervention.get("application_stage") != "pre_quantization_encoding"
        or intervention.get("weight_intervention_required") is not True
        or intervention.get("prompt_only_allowed") is not False
        or intervention.get("manifest_filename") != "qwen-intervention-manifest.json"
        or intervention.get("quantizer_application_report_required") is not True
        or intervention.get("required_evidence") != [
            "source", "tooling", "corpora", "directions", "targets", "tensor_map"
        ]
        or intervention.get("otheru_pipeline") != {
            "repository": "https://git.otheru.ai/akadmin/otheru-quant-pipeline",
            "branch": "ember-contract-and-drafter-fix",
            "revision": "a3c6a728510f91394e991504951ac316cd3a89af",
        }
        or intervention.get("upstream_heretic") != {
            "repository": "https://github.com/p-e-w/heretic",
            "revision": "bedb94ef117a271532ac2058447fbc165d5051bd",
        }
    ):
        raise PipelineError("release profile lacks the pinned Heretic weight-intervention contract")
    artifact_name = str(profile["artifact"].get("filename", ""))
    artifact_repo = str(profile["artifact"].get("repo_id", ""))
    if "Heretic" not in artifact_name or "Heretic" not in artifact_repo:
        raise PipelineError("required directional-ablation artifact must carry the Heretic release name")
    if quantization.get("w4a4_enabled") is not False or quantizer.get("w4a4_default") is not False:
        raise PipelineError("conversion pipeline is exact-dequant only; W4A4 must remain disabled")
    memory_gate = quantization.get("native_262k_memory_gate")
    if not isinstance(memory_gate, dict):
        raise PipelineError("quantization.native_262k_memory_gate is missing")
    if (
        memory_gate.get("native_context_tokens") != 262144
        or memory_gate.get("device_budget_bytes") != 137438953472
        or memory_gate.get("runtime_reserve_bytes") != 34359738368
        or memory_gate.get("certification_host_memtotal_bytes") != 134297894912
        or memory_gate.get("certification_host_gtt_pages_limit") != 32505856
        or memory_gate.get("certification_host_gtt_cap_bytes") != 133143986176
        or memory_gate.get("certification_host_rule")
        != "artifact_bytes + runtime_reserve_bytes + enabled_companion_artifact_bytes <= certification_host_memtotal_bytes"
        or memory_gate.get("companion_artifact_gate_status")
        != "pending_mtp_mmproj_inventory_and_measured_peak_rss_gtt_against_124gib_cap"
        or memory_gate.get("rule") != "artifact_bytes + runtime_reserve_bytes <= device_budget_bytes"
        or memory_gate.get("yarn_1m_math_oracle_passed") is not True
        or memory_gate.get("yarn_1m_runtime_certified") is not False
        or memory_gate.get("yarn_1m_fit_claim") is not False
    ):
        raise PipelineError("native-262K memory gate does not match the audited 128 GiB contract")
    runner = profile["release"].get("conversion_runner_requirements")
    if (
        not isinstance(runner, dict)
        or runner.get("minimum_free_disk_gib") != 1152
        or runner.get("minimum_physical_ram_gib") != 256
        or runner.get("bounded_memory_minimum_physical_ram_gib") != 120
        or runner.get("bounded_memory_mode") != "llama_patched_ple_writeback_use_temp_then_gguf_split"
        or runner.get("bounded_memory_status") != "patched_after_127269264_kib_oom_pending_remeasurement"
        or runner.get("certification_host_memtotal_bytes") != 134297894912
    ):
        raise PipelineError("release conversion runner must require 1152 GiB disk and 256 GiB RAM")
    layout_gate = profile["release"].get("artifact_layout_gate")
    if (
        not isinstance(layout_gate, dict)
        or layout_gate.get("current_quantizer_multi_shard_supported") is not True
        or layout_gate.get("default_split_max_size") != "48G"
        or layout_gate.get("aggregate_preflight_before_output") is not True
        or layout_gate.get("transactional_no_clobber_shards") is not True
        or layout_gate.get("atomic_committed_work_directory") is not True
        or layout_gate.get("directory_commit_method") != "renameat2(RENAME_NOREPLACE)"
        or layout_gate.get("conditional_rollback_unlink") is not False
        or layout_gate.get("publication_blocked") is not False
    ):
        raise PipelineError("release multi-shard artifact-layout contract is missing")
    if profile["conversion"].get("default_split_max_size") != "48G":
        raise PipelineError("release conversion must default to 48G llama.cpp shards")
    inventory_path = profile_path.parent / str(source.get("snapshot_inventory", ""))
    expected_inventory_hash = source.get("snapshot_inventory_sha256")
    if not inventory_path.is_file() or sha256_file(inventory_path) != expected_inventory_hash:
        raise PipelineError("snapshot inventory file is missing or does not match its pinned SHA-256")
    inventory = read_json(inventory_path, "snapshot inventory")
    if inventory.get("repo_id") != source.get("repo_id") or inventory.get("revision") != source.get("revision"):
        raise PipelineError("snapshot inventory repository/revision differs from release profile")
    files = inventory.get("files")
    if not isinstance(files, list) or len(files) != inventory.get("file_count"):
        raise PipelineError("snapshot inventory file count is invalid")
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != inventory.get("inventory_sha256"):
        raise PipelineError("snapshot inventory canonical digest is invalid")
    weight_bytes = sum(row["size"] for row in files if str(row["path"]).endswith(".safetensors"))
    if weight_bytes != source.get("weight_bytes"):
        raise PipelineError("snapshot inventory weight bytes differ from the release profile")
    return profile, inventory, inventory_path


def validate_snapshot(snapshot: Path, revision: str, profile: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if revision != profile["source"]["revision"]:
        raise PipelineError("--snapshot-revision does not match the pinned Hugging Face revision")
    if not snapshot.is_dir():
        raise PipelineError(f"snapshot directory does not exist: {snapshot}")
    if snapshot.parent.name == "snapshots" and snapshot.name != revision:
        raise PipelineError("Hugging Face cache snapshot directory name does not match --snapshot-revision")
    ignored_coordination_files: list[str] = []
    fetch_lock = snapshot / ".ember-fetch.lock"
    try:
        fetch_lock_status = os.lstat(fetch_lock)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise PipelineError(f"cannot inspect snapshot coordination file: {exc}") from exc
    else:
        if not stat.S_ISREG(fetch_lock_status.st_mode):
            raise PipelineError(
                "snapshot coordination file .ember-fetch.lock must be a regular non-symlink file")
        ignored_coordination_files.append(".ember-fetch.lock")
    actual_paths: list[str] = []
    for path in snapshot.rglob("*"):
        try:
            status = os.lstat(path)
        except OSError as exc:
            raise PipelineError(f"cannot inspect snapshot entry {path}: {exc}") from exc
        if stat.S_ISLNK(status.st_mode):
            raise PipelineError(f"snapshot entries must not be symlinks: {path}")
        if stat.S_ISREG(status.st_mode) and path != fetch_lock:
            actual_paths.append(path.relative_to(snapshot).as_posix())
        elif not stat.S_ISREG(status.st_mode) and not stat.S_ISDIR(status.st_mode):
            raise PipelineError(f"snapshot entry must be a regular file or directory: {path}")
    actual_paths.sort()
    expected_rows = inventory["files"]
    expected_paths = [row["path"] for row in expected_rows]
    if actual_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(actual_paths))[:10]
        extra = sorted(set(actual_paths) - set(expected_paths))[:10]
        raise PipelineError(f"snapshot file inventory mismatch; missing={missing}, extra={extra}")
    checked = []
    for row in expected_rows:
        rel = PurePosixPath(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise PipelineError(f"unsafe path in snapshot inventory: {row['path']}")
        path = snapshot / Path(*rel.parts)
        size = path.stat().st_size
        if size != row["size"]:
            raise PipelineError(f"snapshot size mismatch for {row['path']}: expected {row['size']}, got {size}")
        if "sha256" in row:
            algorithm = "sha256"
            actual_digest, read_method = snapshot_artifact_digest(path, size, algorithm)
            if actual_digest != row["sha256"]:
                raise PipelineError(f"snapshot SHA-256 mismatch for {row['path']}")
        else:
            algorithm = "git_blob_sha1"
            actual_digest, read_method = snapshot_artifact_digest(path, size, algorithm)
            if actual_digest != row["git_blob"]:
                raise PipelineError(f"snapshot Git blob mismatch for {row['path']}")
        checked.append({
            "path": row["path"], "size_bytes": size,
            algorithm: actual_digest, "integrity_read_method": read_method,
        })
    license_path = snapshot / profile["source"]["license"]["path_in_source"]
    if sha256_file(license_path) != profile["source"]["license"]["sha256"]:
        raise PipelineError("snapshot license does not match the release profile")
    config = read_json(snapshot / "config.json", "snapshot config")
    architectures = config.get("architectures", [])
    if profile["source"]["architecture"] not in architectures:
        raise PipelineError("snapshot config architecture does not match the release profile")
    if config.get("model_type") != profile["source"]["model_type"]:
        raise PipelineError("snapshot config model_type does not match the release profile")
    return {
        "revision": revision,
        "files_verified": len(checked),
        "total_bytes": sum(x["size_bytes"] for x in checked),
        "ignored_coordination_files": ignored_coordination_files,
        "integrity_read_methods": dict(Counter(
            item["integrity_read_method"] for item in checked)),
    }


def git_revision(directory: Path) -> str:
    git = shutil.which("git", path=os.defpath)
    if git is None:
        raise PipelineError("git is required for source revision validation")
    if not directory.is_dir():
        raise PipelineError(f"source checkout does not exist: {directory}")
    return run_git_checked(git, directory, ["rev-parse", "HEAD"])


def run_git_checked(git: str, directory: Path, arguments: list[str]) -> str:
    """Run Git against one audited checkout despite the credential-free HOME."""
    checkout = directory.resolve(strict=True)
    return run_checked([
        git, "-c", f"safe.directory={checkout}", "-C", str(checkout), *arguments,
    ])


def ember_revision_evidence(ember_dir: Path) -> tuple[str, str, bool]:
    """Read Ember provenance from its build cache, with Git as a source-tree check."""
    cache = ember_dir / "build-rocm" / "CMakeCache.txt"
    configured: str | None = None
    if cache.is_file():
        prefix = "EMBER_CONFIGURED_GIT_HEAD:STRING="
        matches = [line[len(prefix):].strip() for line in
                   cache.read_text(encoding="utf-8", errors="strict").splitlines()
                   if line.startswith(prefix)]
        if len(matches) != 1 or HEX40.fullmatch(matches[0]) is None:
            raise PipelineError(
                "Ember CMake cache must contain exactly one configured 40-character revision")
        configured = matches[0]
    has_git_metadata = path_entry_exists(ember_dir / ".git")
    source_revision = git_revision(ember_dir) if has_git_metadata else None
    if configured is not None and source_revision is not None and configured != source_revision:
        raise PipelineError("Ember source revision differs from its configured ROCm build revision")
    revision = configured or source_revision
    if revision is None:
        raise PipelineError(
            "Ember provenance requires build-rocm/CMakeCache.txt or a Git checkout")
    return revision, ("cmake_cache_and_git" if source_revision else "cmake_cache"), has_git_metadata


def validate_tools(
    llama_dir: Path,
    rocmfpx_dir: Path,
    ember_dir: Path,
    ember_revision: str,
    quantizer_binary: Path,
    profile: dict[str, Any],
    gguf_splitter: Path | None = None,
) -> dict[str, Any]:
    if sys.version_info < (3, 10):
        raise PipelineError("Python 3.10 or newer is required by the pinned converter")
    conversion = profile["conversion"]
    quantizer = profile["quantizer"]
    llama_head = git_revision(llama_dir)
    rocmfpx_head = git_revision(rocmfpx_dir)
    ember_head, ember_revision_source, ember_has_git = ember_revision_evidence(ember_dir)
    if llama_head != conversion["revision"]:
        raise PipelineError(f"llama.cpp checkout must be at {conversion['revision']}, got {llama_head}")
    git = shutil.which("git", path=os.defpath)
    assert git is not None
    git_version = run_checked([git, "--version"])
    llama_parent = run_git_checked(git, llama_dir, ["rev-parse", "HEAD^"])
    if llama_parent != conversion["base_revision"]:
        raise PipelineError("Qwen4Exp rotated-KV head is not directly based on the pinned PR #27742 head")
    if rocmfpx_head != quantizer["revision"]:
        raise PipelineError(f"ROCmFPX checkout must be at {quantizer['revision']}, got {rocmfpx_head}")
    if not HEX40.fullmatch(ember_revision) or ember_head != ember_revision:
        raise PipelineError(f"Ember checkout must be at requested revision {ember_revision}, got {ember_head}")
    bounded = conversion.get("bounded_memory")
    if not isinstance(bounded, dict):
        raise PipelineError("conversion bounded-memory contract is missing")
    patch_path = ember_dir / str(bounded.get("ple_cgroup_writeback_patch", ""))
    if (not patch_path.is_file()
            or sha256_file(patch_path) != bounded.get("ple_cgroup_writeback_patch_sha256")):
        raise PipelineError("pinned Qwen PLE cgroup-writeback patch is missing or changed")
    llama_dirty = run_git_checked(
        git, llama_dir, ["status", "--porcelain", "--untracked-files=all"])
    if llama_dirty != "M conversion/qwen4exp.py":
        raise PipelineError(
            "llama.cpp must differ from the pinned base only by the audited Qwen PLE patch")
    run_checked([
        git, "-c", f"safe.directory={llama_dir}", "-C", str(llama_dir),
        "apply", "--reverse", "--check", str(patch_path),
    ])
    clean_checkouts = [(rocmfpx_dir, "ROCmFPX")]
    if ember_has_git:
        clean_checkouts.append((ember_dir, "Ember"))
    for directory, label in clean_checkouts:
        dirty = run_git_checked(
            git, directory, ["status", "--porcelain", "--untracked-files=all"])
        if dirty:
            raise PipelineError(f"{label} checkout is dirty; pinned conversion requires a clean tree")
    converter = llama_dir / "convert_hf_to_gguf.py"
    qwen_converter = llama_dir / "conversion" / "qwen4exp.py"
    quantizer_source = rocmfpx_dir / "tools" / "quantize" / "quantize.cpp"
    for path in (converter, qwen_converter, quantizer_source):
        if not path.is_file():
            raise PipelineError(f"required tool source is missing: {path}")
    if sha256_file(qwen_converter) != bounded.get("patched_qwen4exp_sha256"):
        raise PipelineError("patched Qwen4Exp converter digest differs from the audited cgroup fix")
    qwen_text = qwen_converter.read_text(encoding="utf-8")
    if "Qwen4ExpForConditionalGeneration" not in qwen_text or "_read_hash_constants" not in qwen_text:
        raise PipelineError("pinned converter lacks Qwen4Exp exact PLE metadata handling")
    quant_text = quantizer_source.read_text(encoding="utf-8")
    if "Q4_0_ROCMI4" not in quant_text or "arg_idx < argc && strncmp" not in quant_text:
        raise PipelineError("pinned ROCmFPX source lacks ROCMI4 or the audited option parser")
    quantizer_binary = quantizer_binary.resolve()
    if not quantizer_binary.is_file() or not os.access(quantizer_binary, os.X_OK):
        raise PipelineError(f"quantizer is not an executable file: {quantizer_binary}")
    try:
        build_info = json.loads(run_checked([str(quantizer_binary), "--build-info-json"]))
    except json.JSONDecodeError as exc:
        raise PipelineError("quantizer --build-info-json did not return a JSON object") from exc
    expected_info = {
        "tool": profile["quantization"]["tool"],
        "ember_revision": ember_revision,
        "rocmfpx_revision": quantizer["revision"],
        "format": profile["quantization"]["format"],
        "ggml_tensor_type": profile["quantization"]["ggml_tensor_type"],
        "intervention_manifest_schema": profile["intervention"]["manifest_schema_version"],
        "per_tensor_formats": list(SUPPORTED_TENSOR_FORMATS),
    }
    if not isinstance(build_info, dict) or any(build_info.get(key) != value for key, value in expected_info.items()):
        raise PipelineError(f"quantizer build provenance mismatch; required fields are {expected_info}")
    result = {
        "python": platform.python_version(),
        "git": git_version,
        "llama_cpp_revision": llama_head,
        "llama_cpp_base_revision": llama_parent,
        "converter_sha256": sha256_file(converter),
        "qwen4exp_converter_sha256": sha256_file(qwen_converter),
        "ple_cgroup_writeback_patch_sha256": sha256_file(patch_path),
        "rocmfpx_revision": rocmfpx_head,
        "ember_revision": ember_head,
        "ember_revision_source": ember_revision_source,
        "quantizer_binary": str(quantizer_binary),
        "quantizer_sha256": sha256_file(quantizer_binary),
        "quantizer_build_info": build_info,
        "composite_support": [
            "qwen4exp-converter", "architecture-agnostic-streaming",
            *SUPPORTED_TENSOR_FORMATS,
        ],
    }
    if gguf_splitter is not None:
        splitter = gguf_splitter.resolve()
        if not splitter.is_file() or not os.access(splitter, os.X_OK):
            raise PipelineError(f"GGUF splitter is not an executable file: {splitter}")
        if not splitter.is_relative_to(llama_dir.resolve()):
            raise PipelineError("GGUF splitter must come from the pinned llama.cpp checkout")
        version = run_checked_combined([str(splitter), "--version"])
        version_commit = re.search(r"\bcommit\s+([0-9a-f]{7,40})\b", version)
        if version_commit is None or not conversion["revision"].startswith(version_commit.group(1)):
            raise PipelineError("GGUF splitter build revision does not match pinned llama.cpp")
        result["gguf_splitter_binary"] = str(splitter)
        result["gguf_splitter_sha256"] = sha256_file(splitter)
        result["gguf_splitter_version"] = version
    return result


def physical_ram_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        raise PipelineError("cannot determine physical RAM on this host")


def existing_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists():
        if current.parent == current:
            raise PipelineError(f"cannot find an existing parent for {path}")
        current = current.parent
    return current


def validate_resources(work_dir: Path, minimum_free_gib: int, minimum_ram_gib: int) -> dict[str, int]:
    if minimum_free_gib < 0 or minimum_ram_gib < 0:
        raise PipelineError("resource minimums cannot be negative")
    free = shutil.disk_usage(existing_parent(work_dir)).free
    ram = physical_ram_bytes()
    if free < minimum_free_gib * GIB:
        raise PipelineError(f"insufficient free disk: need {minimum_free_gib} GiB, have {free / GIB:.1f} GiB")
    if ram < minimum_ram_gib * GIB:
        raise PipelineError(f"insufficient physical RAM: need {minimum_ram_gib} GiB, have {ram / GIB:.1f} GiB")
    return {"free_disk_bytes": free, "physical_ram_bytes": ram, "minimum_free_gib": minimum_free_gib, "minimum_ram_gib": minimum_ram_gib}


def read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise PipelineError("truncated GGUF structure")
    return value


def read_u32(stream: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(stream, 4))[0]


def read_u64(stream: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(stream, 8))[0]


def read_gguf_string(stream: BinaryIO) -> str:
    length = read_u64(stream)
    if length > 64 * 1024 * 1024:
        raise PipelineError("unreasonable GGUF string length")
    return read_exact(stream, length).decode("utf-8")


def read_gguf_value(stream: BinaryIO, value_type: int) -> tuple[Any, int | None]:
    if value_type in FIXED_FORMATS:
        fmt = "<" + FIXED_FORMATS[value_type]
        return struct.unpack(fmt, read_exact(stream, struct.calcsize(fmt)))[0], None
    if value_type == 8:
        return read_gguf_string(stream), None
    if value_type == 9:
        subtype = read_u32(stream)
        if subtype == 9 or subtype not in GGUF_TYPES:
            raise PipelineError(f"unsupported GGUF array subtype {subtype}")
        count = read_u64(stream)
        if count > 100_000_000:
            raise PipelineError("unreasonable GGUF array length")
        values = [read_gguf_value(stream, subtype)[0] for _ in range(count)]
        return values, subtype
    raise PipelineError(f"unknown GGUF metadata type {value_type}")


def inspect_gguf(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        if read_exact(stream, 4) != b"GGUF":
            raise PipelineError(f"not a GGUF file: {path}")
        version = read_u32(stream)
        if version not in (2, 3):
            raise PipelineError(f"unsupported GGUF version {version}: {path}")
        tensor_count = read_u64(stream)
        metadata_count = read_u64(stream)
        if tensor_count < 1 or tensor_count > 1_000_000 or metadata_count > 1_000_000:
            raise PipelineError(f"GGUF header counts are outside audit bounds: {path}")
        metadata: dict[str, dict[str, Any]] = {}
        for _ in range(metadata_count):
            key = read_gguf_string(stream)
            value_type = read_u32(stream)
            value, subtype = read_gguf_value(stream, value_type)
            metadata[key] = {"value": value, "type": value_type, "subtype": subtype}
        tensors = []
        for _ in range(tensor_count):
            name = read_gguf_string(stream)
            dimensions = read_u32(stream)
            shape = [read_u64(stream) for _ in range(dimensions)]
            tensor_type = read_u32(stream)
            offset = read_u64(stream)
            tensors.append({"name": name, "shape": shape, "type": tensor_type, "offset": offset})
    return {"path": str(path), "version": version, "metadata": metadata, "tensors": tensors}


def validate_mtp_companion_gguf(
    path: Path, matrix_contract: str, source_revision: str,
) -> dict[str, Any]:
    """Verify the runtime-visible MTP GGUF contract and exact type partition."""
    inspected = inspect_gguf(path)
    metadata = inspected["metadata"]
    expected_metadata = {
        "general.architecture": "qwen4exp-mtp",
        "qwen4exp-mtp.source_revision": source_revision,
        "qwen4exp-mtp.shared_main_weights": True,
        "ember.mtp.matrix_quant_contract": matrix_contract,
    }
    if any(metadata.get(key, {}).get("value") != value
           for key, value in expected_metadata.items()):
        raise PipelineError("MTP companion GGUF metadata does not match the pinned runtime contract")
    tensors = inspected["tensors"]
    by_name = {item["name"]: item for item in tensors}
    expected_names = MTP_QUANTIZED_MATRIX_NAMES | MTP_BF16_TENSOR_NAMES
    if len(by_name) != len(tensors) or set(by_name) != expected_names:
        raise PipelineError("MTP companion GGUF tensor inventory does not match the pinned exporter")
    matrix_type = SUPPORTED_TENSOR_FORMATS[matrix_contract]
    if any(by_name[name]["type"] != matrix_type for name in MTP_QUANTIZED_MATRIX_NAMES):
        raise PipelineError("MTP companion GGUF matrix tensors do not match its quant contract")
    if any(by_name[name]["type"] != 30 for name in MTP_BF16_TENSOR_NAMES):
        raise PipelineError("MTP companion routers/norms must remain BF16")
    return {
        "version": inspected["version"],
        "architecture": "qwen4exp-mtp",
        "source_revision": source_revision,
        "matrix_quant_contract": matrix_contract,
        "matrix_ggml_tensor_type": matrix_type,
        "matrix_tensor_count": len(MTP_QUANTIZED_MATRIX_NAMES),
        "bf16_tensor_count": len(MTP_BF16_TENSOR_NAMES),
        "tensor_count": len(tensors),
        "tensor_names_sha256": hashlib.sha256(
            "\n".join(sorted(by_name)).encode("utf-8")).hexdigest(),
    }


def validate_mtp_export_manifest(
    path: Path, expected_sha256: Any, artifact: dict[str, Any],
    matrix_contract: str, profile: dict[str, Any],
) -> dict[str, Any]:
    """Bind the MTP artifact to the audited exporter and quantizer evidence."""
    if not isinstance(expected_sha256, str):
        raise PipelineError("MTP export manifest SHA-256 must be a string")
    manifest, evidence = read_exact_json_file(
        path.absolute(), expected_sha256, "MTP export manifest")
    matrix_names = manifest.get("quantized_matrix_tensors")
    build_info = manifest.get("quantizer_build_info")
    expected_type = SUPPORTED_TENSOR_FORMATS[matrix_contract]
    required = {
        "schema": "ember.qwen4exp.mtp-gguf-export.v1",
        "source_revision": profile["source"]["revision"],
        "quantized_output": artifact["path"],
        "quantized_bytes": artifact["size_bytes"],
        "quantized_sha256": artifact["sha256"],
        "quantized_matrix_contract": matrix_contract,
        "quantized_matrix_ggml_type": expected_type,
        "quantized_matrix_tensor_count": len(MTP_QUANTIZED_MATRIX_NAMES),
        "runtime_status": "loadable quantized companion",
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise PipelineError("MTP export manifest does not bind the exact companion artifact")
    if (not isinstance(matrix_names, list) or len(matrix_names) != len(MTP_QUANTIZED_MATRIX_NAMES)
            or set(matrix_names) != MTP_QUANTIZED_MATRIX_NAMES):
        raise PipelineError("MTP export manifest matrix tensor inventory is not exact")
    quantizer = profile["quantizer"]
    if (not isinstance(build_info, dict)
            or build_info.get("tool") != profile["quantization"]["tool"]
            or build_info.get("rocmfpx_revision") != quantizer["revision"]
            or HEX40.fullmatch(str(build_info.get("ember_revision", ""))) is None
            or build_info.get("format") != profile["quantization"]["format"]
            or build_info.get("ggml_tensor_type") != profile["quantization"]["ggml_tensor_type"]
            or build_info.get("per_tensor_formats") != list(SUPPORTED_TENSOR_FORMATS)):
        raise PipelineError("MTP export manifest quantizer build provenance is invalid")
    quantizer_sha = manifest.get("quantizer_sha256")
    if not isinstance(quantizer_sha, str) or SHA256_RE.fullmatch(quantizer_sha) is None:
        raise PipelineError("MTP export manifest quantizer SHA-256 is invalid")
    return {
        **evidence,
        "schema": required["schema"],
        "quantizer_sha256": quantizer_sha,
        "quantizer_build_info": build_info,
        "matrix_tensor_count": len(matrix_names),
    }


def validate_bf16_qwen_mmproj_gguf(path: Path) -> dict[str, Any]:
    """Verify the separate vision artifact before counting it as release-ready."""
    inspected = inspect_gguf(path)
    metadata = inspected["metadata"]
    expected = {
        "general.architecture": "clip",
        "general.file_type": 32,
        "clip.projector_type": "qwen3vl_merger",
        "clip.has_vision_encoder": True,
        "clip.vision.projection_dim": 2560,
        "clip.vision.spatial_merge_size": 2,
    }
    if any(metadata.get(key, {}).get("value") != value for key, value in expected.items()):
        raise PipelineError("vision mmproj GGUF metadata is not the pinned Qwen3.8 BF16 tower")
    tensor_types = {item["type"] for item in inspected["tensors"]}
    if 30 not in tensor_types or not tensor_types.issubset({0, 30}):
        raise PipelineError("vision mmproj tensor inventory is not unquantized BF16/F32")
    return {
        "version": inspected["version"], "tensor_count": len(inspected["tensors"]),
        "tensor_types": sorted(tensor_types), "metadata": expected,
    }


def discover_gguf(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    matches = sorted(base.parent.glob(base.stem + "-?????-of-?????.gguf"))
    if not matches:
        raise PipelineError(f"GGUF output was not produced: {base}")
    parsed = [SHARD_RE.fullmatch(path.name) for path in matches]
    if any(match is None for match in parsed):
        raise PipelineError("invalid GGUF shard filename")
    counts = {int(match.group("count")) for match in parsed if match is not None}
    numbers = [int(match.group("number")) for match in parsed if match is not None]
    if len(counts) != 1 or numbers != list(range(1, len(matches) + 1)) or counts != {len(matches)}:
        raise PipelineError("GGUF shard sequence is incomplete")
    return matches


def expected_gguf_outputs(base: Path, shard_count: int) -> list[Path]:
    if shard_count < 1:
        raise PipelineError("GGUF output shard count must be positive")
    if shard_count == 1:
        return [base]
    return [
        base.with_name(f"{base.stem}-{index:05d}-of-{shard_count:05d}.gguf")
        for index in range(1, shard_count + 1)
    ]


def safetensor_ple_constants(snapshot: Path) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for path in sorted(snapshot.glob("model-*.safetensors")):
        with path.open("rb") as stream:
            header_length = struct.unpack("<Q", read_exact(stream, 8))[0]
            if header_length > 256 * 1024 * 1024:
                raise PipelineError(f"unreasonable safetensors header: {path}")
            header = json.loads(read_exact(stream, header_length).decode("utf-8"))
            data_start = 8 + header_length
            for name, info in header.items():
                if name == "__metadata__" or not isinstance(info, dict):
                    continue
                for suffix in PLE_SUFFIXES:
                    if not name.endswith(suffix):
                        continue
                    if info.get("dtype") != "I64":
                        raise PipelineError(f"PLE constant {name} must be I64 in the source snapshot")
                    start, end = info["data_offsets"]
                    if (end - start) % 8:
                        raise PipelineError(f"invalid I64 byte length for {name}")
                    stream.seek(data_start + start)
                    raw = read_exact(stream, end - start)
                    found[suffix] = list(struct.unpack("<" + "q" * ((end - start) // 8), raw))
    missing = sorted(set(PLE_SUFFIXES) - set(found))
    if missing:
        raise PipelineError(f"source snapshot lacks I64 PLE constants: {missing}")
    return found


def verify_gguf_set(
    paths: list[Path], expected_ple: dict[str, list[int]], *, quantized: bool,
    profile: dict[str, Any], intervention: dict[str, Any] | None = None,
    stock_control: bool = False,
    tensor_overrides: list[str] | None = None,
) -> dict[str, Any]:
    inspected = [inspect_gguf(path) for path in paths]
    all_tensors = [tensor for item in inspected for tensor in item["tensors"]]
    names = [tensor["name"] for tensor in all_tensors]
    if len(names) != len(set(names)):
        raise PipelineError("GGUF tensor inventory contains duplicate names across shards")
    invariant_metadata: dict[str, dict[str, Any]] | None = None
    for shard_index, item in enumerate(inspected):
        metadata = item["metadata"]
        architecture = metadata.get("general.architecture", {}).get("value")
        if architecture != "qwen4exp":
            raise PipelineError(
                f"GGUF shard {shard_index + 1} architecture must be qwen4exp, got {architecture!r}"
            )
        for source_suffix, gguf_key in PLE_SUFFIXES.items():
            field = metadata.get(gguf_key)
            if field is None or field["type"] != 9 or field["subtype"] != 10:
                raise PipelineError(
                    f"GGUF shard {shard_index + 1} PLE metadata {gguf_key} must be ARRAY<UINT64>"
                )
            if field["value"] != expected_ple[source_suffix]:
                raise PipelineError(
                    f"GGUF shard {shard_index + 1} PLE metadata {gguf_key} does not match the source I64 values"
                )
        if quantized:
            if intervention is not None:
                intervention_fields = {
                    "ember.intervention.kind": (8, intervention["kind"]),
                    "ember.intervention.application_stage": (8, intervention["application_stage"]),
                    "ember.intervention.manifest_sha256": (8, intervention["manifest_sha256"]),
                    "ember.intervention.target_names_sha256": (8, intervention["target_names_sha256"]),
                    "ember.intervention.target_count": (4, intervention["target_count"]),
                }
                for key, (expected_type, expected_value) in intervention_fields.items():
                    field = metadata.get(key)
                    if field is None or field.get("type") != expected_type or field.get("value") != expected_value:
                        raise PipelineError(
                            f"GGUF shard {shard_index + 1} intervention metadata {key} does not match the applied manifest"
                        )
            elif stock_control:
                if any(key.startswith("ember.intervention.") for key in metadata):
                    raise PipelineError(
                        f"stock-control GGUF shard {shard_index + 1} carries intervention metadata"
                    )
            else:
                raise PipelineError(
                    "quantized GGUF verification requires intervention evidence or explicit stock-control mode"
                )
        # llama.cpp split files intentionally vary only split.no. Every other
        # metadata value is release-critical provenance/configuration and must
        # remain byte-semantically identical across the set.
        current_invariants = {
            key: value for key, value in metadata.items() if key != "split.no"
        }
        if invariant_metadata is None:
            invariant_metadata = current_invariants
        elif current_invariants != invariant_metadata:
            differing = sorted(
                key for key in set(invariant_metadata) | set(current_invariants)
                if invariant_metadata.get(key) != current_invariants.get(key)
            )
            raise PipelineError(
                f"GGUF shard {shard_index + 1} invariant metadata differs from shard 1: {differing}"
            )
    if len(paths) > 1:
        for index, item in enumerate(inspected):
            metadata = item["metadata"]
            if metadata.get("split.count", {}).get("value") != len(paths):
                raise PipelineError("GGUF split.count does not match discovered shard count")
            if metadata.get("split.no", {}).get("value") != index:
                raise PipelineError("GGUF split.no is not contiguous")
            if metadata.get("split.tensors.count", {}).get("value") != len(all_tensors):
                raise PipelineError("GGUF split.tensors.count does not match tensor inventory")
    type_counts = Counter(tensor["type"] for tensor in all_tensors)
    override_evidence: list[dict[str, Any]] | None = None
    if quantized:
        ple_name = profile["quantization"]["ple_tensor_name"]
        ple = next((tensor for tensor in all_tensors if tensor["name"] == ple_name), None)
        expected_type = profile["quantization"]["ggml_tensor_type"]
        if ple is None or ple["type"] != expected_type:
            raise PipelineError(f"{ple_name} must use GGML tensor type {expected_type}")
        if type_counts[expected_type] == 0:
            raise PipelineError(f"quantized GGUF contains no tensor type {expected_type}")
        selected_overrides = tensor_overrides or [
            profile["quantization"]["ple_tensor_override"]
        ]
        parsed_overrides = [
            parse_profile_tensor_override(value, f"output tensor override {index}")
            for index, value in enumerate(selected_overrides)
        ]
        matched = [False] * len(parsed_overrides)
        matched_names: list[list[str]] = [[] for _ in parsed_overrides]
        for tensor in all_tensors:
            matches = [index for index, override in enumerate(parsed_overrides)
                       if override["compiled"].search(tensor["name"])]
            if len(matches) > 1:
                raise PipelineError(
                    f"multiple selected tensor overrides match output tensor {tensor['name']}")
            if matches:
                index = matches[0]
                matched[index] = True
                matched_names[index].append(tensor["name"])
                expected_override_type = parsed_overrides[index]["ggml_tensor_type"]
                if tensor["type"] != expected_override_type:
                    raise PipelineError(
                        f"output tensor {tensor['name']} has type {tensor['type']}, "
                        f"expected {expected_override_type} from selected override")
        if not all(matched):
            missing = [selected_overrides[index] for index, value in enumerate(matched)
                       if not value]
            raise PipelineError(f"selected tensor overrides matched no output tensor: {missing}")
        override_evidence = []
        for override, names_for_override in zip(
                parsed_overrides, matched_names, strict=True):
            ordered_names = sorted(names_for_override)
            override_evidence.append({
                "override": override["value"],
                "format": override["format"],
                "ggml_tensor_type": override["ggml_tensor_type"],
                "matched_tensor_count": len(ordered_names),
                "matched_tensor_names_sha256": hashlib.sha256(
                    "\n".join(ordered_names).encode("utf-8")).hexdigest(),
            })
    result = {
        "shards": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        ],
        "tensor_count": len(all_tensors),
        "tensor_names_sha256": hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest(),
        "tensor_type_counts": {str(key): value for key, value in sorted(type_counts.items())},
    }
    if override_evidence is not None:
        result["tensor_override_evidence"] = override_evidence
    return result


def planned_commands(
    args: argparse.Namespace, profile: dict[str, Any], work_dir: Path,
    intervention_manifest: Path | None, quantization_arm: dict[str, Any],
) -> tuple[list[str] | None, list[str] | None, list[str], list[str], Path, Path, Path | None]:
    intermediate = work_dir / "Qwen3.8-Flash-Next-BF16.gguf"
    unsplit = (
        work_dir / "Qwen3.8-Flash-Next-BF16.unsplit.gguf"
        if args.bounded_memory_temp and args.split_max_size != "0" else None
    )
    output_name = (
        "Qwen3.8-Flash-Next-Stock-Control-ROCmI4-Strix-Halo.gguf"
        if args.stock_control else profile["artifact"]["filename"]
    )
    output = work_dir / output_name
    converter = args.llama_cpp_dir.resolve() / "convert_hf_to_gguf.py"
    convert: list[str] | None = None
    if args.bf16_cache_manifest is None:
        convert = [
            sys.executable, str(converter), str(args.snapshot_dir.resolve()),
            "--outfile", str(unsplit or intermediate), "--outtype", "bf16",
        ]
    split: list[str] | None = None
    if convert is not None and unsplit is not None:
        convert.append("--use-temp-file")
        assert args.gguf_splitter is not None
        split = [
            str(args.gguf_splitter.resolve()), "--split-max-size",
            args.split_max_size, str(unsplit), str(intermediate),
        ]
    elif convert is not None and args.split_max_size != "0":
        convert.extend(["--split-max-size", args.split_max_size])
    elif convert is not None and args.bounded_memory_temp:
        convert.append("--use-temp-file")
    quantize_options = []
    for override in quantization_arm["per_tensor_overrides"]:
        quantize_options.extend(["--tensor-type", override])
    if intervention_manifest is not None:
        quantize_options.extend(["--intervention-manifest", str(intervention_manifest)])
    if args.split_max_size != "0":
        quantize_options.append("--keep-split")
    memory_gate = profile["quantization"]["native_262k_memory_gate"]
    quantize_options.extend([
        "--device-budget-bytes", str(memory_gate["device_budget_bytes"]),
        "--runtime-reserve-bytes", str(memory_gate["runtime_reserve_bytes"]),
    ])
    # All options intentionally precede all positional arguments.  ROCmFPX's
    # parser stops scanning options at the first positional and otherwise drops
    # later flags without applying them.
    quantize = [
        str(args.quantizer.resolve()), *quantize_options,
        "<converted-first-shard>" if args.split_max_size != "0" else str(intermediate),
        "<private-same-filesystem-staging-output>",
        profile["quantization"]["format"], str(args.threads),
    ]
    preflight = [
        quantize[0], *quantize[1:-4], "--dry-size-json",
        quantize[-4], str(output), *quantize[-2:],
    ]
    return convert, split, preflight, quantize, intermediate, output, unsplit


def validate_memory_preflight(
    value: Any, profile: dict[str, Any],
    companion_inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineError("quantizer --dry-size-json did not return a JSON object")
    gate = profile["quantization"]["native_262k_memory_gate"]
    budget = gate["device_budget_bytes"]
    reserve = gate["runtime_reserve_bytes"]
    for key in (
        "artifact_bytes", "shard_count", "shard_bytes", "runtime_reserve_bytes",
        "budget_bytes", "total_bytes", "headroom_bytes", "fits",
    ):
        if key not in value:
            raise PipelineError(f"quantizer size preflight omitted {key}")
    if value["budget_bytes"] != budget or value["runtime_reserve_bytes"] != reserve:
        raise PipelineError("quantizer size preflight did not use the profile's exact budget and reserve")
    if not isinstance(value["artifact_bytes"], int) or value["artifact_bytes"] < 1:
        raise PipelineError("quantizer size preflight returned an invalid artifact size")
    if (
        not isinstance(value["shard_count"], int)
        or value["shard_count"] < 1
        or not isinstance(value["shard_bytes"], list)
        or len(value["shard_bytes"]) != value["shard_count"]
        or any(not isinstance(size, int) or size < 1 for size in value["shard_bytes"])
        or sum(value["shard_bytes"]) != value["artifact_bytes"]
    ):
        raise PipelineError("quantizer size preflight returned an invalid ordered shard inventory")
    if value["total_bytes"] != value["artifact_bytes"] + reserve:
        raise PipelineError("quantizer size preflight total is inconsistent")
    expected_fits = value["total_bytes"] <= budget
    if value["headroom_bytes"] != budget - value["total_bytes"]:
        raise PipelineError("quantizer size preflight headroom is inconsistent")
    if not isinstance(value["fits"], bool) or value["fits"] != expected_fits:
        raise PipelineError("quantizer size preflight fit result is inconsistent")
    if not expected_fits:
        raise PipelineError(
            "quantized artifact plus provisional non-artifact reserve does not fit the 128 GiB UMA budget"
        )
    certification_memtotal = gate["certification_host_memtotal_bytes"]
    certification_main_only_total = value["artifact_bytes"] + reserve
    if certification_main_only_total > certification_memtotal:
        raise PipelineError(
            "quantized main artifact plus reserve does not fit measured certification-host MemTotal"
        )
    result = dict(value)
    result["certification_host_memtotal_bytes"] = certification_memtotal
    result["certification_host_gtt_pages_limit"] = gate["certification_host_gtt_pages_limit"]
    result["certification_host_gtt_cap_bytes"] = gate["certification_host_gtt_cap_bytes"]
    result["certification_main_only_total_bytes"] = certification_main_only_total
    result["certification_main_only_headroom_bytes"] = (
        certification_memtotal - certification_main_only_total
    )
    companion = companion_inventory or {
        "status": "not_supplied", "enabled_artifact_bytes": None,
    }
    if companion.get("status") == "verified_exact":
        enabled_companion_bytes = companion.get("enabled_artifact_bytes")
        live = companion.get("live_gtt_evidence")
        if (not isinstance(enabled_companion_bytes, int)
                or isinstance(enabled_companion_bytes, bool)
                or enabled_companion_bytes < 1 or not isinstance(live, dict)):
            raise PipelineError("verified companion inventory evidence is incomplete")
        combined_total = certification_main_only_total + enabled_companion_bytes
        live_gtt_cap = live.get("runner_gtt_cap_bytes")
        if live_gtt_cap != gate["certification_host_gtt_cap_bytes"]:
            raise PipelineError("companion inventory live GTT evidence changed before preflight")
        fit_checks = {
            "device_budget": combined_total <= budget,
            "certification_memtotal": combined_total <= certification_memtotal,
            "live_gtt_cap": combined_total <= live_gtt_cap,
        }
        if not all(fit_checks.values()):
            failed = [name for name, fits in fit_checks.items() if not fits]
            raise PipelineError(
                "quantized artifact plus reserve plus enabled exact companions does not fit "
                + ", ".join(failed))
        result.update({
            "enabled_companion_artifact_bytes": enabled_companion_bytes,
            "combined_accounted_bytes": combined_total,
            "combined_device_budget_headroom_bytes": budget - combined_total,
            "combined_certification_memtotal_headroom_bytes": (
                certification_memtotal - combined_total),
            "combined_live_gtt_headroom_bytes": live_gtt_cap - combined_total,
            "combined_fit_checks": fit_checks,
            "combined_fits": True,
            "companion_artifact_fit_status": "verified_exact_fit",
        })
    else:
        result.update({
            "enabled_companion_artifact_bytes": None,
            "combined_accounted_bytes": None,
            "combined_fit_checks": None,
            "combined_fits": None,
            "companion_artifact_fit_status": "pending_exact_companion_inventory",
        })
    result["measured_peak_rss_gtt_status"] = "pending_target_run"
    return result


def validate_intervention_report(
    value: Any, evidence: dict[str, Any], *, applied: bool
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineError("quantizer intervention report is not a JSON object")
    if (
        value.get("intervention_manifest_sha256") != evidence["manifest_sha256"]
        or value.get("intervention_target_names_sha256") != evidence["target_names_sha256"]
        or value.get("intervention_target_count") != evidence["target_count"]
        or value.get("intervention_targets") != evidence["targets"]
        or value.get("intervention_validated") is not True
        or value.get("intervention_applied") is not applied
    ):
        stage = "application" if applied else "preflight"
        raise PipelineError(f"quantizer {stage} intervention evidence does not match the manifest")
    result = {
        "manifest_sha256": value["intervention_manifest_sha256"],
        "target_names_sha256": value["intervention_target_names_sha256"],
        "target_count": value["intervention_target_count"],
        "targets": value["intervention_targets"],
        "validated": True,
        "applied": applied,
    }
    if applied:
        metrics = value.get("intervention_metrics")
        if not isinstance(metrics, list) or len(metrics) != evidence["target_count"]:
            raise PipelineError("quantizer application omitted per-target intervention metrics")
        metric_fields = (
            "source_projection_l2", "stored_projection_l2",
            "stored_projection_ratio", "relative_frobenius_delta",
            "row_norm_relative_rmse", "row_norm_relative_max",
        )
        validated_metrics: list[dict[str, Any]] = []
        for expected_name, metric_value in zip(evidence["targets"], metrics, strict=True):
            metric = require_mapping(metric_value, "quantizer intervention metric")
            if metric.get("tensor_name") != expected_name:
                raise PipelineError("quantizer intervention metrics are not keyed to the exact target order")
            for field in metric_fields:
                number = metric.get(field)
                if (
                    isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(float(number)) or float(number) < 0.0
                ):
                    raise PipelineError(f"quantizer intervention metric {field} must be finite and non-negative")
            signed = metric.get("signed_projection_coefficient")
            if isinstance(signed, bool) or not isinstance(signed, (int, float)) or not math.isfinite(float(signed)):
                raise PipelineError("quantizer intervention signed projection coefficient must be finite")
            validated_metrics.append({
                "tensor_name": expected_name,
                **{field: metric[field] for field in metric_fields},
                "signed_projection_coefficient": signed,
            })
        result["metrics"] = validated_metrics
    elif "intervention_metrics" in value:
        raise PipelineError("quantizer preflight must not claim post-encoding intervention metrics")
    return result


def orchestrate(args: argparse.Namespace) -> dict[str, Any]:
    if args.bf16_cache_manifest is not None:
        return orchestrate_with_snapshot_lease(args)
    snapshot = args.snapshot_dir.resolve()
    with SnapshotReadLease(snapshot):
        return orchestrate_with_snapshot_lease(args)


def orchestrate_with_snapshot_lease(args: argparse.Namespace) -> dict[str, Any]:
    profile_path = args.profile.resolve()
    profile, inventory, inventory_path = validate_profile(profile_path)
    quantization_arms = validated_quantization_arms(profile)
    if args.quantization_arm == DEFAULT_QUANTIZATION_ARM:
        overrides = [profile["quantization"]["ple_tensor_override"]]
        serialized = json.dumps(overrides, separators=(",", ":"), ensure_ascii=True)
        quantization_arm = {
            "id": DEFAULT_QUANTIZATION_ARM,
            "default_matrix_format": "Q4_0_ROCMI4",
            "per_tensor_overrides": overrides,
            "per_tensor_overrides_sha256": hashlib.sha256(
                serialized.encode("utf-8")).hexdigest(),
            "formats": ["Q4_0_ROCMI4"],
            "mtp_matrix_quant_contract": "Q4_0_ROCMI4",
            "override_precedence":
                "exactly_one_matching_regex; overlap_is_an_error",
            "selection": "validated_top_level_profile_default",
        }
    else:
        quantization_arm = quantization_arms.get(args.quantization_arm)
    if quantization_arm is None:
        raise PipelineError(
            f"--quantization-arm must name one of "
            f"{[DEFAULT_QUANTIZATION_ARM, *quantization_arms]}")
    companion_inventory = validate_companion_inventory(
        args.companion_inventory,
        args.companion_inventory_sha256,
        args.ttm_pages_limit_path,
        profile,
        quantization_arm,
    )
    stock_control = bool(args.stock_control)
    if stock_control and args.quantization_arm not in {
            DEFAULT_QUANTIZATION_ARM, "rocmi4-control"}:
        raise PipelineError(
            "stock control must use the unchanged default or rocmi4-control arm")
    if (args.bakeoff_plan is None) != (args.bakeoff_plan_sha256 is None):
        raise PipelineError("--bakeoff-plan and --bakeoff-plan-sha256 are required together")
    if (args.bakeoff_plan is not None
            and (stock_control or args.quantization_arm != "rocmi4-control")):
        raise PipelineError(
            "bakeoff-plan authorization applies only to non-stock rocmi4-control sweep weights")
    intervention_source = (
        args.intervention_manifest.resolve()
        if args.intervention_manifest is not None else None
    )
    intervention: dict[str, Any] | None = None
    if intervention_source is not None:
        intervention_manifest, intervention = validate_intervention_manifest(
            intervention_source, profile
        )
    else:
        intervention_manifest = None
    sweep_authorization = None
    if not stock_control and args.quantization_arm == "rocmi4-control":
        if intervention_manifest is None:
            raise PipelineError("rocmi4-control sweep encoding requires a validated intervention")
        sweep_authorization = validate_rocmi4_sweep_authorization(
            args.bakeoff_plan,
            args.bakeoff_plan_sha256,
            profile_path,
            sha256_file(profile_path),
            quantization_arm,
            intervention_manifest,
        )
    final_work_dir = args.work_dir.parent.resolve() / args.work_dir.name
    runner = profile["release"]["conversion_runner_requirements"]
    minimum_ram_gib = args.min_ram_gib
    if minimum_ram_gib is None:
        minimum_ram_gib = (
            runner["bounded_memory_minimum_physical_ram_gib"]
            if args.bounded_memory_temp or args.bf16_cache_manifest is not None
            else runner["minimum_physical_ram_gib"]
        )
    resources = validate_resources(final_work_dir, args.min_free_gib, minimum_ram_gib)
    conversion_cgroup = (
        validate_conversion_cgroup(args)
        if args.execute and args.bounded_memory_temp else None
    )
    tools = validate_tools(
        args.llama_cpp_dir.resolve(), args.rocmfpx_dir.resolve(), args.ember_dir.resolve(),
        args.ember_revision, args.quantizer.resolve(), profile,
        args.gguf_splitter.resolve() if args.gguf_splitter is not None else None,
    )
    # Tool provenance is cheap and deterministic.  Fail it before direct-I/O
    # validation of the 360 GB source snapshot so a bad image cannot waste a
    # full model read before reporting its actual error.
    bf16_cache = validate_bf16_cache_manifest(
        args.bf16_cache_manifest,
        args.bf16_cache_manifest_sha256,
        profile,
        sha256_file(profile_path),
        tools,
    )
    if bf16_cache is None:
        snapshot = validate_snapshot(
            args.snapshot_dir.resolve(), args.snapshot_revision, profile, inventory)
        expected_ple = safetensor_ple_constants(args.snapshot_dir.resolve())
    else:
        if args.snapshot_revision != profile["source"]["revision"]:
            raise PipelineError("--snapshot-revision differs from the BF16 cache source")
        snapshot = {
            "revision": profile["source"]["revision"],
            "files_verified": None,
            "total_bytes": profile["source"]["weight_bytes"],
            "integrity_source": "content_addressed_bf16_cache",
            "cache_manifest_sha256": bf16_cache["manifest"]["sha256"],
        }
        expected_ple = bf16_cache["main"]["ple"]
    transaction_dir: Path | None = None
    if args.execute:
        requested_record = final_work_dir / "qwen-quant-build-record.json"
        if args.build_record is not None and args.build_record.absolute() != requested_record:
            raise PipelineError(
                "--build-record must name qwen-quant-build-record.json inside --work-dir in execute mode"
            )
        try:
            os.lstat(final_work_dir)
        except FileNotFoundError:
            pass
        else:
            raise PipelineError(
                f"refusing to overwrite existing transaction directory: {final_work_dir}"
            )
        final_work_dir.parent.mkdir(parents=True, exist_ok=True)
        transaction_dir = Path(tempfile.mkdtemp(
            prefix=f".{final_work_dir.name}.transaction-", dir=final_work_dir.parent
        ))
        work_dir = transaction_dir
        record_path = transaction_dir / "qwen-quant-build-record.json"
        if intervention_source is not None and intervention is not None:
            staged_manifest = transaction_dir / profile["intervention"]["manifest_filename"]
            try:
                copy_verified_input(
                    intervention_source, staged_manifest, intervention["manifest_sha256"]
                )
            except PipelineError:
                shutil.rmtree(transaction_dir, ignore_errors=True)
                transaction_dir = None
                raise
            intervention_command_path: Path | None = staged_manifest
        else:
            intervention_command_path = None
    else:
        work_dir = final_work_dir
        record_path = (
            args.build_record.absolute() if args.build_record
            else final_work_dir.with_name(final_work_dir.name + ".plan.json")
        )
        record_path.parent.mkdir(parents=True, exist_ok=True)
        if path_entry_exists(record_path):
            raise PipelineError(f"refusing to overwrite existing build record: {record_path}")
        intervention_command_path = intervention_source
    convert, split, preflight, quantize, intermediate, output, unsplit = planned_commands(
        args, profile, work_dir, intervention_command_path, quantization_arm
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "profile": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
        "snapshot_inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        "snapshot": snapshot,
        "bf16_cache": bf16_cache,
        "companion_inventory": companion_inventory,
        "intervention": intervention,
        "sweep_authorization": sweep_authorization,
        "experiment": {
            "kind": "stock_control" if stock_control else "directional_ablation",
            "stock_weights_unchanged": stock_control,
            "final_release_eligible": False,
            "eligibility_status": (
                "ineligible_stock_control" if stock_control
                else (
                    "pending_exact_companion_inventory_and_measured_bakeoff_and_hardware_certification"
                    if companion_inventory["status"] == "not_supplied"
                    else "pending_combined_memory_preflight_and_measured_bakeoff_and_hardware_certification"
                )
            ),
            "purpose": (
                "activation_capture_and_bakeoff_baseline"
                if stock_control else "measured_bakeoff_candidate"
            ),
        },
        "tools": tools,
        "resources": resources,
        "mode": "execute" if args.execute else "dry-run",
        "publishes": False,
        "credentials_accessed": False,
        "compute_mode": "exact_dequant",
        "w4a4_enabled": False,
        "quantization_recipe": {
            **quantization_arm,
            "selection": quantization_arm.get("selection", "validated_profile_arm"),
            "ple_tensor_override": profile["quantization"]["ple_tensor_override"],
            "ple_override_preserved": True,
            "quantizer_supported_formats": list(SUPPORTED_TENSOR_FORMATS),
        },
        "commands": {
            "convert": convert,
            "convert_shell": shlex.join(convert) if convert is not None else None,
            "split": split,
            "split_shell": shlex.join(split) if split is not None else None,
            "quantize_preflight": preflight,
            "quantize_preflight_shell": shlex.join(preflight),
            "quantize": quantize,
            "quantize_shell": shlex.join(quantize),
            "quantizer_options_precede_positionals": True,
        },
        "ple": {
            "source_dtype": "I64",
            "gguf_metadata_dtype": "ARRAY<UINT64>",
            "keys": sorted(PLE_SUFFIXES.values()),
        },
        "native_262k_memory_gate": profile["quantization"]["native_262k_memory_gate"],
        "conversion_memory": {
            "mode": (
                "reused_content_addressed_bf16_cache" if bf16_cache is not None
                else "bounded_temp_file_then_split" if args.bounded_memory_temp
                else "ordinary_lazy_tensor_registry"
            ),
            "full_in_memory_tensor_registry": (
                False if args.bounded_memory_temp or bf16_cache is not None else None),
            "temporary_directory": (
                "private_transaction_directory" if args.bounded_memory_temp else None),
            "target_measurement_status": (
                "reused_verified_cache_measurement"
                if bf16_cache is not None
                else profile["conversion"]["bounded_memory"]["target_measurement_status"]
                if args.bounded_memory_temp else "not_applicable"
            ),
            "certification_cgroup": (
                bf16_cache["measurement"] if bf16_cache is not None else conversion_cgroup),
        },
        "status": "planned",
    }
    if not args.execute:
        write_json_atomic(record_path, record, create=True)
        return record
    try:
        write_json_atomic(record_path, record, create=True)
        if bf16_cache is not None:
            intermediate_paths = [
                Path(row["path"]) for row in bf16_cache["main"]["shards"]
            ]
            record["intermediate"] = bf16_cache["main"]["gguf"]
            record["conversion_memory"]["measurement"] = bf16_cache["measurement"]
            write_json_atomic(record_path, record, create=False)
        elif args.bounded_memory_temp:
            assert convert is not None
            converter_temp = work_dir / ".converter-tmp"
            converter_temp.mkdir(mode=0o700)
            conversion_started = time.monotonic()
            child_rss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            cgroup_peak_before = (
                read_cgroup_counter(args.cgroup_memory_peak_path, "memory.peak")
                if conversion_cgroup is not None else None
            )
            run_checked(convert, env_overrides={"TMPDIR": str(converter_temp)})
            converter_child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            conversion_wall_seconds = time.monotonic() - conversion_started
            if conversion_cgroup is not None:
                cgroup_peak_after = read_cgroup_counter(
                    args.cgroup_memory_peak_path, "memory.peak")
                memory_limit = conversion_cgroup["memory_limit_bytes"]
                if cgroup_peak_after > memory_limit:
                    raise PipelineError("conversion cgroup peak exceeded its pinned memory limit")
                if converter_child_rss * 1024 > memory_limit:
                    raise PipelineError("converter child maximum RSS exceeded its pinned memory limit")
                record["conversion_memory"]["measurement"] = {
                    "status": "measured_target_cgroup_v2",
                    "memory_limit_bytes": memory_limit,
                    "swap_limit_bytes": conversion_cgroup["swap_limit_bytes"],
                    "cgroup_peak_before_bytes": cgroup_peak_before,
                    "cgroup_peak_after_conversion_bytes": cgroup_peak_after,
                    "child_maximum_resident_set_kib_before_conversion": child_rss_before,
                    "converter_child_maximum_resident_set_kib": converter_child_rss,
                    "converter_wall_seconds": round(conversion_wall_seconds, 3),
                }
                record["conversion_memory"]["target_measurement_status"] = (
                    "measured_within_pinned_no_swap_cgroup"
                )
                write_json_atomic(record_path, record, create=False)
            record["conversion_memory"]["gguf_writer_temp_cleanup"] = {
                "policy": "exact_converter_private_tmp_residue_v2",
                "removed": cleanup_gguf_writer_temp(converter_temp),
            }
            write_json_atomic(record_path, record, create=False)
        else:
            assert convert is not None
            run_checked(convert)
        if bf16_cache is None:
            if split is not None:
                assert unsplit is not None
                run_checked(split)
            intermediate_paths = discover_gguf(intermediate)
            record["intermediate"] = verify_gguf_set(
                intermediate_paths, expected_ple, quantized=False, profile=profile)
            if unsplit is not None:
                try:
                    unsplit.unlink()
                except OSError as exc:
                    raise PipelineError(f"cannot remove private unsplit BF16 intermediate: {exc}") from exc
        if "<converted-first-shard>" in quantize:
            first_shard = str(intermediate_paths[0])
            quantize[quantize.index("<converted-first-shard>")] = first_shard
            preflight[preflight.index("<converted-first-shard>")] = first_shard
        record["commands"]["quantize_preflight"] = preflight
        record["commands"]["quantize_preflight_shell"] = shlex.join(preflight)
        record["commands"]["quantize"] = quantize
        record["commands"]["quantize_shell"] = shlex.join(quantize)
        write_json_atomic(record_path, record, create=False)
        try:
            preflight_json = json.loads(run_checked(preflight))
        except json.JSONDecodeError as exc:
            raise PipelineError("quantizer --dry-size-json returned invalid JSON") from exc
        current_companion_inventory = validate_companion_inventory(
            args.companion_inventory,
            args.companion_inventory_sha256,
            args.ttm_pages_limit_path,
            profile,
            quantization_arm,
        )
        if current_companion_inventory != record["companion_inventory"]:
            raise PipelineError("companion inventory evidence changed during conversion")
        record["memory_preflight"] = validate_memory_preflight(
            preflight_json, profile, current_companion_inventory)
        if current_companion_inventory["status"] == "verified_exact":
            record["companion_inventory"]["fit_status"] = "verified_exact_fit"
            pending_evidence = current_companion_inventory["pending_release_evidence"]
            next_status = (
                "pending_" + "_and_".join([
                    *pending_evidence,
                    "measured_bakeoff_and_hardware_certification",
                ])
                if pending_evidence
                else "pending_measured_bakeoff_and_hardware_certification"
            )
            record["companion_inventory"]["final_release_eligibility"] = next_status
            if not stock_control:
                record["experiment"]["eligibility_status"] = next_status
        if intervention is not None:
            record["intervention"]["quantizer_preflight"] = validate_intervention_report(
                preflight_json, intervention, applied=False
            )
        write_json_atomic(record_path, record, create=False)
        staged_outputs = expected_gguf_outputs(
            output, record["memory_preflight"]["shard_count"]
        )
        quantize[quantize.index("<private-same-filesystem-staging-output>")] = str(output)
        record["commands"]["quantize"] = quantize
        record["commands"]["quantize_shell"] = shlex.join(quantize)
        write_json_atomic(record_path, record, create=False)
        quantize_stdout = run_checked(quantize)
        if intervention is not None:
            try:
                application_json = json.loads(quantize_stdout)
            except json.JSONDecodeError as exc:
                raise PipelineError("quantizer did not return intervention application JSON") from exc
            record["intervention"]["quantizer_application"] = validate_intervention_report(
                application_json, intervention, applied=True
            )
        elif quantize_stdout.strip():
            raise PipelineError("stock-control quantizer unexpectedly returned intervention evidence")
        staged_paths = discover_gguf(output)
        final = verify_gguf_set(
            staged_paths, expected_ple, quantized=True, profile=profile,
            intervention=intervention, stock_control=stock_control,
            tensor_overrides=quantization_arm["per_tensor_overrides"],
        )
        if final["tensor_names_sha256"] != record["intermediate"]["tensor_names_sha256"]:
            raise PipelineError("quantization changed the GGUF tensor inventory")
        final_sizes = [item["size_bytes"] for item in final["shards"]]
        if (
            len(final_sizes) != record["memory_preflight"]["shard_count"]
            or final_sizes != record["memory_preflight"]["shard_bytes"]
            or sum(final_sizes) != record["memory_preflight"]["artifact_bytes"]
        ):
            raise PipelineError("ordered quantized output shard sizes differ from the preflight")
        if staged_paths != staged_outputs:
            raise PipelineError("quantizer output paths differ from the preflight shard plan")
        sync_verified_outputs(staged_paths)
        final_outputs = [final_work_dir / path.name for path in staged_paths]
        for shard, path in zip(final["shards"], final_outputs, strict=True):
            shard["path"] = str(path)
        record["output"] = final
        record["staging_transaction"] = {
            "boundary": "atomic_directory",
            "commit_method": "renameat2(RENAME_NOREPLACE)",
            "committed_directory": str(final_work_dir),
            "same_filesystem": True,
            "verified_before_promotion": True,
            "publication_state": "committed_on_visibility",
            "promoted": [str(path) for path in final_outputs],
            "evidence_promoted": (
                [str(final_work_dir / profile["intervention"]["manifest_filename"])]
                if intervention is not None else []
            ),
        }
        record["status"] = "complete"
        if bf16_cache is None:
            try:
                for path in intermediate_paths:
                    path.unlink()
            except OSError as exc:
                raise PipelineError(
                    f"cannot remove private BF16 intermediate before commit: {exc}"
                ) from exc
        write_json_atomic(record_path, record, create=False)
        fsync_directory(work_dir)
        assert transaction_dir is not None
        rename_directory_noreplace(transaction_dir, final_work_dir)
        transaction_dir = None
    except PipelineError as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        try:
            write_json_atomic(record_path, record, create=False)
        except PipelineError:
            pass
        raise
    finally:
        if transaction_dir is not None:
            shutil.rmtree(transaction_dir, ignore_errors=True)
    return record


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_profile = Path(__file__).resolve().parents[1] / "share" / "release_profiles" / "qwen3.8-flash-next-rocmi4-strix-halo.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=default_profile)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--snapshot-revision", required=True)
    recipe = parser.add_mutually_exclusive_group(required=True)
    recipe.add_argument(
        "--intervention-manifest", type=Path,
        help="completed directional-ablation manifest applied by the quantizer",
    )
    recipe.add_argument(
        "--stock-control", action="store_true",
        help="leave weights unchanged; final-ineligible activation/bakeoff control only",
    )
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument("--rocmfpx-dir", type=Path, required=True)
    parser.add_argument("--ember-dir", type=Path, required=True)
    parser.add_argument("--ember-revision", required=True)
    parser.add_argument("--quantizer", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="execute mode atomically publishes this previously absent directory")
    parser.add_argument("--build-record", type=Path,
                        help="dry-run record path; execute mode only accepts WORK_DIR/qwen-quant-build-record.json")
    parser.add_argument(
        "--companion-inventory", type=Path,
        help="exact MTP/vision companion inventory; omission keeps final release eligibility pending",
    )
    parser.add_argument(
        "--companion-inventory-sha256",
        help="required out-of-band SHA-256 binding for --companion-inventory",
    )
    parser.add_argument(
        "--ttm-pages-limit-path", type=Path,
        default=Path("/sys/module/ttm/parameters/pages_limit"),
        help="live TTM pages_limit evidence (must match the pinned 124 GiB cap)",
    )
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument(
        "--quantization-arm", default=DEFAULT_QUANTIZATION_ARM,
        help="validated profile performance_bakeoff arm (default: profile ROCMI4)",
    )
    parser.add_argument(
        "--bakeoff-plan", type=Path,
        help="canonical selection plan required only for non-stock rocmi4-control sweep weights",
    )
    parser.add_argument(
        "--bakeoff-plan-sha256",
        help="exact SHA-256 binding for --bakeoff-plan",
    )
    parser.add_argument("--split-max-size", default="48G", help="llama.cpp split size; release default is 48G, 0 writes one GGUF")
    parser.add_argument("--min-free-gib", type=int, default=1152)
    parser.add_argument(
        "--bounded-memory-temp", action="store_true",
        help="spill converter tensor payloads under WORK_DIR, then split the single BF16 GGUF",
    )
    parser.add_argument(
        "--gguf-splitter", type=Path,
        help="pinned llama-gguf-split executable; required by bounded mode with split output",
    )
    parser.add_argument(
        "--bf16-cache-manifest", type=Path,
        help="verified content-addressed BF16 cache; skips source conversion",
    )
    parser.add_argument(
        "--bf16-cache-manifest-sha256",
        help="required out-of-band SHA-256 binding for --bf16-cache-manifest",
    )
    parser.add_argument(
        "--conversion-memory-limit-bytes", type=int,
        help="require this exact cgroup-v2 memory.max and zero swap during bounded execution",
    )
    parser.add_argument(
        "--cgroup-memory-max-path", type=Path,
        default=Path("/sys/fs/cgroup/memory.max"),
    )
    parser.add_argument(
        "--cgroup-memory-swap-max-path", type=Path,
        default=Path("/sys/fs/cgroup/memory.swap.max"),
    )
    parser.add_argument(
        "--cgroup-memory-peak-path", type=Path,
        default=Path("/sys/fs/cgroup/memory.peak"),
    )
    parser.add_argument(
        "--min-ram-gib", type=int,
        help="override the profile RAM floor (256 GiB ordinary, 120 GiB bounded)",
    )
    parser.add_argument("--execute", action="store_true", help="run conversion and quantization; default only writes a plan")
    return parser.parse_args(argv)


def reported_record_path(args: argparse.Namespace) -> Path:
    if args.execute:
        work_dir = args.work_dir.parent.resolve() / args.work_dir.name
        return work_dir / "qwen-quant-build-record.json"
    if args.build_record:
        return args.build_record.absolute()
    work_dir = args.work_dir.parent.resolve() / args.work_dir.name
    return work_dir.with_name(work_dir.name + ".plan.json")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        if args.threads < 1:
            raise PipelineError("--threads must be positive")
        if not re.fullmatch(r"0|[1-9][0-9]*[KMGT]", args.split_max_size):
            raise PipelineError("--split-max-size must be 0 or an integer followed by K, M, G, or T")
        if (args.bounded_memory_temp and args.bf16_cache_manifest is None
                and args.split_max_size != "0" and args.gguf_splitter is None):
            raise PipelineError("--bounded-memory-temp with split output requires --gguf-splitter")
        if (not args.bounded_memory_temp and args.bf16_cache_manifest is None
                and args.gguf_splitter is not None):
            raise PipelineError(
                "--gguf-splitter is only valid with bounded conversion or BF16 cache reuse")
        if args.bf16_cache_manifest is not None and args.gguf_splitter is None:
            raise PipelineError("BF16 cache reuse requires the pinned --gguf-splitter")
        if args.bf16_cache_manifest is not None and args.bounded_memory_temp:
            raise PipelineError("BF16 cache reuse and --bounded-memory-temp are mutually exclusive")
        if ((args.bf16_cache_manifest is None)
                != (args.bf16_cache_manifest_sha256 is None)):
            raise PipelineError(
                "--bf16-cache-manifest and --bf16-cache-manifest-sha256 must be supplied together")
        if args.conversion_memory_limit_bytes is not None:
            if args.conversion_memory_limit_bytes < GIB:
                raise PipelineError("--conversion-memory-limit-bytes must be at least 1 GiB")
            if not args.execute or not args.bounded_memory_temp:
                raise PipelineError(
                    "--conversion-memory-limit-bytes requires bounded execute mode")
        if ((args.companion_inventory is None)
                != (args.companion_inventory_sha256 is None)):
            raise PipelineError(
                "--companion-inventory and --companion-inventory-sha256 must be supplied together")
        record = orchestrate(args)
    except PipelineError as exc:
        print(f"qwen_quantize.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "mode": record["mode"], "publishes": False, "status": record["status"],
        "build_record": str(reported_record_path(args)),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
