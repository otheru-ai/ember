#!/usr/bin/env python3
"""Validate and stream-export the pinned Qwen3.8 Flash Next MTP head.

The official checkpoint interleaves the one-layer MTP tensors across the main
131 safetensor shards.  Transformers 36deb0 ignores them and llama.cpp Qwen4Exp
PRs #27742/#27774 explicitly disable MTP export.  This tool creates one
MTP-only safetensors file while using O(header) memory.  Its GGUF path can
optionally run Ember's audited quantizer for either of the two matrix formats
the MTP runtime supports.  The emitted manifest binds the selected format and
exact quantizer build to the artifact so the loader can fail closed rather
than silently mixing formats.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO, Callable


SOURCE_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
TRANSFORMERS_REVISION = "36deb0b53ed0863f4b4dfdea23dcaec7f3df3701"
HALOSPECKV_REVISION = "60ff854bdc25e27ee211ac0c4df896e9379edd3f"
COPY_CHUNK = 16 * 1024 * 1024
MAX_HEADER = 128 * 1024 * 1024


class ExportError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    quant_class: str
    dtype: str = "BF16"

    @property
    def bytes(self) -> int:
        return math.prod(self.shape) * 2


def _spec(name: str, shape: tuple[int, ...], quant_class: str) -> TensorSpec:
    return TensorSpec(name=name, shape=shape, quant_class=quant_class)


# Shapes are in native safetensors/PyTorch order.  The contract is pinned to
# the source headers at SOURCE_REVISION; no multi-architecture guessing occurs.
MTP_TENSORS = (
    _spec("mtp.hyper_connection_mixer.hc_norm.weight", (10240,), "norm"),
    _spec("mtp.hyper_connection_mixer.input_mix_weight_down.weight", (320, 10240), "matrix"),
    _spec("mtp.hyper_connection_mixer.input_mix_weight_up.weight", (10240, 320), "matrix"),
    _spec("mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", (4, 10240), "matrix"),
    _spec("mtp.layers.0.attn_hyper_connection.hc_norm.weight", (10240,), "norm"),
    _spec("mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight", (320, 10240), "matrix"),
    _spec("mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight", (10240, 320), "matrix"),
    _spec("mtp.layers.0.mlp.experts.gate_up_proj", (512, 1280, 2560), "experts"),
    _spec("mtp.layers.0.mlp.experts.down_proj", (512, 2560, 640), "experts"),
    _spec("mtp.layers.0.mlp.gate.weight", (512, 2560), "router"),
    _spec("mtp.layers.0.mlp.shared_expert_gate.weight", (1, 2560), "router"),
    _spec("mtp.layers.0.mlp.shared_expert.gate_proj.weight", (640, 2560), "matrix"),
    _spec("mtp.layers.0.mlp.shared_expert.up_proj.weight", (640, 2560), "matrix"),
    _spec("mtp.layers.0.mlp.shared_expert.down_proj.weight", (2560, 640), "matrix"),
    _spec("mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight", (4, 10240), "matrix"),
    _spec("mtp.layers.0.mlp_hyper_connection.hc_norm.weight", (10240,), "norm"),
    _spec("mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight", (320, 10240), "matrix"),
    _spec("mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight", (10240, 320), "matrix"),
    _spec("mtp.layers.0.self_attn.indexer.index_qk_proj.weight", (640, 2560), "matrix"),
    _spec("mtp.layers.0.self_attn.indexer.k_layernorm.weight", (128,), "norm"),
    _spec("mtp.layers.0.self_attn.indexer.q_layernorm.weight", (128,), "norm"),
    _spec("mtp.layers.0.self_attn.k_norm.weight", (256,), "norm"),
    _spec("mtp.layers.0.self_attn.o_proj.weight", (2560, 6144), "matrix"),
    _spec("mtp.layers.0.self_attn.k_proj.weight", (512, 2560), "matrix"),
    _spec("mtp.layers.0.self_attn.q_proj.weight", (12288, 2560), "matrix"),
    _spec("mtp.layers.0.self_attn.v_proj.weight", (512, 2560), "matrix"),
    _spec("mtp.layers.0.self_attn.q_norm.weight", (256,), "norm"),
    _spec("mtp.pre_fc_norm_embedding.weight", (2560,), "norm"),
    _spec("mtp.pre_fc_norm_hidden.weight", (10240,), "norm"),
    _spec("mtp.fc_embedding.weight", (2560, 2560), "fusion"),
    _spec("mtp.fc_hidden.weight", (2560, 2560), "fusion"),
)
MTP_BY_NAME = {spec.name: spec for spec in MTP_TENSORS}


@dataclasses.dataclass(frozen=True)
class SourceSlice:
    spec: TensorSpec
    shard: str
    file_offset: int
    bytes: int


@dataclasses.dataclass(frozen=True)
class GgufTensor:
    name: str
    shape: tuple[int, ...]  # GGUF ne order, inner dimension first
    source_name: str
    source_skip: int = 0
    source_bytes: int | None = None
    add_one: bool = False


def _gguf(name: str, shape: tuple[int, ...], source: str,
          *, skip: int = 0, size: int | None = None,
          add_one: bool = False) -> GgufTensor:
    return GgufTensor(name, shape, source, skip, size, add_one)


# Short names stay below GGML_MAX_NAME. Raw PyTorch [out,in] storage already
# has the byte order GGUF expects for ne=[in,out]; only index_qk is split.
MTP_GGUF_TENSORS = (
    _gguf("mtp_hc_norm.weight", (10240,), "mtp.hyper_connection_mixer.hc_norm.weight", add_one=True),
    _gguf("mtp_hc_down.weight", (10240, 320), "mtp.hyper_connection_mixer.input_mix_weight_down.weight"),
    _gguf("mtp_hc_up.weight", (320, 10240), "mtp.hyper_connection_mixer.input_mix_weight_up.weight"),
    _gguf("mtp.hc_attn_inject.weight", (10240, 4), "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight"),
    _gguf("mtp.hc_attn_norm.weight", (10240,), "mtp.layers.0.attn_hyper_connection.hc_norm.weight", add_one=True),
    _gguf("mtp.hc_attn_down.weight", (10240, 320), "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight"),
    _gguf("mtp.hc_attn_up.weight", (320, 10240), "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight"),
    _gguf("mtp.ffn_gate_up_exps.weight", (2560, 1280, 512), "mtp.layers.0.mlp.experts.gate_up_proj"),
    _gguf("mtp.ffn_down_exps.weight", (640, 2560, 512), "mtp.layers.0.mlp.experts.down_proj"),
    _gguf("mtp.ffn_gate_inp.weight", (2560, 512), "mtp.layers.0.mlp.gate.weight"),
    _gguf("mtp.ffn_gate_inp_shexp.weight", (2560, 1), "mtp.layers.0.mlp.shared_expert_gate.weight"),
    _gguf("mtp.ffn_gate_shexp.weight", (2560, 640), "mtp.layers.0.mlp.shared_expert.gate_proj.weight"),
    _gguf("mtp.ffn_up_shexp.weight", (2560, 640), "mtp.layers.0.mlp.shared_expert.up_proj.weight"),
    _gguf("mtp.ffn_down_shexp.weight", (640, 2560), "mtp.layers.0.mlp.shared_expert.down_proj.weight"),
    _gguf("mtp.hc_ffn_inject.weight", (10240, 4), "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight"),
    _gguf("mtp.hc_ffn_norm.weight", (10240,), "mtp.layers.0.mlp_hyper_connection.hc_norm.weight", add_one=True),
    _gguf("mtp.hc_ffn_down.weight", (10240, 320), "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight"),
    _gguf("mtp.hc_ffn_up.weight", (320, 10240), "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight"),
    _gguf("mtp.indexer.q_proj.weight", (2560, 512), "mtp.layers.0.self_attn.indexer.index_qk_proj.weight", size=512 * 2560 * 2),
    _gguf("mtp.indexer.k_proj.weight", (2560, 128), "mtp.layers.0.self_attn.indexer.index_qk_proj.weight", skip=512 * 2560 * 2, size=128 * 2560 * 2),
    _gguf("mtp.indexer.k_norm.weight", (128,), "mtp.layers.0.self_attn.indexer.k_layernorm.weight", add_one=True),
    _gguf("mtp.indexer.q_norm.weight", (128,), "mtp.layers.0.self_attn.indexer.q_layernorm.weight", add_one=True),
    _gguf("mtp.attn_k_norm.weight", (256,), "mtp.layers.0.self_attn.k_norm.weight", add_one=True),
    _gguf("mtp.attn_output.weight", (6144, 2560), "mtp.layers.0.self_attn.o_proj.weight"),
    _gguf("mtp.attn_k.weight", (2560, 512), "mtp.layers.0.self_attn.k_proj.weight"),
    _gguf("mtp.attn_q.weight", (2560, 12288), "mtp.layers.0.self_attn.q_proj.weight"),
    _gguf("mtp.attn_v.weight", (2560, 512), "mtp.layers.0.self_attn.v_proj.weight"),
    _gguf("mtp.attn_q_norm.weight", (256,), "mtp.layers.0.self_attn.q_norm.weight", add_one=True),
    _gguf("mtp_pre_emb_norm.weight", (2560,), "mtp.pre_fc_norm_embedding.weight", add_one=True),
    _gguf("mtp_pre_hc_norm.weight", (10240,), "mtp.pre_fc_norm_hidden.weight", add_one=True),
    _gguf("mtp_fc_emb.weight", (2560, 2560), "mtp.fc_embedding.weight"),
    _gguf("mtp_fc_hc.weight", (2560, 2560), "mtp.fc_hidden.weight"),
)

SUPPORTED_MATRIX_QUANTS = {
    "Q4_0_ROCMI4": 108,
    "Q4_0_ROCMFP4_FAST": 101,
}
_QUANTIZED_CLASSES = frozenset(("matrix", "fusion", "experts"))
MTP_QUANTIZED_MATRIX_NAMES = tuple(
    tensor.name for tensor in MTP_GGUF_TENSORS
    if MTP_BY_NAME[tensor.source_name].quant_class in _QUANTIZED_CLASSES
)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExportError(f"JSON root must be an object: {path}")
    return value


def _validate_config(config: dict) -> None:
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ExportError("config.json lacks text_config")
    mtp = text.get("mtp")
    expected = {
        "num_hidden_layers": 1,
        "hybrid": True,
        "layer_types": ["full_attention"],
    }
    if (text.get("model_type") != "qwen4_exp_text" or
            text.get("num_hidden_layers") != 48 or
            text.get("hidden_size") != 2560 or
            text.get("mtp_num_hidden_layers") != 1 or
            text.get("mtp_use_dedicated_embeddings") is not False or
            not isinstance(mtp, dict)):
        raise ExportError("source is not the pinned one-layer Qwen4Exp MTP contract")
    for key, value in expected.items():
        if mtp.get(key) != value:
            raise ExportError(f"unsupported text_config.mtp.{key}")


def read_safetensors_header(path: Path) -> tuple[int, dict]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ExportError(f"truncated safetensors prefix: {path}")
            header_len = struct.unpack("<Q", raw)[0]
            if header_len == 0 or header_len > MAX_HEADER:
                raise ExportError(f"invalid safetensors header length in {path}")
            header_raw = handle.read(header_len)
    except OSError as exc:
        raise ExportError(f"cannot read safetensors shard {path}: {exc}") from exc
    if len(header_raw) != header_len:
        raise ExportError(f"truncated safetensors header: {path}")
    try:
        header = json.loads(header_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"invalid safetensors header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ExportError(f"safetensors header root is not an object: {path}")
    return 8 + header_len, header


def validate_source(
    config: dict,
    index: dict,
    header_reader: Callable[[str], tuple[int, dict, int]],
) -> list[SourceSlice]:
    _validate_config(config)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ExportError("model.safetensors.index.json lacks weight_map")
    actual_names = {name for name in weight_map if name.startswith("mtp.")}
    expected_names = set(MTP_BY_NAME)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise ExportError(f"MTP tensor set mismatch: missing={missing}, extra={extra}")

    cache: dict[str, tuple[int, dict, int]] = {}
    plan: list[SourceSlice] = []
    for spec in MTP_TENSORS:
        shard = weight_map.get(spec.name)
        if not isinstance(shard, str) or not shard or Path(shard).name != shard:
            raise ExportError(f"unsafe or missing shard name for {spec.name}")
        if shard not in cache:
            cache[shard] = header_reader(shard)
        data_start, header, file_size = cache[shard]
        entry = header.get(spec.name)
        if (not isinstance(entry, dict) or entry.get("dtype") != spec.dtype or
                entry.get("shape") != list(spec.shape)):
            raise ExportError(f"wrong dtype/shape for {spec.name}")
        offsets = entry.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(not isinstance(value, int) for value in offsets)):
            raise ExportError(f"invalid data_offsets for {spec.name}")
        begin, end = offsets
        if (begin < 0 or end < begin or end - begin != spec.bytes or
                data_start + end > file_size):
            raise ExportError(f"invalid source byte extent for {spec.name}")
        plan.append(SourceSlice(spec, shard, data_start + begin, end - begin))
    return plan


def _output_header(matrix_quant: str) -> tuple[bytes, list[tuple[int, int]]]:
    header: dict[str, object] = {
        "__metadata__": {
            "format": "ember-qwen4exp-mtp-source-v1",
            "source_model": "Qwen/Qwen3.8-Flash-Next",
            "source_revision": SOURCE_REVISION,
            "transformers_revision": TRANSFORMERS_REVISION,
            "halospeckv_replay_revision": HALOSPECKV_REVISION,
            "matrix_quant_contract": matrix_quant,
            "shared_main_weights": "token_embd.weight,output.weight",
        }
    }
    offset = 0
    extents: list[tuple[int, int]] = []
    for spec in MTP_TENSORS:
        end = offset + spec.bytes
        header[spec.name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [offset, end],
        }
        extents.append((offset, end))
        offset = end
    raw = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    return struct.pack("<Q", len(raw)) + raw, extents


def _pack_gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _pack_gguf_kv(name: str, kind: int, value: object) -> bytes:
    encoded = _pack_gguf_string(name) + struct.pack("<I", kind)
    if kind == 4:  # GGUF_TYPE_UINT32
        return encoded + struct.pack("<I", int(value))
    if kind == 7:  # GGUF_TYPE_BOOL
        return encoded + struct.pack("<?", bool(value))
    if kind == 8:  # GGUF_TYPE_STRING
        return encoded + _pack_gguf_string(str(value))
    raise ExportError(f"unsupported internal GGUF metadata type {kind}")


def _gguf_header(matrix_quant: str) -> tuple[bytes, list[tuple[int, int]]]:
    metadata = (
        ("general.architecture", 8, "qwen4exp-mtp"),
        ("general.name", 8, "Qwen3.8-Flash-Next-MTP"),
        ("general.file_type", 4, 24),  # GGML_FTYPE_MOSTLY_BF16
        ("general.quantization_version", 4, 2),
        ("qwen4exp-mtp.block_count", 4, 1),
        ("qwen4exp-mtp.embedding_length", 4, 2560),
        ("qwen4exp-mtp.hyper_connection_count", 4, 4),
        ("qwen4exp-mtp.hyper_connection_low_rank", 4, 320),
        ("qwen4exp-mtp.attention.head_count", 4, 24),
        ("qwen4exp-mtp.attention.head_count_kv", 4, 2),
        ("qwen4exp-mtp.attention.key_length", 4, 256),
        ("qwen4exp-mtp.indexer.head_count", 4, 4),
        ("qwen4exp-mtp.indexer.key_length", 4, 128),
        ("qwen4exp-mtp.indexer.top_k", 4, 2048),
        ("qwen4exp-mtp.indexer.compress_ratio", 4, 4),
        ("qwen4exp-mtp.expert_count", 4, 512),
        ("qwen4exp-mtp.expert_used_count", 4, 10),
        ("qwen4exp-mtp.source_revision", 8, SOURCE_REVISION),
        ("qwen4exp-mtp.shared_main_weights", 7, True),
        ("ember.mtp.matrix_quant_contract", 8, matrix_quant),
        ("ember.mtp.strict_replay_revision", 8, HALOSPECKV_REVISION),
    )
    prefix = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(MTP_GGUF_TENSORS), len(metadata)))
    for item in metadata:
        prefix += _pack_gguf_kv(*item)

    offset = 0
    extents: list[tuple[int, int]] = []
    for tensor in MTP_GGUF_TENSORS:
        size = tensor.source_bytes
        if size is None:
            size = math.prod(tensor.shape) * 2
        offset = (offset + 31) // 32 * 32
        end = offset + size
        extents.append((offset, end))
        prefix += _pack_gguf_string(tensor.name)
        prefix += struct.pack("<I", len(tensor.shape))
        prefix += struct.pack("<" + "Q" * len(tensor.shape), *tensor.shape)
        prefix += struct.pack("<IQ", 30, offset)  # GGML_TYPE_BF16
        offset = end
    prefix += b"\0" * ((-len(prefix)) % 32)
    return bytes(prefix), extents


def _bf16_add_one(block: bytes) -> bytes:
    if len(block) % 2:
        raise ExportError("BF16 normalization tensor has an odd byte length")
    output = bytearray(len(block))
    for offset in range(0, len(block), 2):
        upper = struct.unpack_from("<H", block, offset)[0]
        value = struct.unpack("<f", struct.pack("<I", upper << 16))[0] + 1.0
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        # Round-to-nearest-even to BF16, matching ggml_fp32_to_bf16.
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        struct.pack_into("<H", output, offset, rounded & 0xFFFF)
    return bytes(output)


def export_gguf(source_dir: Path, output: Path, matrix_quant: str,
                force: bool) -> dict:
    config = _read_json(source_dir / "config.json")
    index = _read_json(source_dir / "model.safetensors.index.json")

    def reader(shard: str) -> tuple[int, dict, int]:
        path = source_dir / shard
        data_start, header = read_safetensors_header(path)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ExportError(f"cannot stat safetensors shard {path}: {exc}") from exc
        return data_start, header, size

    source_plan = validate_source(config, index, reader)
    by_name = {item.spec.name: item for item in source_plan}
    header, extents = _gguf_header(matrix_quant)
    if output.exists() and not force:
        raise ExportError(f"output exists (pass --force to replace): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    handles: dict[str, BinaryIO] = {}
    digest = hashlib.sha256()
    try:
        with temp.open("xb") as dst:
            dst.write(header)
            digest.update(header)
            data_written = 0
            for tensor, (begin, end) in zip(MTP_GGUF_TENSORS, extents):
                padding = begin - data_written
                if padding < 0:
                    raise ExportError("internal GGUF tensor offset overlap")
                if padding:
                    zeros = b"\0" * padding
                    dst.write(zeros)
                    digest.update(zeros)
                source = by_name[tensor.source_name]
                count = end - begin
                if (tensor.source_skip < 0 or count <= 0 or
                        tensor.source_skip + count > source.bytes):
                    raise ExportError(f"invalid GGUF source slice for {tensor.name}")
                src = handles.get(source.shard)
                if src is None:
                    src = (source_dir / source.shard).open("rb")
                    handles[source.shard] = src
                src.seek(source.file_offset + tensor.source_skip)
                remaining = count
                while remaining:
                    block = src.read(min(remaining, COPY_CHUNK))
                    if not block:
                        raise ExportError(f"source ended while writing {tensor.name}")
                    if tensor.add_one:
                        block = _bf16_add_one(block)
                    dst.write(block)
                    digest.update(block)
                    remaining -= len(block)
                data_written = end
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temp, output)
    except Exception:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        for handle in handles.values():
            handle.close()
    return {
        "schema": "ember.qwen4exp.mtp-gguf-export.v1",
        "source_model": "Qwen/Qwen3.8-Flash-Next",
        "source_revision": SOURCE_REVISION,
        "output": str(output),
        "tensor_count": len(MTP_GGUF_TENSORS),
        "artifact_bytes": output.stat().st_size,
        "sha256": digest.hexdigest(),
        "matrix_quant_contract": matrix_quant,
        "normalization": "zero-centered BF16 gamma baked to 1+w",
        "runtime_status": "companion source; quantize before ROCm serving",
    }


def _copy_extent(src: BinaryIO, dst: BinaryIO, offset: int, count: int,
                 digest: "hashlib._Hash") -> None:
    src.seek(offset)
    remaining = count
    while remaining:
        block = src.read(min(remaining, COPY_CHUNK))
        if not block:
            raise ExportError("source shard ended during MTP tensor copy")
        dst.write(block)
        digest.update(block)
        remaining -= len(block)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(COPY_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _quantizer_command(quantizer: Path, source: Path, output: Path,
                       matrix_quant: str, threads: int) -> list[str]:
    if matrix_quant not in SUPPORTED_MATRIX_QUANTS:
        raise ExportError(
            "quantized companion matrix contract must be "
            "Q4_0_ROCMI4 or Q4_0_ROCMFP4_FAST"
        )
    command = [str(quantizer)]
    for name in MTP_QUANTIZED_MATRIX_NAMES:
        command.extend((
            "--tensor-type",
            f"^{re.escape(name)}$={matrix_quant}",
        ))
    # The audited quantizer's positional default remains ROCMI4. Exact FAST
    # selection is expressed by the exhaustive overrides above; routers and
    # vector/norm tensors deliberately remain BF16.
    command.extend((str(source), str(output), "Q4_0_ROCMI4", str(threads)))
    return command


def _quantizer_build_evidence(quantizer: Path) -> dict:
    try:
        completed = subprocess.run(
            [str(quantizer), "--build-info-json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        build_info = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot verify companion quantizer build: {exc}") from exc
    expected_formats = [
        "Q4_0_ROCMI4", "Q6_K", "Q4_0_ROCMFP4_FAST", "Q3_0_ROCMFPX",
    ]
    revision = re.compile(r"[0-9a-f]{40}")
    ember_revision = build_info.get("ember_revision") if isinstance(
        build_info, dict
    ) else None
    rocmfpx_revision = build_info.get("rocmfpx_revision") if isinstance(
        build_info, dict
    ) else None
    if (not isinstance(build_info, dict) or
            build_info.get("tool") != "ember-gguf-quantize" or
            build_info.get("format") != "Q4_0_ROCMI4" or
            build_info.get("ggml_tensor_type") != 108 or
            build_info.get("per_tensor_formats") != expected_formats or
            not isinstance(ember_revision, str) or
            revision.fullmatch(ember_revision) is None or
            not isinstance(rocmfpx_revision, str) or
            revision.fullmatch(rocmfpx_revision) is None):
        raise ExportError("companion quantizer build-info contract mismatch")
    return {
        "quantizer_binary": str(quantizer.resolve()),
        "quantizer_sha256": _sha256_file(quantizer),
        "quantizer_build_info": build_info,
    }


def export(source_dir: Path, output: Path, matrix_quant: str,
           force: bool) -> dict:
    config = _read_json(source_dir / "config.json")
    index = _read_json(source_dir / "model.safetensors.index.json")

    def reader(shard: str) -> tuple[int, dict, int]:
        path = source_dir / shard
        data_start, header = read_safetensors_header(path)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ExportError(f"cannot stat safetensors shard {path}: {exc}") from exc
        return data_start, header, size

    plan = validate_source(config, index, reader)
    output_header, _ = _output_header(matrix_quant)
    total_tensor_bytes = sum(item.bytes for item in plan)
    manifest = {
        "schema": "ember.qwen4exp.mtp-export.v1",
        "source_model": "Qwen/Qwen3.8-Flash-Next",
        "source_revision": SOURCE_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "halospeckv_replay_revision": HALOSPECKV_REVISION,
        "output": str(output),
        "tensor_count": len(plan),
        "tensor_bytes": total_tensor_bytes,
        "matrix_quant_contract": matrix_quant,
        "shared_main_weights": ["token_embd.weight", "output.weight"],
        "quant_classes": {spec.name: spec.quant_class for spec in MTP_TENSORS},
        "runtime_status": "lossless companion source; convert to GGUF before ROCm serving",
    }
    if output.exists() and not force:
        raise ExportError(f"output exists (pass --force to replace): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    handles: dict[str, BinaryIO] = {}
    digest = hashlib.sha256()
    try:
        with temp.open("xb") as dst:
            dst.write(output_header)
            digest.update(output_header)
            for item in plan:
                src = handles.get(item.shard)
                if src is None:
                    src = (source_dir / item.shard).open("rb")
                    handles[item.shard] = src
                _copy_extent(src, dst, item.file_offset, item.bytes, digest)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temp, output)
    except Exception:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        for handle in handles.values():
            handle.close()
    manifest["artifact_bytes"] = output.stat().st_size
    manifest["sha256"] = digest.hexdigest()
    return manifest


def _write_manifest(path: Path, manifest: dict, force: bool) -> None:
    if path.exists() and not force:
        raise ExportError(f"manifest exists (pass --force to replace): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except Exception:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise


def _contract() -> dict:
    return {
        "schema": "ember.qwen4exp.mtp-source-contract.v1",
        "source_revision": SOURCE_REVISION,
        "tensor_count": len(MTP_TENSORS),
        "tensor_bytes": sum(spec.bytes for spec in MTP_TENSORS),
        "tensors": [dataclasses.asdict(spec) for spec in MTP_TENSORS],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--format", choices=("safetensors", "gguf"),
                        default="safetensors")
    parser.add_argument("--matrix-quant", default="inherit-main",
                        choices=("inherit-main", "Q4_0_ROCMI4", "Q4_0_ROCMFP4_FAST"))
    parser.add_argument("--quantizer", type=Path,
                        help="optionally quantize a GGUF export with ember-gguf-quantize")
    parser.add_argument("--quantized-output", type=Path)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-contract", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.print_contract:
            print(json.dumps(_contract(), indent=2, sort_keys=True))
            return 0
        if args.source_dir is None or args.output is None:
            parser.error("--source-dir and --output are required unless --print-contract is used")
        if args.quantizer and (args.format != "gguf" or
                               args.quantized_output is None):
            parser.error("--quantizer requires --format gguf and --quantized-output")
        if args.quantized_output and not args.quantizer:
            parser.error("--quantized-output requires --quantizer")
        if args.threads < 1 or args.threads > 1024:
            parser.error("--threads must be in [1, 1024]")
        if args.format == "gguf":
            manifest = export_gguf(args.source_dir, args.output,
                                   args.matrix_quant, args.force)
        else:
            manifest = export(args.source_dir, args.output, args.matrix_quant,
                              args.force)
        if args.quantizer:
            quantized = args.quantized_output
            assert quantized is not None
            if quantized.exists() and not args.force:
                raise ExportError(
                    f"quantized output exists (pass --force to replace): {quantized}"
                )
            evidence = _quantizer_build_evidence(args.quantizer)
            command = _quantizer_command(
                args.quantizer, args.output, quantized,
                args.matrix_quant, args.threads,
            )
            try:
                subprocess.run(command, check=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                raise ExportError(f"MTP companion quantization failed: {exc}") from exc
            manifest["quantized_output"] = str(quantized)
            manifest["quantized_bytes"] = quantized.stat().st_size
            manifest["quantized_sha256"] = _sha256_file(quantized)
            manifest["quantized_matrix_contract"] = args.matrix_quant
            manifest["quantized_matrix_ggml_type"] = (
                SUPPORTED_MATRIX_QUANTS[args.matrix_quant]
            )
            manifest["quantized_matrix_tensor_count"] = len(
                MTP_QUANTIZED_MATRIX_NAMES
            )
            manifest["quantized_matrix_tensors"] = list(
                MTP_QUANTIZED_MATRIX_NAMES
            )
            manifest["quantize_command"] = command
            manifest.update(evidence)
            manifest["runtime_status"] = "loadable quantized companion"
        if args.manifest_out:
            _write_manifest(args.manifest_out, manifest, args.force)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except ExportError as exc:
        print(f"qwen_mtp_export: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
