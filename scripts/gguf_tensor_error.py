#!/usr/bin/env python3
"""Sample matched GGUF rows and measure source-to-quant reconstruction error.

The decoder is loaded from the exact ggml shared library used to create the
quant, so custom ROCMFP types do not need to be reimplemented in Python.  The
output TSV is accepted directly by quant_quality_report.py.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import math
import random
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any


U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
FIXED = {
    U8: ("<B", 1),
    I8: ("<b", 1),
    U16: ("<H", 2),
    I16: ("<h", 2),
    U32: ("<I", 4),
    I32: ("<i", 4),
    F32: ("<f", 4),
    BOOL: ("<?", 1),
    U64: ("<Q", 8),
    I64: ("<q", 8),
    F64: ("<d", 8),
}


class Reader:
    def __init__(self, fp: Any):
        self.fp = fp

    def number(self, value_type: int) -> int | float | bool:
        fmt, size = FIXED[value_type]
        raw = self.fp.read(size)
        if len(raw) != size:
            raise EOFError("truncated GGUF header")
        return struct.unpack(fmt, raw)[0]

    def string(self) -> str:
        size = int(self.number(U64))
        raw = self.fp.read(size)
        if len(raw) != size:
            raise EOFError("truncated GGUF string")
        return raw.decode("utf-8", "replace")

    def skip_value(self, value_type: int) -> Any:
        if value_type == STR:
            return self.string()
        if value_type == ARR:
            element_type = int(self.number(U32))
            count = int(self.number(U64))
            if element_type in FIXED:
                self.fp.seek(FIXED[element_type][1] * count, 1)
                return None
            for _ in range(count):
                self.skip_value(element_type)
            return None
        return self.number(value_type)


class GGUF:
    def __init__(self, path: Path):
        self.path = path
        self.fp = path.open("rb")
        reader = Reader(self.fp)
        if self.fp.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF: {path}")
        self.version = int(reader.number(U32))
        tensor_count = int(reader.number(U64))
        metadata_count = int(reader.number(U64))
        metadata = {}
        for _ in range(metadata_count):
            key = reader.string()
            metadata[key] = reader.skip_value(int(reader.number(U32)))
        self.tensors: dict[str, dict[str, Any]] = {}
        for _ in range(tensor_count):
            name = reader.string()
            dimensions = [int(reader.number(U64)) for _ in range(int(reader.number(U32)))]
            self.tensors[name] = {
                "dims": dimensions,
                "type": int(reader.number(U32)),
                "offset": int(reader.number(U64)),
            }
        alignment = int(metadata.get("general.alignment") or 32)
        self.data_start = (self.fp.tell() + alignment - 1) // alignment * alignment

    def close(self) -> None:
        self.fp.close()


TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)


class TypeTraits(ctypes.Structure):
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("block_size", ctypes.c_int64),
        ("block_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ctypes.c_void_p),
        ("from_float_ref", ctypes.c_void_p),
    ]


class Decoder:
    def __init__(self, library: Path):
        self.library = ctypes.CDLL(str(library))
        self.library.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self.library.ggml_get_type_traits.restype = ctypes.POINTER(TypeTraits)

    def traits(self, value_type: int) -> TypeTraits:
        pointer = self.library.ggml_get_type_traits(value_type)
        if not pointer:
            raise ValueError(f"ggml has no traits for type {value_type}")
        return pointer.contents

    def row(self, model: GGUF, tensor_name: str, row_index: int) -> tuple[list[float], str]:
        info = model.tensors[tensor_name]
        width = int(info["dims"][0])
        traits = self.traits(int(info["type"]))
        if width % traits.block_size:
            raise ValueError(
                f"{tensor_name}: row width {width} is not divisible by block {traits.block_size}"
            )
        row_bytes = width // traits.block_size * traits.type_size
        model.fp.seek(model.data_start + int(info["offset"]) + row_index * row_bytes)
        raw = model.fp.read(row_bytes)
        if len(raw) != row_bytes:
            raise EOFError(f"{tensor_name}: row {row_index} is truncated")
        name = traits.type_name.decode() if traits.type_name else str(info["type"])
        if traits.to_float:
            packed = ctypes.create_string_buffer(raw)
            values = (ctypes.c_float * width)()
            TO_FLOAT(traits.to_float)(packed, values, width)
            return list(values), name
        return decode_plain_row(name, raw, width), name


def decode_plain_row(name: str, raw: bytes, width: int) -> list[float]:
    formats = {
        "f32": "f",
        "f16": "e",
        "f64": "d",
        "i8": "b",
        "i16": "h",
        "i32": "i",
        "i64": "q",
    }
    if name == "bf16":
        words = struct.unpack(f"<{width}H", raw)
        return [struct.unpack("<f", struct.pack("<I", word << 16))[0] for word in words]
    if name not in formats:
        raise ValueError(f"type {name} has no to_float implementation")
    return [float(value) for value in struct.unpack(f"<{width}{formats[name]}", raw)]


def load_imatrix(path: Path | None) -> dict[str, tuple[float, ...]]:
    if path is None:
        return {}
    result = {}
    with path.open("rb") as fp:
        raw = fp.read(4)
        if len(raw) != 4:
            raise EOFError("truncated importance matrix")
        entries = struct.unpack("<i", raw)[0]
        for _ in range(entries):
            name_size = struct.unpack("<i", fp.read(4))[0]
            name = fp.read(name_size).decode("utf-8", "replace")
            _calls, count = struct.unpack("<ii", fp.read(8))
            raw_values = fp.read(count * 4)
            if len(raw_values) != count * 4:
                raise EOFError(f"truncated importance matrix entry {name}")
            result[name] = struct.unpack(f"<{count}f", raw_values)
    return result


def tensor_class(name: str) -> str:
    if "ffn_gate_exps" in name:
        return "routed_gate"
    if "ffn_up_exps" in name:
        return "routed_up"
    if "ffn_down_exps" in name:
        return "routed_down"
    if "shexp" in name:
        return "shared_expert"
    if "indexer" in name:
        return "indexer"
    if "compressor" in name or "kv_comp" in name:
        return "compressor"
    if "hc_" in name or "_hc_" in name:
        return "hyper_connection"
    if name == "token_embd.weight":
        return "embedding"
    if name == "output.weight":
        return "output"
    if "attn_" in name:
        return "attention"
    if "norm" in name:
        return "normalization"
    if "exp_probs" in name or "router" in name:
        return "router"
    return "other"


class Metrics:
    def __init__(self) -> None:
        self.values = 0
        self.reference_squared = 0.0
        self.candidate_squared = 0.0
        self.error_squared = 0.0
        self.dot = 0.0
        self.weighted_reference_squared = 0.0
        self.weighted_error_squared = 0.0
        self.weighted_values = 0

    def add(
        self, reference: list[float], candidate: list[float], weights: tuple[float, ...] | None
    ) -> None:
        for index, (source, quant) in enumerate(zip(reference, candidate)):
            if not math.isfinite(source) or not math.isfinite(quant):
                continue
            error = quant - source
            self.values += 1
            self.reference_squared += source * source
            self.candidate_squared += quant * quant
            self.error_squared += error * error
            self.dot += source * quant
            if weights is not None:
                weight = weights[index]
                if math.isfinite(weight) and weight >= 0:
                    self.weighted_values += 1
                    self.weighted_reference_squared += weight * source * source
                    self.weighted_error_squared += weight * error * error

    def result(self) -> tuple[float, float | None, float]:
        nrmse = math.sqrt(self.error_squared / self.reference_squared)
        weighted = (
            math.sqrt(self.weighted_error_squared / self.weighted_reference_squared)
            if self.weighted_values and self.weighted_reference_squared
            else None
        )
        cosine = self.dot / math.sqrt(self.reference_squared * self.candidate_squared)
        return nrmse, weighted, cosine


def select_tensors(
    source: GGUF,
    candidate: GGUF,
    pattern: re.Pattern[str] | None,
    per_class: int,
    seed: int,
) -> list[str]:
    shared = []
    for name in sorted(set(source.tensors) & set(candidate.tensors)):
        if source.tensors[name]["dims"] != candidate.tensors[name]["dims"]:
            continue
        if len(source.tensors[name]["dims"]) < 2:
            continue
        if pattern and not pattern.search(name):
            continue
        shared.append(name)
    if per_class <= 0:
        return shared
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in shared:
        grouped[tensor_class(name)].append(name)
    selected = []
    rng = random.Random(seed)
    for name_class in sorted(grouped):
        names = grouped[name_class]
        if len(names) <= per_class:
            selected.extend(names)
        else:
            selected.extend(sorted(rng.sample(names, per_class)))
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True, help="libggml shared library with custom quant decoders")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--imatrix", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows-per-tensor", type=int, default=32)
    parser.add_argument("--tensors-per-class", type=int, default=4, help="0 samples every matched tensor")
    parser.add_argument("--tensor-regex")
    parser.add_argument("--seed", type=int, default=7301)
    args = parser.parse_args()
    for path in (args.library, args.reference, args.candidate, args.imatrix):
        if path is not None and not path.is_file():
            raise SystemExit(f"input does not exist: {path}")
    if args.rows_per_tensor < 1 or args.tensors_per_class < 0:
        raise SystemExit("row and tensor sample counts must be positive")

    source, candidate = GGUF(args.reference), GGUF(args.candidate)
    try:
        decoder = Decoder(args.library)
        imatrix = load_imatrix(args.imatrix)
        pattern = re.compile(args.tensor_regex) if args.tensor_regex else None
        tensors = select_tensors(
            source, candidate, pattern, args.tensors_per_class, args.seed
        )
        if not tensors:
            raise SystemExit("no compatible tensors selected")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "tensor",
            "class",
            "source_type",
            "candidate_type",
            "rows_sampled",
            "values",
            "nrmse",
            "weighted_nrmse",
            "cosine",
        ]
        with args.out.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for tensor_index, name in enumerate(tensors):
                info = source.tensors[name]
                row_count = math.prod(info["dims"][1:])
                sample_count = min(args.rows_per_tensor, row_count)
                rng = random.Random(args.seed + tensor_index)
                rows = sorted(rng.sample(range(row_count), sample_count))
                metrics = Metrics()
                source_type = candidate_type = ""
                importance = imatrix.get(name)
                input_width = int(info["dims"][0])
                output_rows = int(info["dims"][1])
                experts = int(info["dims"][2]) if len(info["dims"]) >= 3 else 1
                expected_importance = input_width * experts
                if importance is not None and len(importance) != expected_importance:
                    importance = None
                for row_index in rows:
                    reference, source_type = decoder.row(source, name, row_index)
                    quant, candidate_type = decoder.row(candidate, name, row_index)
                    row_weights = None
                    if importance is not None:
                        expert = row_index // output_rows
                        start = expert * input_width
                        row_weights = importance[start : start + input_width]
                    metrics.add(reference, quant, row_weights)
                nrmse, weighted_nrmse, cosine = metrics.result()
                writer.writerow(
                    {
                        "tensor": name,
                        "class": tensor_class(name),
                        "source_type": source_type,
                        "candidate_type": candidate_type,
                        "rows_sampled": sample_count,
                        "values": metrics.values,
                        "nrmse": f"{nrmse:.9f}",
                        "weighted_nrmse": (
                            f"{weighted_nrmse:.9f}" if weighted_nrmse is not None else ""
                        ),
                        "cosine": f"{cosine:.9f}",
                    }
                )
                fp.flush()
                print(
                    f"[{tensor_index + 1}/{len(tensors)}] {name}: "
                    f"nrmse={nrmse:.5f} weighted={weighted_nrmse}",
                    flush=True,
                )
    finally:
        source.close()
        candidate.close()
    print(f"tensor error TSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
