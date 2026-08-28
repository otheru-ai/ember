#!/usr/bin/env python3
"""GPU-free end-to-end tests for ember-gguf-quantize."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
import time
import unittest


GGUF_UINT16 = 2
GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_UINT64 = 10

TYPE_F32 = 0
TYPE_F16 = 1
TYPE_BF16 = 30
TYPE_Q6_K = 14
TYPE_ROCMFP4_FAST = 101
TYPE_ROCMFPX_FP3 = 104
TYPE_ROCMI4 = 108

if len(sys.argv) != 2:
    raise RuntimeError("expected ember-gguf-quantize path")
TOOL = pathlib.Path(sys.argv.pop()).resolve()


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def encode_value(kind: int, value: object) -> bytes:
    if kind == GGUF_UINT16:
        return struct.pack("<H", int(value))
    if kind == GGUF_UINT32:
        return struct.pack("<I", int(value))
    if kind == GGUF_INT32:
        return struct.pack("<i", int(value))
    if kind == GGUF_STRING:
        return gguf_string(str(value))
    if kind == GGUF_ARRAY:
        subtype, values = value
        assert subtype == GGUF_UINT64
        return struct.pack("<IQ", subtype, len(values)) + struct.pack(
            "<" + "Q" * len(values), *values
        )
    raise AssertionError(kind)


def type_size(tensor_type: int) -> int:
    return {TYPE_F32: 4, TYPE_F16: 2, TYPE_BF16: 2}[tensor_type]


TENSORS = [
    ("blk.0.attn_q.weight", TYPE_F32, [32, 2]),
    ("per_layer_token_embd.weight", TYPE_F16, [32, 3]),
    ("blk.0.ffn_down_exps.weight", TYPE_BF16, [32, 2, 2]),
    ("blk.0.attn_norm.weight", TYPE_F32, [32, 2]),
    ("output.weight", TYPE_F16, [32, 2]),
]


def write_fixture(
    path: pathlib.Path,
    *,
    split_no: int | None = None,
    split_count: int | None = None,
    split_tensor_count: int | None = None,
    tensors: list[tuple[str, int, list[int]]] | None = None,
) -> None:
    metadata: list[tuple[str, int, object]] = [
        ("general.architecture", GGUF_STRING, "qwen4exp"),
        ("general.file_type", GGUF_UINT32, 1),
        ("test.u64", GGUF_ARRAY, (GGUF_UINT64, [23703573157769, 300001275])),
    ]
    if split_count is not None:
        assert split_no is not None and split_tensor_count is not None
        metadata.extend([
            ("split.no", GGUF_UINT16, split_no),
            ("split.count", GGUF_UINT16, split_count),
            ("split.tensors.count", GGUF_INT32, split_tensor_count),
        ])
    tensors = tensors or TENSORS
    offsets: list[int] = []
    data_size = 0
    for _, tensor_type, shape in tensors:
        offsets.append(data_size)
        size = type_size(tensor_type)
        for dimension in shape:
            size *= dimension
        data_size = (data_size + size + 31) & ~31

    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, kind, value in metadata:
        header += gguf_string(key) + struct.pack("<I", kind) + encode_value(kind, value)
    for (name, tensor_type, shape), offset in zip(tensors, offsets):
        header += gguf_string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *shape)
        header += struct.pack("<IQ", tensor_type, offset)
    header += bytes((-len(header)) % 32)

    data = bytearray(data_size)
    for (name, tensor_type, shape), offset in zip(tensors, offsets):
        elements = 1
        for dimension in shape:
            elements *= dimension
        if tensor_type == TYPE_F32:
            raw = struct.pack("<" + "f" * elements, *[(index % 13 - 6) / 8 for index in range(elements)])
        elif tensor_type == TYPE_F16:
            raw = struct.pack("<" + "e" * elements, *[(index % 9 - 4) / 4 for index in range(elements)])
        else:
            # Exact BF16 encodings of 0.0 are sufficient to cover the trait path.
            raw = bytes(elements * 2)
        data[offset : offset + len(raw)] = raw
    path.write_bytes(header + data)


def read_string(data: bytes, cursor: int) -> tuple[str, int]:
    length = struct.unpack_from("<Q", data, cursor)[0]
    cursor += 8
    return data[cursor : cursor + length].decode(), cursor + length


def inspect(path: pathlib.Path) -> dict[str, object]:
    data = path.read_bytes()
    assert data[:4] == b"GGUF"
    _, tensor_count, metadata_count = struct.unpack_from("<IQQ", data, 4)
    cursor = 24
    metadata: dict[str, object] = {}
    for _ in range(metadata_count):
        key, cursor = read_string(data, cursor)
        kind = struct.unpack_from("<I", data, cursor)[0]
        cursor += 4
        if kind == GGUF_UINT16:
            value = struct.unpack_from("<H", data, cursor)[0]
            cursor += 2
        elif kind == GGUF_UINT32:
            value = struct.unpack_from("<I", data, cursor)[0]
            cursor += 4
        elif kind == GGUF_INT32:
            value = struct.unpack_from("<i", data, cursor)[0]
            cursor += 4
        elif kind == GGUF_STRING:
            value, cursor = read_string(data, cursor)
        elif kind == GGUF_ARRAY:
            subtype, count = struct.unpack_from("<IQ", data, cursor)
            cursor += 12
            assert subtype == GGUF_UINT64
            value = list(struct.unpack_from("<" + "Q" * count, data, cursor))
            cursor += 8 * count
        else:
            raise AssertionError(kind)
        metadata[key] = value
    tensors: dict[str, int] = {}
    tensor_info: dict[str, dict[str, object]] = {}
    for _ in range(tensor_count):
        name, cursor = read_string(data, cursor)
        dimensions = struct.unpack_from("<I", data, cursor)[0]
        cursor += 4
        shape = list(struct.unpack_from("<" + "Q" * dimensions, data, cursor))
        cursor += dimensions * 8
        tensor_type, offset = struct.unpack_from("<IQ", data, cursor)
        cursor += 12
        tensors[name] = tensor_type
        tensor_info[name] = {"type": tensor_type, "shape": shape, "offset": offset}
    data_offset = (cursor + 31) & ~31
    return {
        "metadata": metadata, "tensors": tensors, "tensor_info": tensor_info,
        "data_offset": data_offset, "size": len(data),
    }


def ue4m3_scale(value: int) -> float:
    if value > 0x7E:
        return 0.0
    exponent, mantissa = value >> 3, value & 7
    if exponent == 0:
        return mantissa * 2.0**-10
    return (8 + mantissa) * 2.0 ** (exponent - 11)


def decode_4bit_rows(raw: bytes, rows: int, *, rocmfp4_fast: bool) -> list[list[float]]:
    self_rows = []
    for row in range(rows):
        block = raw[row * 17:(row + 1) * 17]
        if len(block) != 17:
            raise AssertionError("truncated 4-bit row")
        scale = ue4m3_scale(block[16])
        values = []
        low = [byte & 15 for byte in block[:16]]
        high = [byte >> 4 for byte in block[:16]]
        for nibble in low + high:
            if rocmfp4_fast:
                magnitude3 = nibble & 7
                magnitude = magnitude3 if magnitude3 <= 4 else 2 * magnitude3 - 4
                decoded = -magnitude if nibble & 8 else magnitude
            else:
                decoded = nibble - 16 if nibble & 8 else nibble
            values.append(decoded * scale)
        self_rows.append(values)
    return self_rows


def tensor_raw(path: pathlib.Path, name: str) -> tuple[dict[str, object], bytes]:
    result = inspect(path)
    info = result["tensor_info"][name]
    shape = info["shape"]
    assert shape[0] == 32
    rows = math.prod(shape[1:])
    start = result["data_offset"] + info["offset"]
    return info, path.read_bytes()[start:start + rows * 17]


def direction_projection_l2(rows: list[list[float]], direction: list[float]) -> float:
    projection = [
        sum(direction[row] * rows[row][column] for row in range(len(rows)))
        for column in range(len(rows[0]))
    ]
    return math.sqrt(sum(value * value for value in projection))


class QuantizerToolTests(unittest.TestCase):
    tool = TOOL

    def command(self, source: pathlib.Path, output: pathlib.Path, *extra: str) -> list[str]:
        return [
            str(self.tool),
            "--tensor-type", r"^per_layer_token_embd\.weight$=Q4_0_ROCMI4",
            *extra,
            str(source), str(output), "Q4_0_ROCMI4", "3",
        ]

    def test_build_info_and_streaming_contract(self) -> None:
        info_result = subprocess.run(
            [self.tool, "--build-info-json"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        info = json.loads(info_result.stdout)
        self.assertEqual(info_result.stderr, "")
        self.assertEqual(info["tool"], "ember-gguf-quantize")
        self.assertRegex(info["ember_revision"], re.compile(r"^[0-9a-f]{40}$"))
        self.assertEqual(info["rocmfpx_revision"], "c49ebdbd5c9f01ec242369f9e7f7967855f80cba")
        self.assertEqual(info["format"], "Q4_0_ROCMI4")
        self.assertEqual(info["ggml_tensor_type"], 108)
        self.assertEqual(info["per_tensor_formats"],
                         ["Q4_0_ROCMI4", "Q6_K", "Q4_0_ROCMFP4_FAST",
                          "Q3_0_ROCMFPX"])
        self.assertEqual(info["intervention_manifest_schema"], 1)

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, output = root / "input.gguf", root / "output.gguf"
            write_fixture(source)
            dry = subprocess.run(
                self.command(
                    source, output, "--dry-size-json",
                    "--device-budget-bytes", str(128 * 1024**3),
                    "--runtime-reserve-bytes", str(32 * 1024**3),
                ),
                check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            size = json.loads(dry.stdout)
            self.assertFalse(output.exists())
            self.assertTrue(size["fits"])
            self.assertEqual(size["shard_count"], 1)
            self.assertEqual(size["shard_bytes"], [size["artifact_bytes"]])
            self.assertEqual(size["budget_bytes"], 128 * 1024**3)
            self.assertEqual(size["runtime_reserve_bytes"], 32 * 1024**3)
            self.assertEqual(size["total_bytes"], size["artifact_bytes"] + 32 * 1024**3)

            subprocess.run(
                self.command(
                    source, output,
                    "--device-budget-bytes", str(128 * 1024**3),
                    "--runtime-reserve-bytes", str(32 * 1024**3),
                ), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            result = inspect(output)
            self.assertEqual(result["size"], size["artifact_bytes"])
            self.assertEqual(result["metadata"]["general.file_type"], 118)
            self.assertEqual(result["metadata"]["general.quantization_version"], 2)
            self.assertEqual(result["metadata"]["ember.intervention.kind"],
                             "none_control")
            self.assertEqual(
                result["metadata"]["ember.intervention.release_eligibility"],
                "control_only_requires_manifest_for_release")
            self.assertEqual(result["metadata"]["test.u64"], [23703573157769, 300001275])
            self.assertEqual(result["tensors"], {
                "blk.0.attn_q.weight": TYPE_ROCMI4,
                "per_layer_token_embd.weight": TYPE_ROCMI4,
                "blk.0.ffn_down_exps.weight": TYPE_ROCMI4,
                "blk.0.attn_norm.weight": TYPE_F32,
                "output.weight": TYPE_F16,
            })

    def test_non_overlapping_mixed_tensor_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, output = root / "input.gguf", root / "mixed.gguf"
            write_fixture(source, tensors=[
                ("blk.0.attn_q.weight", TYPE_F32, [32, 2]),
                ("per_layer_token_embd.weight", TYPE_F16, [32, 3]),
                ("token_embd.weight", TYPE_F16, [256, 2]),
                ("output.weight", TYPE_F16, [256, 2]),
                ("output_hc_down.weight", TYPE_F16, [32, 2]),
                ("blk.0.ffn_down_exps.weight", TYPE_BF16, [32, 2, 2]),
                ("blk.0.ffn_down_shexp.weight", TYPE_BF16, [32, 2]),
            ])
            profile = json.loads((pathlib.Path(__file__).resolve().parents[1] /
                                  "share" / "release_profiles" /
                                  "qwen3.8-flash-next-rocmi4-strix-halo.json").read_text())
            fast_arm = next(
                arm for arm in profile["quantization"]["performance_bakeoff"]["arms"]
                if arm["id"] == "rocmfp4-fast-matrix-q6k-embedding-head")
            fast_options = [value for override in fast_arm["per_tensor_overrides"][1:]
                            for value in ("--tensor-type", override)]
            command = self.command(source, output, *fast_options)
            completed = subprocess.run(command, check=False, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(inspect(output)["tensors"], {
                "blk.0.attn_q.weight": TYPE_ROCMFP4_FAST,
                "per_layer_token_embd.weight": TYPE_ROCMI4,
                "token_embd.weight": TYPE_Q6_K,
                "output.weight": TYPE_Q6_K,
                "output_hc_down.weight": TYPE_ROCMFP4_FAST,
                "blk.0.ffn_down_exps.weight": TYPE_ROCMFP4_FAST,
                "blk.0.ffn_down_shexp.weight": TYPE_ROCMFP4_FAST,
            })

            fp3_output = root / "fp3-ple.gguf"
            fp3_result = subprocess.run(
                [
                    str(self.tool),
                    "--tensor-type",
                    r"^per_layer_token_embd\.weight$=Q3_0_ROCMFPX",
                    str(source), str(fp3_output), "Q4_0_ROCMI4", "3",
                ],
                check=False, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(fp3_result.returncode, 0, fp3_result.stderr)
            self.assertEqual(
                inspect(fp3_output)["tensors"]["per_layer_token_embd.weight"],
                TYPE_ROCMFPX_FP3,
            )

            expert_arm = next(
                arm for arm in profile["quantization"]["performance_bakeoff"]["arms"]
                if arm["id"] ==
                "rocmfp4-fast-routed-experts-q6k-embedding-head")
            expert_options = [
                value for override in expert_arm["per_tensor_overrides"][1:]
                for value in ("--tensor-type", override)
            ]
            expert_output = root / "expert-only.gguf"
            expert_result = subprocess.run(
                self.command(source, expert_output, *expert_options),
                check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(expert_result.returncode, 0, expert_result.stderr)
            self.assertEqual(inspect(expert_output)["tensors"], {
                "blk.0.attn_q.weight": TYPE_ROCMI4,
                "per_layer_token_embd.weight": TYPE_ROCMI4,
                "token_embd.weight": TYPE_Q6_K,
                "output.weight": TYPE_Q6_K,
                "output_hc_down.weight": TYPE_ROCMI4,
                "blk.0.ffn_down_exps.weight": TYPE_ROCMFP4_FAST,
                "blk.0.ffn_down_shexp.weight": TYPE_ROCMI4,
            })

            overlap = self.command(
                source, root / "overlap.gguf",
                "--tensor-type", r"attn_q=Q4_0_ROCMFP4_FAST",
                "--tensor-type", r"^blk\.0\..*=Q4_0_ROCMI4",
            )
            rejected = subprocess.run(overlap, check=False, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("multiple --tensor-type regexes", rejected.stderr)

            before = hashlib.sha256(output.read_bytes()).digest()
            repeated = subprocess.run(
                self.command(source, output), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(hashlib.sha256(output.read_bytes()).digest(), before)

    def test_streaming_directional_ablation_is_applied_and_self_describing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "input.gguf"
            stock = root / "stock.gguf"
            output = root / "heretic.gguf"
            target_name = "blk.3.attn_output.weight"
            write_fixture(source, tensors=[
                TENSORS[1],
                (target_name, TYPE_F32, [32, 2]),
            ])
            # A direction spanning both rows exercises the preserve-row-norm
            # path without asking the ablation to erase an entire rank-one row.
            direction_value = 2**-0.5
            direction_bytes = struct.pack("<ff", direction_value, direction_value)
            target_names_sha = hashlib.sha256(target_name.encode()).hexdigest()
            manifest = root / "intervention.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "kind": "directional_ablation",
                "status": "complete",
                "weight_intervention": True,
                "prompt_only": False,
                "application_stage": "pre_quantization_encoding",
                "source": {"repo_id": "Qwen/Qwen3.8-Flash-Next"},
                "tooling": {"upstream_heretic": {"revision": "b" * 40}},
                "corpora": [{"id": "fixture"}],
                "directions": [{
                    "id": "r1", "dtype": "F32",
                    "values": [direction_value, direction_value],
                    "sha256": hashlib.sha256(direction_bytes).hexdigest(),
                }],
                "targets": [{
                    "tensor_name": target_name,
                    "direction_id": "r1",
                    "scale": 1.0,
                    "normalization": "row_norm_preserve",
                    "expected_shape": [32, 2],
                }],
                "tensor_map": {
                    "kind": "exact_tensor_names",
                    "target_count": 1,
                    "target_names_sha256": target_names_sha,
                },
            }, sort_keys=True), encoding="utf-8")
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            budget = (
                "--device-budget-bytes", str(128 * 1024**3),
                "--runtime-reserve-bytes", str(32 * 1024**3),
            )
            subprocess.run(self.command(source, stock, *budget), check=True,
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            dry = subprocess.run(
                self.command(source, output, "--intervention-manifest", str(manifest),
                             "--dry-size-json", *budget),
                check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            dry_report = json.loads(dry.stdout)
            self.assertTrue(dry_report["intervention_validated"])
            self.assertFalse(dry_report["intervention_applied"])
            self.assertEqual(dry_report["intervention_manifest_sha256"], manifest_sha)
            self.assertEqual(dry_report["intervention_target_names_sha256"], target_names_sha)
            self.assertEqual(dry_report["intervention_targets"], [target_name])

            completed = subprocess.run(
                self.command(source, output, "--intervention-manifest", str(manifest), *budget),
                check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertTrue(report["intervention_applied"])
            self.assertNotEqual(stock.read_bytes(), output.read_bytes())
            # The target is the final 34-byte ROCMI4 tensor followed by its
            # 32-byte alignment padding. Comparing the final aligned extent
            # proves that this is a weight edit, not metadata-only relabeling.
            self.assertNotEqual(stock.read_bytes()[-64:], output.read_bytes()[-64:])
            result = inspect(output)
            self.assertEqual(result["metadata"]["ember.intervention.kind"],
                             "directional_ablation")
            self.assertEqual(result["metadata"]["ember.intervention.application_stage"],
                             "pre_quantization_encoding")
            self.assertEqual(result["metadata"]["ember.intervention.manifest_sha256"],
                             manifest_sha)
            self.assertEqual(result["metadata"]["ember.intervention.target_names_sha256"],
                             target_names_sha)
            self.assertEqual(result["metadata"]["ember.intervention.target_count"], 1)

            bad = json.loads(manifest.read_text(encoding="utf-8"))
            bad["prompt_only"] = True
            bad_manifest = root / "prompt-only.json"
            bad_manifest.write_text(json.dumps(bad), encoding="utf-8")
            rejected = root / "rejected.gguf"
            failure = subprocess.run(
                self.command(source, rejected, "--intervention-manifest", str(bad_manifest),
                             *budget), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("identity/provenance", failure.stderr)
            self.assertFalse(rejected.exists())

            for label, mutate, expected_error in (
                (
                    "wrong-hybrid-projection",
                    lambda value: value["targets"][0].update({
                        "tensor_name": "blk.4.attn_output.weight",
                    }),
                    "hybrid layer map",
                ),
                (
                    "direction-matches-columns",
                    lambda value: value["targets"][0].update({
                        "expected_shape": [2, 32],
                    }),
                    "shape/direction length",
                ),
            ):
                invalid = json.loads(manifest.read_text(encoding="utf-8"))
                mutate(invalid)
                invalid_names = sorted(
                    target["tensor_name"] for target in invalid["targets"]
                )
                invalid["tensor_map"]["target_names_sha256"] = hashlib.sha256(
                    "\n".join(invalid_names).encode()
                ).hexdigest()
                invalid_manifest = root / f"{label}.json"
                invalid_manifest.write_text(json.dumps(invalid), encoding="utf-8")
                invalid_output = root / f"{label}.gguf"
                invalid_run = subprocess.run(
                    self.command(
                        source, invalid_output, "--intervention-manifest",
                        str(invalid_manifest), *budget,
                    ),
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertNotEqual(invalid_run.returncode, 0)
                self.assertIn(expected_error, invalid_run.stderr)
                self.assertFalse(invalid_output.exists())

    def test_intervention_audit_uses_each_destination_type_not_cross_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "input.gguf"
            target_name = "blk.3.attn_output.weight"
            write_fixture(source, tensors=[
                TENSORS[1],
                (target_name, TYPE_F32, [32, 2]),
            ])
            direction_value = 2**-0.5
            direction = [direction_value, direction_value]
            manifest = root / "intervention.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "kind": "directional_ablation",
                "status": "complete",
                "weight_intervention": True,
                "prompt_only": False,
                "application_stage": "pre_quantization_encoding",
                "source": {"repo_id": "Qwen/Qwen3.8-Flash-Next"},
                "tooling": {"upstream_heretic": {"revision": "b" * 40}},
                "corpora": [{"id": "fixture"}],
                "directions": [{
                    "id": "r1", "dtype": "F32", "values": direction,
                    "sha256": hashlib.sha256(struct.pack("<ff", *direction)).hexdigest(),
                }],
                "targets": [{
                    "tensor_name": target_name, "direction_id": "r1", "scale": 0.75,
                    "normalization": "row_norm_preserve", "expected_shape": [32, 2],
                }],
                "tensor_map": {
                    "kind": "exact_tensor_names", "target_count": 1,
                    "target_names_sha256": hashlib.sha256(target_name.encode()).hexdigest(),
                },
            }, sort_keys=True), encoding="utf-8")
            budget = (
                "--device-budget-bytes", str(128 * 1024**3),
                "--runtime-reserve-bytes", str(32 * 1024**3),
            )

            for label, tensor_type, fast, override in (
                ("rocmi4", TYPE_ROCMI4, False, ()),
                (
                    "rocmfp4-fast", TYPE_ROCMFP4_FAST, True,
                    ("--tensor-type", rf"^{re.escape(target_name)}$=Q4_0_ROCMFP4_FAST"),
                ),
            ):
                output = root / f"{label}.gguf"
                completed = subprocess.run(
                    self.command(
                        source, output, "--intervention-manifest", str(manifest),
                        *override, *budget,
                    ),
                    check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                report = json.loads(completed.stdout)
                metric = report["intervention_metrics"][0]
                info, raw = tensor_raw(output, target_name)
                self.assertEqual(info["type"], tensor_type)
                correct_rows = decode_4bit_rows(raw, 2, rocmfp4_fast=fast)
                cross_rows = decode_4bit_rows(raw, 2, rocmfp4_fast=not fast)
                correct_l2 = direction_projection_l2(correct_rows, direction)
                cross_l2 = direction_projection_l2(cross_rows, direction)
                self.assertAlmostEqual(metric["stored_projection_l2"], correct_l2, places=6)
                self.assertGreater(
                    abs(metric["stored_projection_l2"] - cross_l2), 1e-3,
                    f"{label} audit unexpectedly matches the other format's decoder",
                )

            unsupported = root / "q6-intervention.gguf"
            rejected = subprocess.run(
                self.command(
                    source, unsupported, "--intervention-manifest", str(manifest),
                    "--tensor-type", rf"^{re.escape(target_name)}$=Q6_K", *budget,
                ),
                check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(
                "must use Q4_0_ROCMI4 or Q4_0_ROCMFP4_FAST", rejected.stderr
            )
            self.assertFalse(unsupported.exists())

    def test_budget_failure_happens_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, output = root / "input.gguf", root / "output.gguf"
            write_fixture(source)
            too_large = subprocess.run(
                self.command(
                    source, output,
                    "--device-budget-bytes", "1", "--runtime-reserve-bytes", "1",
                ), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertNotEqual(too_large.returncode, 0)
            self.assertIn("exceeds device budget", too_large.stderr)
            self.assertFalse(output.exists())

    def test_multishard_aggregate_preflight_and_transactional_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = root / "model-00001-of-00002.gguf"
            second = root / "model-00002-of-00002.gguf"
            write_fixture(first, split_no=0, split_count=2,
                          split_tensor_count=len(TENSORS), tensors=TENSORS[:2])
            write_fixture(second, split_no=1, split_count=2,
                          split_tensor_count=len(TENSORS), tensors=TENSORS[2:])
            output = root / "quantized.gguf"
            extra = (
                "--keep-split", "--device-budget-bytes", str(128 * 1024**3),
                "--runtime-reserve-bytes", str(32 * 1024**3),
            )
            dry = subprocess.run(
                self.command(first, output, "--dry-size-json", *extra), check=True,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            size = json.loads(dry.stdout)
            self.assertEqual(size["shard_count"], 2)
            self.assertEqual(sum(size["shard_bytes"]), size["artifact_bytes"])
            outputs = [
                root / "quantized-00001-of-00002.gguf",
                root / "quantized-00002-of-00002.gguf",
            ]
            self.assertFalse(any(path.exists() for path in outputs))
            self.assertEqual(list(root.glob("*.partial.*")), [])

            too_small = subprocess.run(
                self.command(
                    first, output, "--keep-split",
                    "--device-budget-bytes", str(size["artifact_bytes"] - 1),
                    "--runtime-reserve-bytes", "0",
                ), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertNotEqual(too_small.returncode, 0)
            self.assertIn("aggregate artifact", too_small.stderr)
            self.assertFalse(any(path.exists() for path in outputs))
            self.assertEqual(list(root.glob("*.partial.*")), [])

            subprocess.run(
                self.command(first, output, *extra), check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            markers = list(root.glob("quantized.gguf.transaction.*.marker"))
            self.assertEqual(len(markers), 1)
            self.assertEqual(markers[0].read_bytes(), b"COMPLETE\n")
            inspected = [inspect(path) for path in outputs]
            self.assertEqual([item["size"] for item in inspected], size["shard_bytes"])
            for index, item in enumerate(inspected):
                self.assertEqual(item["metadata"]["split.no"], index)
                self.assertEqual(item["metadata"]["split.count"], 2)
                self.assertEqual(item["metadata"]["split.tensors.count"], len(TENSORS))
            tensor_types = {
                name: kind for item in inspected for name, kind in item["tensors"].items()
            }
            self.assertEqual(tensor_types, {
                "blk.0.attn_q.weight": TYPE_ROCMI4,
                "per_layer_token_embd.weight": TYPE_ROCMI4,
                "blk.0.ffn_down_exps.weight": TYPE_ROCMI4,
                "blk.0.attn_norm.weight": TYPE_F32,
                "output.weight": TYPE_F16,
            })

    def test_multishard_missing_or_inconsistent_set_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = root / "model-00001-of-00002.gguf"
            output = root / "quantized.gguf"
            write_fixture(first, split_no=0, split_count=2,
                          split_tensor_count=len(TENSORS), tensors=TENSORS[:2])
            missing = subprocess.run(
                self.command(first, output, "--keep-split"), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("cannot open input GGUF shard", missing.stderr)
            self.assertFalse(any(root.glob("quantized-*.gguf")))

            second = root / "model-00002-of-00002.gguf"
            write_fixture(second, split_no=0, split_count=2,
                          split_tensor_count=len(TENSORS), tensors=TENSORS[2:])
            inconsistent = subprocess.run(
                self.command(first, output, "--keep-split"), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertNotEqual(inconsistent.returncode, 0)
            self.assertIn("split.no/count", inconsistent.stderr)
            self.assertFalse(any(root.glob("quantized-*.gguf")))

    def test_multishard_duplicate_or_incomplete_metadata_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = root / "model-00001-of-00002.gguf"
            second = root / "model-00002-of-00002.gguf"
            output = root / "quantized.gguf"
            write_fixture(first, split_no=0, split_count=2,
                          split_tensor_count=2, tensors=[TENSORS[1]])
            write_fixture(second, split_no=1, split_count=2,
                          split_tensor_count=2, tensors=[TENSORS[1]])
            duplicate = subprocess.run(
                self.command(first, output, "--keep-split"), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn("duplicate tensor", duplicate.stderr)
            self.assertFalse(any(root.glob("quantized-*.gguf")))

            write_fixture(second, tensors=[TENSORS[2]])
            incomplete = subprocess.run(
                self.command(first, output, "--keep-split"), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertNotEqual(incomplete.returncode, 0)
            self.assertIn("split.no/count", incomplete.stderr)
            self.assertFalse(any(root.glob("quantized-*.gguf")))

    def test_late_promotion_conflict_retains_owned_state_and_incomplete_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = root / "model-00001-of-00002.gguf"
            second = root / "model-00002-of-00002.gguf"
            large = ("blk.0.large.weight", TYPE_BF16, [32, 1_000_000])
            first_tensors = [TENSORS[1], large]
            second_tensors = [TENSORS[2]]
            total_tensors = len(first_tensors) + len(second_tensors)
            write_fixture(first, split_no=0, split_count=2,
                          split_tensor_count=total_tensors, tensors=first_tensors)
            write_fixture(second, split_no=1, split_count=2,
                          split_tensor_count=total_tensors, tensors=second_tensors)
            output = root / "quantized.gguf"
            first_output = root / "quantized-00001-of-00002.gguf"
            second_output = root / "quantized-00002-of-00002.gguf"
            process = subprocess.Popen(
                self.command(first, output, "--keep-split"), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 10
            while process.poll() is None and time.monotonic() < deadline:
                if len(list(root.glob("quantized-*.partial.*"))) == 2:
                    second_output.write_bytes(b"promotion race sentinel")
                    break
                time.sleep(0.001)
            else:
                process.kill()
                process.communicate()
                self.fail("quantizer did not stage both partial shards before promotion")
            _, stderr = process.communicate(timeout=30)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("without clobbering", stderr)
            self.assertTrue(first_output.is_file())
            self.assertEqual(second_output.read_bytes(), b"promotion race sentinel")
            self.assertEqual(len(list(root.glob("*.partial.*"))), 2)
            markers = list(root.glob("quantized.gguf.transaction.*.marker"))
            self.assertEqual(len(markers), 1)
            self.assertIn(b"incomplete multi-shard publication", markers[0].read_bytes())

    def test_late_conflict_does_not_unlink_replaced_promoted_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            count = 200
            for index in range(count):
                tensor = TENSORS[1] if index == 0 else (
                    f"blk.{index}.attn_q.weight", TYPE_F16, [32, 1]
                )
                write_fixture(
                    root / f"model-{index + 1:05d}-of-{count:05d}.gguf",
                    split_no=index, split_count=count,
                    split_tensor_count=count, tensors=[tensor],
                )
            first = root / f"model-00001-of-{count:05d}.gguf"
            output = root / "quantized.gguf"
            first_output = root / f"quantized-00001-of-{count:05d}.gguf"
            conflict_output = root / f"quantized-{count:05d}-of-{count:05d}.gguf"
            process = subprocess.Popen(
                self.command(first, output, "--keep-split"), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 10
            while process.poll() is None and time.monotonic() < deadline:
                if len(list(root.glob("quantized-*.partial.*"))) == count:
                    conflict_output.write_bytes(b"late conflict sentinel")
                    break
                time.sleep(0.001)
            else:
                process.kill()
                process.communicate()
                self.fail("quantizer did not stage the complete 200-shard set")

            deadline = time.monotonic() + 10
            while process.poll() is None and time.monotonic() < deadline:
                if first_output.exists():
                    first_output.unlink()
                    first_output.write_bytes(b"foreign replacement sentinel")
                    break
                time.sleep(0.0001)
            else:
                process.kill()
                process.communicate()
                self.fail("quantizer did not promote the first shard before late conflict")
            _, stderr = process.communicate(timeout=30)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("without clobbering", stderr)
            self.assertIn("incomplete shard transaction retained marker", stderr)
            self.assertEqual(first_output.read_bytes(), b"foreign replacement sentinel")
            self.assertEqual(conflict_output.read_bytes(), b"late conflict sentinel")
            self.assertEqual(len(list(root.glob("*.partial.*"))), count)
            markers = list(root.glob("quantized.gguf.transaction.*.marker"))
            self.assertEqual(len(markers), 1)
            self.assertIn(b"incomplete multi-shard publication", markers[0].read_bytes())

    def test_multishard_no_clobber_checks_all_destinations_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = root / "model-00001-of-00002.gguf"
            second = root / "model-00002-of-00002.gguf"
            write_fixture(first, split_no=0, split_count=2,
                          split_tensor_count=len(TENSORS), tensors=TENSORS[:2])
            write_fixture(second, split_no=1, split_count=2,
                          split_tensor_count=len(TENSORS), tensors=TENSORS[2:])
            output = root / "quantized.gguf"
            first_output = root / "quantized-00001-of-00002.gguf"
            second_output = root / "quantized-00002-of-00002.gguf"
            second_output.write_bytes(b"belongs to another process")
            failed = subprocess.run(
                self.command(first, output, "--keep-split"), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("output already exists", failed.stderr)
            self.assertFalse(first_output.exists())
            self.assertEqual(second_output.read_bytes(), b"belongs to another process")

    def test_replaced_transaction_marker_is_preserved_and_commit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            count = 200
            for index in range(count):
                tensor = TENSORS[1] if index == 0 else (
                    f"blk.{index}.attn_q.weight", TYPE_F16, [32, 1]
                )
                write_fixture(
                    root / f"model-{index + 1:05d}-of-{count:05d}.gguf",
                    split_no=index, split_count=count,
                    split_tensor_count=count, tensors=[tensor],
                )
            first = root / f"model-00001-of-{count:05d}.gguf"
            output = root / "quantized.gguf"
            process = subprocess.Popen(
                self.command(first, output, "--keep-split"), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 10
            marker = None
            while process.poll() is None and time.monotonic() < deadline:
                markers = list(root.glob("quantized.gguf.transaction.*.marker"))
                if markers:
                    marker = markers[0]
                    marker.unlink()
                    marker.write_bytes(b"foreign marker sentinel")
                    break
                time.sleep(0.0001)
            else:
                process.kill()
                process.communicate()
                self.fail("quantizer did not create its transaction marker")
            _, stderr = process.communicate(timeout=30)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("marker was replaced", stderr)
            assert marker is not None
            self.assertEqual(marker.read_bytes(), b"foreign marker sentinel")
            self.assertEqual(len(list(root.glob("*.partial.*"))), count)


if __name__ == "__main__":
    unittest.main()
