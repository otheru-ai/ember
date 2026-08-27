#!/usr/bin/env python3
"""GPU-free contract tests for the Qwen4Exp MTP streaming exporter."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "qwen_mtp_export", ROOT / "scripts" / "qwen_mtp_export.py"
)
assert MODULE_SPEC and MODULE_SPEC.loader
mtp = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mtp
MODULE_SPEC.loader.exec_module(mtp)


def valid_config() -> dict:
    return {
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": 48,
            "hidden_size": 2560,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "mtp": {
                "num_hidden_layers": 1,
                "hybrid": True,
                "layer_types": ["full_attention"],
            },
        }
    }


def valid_source() -> tuple[dict, dict, int]:
    offset = 0
    header: dict[str, object] = {}
    weight_map: dict[str, str] = {}
    for tensor in mtp.MTP_TENSORS:
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.bytes],
        }
        weight_map[tensor.name] = "model-00001-of-00001.safetensors"
        offset += tensor.bytes
    return {"weight_map": weight_map}, header, offset


class MtpExportTests(unittest.TestCase):
    def test_exact_pinned_contract(self) -> None:
        index, header, tensor_bytes = valid_source()

        def reader(_: str) -> tuple[int, dict, int]:
            return 4096, header, 4096 + tensor_bytes

        plan = mtp.validate_source(valid_config(), index, reader)
        self.assertEqual(len(plan), 31)
        self.assertEqual(sum(item.bytes for item in plan), 5_214_301_696)
        self.assertEqual(plan[0].file_offset, 4096)
        self.assertEqual(plan[-1].spec.name, "mtp.fc_hidden.weight")

    def test_wrong_shape_fails_closed(self) -> None:
        index, header, tensor_bytes = valid_source()
        header["mtp.fc_hidden.weight"]["shape"] = [2560, 2559]

        def reader(_: str) -> tuple[int, dict, int]:
            return 8, header, 8 + tensor_bytes

        with self.assertRaisesRegex(mtp.ExportError, "wrong dtype/shape"):
            mtp.validate_source(valid_config(), index, reader)

    def test_extra_mtp_tensor_is_not_silently_dropped(self) -> None:
        index, header, tensor_bytes = valid_source()
        index["weight_map"]["mtp.future.weight"] = (
            "model-00001-of-00001.safetensors"
        )

        def reader(_: str) -> tuple[int, dict, int]:
            return 8, header, 8 + tensor_bytes

        with self.assertRaisesRegex(mtp.ExportError, "tensor set mismatch"):
            mtp.validate_source(valid_config(), index, reader)

    def test_truncated_extent_is_rejected(self) -> None:
        index, header, tensor_bytes = valid_source()

        def reader(_: str) -> tuple[int, dict, int]:
            return 8, header, 8 + tensor_bytes - 1

        with self.assertRaisesRegex(mtp.ExportError, "source byte extent"):
            mtp.validate_source(valid_config(), index, reader)

    def test_config_requires_one_full_attention_head(self) -> None:
        config = valid_config()
        config["text_config"]["mtp"]["layer_types"] = ["linear_attention"]
        index, header, tensor_bytes = valid_source()

        def reader(_: str) -> tuple[int, dict, int]:
            return 8, header, 8 + tensor_bytes

        with self.assertRaisesRegex(mtp.ExportError, "layer_types"):
            mtp.validate_source(config, index, reader)

    def test_output_header_is_valid_and_contiguous(self) -> None:
        raw, extents = mtp._output_header("Q4_0_ROCMI4")
        header_len = struct.unpack("<Q", raw[:8])[0]
        self.assertEqual(header_len % 8, 0)
        header = json.loads(raw[8 : 8 + header_len])
        self.assertEqual(
            header["__metadata__"]["matrix_quant_contract"],
            "Q4_0_ROCMI4",
        )
        self.assertEqual(extents[0][0], 0)
        self.assertEqual(extents[-1][1], 5_214_301_696)
        for left, right in zip(extents, extents[1:]):
            self.assertEqual(left[1], right[0])

    def test_gguf_header_contract_and_tensor_offsets(self) -> None:
        raw, extents = mtp._gguf_header("Q4_0_ROCMI4")
        self.assertEqual(raw[:4], b"GGUF")
        version, tensor_count, metadata_count = struct.unpack_from("<IQQ", raw, 4)
        self.assertEqual(version, 3)
        self.assertEqual(tensor_count, 32)
        self.assertEqual(metadata_count, 21)
        self.assertEqual(len({tensor.name for tensor in mtp.MTP_GGUF_TENSORS}), 32)
        self.assertTrue(all(len(tensor.name.encode()) < 64
                            for tensor in mtp.MTP_GGUF_TENSORS))
        self.assertTrue(all(begin % 32 == 0 for begin, _ in extents))
        self.assertEqual(extents[18][1], extents[19][0])
        self.assertEqual(
            extents[18][1] - extents[18][0], 512 * 2560 * 2
        )
        self.assertEqual(
            extents[19][1] - extents[19][0], 128 * 2560 * 2
        )

    def test_zero_centered_norm_is_baked_to_one_plus_weight(self) -> None:
        values = (0.0, -0.5, 1.0, 2.0)
        source = b"".join(
            struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16)
            for value in values
        )
        baked = mtp._bf16_add_one(source)
        decoded = [
            struct.unpack("<f", struct.pack("<I", value << 16))[0]
            for (value,) in struct.iter_unpack("<H", baked)
        ]
        self.assertEqual(decoded, [1.0, 0.5, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
