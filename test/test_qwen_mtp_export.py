#!/usr/bin/env python3
"""GPU-free contract tests for the Qwen4Exp MTP streaming exporter."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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

    def test_rocmfp4_fast_quantizer_command_is_exhaustive(self) -> None:
        command = mtp._quantizer_command(
            Path("/tool/quantizer"), Path("source.gguf"),
            Path("fast.gguf"), "Q4_0_ROCMFP4_FAST", 7,
        )
        overrides = command[1:-4]
        self.assertEqual(len(mtp.MTP_QUANTIZED_MATRIX_NAMES), 21)
        self.assertEqual(len(overrides), 42)
        self.assertEqual(command[-4:], [
            "source.gguf", "fast.gguf", "Q4_0_ROCMI4", "7",
        ])
        emitted = {
            overrides[index + 1]
            for index in range(0, len(overrides), 2)
        }
        expected = {
            f"^{mtp.re.escape(name)}$=Q4_0_ROCMFP4_FAST"
            for name in mtp.MTP_QUANTIZED_MATRIX_NAMES
        }
        self.assertEqual(emitted, expected)
        self.assertTrue(all(
            overrides[index] == "--tensor-type"
            for index in range(0, len(overrides), 2)
        ))

    def test_quantizer_command_rejects_uncontrolled_contract(self) -> None:
        with self.assertRaisesRegex(
            mtp.ExportError, "Q4_0_ROCMI4 or Q4_0_ROCMFP4",
        ):
            mtp._quantizer_command(
                Path("quantizer"), Path("source"), Path("output"),
                "inherit-main", 1,
            )

    def test_quantizer_build_evidence_is_exact(self) -> None:
        build_info = {
            "tool": "ember-gguf-quantize",
            "ember_revision": "a" * 40,
            "rocmfpx_revision": "b" * 40,
            "format": "Q4_0_ROCMI4",
            "ggml_tensor_type": 108,
            "per_tensor_formats": [
                "Q4_0_ROCMI4", "Q6_K", "Q4_0_ROCMFP4_FAST", "Q3_0_ROCMFPX",
            ],
            "intervention_manifest_schema": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            quantizer = Path(directory) / "ember-gguf-quantize"
            quantizer.write_bytes(b"pinned quantizer binary")
            completed = mock.Mock(
                stdout=json.dumps(build_info), stderr="", returncode=0,
            )
            with mock.patch.object(
                mtp.subprocess, "run", return_value=completed,
            ) as run:
                evidence = mtp._quantizer_build_evidence(quantizer)
            run.assert_called_once_with(
                [str(quantizer), "--build-info-json"], check=True,
                stdout=mtp.subprocess.PIPE, stderr=mtp.subprocess.PIPE,
                text=True,
            )
            self.assertEqual(evidence["quantizer_build_info"], build_info)
            self.assertEqual(evidence["quantizer_binary"], str(quantizer.resolve()))
            self.assertEqual(len(evidence["quantizer_sha256"]), 64)

    def test_quantizer_build_evidence_rejects_unpinned_revision(self) -> None:
        build_info = {
            "tool": "ember-gguf-quantize",
            "ember_revision": "unknown",
            "rocmfpx_revision": "b" * 40,
            "format": "Q4_0_ROCMI4",
            "ggml_tensor_type": 108,
            "per_tensor_formats": [
                "Q4_0_ROCMI4", "Q6_K", "Q4_0_ROCMFP4_FAST", "Q3_0_ROCMFPX",
            ],
        }
        completed = mock.Mock(
            stdout=json.dumps(build_info), stderr="", returncode=0,
        )
        with mock.patch.object(mtp.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(mtp.ExportError, "build-info contract"):
                mtp._quantizer_build_evidence(Path("ember-gguf-quantize"))

    def test_fast_gguf_header_carries_loader_contract(self) -> None:
        raw, _ = mtp._gguf_header("Q4_0_ROCMFP4_FAST")
        self.assertIn(b"ember.mtp.matrix_quant_contract", raw)
        self.assertIn(b"Q4_0_ROCMFP4_FAST", raw)

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
