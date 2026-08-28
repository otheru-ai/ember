#!/usr/bin/env python3
"""Expand and validate Ember's exact Qwen3.8 vision-mmproj tensor contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "engine" / "dflash" / "qwen4exp" / "qwen4exp_vision_inventory.json"
)
CPP_INCLUDE_PATH = (
    ROOT / "engine" / "dflash" / "qwen4exp" / "qwen4exp_vision_inventory.inc"
)
SCHEMA = "ember.qwen3.8.vision-mmproj-inventory-contract.v1"


class VisionInventoryError(ValueError):
    pass


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def _shape(value: Any, label: str) -> list[int]:
    if (not isinstance(value, list) or not 1 <= len(value) <= 4
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
                   for item in value)):
        raise VisionInventoryError(f"{label} has an invalid GGML logical shape")
    return list(value)


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisionInventoryError(f"cannot load vision inventory contract: {exc}") from exc
    expected_keys = {
        "schema", "provenance", "constants", "fixed_before_layers", "layer",
        "fixed_after_layers", "tensor_count", "tensor_inventory_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys or raw.get("schema") != SCHEMA:
        raise VisionInventoryError("vision inventory descriptor schema differs")
    constants = raw.get("constants")
    expected_constants = {
        "depth": 27, "hidden_size": 1152, "intermediate_size": 4304,
        "spatial_merge_size": 2, "output_hidden_size": 2560,
        "position_embeddings": 2304,
    }
    if constants != expected_constants:
        raise VisionInventoryError("vision inventory descriptor constants differ")
    expected_provenance = {
        "checkpoint_revision": "f5d08274bafd880402bd16f5e3e6c514136ec06c",
        "llama_cpp_pr_27742_head": "035e22731a7fd70b9854b3a2d64ec68e9b1a45d3",
        "dimension_order": "GGML logical order",
    }
    if raw.get("provenance") != expected_provenance:
        raise VisionInventoryError("vision inventory descriptor provenance differs")
    layer = raw.get("layer")
    if (not isinstance(layer, dict) or set(layer) != {"prefix", "count", "tensors"}
            or layer.get("prefix") != "v.blk.{layer}."
            or layer.get("count") != constants["depth"]
            or not isinstance(layer.get("tensors"), list)):
        raise VisionInventoryError("vision inventory layer template differs")

    rows: list[dict[str, Any]] = []
    for section in ("fixed_before_layers", "fixed_after_layers"):
        values = raw.get(section)
        if not isinstance(values, list):
            raise VisionInventoryError(f"vision inventory {section} must be an array")
        for index, row in enumerate(values):
            if not isinstance(row, dict) or set(row) != {"name", "shape"}:
                raise VisionInventoryError(f"vision inventory {section}[{index}] differs")
            rows.append({"name": row.get("name"),
                         "shape": _shape(row.get("shape"), f"{section}[{index}]")})
        if section == "fixed_before_layers":
            for layer_index in range(layer["count"]):
                for tensor_index, tensor in enumerate(layer["tensors"]):
                    if not isinstance(tensor, dict) or set(tensor) != {"suffix", "shape"}:
                        raise VisionInventoryError(
                            f"vision inventory layer tensor {tensor_index} differs")
                    rows.append({
                        "name": layer["prefix"].format(layer=layer_index) +
                                str(tensor.get("suffix", "")),
                        "shape": _shape(tensor.get("shape"),
                                        f"layer tensor {tensor_index}"),
                    })
    names = [row["name"] for row in rows]
    if any(not isinstance(name, str) or not name for name in names):
        raise VisionInventoryError("vision inventory contains an invalid tensor name")
    if len(set(names)) != len(names):
        raise VisionInventoryError("vision inventory descriptor contains duplicate names")
    if len(rows) != raw.get("tensor_count") or len(rows) != 334:
        raise VisionInventoryError(
            f"vision inventory count differs: expected 334, got {len(rows)}")
    digest = canonical_sha256(rows)
    if digest != raw.get("tensor_inventory_sha256"):
        raise VisionInventoryError("vision inventory descriptor digest differs")
    return {
        "schema": SCHEMA,
        "tensor_count": len(rows),
        "tensor_inventory_sha256": digest,
        "tensors": rows,
        "constants": constants,
    }


def validate_inventory(tensors: list[dict[str, Any]]) -> dict[str, Any]:
    contract = load_contract()
    if not isinstance(tensors, list):
        raise VisionInventoryError("vision mmproj tensor inventory must be an array")
    if len(tensors) != contract["tensor_count"]:
        raise VisionInventoryError(
            f"vision mmproj tensor count mismatch: expected 334, got {len(tensors)}")
    actual: dict[str, list[int]] = {}
    for index, row in enumerate(tensors):
        if not isinstance(row, dict):
            raise VisionInventoryError(f"vision mmproj tensor {index} is malformed")
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise VisionInventoryError(f"vision mmproj tensor {index} has no name")
        if name in actual:
            raise VisionInventoryError(f"duplicate vision mmproj tensor: {name}")
        actual[name] = _shape(row.get("shape"), f"vision mmproj tensor {name}")
    expected = {row["name"]: row["shape"] for row in contract["tensors"]}
    missing = sorted(set(expected) - set(actual))
    unknown = sorted(set(actual) - set(expected))
    if missing or unknown:
        raise VisionInventoryError(
            f"vision mmproj names differ; missing={missing[:3]} unknown={unknown[:3]}")
    for name, shape in expected.items():
        if actual[name] != shape:
            raise VisionInventoryError(
                f"vision mmproj tensor shape mismatch: {name}: "
                f"expected {shape}, got {actual[name]}")
    return {
        "contract_schema": contract["schema"],
        "tensor_count": contract["tensor_count"],
        "tensor_inventory_sha256": contract["tensor_inventory_sha256"],
    }


def generate_cpp_include(contract: dict[str, Any] | None = None) -> str:
    value = contract or load_contract()
    constants = value["constants"]
    rows = value["tensors"]
    before, layer_count = 4, constants["depth"] * 12
    lines = [
        "// Generated by scripts/qwen_vision_inventory.py from qwen4exp_vision_inventory.json.",
        "// Run the GPU-free contract test after changing either file.",
    ]
    for key in ("depth", "hidden_size", "intermediate_size", "spatial_merge_size",
                "output_hidden_size", "position_embeddings"):
        lines.append(f"static_assert(Qwen4ExpVisionContract::{key} == {constants[key]});")
    lines.append(f"tensors.reserve({value['tensor_count']});")

    def push(name: str, shape: list[int], prefix: bool = False) -> str:
        dims = ", ".join(str(item) for item in shape)
        expression = f'prefix + "{name}"' if prefix else f'"{name}"'
        return f"tensors.push_back(tensor({expression}, {{{dims}}}));"

    for row in rows[:before]:
        lines.append(push(row["name"], row["shape"]))
    lines.extend([
        f"for (uint32_t layer = 0; layer < {constants['depth']}; ++layer) {{",
        '    const std::string prefix = "v.blk." + std::to_string(layer) + ".";',
    ])
    first_layer = rows[before:before + 12]
    for row in first_layer:
        suffix = row["name"].removeprefix("v.blk.0.")
        lines.append("    " + push(suffix, row["shape"], prefix=True))
    lines.append("}")
    for row in rows[before + layer_count:]:
        lines.append(push(row["name"], row["shape"]))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-cpp", type=Path)
    parser.add_argument("--emit-cpp", action="store_true")
    args = parser.parse_args()
    contract = load_contract()
    generated = generate_cpp_include(contract)
    if args.check_cpp is not None:
        try:
            existing = args.check_cpp.read_text(encoding="utf-8")
        except OSError as exc:
            parser.error(str(exc))
        if existing != generated:
            parser.error("generated C++ vision inventory include is stale")
    if args.emit_cpp:
        print(generated, end="")
    if args.check_cpp is None and not args.emit_cpp:
        print(json.dumps({key: contract[key] for key in (
            "schema", "tensor_count", "tensor_inventory_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
