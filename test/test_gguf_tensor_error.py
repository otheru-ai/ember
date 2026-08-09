#!/usr/bin/env python3
"""GPU-free classification checks for the GGUF error sampler."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gguf_tensor_error", ROOT / "scripts" / "gguf_tensor_error.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> int:
    cases = {
        "blk.1.ffn_gate_exps.weight": "routed_gate",
        "blk.1.ffn_up_exps.weight": "routed_up",
        "blk.1.ffn_down_exps.weight": "routed_down",
        "blk.1.ffn_gate_shexp.weight": "shared_expert",
        "blk.1.attn_q_a.weight": "attention",
        "blk.1.indexer_norm.weight": "indexer",
        "blk.1.compressor_k.weight": "compressor",
        "blk.1.hc_attn_fn.weight": "hyper_connection",
        "token_embd.weight": "embedding",
        "output.weight": "output",
        "blk.1.ffn_norm.weight": "normalization",
        "blk.1.exp_probs_b.bias": "router",
    }
    for name, expected in cases.items():
        assert MODULE.tensor_class(name) == expected, (name, MODULE.tensor_class(name))
    assert MODULE.decode_plain_row("f32", struct.pack("<2f", 1.5, -2.0), 2) == [1.5, -2.0]
    assert MODULE.decode_plain_row("f16", struct.pack("<2e", 1.5, -2.0), 2) == [1.5, -2.0]
    assert MODULE.decode_plain_row("bf16", struct.pack("<2H", 0x3FC0, 0xC000), 2) == [1.5, -2.0]
    assert MODULE.decode_plain_row("i32", struct.pack("<2i", 7, -9), 2) == [7.0, -9.0]
    metrics = MODULE.Metrics()
    metrics.add([1.0, -1.0], [0.5, -1.5], (2.0, 1.0))
    nrmse, weighted, cosine = metrics.result()
    assert 0 < nrmse < 1
    assert weighted is not None and 0 < weighted < 1
    assert 0 < cosine < 1
    print("gguf tensor error: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
