#!/usr/bin/env python3
"""GPU-free regression tests for the saved-production-ISA gate."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "check_rocmi4_w4a8_isa.py"


def kernel(
    checked: int,
    modifier_low: str = "neg_lo:[1,0,0]",
    scratch: int = 0,
    shift_before_high: bool = False,
) -> str:
    symbol = (
        "_ZL9mul_mat_qIL9ggml_type108ELi32ELb"
        f"{checked}ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii"
    )
    high_then_shift = """\tv_wmma_i32_16x16x16_iu4 v[0:7], v[8:9], v[10:11], v[0:7] neg_lo:[1,1,0]
\tv_lshlrev_b32_e32 v0, 4, v0"""
    if shift_before_high:
        high_then_shift = """\tv_lshlrev_b32_e32 v0, 4, v0
\tv_wmma_i32_16x16x16_iu4 v[0:7], v[8:9], v[10:11], v[0:7] neg_lo:[1,1,0]"""
    return f"""{symbol}:
{high_then_shift}
\tv_wmma_i32_16x16x16_iu4 v[0:7], v[8:9], v[10:11], v[0:7] {modifier_low}
.Lfunc_end{checked}:
\t.amdhsa_kernel {symbol}
\t\t.amdhsa_private_segment_fixed_size {scratch}
\t\t.amdhsa_next_free_vgpr 186
\t\t.amdhsa_next_free_sgpr 36
\t.end_amdhsa_kernel
"""


def run_gate(text: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        assembly = pathlib.Path(tmp) / "kernel.s"
        assembly.write_text(text, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(GATE), str(assembly)],
            check=False,
            capture_output=True,
            text=True,
        )


def main() -> int:
    valid = kernel(0) + kernel(1)
    assert run_gate(valid).returncode == 0
    assert run_gate(kernel(0, modifier_low="neg_lo:[1,1,0]") + kernel(1)).returncode != 0
    assert run_gate(kernel(0, shift_before_high=True) + kernel(1)).returncode != 0
    assert run_gate(kernel(0, scratch=4) + kernel(1)).returncode != 0
    print("PASS: rocmi4 W4A8 ISA gate accepts native exact code and rejects drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
