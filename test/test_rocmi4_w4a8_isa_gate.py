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
    groups: int = 4,
    partial_shift: bool = False,
    dual_shift: bool = False,
    src0: str = "v[8:9]",
    src1: str = "v[10:11]",
    zero_origin: bool = True,
    wave32: int = 1,
    static_lds: int = 0,
    vgpr: int = 186,
    sgpr: int = 36,
) -> str:
    symbol = (
        "_ZL9mul_mat_qIL9ggml_type108ELi32ELb"
        f"{checked}ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii"
    )
    high = f"""\tv_wmma_i32_16x16x16_iu4 v[0:7], {src0}, {src1}, v[16:23] neg_lo:[1,1,0]
\tv_wmma_i32_16x16x16_iu4 v[0:7], {src0}, {src1}, v[0:7] neg_lo:[1,1,0]"""
    lanes = list(range(7 if partial_shift else 8))
    if dual_shift:
        shifts = "\n".join(
            f"\tv_dual_lshlrev_b32 v{lanes[index]}, 4, v{lanes[index]} :: "
            f"v_dual_lshlrev_b32 v{lanes[index + 1]}, 4, v{lanes[index + 1]}"
            for index in range(0, len(lanes), 2)
        )
    else:
        shifts = "\n".join(
            f"\tv_lshlrev_b32_e32 v{lane}, 4, v{lane}" for lane in lanes
        )
    if shift_before_high:
        high_then_shift = f"{shifts}\n{high}"
    else:
        high_then_shift = f"{high}\n{shifts}"
    group = f"""{high_then_shift}
\tv_wmma_i32_16x16x16_iu4 v[0:7], {src0}, {src1}, v[0:7] {modifier_low}
\tv_wmma_i32_16x16x16_iu4 v[0:7], {src0}, {src1}, v[0:7] {modifier_low}"""
    body = "\n".join(group for _ in range(groups))
    zero = 0 if zero_origin else 1
    initialize = "\n".join(
        f"\tv_mov_b32_e32 v{register}, {zero}" for register in range(16, 24)
    )
    return f"""{symbol}:
{initialize}
{body}
.Lfunc_end{checked}:
\t.amdhsa_kernel {symbol}
\t\t.amdhsa_group_segment_fixed_size {static_lds}
\t\t.amdhsa_private_segment_fixed_size {scratch}
\t\t.amdhsa_wavefront_size32 {wave32}
\t\t.amdhsa_next_free_vgpr {vgpr}
\t\t.amdhsa_next_free_sgpr {sgpr}
\t.end_amdhsa_kernel
"""


def program(*kernels: str, target: str = "amdgcn-amd-amdhsa--gfx1151") -> str:
    return f'\t.amdgcn_target "{target}"\n' + "".join(kernels)


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
    valid = program(kernel(0), kernel(1))
    result = run_gate(valid)
    assert result.returncode == 0, result.stderr
    assert "static_lds_bytes=0" in result.stdout
    assert "dynamic_lds_bytes=not_encoded_in_saved_assembly" in result.stdout
    assert run_gate(
        program(kernel(0), kernel(1), target="amdgcn-amd-amdhsa--gfx1150")
    ).returncode != 0
    assert run_gate(program(kernel(0, wave32=0), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, static_lds=4), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, src0="v[8:10]"), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, src1="v[10:10]"), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, zero_origin=False), kernel(1))).returncode != 0
    assert run_gate(
        program(kernel(0, modifier_low="neg_lo:[1,1,0]"), kernel(1))
    ).returncode != 0
    assert run_gate(program(kernel(0, shift_before_high=True), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, scratch=4), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, vgpr=191), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, sgpr=37), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, groups=3), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, partial_shift=True), kernel(1))).returncode != 0
    assert run_gate(
        program(kernel(0, dual_shift=True), kernel(1, dual_shift=True))
    ).returncode == 0
    assert run_gate(program(kernel(0), kernel(0), kernel(1))).returncode != 0
    print("PASS: rocmi4 W4A8 ISA gate accepts native exact code and rejects drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
