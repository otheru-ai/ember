#!/usr/bin/env python3
"""GPU-free regression tests for the saved-production-ISA gate."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "check_rocmi4_w4a8_isa.py"
FACTS = ROOT / "engine/ggml/rocmfpx/rdna3_5_iu4_isa_facts.json"
KERNEL_RE = re.compile(
    r"^(?P<symbol>_ZL9mul_mat_qIL9ggml_type108ELi32ELb[01]ELb1EE[^:]*)\s*:"
)
SCREENED = {
    "register": {
        0: (183, 36, 8),
        1: (190, 36, 8),
    },
    "prepack": {
        0: (143, 32, 10),
        1: (141, 34, 10),
    },
}


def kernel(
    checked: int,
    variant: str = "register",
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
    vgpr: int | None = None,
    sgpr: int | None = None,
    compiler_scratch: int | None = None,
    compiler_occupancy: int | None = None,
    lane_spill: bool = False,
) -> str:
    screened_vgpr, screened_sgpr, screened_occupancy = SCREENED[variant][checked]
    if vgpr is None:
        vgpr = screened_vgpr
    if sgpr is None:
        sgpr = screened_sgpr
    if compiler_occupancy is None:
        compiler_occupancy = screened_occupancy
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
    if lane_spill:
        body += "\n\tv_readlane_b32 s8, v0, 0"
    zero = 0 if zero_origin else 1
    reported_scratch = scratch if compiler_scratch is None else compiler_scratch
    scratch_enable = int(reported_scratch != 0)
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
\t.set .L{symbol}.num_vgpr, {vgpr}
\t.set .L{symbol}.numbered_sgpr, {sgpr}
\t.set .L{symbol}.private_seg_size, {reported_scratch}
\t.set .L{symbol}.uses_flat_scratch, {scratch_enable}
\t.set .L{symbol}.has_dyn_sized_stack, 0
; ScratchSize: {reported_scratch}
; Occupancy: {compiler_occupancy}
; COMPUTE_PGM_RSRC2:SCRATCH_EN: {scratch_enable}
\t.section .text.synthetic
"""


def program(*kernels: str, target: str = "amdgcn-amd-amdhsa--gfx1151") -> str:
    return f'\t.amdgcn_target "{target}"\n' + "".join(kernels)


def raw_words(
    modifier: str,
    opcode: int = 69,
    encoding_tag: int = 204,
    clamp: int = 0,
    neg_hi: int = 0,
    op_sel: int = 0,
    op_sel_hi_low: int = 3,
    op_sel_hi_high: int = 1,
    neg_override: int | None = None,
) -> tuple[int, int]:
    neg = (3 if modifier == "neg_lo:[1,1,0]" else 1)
    if neg_override is not None:
        neg = neg_override
    raw = (
        (encoding_tag << 24)
        | (opcode << 16)
        | (clamp << 15)
        | (op_sel_hi_high << 14)
        | (op_sel << 11)
        | (neg_hi << 8)
        | (neg << 61)
        | (op_sel_hi_low << 59)
    )
    return raw & 0xffffffff, raw >> 32


def object_disassembly(
    assembly: str,
    *,
    opcode: int = 69,
    encoding_tag: int = 204,
    clamp: int = 0,
    neg_hi: int = 0,
    op_sel: int = 0,
    op_sel_hi_low: int = 3,
    op_sel_hi_high: int = 1,
    neg_override: int | None = None,
) -> str:
    output: list[str] = []
    symbol: str | None = None
    address = 0
    for line in assembly.splitlines():
        if match := KERNEL_RE.match(line):
            symbol = match.group("symbol")
            output.append(f"0000000000000000 <{symbol}>:")
            address = 0
            continue
        if symbol is None or "v_wmma_i32_16x16x16_iu4" not in line:
            continue
        modifier_match = re.search(r"neg_lo:\[[01],[01],[01]\]", line)
        if modifier_match is None:
            raise AssertionError(f"synthetic WMMA lacks modifier: {line}")
        modifier = modifier_match.group(0)
        low, high = raw_words(
            modifier,
            opcode=opcode,
            encoding_tag=encoding_tag,
            clamp=clamp,
            neg_hi=neg_hi,
            op_sel=op_sel,
            op_sel_hi_low=op_sel_hi_low,
            op_sel_hi_high=op_sel_hi_high,
            neg_override=neg_override,
        )
        output.append(
            f"{line.strip()} // {address:012X}: {low:08X} {high:08X}"
        )
        address += 8
    return "\n".join(output) + "\n"


def cache(variant: str) -> str:
    prepack = "ON" if variant == "prepack" else "OFF"
    return f"""CMAKE_BUILD_TYPE:STRING=Release
GGML_HIP_EXPORT_METRICS:BOOL=ON
GGML_HIP_ROCMI4_W4A8_IU4:BOOL=ON
GGML_HIP_ROCMI4_W4A8_IU4_PREPACK:BOOL={prepack}
"""


def run_gate(
    text: str,
    variant: str = "register",
    *,
    disassembly: str | None = None,
    cmake_cache: str | None = None,
    isa_facts: str | None = None,
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        assembly = pathlib.Path(tmp) / "kernel.s"
        disassembly_path = pathlib.Path(tmp) / "kernel.disasm"
        cache_path = pathlib.Path(tmp) / "CMakeCache.txt"
        facts_path = pathlib.Path(tmp) / "isa-facts.json"
        assembly.write_text(text, encoding="utf-8")
        disassembly_path.write_text(
            object_disassembly(text) if disassembly is None else disassembly,
            encoding="utf-8",
        )
        cache_path.write_text(
            cache(variant) if cmake_cache is None else cmake_cache,
            encoding="utf-8",
        )
        if isa_facts is not None:
            facts_path.write_text(isa_facts, encoding="utf-8")
        command = [
            sys.executable,
            str(GATE),
            str(assembly),
            "--disassembly",
            str(disassembly_path),
            "--cmake-cache",
            str(cache_path),
            "--variant",
            variant,
        ]
        if isa_facts is not None:
            command.extend(("--isa-facts", str(facts_path)))
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )


def main() -> int:
    valid = program(kernel(0), kernel(1))
    result = run_gate(valid)
    assert result.returncode == 0, result.stderr
    assert "variant=register" in result.stdout
    assert "static_lds_bytes=0" in result.stdout
    assert "dynamic_lds_bytes=not_encoded_in_saved_assembly" in result.stdout
    valid_prepack = program(
        kernel(0, variant="prepack"), kernel(1, variant="prepack")
    )
    result = run_gate(valid_prepack, "prepack")
    assert result.returncode == 0, result.stderr
    assert "variant=prepack" in result.stdout
    assert run_gate(
        valid_prepack, "prepack", cmake_cache=cache("register")
    ).returncode != 0
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
    assert run_gate(
        program(kernel(0, modifier_low="neg_lo:[1,0,0] clamp"), kernel(1))
    ).returncode != 0
    assert run_gate(program(kernel(0, shift_before_high=True), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, scratch=4), kernel(1))).returncode != 0
    assert run_gate(
        program(kernel(0, compiler_scratch=4), kernel(1))
    ).returncode != 0
    assert run_gate(program(kernel(0, lane_spill=True), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, compiler_occupancy=17), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, compiler_occupancy=7), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, vgpr=183), kernel(1))).returncode == 0
    assert run_gate(program(kernel(0, vgpr=184), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0), kernel(1, vgpr=190))).returncode == 0
    assert run_gate(program(kernel(0), kernel(1, vgpr=191))).returncode != 0
    assert run_gate(program(kernel(0, sgpr=37), kernel(1))).returncode != 0
    assert run_gate(
        program(kernel(0, variant="prepack", vgpr=144),
                kernel(1, variant="prepack")),
        "prepack",
    ).returncode != 0
    assert run_gate(
        program(kernel(0, variant="prepack", sgpr=33),
                kernel(1, variant="prepack")),
        "prepack",
    ).returncode != 0
    assert run_gate(
        program(kernel(0, variant="prepack", compiler_occupancy=9),
                kernel(1, variant="prepack")),
        "prepack",
    ).returncode != 0
    assert run_gate(program(kernel(0, groups=3), kernel(1))).returncode != 0
    assert run_gate(program(kernel(0, partial_shift=True), kernel(1))).returncode != 0
    assert run_gate(
        program(kernel(0, dual_shift=True), kernel(1, dual_shift=True))
    ).returncode == 0
    assert run_gate(program(kernel(0), kernel(0), kernel(1))).returncode != 0
    for kwargs in (
        {"opcode": 68},
        {"encoding_tag": 205},
        {"clamp": 1},
        {"neg_hi": 1},
        {"op_sel": 1},
        {"op_sel_hi_low": 2},
        {"op_sel_hi_high": 0},
        {"neg_override": 0},
    ):
        assert run_gate(
            valid, disassembly=object_disassembly(valid, **kwargs)
        ).returncode != 0
    changed_facts = json.loads(FACTS.read_text(encoding="utf-8"))
    changed_facts["instruction"]["encoding"]["opcode"]["value"] = 68
    assert run_gate(
        valid, isa_facts=json.dumps(changed_facts)
    ).returncode != 0
    changed_object_operands = object_disassembly(valid).replace(
        "v[8:9]", "v[12:13]", 1
    )
    assert run_gate(
        valid, disassembly=changed_object_operands
    ).returncode != 0
    missing_object_variant = object_disassembly(valid).split(
        "0000000000000000 <", 2
    )
    assert len(missing_object_variant) == 3
    assert run_gate(
        valid,
        disassembly=(
            "0000000000000000 <" + missing_object_variant[1]
        ),
    ).returncode != 0
    print(
        "PASS: rocmi4 W4A8 ISA gate validates official object encoding, "
        "variant resources, and drift rejection"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
