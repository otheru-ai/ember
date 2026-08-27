#!/usr/bin/env python3
"""Gate the exact gfx1151 ROCMI4 W4A8 production-kernel ISA.

Input is the gfx1151 `.s` emitted by a ROCm build configured with
GGML_HIP_EXPORT_METRICS=ON and GGML_HIP_ROCMI4_W4A8_IU4=ON.  The gate is
GPU-free: it checks the native instruction modifiers, algebraic ordering, and
AMDHSA resource metadata in the saved production translation unit.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


KERNEL_RE = re.compile(
    r"^(?P<symbol>_ZL9mul_mat_qIL9ggml_type108ELi(?P<width>\d+)"
    r"ELb(?P<checked>[01])ELb1EE[^:]*)\s*:"
)
WMMA = "v_wmma_i32_16x16x16_iu4"
TARGET = "amdgcn-amd-amdhsa--gfx1151"
TARGET_RE = re.compile(r'^\s*\.amdgcn_target\s+"([^"]+)"\s*$')
V_RANGE_RE = re.compile(r"v\[(\d+):(\d+)\]")
SHIFT_RE = re.compile(
    r"v_(?:dual_)?lshlrev_b32(?:_e32)?\s+v(\d+),\s*4,\s*v(\d+)"
)
REGISTER_RE = re.compile(r"([sv])(?:\[(\d+):(\d+)\]|(\d+))$")


def fail(message: str) -> None:
    raise ValueError(message)


def function_body(lines: list[str], start: int) -> list[str]:
    end = start + 1
    while end < len(lines) and not re.match(r"^\s*\.Lfunc_end\d+:", lines[end]):
        end += 1
    if end == len(lines):
        fail("candidate function has no .Lfunc_end marker")
    return lines[start:end]


def kernel_metadata(lines: list[str], symbol: str) -> dict[str, int]:
    marker = f"\t.amdhsa_kernel {symbol}"
    try:
        start = lines.index(marker)
    except ValueError as exc:
        fail(f"missing AMDHSA metadata for {symbol}")
        raise AssertionError from exc
    values: dict[str, int] = {}
    for line in lines[start + 1 :]:
        if line.strip() == ".end_amdhsa_kernel":
            break
        match = re.match(r"\s*\.amdhsa_(\w+)\s+(\d+)$", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    return values


def register_range(text: str, operand: int, expected: int) -> list[int]:
    ranges = V_RANGE_RE.findall(text)
    if len(ranges) != 4:
        fail(f"malformed IU4 WMMA operand list: {text.strip()}")
    first, last = (int(value) for value in ranges[operand])
    if last - first + 1 != expected:
        names = ("destination", "SRC0", "SRC1", "accumulator")
        fail(
            f"IU4 WMMA {names[operand]} range is not {expected} registers: "
            f"{text.strip()}"
        )
    return list(range(first, last + 1))


def operand_registers(operand: str) -> tuple[str, list[int]] | None:
    match = REGISTER_RE.fullmatch(operand.strip())
    if match is None:
        return None
    kind = match.group(1)
    if match.group(4) is not None:
        first = int(match.group(4))
        return kind, [first]
    first = int(match.group(2))
    last = int(match.group(3))
    return kind, list(range(first, last + 1))


def update_zero_origins(
    text: str, zero_sgprs: set[int], zero_vgprs: set[int]
) -> None:
    """Track only zero values whose origin remains provable in linear ISA.

    The production accumulator is initialized by scalar zero moves followed by
    scalar-to-vector moves.  Treat every recognized register write as a kill;
    only a move from literal zero or another proven-zero register creates a new
    zero.  Comparison/store/control instructions do not write their first
    register operand and are deliberately excluded.
    """

    for part in text.split("::"):
        instruction = part.strip().split(";", 1)[0].strip()
        if not instruction or instruction.startswith("."):
            continue
        fields = instruction.split(None, 1)
        if len(fields) != 2:
            continue
        mnemonic, operand_text = fields
        operands = [operand.strip() for operand in operand_text.split(",")]
        destination = operand_registers(operands[0])
        if destination is None:
            continue
        kind, registers = destination

        writes_scalar = mnemonic.startswith("s_") and not mnemonic.startswith(
            (
                "s_cmp",
                "s_bitcmp",
                "s_cbranch",
                "s_branch",
                "s_wait",
                "s_barrier",
                "s_end",
                "s_sendmsg",
                "s_nop",
                "s_delay",
                "s_store",
            )
        )
        writes_vector = (
            mnemonic.startswith("v_")
            and not mnemonic.startswith(("v_cmp", "v_readfirstlane"))
        ) or mnemonic.startswith(
            ("ds_load", "global_load", "flat_load", "buffer_load", "scratch_load")
        )
        if kind == "s" and not writes_scalar:
            continue
        if kind == "v" and not writes_vector:
            continue

        zero_set = zero_sgprs if kind == "s" else zero_vgprs
        zero_set.difference_update(registers)
        is_move = mnemonic in {
            "s_mov_b32",
            "s_mov_b64",
            "v_mov_b32",
            "v_mov_b32_e32",
            "v_dual_mov_b32",
        }
        if not is_move or len(operands) < 2:
            continue
        source = operands[1]
        source_register = operand_registers(source)
        source_is_zero = source in {"0", "0x0"}
        if source_register is not None:
            source_kind, source_registers = source_register
            source_set = zero_sgprs if source_kind == "s" else zero_vgprs
            source_is_zero = all(register in source_set for register in source_registers)
        if source_is_zero:
            zero_set.update(registers)


def check_accumulator_dataflow(symbol: str, body: list[str]) -> None:
    # Physical registers are reused aggressively, so carry only the state that
    # matters to the exact high*16+low accumulator chain.  The lane ordinal is
    # retained across compiler register renames performed by v_lshlrev_b32.
    registers: dict[int, tuple[int, int, str]] = {}
    groups: dict[int, dict[str, object]] = {}
    next_group = 0
    zero_sgprs: set[int] = set()
    zero_vgprs: set[int] = set()

    for text in body:
        # A dual-issue assembly line can contain two independent shift
        # mnemonics. Validate both accumulator lanes rather than crediting only
        # the first textual match.
        for shift in SHIFT_RE.finditer(text):
            dst = int(shift.group(1))
            src = int(shift.group(2))
            token = registers.get(src)
            if token is not None and token[2] == "high":
                group, lane, _stage = token
                registers[dst] = (group, lane, "shifted")
                shifted = groups[group]["shifted"]
                assert isinstance(shifted, set)
                shifted.add(lane)

        if WMMA not in text:
            update_zero_origins(text, zero_sgprs, zero_vgprs)
            continue
        destination = register_range(text, 0, 8)
        register_range(text, 1, 2)
        register_range(text, 2, 2)
        accumulator = register_range(text, 3, 8)
        is_signed = "neg_lo:[1,1,0]" in text
        is_unsigned = "neg_lo:[1,0,0]" in text

        if is_signed:
            tokens = [registers.get(reg) for reg in accumulator]
            carry = (
                all(token is not None for token in tokens)
                and len({token[0] for token in tokens if token is not None}) == 1
                and all(token is not None and token[1] == lane and token[2] == "high"
                        for lane, token in enumerate(tokens))
            )
            if carry:
                group = tokens[0][0]  # type: ignore[index]
                groups[group]["high"] = int(groups[group]["high"]) + 1
            else:
                if any(reg not in zero_vgprs for reg in accumulator):
                    fail(f"{symbol}: first signed-high WMMA accumulator is not zero-origin")
                group = next_group
                next_group += 1
                groups[group] = {"high": 1, "low": 0, "shifted": set()}
            for lane, reg in enumerate(destination):
                registers[reg] = (group, lane, "high")
            zero_vgprs.difference_update(destination)
            continue

        if not is_unsigned:
            fail(f"{symbol}: IU4 WMMA has an unrecognized signedness modifier")

        tokens = [registers.get(reg) for reg in accumulator]
        if any(token is None for token in tokens):
            fail(f"{symbol}: unsigned-low WMMA accumulator has no signed-high origin")
        group_ids = {token[0] for token in tokens if token is not None}
        if len(group_ids) != 1:
            fail(f"{symbol}: unsigned-low WMMA mixes signed-high accumulator groups")
        group = next(iter(group_ids))
        stages = {token[2] for token in tokens if token is not None}
        if stages == {"shifted"}:
            shifted = groups[group]["shifted"]
            if shifted != set(range(8)) or any(
                token is None or token[1] != lane for lane, token in enumerate(tokens)
            ):
                fail(f"{symbol}: unsigned-low WMMA sees a partially shifted I32 accumulator")
        elif stages == {"done"}:
            if any(token is None or token[1] != lane for lane, token in enumerate(tokens)):
                fail(f"{symbol}: unsigned-low WMMA accumulator lane order changed")
        else:
            fail(f"{symbol}: unsigned-low WMMA lacks eight matching x16 accumulator shifts")
        groups[group]["low"] = int(groups[group]["low"]) + 1
        for lane, reg in enumerate(destination):
            registers[reg] = (group, lane, "done")
        zero_vgprs.difference_update(destination)

    if not groups or any(
        int(group["high"]) != 2
        or int(group["low"]) != 2
        or group["shifted"] != set(range(8))
        for group in groups.values()
    ):
        fail(f"{symbol}: incomplete two-high -> eight-lane x16 -> two-low accumulator group")


def inspect(path: pathlib.Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    targets = [match.group(1) for line in lines if (match := TARGET_RE.match(line))]
    if targets != [TARGET]:
        fail(f"assembly targets are {targets}, expected exactly ['{TARGET}']")

    candidates: list[tuple[str, int, int, int, int]] = []
    widths: set[int] = set()
    checked_seen: set[int] = set()

    for index, line in enumerate(lines):
        match = KERNEL_RE.match(line)
        if not match:
            continue
        symbol = match.group("symbol")
        width = int(match.group("width"))
        checked = int(match.group("checked"))
        widths.add(width)
        checked_seen.add(checked)
        body = function_body(lines, index)
        all_wmma = [i for i, text in enumerate(body) if WMMA in text]
        signed = [i for i, text in enumerate(body) if WMMA in text and "neg_lo:[1,1,0]" in text]
        unsigned = [i for i, text in enumerate(body) if WMMA in text and "neg_lo:[1,0,0]" in text]
        if len(signed) != 8 or len(unsigned) != 8 or len(all_wmma) != 16:
            fail(
                f"{symbol}: expected exactly 8 signed-high and 8 unsigned-low "
                f"IU4 WMMAs, found {len(signed)}, {len(unsigned)}, total {len(all_wmma)}"
            )

        check_accumulator_dataflow(symbol, body)
        if any("scratch_load" in text or "scratch_store" in text for text in body):
            fail(f"{symbol}: explicit scratch instruction emitted")

        metadata = kernel_metadata(lines, symbol)
        private = metadata.get("private_segment_fixed_size", -1)
        static_lds = metadata.get("group_segment_fixed_size", -1)
        wave32 = metadata.get("wavefront_size32", -1)
        vgpr = metadata.get("next_free_vgpr", -1)
        sgpr = metadata.get("next_free_sgpr", -1)
        if private != 0:
            fail(f"{symbol}: scratch/private segment is {private}, expected 0")
        if static_lds != 0:
            fail(f"{symbol}: static LDS is {static_lds}, expected 0")
        if wave32 != 1:
            fail(f"{symbol}: wave32 metadata is {wave32}, expected 1")
        if vgpr < 1 or vgpr > 190:
            fail(f"{symbol}: VGPR count {vgpr} exceeds screened ceiling 190")
        if sgpr < 1 or sgpr > 36:
            fail(f"{symbol}: numbered SGPR count {sgpr} exceeds screened ceiling 36")
        candidates.append((symbol, checked, vgpr, sgpr, static_lds))

    if widths != {32}:
        fail(f"candidate widths are {sorted(widths)}, expected only [32]")
    if checked_seen != {0, 1}:
        fail(f"candidate checked variants are {sorted(checked_seen)}, expected [0, 1]")
    if len(candidates) != 2:
        fail(f"found {len(candidates)} candidate functions, expected exactly 2")

    reports = []
    for _symbol, checked, vgpr, sgpr, static_lds in sorted(
        candidates, key=lambda item: item[1]
    ):
        # LLVM's TotalSGPRs adds VCC to the numbered count. gfx1151 has 1536
        # vector registers per SIMD; this matches the compiler's 8-wave report.
        occupancy = 1536 // vgpr
        reports.append(
            f"checked={checked} vgpr={vgpr} numbered_sgpr={sgpr} "
            f"scratch=0 spills=0 static_lds_bytes={static_lds} "
            f"occupancy_waves_per_simd={occupancy}"
        )
    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=pathlib.Path)
    args = parser.parse_args()
    try:
        reports = inspect(args.assembly)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS: gfx1151 native exact ROCMI4 W4A8 IU4 ISA")
    for report in reports:
        print(report)
    print("dynamic_lds_bytes=not_encoded_in_saved_assembly (validate at launch/profile time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
