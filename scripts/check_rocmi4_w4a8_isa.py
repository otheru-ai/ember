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
V_RANGE_RE = re.compile(r"v\[(\d+):(\d+)\]")
SHIFT_RE = re.compile(
    r"v_(?:dual_)?lshlrev_b32(?:_e32)?\s+v(\d+),\s*4,\s*v(\d+)"
)


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


def register_range(text: str, operand: int) -> list[int]:
    ranges = V_RANGE_RE.findall(text)
    if len(ranges) != 4:
        fail(f"malformed IU4 WMMA operand list: {text.strip()}")
    first, last = (int(value) for value in ranges[operand])
    if last - first != 7:
        fail(f"IU4 WMMA accumulator range is not eight registers: {text.strip()}")
    return list(range(first, last + 1))


def check_accumulator_dataflow(symbol: str, body: list[str]) -> None:
    # Physical registers are reused aggressively, so carry only the state that
    # matters to the exact high*16+low accumulator chain.  The lane ordinal is
    # retained across compiler register renames performed by v_lshlrev_b32.
    registers: dict[int, tuple[int, int, str]] = {}
    groups: dict[int, dict[str, object]] = {}
    next_group = 0

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
            continue
        destination = register_range(text, 0)
        accumulator = register_range(text, 3)
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
                group = next_group
                next_group += 1
                groups[group] = {"high": 1, "low": 0, "shifted": set()}
            for lane, reg in enumerate(destination):
                registers[reg] = (group, lane, "high")
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

    if not groups or any(
        int(group["high"]) != 2
        or int(group["low"]) != 2
        or group["shifted"] != set(range(8))
        for group in groups.values()
    ):
        fail(f"{symbol}: incomplete two-high -> eight-lane x16 -> two-low accumulator group")


def inspect(path: pathlib.Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    candidates: list[tuple[str, int, int, int]] = []
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
        vgpr = metadata.get("next_free_vgpr", -1)
        sgpr = metadata.get("next_free_sgpr", -1)
        if private != 0:
            fail(f"{symbol}: scratch/private segment is {private}, expected 0")
        if vgpr < 1 or vgpr > 190:
            fail(f"{symbol}: VGPR count {vgpr} exceeds screened ceiling 190")
        if sgpr < 1 or sgpr > 36:
            fail(f"{symbol}: numbered SGPR count {sgpr} exceeds screened ceiling 36")
        candidates.append((symbol, checked, vgpr, sgpr))

    if widths != {32}:
        fail(f"candidate widths are {sorted(widths)}, expected only [32]")
    if checked_seen != {0, 1}:
        fail(f"candidate checked variants are {sorted(checked_seen)}, expected [0, 1]")
    if len(candidates) != 2:
        fail(f"found {len(candidates)} candidate functions, expected exactly 2")

    reports = []
    for _symbol, checked, vgpr, sgpr in sorted(candidates, key=lambda item: item[1]):
        # LLVM's TotalSGPRs adds VCC to the numbered count. gfx1151 has 1536
        # vector registers per SIMD; this matches the compiler's 8-wave report.
        occupancy = 1536 // vgpr
        reports.append(
            f"checked={checked} vgpr={vgpr} numbered_sgpr={sgpr} "
            f"scratch=0 spills=0 occupancy_waves_per_simd={occupancy}"
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
    print("dynamic_lds_bytes=27776 (width32 production launch contract)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
