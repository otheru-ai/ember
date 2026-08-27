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
        signed = [i for i, text in enumerate(body) if WMMA in text and "neg_lo:[1,1,0]" in text]
        unsigned = [i for i, text in enumerate(body) if WMMA in text and "neg_lo:[1,0,0]" in text]
        if not signed or len(signed) != len(unsigned):
            fail(f"{symbol}: signed/unsigned IU4 WMMA counts differ")

        # LLVM may interleave independent accumulator groups.  Track WMMA
        # credits rather than requiring all high products before all low
        # products: a signed-high product becomes consumable only after an
        # intervening I32 x16.  This accepts the real scheduled
        # high,high,x16,low,high,low,high,x16,low,low stream while rejecting a
        # low product that has no exact shifted-high accumulator available.
        unshifted_high = 0
        shifted_high = 0
        shift_count = 0
        for text in body:
            is_signed = WMMA in text and "neg_lo:[1,1,0]" in text
            is_unsigned = WMMA in text and "neg_lo:[1,0,0]" in text
            is_x16 = (
                ("v_lshlrev_b32" in text or "v_dual_lshlrev_b32" in text)
                and re.search(r",\s*4,", text) is not None
            )
            if is_signed:
                unshifted_high += 1
            if is_x16 and unshifted_high:
                shifted_high += unshifted_high
                unshifted_high = 0
                shift_count += 1
            if is_unsigned:
                if shifted_high == 0:
                    fail(f"{symbol}: unsigned-low WMMA lacks a preceding shifted signed-high product")
                shifted_high -= 1
        if unshifted_high or shifted_high or shift_count == 0:
            fail(f"{symbol}: incomplete signed-high -> x16 I32 -> unsigned-low WMMA sequence")
        if any("scratch_load" in text or "scratch_store" in text for text in body):
            fail(f"{symbol}: explicit scratch instruction emitted")

        metadata = kernel_metadata(lines, symbol)
        private = metadata.get("private_segment_fixed_size", -1)
        vgpr = metadata.get("next_free_vgpr", -1)
        sgpr = metadata.get("next_free_sgpr", -1)
        if private != 0:
            fail(f"{symbol}: scratch/private segment is {private}, expected 0")
        if vgpr < 1 or vgpr > 186:
            fail(f"{symbol}: VGPR count {vgpr} exceeds screened ceiling 186")
        if sgpr < 1 or sgpr > 36:
            fail(f"{symbol}: numbered SGPR count {sgpr} exceeds screened ceiling 36")
        candidates.append((symbol, checked, vgpr, sgpr))

    if widths != {32}:
        fail(f"candidate widths are {sorted(widths)}, expected only [32]")
    if checked_seen != {0, 1}:
        fail(f"candidate checked variants are {sorted(checked_seen)}, expected [0, 1]")

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
