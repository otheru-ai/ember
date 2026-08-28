#!/usr/bin/env python3
"""Gate the exact gfx1151 ROCMI4 W4A8 production-kernel ISA.

Inputs are the gfx1151 `.s` emitted by a ROCm build configured with
GGML_HIP_EXPORT_METRICS=ON and GGML_HIP_ROCMI4_W4A8_IU4=ON, plus llvm-objdump
output from its assembled `.o`.  The gate is GPU-free: it checks the native
instruction encoding and modifiers, algebraic ordering, and AMDHSA resource
metadata in the saved production translation unit.  It makes no device-runtime
or performance claim.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys


KERNEL_RE = re.compile(
    r"^(?P<symbol>_ZL9mul_mat_qIL9ggml_type108ELi(?P<width>\d+)"
    r"ELb(?P<checked>[01])ELb1EE[^:]*)\s*:"
)
TARGET_RE = re.compile(r'^\s*\.amdgcn_target\s+"([^"]+)"\s*$')
V_RANGE_RE = re.compile(r"v\[(\d+):(\d+)\]")
SHIFT_RE = re.compile(
    r"v_(?:dual_)?lshlrev_b32(?:_e32)?\s+v(\d+),\s*4,\s*v(\d+)"
)
REGISTER_RE = re.compile(r"([sv])(?:\[(\d+):(\d+)\]|(\d+))$")
DISASSEMBLY_KERNEL_RE = re.compile(
    r"^[0-9a-fA-F]+ <(?P<symbol>_ZL9mul_mat_qIL9ggml_type108ELi(?P<width>\d+)"
    r"ELb(?P<checked>[01])ELb1EE[^>]*)>:$"
)
DISASSEMBLY_FUNCTION_RE = re.compile(r"^[0-9a-fA-F]+ <[^>]+>:$")
RAW_ENCODING_RE = re.compile(
    r"//\s*[0-9a-fA-F]+:\s*([0-9a-fA-F]{8})\s+([0-9a-fA-F]{8})\s*$"
)
DEFAULT_FACTS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "engine/ggml/rocmfpx/rdna3_5_iu4_isa_facts.json"
)
RESOURCE_LIMITS = {
    "register": {
        0: {"vgpr": 183, "sgpr": 36, "occupancy": 8},
        1: {"vgpr": 190, "sgpr": 36, "occupancy": 8},
    },
    "prepack": {
        0: {"vgpr": 143, "sgpr": 32, "occupancy": 10},
        1: {"vgpr": 141, "sgpr": 34, "occupancy": 10},
    },
}


def fail(message: str) -> None:
    raise ValueError(message)


def load_facts(path: pathlib.Path) -> dict[str, object]:
    """Load and validate the checked-in derivation from AMD's pinned XML."""
    try:
        facts = json.loads(path.read_text(encoding="utf-8"))
        architecture = facts["architecture"]
        instruction = facts["instruction"]
        encoding = instruction["encoding"]
        modifiers = instruction["modifiers"]
        operands = instruction["operands"]
        provenance = facts["provenance"]
        exact = (
            facts["schema"] == "ember.amd-machine-readable-isa-derived.v1"
            and architecture == {
                "id": 9,
                "name": "AMD RDNA 3.5",
                "target": "amdgcn-amd-amdhsa--gfx1151",
                "wave_size": 32,
            }
            and instruction["name"] == "V_WMMA_I32_16X16X16_IU4"
            and instruction["assembly_mnemonic"] == "v_wmma_i32_16x16x16_iu4"
            and instruction["wave32_ab_matrix_copies"] == 2
            and encoding == {
                "bit_count": 64,
                "encoding_tag": {"bit_count": 8, "bit_offset": 24, "value": 204},
                "name": "ENC_VOP3P",
                "opcode": {"bit_count": 7, "bit_offset": 16, "value": 69},
            }
            and modifiers["canonical_assembly"] == {
                "signed_high": "neg_lo:[1,1,0]",
                "unsigned_low": "neg_lo:[1,0,0]",
            }
            and modifiers["clamp"] == {
                "bit_count": 1, "bit_offset": 15, "expected": 0}
            and modifiers["neg_hi"] == {
                "bit_count": 3, "bit_offset": 8, "expected": 0}
            and modifiers["op_sel"] == {
                "bit_count": 3, "bit_offset": 11, "expected": 0}
            and modifiers["neg"] == {
                "bit_count": 3,
                "bit_offset": 61,
                "signed_high": 3,
                "unsigned_low": 1,
            }
            and modifiers["op_sel_hi_ranges"] == [
                {"bit_count": 2, "bit_offset": 59, "expected": 3},
                {"bit_count": 1, "bit_offset": 14, "expected": 1},
            ]
            and operands == {
                "SRC0": {
                    "bit_count": 64,
                    "data_format": "FMT_WMMA_AB_16X16_IU4",
                    "register_count": 2,
                },
                "SRC1": {
                    "bit_count": 64,
                    "data_format": "FMT_WMMA_AB_16X16_IU4",
                    "register_count": 2,
                },
                "SRC2": {
                    "bit_count": 256,
                    "data_format": "FMT_WMMA_DC_16X16_I32",
                    "register_count": 8,
                },
                "VDST": {
                    "bit_count": 256,
                    "data_format": "FMT_WMMA_DC_16X16_I32",
                    "register_count": 8,
                },
            }
            and provenance["archive_sha256"] ==
                "82404f1126761b7877595b622afa7e1f311f2f41e89a3abe9aaf8ad045c082e2"
            and provenance["archive_entry_sha256"] ==
                "c36b6d79b1e940d74107221c985f5a7fde248025da251d2c6ef756c4cd31391a"
            and provenance["archive_entry"] == "amdgpu_isa_rdna3_5.xml"
            and provenance["release_date"] == "2026-02-20"
            and provenance["schema_version"] == "1.1.1"
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        fail(f"malformed RDNA3.5 ISA facts fixture {path}: {exc}")
    if not exact:
        fail(f"RDNA3.5 ISA facts fixture {path} does not match the pinned official derivation")
    return facts


def check_build_configuration(path: pathlib.Path, variant: str) -> None:
    values: dict[str, str] = {}
    wanted = {
        "CMAKE_BUILD_TYPE",
        "GGML_HIP_EXPORT_METRICS",
        "GGML_HIP_ROCMI4_W4A8_IU4",
        "GGML_HIP_ROCMI4_W4A8_IU4_PREPACK",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"([^:#]+):[^=]+=(.*)$", line)
        if match is not None and match.group(1) in wanted:
            values[match.group(1)] = match.group(2)
    expected = {
        "CMAKE_BUILD_TYPE": "Release",
        "GGML_HIP_EXPORT_METRICS": "ON",
        "GGML_HIP_ROCMI4_W4A8_IU4": "ON",
        "GGML_HIP_ROCMI4_W4A8_IU4_PREPACK": (
            "ON" if variant == "prepack" else "OFF"
        ),
    }
    if values != expected:
        fail(
            f"build configuration does not attest the {variant} evidence variant: "
            f"{values} != {expected}"
        )


def function_body(lines: list[str], start: int) -> list[str]:
    end = start + 1
    while end < len(lines) and not re.match(r"^\s*\.Lfunc_end\d+:", lines[end]):
        end += 1
    if end == len(lines):
        fail("candidate function has no .Lfunc_end marker")
    return lines[start:end]


def disassembly_body(lines: list[str], start: int) -> list[str]:
    end = start + 1
    while end < len(lines) and DISASSEMBLY_FUNCTION_RE.match(lines[end]) is None:
        end += 1
    return lines[start:end]


def wmma_mode(
    text: str, mnemonic: str, canonical: dict[str, str]
) -> str | None:
    """Return the exact canonical IU4 mode, rejecting appended modifiers."""
    code = text.split("//", 1)[0].strip()
    if mnemonic not in code:
        return None
    if not code.startswith(f"{mnemonic} "):
        fail(f"malformed IU4 WMMA instruction: {text.strip()}")
    ranges = list(V_RANGE_RE.finditer(code))
    if len(ranges) != 4:
        fail(f"malformed IU4 WMMA operand list: {text.strip()}")
    modifier = code[ranges[-1].end() :].strip()
    modes = {value: key for key, value in canonical.items()}
    if modifier not in modes:
        fail(f"IU4 WMMA has noncanonical or additional modifiers: {text.strip()}")
    return modes[modifier]


def bit_field(word: int, field: dict[str, int]) -> int:
    return (word >> field["bit_offset"]) & ((1 << field["bit_count"]) - 1)


def check_object_encoding(
    path: pathlib.Path,
    assembly_modes: dict[str, list[str]],
    assembly_instructions: dict[str, list[str]],
    facts: dict[str, object],
) -> None:
    """Validate assembled VOP3P words against the official XML-derived fields."""
    lines = path.read_text(encoding="utf-8").splitlines()
    instruction = facts["instruction"]
    assert isinstance(instruction, dict)
    mnemonic = instruction["assembly_mnemonic"]
    encoding = instruction["encoding"]
    modifiers = instruction["modifiers"]
    operands = instruction["operands"]
    assert isinstance(mnemonic, str)
    assert isinstance(encoding, dict)
    assert isinstance(modifiers, dict)
    assert isinstance(operands, dict)
    canonical = modifiers["canonical_assembly"]
    assert isinstance(canonical, dict)

    seen: dict[str, list[str]] = {}
    seen_instructions: dict[str, list[str]] = {}
    for index, line in enumerate(lines):
        match = DISASSEMBLY_KERNEL_RE.match(line)
        if match is None:
            continue
        symbol = match.group("symbol")
        body = disassembly_body(lines, index)
        modes: list[str] = []
        instructions: list[str] = []
        for text in body:
            mode = wmma_mode(text, mnemonic, canonical)
            if mode is None:
                continue
            raw_match = RAW_ENCODING_RE.search(text)
            if raw_match is None:
                fail(f"{symbol}: IU4 object disassembly lacks two raw instruction words")
            low = int(raw_match.group(1), 16)
            high = int(raw_match.group(2), 16)
            raw = low | (high << 32)
            if bit_field(raw, encoding["encoding_tag"]) != encoding["encoding_tag"]["value"]:
                fail(f"{symbol}: IU4 instruction is not the official ENC_VOP3P encoding")
            if bit_field(raw, encoding["opcode"]) != encoding["opcode"]["value"]:
                fail(f"{symbol}: IU4 instruction opcode is not official opcode 69")
            for name in ("clamp", "neg_hi", "op_sel"):
                field = modifiers[name]
                if bit_field(raw, field) != field["expected"]:
                    fail(f"{symbol}: IU4 instruction has noncanonical {name} encoding")
            for field in modifiers["op_sel_hi_ranges"]:
                if bit_field(raw, field) != field["expected"]:
                    fail(f"{symbol}: IU4 instruction has noncanonical op_sel_hi encoding")
            neg = modifiers["neg"]
            if bit_field(raw, neg) != neg[mode]:
                fail(f"{symbol}: IU4 {mode} NEG bits disagree with the assembly modifier")
            register_range(text, 0, operands["VDST"]["register_count"])
            register_range(text, 1, operands["SRC0"]["register_count"])
            register_range(text, 2, operands["SRC1"]["register_count"])
            register_range(text, 3, operands["SRC2"]["register_count"])
            modes.append(mode)
            instructions.append(text.split("//", 1)[0].strip())
        if modes:
            seen[symbol] = modes
            seen_instructions[symbol] = instructions

    if set(seen) != set(assembly_modes):
        fail(
            "object disassembly candidate symbols differ from saved assembly: "
            f"{sorted(seen)} != {sorted(assembly_modes)}"
        )
    for symbol, modes in assembly_modes.items():
        if seen[symbol] != modes:
            fail(f"{symbol}: object IU4 modifier sequence differs from saved assembly")
        if seen_instructions[symbol] != assembly_instructions[symbol]:
            fail(f"{symbol}: object IU4 operands differ from saved assembly")


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


def compiler_metrics(lines: list[str], symbol: str) -> dict[str, int]:
    """Read LLVM's per-kernel resource report instead of estimating it."""
    prefix = f".L{symbol}"
    start = next(
        (index for index, line in enumerate(lines)
         if line.strip().startswith(f".set {prefix}.num_vgpr,")),
        None,
    )
    if start is None:
        fail(f"missing compiler resource report for {symbol}")
    end = start + 1
    while end < len(lines) and not (
        lines[end].lstrip().startswith(".section") and ".text." in lines[end]
    ):
        end += 1
    report = lines[start:end]
    values: dict[str, int] = {}
    properties = {
        "num_vgpr", "numbered_sgpr", "private_seg_size",
        "uses_flat_scratch", "has_dyn_sized_stack",
    }
    property_re = re.compile(
        rf"\s*\.set\s+{re.escape(prefix)}\.(\w+),\s*(\d+)\s*$"
    )
    for line in report:
        match = property_re.match(line)
        if match and match.group(1) in properties:
            values[match.group(1)] = int(match.group(2))
        for key, pattern in (
            ("scratch_size", r"\s*; ScratchSize:\s*(\d+)\s*$"),
            ("occupancy", r"\s*; Occupancy:\s*(\d+)\s*$"),
            ("scratch_enable", r"\s*; COMPUTE_PGM_RSRC2:SCRATCH_EN:\s*(\d+)\s*$"),
        ):
            if (match := re.match(pattern, line)) is not None:
                values[key] = int(match.group(1))
    required = properties | {"scratch_size", "occupancy", "scratch_enable"}
    missing = sorted(required - values.keys())
    if missing:
        fail(f"{symbol}: compiler resource report lacks {missing}")
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


def check_accumulator_dataflow(
    symbol: str,
    body: list[str],
    mnemonic: str,
    canonical: dict[str, str],
) -> None:
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

        mode = wmma_mode(text, mnemonic, canonical)
        if mode is None:
            update_zero_origins(text, zero_sgprs, zero_vgprs)
            continue
        destination = register_range(text, 0, 8)
        register_range(text, 1, 2)
        register_range(text, 2, 2)
        accumulator = register_range(text, 3, 8)
        is_signed = mode == "signed_high"
        is_unsigned = mode == "unsigned_low"

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


def inspect(
    path: pathlib.Path,
    disassembly: pathlib.Path,
    cmake_cache: pathlib.Path,
    variant: str,
    facts_path: pathlib.Path,
) -> list[str]:
    facts = load_facts(facts_path)
    check_build_configuration(cmake_cache, variant)
    architecture = facts["architecture"]
    instruction = facts["instruction"]
    assert isinstance(architecture, dict)
    assert isinstance(instruction, dict)
    target = architecture["target"]
    mnemonic = instruction["assembly_mnemonic"]
    modifiers = instruction["modifiers"]
    operands = instruction["operands"]
    assert isinstance(target, str)
    assert isinstance(mnemonic, str)
    assert isinstance(modifiers, dict)
    assert isinstance(operands, dict)
    canonical = modifiers["canonical_assembly"]
    assert isinstance(canonical, dict)
    lines = path.read_text(encoding="utf-8").splitlines()
    targets = [match.group(1) for line in lines if (match := TARGET_RE.match(line))]
    if targets != [target]:
        fail(f"assembly targets are {targets}, expected exactly ['{target}']")

    candidates: list[tuple[str, int, int, int, int, int]] = []
    assembly_modes: dict[str, list[str]] = {}
    assembly_instructions: dict[str, list[str]] = {}
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
        modes = [
            mode for text in body
            if (mode := wmma_mode(text, mnemonic, canonical)) is not None
        ]
        signed = modes.count("signed_high")
        unsigned = modes.count("unsigned_low")
        if signed != 8 or unsigned != 8 or len(modes) != 16:
            fail(
                f"{symbol}: expected exactly 8 signed-high and 8 unsigned-low "
                f"IU4 WMMAs, found {signed}, {unsigned}, total {len(modes)}"
            )
        assembly_modes[symbol] = modes
        assembly_instructions[symbol] = [
            text.split("//", 1)[0].strip()
            for text in body
            if mnemonic in text
        ]

        check_accumulator_dataflow(symbol, body, mnemonic, canonical)
        if any("scratch_load" in text or "scratch_store" in text for text in body):
            fail(f"{symbol}: explicit scratch instruction emitted")
        if any("v_readlane" in text or "v_writelane" in text for text in body):
            fail(f"{symbol}: scalar/vector register-lane spill instruction emitted")

        metadata = kernel_metadata(lines, symbol)
        compiler = compiler_metrics(lines, symbol)
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
        limits = RESOURCE_LIMITS[variant][checked]
        if vgpr < 1 or vgpr > limits["vgpr"]:
            fail(
                f"{symbol}: {variant} VGPR count {vgpr} exceeds checked={checked} "
                f"screened ceiling {limits['vgpr']}"
            )
        if sgpr < 1 or sgpr > limits["sgpr"]:
            fail(
                f"{symbol}: {variant} numbered SGPR count {sgpr} exceeds "
                f"checked={checked} screened ceiling {limits['sgpr']}"
            )
        if compiler["num_vgpr"] != vgpr or compiler["numbered_sgpr"] != sgpr:
            fail(f"{symbol}: compiler and AMDHSA register counts disagree")
        if any(compiler[key] != 0 for key in (
            "private_seg_size", "uses_flat_scratch", "has_dyn_sized_stack",
            "scratch_size", "scratch_enable",
        )):
            fail(f"{symbol}: compiler resource report enables scratch or a dynamic stack")
        occupancy = compiler["occupancy"]
        if occupancy < limits["occupancy"] or occupancy > 16:
            fail(
                f"{symbol}: {variant} compiler occupancy {occupancy} is outside "
                f"screened range {limits['occupancy']}..16"
            )
        candidates.append((symbol, checked, vgpr, sgpr, static_lds, occupancy))

    if widths != {32}:
        fail(f"candidate widths are {sorted(widths)}, expected only [32]")
    if checked_seen != {0, 1}:
        fail(f"candidate checked variants are {sorted(checked_seen)}, expected [0, 1]")
    if len(candidates) != 2:
        fail(f"found {len(candidates)} candidate functions, expected exactly 2")
    check_object_encoding(
        disassembly, assembly_modes, assembly_instructions, facts
    )

    reports = []
    for _symbol, checked, vgpr, sgpr, static_lds, occupancy in sorted(
        candidates, key=lambda item: item[1]
    ):
        reports.append(
            f"variant={variant} checked={checked} vgpr={vgpr} numbered_sgpr={sgpr} "
            f"scratch_bytes=0 scratch_enable=0 register_lane_spill_ops=0 "
            f"static_lds_bytes={static_lds} "
            f"compiler_occupancy_waves_per_simd={occupancy}"
        )
    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=pathlib.Path)
    parser.add_argument("--disassembly", type=pathlib.Path, required=True)
    parser.add_argument("--cmake-cache", type=pathlib.Path, required=True)
    parser.add_argument("--variant", choices=sorted(RESOURCE_LIMITS), required=True)
    parser.add_argument("--isa-facts", type=pathlib.Path, default=DEFAULT_FACTS)
    args = parser.parse_args()
    try:
        reports = inspect(
            args.assembly,
            args.disassembly,
            args.cmake_cache,
            args.variant,
            args.isa_facts,
        )
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
