#!/usr/bin/env python3
"""Query AMD's machine-readable RDNA 3.5 ISA for what an instruction supports.

Two questions cost real time on this project, and both are answerable in one
lookup rather than a build-and-diagnose cycle:

  * Can I hand-write this with a DPP modifier?  -> needs a *_VOP_DPP16 encoding.
  * Can the compiler pack it into a dual-issue pair? -> needs a VOPD encoding.

Getting these wrong is expensive and silent. V_FMA_MIX_F32 is VOP3P-only with
no V_DUAL_ form, so the VOPD pair I hand-wrote could never assemble;
V_PERMLANEX16_B32 is VOP3-only with no DPP, which is why the offset-16 step of a
wave reduction cannot fold into the DPP butterfly. Both were discovered the slow
way first.

Read the VOPD line carefully: RDNA 3.5 carries dual-issue on SEPARATE mnemonics.
V_AND_B32 has no VOPD encoding of its own, but V_DUAL_AND_B32 exists, so the
operation dual-issues even though the base instruction's encoding list says
nothing about it. There are 17 V_DUAL_* instructions; everything else genuinely
cannot pair.

Spec: https://gpuopen.com/download/machine-readable-isa/latest/
Tooling: https://github.com/GPUOpen-Tools/isa_spec_manager

    tools/isa_query.py <spec.xml> V_FMAC_F32 V_FMA_MIX_F32
    tools/isa_query.py <spec.xml> --dpp-capable V_ADD    # prefix search
    tools/isa_query.py <spec.xml> --full V_FMAC_F32      # operands, opcode, group
    tools/isa_query.py <spec.xml> --encoding ENC_VOP2    # bit layout of a format
    tools/isa_query.py <spec.xml> --group VALU           # instructions in a group
    tools/isa_query.py <spec.xml> --operand-type OPR_VGPR

Schema, verified against amdgpu_isa_rdna3_5.xml rather than taken from the docs:

    Spec/ISA/{Architecture,Encodings,Instructions,DataFormats,OperandTypes,
              FunctionalGroups,FunctionalSubgroups}

    Instruction/{InstructionFlags,InstructionName,Description,
                 InstructionEncodings,FunctionalGroup}
      InstructionEncoding/{EncodingName,EncodingCondition,Opcode,Operands}
        Operand/{FieldName,DataFormatName,OperandType,OperandSize}
                 @Input @Output @IsImplicit @IsBinaryMicrocodeRequired @Order
      FunctionalGroup/{Name,FunctionalSubgroups/Subgroup}

    Encoding/{EncodingName,BitCount,EncodingIdentifierMask,EncodingIdentifiers,
              EncodingConditions,Description,MicrocodeFormat/BitMap}
      BitMap field/{FieldName,Description,BitLayout/Range{BitCount,BitOffset}}

Two traps. The published doc lists six ISA subsections; there are seven
(FunctionalSubgroups is its own top-level list). And the spec misspells
`CondtionExpression` inside EncodingConditions -- match that literally.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def _text(e, path, default=""):
    n = e.find(path)
    return (n.text or "").strip() if n is not None and n.text else default


def load(path):
    root = ET.parse(path).getroot()
    out = {}
    for ins in root.findall(".//Instruction"):
        n = ins.find("InstructionName")
        if n is None or not n.text:
            continue
        encs = sorted({e.text for e in ins.findall(".//EncodingName") if e.text})
        out[n.text] = (encs, _text(ins, "Description"))
    return out


def load_root(path):
    return ET.parse(path).getroot()


def find_instruction(root, name):
    for ins in root.findall(".//Instruction"):
        if _text(ins, "InstructionName") == name:
            return ins
    return None


def report_full(root, name):
    ins = find_instruction(root, name)
    if ins is None:
        print(f"{name}: not found in this spec")
        return
    print(name)
    grp = ins.find("FunctionalGroup")
    if grp is not None:
        subs = [s.text for s in grp.findall("FunctionalSubgroups/Subgroup") if s.text]
        subs = [s for s in subs if s != "NOT_ASSIGNED"]
        print(f"  group          : {_text(grp, 'Name')}"
              + (f" / {', '.join(subs)}" if subs else ""))
    flags = [f.tag for f in (ins.find("InstructionFlags") or [])
             if (f.text or "").strip().upper() == "TRUE"]
    print(f"  flags          : {', '.join(flags) if flags else '(none set)'}")
    desc = _text(ins, "Description")
    if desc:
        print(f"  description    : {desc.splitlines()[0][:100]}")
    for e in ins.findall("InstructionEncodings/InstructionEncoding"):
        op = e.find("Opcode")
        radix = op.get("Radix", "10") if op is not None else "10"
        print(f"  encoding {_text(e, 'EncodingName')}"
              f"  condition={_text(e, 'EncodingCondition')}"
              f"  opcode={_text(e, 'Opcode')} (radix {radix})")
        for o in e.findall("Operands/Operand"):
            io = ("in" if o.get("Input") == "true" else "") + \
                 ("/out" if o.get("Output") == "true" else "")
            io = io.lstrip("/") or "-"
            extra = " implicit" if o.get("IsImplicit") == "true" else ""
            print(f"      {_text(o, 'FieldName'):<10} {_text(o, 'OperandType'):<14}"
                  f" {_text(o, 'DataFormatName'):<16} {_text(o, 'OperandSize'):>3}b"
                  f"  {io}{extra}")


def report_encoding(root, name):
    for E in root.findall(".//Encodings/Encoding"):
        if _text(E, "EncodingName") != name:
            continue
        print(f"{name}  bits={_text(E, 'BitCount')}")
        mask = _text(E, "EncodingIdentifierMask")
        if mask:
            print(f"  identifier mask: {mask}")
        conds = [_text(c, "ConditionName")
                 for c in E.findall("EncodingConditions/EncodingCondition")]
        if conds:
            print(f"  conditions     : {', '.join(conds)}")
        print("  bit layout (field: [offset +count]):")
        for f in E.findall("MicrocodeFormat/BitMap/*"):
            rngs = [f"[{_text(r, 'BitOffset')} +{_text(r, 'BitCount')}]"
                    for r in f.findall("BitLayout/Range")]
            print(f"      {_text(f, 'FieldName'):<12} {' '.join(rngs)}")
        return
    print(f"{name}: no such encoding")


def report_group(root, name):
    hits = []
    for ins in root.findall(".//Instruction"):
        g = ins.find("FunctionalGroup")
        if g is None:
            continue
        subs = [s.text for s in g.findall("FunctionalSubgroups/Subgroup")]
        if _text(g, "Name") == name or name in [s for s in subs if s]:
            hits.append(_text(ins, "InstructionName"))
    print(f"{name}: {len(hits)} instructions")
    for h in sorted(hits):
        print(f"  {h}")


def report_operand_type(root, name):
    for t in root.findall(".//OperandTypes/OperandType"):
        if _text(t, "OperandTypeName") != name:
            continue
        print(f"{name}: {_text(t, 'Description')[:120]}")
        vals = t.findall(".//OperandPredefinedValue")
        if vals:
            print(f"  {len(vals)} predefined values:")
            for v in vals[:24]:
                print(f"      {_text(v, 'Value'):>5}  {_text(v, 'Name')}")
            if len(vals) > 24:
                print(f"      ... {len(vals) - 24} more")
        return
    avail = sorted(_text(t, "OperandTypeName")
                   for t in root.findall(".//OperandTypes/OperandType"))
    print(f"{name}: no such operand type. Available: {', '.join(avail)}")


def report(name, encs, desc, table=None):
    dpp = any("DPP16" in e for e in encs)
    vopd = any("VOPD" in e for e in encs)
    # VOPD is carried by SEPARATE mnemonics: V_AND_B32 has no VOPD encoding but
    # V_DUAL_AND_B32 does. Reporting only the base instruction's encodings reads
    # as "this operation cannot dual-issue" when it can, which is the opposite
    # of the truth and the reason this line exists.
    dual_name = name.replace("V_", "V_DUAL_", 1) if name.startswith("V_") else None
    has_dual = bool(table) and dual_name in table
    print(f"{name}")
    print(f"  DPP16 modifier : {'yes' if dpp else 'NO'}")
    if vopd:
        print(f"  VOPD dual-issue: yes (this encoding)")
    elif has_dual:
        print(f"  VOPD dual-issue: yes, via {dual_name}")
    else:
        print(f"  VOPD dual-issue: NO (and no {dual_name or 'V_DUAL_*'} form)")
    print(f"  encodings      : {', '.join(encs)}")
    if desc:
        print(f"  {desc.splitlines()[0][:100]}")


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    spec, args = argv[1], argv[2:]
    for flag, fn in (("--encoding", report_encoding),
                     ("--group", report_group),
                     ("--operand-type", report_operand_type)):
        if flag in args:
            root = load_root(spec)
            for a in [x for x in args if not x.startswith("--")]:
                fn(root, a.upper())
                print()
            return 0
    if "--full" in args:
        root = load_root(spec)
        for a in [x for x in args if not x.startswith("--")]:
            report_full(root, a.upper())
            print()
        return 0
    table = load(spec)
    prefix_mode = "--dpp-capable" in args
    args = [a for a in args if not a.startswith("--")]
    for a in args:
        key = a.upper()
        if key in table:
            report(key, *table[key], table=table)
        elif prefix_mode:
            hits = [k for k in sorted(table) if k.startswith(key)]
            print(f"{key}* -> {len(hits)} matches, DPP16-capable:")
            for k in hits:
                if any("DPP16" in e for e in table[k][0]):
                    print(f"  {k}")
        else:
            print(f"{key}: not found in this spec")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
