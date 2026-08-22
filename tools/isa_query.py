#!/usr/bin/env python3
"""Query AMD's machine-readable RDNA 3.5 ISA for what an instruction supports.

Two questions cost real time on this project, and both are answerable in one
lookup rather than a build-and-diagnose cycle:

  * Can I hand-write this with a DPP modifier?  -> needs a *_VOP_DPP16 encoding.
  * Can the compiler pack it into a dual-issue pair? -> needs a VOPD encoding.

Getting these wrong is expensive and silent. V_FMA_MIX_F32 is VOP3P-only, so
the VOPD pair I hand-wrote could never assemble; V_PERMLANEX16_B32 is VOP3-only
with no DPP, which is why the offset-16 step of a wave reduction cannot fold
into the DPP butterfly. Both were discovered the slow way first.

Spec: https://gpuopen.com/download/machine-readable-isa/latest/
Tooling: https://github.com/GPUOpen-Tools/isa_spec_manager

    tools/isa_query.py <spec.xml> V_FMAC_F32 V_FMA_MIX_F32
    tools/isa_query.py <spec.xml> --dpp-capable V_ADD    # prefix search
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def load(path):
    root = ET.parse(path).getroot()
    out = {}
    for ins in root.findall(".//Instruction"):
        n = ins.find("InstructionName")
        if n is None or not n.text:
            continue
        encs = sorted({e.text for e in ins.findall(".//EncodingName") if e.text})
        desc = ins.find("Description")
        out[n.text] = (encs, (desc.text or "").strip() if desc is not None else "")
    return out


def report(name, encs, desc):
    dpp = any("DPP16" in e for e in encs)
    vopd = any("VOPD" in e for e in encs)
    print(f"{name}")
    print(f"  DPP16 modifier : {'yes' if dpp else 'NO'}")
    print(f"  VOPD dual-issue: {'yes' if vopd else 'NO'}")
    print(f"  encodings      : {', '.join(encs)}")
    if desc:
        print(f"  {desc.splitlines()[0][:100]}")


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    spec, args = argv[1], argv[2:]
    table = load(spec)
    prefix_mode = "--dpp-capable" in args
    args = [a for a in args if not a.startswith("--")]
    for a in args:
        key = a.upper()
        if key in table:
            report(key, *table[key])
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
