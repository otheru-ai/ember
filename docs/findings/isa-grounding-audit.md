# Are our hardware claims grounded in the ISA?

2026-09-02. Every assertion about gfx1151 behaviour written into the engine or
its docs, checked against `/srv/isa/amdgpu_isa_rdna3_5.xml`.

## What the ISA can and cannot settle

It is an **encoding** spec. It answers: does this instruction exist, what
encodings and modifiers does it have, what are its operands and bit layout,
what functional group is it in. It does **not** carry instruction latencies,
issue rates, LDS or register-file capacity, occupancy rules, or what a compiler
chooses to emit. Claims of that kind are not "ungrounded" for failing here —
they need a different source, and this audit says which.

## Verified against the spec

| claim | where | verdict |
|---|---|---|
| `v_and`, `v_lshlrev`, `v_add_nc`, `v_cndmask` can dual-issue | `rocmfp4_hip_scale.cuh:67` | **grounded** — all four have `V_DUAL_*` forms |
| `V_FMA_MIX_F32` is VOP3P-only, so a hand-written VOPD pair cannot assemble | `tools/isa_query.py` docstring | **grounded** — VOP3P encodings only, and no `V_DUAL_FMA_MIX_F32` |
| `V_PERMLANEX16_B32` has no DPP, so it cannot fold into a DPP butterfly | same | **grounded** — `ENC_VOP3` only |
| the quantized dot path cannot dual-issue | `isa-assembly-opportunities.md` | **grounded** — neither `V_DOT4_I32_IU8` nor `V_DOT8_I32_IU4` has a `V_DUAL_` form |
| DP4A is available and used | `common.cuh:737` | **grounded** — `V_DOT4_I32_IU8` exists; code uses `__builtin_amdgcn_sudot4` |

## Corrected by this audit

**`isa_query.py` reported VOPD wrongly**, and I had repeated its answer. RDNA 3.5
carries dual-issue on **separate mnemonics**: `V_AND_B32` has no VOPD encoding,
but `V_DUAL_AND_B32` does. Reporting only the base instruction's encodings said
"VOPD dual-issue: NO" about operations that pair perfectly well. There are 17
`V_DUAL_*` instructions; the tool now resolves the counterpart and names it.

Consequence: my earlier statement that **`V_FMAC_F32` cannot dual-issue was
wrong** — `V_DUAL_FMAC_F32` exists. The conclusions about the dot instructions
and `V_FMA_MIX_F32` survive, because those genuinely have no dual form.

## Refined: "VOPD is unreachable from inline asm"

The ISA grounds half of this. `V_DUAL_AND_B32` encodes as **VOPDXY** with Y-slot
operands (`VDSTY`, `SRCY0`, `VSRCY1`), i.e. the encoding *is* a pair — a lone
dual-issue op cannot be encoded at all, only an X/Y pair can.

What the ISA cannot settle is whether the assembler accepts pair syntax from
inline asm. That is a toolchain question. The accurate form of the claim is:
**the ISA requires VOPD to be emitted as a pair, so a single dual op is not
expressible; whether our inline asm path can express a pair is a separate
question about LLVM, not about the hardware.**

## Not answerable from this spec — needs another source

* **"gfx1151 exposes 64 KiB of LDS per workgroup"** (`deepseek4_internal.h:57`).
  LDS appears in the spec only in `DS_*` instruction descriptions; capacity is an
  architecture property. Cite the RDNA 3.5 architecture doc, not the ISA.
* **"compiles to a `v_cmpx` + `s_and_saveexec_b32` / `s_or_b32` pair"**
  (`rocmfp4_hip_scale.cuh:53`). The instructions exist, but what the compiler
  emits is not an ISA claim. Ground it with disassembly of the built object.
* **"exec-mask writes break VOPD dual-issue"** (`:55`). Plausible and consistent
  with VOPD requiring two independent VALU ops, but the pairing constraints are
  not expressed in the encoding tables. Needs the architecture doc.

## Rule this audit suggests

When writing a hardware claim in a comment, say which source settles it. If the
answer is the ISA, `isa <INSN>` should reproduce it in one command. If it is not
the ISA — latency, capacity, occupancy, codegen — say so, so the next reader
does not go looking in the wrong document and conclude the claim is unfounded.
