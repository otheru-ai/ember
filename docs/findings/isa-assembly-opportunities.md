# ISA review: where assembly (and one non-assembly change) could still pay

Read against the gfx1151 spec mirror at `/srv/isa`, 2026-09-02, focused on the
kernels that run for the shipped recipe: `Q2_0_ROCMFP2` (129 expert tensors, the
largest in the model) and `Q4_0_ROCMFP4_FAST`.

## Already done — do not redo

* **DP4A is in use.** `ggml_cuda_dp4a` maps to `__builtin_amdgcn_sudot4` on
  RDNA3, i.e. `V_DOT4_I32_IU8`. Both quantized vec_dots use it.
* **The 2-bit LUT gather is already one instruction.**
  `rocmfpx_pack4_fp2_bits8_vec_cuda` uses `__builtin_amdgcn_perm`
  (`V_PERM_B32`) for the code-to-value lookup.
* **`V_DOT8_I32_IU4` is already implemented** — behind `GGML_ROCMI4_W4A8_IU4` /
  `DFLASH_ROCMI4_W4A8_IU4` in `mmq.cu`. It serves type 108 `Q4_0_ROCMI4`, which
  the DeepSeek recipe does not use. The instruction exists on gfx1151 (VOP3P, no
  DPP, no VOPD) and the code exists; only the type is unshipped.

## The real opportunity is not assembly — half the dp4a is redundant

`vec_dot_rocmfpx_fp2_q8_1` issues **two** dp4a per iteration:

```c
sumi = ggml_cuda_dp4a(val_packed,  u, sumi);   // weights . activations
sumq = ggml_cuda_dp4a(0x01010101, u, sumq);    // sum of activations
```

Per 32-element block that is 16 dp4a, of which **8 exist only to sum the
activations** for the affine offset term (`value = code*scale - offset`).

`block_q8_1` already carries that sum. `quantize.cu` stores
`ds = make_half2(d, sum)`, and the struct documents `s` as `d * sum(qs[i])`.
Both `iqs` halves of a rocmfp2 block read the *same* `bq8_1` and the *same*
`e[0]`/`e[1]`, so summed over the two calls the offset contribution is
`offset * d * sum(q)` — which is `offset * __high2float(ds)`.

**So the offset term can be applied once per block from an existing field, and
all 8 `sumq` dp4a deleted: a 50% cut in dp4a issue for the largest tensors in
the model.**

*Implemented 2026-09-02 behind `ROCMFP2_OFFSET_FROM_DS` (default off), as
`-0.5*offset*ds.y` on each half so no `iqs` branch is needed; the pairing
question below is answered by `mmvq.cu` (`kqs = vdr*(tid % (qi/vdr))`, both
halves of every block always reduce into one row). `test_rocmfp2_offset.c`
pins the per-block deviation bound. Awaits the behavioural gate on the box.*

Shape of the change: subtract `offset * __high2float(bq8_1->ds)` on the
`iqs == 0` call only, subtract nothing on `iqs == 1`, drop `sumq` entirely.

Two things to settle before trusting it, neither of which I can do off-box:

1. **Numerics.** `ds.y` is the float sum of the original values, while `sumq` is
   the integer sum of the quantised codes. They agree only up to rounding, so
   this is a small numerical change, not a pure refactor. It needs the
   behavioural gate, not just a build.
2. **Call-pairing.** The saving assumes both halves of every block are always
   evaluated and accumulated. If any path evaluates one half alone the offset
   would be wrong. Confirm in the MMVQ driver before landing.

## Tested and rejected

Building the `V_PERM_B32` selector word costs ~11 ALU ops (4 shifts, 4 masks,
3 ORs). A single-multiply bit-spread, `(bits8 * 0x00041041) & 0x03030303`, would
replace all of it. **It does not work**: 96 of 256 byte values disagree, because
the shifted copies carry into one another. Recorded so nobody re-derives it.

## What the ISA says about the remaining ideas

* `V_DOT2_F32_F16` and `V_DOT2_F32_BF16` are VOP3P **and DPP16-capable** — worth
  knowing if any F16 reduction is still scalar.
* Neither `V_DOT4_I32_IU8` nor `V_DOT8_I32_IU4` has a VOPD encoding, so they
  cannot be dual-issued. Consistent with the standing finding that VOPD is
  unreachable from inline asm, and it means dual-issue is not a lever on the
  quantized dot path specifically.


# Addendum: the per-layer causal mask rebuild (2026-09-02)

`build_mla_attention` allocates and zero-fills
`std::vector<float> mvals(n_attn * n_tokens)` and rebuilds the causal mask on
**every call, and it is called once per layer**. The fused path already avoids
this — it pre-builds `fg.mask_bundle` once and hands each layer a view — but the
`layer_major_batch` prefill path does not.

Sizing: at 2048 tokens and n_attn 4096 that is a 32 MB allocate-and-zero per
layer, roughly **1.4 GB of memset per chunk** across 43 layers, on top of the
predicate cost removed in `f4bba5a`.

## My first idea was wrong

I proposed hoisting the mask out of the layer loop as layer-invariant. **It is
not.** `const int ratio = w.compress_ratios[layer_idx]` — the compressor ratio
is per layer, and it feeds `n_comp_live`, hence `n_attn`, hence the mask shape
and contents. A blanket hoist would produce a wrong mask for most layers.

## What is actually available

The mask depends on the layer *only through* `ratio`. The shipped model has
**three distinct ratios across 43 layers**:

    deepseek4.attention.compress_ratios = [0, 0, 4, 128, 4, 128, ...]
    distinct {0, 4, 128}, counts {0: 2, 4: 21, 128: 20}

So the mask can be **memoised by ratio: 43 builds become 3**, provided layers
sharing a ratio also share `lc.n_comp` and `comp_kv->ne[1]`. Both are per-layer
fields, so that is an assumption to verify in the cache, not to assume — it is
the same class of mistake as the hoist above.

`n_prior_rows = min(kv_start, w.n_swa)` and `n_raw` on this path are
layer-independent, so they are not obstacles.

Order of work: verify the shared-ratio assumption first, since getting it wrong
is a correctness bug rather than a slower kernel. Then memoise.

## Second pass (2026-09-02): read the compiler's output, not the ISA index

Method: compile the hot TUs exactly as `build-frontier-rocm/compile_commands.json`
does (`-O3 -DNDEBUG --offload-arch=gfx1151`) with `-S --offload-device-only`
inside `ember-rocm:10.0-dev`, then count instruction classes per kernel. The
kernels are the retained-trace top of prefill (`ds4_flash_attn_d512_*`, 30 %
of prefill kernel time) and decode (`mmvq<101>` 24 %, `mmvq<107>` 18 %).
Every hardware claim below is either the ISA XML (`isa <INSN>` on otheru) or
the emitted assembly; where the XML does not carry a rule, that is said.

### 1. FA D=512: the q rope tail was a flat-load tail — fixed

Before this pass every D=512 attention kernel selected, per element, between
an LDS pointer (rotated q tail) and a global pointer (`qh[d]`). Mixed address
spaces lower to generic `flat_load_b32`, and the compiler could not hoist the
selection. Per kv row, HEADS=4 compact kernel (`fa512_f_h.s`):

| | before | after |
|---|---|---|
| instructions | 7286 | 4576 |
| `flat_load_b32` (each behind `s_waitcnt vmcnt(0) lgkmcnt(0)`) | 258 | 0 |
| `v_readlane_b32` / `v_writelane_b32` (pointer spills into lanes) | 541 / 529 | 93 / 52 |
| scratch bytes ("Folded Reload") | 36 | 0 |
| `s_cselect_b32` | 552 | 40 |
| `s_waitcnt` | 407 | 175 |
| `s_load_b512` (q via scalar cache) | 113 (448 dims) | 129 (all 512) |
| VGPRs / occupancy | 50 / 16 | 42 / 16 |

Fix: `ds4_forward_rope_q_tail_kernel` rotates the tail for every (token, head)
into a `[n_tokens][n_heads][64]` pool buffer before the attention launch; the
kernels select between two *global* wave-uniform pointers, so every q read is
an `s_load_b512` feeding `v_fmac_f32`. Numerics unchanged: same
`ds4_forward_rope_coefficients` and `ds4_apply_inverse_rope_pair`, same F32
coefficient rounding, same per-head accumulation order.

Two variants were tried and rejected, with the counts that rejected them:
staging the tail in LDS unconditionally (removes flat loads but LICM hoists
256 LDS reads into VGPRs: 96 VGPRs, scratch 32 B, `v_dual_fmac` 844 → 60);
adding an `asm volatile("" ::: "memory")` per row to stop the hoist (VOPD
back to 367 but scratch 68 B). Skewing odd heads by one dimension to help the
VOPD pairer made the loop 22 % longer with 285 readlanes.

What remains: `v_dual_fmac_f32` pairs 338 (before: 844, but those paired the
now-deleted tail work too). The XML settles one VOPD rule — `VDSTY` bit 0 is
`~VDSTX[0]`, so paired destinations need opposite parity — and does not carry
a source-bank rule; the compiler only pairs FMAs that read *different* k
VGPRs. The dot loop reads the same `k[d]` for four heads back to back, so the
adjacent-only pairer finds one pair in three. This is compiler scheduling, not
source order (the skew experiment shows the scheduler undoes source order),
so the remaining lever is inline-asm for the 512×4 FMA block — which loses
VOPD entirely ([[ember-gfx1151-kernel-findings]]) — or the WMMA path, already
judged not the win. Leave it.

Falsifier for the perf claim: FA kernel p50 per dispatch (trace, not tok/s)
unchanged after the change means the flat-load latency was hidden by
occupancy 16 and this is a code-size fix only. Expect it to move: the 258
full-drain waits per row were serialising the global k loads.

### 2. UE4M3 scale decode: 9–11 VALU ops → 2 + one convert

`rocmfp4_ue4m3_to_fp32_half_finite` / `_finite` (`rocmfp4_hip_scale.cuh:37-82`)
emit ~9/~11 VALU ops per decode. A UE4M3 byte `x` (bias 7, no sign) is a
subnormal-capable 8-bit float; placing its 7 magnitude bits at the f16
exponent/mantissa boundary and converting is the same function up to a power
of two:

    cvt_f32_f16((x << 7) & 0x3f80) * 128.0f

is bit-identical to both decoders for all 256 inputs (Python enumeration of
the C reference, 0 mismatches; for `x = 0x7f` it yields 240.0, the
`half_finite` value — the `_finite` variant's `x > 0x7e → 0` clamp relies on
the documented pre-validation of scale bytes, so keep that clamp where the
bytes are not validated). Instructions: `v_lshlrev_b32`, `v_and_b32`,
`v_cvt_f32_f16` (`isa V_CVT_F32_F16`: VOP1/VOP3, DPP16 yes), and the ×128 can
fold into the kernel epilogue or `v_ldexp_f32`. *Implemented 2026-09-02
behind `ROCMFP4_SCALE_DECODE_F16` (ember `b307ddf`), bit-identical on all 256
bytes by host test; the kernel-descriptor denormal check and the trace A/B
are with codex.* Where it lands: FP4_FAST mmvq
spends ~9 of ~100 VALU per block on the decode, FP2 mmvq 2×11 of ~66 per
half-block, and the mmq loaders decode per tile. Unimplemented; needs
`test_rocmfp_scale.c` extended to all 256 bytes against the reference.

### 3. FP2 unpack: an identity perm and a layout that would delete the spread

`rocmfpx_pack4_fp2_bits8_vec_cuda` (`vecdotq.cuh:440-461`) ends in
`__builtin_amdgcn_perm(0, 0x03020100, sel)` — with
`ROCMFP2_KVALUE_{0..3}_I8 = 0,1,2,3` (`rocmfpx.h:28-31`) the table is the
identity, so it is a no-op `v_perm_b32` (4 per half-block). Remove it.
*Removed 2026-09-02 (ember `86d8883`), `test_rocmfp2_unpack.c`.*

The 2-bit → byte spread itself costs ~32 ops per 16 codes. With a transposed
storage order (byte j of a 4-byte group holds elements j, j+4, j+8, j+12),
word k of the unpacked bytes is `(x >> 2k) & 0x03030303` — 7 ops for all
four words. It is a lossless bit permutation of the existing type, so it can
be a load-time repack rather than a new GGUF type, and the mmq loader
benefits equally. Unimplemented; pairs naturally with the pending `sumq`
elimination above.

### 4. Already near-optimal, do not spend time here

FP4 codebook gather (2 table perms + 1 select per operand), DPP wave
reductions (`v_add_f32_dpp` + `v_permlanex16`), and `V_DOT4_I32_IU8` use are
what the ISA allows (`isa V_DOT4_I32_IU8`: VOP3P, no DPP16, no VOPD;
`isa V_DOT8_I32_IU4`: VOP3P, no DPP/VOPD — the IU4 route is the existing
ROCMI4 W4A8 design, not a quick win). `V_CVT_F32_FP8` / `V_CVT_PK_F32_FP8`
are not in the rdna3_5 spec.

### 3b. FP2 transposed layout: design, hook points, and why it is not started

Written 2026-09-02 after `86d8883`/`912459f` landed. The layout change in §3
is a load-time repack, so every reader of the type must switch under one
compile-time macro (proposed `ROCMFP2_TRANSPOSED_LAYOUT`, default 0) or the
no-discriminator trap of the two-scale/affine slot 107 repeats:

| reader | where | note |
|---|---|---|
| CPU dequant (`to_float`) | `engine/ggml/rocmfpx/rocmfpx.c:722` `rocmfpx_dequantize_row_fp2` | the only CPU reader; `ggml.c:748` traits carry no CPU vec_dot, so the hybrid cold-expert path (`DFLASH_MOE_COLD_BACKEND`) dequantises through this |
| device dequant | `ggml-cuda/dequantize.cuh` `dequantize_rocmfpx_fp2` | used by `convert.cu:892-1069` and `getrows.cu:210` |
| MMVQ | `vecdotq.cuh` `rocmfpx_pack4_fp2_bits8_vec_cuda` + `vec_dot_rocmfpx_fp2_q8_1` | the ~28→7 op win |
| MMQ prefill loader | `mmq.cuh:594-595`, `:1036` (`load_tiles_rocmfp2_affine`) | same unpack, same win |
| validator | `rocmfpx.c:1032` | scale bytes only, layout-agnostic |
| quantizer | `rocmfpx.c:666-720`, `test_rocmfpx.c` | must stay canonical (on-disk) — do not emit the transposed order |

Repack hook: `deepseek4_loader.cpp:756-793` — in the `fast_managed` worker,
after the `pread` loop for tensor `a`, `if (a.tensor->type ==
GGML_TYPE_Q2_0_ROCMFP2) repack_in_place(dst, a.file_size)`; the pages are
host-writable there (pread just wrote them). The `ggml_backend_tensor_set`
fallback (`:797-800`) reads the read-only mmap, so it needs a staging copy
for FP2 tensors. The repack is a 4×4 transpose of 2-bit fields inside each
4-byte `qs` word — pure host code, unit-testable against the reference
decoder like `test_rocmfp2_unpack.c`.

Why not started: the win is ALU, and whether the FP2 MMVQ is VALU-bound is
not established. Per 4-byte weight word the unpack is ~28 VALU ops against
4 dot ops; at batch 1 that is ~8 ops per weight byte, ~20 % of the CU VALU
rate at DRAM bandwidth, i.e. bandwidth-bound and the repack would be
invisible. For the routed experts the vec_dot runs once per (column, row)
with a different expert per column (`mmvq.cu:525-528` `channel_xs[j]`), so
the unpack cannot be shared across columns and the ratio is unchanged at
verify shapes. Only the non-`ids` FP2 matmuls (shared expert, dense
projections if FP2) multiply the unpack by `ncols_dst`; those are where a
`reuse_rocmfp4_weights`-style variant (`mmvq.cu:508`, FP4FAST only today)
and the transposed layout would both pay.

Pre-registered discriminator before any implementation: from the kernel
trace codex already collects for 940, VALU-busy vs memory-unit-busy on the
FP2 `mul_mat_vec_q` instances (counter names to be confirmed on gfx1151 by
codex; the ISA XML describes instructions, not counters). Implement only if
an FP2 MMVQ instance that carries ≥10 % of decode kernel time shows VALU
busy above memory busy; otherwise close this item as "bandwidth-bound,
not admissible" and leave §3's estimate as an estimate.

### 5. FA D=512 q-rope-tail prepass: measured, and it is a regression

`f5ad83f` (candidate `d466a93` = the patch transplanted onto release
`adc8319`) rotated the forward-RoPE q tail once per call into a pool buffer so
the D=512 kernels select between two uniform global pointers instead of
building the tail in LDS and selecting per element. The compile-time deltas
were large and real — exact gfx1151 TU, `<float,__half,4,true,4>`: 7286 → 4576
instructions, 258 → 0 flat loads, 407 → 175 `s_waitcnt`, 17 → 0 scratch.

Measured 2026-09-02 on gfx1151, one sparse depth-4096 request per arm, 256
tokens, speculation on, `rocprofv3 --kernel-trace`, production quiesced
(`/srv/models/perf/fattn-q-tail-ab-d466a93-20260902`, claude, unreviewed):

| | base `adc8319` | cand `d466a93` |
|---|---:|---:|
| prefill tok/s | 329.4 | 305.8 |
| decode tok/s | 22.34 | 22.28 |
| acceptance | 0.976 | 0.981 |
| total kernel time | 16.567 s | 17.447 s (+5.3 %) |
| `compact<…,true,4>` prefill, grid 447744, n=21 | p50 **63.80 ms** | p50 **84.73 ms** (+32.8 %) |
| `ds4_forward_rope_q_tail_kernel` | absent | 129 dispatches, 112 ms |

The prefill row is the clean comparison: same kernel, same count, same grid,
p50 up by a third. `docs/findings` msg 946 pre-registered "FA kernel p50
unchanged ⇒ code-size fix only"; p50 did not stay unchanged, it got worse, so
the change is **rejected**, not merged.

Mechanism, stated as far as it is established and no further: the tail was
built once per block in LDS and reused across every kv row; it is now read
from a global pool buffer inside the same loop. The instruction count fell
because the addressing became uniform, but the bytes still have to arrive, and
on this kernel they are the binding constraint. That is consistent with the
number and is not proven by it — nobody has read a memory counter here (see
§3b: no PMC pass is scheduled).

Not established, and named so it is not silently reused: the two arms also
differ in decode-phase kernel *selection* (21 dispatches that are
`compact<…,true,4>` at grid 32768 in the baseline appear as
`shared_kv_grouped_kernel<float,__half,4>` in the candidate). Acceptance also
differed (0.976 vs 0.981), so the decode dispatch mix is not matched between
the arms and no decode-phase claim can be made from this run. The LDS terms
the patch removes (`group4 * 64 * sizeof(float)`) do sit inside the 24 KiB
guards at `fattn.cu:2004/2022`, so a selection flip is plausible — but that is
a hypothesis, and the run does not test it.

What this does not say: the removed flat loads were real, and a version that
keeps the tail in LDS while dropping the per-element select may still win.
That is a different patch and needs its own run.
