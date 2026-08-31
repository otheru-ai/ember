# Review: `LaurentZuijdwijk/llama.cpp` (Strix Halo Vulkan fork)

Reviewed 2026-08-31 from the public repository. Same silicon as ours (Radeon
8060S / Strix Halo), overlapping model family (`qwen4exp`, PLE, MTP, QSA), and
an overlapping quant lineage (ROCmFPX). **Vulkan on stock Mesa RADV, not
HIP/ROCm** — which decides what is portable and what is not.

## Assessment of the work itself

The benchmarking is more careful than most published forks, and the
methodology is worth adopting regardless of what we port:

- arms interleaved in **palindrome order** (fork · mainline · mainline · fork)
  so clock drift cancels rather than being charged to one arm;
- a **discarded warmup** per process, because the first run of a set on this
  APU gets a 15-20% boost clock — the same trap our own harness must avoid;
- cells under 2σ explicitly marked "noise" rather than reported;
- two independent sessions on the headline cell, quoting **the lower**;
- a stated thermal caveat: the speculative headline was a 79 °C / 115 W burst
  the chassis cannot sustain, and the everyday profile reads 55.0 t/s not 65.6.

It also states where it does **not** win (generation on stock K-quants is flat
within ±1%; on the MoE the generation win is upstream's, not theirs). A fork
that documents its own null results is one whose positive results are worth
more.

## Findings relevant to Ember

### 1. Batch 3-8 is the underserved regime — independently identified

> "Speculative decoding lives at batch 3-8 … the Vulkan mat-vec kernels fall
> apart in exactly that range … 5× slower at the width the method needs …
> nobody noticed because single-token decode never reaches that width."

They found a register-spill cliff at `NUM_COLS > 4`. We have a **correctness**
failure whose boundary is width 5→6. Different backend, different symptom, same
structural cause: the widths speculation needs are the widths nobody exercises.
Convergent evidence that this region is systematically under-tested, and
support for treating our width-6 red as a real defect rather than an
artefact.

### 2. ROCMI4 (type 108) has no external implementation — our oracle is self-referential

Their hand-port covers types **100, 101, 102, 103, 104, 107**. It does **not**
cover **108 (`Q4_0_ROCMI4`)**, which is the type our shipped checkpoint uses for
all 834 quantized tensors.

This matters for the open blocker. Our operator oracle validates the GPU
decoders against **our own CPU decoder**, so it proves *implementation
agreement*, not *format correctness*. A defect shared by both sides is invisible
to it — and for ROCMI4 there is no independent implementation anywhere to
compare against.

Their validation avoided that trap by building the source fork and using it as
an external oracle (quantizer SHA256 bit-identity, character-identical model
output, 857 `test-backend-ops` cases). We cannot do the same for 108.

**Their two fixed decode bugs do not hit us**: both are fp6
(`Q6_0_ROCMFPX`, 102) — a Vulkan layout mismatch, and a CPU `vec_dot` decoding
`sign|0` as `-0` instead of `-32` (~2% error on every fp6 mat-vec). Our shipped
mix is ROCMI4 + F32 + BF16, with `Q3_0_ROCMFPX` (104) admitted only for the PLE
row table. No fp6.

The transferable lesson is the *shape* of bug 2: two implementations of the same
decode inside one codebase disagreeing. We have three device decoders plus a CPU
row decoder for ROCMI4; the oracle now pins them together, which is exactly the
right guard — but it cannot speak to whether the shared answer is correct.

### 3. Portable, in descending order of value

| item | portability | note |
|---|---|---|
| **Adaptive speculative decoding** (`--spec-draft-adaptive`) | **high — algorithmic** | Draft length tracks measured acceptance; `n_max` is a ceiling. Their data: fixed n=7 collapses to 18% acceptance, fixed n=3 sits at 95% and under-drafts, adaptive holds 96% while drafting longer — 4.7× on structured output. Backend-independent. Our own evidence shows accept_rate flipping 1.0 → 0.0 across widths, so a fixed draft length is demonstrably wrong for us too. |
| **Wide-ubatch MoE insight** | medium — conceptual | Mainline at ubatch 2048 is *slower* than at 512 (870 vs 1144 t/s); their tiled concat-transpose plus the `mul_mat_id` stack turns that regression into their largest win (+89% at pp2048). We have a `ggml_concat` in the GDN path and a MoE `mul_mat_id` stack. The kernels are Vulkan, but "wide ubatch regresses unless concat/transpose is tiled" is a claim we can test on HIP. |
| **f16 B operand for `mul_mat_id` / `mul_mat`** | medium | +6.9-9.1% MoE, +4.5-5.8% dense in their gates. |
| **LDS stride pad** (`SHMEM_STRIDE`) | low — Vulkan coopmat specific | A four-way bank conflict costing 17-18% at kernel level on every non-Intel device. Their claim, not measured by us. The *class* (LDS bank conflict from an unpadded stride) is worth checking in our own hand-written kernels. |
| Vulkan shader fixes (IQ3_S spill, ROCmFPx batch 3-8) | none | Different backend. |

### 4. Numbers are not directly comparable to our gates

Their prefill headline is **440 t/s at pp512 on Qwen3.8-27B**, and 1648 t/s at
ubatch 2048 on a 35B-A3B MoE. Our target is Qwen3.8-Flash-Next at
prefill ~345 peak / decode 23.6-23.8 AR, with gates at
`decode_256_median_tps 39.49` and `prefill_2074_peak_tps 412.0`. Different
models, different context depths, different backend. **Do not quote their
figures as a benchmark for ours** — the transferable content is the mechanisms
and the methodology, not the numbers.

## The bar, after the 2026-08-31 goal change

The user directed that Ember must **exceed** this fork. Their published figures,
with conditions, from `bench/RESULTS.md` (generated from JSONL by script, not
hand-edited):

| model | test | depth | fork t/s | mainline t/s |
|---|---|---:|---:|---:|
| **ornith-35b-a3b-q4km** (MoE, ~3B active) | pp2048 `-ub 512` | 0 | **964.7 ± 4.2** | 856.8 |
| | pp2048 | 4096 | 845.5 | 768.6 |
| | pp2048 | 65536 | 376.5 | 379.9 (noise) |
| | tg64 | 0 | **65.8 ± 0.0** | 65.5 |
| | tg64 | 65536 | 45.0 | 45.2 (noise) |
| qwen38-27b-q4kxl (dense 27B) | pp2048 | 0 | 288.7 | 255.6 |
| | tg64 | 0 | 11.6 | 11.7 |
| ornith-35b-a3b-q4km | pp2048 `-ub 2048` | 0 | **1648.5** | 870.5 |

**The comparable row is the MoE one.** Qwen3.8-Flash-Next is MoE with roughly
3B active parameters (512 experts, top-10, `n_embd` 2560, expert intermediate
640), which is the same activation scale as Ornith-1.5-35B-A3B. The dense 27B
row is not our shape.

### Against our current gates

| | our gate | fork, nearest comparable | ratio |
|---|---:|---:|---:|
| prefill | `prefill_2074_peak_tps` **412.0** | 964.7 (pp2048, depth 0, ub 512) | **2.34x** |
| prefill (wide ubatch) | — | 1648.5 (ub 2048) | **4.0x** |
| decode | `decode_256_median_tps` **39.49** | 65.8 (tg64, depth 0) | **1.67x** |

The superseded DeepSeek-parity target (prefill ~345, decode 23.6-23.8 AR) sits
**below our own gates**, so it is now a floor rather than a goal.

### What this comparison does and does not license

**Does not**: their numbers are on *different models*. Beating 964.7 on
Qwen3.8-Flash-Next would not prove we beat their engine — it would prove our
model is cheaper, or that we measured differently. Their absolute t/s are also
power-profile dependent by their own statement, and their MoE prefill win is
concentrated at wide ubatch where mainline *regresses*.

**Does**: set a defensible bar in the absence of a head-to-head. Until one
exists, treat **prefill ≥ 965 t/s at pp2048 depth 0** and **decode ≥ 66 t/s
tg64 depth 0** as the target on our own model, measured the way they measure —
palindrome-ordered arms, discarded warmup, per-cell σ, depth attached to every
generation figure, and the power profile recorded.

**What would settle it**: the same model on both engines on the same box. That
fork has `qwen4exp/*` and `vulkan/qwen4exp-rocmfpx` branches in flight, so a
head-to-head on Qwen3.8-Flash-Next may become possible. That, not a number
comparison across models, is the measurement to want.

**Precondition**: no valid Qwen performance number exists on our side while the
correctness blocker is open, so the gap above is a target, not a measured
deficit.

## Licensing / provenance

MIT (llama.cpp), and their ROCmFPX port is from the MIT `ciru-ai/ROCmFPX` —
the same lineage `engine/VENDOR.md:55` already records us auditing against.
Porting an idea is unencumbered; porting code would need a `VENDOR.md` entry as
usual.

## Recommendation

1. **Adaptive speculation is the item to take.** It is algorithmic, matches a
   deficiency our own evidence already shows, and targets the decode half of
   the goal, which no current work stream addresses.
2. **Record the ROCMI4 oracle limitation** in the blocker's ledger entry: the
   operator oracle proves implementation agreement, not format correctness, and
   for type 108 no external oracle exists. This does not weaken any result we
   have; it bounds what those results license.
3. Do not port Vulkan kernels. Retest the wide-ubatch concat claim on HIP
   independently if prefill work resumes.
