# Qwen3.8-Flash-Next performance status

Measured status against the parity target. This is a ledger, not a claim: every
row states the exact commit, recipe, and what invalidates it. Update it when a
measurement lands; do not delete superseded rows, mark them.

Recording methodology is `docs/qwen3.8-performance-baseline.md`. The target
figures come from `docs/performance.md` (DeepSeek-V4-Flash on the same
gfx1151 host).

## Where this stands — read this first

This file is 1100 lines and grows by measurement. The state as of
2026-08-31, with pointers rather than repetition:

**Correctness.** Two causes were found, one fixed, one awaiting a decision.

- `sum_rows` selected its reduction tree from the row count, so q1 and batched
  could not agree. **Fixed** (`9f1dc33`), twin in `mean.cu` fixed (`86a5ce1`),
  guarded by `test_sum_rows_shape_invariance` (`f021309`), and screened against
  the shared DeepSeek path with an interleaved A/B that came back flat. Widths
  2, 3, 4 and 5 are now validator-green. → *Open correctness blocker — ROOT
  CAUSE FOUND*
- Widths 6 and 17 remain red, and every width that fails contains physical-16
  **MMQ** work while every width that passes stays on **MMVQ**. Five of five
  follow that boundary. **Decided 2026-08-31**: prefill is judged by a margin
  criterion rather than bit-identity, with MTP's q1 replay unchanged. Widths 6
  and 17 are to be re-run under it. → *DECIDED: prefill uses a margin
  criterion*

**Performance.** No publishable number exists yet, and none may be published
while the above is open.

- Best valid measurement: prefill peak **39.40** against a 412 gate, decode
  median **12.13** against 39.49 (`faa5307`, hard gate, exact binary).
- The gap is architectural, not a missing kernel: a working implementation on
  this same silicon reaches **345 prefill** by building **one graph** for the
  whole model. Ours runs 12 host barriers per layer group at 15.6% GPU busy.
  → *The 345-prefill reference implementation*
- Tranches 1, 2 and 3 are each mapped onto that reference rather than inferred.
  Order: 1, 2, then 3 — the indexer scorer does not execute at our
  certification widths at all.
- Sized but deliberately **not** promoted to a lever: asymmetric KV cache
  quantization, ~101 MB of upload per decode token at ctx 2048.
  → *Candidate, unsized*

**Expect this.** Most published numbers on this part sit at 22.6-28.1 decode
and 345-385 prefill, below our 39.49 and 412 gates, so the first green
measurement will very likely be a real result *and* short of target.

**Correction 2026-08-31**: an earlier version of this line said our gates sit
above *every* published number. They do not.
`agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF` claims **423 t/s
prefill** and **up to 40 tok/s** decode with adaptive drafting on Strix Halo —
both above our gates. The caveat is real and load-bearing: 423 is measured at
**512 tokens** where our gate is at **2074**, and prefill falls with length on
every ladder observed (that model's own drops to 138 at 128k). Not a
like-for-like refutation, but "nobody has exceeded our gates" was too strong.
→ *What the first publishable number requires*,
[`qwen3.8-external-gguf-compatibility.md`](qwen3.8-external-gguf-compatibility.md)

## Target

| metric | target | source |
|---|---|---|
| prefill peak | ~345 tok/s | `docs/performance.md`, spec-off sweep 73.4/300.6/**345.4**/290.9/264.4/220.1/182.0 |
| decode AR | 23.6-23.8 tok/s | `docs/performance.md`, flat across prompt length on unpredictable prose |
| decode structured | ~39.6 tok/s | 2026.8.24 engineering bundle; a different result class, do not conflate |

The hard gates encoded in the certification workflow are `decode_256_median_tps
39.49` and `prefill_2074_peak_tps 412.0`.

## Measurements

### 1. Run 33289399556 @ `c5cb7a2` — SUPERSEDED, do not cite

| metric | measured | gate | ratio |
|---|---|---|---|
| decode 256 median | 4.498 tok/s | 39.49 | 0.114 |
| prefill 2048 median | 24.756 tok/s | 412.0 | 0.060 |
| prefill 2048 peak | 24.896 tok/s | 412.0 | 0.060 |

`hardware_certified: false`, scope `measurement_only_not_certified`,
3 samples, `shape_match: true`.

**Why it is not usable as a baseline — two independent reasons:**

1. **Wrong recipe for a performance claim.** Quant recipe was
   `rocmfp4-fast-matrix-q3-ple-q6k-embedding-head`, which
   `scripts/qwen_real_weight_gate.sh:311-319` maps to
   `no_eligible_rocmi4_mmq`. It is a Q3-PLE first-token proof candidate, not
   the performance candidate. Three `rocmi4-*` recipes map to
   `rocmi4_dense_and_routed` and would exercise the quantized MMQ path.
2. **Predates the correctness validator.** `validation_compare_production_prefill`
   landed at `4b7213c`, after this run. The engine may already have been
   numerically divergent when this throughput was recorded.

**There is currently no trustworthy Qwen performance number.**

### 2. Diagnostic timing @ `a3a50c4`, `LUCE_MMVQ_MAX_NCOLS=5` — NOT PUBLISHABLE

| metric | measured | gate | short by |
|---|---|---|---|
| prefill 2074 median | 37.366 tok/s | — | — |
| prefill 2074 peak | 38.055 tok/s | 412.0 | **10.83x** |
| decode 256 median | 11.757 tok/s | 39.49 | **3.36x** |

MTP depth 3, accept rate 0.767, three exact-shape samples, shape-match and
rounding checks true. Clean timing pass only, no counters.

Improvement over measurement 1: prefill peak **1.53x**, decode median
**2.61x**. Attributable to the fusion/batching commits between `c5cb7a2` and
`a3a50c4` plus the raised MMVQ threshold.

**Why it is not publishable:** the `ncols5` multi-width differential still
fails at prompt widths 3, 6 and 17. Correctness is not established, so this
number describes an engine we would not ship.

### 3. Hard gate @ `faa5307`, `LUCE_MMVQ_MAX_NCOLS=5` — valid, and failing

The first complete unprofiled hard-gate run on an exact binary (codex 215,
evidence `hardgate-timing-faa5307-ncols5-20260830T191000Z/`).

| | measured | gate |
|---|---|---|
| prefill median | 39.1807 | — |
| prefill peak | **39.4019** | 412.0 |
| decode median | **12.1333** | 39.49 |

Prefill samples 39.0978 / 39.1807 / 39.4019, decode 12.1571 / 12.1333 / 12.1138.
Every prefill sample evaluated exactly 2074 tokens and every decode sample
completed exactly 256; all declared-rate rounding checks true. MTP acceptance
0.767 on every retained decode sample. Memory gate passed: peak RSS 68.88 GB,
peak UMA 76.21 GB. Binary SHA-256
`f56b9e2bdf931f486082290813fe1bfd89fc72a79e27bd7f5d5d7b72b36d9e51`. Production
restored, GPU lock free.

**This is a valid measurement of a failing configuration, not a publishable
number** — the correctness blocker is open, so what it measures is not yet a
correct engine. Prefill is 10.5x short of the gate and decode 3.3x short.

Against the earlier `a3a50c4` + ncols5 gate: median prefill +4.86%, peak
+3.54%, median decode +3.20%. Those runs were not interleaved, so that is not
sole attribution to `faa5307`; the dedicated ABBA probe isolated `faa5307` at
**+2.35%** on the calibrated 294-token workload.

That +2.35% is the async tranche's real value, and it settles the prediction
recorded above. A ~1.2x was expected and withdrawn on structural grounds — the
tranche converts 30 copies to async but removes **zero** barriers, and seven of
the fourteen groups hold a single copy where `get_async` immediately followed
by `synchronize` is the blocking copy it replaced. The measurement agrees with
the structure, not with the original guess.

## The 345-prefill reference implementation

`github.com/kingjones30/ROCmFPX` — llama.cpp plus a patch combining qwen4exp
with the ROCmFP4 tensor types. Its README claims **345 tok/s prefill, 22.6 tok/s
generation** for Qwen3.8-Flash-Next on gfx1151; the HF card
`kingjones777/Qwen3.8-Flash-Next-ROCmFP4-STRIX_LEAN-GGUF` gives ROCm 7.2.4,
`-DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_NATIVE=ON`, and
`--n-gpu-layers 999 --flash-attn on --fit off --ctx-size 131072 --threads 16`.
Quantization is comparable to ours: MoE at type 101, attention split 100/101,
PLE at Q5_1, head Q6_K, 4.78 bpw.

The graph builder is `src/models/qwen4exp.cpp`, 1193 lines, kept for reference
at [`docs/reference/qwen4exp_upstream.cpp`](reference/qwen4exp_upstream.cpp)
(MIT, not vendored, not built).

**It is one graph.** `llama_model_qwen4exp::build_arch_graph(const
llm_graph_params &)` at `:187` expands all 48 layers — GDN, QSA, MoE, PLE —
into a single `llm_graph_context` and dispatches once. Grepping the whole file
for host round trips returns **one** hit: an index-array
`ggml_backend_tensor_set` at `:1014`. There is no per-layer `tensor_get` and no
host barrier inside the layer loop. Recurrent state is read on device through
`build_rs(inp, ssm_states_all, ...)` (`:758`) and advanced on device with
`ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst))`
(`:1070`). Their q/k/v extraction from the conv output (`:775-790`) is
structurally the same as ours.

**So the 345 is not a kernel we lack or a flag we failed to set. It is the
absence of our host boundary** — 12 live barriers per layer group, 15.6% GPU
busy, 95.55% of long-tail idle attributable to late host submission.

That makes tranches 1-3 steps toward an implementation that exists and hits the
number, rather than speculative optimizations: tranche 1 maps to their q/k/v
path, tranche 2 to `build_rs` + `ggml_cpy`, tranche 3 to their device-side
cache write. Diff each against the reference rather than designing from
scratch, but do not copy wholesale — our runtime carries a snapshot and
rollback contract their graph does not.

**Caveat on the target.** Their ladder is 345 at ~3.3k tokens and 385 at ~7k
(fixed prompt, run 1 dropped, median of 3). Our 412 gate is above their entire
cluster, so matching this implementation reaches their band; it does not by
itself clear the gate.

## Gap decomposition

Residency headroom alone, if GPU busy went to 100% with no other change:

| | busy now | headroom | gap to close | covered |
|---|---|---|---|---|
| prefill | 13.9% | 7.19x | 10.83x | **66%** |
| decode | 32.4% | 3.09x | 3.36x | **92%** |

So the launch/synchronization problem accounts for most of the remaining gap,
and nearly all of it on decode. That is consistent with the copy attribution
(91.9% of 739,794 copy groups immediately precede `quantize_q8_1`) and argues
the contiguity fix is the highest-value remaining work, not kernel tuning.

Prefill needs roughly 1.5x beyond perfect residency; decode needs almost
nothing beyond it.

## External calibration — the target is achievable, others reach it

Independent published results for this model on gfx1151-class hardware:

| source | hardware | decode | note |
|---|---|---|---|
| llama.cpp PR 27842, Vulkan/RADV | Strix Halo gfx1151 | **25.2 tok/s** baseline, 38.7-48.7 with MTP n-max 3 | UD-IQ4_XS + Q8_0 draft, greedy |
| HF agentionai ROCmFP4-FAST MTP | Radeon 8060S (gfx1151) | **28.1 tok/s** baseline, 31.8-32.4 with MTP | temp 0 |

Extended set, all gfx1151 / Radeon 8060S, all publisher claims except PR 27842:

| source | backend | decode AR | decode w/ spec | prefill |
|---|---|---|---|---|
| llama.cpp PR 27842 | Vulkan/RADV | 25.2 | 38.7-48.7 (n-max 3) | not found |
| HF agentionai | Vulkan | 28.1 | 32.4 (n-max 3) | not found |
| HF EasiiX | EngramHalo.cpp | 23.5 | 35.7 (~85% accept) | not found |
| HF kingjones777 STRIX_LEAN | **HIP ROCmFPX fork** | 22.6-22.87 | not claimed | **345** short, 385 @ 6963 tok |

**Gate calibration.** Against that published cluster:

- prefill target ~345 is **parity** — kingjones777 reaches exactly that on the
  same part with a HIP ROCmFPX fork, the same family as our stack, so 345 is
  demonstrably achievable here and not a Vulkan-only result;
- the encoded prefill gate of **412 is ~1.07-1.19x above** the published
  345-385 cluster, i.e. slightly ambitious rather than parity;
- decode AR target 23.6-23.8 is parity or slightly conservative against the
  22.6-28.1 AR cluster;
- the encoded decode gate of 39.49 is an **MTP-band** number (PR 27842 n-max 3
  reaches 38.7-48.7), not an AR number. Our 11.757 was measured with MTP depth
  3 at accept 0.767, so it is the right comparison, but the AR and MTP bands
  should not be conflated when reporting.

Our 11.757 decode is ~2.1-2.4x below a stock Vulkan AR result on this part.
That is the gap: we are behind a published number, not chasing an unproven
ceiling.

## MTP acceptance is healthy, not a lever

Measured 0.767 at depth 3. Published working band for n-max=3 on this model is
**0.61-0.86** (PR 27842: 0.718 code / 0.652 prose / 0.918 list; agentionai
gfx1151: 0.612). PR 27842 also finds n-max 8 loses on gfx1151 on both
acceptance and rollback copies, and recommends 3 — which is what we run.

No tuning needed here. An accept rate of exactly 0, as seen in the failed run
`33320454087`, is a distinct wiring failure rather than a weak draft head, and
should be read that way if it recurs.

## Known bottleneck evidence (run 33289399556 profiler)

Not throughput, but the shape of the problem. Still believed current.

| | prefill (2074 tok) | decode (256 tok) |
|---|---|---|
| dispatches | 4,455,958 (2,148/token) | 2,746,132 (10,727/token) |
| `__amd_rocclr_copyBuffer` | 1,249,504 (602/tok) | 557,196 (2,177/tok) |
| `quantize_q8_1` | 1,206,107 (582/tok) | 584,108 (2,282/tok) |
| GPU busy / span | **13.9%** | **32.4%** |
| achieved bandwidth | 11.29 GB/s | 12.79 GB/s |

Roofline is 212 GB/s, so this is launch- and synchronization-bound, not
bandwidth-bound.

The `copyBuffer` row was annotated as an undercount, on the theory that
`cudaMemcpy2DAsync` lowers to a separately-named `__amd_rocclr_copyBufferRect`.
**That was checked and is false**: `copyBufferRect` appears zero times in the
trace. Every copy takes the 1D packed branch of `ggml_cuda_cpy_tensor_2d`
(`engine/ggml/src/ggml-cuda/ggml-cuda.cu:1478-1510`). The row is complete.

## Widths 2 and 3 are GREEN; width 6 is a different seam

Codex 365 on `86a5ce1`, with the `sum_rows` fix and its `mean.cu` twin landed:

- the HIP invariance guard passes — `backend=hip`, `first_diff_row=-1`
- widths 2 and 3 are validator-green
- at width 3 **every captured GDN layer is bit-exact**: output, convolved
  input, normalized Q and K, decay, beta, conv state, recurrent state
- **width 6 remains red**, as a real production-prefill mismatch — q1 sampled
  an immediate stop while batched produced a token

So `sum_rows` was the width-3 cause and is closed. Width 6 is a second seam.

### Width 6 is the first width whose dense path changes kernel *family*

`moe_cached_width` maps logical 1→1, 2→5, 3→5, **6→16**. `use_mul_mat_vec_q`
requires `src1->ne[1] <= luce_mmvq_max_ncols`, and `MMVQ_MAX_BATCH_SIZE` is 8
(`mmvq.cuh:3`). So q1 and widths 2-3 are all **MMVQ**, while width 6 at
physical 16 can only be **MMQ**, at any ceiling. Widths 2 and 3 went green
because both sides now run the same family through the same tree; width 6 is
the first width where the batched path is a different quantized matmul kernel
from the q1 reference it is asserted equal to.

**This is not another `sum_rows`, and the difference matters.** The `sum_rows`
shape-dependence was gratuitous — nothing forces a reduction's arithmetic to
depend on how many rows share a launch — so it was a defect and the fix cost
nothing, as the flat DeepSeek A/B showed. MMQ versus MMVQ is not gratuitous:
MMQ exists precisely because it is faster at larger batches, and at physical 16
MMVQ is not even available. Making the two agree means giving up the crossover.

So width 6 is where "batched prefill must be bit-identical to q1" meets "the
engine switches kernel family by batch size for performance". One of those has
to yield, and which one is the release-criterion question below.

**Correction**: an earlier version named `GGML_CUDA_FORCE_MMQ` as the
falsifier. That was wrong — the macro gates `ggml_cuda_should_use_mmq`, not the
earlier `use_mul_mat_vec_q` branch at `ggml-cuda.cu:2591`, so it never moves q1
off MMVQ. Codex ran it and correctly voided the result (codex 371).

**Falsifier that needs no code**: run **widths 4 and 5**. `moe_cached_width`
maps both to physical **5**, so they are batched, MMVQ, and identical in
composition to width 6, which maps to physical **16** and can only be MMQ.

| logical | physical | family | result |
|---|---|---|---|
| 2, 3 | 5 | MMVQ | **green** |
| 4, 5 | 5 | MMVQ | **green** (codex 374) |
| 6 | 16 | MMQ | **red** |
| 17 | chunks to **16 + 1** | MMQ then MMVQ | **red** (codex 376) |

**Run, and the boundary is exact.** Widths 4 and 5 are validator-green on
`86a5ce1`. Every width whose physical bucket is 5 passes; the first width whose
bucket is 16 fails. The transition coincides precisely with the MMVQ→MMQ
crossover, established with no diagnostic code path. (Codex is right that this
does not rehabilitate the invalid `GGML_CUDA_FORCE_MMQ` run; it replaces it.)

### The decision this leaves, which is the user's

The engine switches quantized matmul kernel family by batch size because MMQ is
faster at larger batches, and at physical 16 MMVQ is not available at all
(`MMVQ_MAX_BATCH_SIZE` is 8). Two different kernels will not agree bit-exactly.

So the question is not "where is the width-6 bug" — the boundary says there may
not be one. It is: **does release correctness require bit identity between
batched prefill and q1 stepping, across kernel families that exist because they
differ?**

Three things bear on it, all recorded above:

- Upstream has **no q1 path at all**, so the equality is unaskable there.
- Ember's prefill does not need it: prefill verifies nothing and nothing
  downstream consults a q1 prefill. MTP's q1 replay is separate and stays.
- `qwen4exp_mtp.cpp:320-327` already declines to rely on this kind of equality
  where it *does* matter, and answers it architecturally.

**Width 17 is measured, and it strengthens the case rather than complicating
it.** I had recorded it as a third question because `moe_cached_width(17)`
returns 0. Codex corrected that from source: `dense_eval_rows` processes 17 as
a **max-16 chunk plus one row**, so the path still crosses the physical-16 MMQ
boundary. It is a fourth data point on the same question, not an independent
one.

**The correlation is now complete across every width measured**: 2, 3, 4 and 5
stay entirely on MMVQ and are green; 6 and 17 both contain physical-16 MMQ work
and are red against the q1 MMVQ reference. Five of five widths follow the
family boundary exactly, with no exceptions in either direction.

Width 17 maps to physical **0** — the dense/MoE cache does not serve it — so it
is a third question again, and it has not been run since the fix.

## Open correctness blocker — ROOT CAUSE FOUND

**`sum_rows` selects its reduction tree from the row count, so q1 and batched
cannot agree by construction.**

`sumrows.cu`, `ggml_cuda_op_sum_rows`:

    if ((nrows / nsm) < 2) { block_dims(512) }               // A
    else { ...; block_dims(ncols < 1024 ? 32 : 128) }        // B

and `reduce_rows_f32` (`reduce_rows.cuh:109-144`) strides by exactly that
width: `for (int i = col; i < ncols; ) { ... i += blockDim.x; }`.

`exact_l2_norm`'s `ggml_sum_rows` sees `ncols = head_dim = 128` and
`nrows = n_key_heads * n_tokens = 16 * n_tokens`:

| | nrows | branch | blockDim | per thread | tree |
|---|---|---|---|---|---|
| q1 | 16 | A | 512 | one element, lanes 128-511 add zero | butterfly over 512 |
| q3 | 48 | B | 32 | four elements accumulated serially, then summed | butterfly over 32 |

Two different accumulation trees over the same 128 values. Neither is wrong;
they simply cannot round the same.

**On `nsm`.** That branch assignment holds for `nsm <= 24`. An earlier version
of this section asserted "gfx1151's 20 CUs" as established fact; published
figures give Strix Halo **40** CUs, which would put q1 and q3 on the *same*
branch and contradict the measurement. The sound reasoning runs the other way:
the fix closed width 3, so the branch did differ between them, so `nsm <= 24` —
consistent with HIP reporting the part's 20 WGPs rather than its 40 CUs through
`multiProcessorCount`. One line settles it, and it is worth printing once:
`ggml_cuda_info().devices[id].nsm`.

This accounts for the whole measured signature (codex 354): convolved, decay
and beta exact — none of them touch `sum_rows`; normalized **Q and K both**
non-exact — both go through `exact_l2_norm`, at `1.1920929e-07` and
`5.96046448e-08`; recurrent state non-exact afterward, since K is a direct
recurrence input; and nothing visible at q1, where there is no second shape to
disagree with.

It also explains why the double-precision control tied on HIP at
`6.24756508e-09` for both orders: the recurrence was never the seam.

**A comment helped hide this.** `sumrows.cu` says "The per-row reduction is
unchanged, so this is bit-identical to the uncapped launch" — true, and about
the *grid cap*. It says nothing about the block-width branch two lines below,
which is where the shape-dependence lives. Correct it with the fix.

**The same defect exists twice.** `mean.cu:70-83` carries it byte-for-byte —
same `(nrows / nsm) < 2` branch, same `reduce_rows_f32`, differing only in
`norm=true`, dispatched from `ggml-cuda.cu:3141`. There is no current
`ggml_mean` call in `engine/dflash/` or `src/`, so it is latent; fix it while
the reason is understood. The rest of the HIP tree is clean: `norm.cu` selects
on `ncols`, `gated_delta_net.cu` on `S_v`, and everything else conditional on
`nrows`/`nsm` sets grid dimensions, which changes block-to-row assignment and
not the per-row tree.

**Guard**: `test_sum_rows_shape_invariance` (`f021309`) sums the same rows in a
16-row and a 48-row tensor — `n_key_heads * n_tokens` at q1 and q3 — and
requires the shared rows bit-identical. Trivial on CPU, meaningful under
`DFLASH_QWEN_GDN_TEST_HIP=1`, where it fails on any pre-fix build. It states
the invariant generally: *a reduction's arithmetic must not depend on how many
independent reductions are launched alongside it.*

**Fix**: make the block width a function of `ncols` only. One line, and q1 and
batched then take the same tree. Narrowing it to the `exact_l2_norm` call sites
would leave the hazard live for every other `sum_rows` whose row count crosses
the threshold — a general q1-versus-batched hazard, not a GDN one. Changing the
release criterion is now clearly wrong: the shape-dependence is gratuitous,
nothing forces a reduction tree to depend on row count.

**DeepSeek regression safety A/B, diagnostic only.** `sumrows.cu` is shared by
the production DeepSeek graph, including its routing reductions, so the
reviewed diff was screened before landing.  The unprofiled ABBA sequence was
baseline/candidate/candidate/baseline, with a fresh server and KV directory per
arm, the production DSpark environment, and three exact 256-token decode
samples per arm.  Baseline binary SHA-256 was
`e8e4dd620ec2ca8160d3b1e1849af96fc7750aba2405b2b04c23b8f6c3b0eabc`;
candidate binary SHA-256 was
`e5f3a8eca4e1faaf39df68600e50d15537b2a93de3c16ad6f4ecc2fd66c93d2e`,
from reviewed diff SHA-256
`a03c5da4a1cae62b129a07bcbac48553c730f4552abe7f93ebe940676f685926`.
All twelve samples completed 256 tokens with speculation active and matching
acceptance.  Across all six samples per side, baseline decode was p50 39.972,
p90 40.421, p99 40.434, max 40.435 tok/s; candidate was p50 40.394, p90
40.443, p99 40.446, max 40.446 tok/s.  Restricting each arm to repeats 2-3 to
exclude its first long-decode sample gives medians 40.391 and 40.419 tok/s,
respectively.  This is flat and clears the shared-kernel regression concern;
it is **not a publishable Qwen performance result** while the correctness
blocker remains open.  Raw evidence:
`sumrows-ds4-ab-20260830T210633Z/`.  Production restored healthy and the GPU
lock was free after the run.

## Open correctness blocker

> **A release-criteria question is now attached to this section and belongs to
> the user, not to any agent.** See "Is bit-exactness the right criterion?"
> below.


q=1 and batched prefill disagree from batch width 2.

**The "root cause isolated" claim is withdrawn.** It was true of width 2 only.
Codex ran the sweep under `LUCE_MMVQ_MAX_NCOLS=5` at `a3a50c4` (`.coord/msg/`
codex 106, evidence under
`ncols5-width-sweep-a3a50c4-20260830T191000Z/`):

| prompt tokens | result at ceiling 5 |
|---|---|
| 2 | pass, seed and AR logits bit-identical |
| 3 | fail index 1, expected 830 actual 198 |
| 6 | fail index 0, expected 10459 actual 87 |
| 17 | fail index 0, expected 87 actual 830 |

This is what the width map predicts (see below): the ceiling can only move
logical widths 2-5, and it moved 2. Widths 3, 6 and 17 have **no candidate
cause**.

**And the second cause is not a kernel-precision story.** At width 3 the
batched seed logit for token 830 is 13.4118 against 19.5071 at q=1 — a 6.1
absolute shift that flips the argmax. MMQ-versus-MMVQ is quantization rounding;
it perturbs a logit far below that and does not reorder the top of the
distribution. Whatever is wrong at 3, 6 and 17 is structural — wrong data,
wrong positions, wrong routing, or padding reaching a real row — not
arithmetic. The entire MMQ/MMVQ family can be set aside for these three.

`LUCE_MMVQ_MAX_NCOLS=5` remains justified on gfx1151 as a *throughput* default
(35.65 versus 34.69 tok/s), and must not be gated as the correctness closer.

### Named suspect: the pad-independence assumption

`qwen4exp_frontier.h:104-107` states that logical 2-5 are zero-padded to
physical 5 and that "MoE rows are independent, so padding cannot change a real
row". That is an assertion in a comment, not a tested invariant, and it sits
exactly where the failures are.

**Run, GPU-free, and the dense half is eliminated** (`99dcc3d`).
`test_qwen4exp_frontier.cpp` now evaluates every row inside a padded batch
against the same row evaluated alone at q=1 — the comparison the failing
differential makes — at widths 1, 2, 3, 4, 5, 6, 16 and 17. It passes at all of
them, and mutation-testing (comparing a row against its neighbour's q1 result)
fails at width 2, so the check bites.

Scope of that elimination, stated exactly because an over-read costs a run:
`dense_eval_rows` is a plain `mul_mat` over `weights.router` on the CPU backend
with F32 weights. It clears the **padding algebra on the dense path**. It does
**not** touch MoE routing or the experts — the comment's "MoE rows are
independent" has had its dense half tested and its MoE half not — nor the
type-101 ROCMFPX quantized path, nor the HIP kernels, nor positions and state.

**The MoE half is now eliminated too** (`b5d0bb5`). A batch graph at the cached
physical width, `width` real rows, zeros to the bucket edge, each real row
compared against the same row through the q1 graph, at widths 2, 3, 5, 6 and
16. Passes; mutation-testing fails it at width 2 on both rows. This was the
half with a plausible coupling — routing picks top-k experts per row, so a
batch-axis reduction would change which experts a real row dispatches to, at
whole-logit scale.

Both halves of the sentence at `qwen4exp_frontier.h:104-107` are now tested
rather than asserted, and both hold.

### Eliminated so far, all at the exact failing widths

| claim | status |
|---|---|
| MMQ/MMVQ crossover explains 3/6/17 | refuted — codex 106; bounded impossible for 6 and 17 |
| magnitude consistent with kernel precision | refuted — 6.1 logits with an argmax flip |
| dense padding couples rows | refuted — `99dcc3d` |
| MoE routing couples rows across the batch | refuted — `b5d0bb5` |
| `sync_fallback` | refuted — **0 of 4924** MoE dispatches, measured twice |

`sync_fallback`'s denominator is load-bearing and belongs here rather than in
`docs/dead-code-candidates.md`, which points at this table. "0 dispatches" on
its own is a much weaker claim than 0 of 4924 — a null result is only as strong
as its sample. The figure matters twice over: it is the evidence for dead-code
entry 3, and it is the reason the F32 dequantized reference gates on width 2,
since that reference routes every expert through this never-exercised path.

### What is left

1. the type-101 ROCMFPX quantized path — the CPU tests use F32 weights
2. the HIP kernels themselves
3. state the frontier fixtures do not carry: positions, KV, GDN recurrent and
   conv history across a batch

3 ranks above 2. A quantization or kernel fault should be roughly
width-uniform; the observed pattern is widths 1 and 2 passing with 3, 6 and 17
failing, which looks like something that engages only once a batch carries more
than two tokens of *history*.

**GDN batching is already covered.** `test_qwen4exp_frontier.cpp:478-545`
builds a batch graph at n=3 and compares against three sequential scalar rows
chaining conv and recurrent state; `:547+` repeats at n=16. Output, conv
frontier, final recurrent state, tolerance 2e-5, plus an exact-replay check. It
passes. GDN's batched kernel and its state chaining are clear at both failing
bucket widths.

### The run that separates what is left

`batch_q1_numerics_mask()` (`qwen4exp_runtime.cpp:1618-1641`), consumed at
`:1695-1700`, exists for exactly this question: it retains the layer-major
schedule and causal state order while forcing every normally batched subsystem
through its q=1 graph.

    DFLASH_QWEN_BATCH_FORCE_Q1_NUMERICS=1   -> mask 31

Bits: `Ple=1`, `AttentionHc=2`, `Attention=4`, `FfnHc=8`, `Moe=16`. Codex 106
ran the ncols5 sweep at **mask 0**; no run at width 3 with mask 31 is on
record.

- **still fails at mask 31** — no batched kernel is involved, so the defect is
  in scheduling, composition, or causal state order. That eliminates all five
  batched subsystems and the batched type-101 kernels in one run, and moves the
  search into host C++ that can be reviewed GPU-free.
- **passes** — the defect is in exactly one of the five, and the mask bisects
  it in three more runs.

Width 2 production runs the **mask 0 batched kernels**, not q=1 graphs, so mask
31 at width 3 is a third path rather than a return to a proven one.

### Read GPU-free while that run is outstanding — no defect found

Eight things are eliminated from the fail branch:

| | why |
|---|---|
| `mrope_positions` pre-pushed for the whole batch (`:1915-1916`) | only the `!dense_selection` scorer reads it, and it never runs at ctx ≤ 2048 |
| `cur_pos` held at the pre-batch value until `:1978` | not read anywhere inside the layer loop — `:1888`/`:1890` are bounds checks, `:1978` the advance |
| PLE trigram history | `state.ple_tokens` is a 2-token history, which matches the pass-1,2/fail-3 signature exactly — but the shift is identical in serial (`:260`) and batched (`:284`, `:298`, `:384`) |
| PLE dilated convolution ring | 9 slots, taps at t-9/t-6/t-3/t (dilation 3), shifted one slot per token; same code in `:253-257` and `:374-377` |
| QSA per-row cache appends | `finish_qsa_row:905-907`, strictly row-ordered |
| dense padding | tested, `99dcc3d` |
| MoE routing across the batch | tested, `b5d0bb5` |
| GDN batch versus sequential | pre-existing test at n=3 and n=16 |

HC turned out to be covered by an existing test:
`test_qwen4exp_frontier.cpp:297-312` runs `hc_eval` at n=3 against the scalar
reference including injection values, then at n=1 against the first row of the
same reference. `hc_mix` and `hc_mix_rows` differ only in the `n_tokens`
argument and the row-major slice of `raw_injection` (`:1418-1422`);
`hc_combine` (`:151-158`) is per-row scalar arithmetic, identical in both.

### ISOLATED: `run_gdn_batch()` at width 3

Codex 234, run `q3-gdn-mask4-eec1c68-static`, binary SHA-256
`a095fb75cc7e1c6a4636b458a1b446a40b8c95ac90313cc6b38998955e7bc64c`.

With the corrected diagnostic bit 4 serializing **only** the attention families
— QSA through `run_qsa()` and GDN through causal row-ordered `run_gdn()` — the
exact Q3 width-3 differential goes **green**:

    q1 top2       830 @ 19.5070915, 1543 @ 15.118576
    batched top2  identical
    production prefill exact: true

QSA, PLE, both HC mixers and MoE all remained on their **batched** paths. The
only new serialization against the previously red mask 4 is GDN. So the defect
is in `run_gdn_batch()` at width 3, or in its three-step HIP graph/state
boundary.

**This refutes the MMVQ specialization suspect recorded below, and it is
withdrawn.** MoE, HC, PLE and QSA all ran batched at physical width 5 through
the generic MMVQ path in the green run. If the generic-versus-`unroll2` kernel
difference were the cause, that run could not have been exact. It was the
strongest-looking lead available and it was wrong.

Codex 325 then ran the corrected combination sweep: **every bit-4 superset is
exact and every bit-4-absent mask is red.** The isolation holds across the
whole sweep, not just the single mask-4 run.

### Grouped-cols exonerated; suspicion moves to GDN's inputs

Codex 338 ran `DFLASH_GDN_NO_GROUPED_COLS` at width 3. **Real width 3 is red
both with and without it** — two different kernels, the same wrong answer. They
share very little code, so both carrying the same arithmetic fault is unlikely.
That moves the suspicion **off the kernel and onto its inputs**: the batch
graph is handing GDN something different from what three sequential q1 steps
hand it.

The seam is `qwen4exp_frontier.cpp:954-1000` — `ggml_ssm_conv`, the q/k/v
`view_4d`s into `convolved` (strided `nb2 = conv_channels`), `exact_l2_norm`,
and the `repeat_4d` GQA expansion. Those views are the part of the seam whose
shape changes with width; the q1 graph builds them at `n_tokens = 1`.

**Correction (codex 345).** An earlier version of this section claimed the
synthetic half of that run was vacuous because the fixture ran `head_dim = 16`.
That was wrong, and it conflated two different controls.

Codex 338 changed `kControlRows` inside
`qwen4exp_frontier_run_projection_numerics_control()`, which uses
`weights.layers.front()` from the **loaded real model** and calls the
production GDN frontiers. Its recurrent record holds 786,432 values — exactly
48 x 128 x 128 — so it exercised production `S_v = 128` in both grouped and
no-grouped modes. It was not vacuous.

The `head_dim = 16` mistake belonged to a *different*, GPU-free fixture,
`test_gdn_batch_at_hip_legal_conv_channels()`, fixed in `4e9a6aa`. That fix
stands on its own, but it does not explain codex 338 and must not be used to.

Read together with codex 344, the synthetic production-shape control had
**already shown the same pattern as real inputs**: output and conv state exact,
recurrent state non-exact at floating-noise scale. **Real data is not required
to reproduce the first recurrent divergence** — depth is what makes it visible
in the output. That makes the defect reproducible far more cheaply than a full
model run.

### Where in GDN, from the pass/fail pattern

A fresh GDN chunk starts from `S = 0`, and that alone explains why n=1 is
exact. The recurrence (`gated_delta_net.cu:104-127`, `:288-326`) is:

    kv[col]    = sum_i S[i][col] * k[i]
    delta[col] = (v[col] - g * kv[col]) * beta
    S[i][col]  = g * S[i][col] + k[i] * delta[col]
    attn[col]  = sum_i S[i][col] * q[i]

At n=1, `S = 0`, so `kv = 0` and `delta = v * beta` exactly. **Every term that
touches the carried state is multiplied by zero and cannot be observed.** At
n=2 the error is one update deep — enough to perturb, not to reorder the top of
the distribution. At n=3 it compounds and the argmax flips.

So the defect is in a term multiplied by the carried state: the `g * kv`
correction in `delta`, the `g * S` decay, or the `S * k` reduction that
produces `kv`. It is **not** in `v`, `beta`, `q`, the attention output path,
the register loads, or the initial state load — all exercised identically at
n=1, which is exact.

Statically the two kernels are **algebraically identical** on the non-KDA path:
both keep `g` out of the kv reduction and apply it as `v - g*kv`, then decay
with `g*S + k*delta`. Subgroup indexing checks out — `lane = threadIdx.x %
WIDTH`, `rows_per_lane = 8`, `row = r*WIDTH + lane` covers 0..127, and
`__shfl_sync(..., width=16)` confines each exchange to its 16-lane segment.
That raises the odds of a wave32/subgroup-width assumption that only bites once
`S` is non-zero.

### Where in GDN, from source

The suspect kernel is `gated_delta_net_cuda_grouped_cols`
(`engine/ggml/src/ggml-cuda/gated_delta_net.cu:190`), a hardware specialization
`static_assert`ed to `S_v == 128`, 16-lane subgroups and 4 columns per
subgroup. Selection conditions, read at `:388-440`:

    use_grouped_cols = FORCE || (!DISABLE && !ampere_nvidia)
    taken when:  S_v == 128  &&  !KDA  &&  (AMD || NVIDIA >= Ampere)

Qwen is `S_v = 128` with `KDA` false on an AMD device, so **the grouped kernel
is always selected on gfx1151, at every width** — it is not the discriminator
between n=2 and n=3, and a story about kernel *selection* will not explain the
failure. What it does mean is that the generic `gated_delta_net_cuda` path is
never exercised in production here, so its equivalence to the grouped kernel is
untested at every width.

`DFLASH_GDN_NO_GROUPED_COLS` already exists as a runtime guard, so that A/B
costs no code change.

Structural comparison of the two, since it bounds what the difference can be:
both carry state in registers across the token loop and write back once at the
end, both advance `attn_data` by `S_v * H` per token, and neither uses
`__syncthreads` — they are warp-level with `__shfl_sync`. The visible
divergence is the decay term: the generic kernel applies `expf(g_t[i])`
per row, supporting a KDA vector, while the grouped kernel broadcasts a single
scalar `g_val` from lane 0. That is consistent with grouped being gated on
`!KDA`, and it is not width-dependent.

It also resolves the 2-versus-3 residual that broke every other hypothesis.
GDN is the one **recurrent** subsystem: a chunked recurrent kernel failing at
n=3 while passing at n=2 needs no width-keyed story at all, only a tile or
grouped-column boundary that n=2 does not cross.

**And it vindicates the caveat on our own GDN coverage.** Every CPU GDN test
passes — batch versus sequential at n=3 and n=16, output, conv frontier and
final recurrent state — because they run head_dim 4 with 40 convolution
channels and head_dim 16 with 128. Production is head_dim 128 with 10240
channels, and `gated_delta_net.cu` at `S_v = 128` is what none of them reach.
Grok raised exactly this (msgs 273, 313) and it was pointing at the defect.
`56dfb0f` is now the cheapest way to reproduce on hardware without a full model
load: it is the only GDN fixture whose shape HIP accepts, so pointing
`test_qwen4exp_frontier` at a HIP backend exercises the real kernel.

### WITHDRAWN suspect: the gfx1151 type-101 MMVQ specializations

`engine/ggml/src/ggml-cuda/mmvq.cu:1495-1516` selects a **different kernel** by
`ncols_dst`, for `Q4_0_ROCMFP4_FAST` on gfx1151 only:

    ncols_dst == 1  ->  mul_mat_vec_rocmfp4_unroll2_launch
    ncols_dst == 4  ->  mul_mat_vec_rocmfp4_4col_reuse_launch
    otherwise       ->  the generic path

Their bit-exactness is asserted only in a comment — "Each preserves the
original per-lane K traversal and accumulation order" — which is the same kind
of claim the pad-independence comment made before it was tested.

The consequence is sharp: **q=1 runs `unroll2` and batched runs generic**, and
the differential compares batched against q=1. If those two kernels are not
bit-identical, the differential fails *and the reference itself may be the
wrong side*. "Mask 31 green" would then mean only that `unroll2` agrees with
`unroll2`.

Profile matches the failure exactly: HIP-only, type-101-only, `ncols`-dependent,
and invisible to every CPU/F32 test above.

Falsifier, one run, no new kernel: add an env guard forcing the generic path
(the neighbouring code already uses `DFLASH_CUDA_MMVQ_*`), then compare q=1 with
the specialization against q=1 without it. If they differ, the comment is false.

Residual that does not fit yet, stated rather than papered over: widths 2 and 3
both map to physical 5, so any kernel-selection story still has to explain why
2 passes.

### Coverage is now complete at width 3

| subsystem | test | comparison |
|---|---|---|
| HC mixer | `:297-312` | n=3 and n=1 vs scalar reference, incl. injections |
| GDN | `:478-545`, `:547+` | n=3 and n=16 vs sequential scalar, incl. conv and recurrent state — but at spec `{4, 6, 2, 4, 4}`, **head_dim 4** |
| MoE | `b5d0bb5` | widths 2, 3, 5, 6, 16 vs q1 |
| dense projections | `99dcc3d` | widths 1-6, 16, 17 vs q1 |
| PLE | — | code-identical chain, read and verified |
| QSA | — | `run_qsa_batch` is row-serial; it differs from `run_qsa` only in taking the five projections through `matmul_rows`, which is `dense_eval_rows` |

**Every batched subsystem agrees with serial on the CPU backend at the failing
width** — with one dimensional caveat that matters. Production GDN is
`n_heads = 48`, `n_key_heads = 16`, `head_dim = 128`
(`qwen4exp_runtime.cpp:21-23`); the CPU test runs `head_dim = 4`. It covers the
recurrence algebra and the state chaining, not the 128-wide HIP
`gated_delta_net.cu` kernel. Raised by grok (msg 273) and verified.

**And it is worse than a size difference.** HIP's `supports_op` for `SSM_CONV`
requires `src0->ne[1] % 128 == 0` (`ggml-cuda.cu:5526-5528`). Convolution
channels are `(2 * n_key_heads + n_heads) * head_dim`
(`qwen4exp_frontier.cpp:811-813`): the test fixture gives **40**, production
gives **10240**. So that fixture is a shape HIP *refuses* — point it at a HIP
backend and `SSM_CONV` falls back to CPU while the test still passes. It cannot
exercise the production dispatch on any backend.

`56dfb0f` adds a control at the smallest spec satisfying the predicate,
`{8, 4, 2, 16, 4}` → 128 channels, with the channel count asserted rather than
assumed. Still CPU and still not the production kernel, but it is a shape HIP
accepts, so it can be pointed at a HIP build later and will exercise the real
dispatch. This is also grok 199's `conv_channels % 128 == 0` constraint
arriving from the other side: our own fixture already violated it.

So a green mask 31 does **not** collapse straight to the type-101 dense path.
The next step there is `DFLASH_QWEN_BATCH_Q1_MASK=4`, which forces GDN and QSA
onto q=1 graphs while leaving MoE, HC and PLE batched:

- green → the defect is in the Attention bit, and it splits again between GDN's
  128-wide HIP kernel and QSA's projections (which *are* the dense path)
- red → not Attention, and the type-101 dense MoE/HC path is implicated despite
  the CPU tests

Worth knowing in advance either way: `dense_eval_rows(3)` versus
`dense_eval(1)` is exactly the q5-graph-versus-q1-graph comparison, it passes
on CPU/F32, and at ceiling 5 both sides are MMVQ on HIP. So if the trail does
end at the dense path, it means two MMVQ paths disagreeing by six logits, which
is not a rounding story either.

A red mask 31 puts it outside `qwen4exp_batch_layer` entirely — prefill
chunking, the embedding path, or state carried across the batch boundary.

The two composition asymmetries found (`mrope_positions`, `cur_pos`) are real
but inert. They stay filed as guard-correctness issues for after the blocker —
pre-filling the position history turns the bounds check at `:832-836` from a
real guard into one that always passes. Our default of 3 is an inherited sm_86 (RTX 3090)
crossover measurement, not a gfx1151 one — see the comment at
`engine/ggml/src/ggml-cuda/ggml-cuda.cu:2545-2559`, which explicitly says to
override for other hardware, and note that DeepSeek already overrides it to 4
via `configure_gfx1151_dspark_mmvq_default()`.

No performance number should be published until this closes, because the cost
of whatever fix lands is itself part of the number.

**Two things found on 2026-08-30 that qualify the "root cause isolated" claim.**

*Nothing in the Qwen path sets that ceiling.* `LUCE_MMVQ_MAX_NCOLS` is read once
into a function-local `static` at `ggml-cuda.cu:2576-2580`, default **3**. The
only writer is `configure_gfx1151_dspark_mmvq_default`
(`deepseek4_backend.cpp:141-184`), which early-returns unless `DFLASH_DS4_SPEC`
is set — a DeepSeek4 flag in the DeepSeek4 backend. `engine/dflash/qwen4exp/`
contains no `setenv` and no reference to the variable; `qwen4exp_backend.cpp`
reads five env vars and this is not one. The single hit under `scripts/` is
`benchmark_bundle.sh:106`, inside the `SPEC_ENV` block next to
`DFLASH_DS4_SPEC=1`. So Qwen takes MMVQ only at `src1->ne[1] <= 3`, from an
RTX 3090 measurement, on hardware whose only in-tree measurement says 4. The
comment at `:2560-2574` predicts this in writing.

*The red at width 3 is not explained by a ceiling of 3.* At that ceiling
logical width 3 already takes MMVQ — the same kernel as q=1 — so it should be
green, and it is not, while width 2 is. Either the physical `src1->ne[1]`
differs from the logical width (`qwen4exp_frontier.h:266-269` already speaks of
"physical q=1 [...] q=5/q=16", so they are known to diverge here), or there is
a second cause and raising the ceiling will move 6 and 17 but leave 3 red.

**The width map bounds what the ceiling could ever have explained.**
`qwen4exp_frontier_moe_cached_width` (`frontier.cpp:309-317`), which
`dense_cached_width` reuses verbatim (`:319-321`), maps logical to physical:

| logical | physical | ceiling 3 | ceiling 5 |
|---|---|---|---|
| 1 | 1 | MMVQ | MMVQ |
| 2, 3, 4, 5 | 5 | MMQ | MMVQ |
| 6 … 16 | 16 | MMQ | MMQ |
| 17+ | 0 — cache does not serve it | — | — |

QSA uses a different map (`:323-333`): 3→16, 6→16, 17→64.

`MMVQ_MAX_BATCH_SIZE` is **8** (`mmvq.cuh:3`), so physical width 16 cannot take
MMVQ at any ceiling. Therefore **`LUCE_MMVQ_MAX_NCOLS` can only ever change
logical widths 2-5. It cannot touch 6, and 17 never reaches that cache.** The
measured sweep above matches that bound exactly, which is the strongest reason
to trust it.

If the fix is a Qwen-side default, note that the ceiling is latched on the
first `mul_mat` and never re-read, so any `setenv` must precede it — backend
init, ahead of warmup.

## Counter-unit correction (affects any older bandwidth figure)

ROCm 10 gfx1151 calibration (run 33288846711) measured `FETCH_SIZE` at 64-byte
and `WRITE_SIZE` at 128-byte transactions, relative RMSE ~0.0012. `AGENTS.md`
previously asserted KiB. Any bandwidth figure computed under the KiB assumption
is wrong by 16x/8x.

## F32 dequantized reference @ `8815442` — width 2 gate RED, `f32-reference-8815442-20260830T235900Z`

The reference build (`GGML_CUDA_FORCE_CUBLAS=ON` + `DFLASH_CUBLAS_F32_REFERENCE=1`
+ `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1`) positively logged true-F32 compute,
dense `cublas_f32` and routed `sync_fallback_f32`, with no quantized MMVQ/MMQ
route. It still failed its own trust gate.

At width 2 the default build's q1 and production are bit-identical, so any
distance from the reference is **the reference's own error**:

    d_q1 = d_prod = 12.0171

Verified independently by claude with `scripts/qwen_f32_reference_compare.py`
against the raw rows.

### The reference is wrong at the top of the distribution, not just in the tail

Max `|delta|` against the trusted default, restricted to the default's own
top-ranked logits:

| rank window | row 0 | row 1 |
|---|---|---|
| top-1 | 0.519 | **4.827** |
| top-2 | 7.120 | 5.813 |
| top-10 | 7.220 | 7.557 |
| top-50 | 11.968 | 8.689 |
| top-2 margin | 2.167 → **8.768** | 3.313 → 4.299 |

Row 1's single most probable token moves by 4.83. Both rows keep argmax 830,
which is why the run completed, but argmax agreement alongside a top-2 margin
moving 2.167 → 8.768 is not agreement. **`sync_fallback` + dequantize-to-F32 is
not a trustworthy oracle for this model** — the outcome the 0-of-4924 evidence
predicted, and the reason the width-2 gate existed. No conclusion about MMQ
follows from this run.

### Full matrix, all four widths (claude, `0aadac3 --ranks`)

`d_prod` is the reference's distance from the trusted default; the `top-K`
columns restrict that same statistic to the **default's** own top-ranked
logits; `rank` locates the single worst deviation in the default's ordering.

| w | row | d_prod | top-1 | top-2 | top-10 | top-50 | rank(worst) | argmax same | margin def→ref |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0 | 12.017 | 0.519 | 7.120 | 7.220 | 11.968 | 247775 | yes | 2.167 → 8.768 |
| 2 | 1 | 11.805 | 4.827 | 5.813 | 7.557 | 8.689 | 4019 | yes | 3.313 → 4.299 |
| 3 | 0 | 12.354 | 4.025 | 5.127 | 12.354 | 12.354 | **4** | yes | 2.643 → 3.745 |
| 3 | 1 | 10.265 | 2.637 | 5.847 | 10.265 | 10.265 | **7** | yes | 0.776 → 3.986 |
| 6 | 0 | 12.795 | 3.604 | 5.621 | 6.605 | 10.938 | 197 | **no** | 2.306 → 4.323 |
| 6 | 1 | 10.741 | 1.855 | 3.055 | 4.681 | 7.825 | 246579 | yes | 2.628 → 3.828 |
| 17 | 0 | 9.029 | 3.781 | 7.345 | 7.345 | 7.892 | 246734 | yes | 3.843 → 7.408 |
| 17 | 1 | 8.987 | 3.929 | 6.556 | 6.556 | 7.223 | 115266 | **no** | 2.048 → 4.675 |

Three readings:

- **The reference is broken at widths 2 and 3, where the default is
  trustworthy.** Width 3 is sharpest: its worst deviation sits at rank **4** and
  rank **7** — not in the tail — against a default that is token-exact with q1.
  Not a routing-amplification artefact at those widths.
- **The magnitude is near-constant, 9.0-12.8, at every width.** That is the
  signature of a systematic decode error, not a discontinuity: amplification
  would predict width dependence, and there is none.
- **The reference flips the argmax** at width 6 row 0 and width 17 row 1 — it
  produces different tokens from the trusted default.

The near-constant magnitude has a consequence beyond this run: **~10-12 is
simply what this model's logits do when a compute path changes.** The ~11.9
observed at widths 6/17 in the tranche 1 table above is therefore *not* by
itself evidence that MMQ is defective, and the earlier reading of it as "strong
evidence" is withdrawn.

### `max_abs_logit_delta` is rank-blind — a criterion defect in its own right

Row 0's largest single deviation sits at **rank 247775 of 248320** (logit −8.5
against a maximum of 19.7 — unsamplable at any temperature we ship). Row 1's
sits at rank 4019.

The statistic cannot distinguish that from an error at rank 2. The release
criterion therefore compares `q1_top2_margin`, a strictly top-of-distribution
quantity, against a value that may be set by a token a quarter-million deep.
**These are incommensurable.** This is separate from the inverted incentive
recorded above, and separate from whether MMQ is defective.

It also weakens the earlier reading of widths 6/17: a delta of ~11.9 says
nothing until its rank is known. The rank-restricted statistics above are what
that question needs.

## Tranche 1 correctness @ `8c67086` — hardware, `tranche1-qsa-8c67086-20260830T232419Z`

Copied from the runner evidence at codex's request (msg 402). Correctness run;
**no performance claim** — the `ar_tok_s` column is included only because it is
in the same file and must not be read as a benchmark (2-token decodes, validator
build, production quiesced).

| width | prefill exact | accepted | `q1_top2_margin` | `max_abs_logit_delta` | spec accept_rate | ar tok/s |
|---|---|---|---|---|---|---|
| 2 | yes | yes | 2.1667 | **0** | 1.0 | 11.43 |
| 3 | yes | yes | 2.65555 | **0.057539** | 1.0 | 14.85 |
| 6 | no | yes | 0.780952 | **11.9231** | 0.0 | 11.37 |
| 17 | no | yes | 0.270742 | **11.7909** | 0.0 | 11.07 |

### Correction: "widths 2-5 are bit-identical" is wrong

That claim has been repeated in several places, including my own msgs 361 and
364. **Width 2 is bit-identical (delta exactly 0). Width 3 is not**
(`0.057539`). Both are token-exact, which is what the validator reports as
"exact" — token equality, not logit equality.

This has a direct consequence that argument got backwards: the green widths
**do** supply a quantization-noise scale. Width 3 changes the reduction shape
without crossing the MMVQ/MMQ family boundary, and moves the logits by
**0.058**. That is the calibration msg 361 claimed could only come from an F32
reference. It is already in hand.

### The margin criterion is accepting a ~12-logit perturbation

Widths 6 and 17 move the logits by **11.92** and **11.79** — **207x and 205x**
the width-3 scale. They are accepted, and the mechanism is the inverted
incentive recorded in msg 361: `accepted = margin < delta`, so a *larger* error
is *more* likely to be accepted. Here the margins are small (0.78, 0.27) and
the deltas enormous, which is the most permissive combination the rule admits.

**An independent signal in the same files agrees.** Speculative
`accept_rate` is **1.0 at widths 2 and 3 and 0.0 at widths 6 and 17** — a
perfect correlation with the non-exact widths, from a measure the margin
criterion does not feed. A drafter whose every token is rejected is not
describing a rounding difference.

**What this does and does not establish.** Width 3 and width 6 differ in two
ways at once: the matmul family (MMVQ to MMQ) and the batch shape. So 0.058 is
the scale for a shape change *within* MMVQ, and a correct MMQ could legitimately
sit somewhat above it. What it cannot plausibly do is sit 200x above it. This is
strong evidence that the MMQ path is wrong rather than merely coarser — but it
is evidence, not proof, and the F32 reference run (msg 364) is what separates
the two.

## F32 dequantized reference @ `8815442` — trust gate failed

Evidence:
`f32-reference-8815442-20260830T235900Z`. This is a correctness diagnostic,
not a performance run; no timing from the force-cuBLAS build is usable.

The default and reference width-2 validators both completed successfully with
equal, finite two-row captures. The reference build positively logged
`GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F`, `path=cublas_f32`, and
`path=sync_fallback_f32`, with no quantized MMVQ or MMQ route. Thus C reached
the intended dequantize-to-F32 plus SGEMM family for both dense and routed
expert matrices.

| width | `d_q1 = max|B-C|` | `d_prod = max|A-C|` | result |
|---|---:|---:|---|
| 2 | 12.0171 | 12.0171 | **reference trust gate failed** |
| 3 | 12.3842 | 12.3542 | characterization only |
| 6 | 12.9528 | 12.7954 | characterization only |
| 17 | 11.1705 | 9.02933 | characterization only |

At width 2 the default production and q1 captures are bit-identical, so the
equal distances are the reference path's own error. That error is on the same
scale as the width-6/17 effect the experiment was meant to adjudicate, rather
than the within-MMVQ width-3 scale. Per the predeclared gate, this run says
that the newly exercised `sync_fallback`/F32 composite is not a trustworthy
reference; it says nothing for or against MMQ. The requested wider captures
are shown only to characterize the invalid reference path: every distance
remains on the same large scale, so none may be interpreted as an MMQ verdict.

### Rank-aware default q1-versus-production deltas

Computed offline from the default-build rows in the same evidence bundle; no
additional GPU run. `top-K delta` is the maximum absolute q1/production delta
restricted to the K highest logits in the q1 ordering. `worst rank` is the
one-based position, in that same ordering, of the unrestricted maximum delta.
Margins are each side's own top-1 minus top-2. An exact row has no meaningful
worst rank and is shown as `—`.

| width | row | q1 argmax | prod argmax | q1 margin | prod margin | top-1 delta | top-2 delta | top-10 delta | top-50 delta | worst rank | max delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 830 | 830 | 2.16670 | 2.16670 | 0 | 0 | 0 | 0 | — | 0 |
| 2 | 1 | 830 | 830 | 3.31295 | 3.31295 | 0 | 0 | 0 | 0 | — | 0 |
| 3 | 0 | 830 | 830 | 2.65555 | 2.64312 | 0.003332 | 0.009094 | 0.029988 | 0.029988 | 172739 | 0.057539 |
| 3 | 1 | 830 | 830 | 0.776179 | 0.776179 | 0 | 0 | 0 | 0 | — | 0 |
| 6 | 0 | 17962 | 87 | 0.780951 | 2.30577 | 6.84558 | 6.84558 | 6.84558 | 6.96886 | 247804 | 11.9231 |
| 6 | 1 | 11966 | 830 | 0.103767 | 2.62778 | 7.07121 | 7.07121 | 7.97384 | 8.71832 | 247935 | 10.3123 |
| 17 | 0 | 23295 | 87 | 0.270742 | 3.84308 | 4.60085 | 5.02064 | 5.02064 | 6.43461 | 247767 | 11.7909 |
| 17 | 1 | 11966 | 830 | 1.16483 | 2.04813 | 1.44372 | 5.94212 | 6.66708 | 7.81858 | 16995 | 9.99369 |

#### Isolated ROCMI4 operator oracle @ `b4fb6fe` — green, narrow

Evidence:
`rocmi4-operator-oracle-5b8e368-20260831T003000Z` and
`rocmi4-partial-k-oracle-b4fb6fe-20260831T005301Z`. These are model-free
correctness diagnostics; they collected no timing.

All 23 predeclared checks passed on gfx1151. The exact fixture independently
covered the production `convert.cu` decoder, dense MMVQ/MMQ, routed-expert
MMVQ/MMQ, and a CPU ROCMI4 decode plus scalar F32 matmul with a zero-error
budget at K = 160, 256, 320, and 640. Thus the full MMQ iteration control and
all shipped partial-iteration K values are green. Dispatch logs prove that
both dense and routed cases reached both kernel families. The non-grid fixture
also kept the direct and routed MMVQ/MMQ outputs within the predeclared
per-output bound of one Q8_1 activation-quantization step per K32, weighted by
the row L1 norm.

This eliminates an isolated ROCMI4 device decoder or accumulation failure on
these fixed fixtures. It does **not** validate the full-model activation,
routing, masking, or state-selection context, so it remains compatible with
the controlled full-model red result below.

#### REFUTED: MMQ partial-K handling is not the defect

The predeclared falsifier was a red K = 320 or 640 result beside the green
K = 256 control. All four K shapes instead pass the zero-error oracle through
both dense and routed MMQ, so the partial-K hypothesis is eliminated.

The source reading missed the paired activation padding. gfx1151 does take the
conventional non-CDNA HIP path, and its ROCMI4 weight loader reads a complete
eight-block tile on a short final iteration. But `ggml_cuda_op_mul_mat` pads K
to `MATRIX_ROW_PADDING` before `quantize_mmq_q8_1_cuda`; the quantizer emits a
real zero Q8 block whenever `i0 >= ne00` (`quantize.cu:221-223`, with zero
scale and integers). The foreign weight blocks are therefore multiplied by
zero. The model-free exact oracle confirms that this mechanism is sound at
every shipped partial K rather than merely avoiding a fault.

The next missing isolated dimensions are the production output-row values and
the non-contiguous/view activation layouts absent from the current fixture.

#### PROPOSED CRITERION: total-variation distance at the serving temperature

Computed by claude from the already-retained rows; no GPU. `TV` is the total
variation distance between `softmax(q1/T)` and `softmax(production/T)`.

| case | validator | r | TV @ 0.6 | TV @ 1.0 | argmax |
|---|---|---|---|---|---|
| w2 r0/r1 | green | 1.0000 | 0.0000 | 0.0000 | same |
| w3 r0 | green | 1.0000 | **0.0002** | 0.0009 | same |
| w3 r1 | green | 1.0000 | 0.0000 | 0.0000 | same |
| **w4 r0, `NCOLS=3`** | **GREEN** | 0.5558 | **0.5073** | 0.8006 | same |
| **w4 r1, `NCOLS=3`** | **GREEN** | 0.6014 | **0.1390** | 0.4228 | same |
| w6 r0 | red | 0.5986 | 0.8236 | 0.7288 | differs |
| w6 r1 | red | 0.5402 | 0.9412 | 0.8578 | differs |
| w17 r0 | red | 0.6750 | 0.7070 | 0.7227 | differs |
| w17 r1 | red | 0.6222 | 0.8943 | 0.8162 | differs |

**TV has an operational meaning the current criterion lacks.** It is the maximum
probability difference over any event, so `TV = 0.507` says **up to ~51% of
sampled tokens could differ** on that row — in a run the validator reported as
token-exact and accepted.

**The separation is not marginal.** Genuinely-good rows sit at 0.0000-0.0002;
the worst validator-green row sits at 0.5073. That is ~2500×, so any threshold
in between works and the choice is not a tuning exercise. **0.01 at the serving
temperature** leaves two orders of magnitude of headroom on both sides.

**Why TV rather than the correlation floor suggested earlier.** Pearson `r` is
invariant to scale and shift, so a uniformly rescaled logit vector passes it
while sampling very differently; TV is not. TV is also temperature-aware, and
`w4 r1` shows why that matters — `r` is essentially the same as `w4 r0` (0.601
vs 0.556) while TV differs by 3.6× (0.139 vs 0.507), because TV weights the part
of the distribution that is actually sampled. Evaluate at the **served**
temperature (`--default-temperature 0.6`); note TV@1.0 is larger for these rows,
so a criterion fixed at 0.6 is not conservative for hotter sampling and the
threshold should be checked at the highest temperature the deployment allows.

**Cost: none.** The vectors are already captured by
`EMBER_VALIDATION_LOGITS_DIR`, and this is arithmetic over them.

**Status: proposed, not adopted.** Criterion B is the user's decision;
`engine/dflash/common/prefill_validation.h` is codex's file and has not been
touched.

#### DECISIVE: a token-exact, validator-GREEN run with a destroyed logit distribution

Width 4, `LUCE_MMVQ_MAX_NCOLS=3`, capture-capable runtime (`5b8e368`), evidence
`qwen-width4-ncols3-correlation-5b8e368-20260831T004201Z`. Verified independently
by claude from the raw rows.

**The validator passed:**

    ok: true
    prefill: exact: true, accepted: true
    q1_top2_margin 0.712470055, max_abs_logit_delta 11.7082748
    detail: "... are token-exact (0 speculative rows)"

**The logits did not:**

| row | r (all) | r (top 1000) | r (top 100) | max Δ | top-1 Δ | argmax | q1 margin → prod |
|---|---|---|---|---|---|---|---|
| 0 | **0.556** | 0.333 | 0.489 | 11.708 | 4.308 | same | 0.712 → 4.019 |
| 1 | **0.601** | 0.495 | 0.567 | 10.181 | 0.282 | same | 2.045 → 4.410 |

The same structural collapse as widths 6/17 (r = 0.54-0.67), at a width the
validator calls **exact and accepted**, because both argmax tokens happen to
survive.

**This is the criterion's failure mode, measured rather than argued.** Token
equality and the top-2 margin can both be satisfied while the distribution is
~45% uncorrelated with the reference. Greedy argmax is one order statistic; it
does not constrain the distribution that produced it.

**It matters for what we actually serve.** The production configuration runs
`--default-temperature 0.6`, not greedy. A perturbation this large changes
sampled output even where greedy argmax coincides, so a validator that checks
only greedy tokens cannot certify shipped sampling behaviour.

**Consequence for the release criterion.** No margin-versus-delta rule over
top-of-distribution order statistics can catch this at any threshold — the
quantity needing constraint is distributional agreement. A correlation floor
(or an equivalent divergence bound) on the retained rows is cheap: the vectors
are already captured and `scripts/qwen_f32_reference_compare.py --ranks` already
reads them.

#### RESOLVED: the dense MMVQ→MMQ crossover alone is sufficient (codex 435/437)

Evidence:
`qwen-width4-ncols3-86a5ce1-20260831T003501Z` and the literal one-variable
confirmation
`qwen-width4-ncols3-silent-86a5ce1-20260831T003800Z`. The first retained run
enabled passive MMID logging; the confirmation omitted it and reproduced the
same result, leaving `LUCE_MMVQ_MAX_NCOLS` as the sole environment change from
the earlier green width-4 run. The discarded setup attempt failed during an
optional dispatch-control preflight before validation and contributes no
correctness evidence.

Width 4, `LUCE_MMVQ_MAX_NCOLS=3`, same binary (`86a5ce1`) and prompt as the
prior green: **validator-red**, `token mismatch at 0: expected=198 actual=87`.

That moves dense quantized matmuls from MMVQ to MMQ while holding the MoE
bucket at 5 and — per the collinearity below — the routed-expert family at MMVQ.
**The dense crossover alone reproduces the divergence; the bucket transition is
not required.**

The bucket hypothesis recorded below is therefore **refuted as necessary**. It
is not excluded as an additional contributor at widths 6/17, but it is no longer
needed to explain them.

**The token is the corroborating detail.** Production picks **87** here at
bucket 5, and also at width 6 row 0 and width 17 row 0 at bucket 16
(cross-evaluation table above). Bucket varies, dense MMQ is constant, and the
same token appears. That is the signature of one mechanism keyed to dense MMQ,
not two mechanisms that happen to coincide.

##### Current-runtime magnitude capture @ `5b8e368` — structural collapse confirmed

Evidence:
`qwen-width4-ncols3-correlation-5b8e368-20260831T004201Z`. The reviewed Release
binary SHA-256 is
`4b97987aa5299aa3ca1ec726ec8416e605da44d87be13c038b2ccc5e21cd487c`.
Raw q1 and production rows were written by `EMBER_VALIDATION_LOGITS_DIR`; the
offline calculation verified equal finite row shapes before correlation. This
is correctness evidence only; no timing from the run is interpreted.

| row | r (all) | r (q1 top 1000) | r (q1 top 100) | slope | residual max | q1 argmax | prod argmax | q1 margin | prod margin | max delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.555806 | 0.332564 | 0.488637 | 0.537101 | 11.7434 | 87 | 87 | 0.712470 | 2.34581 | 11.7083 |
| 1 | 0.601413 | 0.494935 | 0.566995 | 0.589904 | 8.92388 | 830 | 830 | 2.04495 | 1.60067 | 10.1805 |

This is the same structural failure class as widths 6/17, not a near-tie
argmax flip. The dense crossover is independently movable at width 4 while the
MoE bucket and routed-expert family stay fixed, so dense MMQ alone is
sufficient to collapse the logit structure.

Both rows retain their q1 argmax and the validator reports
`prefill.exact=true`, `accepted=true` under the user-decided token/margin
criterion. That criterion therefore certifies token stability for this prompt;
it does **not** certify that the batched verifier computed the same logit
function. The accepted result must not be described as kernel-precision noise.
The isolated operator oracle above remains green only because its fixed
fixtures omit the production activation/state context that triggers the defect.

#### Historical confound: two mechanisms change at width 5→6

This was the confound before the controlled width-4 run above. It is retained
to document why the earlier table could not distinguish the mechanisms; the
one-variable result now proves that the dense crossover alone is sufficient.

1. **Matmul family.** `LUCE_MMVQ_MAX_NCOLS=5` in every run
   (`ggml-cuda.cu:2624-2632`): MMVQ at `ne[1] <= 5`, MMQ above. Boundary 5/6.
   The documented **default is 3** — the 5 is our choice.
2. **MoE graph bucket.** `qwen4exp_frontier_moe_cached_width`
   (`qwen4exp_frontier.cpp:325-333`, constants `qwen4exp_frontier.h:115-116`):
   width 1 → 1; widths 2-5 → 5; widths 6-16 → 16; width 17+ → 0.
   Boundary **also** 5/6, with width 17 a third case again.

| width | MoE bucket | matmul | result |
|---|---|---|---|
| 2 | 5 | MMVQ | green |
| 3 | 5 | MMVQ | green (δ 0.058) |
| 6 | **16** | **MMQ** | red |
| 17 | **0** | **MMQ** | red |

Both hypotheses predict this table exactly. The alignment is an artefact of
setting `LUCE_MMVQ_MAX_NCOLS` to 5, which places the family boundary on top of
the bucket boundary.

3. **Routed-expert dispatch.** `ggml_cuda_mul_mat_id` has its **own** MMVQ/MMQ
   threshold, `get_mmvq_mmid_max_batch(src0->type, cc)` (`mmvq.cu:298`) with
   ceiling `MMVQ_MAX_MOE_BATCH_SIZE = 16` (`mmvq.cuh:8`). It does **not** read
   `LUCE_MMVQ_MAX_NCOLS`, which governs dense `ggml_cuda_mul_mat` only.

**Discriminating run (one env var, no code):** width 4 or 5 with
`LUCE_MMVQ_MAX_NCOLS=3`, which moves the **dense** family boundary to 3/4 while
holding the MoE bucket at 5.

- **Red** ⇒ a dense matmul-family change alone is sufficient; the bucket is not
  required. Decisive.
- **Green** ⇒ the *dense* family is exonerated at width 4. It does **not**
  implicate the bucket on its own, because arm 3 above did not move — two
  hypotheses remain live (bucket, routed dispatch).

**Arms 2 and 3 are collinear by construction, and no env var separates them.**
`DFLASH_CUDA_MMVQ_MOE_KERNEL` is consumed inside the NVIDIA branch
(`mmvq.cu:307-310`); the AMD branch at `:326` returns first, so it is inert on
gfx1151 (codex, msg 432). The routed ceiling for ROCMI4 on RDNA3 comes from
`get_mmvq_mmid_max_batch_rdna3`, which has no ROCMI4 entry and falls to
`default: return MMVQ_MAX_BATCH_SIZE` = **8** (`mmvq.cu:219`, `mmvq.cuh:3`).

The MoE graph is built at the **bucket** width with inputs zero-padded to it
(`qwen4exp_frontier.cpp:3063-3068`), so the routed `MUL_MAT_ID` sees `ne2` =
bucket width, not the real token count:

| real width | bucket | routed `ne2` | vs 8 | routed family |
|---|---|---|---|---|
| 2-5 | 5 | 5 | ≤ | MMVQ |
| 6-16 | 16 | 16 | > | MMQ |
| 17 | **16 then 1** | 16, then 1 | >, then ≤ | **MMQ then MMVQ** |

Width 17 does not fail the `graph_width == 0` guard in practice:
`qwen4exp_prefill_chunk_rows` (`qwen4exp_runtime.cpp:2220-2232`) caps a chunk at
`kQwen4ExpFrontierMoeMaxBatch`, so 17 rows run as **16 + 1** — a bucket-16 chunk
followed by a bucket-1 chunk. Its red is therefore consistent with bucket 16
alone being at fault, and it is *not* an independent third case. (An earlier
line in this document already recorded the 16+1 chunking; the bucket table above
is the mechanism underneath it.)

The bucket *determines* the routed width, so the routed family flips exactly
when the bucket does. Arm 3 is a **consequence** of arm 2, not an independent
suspect. The question narrows to: does bucket 16 fail *because* it routes
experts through MMQ, or because of something else about the wider graph?

**Separating them requires a small code change**, not another sweep: a
default-off diagnostic override returning `MMVQ_MAX_MOE_BATCH_SIZE` for ROCMI4
on RDNA3, so bucket 16 keeps its graph and arena but routes through MMVQ. Still
red ⇒ the bucket's non-routing differences; green ⇒ routed MMQ at `ne2` = 16.

The correlation evidence below favours the bucket: width 3 also crosses a
reduction shape and is also non-bit-identical, yet r = 0.99999, while widths
6/17 fall to 0.54-0.67. A kernel swap should look like width 3.

#### Correlation: widths 6/17 are not a perturbation of q1 at all

Offline from the same retained default-build vectors (claude, no GPU). Pearson
correlation between the q1 and production logit vectors, over the full
248320-token vocabulary and restricted to q1's top-ranked entries:

| w | row | r (all) | r (top 1000) | r (top 100) | slope | residual max |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0 |
| 2 | 1 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0 |
| 3 | 0 | 0.99999 | 0.99996 | 0.99998 | 1.00017 | 0.059 |
| 3 | 1 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0 |
| 6 | 0 | **0.59862** | 0.53633 | 0.62863 | 0.612 | 9.467 |
| 6 | 1 | **0.54019** | 0.52150 | 0.45133 | 0.530 | 9.540 |
| 17 | 0 | **0.67497** | 0.67500 | 0.59550 | 0.686 | 12.024 |
| 17 | 1 | **0.62216** | 0.50493 | 0.51184 | 0.654 | 11.358 |

**This reframes the blocker.** At widths 6 and 17 the two paths share roughly
25-45% of their variance. That is not a perturbed version of the same
computation — the logit vectors are substantially unrelated. And it is not a
tail artefact: restricting to q1's top 100 gives r as low as 0.451.

Width 3 is the control that makes the contrast unambiguous. It **also** crosses
a reduction shape and is **also** non-bit-identical (residual 0.059), yet
r = 0.99999. A kernel-precision difference looks like width 3. Widths 6 and 17
do not look like a kernel-precision difference.

The standard deviations stay comparable on both sides (2.05-2.23), so this is
not a scale or normalisation factor either; the structure itself differs.

**Consequence for the MMVQ/MMQ hypothesis.** MMQ and MMVQ compute the same
matmul over the same int8-quantized activations. Their disagreement should look
like width 3's 0.059, not like r = 0.5. On this evidence the quantized-matmul
family boundary is unlikely to be the *cause* of the width-6/17 divergence, even
though it coincides exactly with the observed threshold. Something that also
changes at that width — routing, masking, or state selection — is the better
suspect. This is a lead from an offline measurement, not a proof.

#### Cross-evaluation: how each side scores the other's winner

Requested in msg 380 and computed offline from the same retained default-build
vectors (claude, no GPU). When the argmax changes, the margins above do not say
whether the two sides are near-tied and flipping on noise or one is confidently
wrong. Scoring each side's winner under *both* distributions does.

| w | row | q1 argmax | prod argmax | q1's lead over prod's pick | prod's lead over q1's pick |
|---:|---:|---:|---:|---:|---:|
| 6 | 0 | 17962 | 87 | 0.786 | **8.926** |
| 6 | 1 | 11966 | 830 | 1.298 | **6.192** |
| 17 | 0 | 23295 | 87 | 0.271 | **9.351** |
| 17 | 1 | 11966 | 830 | 1.165 | **6.221** |

The asymmetry is consistent and large, and it is the same on every divergent
row: **q1 is nearly tied between the two candidates (0.27-1.30) while production
prefers its own winner by 6.2-9.4.** Production's winner is, in every case,
q1's close runner-up.

Two further observations, offered as leads rather than conclusions:

- Production's winners are **87 and 830** on all four divergent rows, across two
  different widths and two different prompts — and 830 is also the argmax that
  both paths agree on at widths 2 and 3. q1's winners are varied and
  prompt-specific (17962, 11966, 23295).
- Symmetric numerical noise would produce symmetric near-ties. A one-sided
  6-9 logit lead is not a tie-break; something is amplifying one candidate.

This does **not** identify which side is correct — that still requires the
operator oracle. But it narrows what to look for: not a diffuse precision
difference, rather a mechanism that inflates a small set of high-prior tokens as
batch width grows.

The unrestricted statistic is rank-blind: at width 3 its maximum sits deep in
the tail while the top-ranked logits barely move. That caveat does not rescue
widths 6 and 17: their q1 argmax changes, and the q1 top-ranked token itself
moves materially on every divergent row. This establishes that the current
margin rule accepts decision-relevant perturbations, not merely a remote-tail
outlier. It still does **not** establish which side is closer to truth; the
full-model F32 reference that was meant to answer that failed its control.

## Host-barrier census @ `faa5307` (static, `qwen4exp_frontier.cpp`)

Grouping each run of `ggml_backend_tensor_get_async` / `_set_async` by the
`ggml_backend_synchronize` that terminates it:

| copies | barrier | function |
|---|---|---|
| 1 | :583 | `dense_eval` (:542) |
| 1 | :603 | `dense_eval` |
| 1 | :697 | `hc_eval` (:639) |
| 3 | :732 | `hc_eval` |
| 3 | :1140 | `gdn_eval_batch` (:1096) |
| 3 | :1162 | `gdn_eval_batch` |
| 1 | :1486 | `qsa_project_q1` (:1468) |
| 5 | :1513 | `qsa_project_q1` |
| 2 | :1550 | `qsa_rotate_q1` (:1517) |
| 2 | :1561 | `qsa_rotate_q1` |
| 5 | :1648 | `qsa_attend_q1` (:1567) |
| 1 | :1657 | `qsa_attend_q1` |
| 1 | :1887 | `moe_eval` (:1869) |
| 1 | :1896 | `moe_eval` |

**14 barriers, 30 copies, 7 stages, every stage a barrier pair** — one closing
the upload run, one closing the download run.

**Two of those rows never execute.** `:1550` and `:1561` belong to
`qsa_rotate_q1`, which is the #27774 Hadamard KV-cache rotation, not the YaRN
stage. Its subgraph is built only when the GGUF carries `attn_k_rot.weight` /
`attn_v_rot.weight`, and the published checkpoint carries neither — see
[`dead-code-candidates.md`](dead-code-candidates.md). So the **live** census is
**12 barriers / 26 copies**, and over 12 QSA layers that is 24 barriers and 48
copies per token that no A/B may be credited with removing.

### Re-derived @ `1ee72b8` after tranche 1 (static, verified by claude 20260831T074500Z)

Tranche 1 did **not** edit the stages above; it added two alternative
functions that replace them when `keep_resident` is set:

| copies | barrier | function | direction |
|---|---|---|---|
| 2 | :1734 | `qsa_project_prepared_q1` (:1700) | upload |
| **1** | :1769 | `qsa_project_prepared_q1` | download |
| 3 | :1998 | `qsa_attend_prepared_q1` (:1918) | upload |
| 3 | :2016 | `qsa_attend_prepared_q1` | download |

**A naive static count of this file now reads 19 barriers / 45 copies. That
number is wrong and must not be quoted.** Three reasons, all of which the
grep hides:

1. The `prepared_*` functions **replace** `qsa_project_q1` / `qsa_attend_q1`,
   they do not run alongside them. Counting both double-counts the stage.
2. The `:1769` group contains six `get_async` calls, but **five of them sit
   behind `if (!keep_resident)`**. On the resident path the group is **one
   copy** — raw index-K, which feeds the host snapshot history.
3. `gdn_capture_inputs` (:2945) is the default-off diagnostic capture, not a
   production stage.

So codex's tranche 1 claim is **confirmed on its static half**: the download
group at the old `:1513` goes from **depth 5 to depth 1**, and the barrier
count is unchanged because both stages still close an upload run and a
download run. The live census stays **12 barriers**, and copies fall from
**26 to 22** per token-layer set.

What remains unverified is the runtime half — that these static groups map to
the barrier timing on the shipped decode path. Only a runner measurement shows
that, and it should report the barrier count next to the timing.

The YaRN rope is not in this table at all: it runs on the host in
`prepare_qsa_row` (`qwen4exp_runtime.cpp:745-770`), between the project
download barrier and the attend upload barrier. What tranche 1 targets is the
**5-get at `:1513`**, which exists so the host can run `rms_norm` + `rope` on
those tensors.

It does **not** delete that barrier. `finish_qsa_row`
(`qwen4exp_runtime.cpp:775-909`) still consumes host `index_query` and
`index_key` for the 4-token pool, the ReLU score and the top-512 selection
(`:811`, `:812`, `:843`) and appends `index_key` to `state.index_key` at
`:908`. Those are two of the five gets.

Below 2049 tokens it is better than that, and that is where we certify.
`qwen4exp_qsa_dense_selection` (`qwen4exp_internal.h:202-210`) is true for
`n_tokens <= 2048`, and both scorer bodies are gated on `!dense_selection`
(`qwen4exp_runtime.cpp:640`, `:801`). So `index_query` is downloaded and never
read, and `index_key`'s payload is never read either — outside the gate only
`state.index_key.size()` is used, as a token counter (`:599`, `:796`), while
the append at `:907` is unconditional. See
[`dead-code-candidates.md`](dead-code-candidates.md) entry 5.

**On the shipped decode path the group therefore goes from depth 5 to depth
1**, the survivor being `index_key` purely to keep a history for contexts above
the boundary. It reaches depth 0, and takes the barrier with it, only when that
history moves on-device. The barrier count stays at 12 until then.

The tranche is still worth taking — three fewer host copies, the head-wise
`rms_norm` and `rope` off the CPU, and the numerics improvement above — but it
must not be measured as "one barrier deleted".

Consequence for `faa5307`, which converted those 30 copies to async: it removes
**zero** barriers. It only lets copies overlap *within* a group, and seven of
the fourteen groups hold exactly one copy, where a `get_async` immediately
followed by `synchronize` is the blocking `get` it replaced. So it can act on
23 of 30 copies across 7 groups of depth 2,2,3,3,3,5,5.

An earlier ~1.2x prediction for this tranche is **withdrawn** — it was not
derived from this structure. Whether the reachable saving is large depends on
whether copy cost is latency- or transfer-dominated, which the pending A/B
answers and the structure does not.

What the structure does establish: whatever fraction of decode time is barrier
latency is untouched by `faa5307` and moves only with the stage-removal
tranches. The measurement that sizes it is a per-barrier counter dumped per
decode token, not a static call count — `dense_eval`, `hc_eval` and `moe_eval`
are called from a dozen sites across `qwen4exp_frontier.cpp` and
`qwen4exp_runtime.cpp`.

## RoPE numerics: the host scalar degrades with position

From `test/test_qwen_rope_graph_oracle.cpp` (`3cc509e`), which carries a
double-precision reference. Max absolute error of the host scalar
`ember_qwen_yarn_apply` against exact, yarn off / on:

| pos | yarn off | yarn on |
|---|---|---|
| 7 | 1.56e-07 | 2.05e-07 |
| 1024 | 1.76e-05 | 2.00e-05 |
| 2074 | 4.37e-05 | 4.97e-05 |
| 65536 | 9.33e-04 | 1.06e-03 |
| 131072 | 2.11e-03 | 2.40e-03 |
| 262143 | 5.65e-03 | 6.43e-03 |

`powf(1e7, 2k/64)` carries ~1 ulp and the angle multiplies that by `pos`; there
is no mod-2pi reduction (`qwen_yarn.c:107`). The worst pair is k=1 or 2, not
k=0, because `inv_freq[0]` is exactly 1.0 and `pos * 1.0` is exact for integer
`pos` below 2^24.

The HIP `ggml_rope_multi` kernel computes theta in double and defers the
reduction (`rope.cu:15-29`, added explicitly because `freq_base = 1e7` is past
the f32 precision wall). So:

- A HIP-vs-host disagreement at long positions is HIP being right. Do not gate
  a HIP differential on host agreement at long `pos`, and do not tune the
  kernel to match.
- Tranche 1 **improves** numerics rather than risking them. At pos 262141 under
  YaRN the graph path is already closer to exact than the host scalar it
  replaces — 1.86e-3 vs 2.75e-3 — on the CPU backend alone, before the fp64
  theta the HIP kernel adds.

## DECIDED: prefill uses a margin criterion, not bit-identity

**User decision, 2026-08-31.** Option B of the three put to them. Recorded here
because it is a release criterion and belongs to the user, not to any agent;
do not relitigate it from the ledger.

### The criterion

> A prefill disagreement between batched and q1 is acceptable **only if** the
> q1 top-2 margin at the diverging position is smaller than the maximum
> absolute logit difference measured between the two paths on that same row.
> Otherwise it fails.

It is self-calibrating — there is no constant to pick or defend. If the two
paths differ by `d` in logits and the winning token led the runner-up by more
than `d`, the flip cannot be explained by the observed numerical difference and
is a real defect. If the margin was inside `d`, the argmax was tied at the
precision available and either answer is as correct as the other.

**CORRECTION 2026-08-31: it does weaken the test, and the claim that it did
not was mine and was wrong.**

When this was put up for decision I wrote that width 3 would still fail under
it. Checking the arithmetic against the evidence:

    q1 top-2   830 @ 19.5070915, 1543 @ 15.118576   ->  margin ~= 4.389
    batched    830 @ 13.4118                        ->  |delta[830]| ~= 6.095
    accepted = margin < max_abs  ->  4.389 < 6.095  ->  ACCEPTED

**The `sum_rows` defect would have passed this criterion.** (The fix changed
q1's reduction too, so the pre-fix q1 margin may differ slightly from 4.389 —
not by 1.7 logits.)

The flaw is that the criterion asks whether a flip is *explicable* by the
observed perturbation and accepts when it is. But a six-logit perturbation **is
itself the defect**, whatever the margin. It conflates explicable with
acceptable.

The missing clause is that the perturbation must be small in absolute terms:

    accepted = (max_abs_logit_delta < noise_bound)
            && (q1_top2_margin < max_abs_logit_delta)

`noise_bound` is not self-calibrating and needs a number — which is what the
decision was framed to avoid. It is derivable rather than invented: measure
`max_abs_logit_delta` on the widths that **pass**. Widths 2-5 are green, and
whatever perturbation they carry is the real floor for this model on this
hardware.

**This is back with the user**, who decided the criterion partly on the
assurance that it would still have caught `sum_rows`. That assurance was false.

Separately, a rewrite I proposed — compare the margin against
`|delta(expected)| + |delta(actual)|` — was refuted by codex and must not be
used: for greedy decoding `production[A] > production[E]` makes
`q1[E] - q1[A] < |delta[A]| + |delta[E]|` true by construction, so it accepts
every flip. A rubber stamp with the appearance of rigour.

### What stays bit-exact

MTP verification is unchanged. `qwen4exp_mtp.cpp:320-334` remains the authority
boundary: no token is committed from batched logits alone, and the accepted
prefix is replayed through q=1. That equality is load-bearing and this decision
does not touch it.

### Why prefill differs from MTP

- Upstream has **no q1 path at all**, so the equality is unaskable against the
  implementation that reaches 345 prefill.
- Nothing downstream of Ember's prefill consults a q1 prefill; prefill verifies
  nothing.
- Enforcing bit-identity would require capping prefill batching at physical
  width 5, since physical 16 cannot use MMVQ at all — which discards the
  batching the performance target depends on.

### Implementation

`qwen4exp_backend.cpp` already logs `top1`, `top2` and `margin` through
`log_numerics_top2`, so the inputs exist. The differential needs, per diverging
row: the q1 top-2 margin, the max |logit delta| between paths, and a verdict.
Report all three whether it passes or fails — a pass at a margin close to `d`
is worth seeing.

Widths 6 and 17 are then re-run under it. Their outcome is an *output* of this
criterion, not an input to it: if the margins are inside the deltas they pass,
and if they are not there is a second defect and the criterion has found it.

## Superseded: is bit-exactness the right criterion for batched prefill?

Raised 2026-08-31 after codex 344 measured the first divergence as *floating-
noise scale* in layer 0's recurrent state, compounding to an output flip only
by layer 2 row 2.

**The engine already documents this phenomenon.** `qwen4exp_mtp.cpp:320-327`,
in shipped code:

> The layer-major verifier is a proposal accelerator, not an authority
> boundary. A different reduction order can move a near-tied argmax even when
> every tensor/state update is otherwise valid. The gfx1151 Q3 differential
> caught exactly that after 29 emitted tokens...

The response there was **architectural, not numerical**: never commit a token
from batched logits alone, replay the accepted prefix through q=1 (`:326-334`).

**The analogy is not automatic, and the difference is the whole question.** MTP
verification produces a *proposal*, and a wrong one is caught by the q=1 replay
behind it. Prefill produces the *hidden state every later token depends on*,
with no replay behind it — so a merely differently-rounded batched prefill
still hands decoding a different starting state and nothing downstream catches
it.

So: does the width-3 differential assert bit-exactness between batched and q=1
prefill — a property this engine's own MTP design declines to rely on — or
something weaker that should be written down?

**Do not answer this from the ledger.** What should come first is both paths
compared against a **double-precision scalar reference**, the way the rope
numerics resolved the same shape of question (where the graph path turned out
*closer* to exact than the host scalar it was being measured against). If
batched prefill is closer to exact than three sequential q=1 steps, then q=1 is
the worse reference and the differential is comparing against the wrong side.
That test is GPU-free.

**It is live.** Codex 348 ran `DFLASH_CUDA_MMVQ_ROCMFP4_UNROLL2=0` and the
signature is unchanged: layer 0 output and conv state still exact, recurrent
state still first differing at head 12 with the same maximum absolute delta,
layer 2 row 2 still the first output divergence. That clears the q1
specialization, including the `ssm_alpha`/`ssm_beta` projection-shape hole.

**And the magnitude names the phenomenon.** That delta is `1.1920929e-07` =
`2^-23`, exactly one float32 ULP near 1.0 — the *floor* for two valid
roundings, not a defect magnitude. Two paths with different accumulation order
cannot do better. So "batched disagrees with q1" is no longer evidence of a
bug; it is evidence that they round differently, which they must.

### The instrument that can answer it

`18e1253` adds a **double-precision** chain of the same recurrence to
`test_gdn_batch_at_hip_legal_conv_channels`, printing

    [gdn-precision] batched_vs_exact=... serial_q1_vs_exact=... batched_closer=...

Comparing the two float paths to each other cannot say which is correct;
comparing both to double can. On the CPU backend they land on an identical
error, `6.24756508e-09`, because the CPU op has no register-versus-memory
distinction to expose — the control builds the instrument but cannot
discriminate there.

**Run `test_qwen4exp_frontier` on a HIP build.** The fixture already reaches the
S_v=128 kernel, so there is no model load and no production quiesce, and:

- `batched_vs_exact < serial_q1_vs_exact` → the batched path is closer to truth
  than the q=1 reference it is measured against, and the differential is
  comparing against the worse side — which is what happened with M-RoPE.
- `serial_q1_vs_exact < batched_vs_exact` → q=1 is the better reference, the
  batched path is genuinely drifting, and the question becomes how much drift a
  prefill may carry.

## What the first publishable number requires

Recorded now so the moment the blocker closes is not fumbled.

1. **The width-3 differential green**, with the `sum_rows` fix landed and the
   prediction confirmed: normalized Q and K exact, recurrent state exact. If Q
   and K go exact and the validator stays red, there is a second seam — the
   per-layer comparator names it the same way it named this one.
2. **A DS4 non-regression result.** `sumrows.cu` is shared with the production
   DeepSeek path (`deepseek4_graph.cpp:447`, `:458`, `:869`, `:1272`, `:2655`;
   `moe_hybrid_ffn_eval.cpp:81`). The removed branch fired at `nrows < 40`,
   the routing shape. A Qwen correctness fix must not silently cost DeepSeek
   throughput.
3. **A hard gate on an exact binary**, unprofiled, with prefill evaluating
   exactly 2074 tokens and decode exactly 256, rounding checks true, and the
   memory gate passed — the protocol codex 215 already followed.

Only then is a number publishable, and it should be published against the
right comparison: the measured cluster on this silicon is 22.6-28.1 decode and
345-385 prefill (see the reference implementation section). Our gates of 39.49
and 412 sit above that cluster, so the first green number will very likely be
a real result *and* short of the gate. Both things being true at once is the
expected outcome, not a contradiction, and the ledger should say so before
anyone has to interpret it under pressure.

## Candidate, unsized: asymmetric KV cache quantization

From a survey of other engines on this part. `julianmb/q38rocm` ships
**TurboQuant** — K at Q8, V at 4-bit (`-ctk q8_0 -ctv turbo4`) — reporting a
262K context dropping from 61.4 GB to 20.08 GB. It is a cache *format*, not a
shader, so unlike the Mesa RADV Wave64 dual-issue work it is portable to a HIP
engine.

**Ember's QSA KV cache is F32**: `Qwen4ExpCowBuffer` stores `float`
(`qwen4exp_internal.h:109-124`), and `qwen4exp_model.h:20` states it. At ctx
2048, where dense selection selects every token:

    per QSA layer   2 x 2048 x 2 heads x 256 dim x 4 B  =  8.4 MB
    x 12 QSA layers                                     =  101 MB per decode token
    at Q8 K / 4-bit V                                   =   19 MB  (5.3x less)

**This is a size, not a measurement, and must not be treated as a lever until
it is one.** The share of decode wall time spent in those uploads is unknown.
The measurement that decides it: time the `qsa_attend_q1` upload group
(`qwen4exp_frontier.cpp:1612-1648`) across a 256-token decode as a fraction of
decode wall time. Low single digits means leave it alone; a double-digit share
makes it the largest decode-side item on the list.

It also reactivates [`dead-code-candidates.md`](dead-code-candidates.md) entry
1: the #27774 Hadamard rotations are inert *because* the cache is F32, and they
are exactly what quantized KV requires.

### Corroborated, no action

- An independent Strix Halo comparison finds "the safe-core / no-graphs control
  barely changed generation", agreeing with dead-code entry 4 on HIP graph
  replay.
- `julianmb/q38rocm` measures 329.86 tok/s prefill at 16K on the HIP backend,
  sitting with kingjones777's 345/385. Our 412 gate remains above every
  published number on this part.
- Mesa RADV Wave64 dual-issue (`KHR_coopmat`) is Vulkan-only and does not port.

## The reference has no q1 path at all

`docs/reference/qwen4exp_upstream.cpp`, all 1193 lines, grepped for
`n_tokens == 1`, `n_seq_tokens == 1`, `== 1)`, `q1`, `single token`: **zero
matches**. There is one `build_arch_graph`, and a single token is a ubatch of
one through it. No q1 builder, no q1 arena, no q1 kernel selection — and
therefore no possibility of a q1-versus-batched disagreement. The question this
blocker asks cannot be posed against that implementation.

**So the bit-equality requirement is self-imposed**, and its two sources are
not equally load-bearing:

- **MTP verification** — real. A proposal accelerator needs an authority to
  check against, and `qwen4exp_mtp.cpp:320-327` already handles the argmax
  consequence architecturally rather than numerically. That stays.
- **Prefill** — a *test* choice. Prefill verifies nothing, and nothing
  downstream consults a q1 prefill. The width-N differential compares batched
  prefill against q1 stepping because we built it that way.

Width 6 is therefore not "the engine is broken at width 6" but "our prefill
test asserts an equality that upstream's architecture makes meaningless,
between two kernel families that exist because they differ".

**And the two threads are one piece of work.** Collapsing to a single graph is
the entire 345-prefill result, and it also dissolves this class of correctness
question, because there stops being a second path to disagree with. Every stage
moved into the graph is a stage that can no longer disagree with itself.
