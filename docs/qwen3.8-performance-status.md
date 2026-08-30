# Qwen3.8-Flash-Next performance status

Measured status against the parity target. This is a ledger, not a claim: every
row states the exact commit, recipe, and what invalidates it. Update it when a
measurement lands; do not delete superseded rows, mark them.

Recording methodology is `docs/qwen3.8-performance-baseline.md`. The target
figures come from `docs/performance.md` (DeepSeek-V4-Flash on the same
gfx1151 host).

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

## Open correctness blocker

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
| `sync_fallback` | refuted — 0 dispatches, measured twice |

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

**The synthetic half of that run proved nothing, and the fixture was at fault.**
`launch_gated_delta_net` switches on `S_v` (`gated_delta_net.cu:397-440`), so
the control's `head_dim = 16` selected `gated_delta_net_cuda<16, ...>` — a
separate template instantiation sharing no code with the S_v=128 path, with the
grouped-cols specialization not even compiled into that call. Green with and
without the guard is exactly what a fixture that never reached either kernel
would produce. Fixed in `4e9a6aa`: `head_dim = 128`, 1024 conv channels (still
`% 128 == 0`), both properties asserted in the test.

### The S_v=128 kernel passes on HIP at n=3 with non-zero state

Codex 341, run `qwen-gdn-sv128-unit-3e2047c`, on the corrected fixture
(`4e9a6aa`): the HIP graph matches the scalar reference at n=3 **with non-zero
initial state** — the condition the zero-state argument says is necessary to
observe the fault at all.

So at 4 heads / 2 key heads / 1024 conv channels, the S_v=128 GDN kernel, its
`ssm_conv` window, the q/k/v views and the state carry are all **correct on
real hardware at the failing width**. Combined with grouped-cols being
exonerated, that is the kernel largely cleared.

What the control does *not* cover, and therefore what is left:

- **Scale.** 4 heads against 48, 2 key heads against 16, 1024 conv channels
  against 10240, `n_embd` 8 against 2560. Grid extent in the grouped kernel is
  `(H, n_seqs, ...)`, so `H` is the one axis that changes with head count.
- **Real data.** The fixture uses patterned weights. A data-dependent fault —
  range, overflow, a denormal — would not show.
- **Depth.** The real path runs 36 GDN layers. A per-layer-exact kernel can
  still produce a wrong answer if something accumulates across layers, and
  nothing so far distinguishes "one GDN call is wrong" from "GDN calls are
  right and something between them is not".

The per-layer comparator (codex 339, reviewed and approved) separates exactly
those: it holds `attention_inputs` fixed and varies only batched-versus-serial,
per layer, on the real model. Its first diverging layer is the answer; later
layers are fed contaminated inputs and are not independent evidence.

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
