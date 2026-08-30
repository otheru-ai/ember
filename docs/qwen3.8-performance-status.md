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

q=1 and batched prefill disagree from batch width 2. Root cause isolated to
choosing MMQ at physical width q5: `LUCE_MMVQ_MAX_NCOLS=5` makes seed and both
AR steps bit-identical. Our default of 3 is an inherited sm_86 (RTX 3090)
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

Logging `src0->type`, `src1->ne[1]` and the chosen kernel at
`ggml-cuda.cu:2582-2585` for logical widths 2, 3, 6, 17 separates them in one
run. Until it does, treat "`LUCE_MMVQ_MAX_NCOLS=5` makes it bit-exact" as a
measurement whose mechanism is not yet established.

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
