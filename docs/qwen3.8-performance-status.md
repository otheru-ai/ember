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
bandwidth-bound. `copyBuffer` is undercounted: `cudaMemcpy2DAsync` lowers to
`__amd_rocclr_copyBufferRect`, a separate rocprof name not included above.

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

## Counter-unit correction (affects any older bandwidth figure)

ROCm 10 gfx1151 calibration (run 33288846711) measured `FETCH_SIZE` at 64-byte
and `WRITE_SIZE` at 128-byte transactions, relative RMSE ~0.0012. `AGENTS.md`
previously asserted KiB. Any bandwidth figure computed under the KiB assumption
is wrong by 16x/8x.
