# `wmma-ops` gfx1151 applicability audit

This is a source-contract audit, not an Ember performance result. The audited
source is `otheru/wmma-ops`, branch `perf/50tflops-frontier-research`, pinned at
commit [`c319068f651efb34788e77c7e6e61577d718e65d`](https://git.otheru.ai/otheru/wmma-ops/src/commit/c319068f651efb34788e77c7e6e61577d718e65d).
`git ls-remote` resolved the branch to that exact object on 2026-08-26. No
gfx1151 device was available for this audit, so Ember has not reproduced any
number below and makes no throughput claim from it.

The source repository describes itself as a performance laboratory, not a
drop-in BLAS implementation. Its own [current-status table](https://git.otheru.ai/otheru/wmma-ops/src/commit/c319068f651efb34788e77c7e6e61577d718e65d/README.md)
and [50-TFLOPS qualification](https://git.otheru.ai/otheru/wmma-ops/src/commit/c319068f651efb34788e77c7e6e61577d718e65d/docs/REPRODUCING_50_TFLOPS.md)
are authoritative for the contracts quoted here. The repository has no
top-level license. On 2026-08-26, the user clarified that OtherU owns the kernel
work hosted in this repository and supplied authority for Ember's internal
evaluation and porting. The absent license is therefore not a blocker in the
technical classifications below. It remains important for public distribution:
before Ember redistributes source derived from this work, OtherU should add an
explicit repository license and preserve separate provenance for the pinned
MIT-licensed upstream components used by the standalone FP16 harness.

## Result

There is no technically direct kernel import for the current Qwen4Exp or
DeepSeek q=1 paths. The useful outcomes are:

- an independent confirmation of Ember's gfx1151 WMMA fragment assumptions;
- schedule ideas for a future, genuinely batched FP16 prefill path;
- a paired-K experiment to compare with Ember's already-existing, lossy and
  off-by-default ROCMI4 W4A4 prefill path; and
- a profiling warning: large square GEMM peak is not a proxy for q=1 decode.

No adapter or vendored-engine source was added by this audit.

## 2026-08-27 superseding update

Two implementation facts changed after the original audit. Qwen now runs
persistent fused MoE, QSA, and GDN graphs at the q=1, q=5, and q=16 frontiers;
it is no longer accurately described as a tokenwise scalar-expert-only path.
Ember also has an off-by-default, numerically exact W4-by-A8 IU4 MMQ experiment
for ROCMI4. It reaches dense q=4/q=5 target verification and q=16 dense and
routed-expert MMQ. Routed experts at q=4/q=5 remain on MMVQ, while q=1 remains
outside MMQ entirely. The historical matrix and plan below remain useful as
provenance for the 2026-08-26 decision, but these newer facts control current
experiments.

The closest reusable frontier idea is now the paired-K/publication schedule in
the pinned remote `bench_wmma_iu4_gemm.hip`, applied only to the exact W4A8
experiment. That source is not a checked-in Ember device microtest. Do not
spend the release bakeoff budget adapting it to lossy W4A4. Qualification order
is strict:

1. correct and device-differentiate the exact W4A8 fragment loader against the
   existing exact int8 path, including nonuniform K lanes and checked/ragged
   tiles;
2. regenerate the ROCm 10 gfx1151 assembly and pass
   `scripts/check_rocmi4_w4a8_isa.py` with no scratch or spills;
3. measure existing exact int8 versus exact W4A8 on real dense q=4/q=5 and
   q=16 dense/routed-expert shapes, using q=5 routed experts as a negative
   dispatch control, plus the full 2,074-prompt/256-generation gate; and
4. only when counter profiling attributes time to LDS publication/barriers,
   screen two K16 slices before four, rejecting any occupancy loss or changed
   scale/float-accumulation contract.

The real-weight gate now records those dispatch controls in a separate
differential run. `scripts/qwen_w4a8_dispatch_evidence.py` fails closed when an
enabled startup variant lacks an actual exact-gfx1151 launch, when q=4 is not
bound to its completed cached-q5 dense scope, or when the q=1/q=5 negative
controls enter W4A8. That proves routing and launch selection only. Clean
timing and counter passes remain separate and are the only sources of
performance evidence.

The source-reported four-slice IU4 result (94.044 TOPS versus its 84.560-TOPS
one-slice control) is evidence for that schedule experiment, not an Ember speed
claim. Saved ROCm 10 gfx1151 assembly shows that activation prepacking reduces
the screened kernels' VGPR counts. Their source- and launch-derived 27,776-byte
dynamic LDS allocation allows at most four workgroups per 128-KiB WGP, an
eight-wave-per-SIMD LDS upper bound; the compiler's higher register-only
occupancy therefore does not establish extra resident workgroups.
The large prepacked `4096^3` FP16 kernels still do not match Qwen's quantized
q=1/q=5/q=16 text shapes. QSA bank interleaving/padding and physical VGPR phase
placement remain separate, profile-led experiments; neither may copy constants
or a code object from the laboratory kernel.

## Applicability matrix

Classification means:

- **direct reuse**: the numerical, memory-layout, shape, and ABI contracts
  already match;
- **adapter**: the arithmetic can plausibly be reached through Ember's ggml/HIP
  seam after bounded glue and a new correctness/performance gate;
- **research-only**: a useful schedule or architecture lead, but adopting it
  changes a numerical/layout contract or needs substantial kernel work; and
- **no fit**: it does not execute the workload represented by the Ember call
  site.

| Remote kernel or result | Precision, layout, and measured shape | Source-reported result | Ember call site | Class | Blocking facts and next proof |
|---|---|---:|---|---|---|
| Public `wmma_gemm_kernel*` / PyTorch extension | Ordinary FP16 A/B, FP32 accumulation and output; tiled GEMM, historically reported at `4096^3` | 21.6 TFLOPS historical extension result | Qwen dense `matvec()` in `engine/dflash/qwen4exp/qwen4exp_runtime.cpp`; DeepSeek `ggml_mul_mat` projections in `engine/dflash/deepseek4/deepseek4_graph.cpp` | **adapter** for future batched prefill; **no fit** for current q=1 | Qwen supplies one F32 activation row and quantized/F16/BF16 weights through ggml, not two FP16 matrices through PyTorch. Current Qwen prefill is tokenwise q=1. DeepSeek can have batched prefill, but its shapes and quant dispatch differ. A candidate needs a C/ggml dispatch seam, model-shape sweep, full numerical comparison, and unprofiled end-to-end timing. |
| Pinned upstream `rocm_wmma_gemm` ordinary-layout harness | FP16 A/B, FP32 accumulation/output, `4096^3`; every output compared with rocBLAS | Same validated 41.322-TFLOPS peak contract | Same dense projection sites | **research-only** | The harness is reproducible evidence for one large shape, not an Ember API. Its separately MIT-licensed upstream kernel can be audited independently, but the record schedule is not a q=1 kernel and its `4096^3` specialization is not representative of Ember's rectangular projections. |
| Block/K16-prepacked persistent-input FP16 kernel | Both A and B prepacked before timing; FP16 WMMA accumulation and FP16 output; `4096^3`, 256x128 block, K16, eight waves | 50.073879 TFLOPS average, five-process floor 50.014850 | Future batched Qwen/DeepSeek dense prefill only | **research-only** | Packing is excluded, both operands must use a private layout, and the numerical output contract changes. Model weights can persist, but activations are request-dependent. The source's own B-only persistent-weight experiment reached 20.968 TFLOPS, showing that the two-input contract is load-bearing. Any persistent weight repack also consumes UMA and must pass Ember's 128-GiB residency planner. |
| Upstream `bench_half_half` control | Ordinary FP16 input/output, `4096^3` | 46.082 TFLOPS median | Comparison control for a future dense-prefill study | **no fit** | This is an external comparison with a different output contract, not a new Ember kernel. It gives no q=1, quantized-weight, or end-to-end result. |
| `bench_wmma_iu4.hip` | Signed linear INT4 operands, INT32 accumulate; independent register-resident WMMA chains, no GEMM data movement | 110.229 INT4 TOPS with 16 chains | Ember gfx1151 IU4 primitive in `engine/ggml/src/ggml-cuda/mma.cuh` | **no fit** as a runtime kernel; ISA oracle only | It measures instruction issue, not weight reads, activation quantization, scales, routing, or output conversion. Its fragment/result mapping is useful for a focused device test, but Ember already uses the same builtin behind `GGML_HIP_ROCMI4_W4A4=ON`. |
| `bench_wmma_iu4_gemm.hip`, one-slice pipeline | Both operands prepacked linear signed W4A4, INT32 output, `4096^3`, 128x128 block, K16 | 85.907 INT4 TOPS, exact full INT32 output | Optional ROCMI4 W4A4 MMQ in `engine/ggml/src/ggml-cuda/mmq.cuh` and activation quantization in `quantize.cu` | **research-only** | Ember's stored weights are 32-value ROCMI4 blocks with a UE4M3 scale, activations arrive as floats and are quantized with scales, and output is rescaled to float. The remote kernel has no such scale path and prepackages both matrices. W4A4 is already a separate lossy release mode in Ember, not the exact decode path. |
| Paired-K / four-slice IU4 pipeline (`IU4_PAIR_K=2`) | Same prepacked linear W4A4/INT32 contract; four K16 slices in eight rotating LDS slots | 94.044 INT4 TOPS average, five-process floor 93.445 | Same optional ROCMI4 W4A4 MMQ prefill path | **adapter experiment** | The publication cadence and LDS ring can be adapted inside Ember's existing off-by-default W4A4 MMQ without changing GGUF storage. The remote layout, scale handling, tile geometry, and square workload still do not match, so the bounded experiment below preserves Ember's pack/scale arithmetic and changes only K-slice staging. It says nothing about default exact MMVQ decode. |
| `rocwmma_gfx1151.hpp` fragment helpers and fragment-layout notes | Wave32 FP16 WMMA: 16 A/B elements per lane, lanes 16-31 replicate 0-15; FP32 D has eight elements/lane with even/odd row split | Correctness/layout reference, not a throughput result | `engine/ggml/src/ggml-cuda/mma.cuh` and `tools/bench_wmma_decode.hip` | **adapter** as a reference, not source | The mapping agrees with Ember, so no correction is needed. Ember already consumes native builtins through its own ggml tile ABI. Replacing that ABI with the helper is unnecessary; retain the helper as an external oracle when changing fragment loads. |
| Decode-attention findings document | DeepSeek MLA q=1: 64 Q heads, one K/V head, D=512, FP16 K, context 128 to about 33k; scalar-F32 attention study | 512.1 us / 142 GB/s at 8,896 rows; source reports about 148 GB/s achievable for that experiment | DeepSeek D=512 `ggml_flash_attn_ext` / explicit reduction in `deepseek4_graph.cpp`; local `tools/bench_wmma_decode.hip` | **research-only**, and already represented locally | The remote document points back to Ember's local standalone source; it contributes analysis, not an importable kernel. Ember's production path has sparse/compressed-cache and inverse-RoPE semantics beyond the benchmark. Revalidate against current full-ring/sparse paths and the profiler's separate measured roofline before any promotion. |
| Decode-attention WMMA proposal | FP16 Q/K WMMA, proposed larger head grouping; Q would be demoted from F32, with optional hi/lo recovery | No completed WMMA result is claimed | DeepSeek D=512 attention above; Qwen QSA CPU loops in `qwen4exp_runtime.cpp` | **research-only** | DeepSeek requires softmax, sinks, raw/compressed boundaries, inverse RoPE, and exact restore parity. Qwen differs materially: 24 Q heads, two K/V heads, D=256, a separate four-head D=128 indexer, and a 2,048-token selection budget (top-512 complete four-token blocks, with the incomplete causal tail handled separately) rather than one dense D=512 latent span. gfx1151 cannot co-execute this WMMA with VALU, so CDNA scheduling advice does not transfer directly. |
| Large FP16 GEMM schedules applied to the Qwen vision tower | Remote promotion data is `4096^3`; Qwen vision uses hidden width 1,152 and intermediate width 4,304, with request-dependent patch-row M | No matching remote shape is measured | Future encoder-provider implementation behind `qwen4exp_vision_loader.cpp`; the text backend currently reports vision unsupported | **adapter** for a future batched vision graph only | Neither 1,152 nor 4,304 matches the promoted square/tile contract, and the number of patch rows varies by image. The current source tree validates the 334-tensor/processor seam but has no installed HIP vision encoder. Require real ViT projection/attention shapes, ordinary-layout packing-inclusive timings, 2x2 merger correctness, and image quality tests. The q=1 text result is unaffected. |
| Hand-assembled hot-B register-phase schedule | Exact block/K16 FP16 prepacked code object; 18,432-byte LDS, 120 VGPR, 22 SGPR, two blocks/16 waves per CU | Schedule underlying the qualified 50.074-TFLOPS contract | No current Ember call site has this exact geometry/contract | **no fit** | Physical register placement and instruction order are tied to one compiler image and exact tile. Use it only as evidence that register phase can matter after a matching Ember kernel is already measured hot; it is not a source-independent adapter. |

There is deliberately no **direct reuse** row: no audited artifact clears every
technical and numerical contract.

## Why the headline rates do not predict Ember q=1

A `4096^3` GEMM performs about 137.4 billion floating-point operations. With
ideal one-time traffic, FP16 A/B/C is roughly 100.7 MB, or about 1,365 FLOP per
byte; FP16 A/B plus FP32 C is roughly 134.2 MB, or about 1,024 FLOP per byte.
Those high-reuse workloads can be matrix-compute bound.

A q=1 matrix-vector product instead consumes a large weight matrix for one
activation row. Ignoring scales and every other cost, FP16 weights provide only
about 1 FLOP per byte, and 4.25-bit ROCMI4 weights provide about 3.76 FLOP per
byte. There is too little row reuse to approach the square-GEMM arithmetic
intensity. Qwen's current runtime compounds this with one-op graph creation,
host synchronization, scalar expert dequant/dot products, GDN recurrence, and
CPU QSA selection. DeepSeek's default q=1 path uses MMVQ/fused MoE/attention
paths selected for its quantized and bandwidth-sensitive shapes, not the
ordinary FP16 square-GEMM path.

The persistent-prepacked results also cannot be imported as a second full
weight representation. The released Qwen artifact is already roughly 94 GB at
4-bit weight precision; Ember's 128-GiB planner reserves capacity for runtime
and native-context QSA/index state. A duplicate full-model packed copy cannot
fit that budget, and an FP16 packed copy would be larger still. A future layout
must replace a certified resident representation tensor-by-tensor, or use a
bounded hot subset whose aggregate plan still passes before allocation.

Consequently, the source's 41/50 TFLOPS and 86/94 TOPS results can prioritize
experiments, but they cannot be converted into tokens/second or an Ember
speedup. Timing must use the real artifact, request shape, implementation path,
and unprofiled end-to-end recorder; counters belong to a separate pass.

## Fragment-layout cross-check

The remote [fragment-layout document](https://git.otheru.ai/otheru/wmma-ops/src/commit/c319068f651efb34788e77c7e6e61577d718e65d/docs/wmma_fragment_layout_rdna3.md)
matches Ember's verified gfx1151 facts:

| Fragment | gfx1151 wave32 mapping | Ember evidence |
|---|---|---|
| FP16 A | Lane `L` holds row `L % 16`, all 16 K values; lanes 16-31 replicate lanes 0-15 | Header and loads in `tools/bench_wmma_decode.hip`; `halfx16_t` operand in `engine/ggml/src/ggml-cuda/mma.cuh` |
| FP16 B | Lane `L` holds column `L % 16`, all 16 K values; same half-wave replication | Same local benchmark and builtin wrapper |
| FP32 C/D | Eight floats per lane; lanes 0-15 cover even rows, lanes 16-31 odd rows | Local benchmark's `D[2*i + lane/16][lane%16]` mapping; `floatx8_t` accumulator wrapper |
| IU4 A/B | Two packed 32-bit VGPRs per lane for 16 nibbles, replicated by half wave | Ember's opt-in `mma_iu4()` uses two `int32x2_t` products and an `int32x8_t` accumulator |

This is RDNA 3.5 behavior. The gfx12/RDNA 4 branch uses eight FP16 elements per
lane without the same replication rule and must not be substituted. Ember's
vendored engine deliberately excludes its unsupported gfx12 path.

## Bounded paired-K ROCMI4 W4A4 experiment

The technically closest port is the four-slice IU4 publication schedule, but
only inside Ember's already lossy, gfx1151-only, off-by-default W4A4 MMQ. It is
not a replacement for ROCMI4 storage and it does not apply to exact q=1 MMVQ.
No kernel was changed during this audit: changing shared-memory publication and
barrier cadence without a device correctness run is not a safe source-only
edit.

The bounded implementation plan is:

1. **Keep the numerical contract fixed.** Preserve
   `load_tiles_rocmi4_w4a4()` and
   `vec_dot_rocmi4_w4a4_wmma()` in
   `engine/ggml/src/ggml-cuda/mmq.cuh`, including 32-value ROCMI4 blocks,
   signed-nibble packing, UE4M3 weight scales, activation scales, and float
   write-back. Preserve `mma_iu4()` in `mma.cuh` and both activation-quantizer
   entry points in `quantize.cu`. The remote linear-INT4 prepacked format must
   not become a second GGUF or resident weight format.
2. **Add one nested experimental schedule.** Beside
   `mul_mat_q_process_tile()` in `mmq.cuh`, retain today's load/synchronize/
   `vec_dot` sequence as the control and add an opt-in ROCMI4-W4A4-only
   candidate with rotating LDS slots. First establish the mapping between an
   Ember packed half-tile and the remote K16 slice with static assertions and a
   device oracle; do not assume `QI8_0` is a K16 publication unit. Screen a
   two-slice pair before the four-slice/eight-slot form. Do not alter generic
   MMQ, exact ROCMI4 MMQ, MMVQ, or another architecture's specialization.
3. **Keep dispatch visibly experimental.** The outer
   `GGML_HIP_ROCMI4_W4A4=OFF` default remains authoritative. A separate nested
   build option should select candidate versus control on exact gfx1151 so the
   two implementations can be built and bracketed from the same source
   revision. Record the pinned OtherU commit and this ownership clarification
   in `engine/VENDOR.md` if source is ported. Before a public source release,
   also record the explicit OtherU license.
4. **Use real Ember shapes, not only `4096^3`.** At minimum cover DeepSeek
   grouped top-6 expert projections `4096x2048` and `2048x4096`, then future
   batched-Qwen expert projections `2560x1280` and `640x2560`. Sweep activation
   columns including small/ragged and production-prefill cases (for example 2,
   7, 32, 128, and 512), expert-ID indirection, final partial tiles, and the
   actual `mmq_x/mmq_y` selections. Qwen's current tokenwise runtime is a
   negative control because it must not start dispatching this MMQ candidate.
5. **Gate correctness before timing.** Compare candidate and current W4A4
   control on zero, alternating, extremal signed-nibble, random, and finite
   UE4M3-scale inputs. Require identical INT32 tile results and bitwise-equal
   float output where accumulation order is preserved; if scheduling changes
   float addition order, set a reviewed tolerance and require greedy-logit
   differential parity. Exercise ordinary and grouped quantizer strides and
   confirm a build with W4A4 disabled is unchanged. Then rerun the independent
   W4A4 perplexity, behavior, vision, and runtime certification gates; parity
   with the existing lossy mode is required, not parity with the remote linear
   microbenchmark.
6. **Gate performance without a target claim.** On the exclusive gfx1151 host,
   alternate candidate/control fresh processes, time the complete MMQ operation
   including activation quantization and scale/write-back, and profile counters
   separately. Inspect code-object VGPR, LDS, spills, waves, and achieved
   occupancy. Promote only if real model shapes improve outside run noise and
   unprofiled end-to-end prefill also improves; reject the schedule if extra LDS
   reduces occupancy or q=1/default behavior changes.

The candidate must reuse the resident ROCMI4 bytes in place. Ember's 128-GiB
planner forbids solving the layout problem by retaining a duplicate packed
full-weight model.

## Profiling decisions

The first useful gfx1151 run is not another `4096^3` replay. Use
`scripts/qwen4exp_baseline.py` for an unprofiled Qwen end-to-end record and
`scripts/profile_gpu.sh` for separate timing/counter passes. Rank source work by
total phase contribution:

1. measure current q=1 routed-expert MMVQ/scalar fallback and QSA/index traffic;
2. measure actual batched DeepSeek prefill GEMMs by `(M,N,K,type)` before
   considering the ordinary FP16 schedule;
3. only if opt-in ROCMI4 W4A4 prefill is both quality-certified and hot, compare
   Ember's K-loop with the one-, two-, and four-slice IU4 publication cadence;
4. measure total packing/activation-quantization and any persistent-layout UMA
   cost, never just the matrix instruction interval; and
5. preserve full-output/reference checks and fresh-process brackets for every
   promoted kernel result.

The concrete kernelization order for the correctness-first Qwen runtime remains
the one in `docs/qwen3.8-performance-baseline.md`: selected-expert execution,
persistent q=1 graphs, GDN recurrence, then native-depth QSA, reordered only by
measured total contribution. The audited large-GEMM source does not overturn
that bytes/operations analysis.
