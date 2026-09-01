# DeepSeek-V4-Flash Vision status ledger

This is the single source of truth for measured DeepSeek-V4-Flash-Vision-Exp
correctness and performance results. Do not copy these numbers into backlog,
coordination, release, or criteria documents. Raw evidence remains at the
named retained path.

The release criteria are fixed separately in
`docs/vision-benchmark-criteria.md`. A green row here does not revise those
criteria or imply that a later gate has passed.

## Current gate state

| Gate | State | Reason |
|---|---|---|
| C1 text quality parity | not measured | The full-model text comparison is still open. |
| C1 text throughput parity | not measured | No timing run is admissible before its correctness prerequisites. |
| C1 text/vision routing inertness | not measured | Requires the full text-path gate on the release artifact. |
| C1 one-layer BF16/quant discriminator | void | The identically quantized known-good 0731 control is similarly gross, so the one-layer method cannot localize a Vision-specific defect. |
| C1 type-107 reader/writer compatibility | **artifact defect confirmed** | The Vision artifact writer emits a two-scale layout while every selected Ember CPU, MMQ, MMVQ, and conversion reader uses affine scale/offset. Actual coherent production 0731 was written affine and matches Ember; the earlier contrary control used a new fixture from the same bad writer. |
| C2 block-0 numeric differential | **green** | All 13 records pass the corrected compounding budgets at `770d2b2`. |
| C2 tower-depth numeric differential | diagnostic only | The reference is not a numerical authority after block 0. |
| C2 full-depth native execution | **green** | Four image requests and one text-only control completed on the 43-layer artifact at `5df70f6`. |
| C2 synthetic behavioural differential | deliberately withheld | Policy v1 is preregistered, but the current artifact is already incoherent on text; scoring it would not diagnose vision. |
| C2 natural-image behavioural differential | deliberately withheld | The digest-bound policy is retained for an artifact that could plausibly pass its text prerequisite. |
| C2 image throughput | not measured | Correctness is not complete. |

## 2026-09-01 — native block-0 numeric gate

This was a correctness-only run. It collected no timing or throughput data.

### Identity and method

- Ember commit: `770d2b2`
- Evidence: `/tmp/ember-vision-native-block0-731.XhRhcY` on `otheru`
- Binary SHA-256:
  `a4ae12bc3aaf49efca67b6f2dca98b2316025ff4f722c2caad9c639f6d4a982e`
- Source archive SHA-256:
  `cb9ce655246be1a75cbe257348676bf35f79f48ae71f6e6f748447bb3cc1d42c`
- mmproj SHA-256:
  `9225c5562c05bd910245ab24c9274ca777eba2a801990f47ebe0c6344f144002`
- Raw patch SHA-256:
  `560aab9a6fc1a99b205dad8a7942d4435c2c35a54013f5e610460bb08b19f7c2`
- BF16 block-oracle SHA-256:
  `edb2b3aa1bdf38828a06a532085a2d130cea18b0a3b560cdf6e409fc8fecb326`
- F32 block-oracle SHA-256:
  `9d3f515dff0db962d8afbb31de2d80e91782f041acb49bab9f0fcd161b10463b`
- Budget producer: `otheru-quant-pipeline@e619204`, corrected block-input
  injection, anchor `6e-5`, six directions, worst-of-N multiplied by `1.5`.
- Device: gfx1151 AMD Radeon 8060S Graphics under the retained host record.

The verdict is first-red through `norm2_out`. The final four MLP records still
fail closed, but a failure there is labelled `budget-exceeded` because the
budget no longer localizes a known-size defect. No record failed in this run.

| Block-0 record | Engine vs BF16 rel L2 | Budget | Consumption | Result |
|---|---:|---:|---:|---|
| `norm1_out` | 1.27358643e-05 | 3.161e-05 | 0.402906178 | pass |
| `qkv_biased` | 5.3923188e-05 | 1.738e-04 | 0.31026 | pass |
| `q_roped` | 4.75761193e-05 | 1.848e-04 | 0.257446533 | pass |
| `k_roped` | 5.88615166e-05 | 1.847e-04 | 0.31868715 | pass |
| `v_in` | 9.83692164e-05 | 3.142e-04 | 0.313078346 | pass |
| `sdpa_out` | 1.50197666e-04 | 4.987e-04 | 0.301178395 | pass |
| `wo_biased` | 1.81111729e-04 | 4.515e-04 | 0.401133398 | pass |
| `post_attn_residual` | 3.56469089e-04 | 8.579e-04 | 0.415513566 | pass |
| `norm2_out` | 3.9668047e-04 | 1.054e-03 | 0.376357182 | pass |
| `mlp_gate` | 6.96249745e-04 | 1.540e-03 | 0.452110224 | pass |
| `mlp_up` | 6.8023853e-04 | 1.501e-03 | 0.453190227 | pass |
| `mlp_silu_act` | 9.98065905e-04 | 1.975e-03 | 0.505349825 | pass |
| `mlp_down_out` | 1.05122839e-03 | 2.000e-03 | 0.525614196 | pass |

### Tower-depth diagnostics from the same run

These records are retained for diagnosis only. They do not gate C2 because the
official BF16 and F32 reference lanes diverge too far at tower depth for either
lane to be a numerical authority there.

| Tower record | Engine vs BF16 rel L2 | BF16 vs F32 rel L2 | Ratio | Policy |
|---|---:|---:|---:|---|
| `post_patch_projection` | 1.23500535e-05 | 1.60245326e-03 | 0.00770696643 | diagnostic only |
| `post_block_0` | 9.90568248e-04 | 3.85666808e-03 | 0.256845606 | diagnostic only |
| `post_vit` | 9.76215826e-02 | 1.01251956e-01 | 0.964145154 | diagnostic only |
| `post_aligner` | 6.1578665e-02 | 7.16612282e-02 | 0.859302395 | diagnostic only |

## 2026-09-01 — full-depth native execution and tower latency

This run did **not** execute either arm of the behavioural policy. The loaded
artifact was already known to be incoherent on text under both sparse and exact
prefill, so output text was retained only as an opaque response digest. Four
image requests and one text-only request each returned HTTP 200 with a positive
completion-token count. That establishes tower execution, learned-block
insertion, image routing through all 43 language layers, prefill, and decode;
it does not establish answer quality.

### Identity and method

- Ember commit: `5df70f6`
- Evidence: `/tmp/ember-vision-e2e-full-748.qrpY0s` on `otheru`
- Binary SHA-256:
  `bad79f3e33c9224fc9d87475bb7b81c1d9f9a51393838aa6de24041424b44c06`
- Source archive SHA-256:
  `051b5528bdaa11c91bd3c3284c925ef5d0bc23c624cff605052b8375798754b1`
- Model: `/srv/models/DeepSeek-V4-Flash-Vision-Exp-ROCMFP4FAST-affine.gguf`
- Model SHA-256:
  `371721f46e25db6bb54e0e7f5dbe9bfc0e5978acf035bcf15e7bfd9a1555fb9a`
- mmproj SHA-256:
  `9225c5562c05bd910245ab24c9274ca777eba2a801990f47ebe0c6344f144002`
- Image: `count_000.png`, 448x448 RGB PNG, SHA-256
  `fd1959b422e2baa18e2ccb3202cb19fb0868989a2e740a0bd279794d918512b3`
- Grid: ViT 32x32; LLM 11x11.
- Method: `steady_clock` around `deepseek4_run_vision_tower`, one discarded
  warmup followed by three serial samples on the same image. Production was
  quiesced and no competing GPU workload was admitted.

The boundary includes per-image tower graph construction and allocation, input
transfer, all 32 vision blocks, synchronization, checkpoint readback, learned
marker assembly, and cleanup performed inside `deepseek4_run_vision_tower`.
It explicitly **excludes** PNG decode, preprocessing, lazy mmproj load, time
waiting for the tower mutex, language prefill, and decode. It is therefore a
standalone tower-compute measurement, not the end-to-end latency experienced by
an image request.

| Warmup (discarded) | Sample 1 | Sample 2 | Sample 3 | Median | Range |
|---:|---:|---:|---:|---:|---:|
| 511.561 ms | 494.950 ms | 465.161 ms | 495.279 ms | 494.950 ms | 465.161–495.279 ms |

This grid-dependent result is a **floor, not a typical image latency**. The
32x32 ViT grid is near the bottom of the admissible range by construction;
`vision_max_n_token` constrains the LLM grid after the 3x pixel shuffle and
does not cap the ViT patch grid, so larger natural and real-request images
perform more tower work.

Sample 1 is not a first-sample outlier after warmup: it is effectively the same
as sample 3, while sample 2 is the low observation. This run therefore gives no
evidence that graph construction or allocation has an additional unamortized
first-measured-sample cost beyond the discarded warmup.

## 2026-09-01 — one-layer production-quant discriminator

This is a correctness discriminator, not a performance measurement. The BF16
and production-quant artifacts came from the same converter run and have
identical 32-tensor name/shape inventories. The only artifact transformation
between them is the production quantization step. Separate processes emitted
one exact-q1 full-vocabulary row for each artifact using the same probe binary
and frozen token sequence.

Attempt 754 is void and contributes no value here: its reused CMake directory
embedded a stale revision. This section uses only the fresh-build rerun.

**VOID methodology:** the identically quantized known-good 0731 control below
is similarly gross. A one-layer model has no depth over which weight error can
average, and its quantized `output.weight` perturbs the compared logit row
directly. The pre-registered bands classify distance but are not a valid
correctness gate for this construction. All causal and localization readings
from the Vision-Exp logits, layer-0, and selective-expert comparisons are
withdrawn. Their raw observations remain for audit only.

### Identity and method

- Ember commit: `326255b07d37843399b84a3da1b3cf0154b79130`
- Evidence: `/tmp/ember-vl-quant-logits-756.zRtkaG` on `otheru`
- Probe binary SHA-256:
  `dd05a1a116650c41c05286ecc6c822fc15a3dd62d50ffa05ea364eb7a97648d8`
- BF16 model SHA-256:
  `68ecfaaf8d9bfafa402279b4642fb80f3843c86962e60f541dd75503a76c0b9c`
- Production-quant model SHA-256:
  `43b6d6cb8fa7ca509ae48d4598a821b84dde341f1311d9ea0b590fa3069a155d`
- Tokens: `[1, 1000, 5000, 20000, 40000, 70000, 100000, 129279]`;
  compared row predicts the token after the final supplied ID.
- Both manifests assert exact q1, effective prefill chunk 1, monolithic GPU
  placement, vocabulary width 129280, the same binary digest, and the exact
  current 40-hex Ember revision.
- Pre-registered reading from msg 659: relative L2 below `5e-2` is consistent
  with ordinary quantization; above `1e-1` is a finding; the interval between
  those bounds is unresolved.

| Comparison | Relative L2 | Cosine | BF16 argmax | Quant argmax | Top-10 overlap | Result |
|---|---:|---:|---:|---:|---:|---|
| BF16 vs production quant, exact q1 | 8.526334999767808e-01 | 6.39186154175101e-01 | 39935 | 17174 | 1/10 | gross under uncalibrated band |

The result is far beyond the pre-registered numerical boundary, but without the
calibration supplied by the known-good control it does not establish that the
production-quant model path is defective. It also cannot distinguish artifact
quantization from type-specific runtime execution.

### Known-good 0731 calibration that voids the method

Evidence `/tmp/ember-0731-one-layer-control-771.vmseRi` emitted fresh exact-q1
rows for both 0731 fixtures with the same probe binary used above. The BF16
model SHA-256 is
`dcbdc45f0faf7e36968b46f165192071910e5cad1d78f34814f8cd50f448b3b4`;
the production-quant model SHA-256 is
`3a85716d520e08c7a255617a5a7cc8d1cb0fe310afea65258dde045a7ff15208`.
Both manifests bind the frozen tokens, row, vocabulary width, current revision,
same binary, monolithic placement, and observed q1 chunk. Model stat identities
were unchanged across the run, and production remained live and healthy.

| Comparison | Relative L2 | Cosine | BF16 argmax | Quant argmax | Top-10 overlap | Result |
|---|---:|---:|---:|---:|---:|---|
| known-good 0731 BF16 vs production quant, exact q1 | 8.304892828205547e-01 | 6.556967254553592e-01 | 19454 | 123327 | 1/10 | gross; method void |

This matches Vision-Exp's gross regime and rank-collapse signature. Under msg
664's reading, the one-layer method is therefore void rather than a minimal
reproducer. No Vision-specific defect, tensor-family attribution, or artifact-
versus-runtime conclusion follows from any comparison in this section.

The production type assignment is also not a differentiator: the coherent
full-depth 0731 artifact and incoherent Vision-Exp artifact use the same types
for HC projections, routing inputs and biases, output and embedding weights,
attention output-B, and experts. In particular, keeping the HC projections in
ROCMFP4_FAST cannot explain the Vision-specific failure because coherent 0731
does the same. The remaining quantizer-side lead is instead the provenance of
the importance matrix: the Vision recipe currently consumes one produced for a
different checkpoint. That hypothesis requires a paired CPU-only weight-space
round trip with and without the matrix, plus the same 0731 control, before any
engine run.

### Passive layer-0 boundary split

Evidence `/tmp/ember-vl-layer0-split-761.JBEHO5` re-ran `layer0-q1` once per
artifact, each bound to its own exact-q1 authority bundle above. In both arms,
ordinary logits and capture-enabled logits were byte-identical to the retained
authority payload. The passive checkpoint therefore changed neither logits nor
reset state.

The boundary reading was frozen before this run: relative L2 at or above
`1e-1` means damage is already gross before the output head; at or below
`2e-2` is close and localizes downstream; the open interval is inconclusive.
The close boundary comes from approximately twice the already measured
same-model Ember-vs-CPU checkpoint floor.

| Checkpoint | Width | Relative L2 | Cosine | Result |
|---|---:|---:|---:|---|
| `post_layer_0_mean_hc` | 4096 | 7.059920265555953e-01 | 7.833478043036031e-01 | gross under uncalibrated band |

The distance is already present at the output of layer 0, but the known-good
control demonstrates that this one-layer construction cannot turn that fact
into a Vision-specific localization. No component is localized from this
value.

### Selective routed-expert type-path isolation

Evidence `/tmp/ember-expmxfp4-isolation-765.gzI4vj` used model
`dsv4-vision-1layer-vl-EXPMXFP4.gguf`, SHA-256
`297ebede2b5b4fc542216e528bbaf32f74980705e872ffa824a54976c6465f68`.
Payload-range hashing before the run established that its three routed-expert
tensors are type/shape/byte-identical MXFP4 to the BF16 fixture, while all 29
non-expert tensors are type/shape/byte-identical to the production-quant
fixture. This is therefore an exact one-family intervention. The probe binary,
revision, token sequence, exact-q1 contract, and retained BF16 authority are
the same as in the preceding discriminator. Production remained live and
healthy throughout; this correctness run did not quiesce it.

The logits bands from msg 659 and layer-0 bands from msg 662 were retained
unchanged. The capture-enabled run reproduced the selective artifact's
authority logits byte-for-byte before its checkpoint was interpreted.

| Comparison | Relative L2 | Cosine | Result |
|---|---:|---:|---|
| BF16 vs selective EXPMXFP4, exact-q1 logits | 2.0289609523323635e-01 | 9.793404476355955e-01 | gross |
| BF16 vs selective EXPMXFP4, `post_layer_0_mean_hc` | 1.5663958802217992e-01 | 9.877560285808963e-01 | gross before output head |

Retaining the routed experts at MXFP4 does not close either boundary under the
original bands. That observation is retained for audit, but the 0731 control
voids its causal reading. No mirror artifact or further family split is
admissible on this premise. Artifact-versus-runtime remains open.

### Type-107 CPU reader compatibility control

Evidence `/tmp/ember-type107-layout-776.CxQULx` on `otheru` applies the same
`gguf_tensor_error.py` sampling contract to both fixture pairs with two
independently built CPU readers. Each cell samples 64 rows from the named
routed-expert tensor with seed 107. The evidence binds all four model digests,
both reader-library digests, the script digest, each TSV digest, and the dirty
writer-source checkout whose exact library—not its checkout name—is the reader
identity. The earlier `comparison.json` in that directory used the wrong
writer provenance and is explicitly superseded by the four named TSVs.

| Model pair | Reader | Tensor | NRMSE | Cosine |
|---|---|---|---:|---:|
| Vision-Exp | artifact writer | `ffn_down_exps` | 3.84028948e-01 | 9.23884845e-01 |
| Vision-Exp | artifact writer | `ffn_gate_exps` | 3.67998019e-01 | 9.31135696e-01 |
| Vision-Exp | artifact writer | `ffn_up_exps` | 3.67611867e-01 | 9.31221298e-01 |
| Vision-Exp | Ember CPU | `ffn_down_exps` | 7.89703297e-01 | 7.55730054e-01 |
| Vision-Exp | Ember CPU | `ffn_gate_exps` | 7.34838341e-01 | 7.84502896e-01 |
| Vision-Exp | Ember CPU | `ffn_up_exps` | 7.45734482e-01 | 7.81108858e-01 |
| one-layer 0731, same two-scale writer | artifact writer | `ffn_down_exps` | 3.84355531e-01 | 9.23672681e-01 |
| one-layer 0731, same two-scale writer | artifact writer | `ffn_gate_exps` | 3.66127981e-01 | 9.31825664e-01 |
| one-layer 0731, same two-scale writer | artifact writer | `ffn_up_exps` | 3.66237865e-01 | 9.31680575e-01 |
| one-layer 0731, same two-scale writer | Ember CPU | `ffn_down_exps` | 8.03484554e-01 | 7.47012596e-01 |
| one-layer 0731, same two-scale writer | Ember CPU | `ffn_gate_exps` | 7.45473016e-01 | 7.81244991e-01 |
| one-layer 0731, same two-scale writer | Ember CPU | `ffn_up_exps` | 7.64949239e-01 | 7.72443083e-01 |

The writer and Ember CPU readers do not give equivalent reconstructions for
numeric type 107. A source audit extends the incompatibility to every selected
Ember reader: MMQ uses `load_tiles_rocmfp2_affine`, MMVQ uses
`vec_dot_rocmfpx_fp2_q8_1`, and CUDA conversion/get-rows use
`dequantize_rocmfpx_fp2`; all interpret the two trailing bytes as a whole-block
scale and offset. The fixture writer recorded by its quantization log emits one
scale per half-block with codebook `{-1,0,1,2}`. The unused type-107
`rocmfpx_dual_mmq_traits` specialization does not alter that result because
only type 104 instantiates the dual loader. Equal numeric type and block size
therefore mask a real writer/runtime ABI mismatch.

The apparent non-differential in the first table was an artifact-lineage error:
the one-layer 0731 candidate was emitted by the same two-scale writer as the
Vision fixture. It is a control for the bad tool, not a sample of the coherent
published model.

The actual serving 0731 artifact
`/srv/models/DeepSeek-V4-Flash-0731-ablit1042-v2.gguf` was written by the older
affine quantizer tree under `/srv/lucebox/rocmfpx`. Against its matching BF16
source, that writer's reader and current Ember's reader produce identical
metrics for all 129 routed-expert tensors sampled at 64 rows each. Their NRMSE
distribution is p50 `3.09347596e-01`, p90 `3.18139365e-01`, maximum
`3.20699470e-01`; cosine has minimum `9.48057067e-01` and median
`9.51756836e-01`. The exact TSVs are `writer-production-0731.tsv` and
`ember-production-0731.tsv` in the retained evidence directory.

The ABI mismatch is therefore a real artifact-lineage differential: production
0731 is affine-writer to affine-runtime and coherent; the current Vision
artifact is two-scale-writer to affine-runtime and incoherent. This establishes
that the current Vision artifact's routed-expert payload is misinterpreted; it
does not assert that no other quality issue exists. Ember's existing numeric
type must remain affine for published 0731 compatibility. Any retained
two-scale format requires a distinct tensor type rather than another silent
meaning for type 107.

The stock-types full artifact now in progress remains useful: it holds the
cross-checkpoint importance matrix fixed while removing every ROCmFPX type and,
if coherent, can bootstrap the authorized Vision-owned matrix. The next
ROCmFPX artifact must use the affine writer matching Ember regardless of which
matrix it consumes. No hardware run follows before the stock-types artifact is
complete.

## Pre-measurement runtime facts

- The native tower is lazy: text-only startup does not open or allocate the
  mmproj.
- Malformed PNG input is decoded and rejected before the tower mutex or lazy
  load, so malformed requests cannot force tower residency.
- The tower mutex covers execution as well as first load. Image encodes
  therefore serialize across the server. This is intentional until the tower's
  buffers are proven re-entrant. Any later concurrent-throughput report must
  state that its ceiling is one completed tower encode per encode-latency
  interval; it must not present serialized image throughput as a scheduler
  scaling result.
- Resident batching, exact-q1 prefill, speculation, compaction, and
  image-spanning snapshot reuse remain unavailable for image requests.

## Next admissible result

The block-0 and full-depth execution prerequisites are satisfied. Behavioural
policy v1 is versioned and digest-bound, but it must not be spent on the current
artifact: its known text incoherence predetermines an uninformative red. The
next C2 correctness result is the two-arm behavioural differential on a model
that first passes the text prerequisite. No image-path throughput number may be
published before that gate passes.
