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
| C1 text throughput parity | **measured, red, confounded** | Rows A and B (2026-09-01/02 section) ran the affine Vision artifact at `--ds4-expert-top-k 0` (model default, 6 routed experts) while the 2026.8.24 reference and production run top-k 4. The text-path deficit is the size the extra experts predict; a like-for-like row at top-k 4 does not exist yet. |
| C1 text/vision routing inertness | not measured | Requires the full text-path gate on the release artifact. |
| C1 one-layer BF16/quant discriminator | void | The identically quantized known-good 0731 control is similarly gross, so the one-layer method cannot localize a Vision-specific defect. |
| C1 type-107 reader/writer compatibility | **resolved on the current artifact** | The 2026-08-31 artifact was written two-scale under the affine type id and was incoherent. The current artifact `a2afd9f4…` (recipe `8765ebe`) is affine, passes gates 1–2, and answers image and text questions coherently (behavioural rows below). The format still has no discriminator; writer lineage is recorded by the pipeline, not by the file. |
| C2 block-0 numeric differential | **green** | All 13 records pass the corrected compounding budgets at `770d2b2`. |
| C2 tower-depth numeric differential | diagnostic only | The reference is not a numerical authority after block 0. |
| C2 full-depth native execution | **green** | Four image requests and one text-only control completed on the 43-layer artifact at `5df70f6`. |
| C2 synthetic behavioural differential | **green, twice** | Policy v2 on the current artifact: PASS at ember `bbdaee1` (released-image build) and again at `f8efd7a` on the ROCm 10 image, identical per-class counts. Section 2026-09-01/02 below. |
| C2 natural-image behavioural differential | **green at policy v4; v3 red retained** | 100 TextVQA items, arm A significant, arm B at chance, 0 cuts, 0 errors at v4 (cap 2048). The v3 run (cap 512) stays red on the record: one image-arm 422 and six `length` finishes. |
| C2 image throughput | **measured** (rows A, B) | Image requests decode autoregressively by design (speculation is inert on image requests, speedup ≈ 1.0 in both rows). Numbers in the 2026-09-01/02 section. Not comparable to any text figure. |

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

## 2026-09-01/02 — behavioural gates on the affine artifact

All numbers here are read from the retained results files, not from console
summaries or coordination messages. The pipeline's own acceptance record is
`otheru-quant-pipeline/docs/ACCEPTANCE.md`; where the two disagree, the results
files named below win.

### Identity

- Artifact: `/srv/models/DeepSeek-V4-Flash-Vision-Exp-affine-strix-lean-8765ebe.gguf`,
  SHA-256 `a2afd9f42881e57f95a13054900584c69cfb2a94aeba581924ae5dbd702dd55a`;
  mmproj `9225c5562c05bd910245ab24c9274ca777eba2a801990f47ebe0c6344f144002`.
- Gate runner `scripts/81-run-behavioural-gate.py`
  (`e2e78588…` for the v2/v4 runs, `ff222128…` for v3).
- Server: `--ds4-expert-top-k` **model default** (`expert_top_k=model_default`
  in the 0d attestation; no run passed 4). See the throughput section for why
  this matters.

### Results

| run | ember | image | policy | items | retained | arm A | arm B | pass |
|---|---|---|---|---|---|---|---|---|
| `run-0g-synthetic-bbdaee1-3eea1db` | `bbdaee1` | `ghcr.io/otheru-ai/ember:dev-sha-63435cf365d9` | v2 (`46756403…`) | 100 | 97 | colour 25/25, count 24/25, ocr 25/25, spatial 22/22 | 0, 0, 0, 3/25 | **true** |
| `run-rocm10-synthetic-f8efd7a-20260901` | `f8efd7a` (ROCm 10) | `sha256:88f78d3f…` | v2 | 100 | 97 | identical to the row above, per class | identical | **true** |
| `run-0g-natural-scored-bbdaee1-9d6ea49-policyv3` | `bbdaee1` | as 0g | v3 (`0ff8c4fa…`, cap 512) | 99 (+1 error) | 99 | 87/99 = 0.879 | 0/99 | **false** |
| `run-0l-natural-policyv4-bbdaee1-ae78f39` | `bbdaee1` | as 0g | v4 (`b82d86ff…`, cap 2048) | 100 | 100 | 90/100 = 0.900 | 0/100 | **true** |

Per class, every arm A p-value against chance is below 1e-6 and every arm B
`not_above_chance` is true; the spatial class cut 3 items on arm B (chance
answers) in both synthetic runs. The synthetic result is byte-identical in its
per-class counts across the two images, which is what a compiler bump that
does not move numerics should look like; it is not evidence that it moved
nothing (exact digests of the responses were not compared).

The v3 → v4 flip is three items and the disappearance of one image-arm 422
(`tvqa_34733`, the detected degeneration tracked as backlog 0j). The cap was
raised by user decision with the flip stated in advance.

## 2026-09-01/02 — throughput rows A and B, and what they do not compare

Bundles, both `scripts/benchmark_bundle.sh` in Vision mode
(`EMBER_BENCH_VISION=1`, 12 workloads, 7 depths, spec on/off arms):

| row | bundle | ember | image | drafter |
|---|---|---|---|---|
| A | `/srv/models/perf/vision-baseline-current-ed6cf9f-20260901/ember-2026-09-01` | `bbdaee1` binary from the 0g run, harness `ed6cf9f` | `ghcr.io/otheru-ai/ember@sha256:14b277f8…` (2026.8.24-era) | `DeepSeek-V4-Flash-0731-ablit1042-DSpark-draft.gguf` (`1a01c80e…`) |
| B | `/srv/models/perf/vision-rocm10-matched-draft-6575096-20260902/ember-2026-09-02` | `f8efd7a`, harness `6575096` | `ember-rocm:10.0` (`88f78d3f…`) | `DeepSeek-V4-Flash-Vision-Exp-DSpark-draft-ember.gguf` (`97ba6019…`, built at pipeline `b8c2abc`) |

Server args in both: `--vision-mmproj … --max-ctx 65536 --ds4-prefill sparse
--ds4-expert-top-k 0 --default-temperature 0.6`. The text reference rows
(2026.8.24 and both ROCm 10 text rows in `docs/perf/data.json`) and production
(`docker inspect ember-server`, 2026-09-02) run `--ds4-expert-top-k 4`. The
model's `expert_used_count` is 6, so `0` means six routed experts per token
against production's four.

### Text workloads, three-way (tok/s; spec-on decode / autoregressive)

| workload | 2026.8.24 | row A | row B |
|---|---:|---:|---:|
| alphabet | 40.61 / 23.63 | 35.50 / 22.08 | 34.00 / 22.12 |
| code | 29.86 / 23.79 | 33.51 / 22.08 | 21.46 / 22.11 |
| count | 40.44 / 23.33 | 36.24 / 21.83 | 23.33 / 21.87 |
| creative | 23.19 / 23.65 | 21.70 / 22.05 | 21.40 / 22.14 |
| essay | 23.18 / 23.64 | 21.70 / 22.09 | 21.41 / 22.14 |
| factual | 30.81 / 23.64 | 20.98 / 22.26 | 21.22 / 22.28 |
| json | 39.60 / 23.60 | 35.30 / 22.07 | 34.69 / 22.09 |
| multiples | 40.40 / 23.62 | 35.37 / 22.09 | 20.53 / 22.12 |
| prose | 23.14 / 23.65 | 21.68 / 22.08 | 21.38 / 22.15 |
| repeat | 39.81 / 23.63 | 34.93 / 22.09 | 23.07 / 22.13 |
| throughput median (256 tok) | 39.49 | 35.21 | 35.14 |

Depth series (prefill tok/s / spec-on decode / autoregressive; acceptance):

| prompt tokens | 2026.8.24 | row A | row B |
|---:|---|---|---|
| 43 | 71.0 / 39.19 / 23.36; 0.981 | 64.4 / 35.06 / 21.81; 0.981 | 65.1 / 22.62 / 21.83; 0.833 |
| 862 | 281.0 / 39.43 / 22.66; 0.981 | 250.4 / 33.25 / 21.22; 0.980 | 249.5 / 28.74 / 21.21; 0.895 |
| 3,925 | 342.0 / 38.00 / 22.74; 0.981 | 308.1 / 34.13 / 21.27; 0.981 | 297.2 / 20.49 / 21.30; 0.775 |
| 18,553 | 299.2 / 32.62 / 20.91; 0.936 | 273.5 / 29.28 / 19.66; 0.981 | 273.7 / 19.69 / 19.67; 0.875 |
| 38,059 | 271.0 / 24.79 / 19.01; 0.974 | 250.4 / 22.93 / 18.02; 0.978 | 249.6 / 19.55 / 18.05; 0.926 |
| 77,068 | 223.5 / 16.58 / 16.61; — | 211.3 / 15.79 / 15.79; — | 209.8 / 15.81 / 15.80; — |
| 116,077 | 184.1 / 14.86 / 14.88; — | 176.3 / 14.21 / 14.21; — | 175.6 / 14.23 / 14.21; — |

Fixed-prompt prefill groups, median tok/s (128 / 512 / 2048 / 8192 / 16384 /
32768 tokens): 2026.8.24 216.3 / 336.7 / 412.0 / 351.0 / 322.5 / 291.4;
row A 188.6 / 306.3 / 381.2 / 324.7 / 303.4 / 278.3;
row B 188.4 / 306.3 / 364.8 / 325.1 / 303.0 / 277.5.

### Image workloads (rows A, B only; no text row has them)

| workload | row A tok/s / AR / speedup | row B tok/s / AR / speedup |
|---|---|---|
| image_short | 25.76 / 25.79 / 0.999 | 25.37 / 25.85 / 0.981 |
| image_long | 21.76 / 21.73 / 1.001 | 21.48 / 21.75 / 0.988 |

Image prefill in row A: 195.6 (short) and 194.3 (long) tok/s. Speculation is
inert on image requests in both rows, as pre-registered from the upstream
zero MTP VL biases; the image decode figure is the autoregressive rate.

### Readings

1. **2026.8.24 → row A is not an artifact-versus-artifact comparison.** Every
   autoregressive text workload is −5.8…−7.2 % and every prefill point is
   −6…−13 %, uniformly, with acceptance unchanged (0.98 with the 0731 drafter).
   Row A ran six routed experts per token; the reference ran four. Routed
   expert matmuls (type 107) are roughly 18 % of decode in the release trace,
   so 4 → 6 predicts about −8 % autoregressive, which is the size observed.
   The prediction is stated before the measurement: **a row A rerun at
   `--ds4-expert-top-k 4` should land within ±1 % of 2026.8.24 on every
   autoregressive workload.** If it does not, the Vision artifact is slower
   than 0731 on the same engine and that becomes a finding about the recipe.
   Until that row exists, no text-throughput parity verdict for the Vision
   artifact can be read from these bundles.

2. **The behavioural gates were run at the same top-k 0.** A Vision release
   served at top-k 4 (production's setting) has not been gated. Routing
   truncation changes outputs; gates 3–4 would have to be re-run at the
   serving top-k before any such release.

3. **The matched Vision drafter (row B) is worse than the 0731 drafter (row A)
   on this target, and it is the weights, not the wiring.** Acceptance on the
   structured workloads falls from ≈0.98 to 0.70–0.89 (code 0.705, multiples
   0.707, repeat 0.850, count 0.886) and on prose to 0.18–0.29; spec-on tok/s
   on code/count/multiples/repeat drops 34–42 % while the autoregressive
   controls move +0.1…+0.4 %. The two drafter files have byte-identical
   metadata (37 keys: `capture_layer_ids [40,41,42]`, `block_size 5`,
   `markov_rank 256`, `mask_token_id 128799`, …) and identical tensor
   inventories except the three `exp_probs_b_vl.bias` tensors row B adds; all
   81 shared tensors differ in content, including
   `dflash.dspark.markov.w1/w2` and `dflash.dspark.confidence.weight`
   (per-tensor SHA-256, 2026-09-02). So the drafter is a genuinely different
   set of weights built from Vision-Exp's own `mtp.*` tensors, and those
   weights predict this target's text worse than 0731's abliterated drafter
   does. What is not yet separated: whether that is the Vision-Exp MTP head
   itself, or something in the `b8c2abc` build (the DSpark Markov/confidence
   tensors have no independent reference on this box). The decision this
   forces — ship Vision with the cross-model 0731 drafter, or investigate the
   build — is the user's.

   Host-side audit of the matched drafter, 2026-09-02 (pure-python GGUF
   reader on otheru, both files): type inventory identical (25 Q8_0, 9
   type-101 expert tensors, 2 F16, F32 for the rest, +3 F32 `bias_vl`);
   Vision-Exp `config.json` and `inference/config.json` both carry
   `dspark_target_layer_ids [40,41,42]`, `dspark_block_size 5`,
   `dspark_markov_rank 256`, `dspark_noise_token_id 128799`, matching the
   metadata the build copied from the 0731 template; no NaN and no zero
   tensor in the 22 small tensors sampled (`dflash.*`, `output_*`, block-0
   norms, router, `exp_probs_b`); the drafter's three stages carry
   `exp_probs_b_vl` all-zero, as upstream ships them — the 43 LM layers of
   the affine artifact do NOT (measured 2026-09-02: mean norm 27.1, none
   zero), so do not read this line as a statement about the main model;
   Q8_0 block scales on `dflash.fc`, `attn_output_a/b`, `ffn_down_shexp`
   have no zero blocks and RMS within 0.65–0.8× of 0731's. Norm-weight
   RMS differs 2–4× between the two (`attn_norm` 0.048 vs 0.196,
   `hidden_norm` 0.073 vs 0.211, `markov.w1` 0.50 vs 0.22), which is what
   different trained weights look like, not a scale error. Nothing at the
   artifact level says "build defect"; what remains is GPU-bound: the
   Vision drafter against the 0731 target (low acceptance there too → the
   head is weak in general; high → a pairing effect on this target).

4. **ROCm 10 alone (row A → row B, autoregressive) is +0.1…+0.4 %** on every
   text workload and within ±1 % on every prefill point except the 2048 group
   (−4.3 %), consistent with the text rows in `docs/perf/data.json`.

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

Two rows are missing before any Vision text-throughput verdict, both one
bundle each on the box, and both were pre-registered above:

1. Row A rerun with `--ds4-expert-top-k 4` and the 0731 drafter, everything
   else identical. Reading 1 says where it must land.
2. Gates 3–4 at top-k 4 if that is the serving configuration.

Row B's drafter question is a build-versus-model discriminator, not a rerun:
the matched drafter needs an independent reference for its DSpark tensors or a
second build from a clean tree before another benchmark is spent on it.

The 0j degeneration (`tvqa_34733`, a detected 422, not silent corruption) is
still open and is the only correctness item standing between the current
artifact and the C1 text quality gate.
