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
