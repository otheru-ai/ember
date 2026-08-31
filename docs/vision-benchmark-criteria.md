# Part C — what "meets or exceeds our existing quant" has to mean

Written **before** the first measurement, deliberately. The Qwen work is the
standing lesson: a criterion chosen after seeing the numbers is not a release
gate, it is a rubber stamp. Nothing in this file may be revised once a number
exists for the case it governs; revise it now if it is wrong.

No measurements live here. They live in the performance ledger.

## The comparison is narrower than the goal statement

The goal is "meets or exceeds our existing quant" — production
`DeepSeek-V4-Flash-0731` affine, 85.6 GiB at 2.59 BPW.

**The 0731 baseline has no vision.** So there is no image-path number to meet or
exceed, and any claim that we beat the baseline on images would be comparing
against nothing. Only the **text path** is comparable at all.

That splits part C into two questions that must be reported separately and never
averaged into one verdict:

| | question | comparable to 0731? |
|---|---|---|
| **C1** | Does adding vision cost us anything on text? | **yes** — this is the real regression gate |
| **C2** | Is the image path correct and fast enough to ship? | **no** — standalone, against the reference implementation |

Reporting a single "meets or exceeds" verdict across both is the specific
failure this document exists to prevent.

## C1 — text-path parity against 0731

The bar is **parity, not improvement**. The vision checkpoint is a different
model; we are not claiming a text win, we are proving vision costs nothing.

**Quality.** The total-variation criterion in
`engine/dflash/common/prefill_validation.h` applies unchanged: TV between the
reference and production logit distributions, evaluated at both T=1.0 and the
serving temperature, worst case taken, threshold 0.01. It is already the
user-decided release criterion and it is model-agnostic — it bounds how
differently the two paths could sample, which is what matters for a server that
samples rather than taking argmax.

**Throughput.** Prefill and decode tokens/s on the existing text workloads, same
shapes as the 0731 ledger entries. A regression outside measurement noise is a
release blocker; noise must be established from repeated runs before any
comparison, not asserted.

**The inertness gate.** The new `exp_probs_b_vl` weights, the routing branch and
the new request fields must be provably inert on a text-only request. If a
text-only run touches the vision routing branch even once, C1 is void regardless
of the numbers — the paths are not separated and the parity result means nothing.

## C2 — image path, standalone

### Correctness comes first and gates everything

No performance number for the image path is meaningful until the path is proven
correct, because every one of the known failure modes produces *fluent, confident,
wrong* output that a throughput harness cannot see:

- raster order instead of the two-row interleave (`build_image_block`);
- text routing bias applied to image tokens (`bias_vl` ignored);
- causal-only attention inside an `IMAGE_START..IMAGE_END` span;
- an image block split across a prefill chunk;
- prefix-cache collision returning image A's KV for image B.

The gate is a differential against the official Python reference
(`inference/vision.py`, `inference/image_processor.py`, `inference/model.py`),
not against our own intuition about what looks right. Codex owns this; the
mutation tests in its part-B plan cover exactly the five failures above.

### Quantization quality, once correct

Same instrument as C1 — TV distance — but the reference is our own model at
higher precision on the same image and prompt, not 0731.

**The obstacle, stated now rather than discovered later.** The natural reference
is the BF16 model, and it does not fit: BF16 is ~336 GB on a 128 GiB box. So the
reference must itself be a quant, and a quant reference is only admissible under
the rule already established for the Qwen F32 work: **the reference must be at
least an order of magnitude more accurate than the effect under test**, and that
ratio must be demonstrated, not assumed. If it cannot be demonstrated, the
comparison is not reportable and we say so.

### Throughput, reported standalone

Three numbers, never merged, because they have different denominators:

1. **Tower encode latency** per image, by resolution — a fixed per-request cost
   that decode-rate figures hide entirely. ~466 M params runs once per image.
2. **TTFT with one image**, end to end.
3. **Decode tokens/s after an image**, which should match C1's text decode; if
   it does not, the image span is affecting the decode path and that is a bug,
   not a benchmark result.

Resolution must be reported with every image number. This model is
dynamic-resolution under a 384-token budget, so an unqualified image throughput
figure is meaningless.

## Rules carried over

- A run missing any required attestation is **void**, not "indicative".
- No interpolated comparison values.
- Measurements go to the ledger; coordination messages and this file carry
  method and status only.
- GPU claims and production quiesce are announced before they are taken.
