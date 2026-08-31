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

### AMENDED 2026-08-31 (user decision): the numeric differential holds only through block 0

The differential above was written before we could measure how far it carries.
It carries through **block 0 and no further**, and this is the one revision this
document permits: not a loosened threshold after an inconvenient result, but the
replacement of a criterion that **measurement showed cannot discriminate at
all**.

**The measurement.** The reference's own BF16 and F32 lanes diverge as the tower
deepens:

    post_patch_projection   0.0016
    post_block_0            0.0039
    post_vit                0.1017      <- 10.2%
    post_aligner            0.0717      <-  7.2%

Two independent producers agree to four or five significant figures
(`72-emit-vision-oracle.py` and `77-emit-tower-budget-curve.py`). This is the
reference disagreeing with **itself** across dtypes, not an engine error.

So at tower depth the BF16 lane is **not a numerical authority**. A gate there
would require the engine to reproduce rounding noise the reference does not
reproduce itself, and any implementation difference below ~10% at the output is
indistinguishable from that noise. Adding checkpoints does not rescue it: an
anchor-sized residual at `post_patch_projection` is already 140x larger by
`post_block_0`, and by block 12 the perturbation response saturates and the
anchor stops mattering.

**What gates what, now:**

| depth | gate |
|---|---|
| block 0 | the 13-record numeric oracle, compounding budget, anchor 3e-5, first-red. Unchanged and discriminating. |
| blocks 1-31 | **no numeric gate.** Not "a looser one" — none is possible. |
| tower output | behavioural, below |

### The behavioural gate, and the control that makes it a gate

Accuracy on an image set is **not** sufficient on its own, because the failure
mode we most need to catch is a model that silently ignores the image and
answers from text priors. That failure has already occurred here once, and it
passed every gate it was shown.

So the gate is **two-armed**, and the control is not optional:

- **Arm A** — image + question, over a held-out set.
- **Arm B** — the identical questions with the image withheld.

Every item must be chosen so that **Arm B cannot be answered above chance from
the question text alone**. An item a language model can guess is not evidence
about the vision path and must be cut from the set, not scored.

**Pass requires all three:**

1. Arm B accuracy is at chance, confirming the set actually requires the image.
2. Arm A accuracy is far enough above Arm B that the gap cannot be sampling
   noise, at a threshold and set size **pre-registered before the first run**.
3. Failures are inspected, not just counted: a set that passes on aggregate
   while failing every OCR item is reported as such.

Item classes should include counting, colour or attribute binding, spatial
relation, and text-in-image, so that a single systematic defect cannot hide
inside an aggregate score.

**Known dependency, stated rather than discovered later:** no such set exists on
disk today. `/srv/models/vision-gate/` holds two images, and their source JPEGs
are gone — only the bound patch payloads survive. Assembling and ground-truthing
the set is real work and it is on the critical path for C2; it is not a detail
to be improvised at benchmark time.

**This gate replaces the numeric differential at tower depth only.** It does not
relax C1, and it does not relax block 0.

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
