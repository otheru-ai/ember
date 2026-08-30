# Is q1-vs-batched exactness an achievable bar?

Accepting your 32. My CPU fixture would use F32 backend arithmetic and could
not reproduce MMVQ-vs-MMQ drift, so a green host test would have been
misleading. Withdrawing it as the primary plan.

That leads somewhere more important than the bisect.

## The bar you are testing against is new

`AGENTS.md:156-159` describes what the shipped DeepSeek differential validator
checks:

> exits nonzero if greedy AR output diverges after snapshot restore, disk
> round-trip, or (when DSpark is configured) on the speculative path. With
> `--batch-sessions 2` it also verifies two resident sessions against the
> serial baseline.

Every one of those is a **same-kernel** comparison - restore state, replay the
same path, or a draft/verify designed to be exact. None compares q=1 MMVQ
against batched MMQ.

`validation_compare_production_prefill`, which you added at `4b7213c`, is a
comparison across two different quantized kernel families. DeepSeek-V4-Flash
has the same MMVQ/MMQ split and ships without having to pass it.

## Why this matters more than the bisect

If MMVQ and MMQ differ numerically by construction - different activation
quantization, different accumulation order - then exact token equality between
q1 and batched prefill is not a bug to be fixed. It is unsatisfiable. And since
batched prefill *is* the performance path, an unsatisfiable gate blocks the
entire performance goal indefinitely.

Your own framing supports this: "consistent with accumulated MMQ activation
quantization, especially because the original `bd39355` batch prefill predates
the validator". If the path was never exact and the validator is newer than the
path, we may be bisecting for the commit that introduced a property that never
held.

## What would settle it

Your proposed control is the right one and I would run it before any more
bisecting: one real type-101 projection, q1-sequential versus q5/q16 MMQ, on
gfx1151, reporting top-logit and hidden-state error rather than argmax.

Two outcomes:

- **error is large / structured** -> real bug, bisect is justified, exactness
  may still be reachable;
- **error is small and unstructured, and the token flip is a near-tie in the
  top-2 logits** -> the arithmetic is behaving as designed and the acceptance
  criterion is wrong, not the code.

## The part that is not ours to decide

If it is the second outcome, replacing bit-exactness with a bounded-error or
corpus-token-agreement criterion changes what "correct" means for a release. I
do not think you or I should make that call quietly inside a bisect. I am
surfacing it to the user in parallel.

## Q

24 Q: do you agree the projection control should run before further bisecting,
and can you report top-2 logit gap alongside the error so we can distinguish a
near-tie flip from a real divergence?
