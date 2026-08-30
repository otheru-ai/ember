# Review of c561212 numerics control

Reviewed before trusting its output. Design is sound; two caveats.

## Correct

- uses a real type-101 `attn_qkv` weight, not synthetic - so it measures the
  shipping kernel pair;
- same weight, same deterministic input, only row count differs, so MMVQ vs
  MMQ is the only free variable;
- `signed_error` alongside squared error is the right discriminator: unbiased
  noise gives signed ~0, a systematic fault does not. Please report it, not
  just compute it;
- top-2 scan in `log_prefill_top2` is correct, including the initial pair swap;
- gated on `DFLASH_QWEN_NUMERICS_EVIDENCE=1`, so normal runs are unaffected.

## Caveat 1: single projection is a lower bound

The control measures one projection in one layer. The end-to-end divergence
accumulates over 48 layers and several projections per layer. A small relative
error here does not by itself prove the end-to-end flip is benign - it could
compound. Read the control as a lower bound on drift, and let the end-to-end
top-2 margin carry the actual argument.

## Caveat 2: both paths must log

For the margin to be usable I need it from the same prompt under
`force_exact=true` and `force_exact=false`. If only one path logs, there is
nothing to compare. The diff shows one call site; confirm the validator drives
both and each emits a line.

## The number that decides it

q=2, index 1, margin under `force_exact=true`. Top-1 was still correct there,
so if that margin is within the projection error scale, the flip is noise
crossing a near-tie and there is no functional fault. If the margin is wide,
something real is wrong and the bisect resumes.
