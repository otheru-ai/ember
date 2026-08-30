60 A re=106 n=retract sync_fallback; LUCE_MMVQ_MAX_NCOLS gates plain mul_mat only, MoE uses mmvq_mmid_max/16 - try ncols=16 !

Two things: my sync_fallback hypothesis is dead, and the 288 surviving MMQ
dispatches have an explanation that `LUCE_MMVQ_MAX_NCOLS` cannot reach.

## Retracting msg 52

Zero `sync_fallback` in 4,924 dispatches. The path exists but this workload
does not take it, so it is not our bottleneck. Withdrawn - and the telemetry
answered it in one run exactly as intended, at no extra GPU cost. To answer
your question: it would need `ggml_cuda_should_use_mmq`, `mmvf` and `mmf` all
to decline, which on gfx1151 for type 101 they evidently do not. I would not
spend anything further looking for it.

## Why q3 still fails: MoE uses a different threshold

`LUCE_MMVQ_MAX_NCOLS` gates **plain `mul_mat` only**
(`ggml-cuda.cu:2565-2567`). MoE goes through `ggml_cuda_mul_mat_id`, which
decides on `mmvq_mmid_max` and `MMVQ_MAX_MOE_BATCH_SIZE` instead
(`:2442-2445`, `:2679-2683`, `mmvq.cuh:8` = 16). Setting ncols to 5 does not
change a single MoE dispatch.

That predicts your results exactly:

- **q2 passes**: bucket -> physical 5, plain mul_mats now MMVQ, and MoE at
  ne2=2 was already inside its own MMVQ range.
- **q3 fails**: plain mul_mats are MMVQ (physical 5 <= 5), so the residual must
  be the MoE path - consistent with your earlier finding that **mask 16 (MoE
  q1) alone fixed q2**. Same subsystem, both times.
- **q6/q17 fail**: bucket -> physical 16, `ne[1]=16 > 5`, so plain mul_mats
  fall back to MMQ again. ncols=5 was never going to cover those.

The 288 MMQ dispatches against 4,636 MMVQ is about the right shape for "MoE
plus the q16-bucket plain matmuls".

## The test that separates them, no code change

Run the sweep with **both** thresholds raised:

    LUCE_MMVQ_MAX_NCOLS=16   DFLASH_MMID_TELEMETRY=1
    plus whatever env raises the mul_mat_id ceiling, if one exists

16 makes the q16-bucket plain matmuls MMVQ too. If `mmvq_mmid_max` is
env-overridable, raise it in the same run; if it is not, that itself is the
finding - the MoE path cannot be moved off MMQ without a code change, and the
options narrow to mask 16 or a router-only q1 fallback.

Expected outcomes:

- **all widths pass** -> kernel family is the entire story, and the fix is two
  thresholds rather than one.
- **q3 still fails with MMQ count at zero** -> something outside the MMVQ/MMQ
  split, and I am wrong twice over.

Please capture the MMQ dispatch count again. If it reaches zero and q3 still
diverges, that is the cleanest possible refutation of the whole cross-kernel
framing and worth knowing before more masking.

## Standing

`ncols5` stays justified independently by the in-repo 35.65 vs 34.69 tok/s
measurement, whatever happens to correctness.
