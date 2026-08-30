51 A re=95 n=the 3 is an sm_86 RTX3090 number not gfx1151; deepseek already overrides to 4; measure A/B/C tok/s !

Excellent isolation. The cause is MMQ-at-physical-q5 and nothing else.

Before treating `LUCE_MMVQ_MAX_NCOLS=5` as the fix, read its own comment -
the default of 3 is **not** a gfx1151 measurement.

`ggml-cuda.cu:2545-2559`:

    // Default 3: measured crossover on sm_86 (RTX 3090, Q4_K_M/Q6_K dense
    // GEMVs) - MMVQ wins at ncols<=3, MMQ wins at 4-8 ... Override via env
    // for other hardware.
    // SECOND DECLARATION WARNING: this fallback of 3 is the generic one ...
    // It is NOT what a gfx1151 DSpark run sees -- deepseek4_backend.cpp's
    // configure_gfx1151_dspark_mmvq_default() setenv()s this variable
    // (overwrite=0) to 4, or to the speculative verify width when that is
    // larger ... Keep the two in step, and do not "simplify" either away
    // without re-measuring both paths.

Three consequences.

**1. The 3 is an RTX 3090 number.** MMVQ-vs-MMQ crossover on sm_86 for
Q4_K_M/Q6_K. It was never measured on gfx1151, and never for type 101. The
comment explicitly says to override for other hardware. So raising it for Qwen
is not fighting a tuned value; it is replacing an inherited one.

**2. DeepSeek already overrides it to 4** via
`configure_gfx1151_dspark_mmvq_default()`, and only when DSpark is on and the
device is gfx1151. Qwen presumably never calls that, so Qwen runs on the
sm_86-derived 3. Worth confirming - if Qwen does inherit 4, then q5 is only one
step above an already-overridden value.

**3. The comment demands both paths be re-measured together.** Whatever we set
for Qwen must not silently change DeepSeek's DSpark verify width.

## What I would measure, in one session

    A. mask 0, LUCE_MMVQ_MAX_NCOLS=3   (today)      -> prefill tok/s, baseline
    B. mask 0, LUCE_MMVQ_MAX_NCOLS=5   (exact)      -> prefill tok/s
    C. mask 31 (all-q1)                             -> prefill tok/s

A vs B is the price of exactness via the threshold. C is the price of
exactness via full q1 fallback. If B is close to A, we have a fix that costs
almost nothing and needs no code change - just a Qwen-scoped default, set the
way DeepSeek already sets its own.

Also worth one datapoint at `LUCE_MMVQ_MAX_NCOLS=16`, which would keep the q16
graphs on MMVQ too and should make the 6/17 widths exact as well. That tells us
whether the whole batched path can stay one kernel family, and at what cost.

## Naming

If this lands, it should be a Qwen-scoped default set in the Qwen backend the
same way `configure_gfx1151_dspark_mmvq_default()` does it for DeepSeek - not a
global change to the fallback constant. The comment is explicit that the two
paths must stay in step, and a global edit would silently retune DeepSeek.
