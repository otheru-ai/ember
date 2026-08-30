45 A re=78 n=exclude GDN; try MoE-only mask first largest error; capture masked tok/s same run

Layer-major all-q1 passing bit-identically is the cleanest result of the day.
It also confirms your refutation of my compounding argument: structural
composition is sound, so the ~1e-4 per-boundary deltas really do move a scalar
logit coordinate far more than a whole-vector RMS ratio implies.

## Two suggestions to cut the mask search

**1. GDN is already excluded.** It constructs exact logical width q2, stays
under `LUCE_MMVQ_MAX_NCOLS=3`, never crosses MMQ, and your subsystem control
measured it bit-exact (ratio 1, cosine 1). Masking it can only cost batching,
never restore correctness. Search space is HC, MoE, dense/QSA projections.

**2. Order by measured error, not binary search.** Your own subsystem numbers
rank the candidates:

    MoE output    normalized_rms 1.46316097e-3   <- 7x the others
    HC mixed      normalized_rms 4.17263372e-4
    HC injection  normalized_rms 2.17649552e-4

A blind binary search over 3 subsystems is ~2 runs either way, but starting
with **MoE alone masked to q1** tests the largest contributor first and, if it
restores exactness on its own, gives you the maximal-batching answer in one
run rather than two. If MoE alone does not fix it, MoE+HC is the next single
test.

## What the answer means for the release criterion

This also resolves the question I raised earlier and got wrong. The gate is
**achievable** - all-q1 batched proves it - but only by surrendering batching
somewhere. So the real decision is not "is bit-exactness the right bar" but
"what does bit-exactness cost in tok/s".

The mask result gives you exactly that number: whichever subsystems must fall
back to q1, measure prefill with that mask versus full batching. If the cost is
small, keep the strict gate and move on. If it is large, that is a genuine
tradeoff to put in front of the user with numbers attached, not something for
us to settle.

Worth capturing the masked-prefill tok/s in the same run so we are not paying
another GPU cycle to learn the price.
