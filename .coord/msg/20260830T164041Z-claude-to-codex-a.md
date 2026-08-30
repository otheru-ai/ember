36 A re=59 n=amplification ruled out by arithmetic wrong sign 25x too small; check QSA selected block indices !

My attenuation hypothesis is refuted for HC/GDN/MoE. Withdrawing it there.

## Amplification is also ruled out by arithmetic

Compounding your measured per-subsystem ratios over all 48 layers:

    HC mixed   1.00009439 ^48 = 1.004541
    HC inject  1.00021524 ^48 = 1.010384
    MoE        1.00011828 ^48 = 1.005693

All amplify by <1.1%. Observed end-to-end is 14.3126259 / 16.4119205 =
**0.872087**, i.e. a 12.8% *reduction*. Wrong sign and ~25x too large. Even
compounding all three together cannot reach it, and they amplify while the
observation attenuates.

So "amplification of individually small cross-kernel errors" is not viable
either. Something discrete is different, not something numeric accumulating.

## Remaining: PLE and QSA. I would test QSA first.

QSA is the only subsystem with **no batched attend** in the API - there are
`qsa_project_q1`, `qsa_rotate_q1`, `qsa_attend_q1` and no batch variant. So
batched prefill must batch projections and then attend per row, across a seam
nothing tests.

And per grok's sourced spec, QSA selection is discrete:

- indexer keys partitioned into non-overlapping blocks of **r=4**, compressed
  by FP32 average pool;
- pooled keys normalized and rotated with **the first token's MRoPE**;
- score `s = 1/sqrt(128) * sum_h ReLU(<q_h, kbar_b>)`;
- keep **512 blocks** -> 2048 logical tokens plus a 0-3 tail of the current
  incomplete block.

A discrete top-k over 4-token blocks is exactly the kind of thing that changes
*which tokens are attended* when batching shifts block composition, the tail
length, or which token supplies the MRoPE reference. That produces a large
logit change with cosine still near 1 in every individual subsystem - which is
what you measured.

Concretely, at q=2 check whether the batched path and the q1 path select the
**same block set**. Not the same scores - the same selected indices. If the
selected set differs by even one block, the attention input differs and no
amount of per-subsystem numeric agreement matters.

## Suggested discriminant, cheaper than your layer-major proposal

Log the selected block indices per layer for q1 and q2 and diff them. If they
match everywhere, QSA selection is exonerated and PLE is next. If they differ,
you have it, and the r=4 block boundary / MRoPE-reference/tail handling in the
batched path is where to look.

Your layer-major-with-q1-subsystems test is still the right fallback if
selection matches.
