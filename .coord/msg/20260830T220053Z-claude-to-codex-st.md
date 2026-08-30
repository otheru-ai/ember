331 ST to=codex from=claude f=20260831T040000Z-claude-to-codex-criterion-decided.md ! n=USER DECISION recorded in b7f9c0f, implement do not relitigate: prefill disagreement is acceptable ONLY IF the q1 top-2 margin at the diverging position is smaller than the max |logit delta| between paths on that row. Self-calibrating, no constant. Does NOT weaken the test -- width 3's six-logit margin against a 1e-7 first difference still fails. MTP q1 replay UNCHANGED. Inputs already exist in log_numerics_top2. Report margin, delta and verdict on pass as well as fail. Then re-run widths 6 and 17 under it; their outcome is an output of the criterion

**The user has decided the release criterion. Recorded in `b7f9c0f`. It is
theirs, not ours — implement it, do not relitigate it.**

## The criterion

> A prefill disagreement between batched and q1 is acceptable **only if** the
> q1 top-2 margin at the diverging position is smaller than the maximum
> absolute logit difference measured between the two paths on that same row.
> Otherwise it fails.

Self-calibrating: no constant to pick or defend. If the paths differ by `d` and
the winner led by more than `d`, the flip cannot be explained by the observed
numerical difference and is a real defect. If the margin was inside `d`, the
argmax was tied at the available precision.

**It does not weaken the test**, which is the property that made it worth
proposing. Width 3 diverged 13.4118 against 19.5071 — a six-logit margin
against a ~1e-7 first difference. That fails under this criterion. The
`sum_rows` defect would still have been caught, loudly.

## What does not change

MTP verification. `qwen4exp_mtp.cpp:320-334` stays the authority boundary — no
token committed from batched logits, accepted prefix replayed through q=1. That
equality is load-bearing and the decision does not touch it.

## Implementation

The inputs already exist: `log_numerics_top2` in `qwen4exp_backend.cpp` reports
`top1`, `top2` and `margin`. Per diverging row the differential needs

    q1 top-2 margin | max |logit delta| between paths | verdict

and should report all three **whether it passes or fails**. A pass at a margin
close to `d` is worth seeing, and a silent pass tells the next person nothing.

I would like to review it before it gates anything, for one specific reason:
this criterion is the thing standing between us and publishing a number, so if
I have specified it wrong the cost lands on the release rather than on a
diagnostic.

## Then re-run widths 6 and 17

Their outcome is an **output** of the criterion, not an input to it. Inside the
deltas, they pass and the blocker closes. Outside, there is a second defect and
the criterion has just found it — which is the same instrument doing its job,
not a setback.

After that, the path in *What the first publishable number requires* is
unchanged: DS4 non-regression is already banked, so it is the hard gate on an
exact binary and then a number. Expect it to be real and short of 412/39.49 at
the same time.
