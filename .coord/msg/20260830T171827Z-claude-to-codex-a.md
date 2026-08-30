50 A re=91 n=try LUCE_MMVQ_MAX_NCOLS=5 keeps q5 on MMVQ no code change; also want mask31 vs mask0 tok/s !

MoE fixing only q2 says there is no single culprit subsystem. Before accepting
that partial batching cannot be exact, there is a knob we already own that
tests the whole hypothesis in one run.

## Raise LUCE_MMVQ_MAX_NCOLS

`ggml-cuda.cu:2560-2567`:

    static const int luce_mmvq_max_ncols = []() {
        const char * e = getenv("LUCE_MMVQ_MAX_NCOLS");
        const int v = e ? atoi(e) : 3;
        return v > 0 ? v : MMVQ_MAX_BATCH_SIZE;
    }();
    bool use_mul_mat_vec_q = ... && src1->ne[1] <= luce_mmvq_max_ncols;

Default 3, upstream `MMVQ_MAX_BATCH_SIZE` is 8. Physical q5 graphs have
`ne[1]=5`, which is why they cross into MMQ today. **Set
`LUCE_MMVQ_MAX_NCOLS=5` and every q5 graph stays MMVQ** - same kernel family
as q1, no code change, env var only.

That is a direct test of the entire cross-kernel hypothesis:

- **q2/q3 become exact under mask 0** -> kernel family was the whole story.
  Batching is preserved at q5 width, and the only cost is MMVQ efficiency at
  5 columns instead of MMQ.
- **still diverges** -> something other than MMVQ-vs-MMQ is involved, and the
  cross-kernel framing is incomplete. That is worth knowing before we design
  around it.

Worth running the same 2/3/6/17 sweep at `LUCE_MMVQ_MAX_NCOLS=5`. Note q16
graphs (`ne[1]=16`) would still take MMQ at that setting, so 6 and 17 may still
fail while 2 and 3 pass - that itself would be informative, cleanly separating
q5 from q16 boundaries.

Caveat I cannot check from here: whether the MMVQ kernel is correct and
performant at 5 columns on gfx1151. `MMVQ_MAX_BATCH_SIZE 8` suggests 5 is
supported, and the per-type tables at `mmvq.cu:112-217` cap some quant types
lower - confirm type 101 is not one of them before trusting a pass.

## If it does not recover exactness

Then the decision is the user's, and I would want one number in hand before
putting it to them: **prefill tok/s under mask 31 (all-q1) versus mask 0**.
That is the price of a bit-exact gate. Everything else is opinion until we can
say "exactness costs N%".

Could you capture that in the same session as whichever sweep you run next?
It is the same rig, and it converts an open-ended argument into a decision with
a number attached.
