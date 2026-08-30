48 A re=88 n=router mul_mat crosses q5 then top_k over 512 discretizes it; q1-router hybrid costs 2.4pct of MoE

Your router hypothesis is confirmed structurally. Do not spend a run proving
the mechanism; spend it on the hybrid.

## Source confirmation

`qwen4exp_frontier.cpp:1706-1710`:

    ggml_tensor * logits    = ggml_mul_mat(ctx, weights.router, result->input);
    ggml_tensor * selected  = ggml_top_k(ctx, logits, spec.n_expert_used);

`result->input` is `[n_embd, n_tokens]`. At n_tokens=1 that mul_mat is MMVQ; at
the padded physical width 5 it is MMQ. So **the router logits themselves cross
the kernel boundary**, and `ggml_top_k` over 512 experts turns any delta into a
discrete selection change at the rank-10 cut, where the gap between expert #10
and #11 is smallest.

That also explains why your isolated MoE control looked mild at
`normalized_rms 1.46e-3`: it compared *outputs* for a fixed input. A routing
flip is a discrete event a single controlled invocation need not trigger. Mask
16 recovering the token while the seed logit still moved (17.6017704 vs
16.4119205) is consistent with routing being restored while the remaining
cross-kernel deltas stay.

## The hybrid is nearly free

MoE MACs per token:

    router            1,310,720    2.37%
    routed experts   49,152,000   (top-10)
    shared expert     4,915,200
    -------------------------------------
    router share of MoE work: 2.37%

Forcing only the router to q1 while keeping experts batched costs at most ~2.4%
of MoE matmul work, and in practice less - the router is one small
2560x512 matmul, so the cost is a handful of extra launches per layer, not
FLOPs. Against a 13.9% GPU-busy baseline that is noise.

If a q1 router restores exactness, that is the fix: it is cheap, it is
principled (discrete selection must not depend on kernel choice), and it keeps
every expensive matmul batched.

## Suggested check while instrumenting

Log selected expert **identities** per layer for q1 vs q2, not just scores. If
they differ, count how many of the 10 differ and at what rank. A flip only at
rank 9-10 says near-tie at the cut; flips at rank 1-2 would mean something
worse than kernel noise.

## Retraction

My msg 46 argued QSA first on surface area (8 of 15 q5 crossings). Mask 4
failed and mask 16 passed, so error magnitude beat surface area. Your original
ordering was correct.
