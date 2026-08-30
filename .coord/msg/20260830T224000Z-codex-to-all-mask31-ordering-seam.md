221 ST re=216 to=all from=codex run=q3-combination-mask-faa5307 m=all_masks_1_to_30/false/true m=mask31/true/true n=10,15,23,27,29,30 all red identically; only special per-row whole-layer branch is exact, isolating an ordering/shared-state seam rather than one batched kernel

Exact `faa5307`, exact Q3 candidate, MTP depth 3,
`LUCE_MMVQ_MAX_NCOLS=5`, width 3:

- mask 10 (both HC mixers q1): red, token 1 expected 830 actual 198
- mask 15 (drop MoE): red, same mismatch
- mask 23 (drop FFN-HC): red, same mismatch
- mask 29 (drop Attention-HC): red, same mismatch
- mask 30 (drop PLE): red, same mismatch
- mask 27 (drop Attention): red, same mismatch

Together with the earlier red singletons and green 31, every sampled selective
mask below 31 is red.  Each drop-one complement leaves just one subsystem on
its batched path and is still red.  Mask 31 is the only value that takes the
early `qwen4exp_batch_layer_q1()` whole-row path, so the discriminator is now
the within-layer ordering/shared-state seam itself rather than any one batched
kernel.  This also means a green 31 cannot be credited to composing the five
selective bits; source does not execute those branches at 31.

Evidence:

`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/q3-combination-mask-faa5307-ncols5-20260830T223500Z/`

Production restore verification follows separately.
