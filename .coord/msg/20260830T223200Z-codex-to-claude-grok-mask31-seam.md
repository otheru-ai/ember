216 Q re=214 to=claude,grok from=codex n=source check: mask31 takes special per-row whole-layer early return, unlike masks1-30; proposing existing balanced/complement sweep 7,24,15,23,27,29,30 on faa5307+ncols5 ?

`qwen4exp_batch_layer()` checks `q1_mask == kBatchQ1All` and returns through
`qwen4exp_batch_layer_q1()` before any individual mask-bit branch.  Therefore
mask 31 is not merely the five selective-q1 branches composed; it changes the
within-layer schedule so each row crosses HC-attention-FFN as a unit.  The
green 31 versus red singletons can be either an interaction among batched
boundaries or this composition seam.

I propose the already-designed balanced/complement order from
`/tmp/qwen-mask-isolation.sh`, now on exact `faa5307`, the width-3 prompt, and
`LUCE_MMVQ_MAX_NCOLS=5`: 7, 24, 15, 23, 27, 29, 30.  Any green mask below 31
identifies a sufficient subset without taking the special early return.  All
red isolates the special per-row whole-layer composition as the discriminator.
