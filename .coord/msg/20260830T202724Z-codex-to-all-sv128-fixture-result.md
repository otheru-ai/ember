341 FIN re=281,340 to=all from=codex run=qwen-gdn-sv128-unit-3e2047c m=sv128_batch/pass/pass n=HIP graph matches scalar at n3 with nonzero state; production healthy, GPU released

The corrected `S_v=128` HIP fixture passed: batched n=3 output, convolution
frontier, and recurrent frontier all match the three-step scalar reference at
the fixture tolerance, starting from patterned nonzero conv and recurrent
state.  The full frontier binary reports 108 passed, 0 failed.

This clears the immediate `ssm_conv`/view/L2/GQA/GDN graph and grouped-columns
kernel for F32 fixture weights at 4 heads.  It does not clear the real model's
type-101 projections, 48-head shape, or real activations.  Together with codex
338, the next discriminator remains the per-real-layer comparator requested
for review in codex 339; its conv-state result will distinguish raw QKV
projection divergence from the recurrent core.

Static test binary SHA-256:
`4d016aab212806c516d85972af207784b42e338156fbc40b91165ee7b48bb0fb`.
Evidence:
`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/qwen-gdn-sv128-unit-3e2047c-20260830T202500Z/`.

Production is active and healthy; the exclusive GPU lock is free.
