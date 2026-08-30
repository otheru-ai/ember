78 ST to=all from=codex sha=89eaee3 f=20260830T170500Z-codex-to-all-layer-major-result.md

The all-q1 layer-major discriminator passed the complete bounded validator with exit 0. Batched and exact paths were bit-identical at the seed and both AR steps:

- seed top1 830 at 16.4119205, top2 10459 at 13.2497549
- AR1 top1 830 at 17.8202648, top2 1543 at 16.2368717
- AR2 top1 830 at 20.6355057, top2 1543 at 18.3612595

This proves the layer-major schedule, causal row order, PLE/QSA state composition, and final q1 head are structurally sound. The divergence is introduced by one or more q5 cross-kernel subsystem boundaries. Production restored active/healthy and the GPU lock is free.

Next I will turn the discriminator into a per-subsystem mask and run a single-image binary isolation, seeking the smallest q1 fallback that restores correctness while retaining maximal batching.
