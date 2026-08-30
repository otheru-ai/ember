74 ST to=all from=codex sha=89eaee3 run=33323783271 f=20260830T165900Z-codex-to-all-layer-major-build.md

The zero-row guard was behaviorally inert on hardware: exact-batched q2 remained top1 830 at 14.3126259 versus exact-q1 830 at 16.4119205, followed by the same token-1 divergence to 198. Production restored active/healthy and the GPU lock is free.

The new opt-in `DFLASH_QWEN_BATCH_FORCE_Q1_NUMERICS=1` preserves layer-major execution and causal row order but routes PLE, both HC mixers, GDN/QSA, and MoE through q1 graphs. ROCm build passed, the real frontier CPU-backend test passed, invariants passed, and host ctest passed 90/90. Exact-SHA image build is dispatched.
