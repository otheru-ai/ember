25 ST re=17 f=20260830T160420Z-host-repro-feasibility.md

One feasibility warning before implementation: existing CPU frontier tests use
small per-component specs, but the production composition path
`qwen4exp_step_prefill_batch_mrope()` is hard-coded to 48 layers, embedding
2560, 512 experts, and the real tensor shapes. Constructing a literal synthetic
`Qwen4ExpWeights` for the end-to-end public runtime would require production-
scale matrices, not a seconds-long fixture.

A cheap permanent guard probably needs one of:

1. factor the per-row composition/state logic into a dimension-parameterized
   pure helper and test that with small component outputs;
2. add an injectable/mock projection seam for the runtime test; or
3. reproduce only the suspected QSA/HC/PLE composition with the existing small
   frontier specs, rather than calling the full production step.

`15f6a97` changes MTP synchronization/proposal positions only. The production
prefill comparison target path should be independent of MTP side-cache state,
so I rank it below the target QSA/HC/PLE batch commits despite the RoPE angle.
