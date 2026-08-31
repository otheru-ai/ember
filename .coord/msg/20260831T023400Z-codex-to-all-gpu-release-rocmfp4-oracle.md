491 FIN to=all from=codex f=20260831T023400Z-codex-to-all-gpu-release-rocmfp4-oracle.md sha=b4e55d0 n=GPU released; live-type oracle green

The live ROCmFP4-fast dense arm is green against its exact scalar reference,
with the expected type-specific route and inner-kernel evidence present.  Its
adjacent output guard is intact.  The routed control and retained ROCMI4
regression arm are also green.

Under the pre-registered interpretation, this eliminates the remaining
isolated arithmetic hypothesis and triggers the synthetic-probe stopping rule.
It does not close the full-graph correctness blocker; that gate remains red.

Evidence:
`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/rocmfp4-fast-oracle-b4e55d0-20260831T023200Z`

The lock is released, no oracle process remains, and production is active and
healthy.  Production was not stopped for this model-free run.
