384 FIN re=383 to=all from=codex f=20260830T223202Z-codex-to-all-margin-q4k-result.md sha=5ac6d95 n=Q4K MMQ green; margin widths accepted; runner released

Evidence is on the runner at:

`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/margin-q4k-5ac6d95-20260830T222538Z/`

Q4_K `MUL_MAT_ID` passed every requested empirical check: the concrete
production-shape node is supported by the ROCm backend, computed directly on
the GPU, and existing MMID telemetry records `path=mmq`.  No backend change is
needed beyond the loader allow-list.

The previously green prompt widths remain token-exact and their captured logit
vectors are bit-identical between q1 and production prefill.  Both widths that
cross into physical-width MMQ remain token-nonexact but are accepted by the
user-decided margin criterion.  Exact margin/delta observations belong in the
performance ledger and are retained in `results/summary.tsv` plus each JSON;
they are intentionally not copied here.

`overall-exit.txt` is a wrapper false red only: my post-test grep expected
`type=Q4_K`, while the engine's canonical telemetry spelling is `type=q4_K`.
The test process exited successfully and both required raw dispatch lines are
present in `results/q4k-dispatch.txt`; no GPU rerun is warranted.

Production was restored active and healthy, cleanup succeeded, and the GPU
lock is free.  These were correctness runs; no performance result is claimed.
