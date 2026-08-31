503 FIN re=421 to=all from=codex f=20260831T112605Z-codex-to-all-gpu-release-bare-ar-complete.md !

Both user-ordered bare-AR context cells are complete.  Each has a discarded
same-workload warmup, the required retained repetitions, exact prompt and
completion shapes, `prefill_mode=exact-q1`, speculation absent, consistent
serialized timing, the required inventory enabled, and no dense MMQ dispatch
above the accepted width.  The result is ready for the measurements ledger;
no performance figure is copied into coordination.

The fixed-purpose wrapper restored production; it reports active, the real
port-8000 health endpoint returns `ok`, the GPU lock reports free, and no
evidence-local process remains.  Sealed evidence:

- shallow cell: `/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/bare-ar-exact-prefill-ecf6996-20260831T030532Z/`
- deep cell: `/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/bare-ar-exact-prefill-deep-ecf6996-20260831T032805Z/`
