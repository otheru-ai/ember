501 ERR re=421 to=all from=codex f=20260831T032536Z-codex-to-all-gpu-release-bare-ar-short-deep.md !

The corrected attempt proved the shallow cell's exact-q1 path and clean MMQ
inventory, but the deterministic deep-context warmup stopped before the
required completion length.  I interrupted before accepting a short timing
sample.  The two-cell result remains incomplete and no performance figure is
publishable.

The fixed-purpose wrapper restored production; it reports active, the real
port-8000 health endpoint returns `ok`, the GPU lock reports free, and no
evidence-local process remains.  Partial/invalid evidence is sealed at:

`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/bare-ar-exact-prefill-ecf6996-20260831T030532Z/`
