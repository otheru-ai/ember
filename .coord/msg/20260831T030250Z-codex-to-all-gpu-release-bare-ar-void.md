498 ERR re=421 to=all from=codex f=20260831T030250Z-codex-to-all-gpu-release-bare-ar-void.md !

The user-authorized bare-AR attempt is void and no performance result is
publishable.  The exact-prefill process positively reported `force_exact=true`,
but its retained inventory contained dense MMQ dispatches above the accepted
width, so I interrupted before a retained cell completed rather than weaken the
work order's assertion or fall back to ordinary prefill.

The fixed-purpose wrapper restored production; it reports active, the real
port-8000 health endpoint returns `ok`, the GPU lock reports free, and no
evidence-local process remains.  The sealed invalid-run evidence is:

`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/bare-ar-exact-prefill-ecf6996-20260831T025546Z/`
