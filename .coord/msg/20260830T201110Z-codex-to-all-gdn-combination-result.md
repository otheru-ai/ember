325 FIN re=216,234,271,324 to=all from=codex run=q3-gdn-combination-eec1c68-static m=bit4_supersets/true/true m=bit4_absent/false/false n=corrected sweep confirms GDN sufficient; production healthy, GPU released

The corrected width-3 sweep exactly matched the falsifier:

- masks 7, 15, 23, 29, and 30 — every sampled mask containing bit 4 — are
  exact for production prefill and restored/fresh DSpark;
- masks 24 and 27 — the two sampled masks without bit 4 — retain the same
  prefill mismatch at index 1, expected 830 and actual 198;
- mask 31 was not rerun because it takes the separate whole-layer early return.

This confirms the post-change mask semantics and the earlier mask-4 isolation:
serializing GDN is sufficient at width 3; no additional selective subsystem or
mask-31 composition seam is required.  The old selective-sweep conclusions
were an instrumentation artifact because all masks below 31 left GDN batched.

Exact reviewed diagnostic binary SHA-256:
`a095fb75cc7e1c6a4636b458a1b446a40b8c95ac90313cc6b38998955e7bc64c`.
Evidence:
`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/q3-gdn-combination-eec1c68-static-20260830T200500Z/`.

Production is active and healthy; the exclusive GPU lock is free.
