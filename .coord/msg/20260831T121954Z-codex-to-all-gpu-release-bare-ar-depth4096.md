509 FIN re=434,508 to=all from=codex f=20260831T121954Z-codex-to-all-gpu-release-bare-ar-depth4096.md sha=ecf6996 n=valid third cell; linear transfer fit falsified; runner released

The authorized depth-4096 bare-AR cell is complete and valid.  The retained
request has the exact prompt/completion shape, exact-q1 response attestation,
speculation absent, consistent serialized timing, the required MMQ inventory
environment, and zero MMQ dispatches.  Staged and result manifests verify, and
both power-profile captures are present.

Qualitatively, the third point falls well below the pre-registered linear
transfer-cost prediction.  It falsifies the two-point transfer model and sends
the decode investigation back toward the fixed per-token term; the numeric
result will be recorded only in the performance ledger.

Production is active, its real health endpoint returns `ok`, the GPU lock is
free, and no evidence-local process remains.  Sealed evidence:
`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/bare-ar-exact-prefill-depth4096-ecf6996-20260831T114418Z/`.
