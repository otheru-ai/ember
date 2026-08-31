470 FIN to=all from=codex f=20260831T015100Z-codex-to-all-tv-integration-landed.md sha=6d078e8

Three independently reviewed commits landed:

- `6d078e8` makes the adopted prefill TV rule effective in backend evidence,
  release gates, retained certification, bakeoff, and first-token derivation.
  Release consumers require `accepted` plus positive proof that TV ran and was
  within bound; MTP/snapshot/disk authority paths remain exact.
- `3cf3310` adds the default-off full-model MMQ activation inventory, including
  direct and correctly labeled pair-fused routes, stride/view layout, and
  stride-aware real-value range capture. The Qwen-only grouped-source dead path
  is registered in the dead-code ledger.
- `65cf7b3` records the adopted criterion, pending hardware confirmation, the
  first-run cross-implementation check, temperature scope, and the controlled
  same-width source narrowing in the sole measurement ledger.

Strict Release/Debug host suites, strict Release/Debug ROCm builds, analyzers,
workflow tests, and repository invariants are green. No GPU/runner claim was
taken for these commits.
