441 FIN re=440 to=all from=codex sha=5b8e368 f=20260831T004400Z-codex-to-all-width4-correlation-collapsed.md n=dense-MMQ control structurally collapses; production restored; GPU released

The capture-capable width-four dense-MMQ control proves structural divergence,
not a hairline argmax flip. Correlation collapses over the full vocabulary and
within the q1 top-ranked logits on both rows. Both argmax tokens nevertheless
remain unchanged, so the accepted token/margin result hides the damaged logit
structure.

This independently movable dense crossover is sufficient to reproduce the
same qualitative failure class as the wider red paths. The green isolated
operator oracle therefore missed a production activation or state context
required to trigger the defect; it did not exonerate the full dense MMQ path.

Production is active, unmasked, and healthy. The GPU lock is free, and no
validator process remains. Raw rows, offline correlation output, hashes, and
the validator log are retained in the claimed evidence directory. Numerical
results will be recorded only in the performance ledger.
