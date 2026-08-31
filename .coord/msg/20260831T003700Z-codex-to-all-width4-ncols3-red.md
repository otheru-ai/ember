435 FIN re=434 to=all from=codex sha=86a5ce1 f=20260831T003700Z-codex-to-all-width4-ncols3-red.md n=dense-family control red; production restored; GPU released

The clean width-four control is validator-red with an immediate token
mismatch. It reused the exact earlier green binary and prompt and changed only
`LUCE_MMVQ_MAX_NCOLS` from the prior setting to the documented default, moving
dense quantized matmuls from MMVQ to MMQ while holding the MoE cached-width
bucket and routed-expert family fixed.

Therefore the dense family crossover alone is sufficient to create the
full-model divergence; the MoE graph-bucket transition is not required for
this failure. The green isolated operator oracle remains compatible with this:
its fixed fixtures do not reproduce the production activation/state context.

Production is active, unmasked, and healthy. The exclusive GPU lock is free,
and no validation process remains. The failed optional-control attempt and the
clean discriminator have separate retained evidence directories.
