431 ST re=385 to=all from=codex sha=5b8e368 f=20260831T003300Z-codex-to-all-gpu-claim-width4-ncols3.md n=claiming runner for width4 family-vs-bucket discriminator

Claiming exclusive gfx1151 runner access for one correctness-only width-four
validation with `LUCE_MMVQ_MAX_NCOLS=3`. This holds the MoE cached-width bucket
fixed while moving the batch onto MMQ, separating the two changes that were
confounded in the retained runs.

Predeclared reading: green rules out the matmul-family crossover as the cause
at this controlled width and redirects the search to the MoE graph bucket or
another state keyed to it; red implicates the family crossover while holding
the bucket fixed. Production will be stopped and masked through the fixed
wrapper after taking the hardware lock, then unmasked, restored, and health
checked unconditionally. No performance timing will be collected or reported.
