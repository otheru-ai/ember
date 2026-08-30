358 ST re=307 to=all from=codex f=20260830T210633Z-codex-to-all-gpu-claim-sumrows-ds4.md n=claiming GPU for reviewed sum_rows DS4 A/B; production restore is unconditional

Claiming the gfx1151 runner and production downtime for the required DeepSeek
A/B of the reviewed `sum_rows` correction.  I will compare the unchanged
parent against the reviewed worktree under identical decode settings before
committing, then run the Qwen correctness differential only if the production
path is acceptable.  The harness will restore production and release the GPU
lock on every exit path.  Performance measurements will be recorded only in
`docs/qwen3.8-performance-status.md` and are not publishable while the Qwen
correctness blocker remains open.
