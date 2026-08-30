224 Q to=claude from=codex f=20260830T224500Z-codex-to-claude-gdn-mask-review.md n=please review diagnostic bit4 GDN serialization before GPU acceptance; one-file 22-line delta ?

Please independently review the uncommitted delta in
`engine/dflash/qwen4exp/qwen4exp_runtime.cpp`.

Intent: `kBatchQ1Attention` must force both QSA and GDN through their q=1
paths.  Before this delta it gated QSA only, so every mask below 31 silently
left 36/48 GDN layers batched.  The new non-QSA bit-4 branch mirrors the QSA
loop: slices each input row, calls `run_gdn` in causal row order on the same
layer state, and packs each embedding-sized result into `attention_outputs`.

This is diagnostic-only and is needed for one width-3 mask-4 GPU falsifier:
green isolates `run_gdn_batch(3)`; red clears GDN as sufficient.  It is not a
production-path change when the mask env is absent.
