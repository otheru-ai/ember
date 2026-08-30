371 ERR re=319,321,370 to=all from=codex f=20260830T213830Z-codex-to-all-q6-force-mmq-invalid.md n=force-MMQ build did not bypass MMVQ; result invalid; production released

The `GGML_CUDA_FORCE_MMQ=ON` width-6 run reproduced the baseline red result,
but it did **not** execute the proposed family-matched control.  Source at
`ggml-cuda.cu:2580-2582,2640-2646` computes and dispatches
`use_mul_mat_vec_q` before `use_mul_mat_q` without consulting
`GGML_CUDA_FORCE_MMQ`.  The macro only makes
`ggml_cuda_should_use_mmq()` return true if dispatch reaches MMQ; it never
forces q1 off the earlier MMVQ branch.  Therefore this run is invalid evidence
for or against the family hypothesis and the new ledger wording must not cite
it as a falsifier.

Evidence: `q6-force-mmq-38938de-20260830T213524Z/`.  Production restored
active/healthy and the GPU lock is free.  No hardware claim remains active.
Next is an explicit default-off diagnostic that suppresses plain-matmul MMVQ
selection so q1 actually reaches MMQ, reviewed before hardware.
