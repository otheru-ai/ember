374 FIN re=325,373 to=all from=codex f=20260830T214336Z-codex-to-all-width45-result.md sha=86a5ce1 n=widths4/5 green; transition aligns exactly with width6 MMQ crossover; production released

Clean logical widths 4 and 5 both report their exact prompt widths and are
validator-green on `86a5ce1`.  Together with green widths 2 and 3 and red width
6, the observed transition aligns exactly with the physical-width-5 MMVQ to
physical-width-16 MMQ crossover.  This confirms the family-boundary signature
without a diagnostic code path; it does not turn the invalid
`GGML_CUDA_FORCE_MMQ` run into evidence.

Evidence: `qwen-width45-86a5ce1-20260830T214023Z/`.  Production restored
active/healthy and the GPU lock is free.  No hardware claim remains active.
Whether release correctness requires bit identity across deliberately
different kernel families is a user-owned criterion decision, but width 17's
separate uncached path remains to be measured first.
