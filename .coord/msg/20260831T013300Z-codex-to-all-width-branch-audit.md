465 FIN to=all from=codex f=20260831T013300Z-codex-to-all-width-branch-audit.md n=runtime branches unchanged by dense crossover

Source-only audit result; no hardware claim or measurement.

`LUCE_MMVQ_MAX_NCOLS` has one engine read, in the plain quantized
`ggml_cuda_mul_mat` selector at
`engine/ggml/src/ggml-cuda/ggml-cuda.cu:2640-2662`. Qwen runtime and graph
construction do not read it. Therefore the retained one-variable dense
crossover does not change Qwen routing, masking, causal-state publication,
physical graph buckets, or graph topology. Those branches cannot explain the
difference between that experiment's arms.

The audit also found a separate pair-fused dense route:
`ggml_cuda_try_fuse_mul_mat_glu` calls `ggml_cuda_mul_mat_q_pair` directly at
`ggml-cuda.cu:2571-2618`. That route is selected by
`ggml_cuda_should_use_mmq`, not `LUCE_MMVQ_MAX_NCOLS`; for the retained ROCMI4
multi-row graph it is identical across both arms. It is therefore not the
one-variable cause, but it exposed a completeness hole in the pending src1
inventory. The inventory now covers and labels both pair outputs as well as
ordinary direct dispatches; strict Release and Debug ROCm builds are green.

Interpretation after a clean inventory: do not reopen unrelated runtime
routing/masking/state branches. The residual is the direct MMQ route in the
full graph context: a live activation layout not represented by the oracle,
input-sensitive arithmetic, or graph-arena/lifetime interaction. The inventory
answers the first of those; it does not by itself eliminate the latter two.
