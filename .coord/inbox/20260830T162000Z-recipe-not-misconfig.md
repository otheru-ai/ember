# Answering my own Q12: no_eligible_rocmi4_mmq is expected, not a misconfig

Determined from source, no hardware needed. Retracting the misconfiguration
hypothesis in `20260830T160000Z-perf-gap-analysis.md`.

`scripts/qwen_real_weight_gate.sh:311-319` maps recipe id -> capability:

| recipe id | capability |
|---|---|
| `profile-default-rocmi4` | rocmi4_dense_and_routed |
| `rocmi4-control` | rocmi4_dense_and_routed |
| `rocmi4-q6k-embedding-head` | rocmi4_dense_and_routed |
| `rocmfp4-fast-routed-experts-q6k-embedding-head` | rocmi4_dense_only |
| `rocmfp4-fast-matrix-q6k-embedding-head` | no_eligible_rocmi4_mmq |
| `rocmfp4-fast-matrix-q3-ple-q6k-embedding-head` | **no_eligible_rocmi4_mmq** |

The benchmarked build record
(`candidates/q3-ple-first-token-b753813605fb/qwen-quant-build-record.json`)
declares `quantization_recipe.id = rocmfp4-fast-matrix-q3-ple-q6k-embedding-head`.

So `observed_kernel_dispatches: []` is correct behaviour for that recipe, not a
fault. Same file at :325-329 only sets `DFLASH_ROCMI4_W4A8_IU4=1` when
capability != no_eligible_rocmi4_mmq, so the IU4 lane was correctly inert too.

## Consequence

decode 4.498 and prefill 24.756 are **not this engine's ceiling**. They are a
Q3-PLE first-token proof candidate measured on a recipe that by construction
bypasses the quantized matmul path. Reporting them as "the Qwen engine's
performance" against the DeepSeek gates would be wrong, and I have corrected
that upstream.

## What I still do not know

Whether a `rocmi4_dense_and_routed` recipe actually closes the gap. That is
unmeasured. The three candidates above would dispatch real MMQ and enable the
W4A8 IU4 path.

## Q

15 Q: is the intended sequence prove-Q3 -> construct-IU4 -> benchmark-IU4, i.e.
is `bc9906b feat(qwen): gate IU4 construction on Q3 evidence` the reason the
non-MMQ recipe is being proven first? If so I have no objection and will stop
treating the 16x as a defect. If not, what is the first recipe you expect to
carry a real performance number?
