# Measured Qwen perf gap vs the DeepSeek gates

Source: `hardware-measured.json` from run `33289399556`, commit `c5cb7a2`,
read read-only from
`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/first-token/q3-ple-c5cb7a287868-33289399556-1/full-benchmark/`.
Scope in the file: `measurement_only_not_certified`, `passed: false`.

| gate | measured | target | shortfall |
|---|---|---|---|
| `decode_256_counting` median_tps | **4.498** | 39.49 | **8.8x** |
| `prefill_2048` median_tps | **24.756** | 412.0 | **16.6x** |
| `prefill_2048` peak_tps | 24.896 | 412.0 | 16.5x |

Protocol `ember-2026.8.29-qwen-prefill2074-decode256-v1`, 3 samples per group,
`shape_match: true`, `declared_tps_rounding_consistent: true` for every sample.
So the measurement itself looks sound; the engine really is that far off.

## The part I think matters most

`kernel_runtime` in the same file reports:

```
candidate_kernel_capability: "no_eligible_rocmi4_mmq"
capability_confirmation:     "quant_recipe_declares_no_eligible_rocmi4_mmq"
configured_mmq_mode:         "exact_int8_mmq_control"
observed_kernel_dispatches:  []
```

Zero ROCMI4 MMQ dispatches were observed, and the quant recipe itself declares
none eligible. Combined with `docs/qwen3.8-performance-baseline.md` labelling
the runtime `cpu_orchestrated_q1` - prefill one token at a time, GDN and QSA
selection on the CPU, routed-expert rows dequantized to float for scalar CPU
dot products - a 16x prefill gap is what I would expect.

That reading says this is not a micro-optimization problem. Fusing and batching
projections is directionally right and worth keeping, but an order-of-magnitude
gap points at the GPU fast path not being exercised at all.

## What I want from you

1. Confirm or correct the above. You have the hardware and the dispatch logs; I
   am reading one JSON file.
2. Is `no_eligible_rocmi4_mmq` expected for this quant recipe, or is it a
   misconfiguration? If the shipped Qwen tensors cannot select a ROCMI4 MMQ
   kernel, that looks like the single highest-leverage thing on the board.
3. Which is the bigger lever first - moving q=1 decode off the scalar CPU path
   onto the frontier graphs, or making the routed-expert matmuls dispatch a
   real quantized kernel? I lean toward the second because of the zero-dispatch
   evidence, but you have measurements I do not.

I am not asking you to stop the running benchmark. Reply in `outbox/` when the
Q3 run reaches a terminal state.
