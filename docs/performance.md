# Measured performance

All figures below come from one host: AMD Strix Halo (`gfx1151`), 128 GiB
unified memory, native Linux, ROCm 7.1. The model is the 85.3 GiB
DeepSeek-V4-Flash ROCMFPx quant with the DSpark drafter loaded and speculative
decoding on unless a column says otherwise. One measurement per data point --
these are not averaged over repeats, so treat single-digit-percent differences
between adjacent rows as noise.

## Time to first token

TTFT is dominated by prefill, and prefill on this model is superlinear in a way
that matters: throughput peaks in the low thousands of prompt tokens and then
falls, so TTFT grows faster than the prompt does. At the supported ceiling of 131,072 tokens a
cold prompt takes over ten minutes before the first token appears.

| prompt tokens | prefill tok/s | **cold TTFT** |
| ---: | ---: | ---: |
| 43 | 71.1 | 0.6 s |
| 862 | 274.5 | 3.1 s |
| 3,925 | 330.9 | 11.9 s |
| 18,553 | 289.7 | 1 min 4 s |
| 38,059 | 263.2 | 2 min 25 s |
| 77,068 | 218.8 | 5 min 52 s |
| 116,077 | 182.1 | **10 min 37 s** |

TTFT here is `prompt_tokens / prefill_tok_s`. It excludes queueing and the first
decode step, both of which are small against these numbers but are not zero.

The low figure at 43 tokens is not a regression: a prompt that short cannot fill
the GPU, so fixed per-request work dominates and the tok/s ratio looks bad while
the absolute latency (0.6 s) is the best in the table.

Speculative decoding does not affect prefill. Measured with it disabled, the
same prompts prefill at 73.4 / 300.6 / 345.4 / 290.9 / 264.4 / 220.1 / 182.0
tok/s -- within noise of the spec-on column at every length except the two
shortest, where the drafter's own warmup is being charged to a very small
denominator.

## Time to first token with a warm prefix cache

The number above is the **cold** cost, and for the workload Ember is built for
it is paid once per conversation, not once per turn. The prefix cache commits at
turn boundaries and restores an exact match:

| turn | prompt tokens | prefill | restored |
| --- | ---: | ---: | --- |
| 1 (cold) | 6,011 | 17,693 ms | 0 (0%) |
| same prompt repeated | 6,011 | 17,145 ms | 0 (0%) |
| 2 (history + new) | 6,053 | 17,502 ms | 0 (0%) |
| 3 (history + new) | 6,063 | **194 ms** | 6,053 (100%) |

Turn 3 is **90x faster** than turn 1 for a marginally longer prompt.

Repeating an identical single-turn prompt caches nothing, and that is by design,
not a defect: `anchor_cut` in `src/model/kv_cache.c` requires an anchor at least
`anchor_min = 512` tokens in, at a turn boundary, so a prompt with no turn
structure has no valid cut point. Benchmarks that measure cache behaviour by
resending one prompt will therefore report no speedup. A growing conversation --
the agent and chat workloads this is for -- commits on turn 2 and hits fully on
turn 3.

Practical consequence: quote cold TTFT for a first request against a long
document, and warm TTFT for every turn after it. They differ by two orders of
magnitude and neither one alone describes the system.

## Decode

| prompt tokens | spec on | spec off | speedup | acceptance |
| ---: | ---: | ---: | ---: | ---: |
| 43 | 37.78 | 23.32 | 1.62x | 0.981 |
| 862 | 37.98 | 22.69 | 1.67x | 0.981 |
| 3,925 | 36.56 | 22.70 | 1.61x | 0.967 |
| 18,553 | 30.65 | 20.92 | 1.47x | 0.944 |
| 38,059 | 24.32 | 19.00 | 1.28x | 0.974 |
| 77,068 | 18.01 | 16.61 | 1.08x | 0.969 |
| 116,077 | 14.50 | 14.86 | **0.98x** | 0.969 |

Decode tok/s, 256 generated tokens, identical generation task at every context
length so that only the prompt varies.

This sweep predates the `ggml_cpy` collapse in the compressor step (16 dispatches
per layer per token down to 4, and 8 down to 2). On the current build the
benchmark harness measures 39.0 tok/s at short context, so the short-context rows
here read about 3% low; the shape of the curve is unaffected.

Speculation's advantage decays monotonically with context and reaches
break-even somewhere near 100k tokens. Acceptance stays high throughout
(0.94-1.00), so the decay is not the drafter getting worse -- it is the verify
step growing with the KV span while the draft step does not, which raises the
cost coefficient in the speculative speedup until the extra tokens no longer pay
for it. `DFLASH_DS4_SPEC_MAX_CTX` defaults to 131072; the measurement above
means speculation is roughly neutral, not beneficial, above ~100k.

## Roofline position

Measured with `rocprofv3` PMC counters (`FETCH_SIZE`, `WRITE_SIZE`) against
durations from a separate trace pass, because collecting counters serializes
dispatches and inflates every duration.

| phase | achieved | vs 212 GB/s peak | GPU busy | dispatches/token |
| --- | ---: | ---: | ---: | ---: |
| decode | 159.0 GB/s | 75% | 70.3% | 2,249 |
| prefill | 125.6 GB/s | 59% | 92.9% | 11 |

Decode is bandwidth-bound and its remaining headroom is mostly launch gaps --
2,249 dispatches per token leaves 29.7% of the wall clock idle. Prefill keeps
the GPU busy but at lower bandwidth, because it is limited by specific kernels
rather than by memory:

| prefill kernel | share | achieved | vs peak |
| --- | ---: | ---: | ---: |
| D=512 flash attention (sparse/indexed) | 22.5% | 194 GB/s | 92% |
| `mul_mat_q<ROCMFP2>` (MoE experts) | 22.1% | 67 GB/s | 32% |
| `mul_mat_q<ROCMFP4_FAST>` | 10.1% | 136 GB/s | 64% |
| D=512 flash attention (dense) | 8.9% | 53 GB/s | 25% |

`FETCH_SIZE` counts L2 misses, which the 32 MB MALL can serve without touching
DRAM, so "vs peak" is an upper bound on true memory utilisation -- one kernel
(`rms_norm_f32`) reads at an apparent 131% of DRAM peak. The ordering is still
meaningful, but a high score here is not evidence that a kernel is DRAM-bound.

The flash-attention row is a worked example. At an apparent 92% it looks like
only bytes could matter, so the compressed half of its KV span was made readable
at its native F16 width -- exactly token-exact, and 40% fewer bytes for the span
each query token actually reads. It measured **3.6-5.9% slower** at every prefill
length above 2k, because the kernel's 512-deep unrolled dot product has to be
instantiated once per KV width and the second copy costs more VALU throughput
than the bandwidth saves. The change was reverted. Settle bandwidth-versus-VALU
questions with an A/B, not with this counter.

## Reproducing

The context sweep holds the generation task fixed and varies only prompt length,
alternating `DFLASH_DS4_SPEC=1` and `DFLASH_DS4_SPEC=0` per point. Both prefill
and decode throughput are read from the server's `usage.timings` block, not from
wall-clock timing at the client, so client-side overhead is excluded.

Profiling uses one `rocprofv3` pass per counter -- requesting two `--pmc`
counters in one pass faults on `gfx1151` -- plus a separate trace pass for
durations.
