# Measured performance

All figures below come from one host: AMD Strix Halo (`gfx1151`), 125 GiB
installed unified memory, native Linux, ROCm 7.14 (the version pinned by the
release and development images). The model is the 85.3 GiB
DeepSeek-V4-Flash ROCMFPx quant with the DSpark drafter loaded and speculative
decoding on unless a column says otherwise. One measurement per data point --
these are not averaged over repeats, so treat single-digit-percent differences
between adjacent rows as noise.

The current release-candidate bundle (`2026.8.24`, measured 2026-08-23) is
tracked separately in [`docs/perf/data.json`](perf/data.json) and the
[performance dashboard](https://otheru-ai.github.io/ember/perf/). It reports a
39.59 tok/s median over three 256-token throughput samples, with a 98.1% median
draft acceptance rate. That bundle is explicitly **not certified**: it has no
depth series and was produced by the engineering harness. The tables below
retain the longer context-depth series used for the certified-reference
workflow, and are not interchangeable with the dashboard's release history.

## Time to first token

TTFT is dominated by prefill, and prefill on this model is superlinear in a way
that matters: throughput peaks in the low thousands of prompt tokens and then
falls, so TTFT grows faster than the prompt does. At the supported ceiling of 131,072 tokens a
cold prompt takes over ten minutes before the first token appears.

| prompt tokens | prefill tok/s | **cold TTFT** |
| ---: | ---: | ---: |
| 43 | 72.6 | 0.6 s |
| 862 | 280.3 | 3.1 s |
| 3,925 | 343.3 | 11.4 s |
| 18,553 | 300.6 | 1 min 02 s |
| 38,059 | 272.3 | 2 min 20 s |
| 77,068 | 226.2 | 5 min 41 s |
| 116,077 | 186.7 | **10 min 22 s** |

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
| 43 | 39.24 | 23.37 | 1.68x | 0.981 |
| 862 | 39.47 | 22.69 | 1.74x | 0.981 |
| 3,925 | 37.98 | 22.73 | 1.67x | 0.981 |
| 18,553 | 32.22 | 20.94 | 1.54x | 0.981 |
| 38,059 | 24.94 | 19.02 | 1.31x | 0.953 |
| 77,068 | 18.25 | 16.61 | 1.10x | 0.978 |
| 116,077 | 14.63 | 14.88 | **0.98x** | 0.978 |

Decode tok/s, 256 generated tokens, identical generation task at every context
length so that only the prompt varies.

**Read the generation task before reading the numbers.** It is
`benchmark.py`'s decode prompt: *"Write a very long comma-separated sequence of
consecutive positive integers beginning at 1."* Counting is close to the easiest
possible workload for a drafter, and every decode figure this project has
published -- 33.60, 37.49, 39.0 -- is that task. It is a ceiling, not a typical
result.

The same server, same configuration, measured across ten prompts spanning how
predictable the *continuation* is (speculation on vs off, same prompt both
sides):

Medians of three repeats each. `p` is the fraction of offered draft blocks that
come back FULLY accepted, which is what decides whether the wide verify path is
worth taking.

| workload | p | spec | AR | speedup |
| --- | ---: | ---: | ---: | ---: |
| alphabet | 0.976 | 40.70 | 23.66 | 1.720x |
| multiples of 7 | 0.952 | 40.50 | 23.67 | 1.711x |
| repeat a sentence | 0.976 | 39.97 | 23.67 | 1.689x |
| count integers | 0.952 | 39.73 | 23.64 | 1.681x |
| JSON array | 0.952 | 39.70 | 23.64 | 1.679x |
| code | 0.760 | 28.92 | 23.84 | 1.213x |
| factual list | 0.714 | 27.88 | 23.71 | 1.176x |
| prose | 0.056 | 23.15 | 23.69 | **0.977x** |
| essay | 0.000 | 23.20 | 23.67 | **0.980x** |
| creative | 0.000 | 23.20 | 23.69 | **0.979x** |

Speculation pays on seven of ten and is a small loss on the three where the
drafter never lands a whole block. The AR baseline is flat at 23.6-23.8 tok/s
regardless of prompt, so decode on genuinely unpredictable prose is about
**23 tok/s** against 40 on highly predictable output.

Note that `p` is not acceptance. `factual list` averages 4.13 of 5 tokens
accepted -- excellent -- but only 71% of its blocks are whole, and it is
wholeness that the batched verifier qualifies on. Mean acceptance and `p` rank
these workloads differently, which is why acceptance-based gating could not
price them.

Getting the bottom three to parity took a per-request abandon once the verifier
has visibly failed to qualify; getting `code` and `factual list` above 1.0x took
replacing the qualification rule itself. Before that work:

| workload | before | after |
| --- | ---: | ---: |
| code | 0.939x | 1.213x |
| factual list | 0.931x | 1.176x |
| prose | 0.877x | 0.977x |
| essay | 0.876x | 0.980x |
| creative | 0.856x | 0.979x |
| JSON array | 1.528x | 1.679x |
| alphabet | 1.592x | 1.720x |

Acceptance does not predict which side a prompt lands on: the factual list
accepts 4.13 of 5 drafts and still loses 7%. What separates them is whether the
batch verifier qualifies, which needs consecutive *fully* accepted blocks --
see `DSparkBatchVerifyGate` and the note in `deepseek4_dspark_spec.cpp`.

This also qualifies the acceptance column in the context sweep above. Those
0.94-1.00 figures are not evidence that the drafter holds up at depth; they are
the counting task pinning acceptance near 1.0 at every length.

Measured on the current build, so this includes the `ggml_cpy` collapse in the
compressor step, the `reduce_rows`/`rope` launch fixes, and the speculation
gating work. Acceptance now holds at 0.98 out to 116k where the previous sweep
dipped to 0.944 at 18.5k -- the drafter did not change, the qualification gate
simply stopped ejecting the request on stray partial blocks.

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

**Two ways this table misleads, both of which cost real work before they were
noticed.**

`FETCH_SIZE` counts L2 misses, which the 32 MB MALL can serve without touching
DRAM, so "vs peak" is an upper bound on memory utilisation, not a measurement of
it -- `rms_norm_f32` reads at an apparent 131% of DRAM peak. A high score here is
not evidence that a kernel is DRAM-bound.

And GB/s is not comparable across quantisation formats. The ROCMFP2 row looks
like half the efficiency of the ROCMFP4_FAST row, but ROCMFP2 is 2.50 bpw
against 4.25, so it moves fewer bytes for the same arithmetic. Per weight the
two are within 5% of each other:

| kernel | bytes/weight | weights processed |
| --- | ---: | ---: |
| `mul_mat_q<ROCMFP2>` | 0.3125 | 1.88e11 /s |
| `mul_mat_q<ROCMFP4_FAST>` | 0.53125 | 1.98e11 /s |

Both are instruction-issue bound -- 16 WMMA instructions in roughly 1,315 -- and
neither has a bandwidth problem. There is no 2x sitting in the ROCMFP2 unpack,
and an earlier revision of this document claimed there was.

The flash-attention row is a worked example. At an apparent 92% it looks like
only bytes could matter, so the compressed half of its KV span was made readable
at its native F16 width -- exactly token-exact, and 40% fewer bytes for the span
each query token actually reads. It measured **3.6-5.9% slower** at every prefill
length above 2k, because the kernel's 512-deep unrolled dot product has to be
instantiated once per KV width and the second copy costs more VALU throughput
than the bandwidth saves. The change was reverted. Settle bandwidth-versus-VALU
questions with an A/B, not with this counter.

## Prefill kernel work

Two kernels were 5.3% of prefill between them and neither was slow -- both were
launched wrong, for a tensor shape DS4 does not have.

| kernel | before | after | |
| --- | ---: | ---: | --- |
| `reduce_rows_f32` | 1001.8 ms | 131.5 ms | 7.6x |
| `rope_norm` | 420.0 ms | 212.5 ms | 2.0x |
| **prefill total** | **26632 ms** | **25577 ms** | **-4.0%** |

`reduce_rows_f32` mapped one block per row irrespective of row length. DS4 sums
its 4 hyper-connection streams as a `[4 x 8388608]` `sum_rows`, so that launched
8.4 million workgroups of 32 threads to add four floats each, with 28 lanes idle
in every one. Fixed in two steps: cap `gridDim` and walk rows with a stride
(-36%), then collapse the reduction for short rows to its minimal exact form
(a further 4.9x).

`rope_norm` hardcodes a 256-thread block, and each thread rotates one dim-pair --
so a block always covers 512 dims. DS4 ropes `n_rot = 64`, leaving 32 of 256
threads with work and launching seven dead waves per block. Fixed by sizing the
block to the row.

Both changes are bit-exact, and both were verified by differential test rather
than by the DSpark validator. That validator compares AR against speculative
against batched decode *within one build*; when every path uses the changed
kernel it will pass whether or not the arithmetic moved. `GGML_REDUCE_ROWS_SHORT`
exists so one binary can produce both reductions and they can be diffed
directly. Output over four prompts x 256 deterministic tokens is byte-identical
in both cases.

What is left in these two: `rope_norm`'s remaining 212 ms is dominated by
`rope_theta_fp64`, which computes theta in double precision at a small fraction
of fp32 rate on gfx1151 -- narrowing it would change every rotated value, so it
is a quality decision rather than a launch fix. `reduce_rows_f32` now runs at
about 0.75 ms per call against a ~0.8 ms memory-bound floor and is finished.

### What was investigated and rejected

The two largest prefill kernels -- D=512 flash attention at 23.3% and
`mul_mat_q<ROCMFP2>` at 23.0% -- were both examined for a rewrite and neither
was taken.

An instruction census of the shipped flash-attention kernel
(`<float, half, 4, indexed, 4>`, 7,287 instructions) is:

| class | share |
| --- | ---: |
| select (`v_cndmask`, `s_cselect`) | 23.6% |
| FMA | 19.4% |
| address arithmetic | 18.3% |
| waits, `s_delay_alu` | 14.9% |
| memory | 9.7% |
| moves | 5.6% |

Two conclusions follow. WMMA is not the lever: the dot product is under a fifth
of the work, so collapsing it caps at 1.24x by Amdahl -- and every matrix form on
RDNA 3.5 takes f16/bf16/iu8/iu4 (ISA table 33; there is no fp32 matrix path), so
that 1.24x would also cost prefill its fp32 baseline.

The better theory was that head grouping amortises overhead: the first three
eighths of the kernel are sparse-index bookkeeping with almost no FMA, and the
grid is (n_tokens, n_heads/4), so a token pays it 16 times over identical rows.
Widening the group to 8 heads -- the ceiling this kernel body allows -- gave 34%
fewer instructions per head and 58% fewer selects per head, and ran **8.9%
slower**, taking prefill from 25520 to 26349 ms. Two costs scale against it:

  * `scores[HEADS * score_stride]` is staged in LDS, so 15.4 KB becomes 30.1 KB
    and occupancy halves from 4 resident blocks per CU to 2.
  * The value pass walks the union of per-head envelopes, so a wider group loads
    more rows only to discard them per head.

Both worsen monotonically with group width, which is why 4 is the setting.

So a 64-head kernel cannot be this kernel with more heads. It would have to be
row-tiled the way `ds4_decode_attn_wmma_partial` is -- staging a 16-row tile
whose LDS is constant in the span, with online softmax and no envelopes -- which
is a rewrite, not a parameter change, and it would still carry the f16 numerics
change. Not attempted.

`mul_mat_q<ROCMFP2>` was rejected for a different reason: it is not slow. See
the per-weight comparison under Roofline position.

Prefill is cold-start work. The prefix cache removes nearly all of it after the
first turn, so a 4% prefill gain is worth far less than it looks next to the
decode figures above.

## Reproducing

The context sweep holds the generation task fixed and varies only prompt length,
alternating `DFLASH_DS4_SPEC=1` and `DFLASH_DS4_SPEC=0` per point. Both prefill
and decode throughput are read from the server's `usage.timings` block, not from
wall-clock timing at the client, so client-side overhead is excluded.

Anything measured through `benchmark.py` inherits its decode prompt, which is
integer counting. Quoting a decode number without saying so overstates ordinary
throughput by roughly 1.6x -- see the workload table above.

Profiling uses one `rocprofv3` pass per counter -- requesting two `--pmc`
counters in one pass faults on `gfx1151` -- plus a separate trace pass for
durations.

### Which number should be quoted?

Use the workload table for user-facing expectations and always name the
workload. The integer-counting task is intentionally drafter-friendly and is a
ceiling, not a typical decode rate. Use the autoregressive column when comparing
against a server without DSpark. Use the dashboard only for release-to-release
comparisons, because it records the image, model digests, sample count, and
certification flag alongside every number. Hardware certification is the
authoritative release gate; a faster local harness run does not replace it.
