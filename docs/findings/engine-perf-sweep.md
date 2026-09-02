# Methodical engine performance sweep

2026-09-02. `tools/perf/hotpath_scan.py` enumerates allocation, sync, string,
copy and growth patterns inside loops across `engine/dflash`. It produces
signal, not proof — every hit below was read before being kept or discarded.

## Triage rule applied

A hit only matters if it is on a path the **shipped** configuration executes.
Two large groups failed that test and are recorded so nobody re-finds them:

* **`deepseek4_step_hybrid` (graph.cpp ~3923-4300)** holds the worst raw hits —
  a `std::vector<int32_t> top(n_used)` allocated **per token** inside a
  256-expert x 6-slot selection loop, which also recomputes
  `probs[cur] + bias[cur]` for incumbents on every comparison and re-tests
  `!bias_host.empty()` in the innermost loop. **Not shipped**:
  `deepseek4_loader.cpp:1170` sets `w.moe_hybrid = false`. Worth fixing only if
  hybrid MoE is ever turned on.
* Loader, vision-tower init and contract code dominate the `DEVICE_SYNC` and
  string hits. All one-time.

## Kept: prefill embed buffer re-allocated per chunk

`deepseek4_backend.cpp:1652`, the only allocation in `do_prefill`'s chunk loop:

```cpp
std::vector<float> embed(w_.n_embd * n_tok);
```

`std::vector<float> v(N)` value-initialises, so this **zero-fills 32 MiB** at
n_tok = 2048 and the embedder then overwrites all of it. Fresh each chunk, so
the allocator also faults in the pages again every time.

Measured, 48 chunks of 32 MiB (about a 98k-token prefill):

| | |
|---|---:|
| fresh vector per chunk | 246.97 ms |
| reused member buffer | 7.03 ms |
| **saved** | **239.9 ms (97.2%)** |

**Perspective, so this is not oversold:** 240 ms is a large fraction of *that
buffer's* cost and a small fraction of the prefill it sits in — well under 0.1%
of a 98k-token prefill. It is a free win, not a fix for the row C regression.
Hoist to a member, `resize()` once to the maximum chunk, reuse.

## Already-known items confirmed by the sweep

* The per-layer causal mask rebuild — see the addendum in
  `isa-assembly-opportunities.md`. Larger than the embed buffer, but a blanket
  hoist is wrong and the memoise-by-ratio form needs a correctness check first.
* The cross-TU predicate in the mask loop — fixed in `f4bba5a`.

## What the sweep did not find

No device synchronisation inside a per-token or per-layer loop. No string
formatting on the decode path. No unreserved `push_back` in a loop whose trip
count is both large and known in advance. The engine's hot paths are, on this
axis, already clean — which is worth recording as a negative result rather than
leaving the impression that a sweep found nothing because it looked poorly.


## Stop hand-auditing: the engine already measures this

`DFLASH_DS4_TIMING=1` emits `[deepseek4-timing]` per phase, and it separates
**build from compute from read** for attention, FFN and routing:

    attn_build_us   attn_compute_us   attn_read_us
    ffn_build_us    ffn_compute_us    ffn_read_us
    route_build_us  route_compute_us  route_read_us  route_select_us
    hc_pre_build_us hc_pre_input_us   hc_pre_compute_us
    embed_us  output_us  sample_us  emit_us

That is exactly the host-versus-device split every finding in this document has
been arguing about from source. `attn_build_us` is graph construction — the
phase that contains the 48 MiB-per-token arena — measured apart from the GPU
work it feeds.

**One prefill with `DFLASH_DS4_TIMING=1` on the pre-fix build settles the arena
claim** (`64ae212`) in a way no microbenchmark can: if `attn_build_us` is a small
fraction of `attn_compute_us`, my 917 ms/chunk allocator measurement does not
transfer and the change is not worth its risk. If it dominates, it does.

Ordering rule this suggests for anything after this document: read the telemetry
first, pick the phase that actually costs, then audit that code. Everything
above was found the other way round, which is why two of the four findings had
to be discarded after measurement (`moe_hybrid` disabled, the blanket mask
hoist wrong).
