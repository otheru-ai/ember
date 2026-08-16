# Continuous batching

Ember's original backend contract runs one request from prefill through its last
decoded token while owning one live DeepSeek KV cache. That shape cannot safely
be made concurrent at the HTTP layer: interleaving requests would overwrite the
same KV, sampler, logits, and speculative-decoder state.

The batching work therefore starts inside the engine.

## Implemented path

`engine/dflash/common/continuous_batch_scheduler.*` implements the resident
session lifecycle and produces one serialized engine submission at a time:

- bounded admission with generation-stamped session IDs;
- round-robin, quantum-limited prefill;
- decode coalescing, followed by one batch containing every ready session;
- an optional prefill quantum alongside a decode batch;
- cancellation while queued or in flight;
- terminal slot retention and explicit release;
- submission IDs that reject late or duplicate completions;
- instantaneous and lifetime counters for batch size, scheduled/completed work,
  coalescing, cancellation, and failure.

The default policy mirrors Dwarfstar's server:

- idle prefill quantum: 2048 tokens;
- mixed prefill quantum: 128 tokens;
- decode coalescing window: 2000 microseconds.

The mixed quantum applies whenever any generation is active, not only when a
decode row happens to be ready in the current scheduling pass. This prevents a
large prefill chunk from delaying a generation that becomes decode-ready just
after the plan is issued.

`engine/dflash/common/continuous_batch_executor.*` is the scheduler-to-engine
transaction boundary. It:

- executes native mixed prefill/decode submissions when a backend advertises
  them;
- otherwise uses a correctness-first fallback: prefill, then one decode batch;
- accepts decode results in any order but requires every scheduled session
  exactly once;
- rejects zero/oversized successful prefill progress;
- converts malformed results and exceptions into terminal scheduler failures
  so no submission remains wedged in flight;
- records native/fallback use, backend failures, malformed results, aborted
  components, and completion rejections.

`engine/dflash/common/resident_batch_coordinator.*` binds that transaction layer
to a model-coupled resident backend. `DeepSeek4Backend` now owns a session object
for each admitted request containing:

- an independent KV/HC/compressor cache;
- last logits and the absolute prefill/decode frontier;
- sampler configuration, RNG, penalty history, and force-close state;
- generated and pending tokens;
- streaming/keepalive callbacks, cancellation, errors, and timing;
- inline-snapshot state.

Model weights and the engine submission thread remain shared. Snapshot restore
copies a stable parked prefix into the resident cache before its read lease is
released. Released sessions return their cache allocation to a warm pool, so
slot reuse resets resident state without repeatedly allocating the full context
buffer. DSpark is disabled for resident sessions because its drafter and rolling
verifier feature state are not yet session-isolated.

The real C backend bridge owns a coordinator thread. Concurrent HTTP dispatchers
enqueue generation calls, then block on their individual completions while that
thread admits sessions, pumps plans, and runs snapshot/disk control operations.
Terminal sessions retain a short lease so the originating dispatcher can park a
post-tool-call frontier; explicit generation release then frees the resident
cache and returns scheduler capacity.

The server enables this path with:

```text
--batch-sessions N
```

`N=1` preserves the original one-worker monolithic path. Values `2..64` create
that many HTTP dispatchers and resident engine slots. Prefix-cache bookkeeping
is mutex-protected; write targets are reserved and restore sources are pinned so
neither concurrent disk staging nor LRU eviction can alias an in-flight request.
The Dwarfstar-compatible `DS4_SERVER_PREFILL_QUANTUM`,
`DS4_SERVER_MIXED_PREFILL_QUANTUM`, and
`DS4_SERVER_DECODE_COALESCE_US` environment variables tune the three policy
defaults above.

`GET /status` reports `continuous_batching` with capacity, pending/resident
counts, current scheduler states, admission/release/submission totals, decode
batch/row counts, prefill tokens, mixed plans, coalesce waits, backend failures,
and the largest planned decode batch.

## Validation

GPU-free tests cover scheduler state-space invariants, executor result
validation, coordinator lifecycle, prefix read/write leases, and a live server
test that observes two overlapping HTTP generations. DeepSeek and bridge
translation units also receive a host C++ syntax build when ROCm is unavailable.
On the ROCm host, `--validate-prompt PATH --batch-sessions 2` runs two resident
sessions through the coordinator and requires both token streams to match the
serial autoregressive baseline; its JSON `batch` object records the rows and
tokens compared. Repeating that oracle across ragged prompts, cancellation, slot
reuse, and mixed plans remains the hardware acceptance gate.

The first gfx1151 acceptance run passed on 2026-08-09 using exact prefill and
two resident sessions. A 3,926-token prompt produced 32 baseline tokens and 64
resident-session tokens; both resident streams and the disk snapshot round trip
were token-for-token identical to serial autoregressive generation. This proves
the engine coordinator path for that shape. The GPU-free server test separately
exercises overlapping HTTP requests, while broader ragged/cancellation coverage
remains part of the release hardware gate.

Server bookkeeping does not rely on a single `run_chat()` thread. With
`--batch-sessions N`, N persistent generation workers may overlap and bypass
`gen_lock`. KV, continuation, and tool-memory access is serialized by
`state_lock`; the DSML tracker is request-local. Tool-memory getters return
borrowed interior pointers, so their callers must retain `state_lock` until the
pointer is no longer used.

## Measured HIP GEMM batch limit

The gfx1151 sweep added as `--validate-gemm-batch N` was run through batch count
64 on ROCm 7.14. Three representative decode projections were bit-identical to
the one-row baseline at every count, with no HIP fault. This removes the need to
copy Dwarfstar's CUDA-specific four-row ceiling, but it does not by itself prove
that the full attention, MoE, and ROCmFP decode path can combine independent
sessions. The differential validator remains authoritative.

The sweep uses deterministic synthetic FP16 inputs, `HIPBLAS_COMPUTE_32F`, and
`HIPBLAS_GEMM_DEFAULT`. It needs no model weights and can be rerun with:

```bash
./build-rocm/ember-dflash --validate-gemm-batch 64
```

## Remaining performance work

The scheduler already forms one logical decode batch containing every ready
session, but DeepSeek's current `decode_batch()` evaluates those rows one at a
time. This matches Dwarfstar's correctness fallback and provides concurrency,
fairness, and bounded latency without pretending independent caches are a dense
token sequence. Dwarfstar's CUDA fast path records independent session graphs
into one command batch; Ember's remaining throughput work is the HIP equivalent
(or a ragged target step whose rows carry independent cache pointers and
absolute positions) while sharing as many embedding, attention, HC, MoE, and
output launches as the backend permits.

After that graph is validated, DSpark needs the same per-session treatment for
its draft cache, feature window, verifier rollback state, and profitability
scheduler before speculative resident batching can be enabled. The XDNA2
prototype now has an asynchronous whole-draft-forward provider contract with
two-job session-isolation coverage. The monolithic path collects it
immediately; the resident implementation must retain one proposal job per
session and schedule GPU verify for a different ready session before waiting.
