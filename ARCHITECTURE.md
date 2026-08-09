# Ember — architecture

Ember is a from-scratch inference engine for **DeepSeek-V4-Flash on AMD Strix
Halo (gfx1151)**. It combines two existing engine lineages:

- **ds4-server (DwarfStar)** — battle-tested *server architecture*: robust SSE
  streaming, tokenizer/UTF-8 handling, tool-calling, reasoning separation, KV
  cache semantics and request lifecycle.
- **lucebox** — the *performance kernels*: ROCMFP2/ROCMFP4 mixed-precision quant,
  rocWMMA flash prefill, DSpark speculative decode, and KV snapshot/restore.
  HIP graph replay is disabled on gfx1151 because it regressed measured
  throughput; the rationale and numbers live in `engine/CMakeLists.txt`.

**Ember = ds4's server architecture, rewritten clean in C, driving lucebox's
tuned kernels.** We keep the 2× decode speed and stop reinventing ds4's server
robustness one bug at a time.

## The one load-bearing decision: reuse the kernels, rewrite the server

The GPU kernels (attention, 256-expert MoE, sparse indexer, hyper-connections,
the ROCMFP quant decode and DSpark verify) are the *entire*
performance advantage and represent person-years of gfx1151-specific tuning.
**Rewriting them from scratch would throw away the win and start slower.** So:

- **Reused through a maintained vendored fork** (via a stable C ABI,
  `src/backend/ember_backend.h`): the lucebox `dflash` DeepSeek4 forward pass —
  load, prefill, decode,
  snapshot_save/restore, DSpark. Linked as a C++ static lib behind an
  `extern "C"` shim. Ember's C server never sees ggml/HIP directly.
- **Rewritten fresh in C** (this repo): everything above the forward pass — the
  parts where ds4's architecture is superior and lucebox is weak.

This is a *server rewrite with a kernel bridge*, not a kernel rewrite. Over time
the backend can be progressively reimplemented natively behind the same ABI
without touching the server.

## Layers (bottom to top)

```
  ┌─────────────────────────────────────────────────────────────┐
  │ src/server/   http.c  sse.c  chat_api.c  tool_loop_guard.c     │  fresh C
  │   HTTP/1.1 · SSE streaming · OpenAI chat completions ·        │  (this repo)
  │   graceful drain · scheduling · agent-loop circuit breaker     │
  ├─────────────────────────────────────────────────────────────┤
  │ src/model/    chat_template.c  tool_parser.c  tool_memory.c    │  fresh C
  │   DSML render/replay · DSML/short/ascii tool parse ·           │  (this repo)
  │   reasoning split · thinking-budget force-close               │
  ├─────────────────────────────────────────────────────────────┤
  │ src/model/    kv_cache.c  (prefix + anchor + disk layout)     │  fresh C
  ├─────────────────────────────────────────────────────────────┤
  │ src/backend/  ember_backend.h  (stable C ABI)                 │  bridge
  │   ── extern "C" shim ──                                       │
  │   engine/  vendored ggml + DeepSeek4  (ROCMFP, DSpark)      │  reused
  └─────────────────────────────────────────────────────────────┘
```

## Why ds4's SSE model, not lucebox's

Lucebox streams token-by-token through an incremental `window_` state machine
with a fixed holdback. It broke five different ways this month (partial tool
markers, split `</think>`, split emoji, prefill silence, blind holdback), each
fixed by re-deriving what ds4 already does. Ds4 instead **keeps the full
accumulated output and re-splits it every update** (`split_reasoning_content`,
`emit_pos` diffing, `utf8_stream_safe_len`, `text_stream_safe_limit`). A marker
or codepoint split across *any* number of tokens is always re-findable. Ember's
`src/server/sse.c` is built on that model from line one — correct by
construction, not by accumulated patches.

## Parity roadmap

1. **Streaming and OpenAI chat** — implemented, including buffer-and-resplit
   SSE, bounded overload handling, disconnect cancellation, and graceful drain.
2. **Chat template + prompt path** — implemented. Tokenization is NOT
   rewritten: this model's pre-tokenizer is a custom `joyai-llm` variant that
   must be byte-exact or the prompt is silently corrupted, and lucebox already
   ships that proven tokenizer. It is exposed across the backend ABI as
   `encode`/`decode` — a model-coupled piece we reuse, like the kernels.
3. **Backend and tool continuation** — implemented through the vendored dflash
   bridge, exact-token tool replay, post-call KV, and disk prefix persistence.
4. **Responses and Anthropic adapters** — implemented, including token-live
   native reasoning, text, tool, error, and terminal event lifecycles driven by
   the shared buffer-and-resplit parser.
5. **Multi-session continuous batching** — implemented end-to-end behind
   `--batch-sessions`. DeepSeek keeps KV/cache, logits, sampler/RNG,
   HC/compressor state, decode history, and callbacks per resident session; one
   engine coordinator owns scheduling and execution. The HTTP dispatcher admits
   concurrent requests, prefix snapshots use reservation/read leases, and
   `/status` publishes scheduler counters. Decode rows currently use the
   executor's correctness-first serial target fallback; a ragged native
   multi-cache target graph and session-local DSpark are the remaining
   throughput optimizations. See
   [docs/continuous-batching.md](docs/continuous-batching.md).

## Non-goals (for now)
- Metal / multi-GPU / tensor-parallel (ds4 has these; we run one gfx1151 box).
- GLM support.
- Rewriting the HIP kernels natively (reused via ABI; revisit later).
