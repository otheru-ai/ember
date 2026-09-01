# Ember — architecture

Ember is Otheru's from-scratch inference engine for **DeepSeek-V4-Flash on AMD
Strix Halo (gfx1151)**. It fuses two existing engines we run in production:

- **ds4-server (DwarfStar)** — battle-tested *server architecture*: robust SSE
  streaming, tokenizer/UTF-8 handling, tool-calling, reasoning separation, KV
  cache semantics, request lifecycle. Slower base decode (~14 t/s) on our box.
- **lucebox** — the *performance kernels*: ROCMFP2/ROCMFP4 mixed-precision quant,
  graphs-ON HIP decode, rocWMMA flashprefill, DSpark speculative decode, KV
  snapshot/restore. Faster base decode (~25 t/s, ~33 t/s median with DSpark),
  but a rough, repeatedly-patched server layer.

**Ember = ds4's server architecture, rewritten clean in C, driving lucebox's
tuned kernels.** We keep the 2× decode speed and stop reinventing ds4's server
robustness one bug at a time.

## The one load-bearing decision: reuse the kernels, rewrite the server

The GPU kernels (attention, 256-expert MoE, sparse indexer, hyper-connections,
the ROCMFP quant decode, HIP graph capture, DSpark verify) are the *entire*
performance advantage and represent person-years of gfx1151-specific tuning.
**Rewriting them from scratch would throw away the win and start slower.** So:

- **Reused as-is** (via a stable C ABI, `src/backend/ember_backend.h`): the
  lucebox `dflash` DeepSeek4 forward pass — load, prefill, decode,
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
  │ src/server/   http.c  sse.c  chat_api.c  queue.c             │  fresh C
  │   HTTP/1.1 · SSE streaming (ds4 buffer-and-resplit) ·         │  (this repo)
  │   OpenAI/Anthropic/Responses shapes · cancel-on-disconnect    │
  ├─────────────────────────────────────────────────────────────┤
  │ src/model/    tokenizer.c  chat_template.c  tool_parser.c     │  fresh C
  │   gpt2 BPE · DSML render · DSML/short/ascii tool parse ·      │  (this repo)
  │   reasoning split · thinking-budget force-close               │
  ├─────────────────────────────────────────────────────────────┤
  │ src/model/    kv_cache.c  (prefix + anchor + disk layout)     │  fresh C
  ├─────────────────────────────────────────────────────────────┤
  │ src/backend/  ember_backend.h  (stable C ABI)                 │  bridge
  │   ── extern "C" shim ──                                       │
  │   lucebox libdflash_common.a  (ROCMFP kernels, graphs, DSpark)│  reused
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

## Milestones (feature-parity target = ds4-server)

1. **Scaffold + streaming core** — repo, build, `sse.c` (ds4 model) + tests. ← start
2. **HTTP server + OpenAI chat API** — request lifecycle, single-slot queue with
   cancel-on-disconnect (a real gap in *both* ds4 and lucebox).
3. **Chat template + prompt path** — DSML render (done). Tokenization is NOT
   rewritten: this model's pre-tokenizer is a custom `joyai-llm` variant that
   must be byte-exact or the prompt is silently corrupted, and lucebox already
   ships that proven tokenizer. It is exposed across the backend ABI as
   `encode`/`decode` — a model-coupled piece we reuse, like the kernels.
4. **Backend ABI + lucebox bridge** — link the tuned forward pass AND its
   tokenizer; first real token out of Ember end-to-end. ← next major phase
   (needs the HIP build container).
5. **Tool-calling + KV cache + disk persistence** — DSML multi-spelling parse,
   anchor checkpoints, longest-prefix match, layout-fingerprinted disk cache.
6. **Parity pass** — model card / effort tiers, /status, metrics, the
   QA_BEFORE_RELEASES checklist ds4 ships.

This is a multi-session build. Each milestone lands as compiling, tested C.

## Non-goals (for now)
- Metal / multi-GPU / tensor-parallel (ds4 has these; we run one gfx1151 box).
- GLM support.
- Rewriting the HIP kernels natively (reused via ABI; revisit later).
