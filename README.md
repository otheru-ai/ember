# Ember

Otheru's from-scratch **C** inference server for **DeepSeek-V4-Flash on AMD Strix
Halo (gfx1151)**. It fuses [antirez/ds4](https://github.com/antirez/ds4)'s
battle-tested server architecture (rewritten clean in C) with lucebox's tuned
HIP kernels (ROCMFP quant, graphs-ON decode, DSpark speculative decode, KV
snapshotting) via a stable C ABI. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Status — production

Ember is **live in production** on `:8000`, serving the Hermes agent (it replaced
the lucebox `dflash_server`). The real forward pass runs through
`backend_dflash.cc` — an `extern "C"` shim over lucebox's `libdflash_common.a`;
nothing above the backend ABI changed from the stub bring-up.

Production runs the **full-ROCMFP affine model** (below) and **meets the
Strix-Halo reference benchmarks**:

| metric | target | ember |
|---|---|---|
| sparse prefill | 245 tok/s | **248–253 tok/s** |
| decode (DSpark) | 32 tok/s | **32 tok/s** clean-gen (~28–30 on structured/reasoning) |

Decode is a pure function of the DSpark accept rate (same backend/model/config →
same speed). ds4 server-behavior parity was driven by a multi-agent audit
([docs/ds4-fidelity-audit.md](docs/ds4-fidelity-audit.md)); every actionable
finding is implemented and verified live.

## The model — full-ROCMFP affine fp2

Every MoE expert — **including the sensitive down-projection** — is carried at
**2.5 bpw** using an *affine* fp2 format: `value = code·scale − offset`, uniform
codes `{0,1,2,3}` with one UE4M3 `scale` + one UE4M3 `offset` per 32-weight block
(Q2_K's scale-and-min scheme in ROCMFP's 10-byte block). The stock symmetric fp2
codebook is too coarse for the down-projection (incoherent output); the affine
form matches Q2_K quantization error (NRMSE **0.292 vs 0.287**) at ROCMFP kernel
speed. Matching **mvvq** (decode) and **mmq** (WMMA prefill/verify) kernels are
implemented for gfx1151.

Result: a coherent **85.3 GiB / 2.58 bpw** model — **14 GiB smaller in RAM** than
the prior Q3-down build, which is what closed the memory gap on a 128 GB box.

**Weights:** [otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF](https://huggingface.co/otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF)
· recipe: experts `Q2_0_ROCMFP2` (affine), dense/attn `Q4_0_ROCMFP4_FAST`,
`token_embd` Q6_K, imatrix-calibrated.

## Features

- **HTTP/1.1** — threaded (thread-per-connection), single-slot generation lock
  (health/status stay responsive during a generation); socket send/recv timeouts
  so a stuck client can't wedge the slot; 64 MB body cap; query-string strip.
  Endpoints: `/health`, `/v1/models`, `/status`, `/v1/chat/completions`.
- **SSE streaming** — ds4 buffer-and-resplit: split emoji, `</think>`, and tool
  markers are correct by construction. Initial `role` chunk; `event: error` on
  backend failure; `usage` gated on `stream_options.include_usage`.
- **Chat template** — byte-exact DSML render (BOS, `<｜User｜>`/`<｜Assistant｜>`,
  tool preamble, `<think>` gating + MAX-effort prefix, assistant-turn collapse,
  `developer` role as system, `<tool_result>` sentinel-escaping, assistant
  tool-call replay).
- **Tool calling** — parses DSML (full / short / plain-XML) **and the model's
  native `ds_engine_tool_use` format** (the leaked-tool-call fix); DSML entity
  unescape; conservative truncated-block repair; structured `tool_calls` in the
  non-stream response *and* **incremental streaming deltas** (header + per-argument
  fragments); tool-call end-marker early-stop.
- **Sampling** — full surface threaded to the backend: `temperature`, `top_p`,
  `top_k`, **`min_p`** (added to lucebox's `SamplerCfg`), `seed`,
  `repetition_penalty` (+ window), `frequency_penalty`, `presence_penalty`.
  Greedy (temp 0) takes the DSpark fast path; `min_p` is correctly a no-op there.
- **Thinking / reasoning** — `reasoning_effort` → NONE/HIGH/MAX; thinking ON by
  default (ds4); Level-2 force-close reserves a reply budget and injects the
  terminator so the model always emits a visible answer (the reasoning-leak fix).
- **Stop sequences** — stream-safe holdback + truncation.
- **KV prefix cache** — longest-token-prefix match, anchor + turn-boundary cuts,
  keep-ancestors LRU; **cross-restart disk persistence**; `cached_tokens` reported.
- **Background-gating** — `ember_background:true` requests defer under user load
  (foreground stamps activity; background work waits for an idle window), so
  Hermes memory consolidation never contends with a live user turn.
- **Usage extensions** — `timings` (prefill/decode tok/s) + `accept_rate`, and a
  `backend` object (`spec_ran`, `degenerate`, `forced_close`, `empty`).
- **Robustness** — `context_length_exceeded` → 400; JSON-escaped output
  everywhere; validated surrogate pairs; balanced allocations (leak-audited).

## Configuration (flags)

```
-m <path>                 model gguf (required)
--model-name <id>         advertised id (default deepseek-v4-flash)
--model-card <path>       sampling defaults + thinking budget + terminator hint
--kv-cache-dir <path>     enable cross-restart disk KV cache
--ds4-expert-top-k <n>    routed experts (default 4 — matches the DSpark draft)
--default-temperature <t> temperature when a request omits it (code default 1.0;
                          production runs 0 so temp-omitting clients hit DSpark)
--prefix-cache-slots <n>  in-memory KV snapshot slots (default 32; production 8 —
                          each holds a full-KV buffer, so this bounds memory)
--max-ctx <n>             KV context (default 65536; production 49152)
--port <n>
```

Runtime env: `DFLASH_DS4_SPEC=1` + `DFLASH_DS4_DRAFT=<draft.gguf>` enable DSpark;
`EMBER_BG_IDLE_SECS` / `EMBER_BG_MAX_WAIT_SECS` tune background-gating.

## Build & run

The real backend must be built in the ROCm/HIP container (it links
`libdflash_common.a`); the pure-C server + test gauntlet build with plain gcc.

```
# real backend (inside lucebox-rocm:7.2.4)
bash scripts/build_dflash.sh              # -> ./ember-dflash

# GPU-free stub server + tests (any host)
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

Production is a systemd **user** unit (`ember-server.service`) with a RAM
drain-guard and `Conflicts=` lucebox-server; rollback to lucebox is one script.

## Memory

The affine model is **85.3 GiB** on a **125 GiB** box (~89 GB resident) — roughly
**36 GiB of headroom**, enough to absorb concurrent large-context load without the
swap-thrash the prior ~96 GiB (Q3-down) build hit. Combined with background-gating
and ember's lean footprint (8 vs lucebox's 32 snapshot slots; ~28 MB process RSS;
no leaks), memory is no longer the operational limiter it once was. The remaining
pressure is transient (e.g. streaming the model file off disk while serving it).
