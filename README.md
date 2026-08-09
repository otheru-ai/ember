# Ember

<p align="center">
  <img src="docs/assets/ember-logo.png" alt="Ember logo" width="240">
</p>

<p align="center">
  <strong>DeepSeek-V4-Flash inference for AMD Strix Halo</strong>
</p>

A from-scratch **C** inference server for **DeepSeek-V4-Flash on AMD Strix Halo
(gfx1151)**. Ember combines [antirez/ds4](https://github.com/antirez/ds4)'s
battle-tested server architecture (rewritten clean in C) with lucebox's tuned
HIP kernels (ROCMFP quant, DSpark speculative decode, KV
snapshotting) via a stable C ABI. See [ARCHITECTURE.md](ARCHITECTURE.md).

HIP graph replay is deliberately disabled on gfx1151 because measured capture
churn made it slower; see `engine/CMakeLists.txt` for the benchmark and gate.

## Docker quick start

On a native Linux Strix Halo host with Docker Engine, Docker Compose v2,
`/dev/kfd`, and `/dev/dri`, run this from the repository root:

```bash
scripts/preflight.sh && docker compose up --build -d
```

That command validates the host, builds the minimal `release` image, and starts
Ember on `http://127.0.0.1:8080`. On the first start, the container downloads
the release-pinned
[DeepSeek-V4-Flash Strix Halo GGUF](https://huggingface.co/otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF)
(approximately 85 GiB), resumes interrupted downloads, and verifies its SHA-256
before serving it. Model files live in `./models` and the persistent KV cache in
`./cache`; neither is baked into the image.

Once the container reports healthy, verify the API:

```bash
scripts/smoke_test.sh --generate
```

Later starts need only `docker compose up -d`. For a shell with the complete
ROCm compiler, headers, source tree, and debugging toolchain, use the separate
development container:

```bash
docker compose --profile dev run --rm ember-dev
```

See [Build & run](#build--run) for manual image builds and
[docs/operations.md](docs/operations.md) for hardware requirements, updates,
security, and troubleshooting.

## Status

The real forward pass runs through
`backend_dflash.cc` — an `extern "C"` shim over the vendored engine in `engine/`;
nothing above the backend ABI changed from the stub bring-up.

The published **full-ROCMFP affine model** (below) meets the
Strix-Halo reference benchmarks:

| metric | target | ember |
|---|---|---|
| sparse prefill | 245 tok/s | **248–253 tok/s** |
| decode (DSpark) | 32 tok/s | **32 tok/s** clean-gen (~28–30 on structured/reasoning) |

Decode is a pure function of the DSpark accept rate (same backend/model/config →
same speed). Compatibility behavior is pinned by the GPU-free regression suite
and the target-hardware differential validator.

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

- **HTTP/1.1** — threaded (thread-per-connection), with one persistent dispatcher
  on the legacy path or a bounded resident-session pool under
  `--batch-sessions`; health/status stay responsive during generation. Socket
  send/recv timeouts keep a stuck client from wedging a slot; 64 MB body cap;
  query-string strip.
  Endpoints: `/health`, `/status`, `/v1/models`, `/v1/models/{id}`,
  `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, and
  `/v1/completions`; optional browser CORS via `--cors`. Setup recipes for
  Claude Code, Codex, OpenCode, pi, and OMP are in
  [docs/client-compatibility.md](docs/client-compatibility.md).
- **API adapters** — OpenAI Chat, Responses, legacy Completions, and Anthropic
  Messages request/response shapes share one inference pipeline. Responses and
  Anthropic streams use the same buffer-and-resplit parser as Chat. Tool-free
  turns emit token-live reasoning where the target protocol can represent it
  safely, plus token-live text; tool-bearing turns remain
  active with SSE keepalives while output is validation-gated, so live events
  and terminal objects cannot disagree. Malformed calls never trigger a hidden
  second assistant inside an open stream; they end in one typed model-output
  error, matching ds4's streaming turn boundary. Responses
  preserves `instructions`, hosted-tool replay, and `tool_search` discoveries
  (including namespace flattening). `tool_choice` supports auto/none/required,
  named functions, Responses `allowed_tools`, and Anthropic any/tool forms;
  parallel-call disabling is prompt-steered and enforced before emission.
  Unsupported item types fail explicitly instead of changing semantics.
- **SSE streaming** — ds4 buffer-and-resplit: split emoji, `</think>`, and tool
  markers are correct by construction. Tool-call frames are held until the full
  block passes executable validation. Chat uses an initial `role` chunk and
  `event: error`; Responses emits its top-level sequenced `type:error` event,
  and Anthropic uses its native error envelope. Chat `usage` remains gated on
  `stream_options.include_usage`.
- **Chat template** — byte-exact DSML render (BOS, `<｜User｜>`/`<｜Assistant｜>`,
  tool preamble, `<think>` gating + MAX-effort prefix, assistant-turn collapse,
  `developer` role as system, `<tool_result>` sentinel-escaping, assistant
  tool-call replay).
- **Tool calling** — parses DSML (full / short / ASCII-degraded / plain-XML)
  **and the model's native `ds_engine_tool_use` format**; DSML entity unescape;
  structured `tool_calls` in atomic responses and validation-gated streaming
  deltas; tool-call end-marker early-stop. The executable boundary rejects
  incomplete repair, nested/mixed/trailing protocol markup, malformed raw JSON,
  duplicate argument keys, unknown or disallowed tool names, forbidden parallel
  calls, and non-object arguments. Arguments are recursively checked against
  the advertised JSON Schema (types, enums/constants, numeric/string/array/object
  constraints, combinators, and local `$ref`); strict schemas fail closed on
  unsupported keywords. Atomic
  requests get one model-visible correction attempt; if it also fails, Ember
  returns typed `422 invalid_tool_call`. Streaming requests do not splice a
  hidden replacement turn into an already-open response: the first violation
  emits one `model_output_error` with `retry_exhausted:false` and
  `partial_tool_call:false`, without exposing a partial call to the harness.
  Only exact sampled **tool-call blocks** and their token IDs are persisted beside
  the disk KV cache—reasoning/content are always rendered canonically—along
  with model-scoped call-ID/`previous_response_id` frontier bindings. Call-id
  storage is dynamically sized, so parallel calls beyond the former 16-call
  ceiling retain their replay identity.
  ID-only post-tool continuations therefore survive restarts without allowing a
  tool id to replay a hidden or force-closed assistant trajectory.
- **Agent progress and recovery** — long streaming prefills emit `: prefill`
  keepalives, responses report `cached_tokens`/`restored_prefix`, and the
  visible-output cycle guard reports `usage.backend.degenerate:true`. Ember does
  not impose a fixed tool-round ceiling: long-running agents may legitimately
  repeat polling or iterative actions, so cross-turn progress policy remains the
  harness's responsibility. Instead, Ember derives stateless trailing-history
  diagnostics for repeated call signatures, repeated call-and-result rounds,
  and tool rounds that produce no novel `(tool name, exact result)` effect.
  Reports retain `finish_reason:"tool_calls"` / `stop_reason:"tool_use"` and
  appear in logs, response metadata, and `/status`. Empty visible turns,
  backend-flagged degenerate turns, and tool markup delivered as text are also
  counted. Optional `--auto-answer-after-loop` can suppress tools for one turn
  after repeated calls and inject a plain-prose recovery instruction; it is
  behavior-changing, stateless, never overrides required tool choice, and is
  disabled by default.
- **Sampling** — full surface threaded to the backend: `temperature`, `top_p`,
  `top_k`, **`min_p`** (added to lucebox's `SamplerCfg`), `seed`,
  `repetition_penalty` (+ window), `frequency_penalty`, `presence_penalty`.
  Omitted temperature/top-p/top-k/min-p/repetition/presence values resolve from
  the model card; explicit request values (including zero) win. The shipped
  DeepSeek card uses temperature 0.6, top-p 0.95, top-k 40, min-p 0,
  repetition penalty 1, and presence penalty 0. `/status` reports the effective
  defaults so production policy is observable without reading the unit file.
  Greedy (temp 0) takes the DSpark fast path when configured; a worker-scoped
  profitability scheduler temporarily falls back to target-only decoding when
  drafting costs more than it saves and hands short tails back to autoregressive
  decode. `min_p` is correctly a no-op on the greedy path.
- **Thinking / reasoning** — `reasoning_effort` → NONE/HIGH/MAX; thinking ON by
  default (ds4). `reasoning_budget_tokens` (plus the vLLM-compatible
  `thinking_token_budget` alias) can impose an explicit phase-1 cap. Level-2
  force-close reserves a reply budget and injects the terminator so the model
  always emits a visible answer (the reasoning-leak fix).
  A natural `</think>` permanently disarms that hook for the turn, so it cannot
  inject a second close directive into an answer that is already in progress.
  If the model starts a tool stanza before closing thinking, Ember performs one
  bounded same-assistant `</think>\n\n` continuation (ds4 parity) without
  repeating the system prompt or opening a synthetic tool-result turn. An
  opener whose preceding token suffix exactly overlaps the rendered prompt is
  treated as quoted/echoed protocol text and is never promoted by this path.
  The progress guard stops sustained visible cycles (four copies / 256 repeated
  tokens), substantially longer hidden-reasoning cycles (eight copies / 1024
  tokens), and exact 512-token prompt echoes. These are typed
  `model_output_error` failures (`repetition_detected`,
  `reasoning_cycle_detected`, or `prompt_echo_detected`), not ordinary
  `finish_reason:"length"` completions: an agent harness must replan rather than
  auto-continue the same failed trajectory. Watchdog output is never fed into
  malformed-tool recovery merely because echoed prompt text contains DSML.
- **Bounded omitted output limits** — when a request omits `max_tokens`,
  `max_completion_tokens`, or `max_output_tokens`, Ember uses the model card's
  `max_tokens` (16,384 for DeepSeek-V4-Flash) rather than granting the entire
  remaining context. Explicit limits remain available for intentional long
  generations and are capped only by context capacity.
- **Stop sequences** — stream-safe holdback + truncation.
- **KV prefix cache** — longest-token-prefix match, anchor + recent
  turn-boundary cuts (the scan retains the newest boundaries even after long
  histories exceed its scratch buffer),
  keep-ancestors LRU; **cross-restart disk persistence**; `cached_tokens` reported.
  Concurrent restores are pinned and snapshot targets are reserved, so two
  sessions cannot evict or overwrite each other's physical KV slots.
  Snapshots store only live compressed-KV rows. Disk format v3 writes append-only
  compressed-KV suffixes against a bounded parent chain, protects parent files
  during eviction, and uses fsync + atomic rename; existing v1 snapshots remain
  readable.
- **Background-gating** — `ember_background:true` requests defer under user load
  (foreground stamps activity; background work waits for an idle window), so
  background agent maintenance never contends with a live user turn.
- **Continuous batching** — opt-in engine-owned resident sessions with isolated
  DeepSeek KV, logits, sampler/RNG, HC/compressor frontier, and decode history.
  The scheduler coalesces decode-ready sessions, advances long prefills fairly
  in quanta, mixes a small prefill quantum beside active decode, and propagates
  cancellation without returning the resident slot before post-generation
  snapshots are complete. Cache allocations return to a warm pool on slot
  release. `/status` exposes live and lifetime scheduler metrics.
- **Usage extensions** — `timings` (evaluated prefill tokens, prefill/decode
  tok/s) + `accept_rate`, and a `backend` object (`spec_ran`, `degenerate`,
  `forced_close`, `empty`, `prefill_mode`, `prefill_reason`). Streaming clients
  receive the same performance attribution when `include_usage` is enabled.
- **Robustness** — `context_length_exceeded` → 400; JSON-escaped output
  everywhere; validated surrogate pairs; balanced allocations (leak-audited).

## Context compaction

Ported from Dwarfstar's agent-side compaction (`ds4_agent.c:8010-8292`) into
`src/server/compaction.c`, opt-in via `--auto-compact`. When a prompt reaches 85%
of context — or when free space drops below `min(8192, ctx/8)` — the server asks
the model for a durable task-state summary and rebuilds the history as
`system + summary + verbatim tail`, with the tail budgeted at `ctx/10` (capped at
50k) and snapped to a user-turn boundary. Because the rebuild is assembled within
a fixed budget rather than produced by shrinking the old one, the result is
bounded by construction: it cannot fail to shrink, whatever the summary's quality.

The compaction turn runs with thinking OFF, stops if the model starts emitting
DSML, and never snapshots its KV — the prompt is private and must not become a
cache entry a later real turn could restore from. Any failure leaves the request
untouched, so the normal context guard still applies.

It runs ahead of the context guard, so a history that would otherwise be rejected
with `context_length_exceeded` is served instead. Skipped for `/v1/completions`
(no message structure) and for tool-output continuations (their whole point is
exact token identity against a KV frontier).

Two behaviours differ from ds4, both because a server is stateless where an agent
is not:

- ds4 compacts once and keeps the shrunken transcript. Ember re-derives compaction
  each request, since the client re-sends full history.
- ds4's transcript is always under context, so it summarizes all of it. Ember can
  be handed a history that *already* exceeds context, which cannot be read in one
  pass; it then summarizes the most recent window that fits and reports the
  remainder as `dropped_unsummarized` rather than losing it silently.

Every compacted generation is reported in `usage.compaction` (both the JSON and
SSE paths) and logged to stderr. `usage.prompt_tokens` continues to report what
the client sent, not the compacted size.

## Remaining Dwarfstar parity work

- Multi-session serving is now end-to-end and opt-in. DeepSeek resident state,
  server admission, cancellation, snapshot leases, metrics, and concurrent HTTP
  dispatch are implemented. The current DeepSeek executor evaluates coalesced
  decode rows sequentially inside one engine-owned submission; the remaining
  throughput optimization is a ragged native target graph whose rows reference
  independent KV/cache state. DSpark is deliberately disabled for resident
  sessions until its draft/verifier frontier is isolated per session. See
  [docs/continuous-batching.md](docs/continuous-batching.md).

## Known API limitations

- Responses `conversation` objects and non-tool `previous_response_id` chains
  are rejected explicitly; persisted tool-output continuations are supported.
  Current Dwarfstar also rejects official durable conversation references, so
  this is API-surface work rather than a behavioral-parity gap.
- DeepSeek reasoning is omitted from Anthropic Messages output because Ember
  cannot produce Anthropic's cryptographic thinking signature. It deliberately
  does not emit an empty signature or label unsigned text as a `thinking` block.

## Configuration (flags)

```
-m <path>                 model GGUF (required)
-h, --help                print CLI help and exit
--port <n>                loopback port (default 8080)
--model-name <id>         advertised id (default deepseek-v4-flash)
--model-card <path>       sampling defaults + thinking budget + terminator hint
--cors                    allow browser cross-origin requests
--kv-cache-dir <path>     enable cross-restart disk KV cache
--kv-cache-mb <n>         disk KV cache budget in MiB (default 131072 / 128 GiB)
--ds4-expert-top-k <n>    routed experts (default 4 — measured production
                          override; model/drafter metadata use 6)
--default-temperature <t> override the model-card temperature (card/fallback 0.6)
--tool-loop-report <n> report after N repeated call signatures or identical
                       call+result rounds (default 8;
                       0 disables; reporting never stops a tool call)
--no-progress-report <n>
                       report after N tool rounds return no novel effect
                       (default 8; 0 disables)
--auto-answer-after-loop <n>
                       suppress tools for one turn after >N identical trailing
                       calls (default 0/off; never overrides required tools)
--prefix-cache-slots <n>  in-memory KV snapshot slots (default 8 —
                          each holds live compressed rows plus fixed rolling state)
--batch-sessions <n>      resident concurrent sessions (default 1; max 64).
                          Values >1 enable continuous batching.
--max-ctx <n>             KV context (default 65536; production 131072)
--auto-compact            ds4-style context compaction (default OFF). At 85% of
                          context, rebuild history as system + summary + verbatim
                          tail. See "Context compaction" below.
--validate-prompt <path>  run deterministic engine validation, print JSON, exit
--validate-tokens <n>     generated tokens for validation (default 32)
```

Combining `--validate-prompt` with `--batch-sessions 2` also runs two resident
sessions and requires both token streams to match the serial autoregressive
baseline; the JSON report includes a `batch` result.

## Quantization quality reports

`scripts/quant_quality_report.py` turns reference/quant continuation scores,
full-logit statistics, agentic replays, tensor error, and runtime measurements
into an auditable Markdown + JSON + CSV report with SVG plots. Reports keep
upstream-to-source changes separate from source-to-quant degradation, which is
essential for transformed sources such as abliterated checkpoints. See
[`docs/quant-quality-reports.md`](docs/quant-quality-reports.md) for the frozen
corpus workflow, behavioral suite, provenance requirements, and release gates.

Runtime env: `DFLASH_DS4_SPEC=1` + `DFLASH_DS4_DRAFT=<draft.gguf>` enable DSpark;
`DFLASH_DS4_SPEC_SCHEDULER=0` disables its adaptive profitability scheduler.
Compose users may set `EMBER_TOOL_LOOP_REPORT`, `EMBER_NO_PROGRESS_REPORT`, and
`EMBER_AUTO_ANSWER_AFTER_LOOP` for the corresponding server flags.
`--ds4-prefill exact` selects tokenwise reference prefill for quality evaluation;
`sparse` remains the faster default and may change generated tokens.
`dense` has a context ceiling of roughly 50k tokens (~53.5k at a 2048-token
prefill band, ~45.6k at 4096) and aborts above it. The D=512 flash kernel stages
one score per visible KV row in shared memory, and only the sparse path bounds
that span — its indexer publishes a negative keep count, which pins the kernel's
score stride at `n_swa + n_indexer_top_k` (128+512) no matter how long the
compressed span grows (`fattn.cu:1706`). Dense sets a zero keep count, so the
stride follows the compressed row count without limit and eventually exceeds the
64 KiB per-workgroup LDS budget; a DS4-marked flash node has no fallback and
aborts. `sparse` and `exact` are unaffected at any supported context, and
ratio-128 layers stay within budget past 131072.
Exact q=1 prefill skips the stateless vocabulary projection on intermediate
prompt tokens by default and still computes it at the final token and snapshot
boundaries. Set `DFLASH_DS4_EXACT_PREFILL_SKIP_LOGITS=0` for the former
every-token projection during diagnostics. `DFLASH_DS4_EXACT_PREFILL_CHUNK`
accepts 1–4, but values above 1 are experimental: the 0731 production model
changed final-logit fingerprints and ran slower at q=2/q=4, so production must
retain the default q=1.
`DFLASH_DS4_DECODE_FLASH=1` runs q=1 attention through the same exact D=512
flash kernel prefill uses, replacing the explicit F32 reduction (scores matmul,
softmax, transposed value matmul) that costs ~23.5% of decode. It applies only
to the fused decode graph's masked full-ring path, where the host-filled row
mask is the sole visibility authority — no other q=1 shape carries a mask, and
none may bake a position into the kernel, since one cached q=1 graph is reused
at every later position. Compressed rows stay dense (the padded span would
pollute a block mean), the inverse tail RoPE stays a graph node reading the
per-step position, and a compressed span too long for the kernel's shared-memory
budget falls back to the explicit path. Off by default: it changes decode
reduction order and therefore sampled tokens.
DSpark-eligible greedy requests keep the configured batched prefill for the
prefix, then switch to exact q=1 only for the final SWA feature-capture window.
This preserves the drafter's exact feature contract without reducing a long
prompt to one graph launch per token. The production verifier evaluates proposed
tokens through that same q=1 graph until the first disagreement. A rejected
token is never written, so target KV and compressor state need no speculative
rollback and greedy output remains target-authoritative.
`DFLASH_DS4_BATCH_VERIFY=1` selects the Dwarfstar-style throughput path: verify
the proposal in one q-wide target graph, retain fully accepted blocks, and roll
back/replay partial accepts through the ordinary q=1 graph. This is the path
used to reach the clean-generation decode target. By default each request must
first produce 48 consecutive fully accepted tokens under exact q=1 verification;
a partial batched block returns to exact mode and must requalify. Override that
gate with `DFLASH_DS4_BATCH_WARMUP_TOKENS` (`0` restores immediate batching).
Batched floating-point reductions can choose a different token when target
logits are nearly tied, so the q=1 verifier remains the default for strict token
reproducibility.
`DFLASH_DS4_APPROX_VERIFY=1` is retained as a compatibility alias.
Set `DFLASH_DS4_SPEC_DEBUG=1` to log the precise eligibility decision for
each request (`context`, `sampling`, `force_ar`, `short_budget`, or
`no_drafter`) without changing decode behavior.
Set `DFLASH_DS4_AGREEMENT_LOG=1` for an online draft/target agreement trace
(`ctx`, offered candidates, matched candidates, and a run summary). This is
diagnostic only and does not alter proposal or acceptance decisions.
The worker-scoped profitability gate uses an acceptance EWMA by default:
`DFLASH_DS4_SPEC_GATE_DISABLE_MILLI` / `DFLASH_DS4_SPEC_GATE_ENABLE_MILLI`
set its hysteresis thresholds (defaults 450/700), and
`DFLASH_DS4_SPEC_GATE_REPROBE_REQUESTS` controls periodic re-probes (default 3).
`DFLASH_DS4_SPEC_GATE_ALPHA_MILLI` / `DFLASH_DS4_SPEC_GATE_PROBE_ALPHA_MILLI`
weight the average (defaults 250 / 500). Every one of these counters advances
in *spec-eligible* requests, not wall requests: sampled and forced-AR turns
short-circuit before the gate, and in an agent workload most turns are
forced-AR tool continuations. Samples taken while gated off are therefore rare
and are weighted harder so recovery costs ~2 probes rather than 4-5.
The enable threshold is absolute, so a workload whose honest acceptance sits
inside the `[disable, enable)` band would be disabled by one transient dip and
never recover. `DFLASH_DS4_SPEC_GATE_RECOVER_PROBES` (default 6) bounds that:
after that many probes with the gate still off, recovery falls back to the
disable threshold, asking only whether speculation is still *unprofitable*
rather than whether it is excellent. Set 0 to require the full enable threshold
forever. A reprobe interval of 0 is clamped to 1 — left as 0 it would stop every
probe, so no sample would ever be taken and the gate could never clear; set
`DFLASH_DS4_SPEC=0` to turn speculation off instead.
Only an immediate tool-result continuation disables DSpark and uses target-only
autoregressive decode. It still uses the configured batched prefill policy;
decode strategy no longer implicitly selects q=1 exact prefill. Historical tool
messages elsewhere in a growing conversation do not poison later user turns.
Explicit reference validation continues to force exact prefill.
Prefill capture uses the same request eligibility gate as speculative decode:
over-ceiling contexts, sampled requests, short reply budgets, and forced-AR
turns do not attach capture hooks or retain stale DSpark features.
`EMBER_BG_IDLE_SECS` / `EMBER_BG_MAX_WAIT_SECS` tune background-gating (defaults:
5 / 60 seconds). `EMBER_IDLE_RECLAIM_SECS` controls how long the persistent
worker stays idle before releasing cached compute graphs (default 300 seconds;
`0` disables reclamation). Batched mode accepts Dwarfstar's scheduler knobs:
`DS4_SERVER_PREFILL_QUANTUM` (default 2048),
`DS4_SERVER_MIXED_PREFILL_QUANTUM` (128), and
`DS4_SERVER_DECODE_COALESCE_US` (2000).

Two off-by-default forensic traces diagnose output corruption without changing
the response: `EMBER_TRACE_TOKENS=1` logs every committed token id and its exact
decoded bytes, while `DFLASH_TRACE_SAMPLER=1` logs the selected token, RNG draw,
final post-filter weight, and top ten candidates. Production's `top_k=40`
configuration takes the CPU sampler path, so those reported weights are the
values used by its inverse-CDF draw.

## Build & run

The real backend must be built in the ROCm/HIP container (it compiles the
vendored ggml fork + DeepSeek4 engine under `engine/`); the server and GPU-free
test gauntlet build with a normal C/C++ toolchain.

### One-command container start

The condensed path is in [Docker quick start](#docker-quick-start). To keep the
container attached and see model-download and server logs directly, omit `-d`:

```bash
scripts/preflight.sh && docker compose up --build
```

The preflight catches missing devices, Docker access, incompatible GPUs, and
insufficient disk space before the expensive path begins. On first start,
Compose builds Ember and downloads the approximately 85 GiB
Strix Halo ROCMFPx GGUF from
[`otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF`](https://huggingface.co/otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF)
into `./models`. The download is pinned to the release-tested Hugging Face
revision and verified by SHA-256 before it is made visible to the server.
Downloads resume after interruption, and both model data and the KV cache
survive container replacement. Later starts are simply:

```bash
docker compose up -d
```

The API listens at `http://127.0.0.1:8080`; host networking deliberately keeps
Ember's loopback-only security default intact. Use `.env.example` only when you
need to select a different model directory, artifact, port, or sampler default.
The downloaded model is governed by the license shown on its Hugging Face model
card and is not redistributed inside the Ember image.

The default Compose service builds the `release` target. It contains the
stripped Ember server and only its recursive ROCm runtime-library closure; the
ROCm compiler, headers, source tree, build directory, and debugging utilities
remain in the separate `dev` target. To open the development container with the
repository mounted at `/workspace`:

```bash
docker compose --profile dev run --rm ember-dev
```

Check readiness with:

```bash
curl http://127.0.0.1:8080/health
```

Compose also reports this state through its container health check. Run
`scripts/smoke_test.sh` for health/model/status checks or add `--generate` for
a short inference check. See the [operations guide](docs/operations.md) for the
supported host contract, artifact verification, updating, security boundary,
and troubleshooting. The [upstream lessons review](docs/upstream-lessons.md)
records which release patterns Ember adopted or deliberately deferred.

Docker Engine, the ROCm-supported kernel/device stack, roughly 90 GiB for the
model download, and sufficient unified memory are prerequisites. Docker Desktop
is not supported because this deployment requires `/dev/kfd`, `/dev/dri`, and
Linux host networking.

### Manual build

Build the ROCm development image once, then let the repository script configure
and compile the real backend into the host checkout:

```
# from the repository root
docker build --target dev -f docker/Dockerfile -t ember-rocm:7.14-dev .
scripts/build.sh
```

Build only the deployable image with:

```bash
ember_version="$(tr -d '\n' < VERSION)"
docker build --target release -f docker/Dockerfile \
  --build-arg EMBER_VERSION="$ember_version" \
  -t "ember:$ember_version" .
```

To configure manually from inside an existing ROCm/HIP build environment:

```
cmake -S . -B build-rocm -DEMBER_ENGINE=ON -DEMBER_ENGINE_BACKEND=hip
cmake --build build-rocm --target ember-dflash -j
```

The GPU-free stub and test suite build on a normal host:

```
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

Start the real server, then make an OpenAI-compatible request:

```
DFLASH_DS4_SPEC=1 \
DFLASH_DS4_DRAFT=/models/DeepSeek-V4-Flash-DSpark-draft.gguf \
./build-rocm/ember-dflash \
  -m /models/DeepSeek-V4-Flash.gguf \
  --kv-cache-dir /tmp/ember-kv-cache \
  --default-temperature 0.6

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

Before deploying an engine build, run a representative prompt through the
differential validator:

```
./build-rocm/ember-dflash -m /models/model.gguf \
  --kv-cache-dir /tmp/ember-validation-cache \
  --validate-prompt /path/to/representative-prompt.txt --validate-tokens 32
```

The command exits nonzero if greedy AR output differs after snapshot restore or
disk round-trip. With DSpark configured, `spec.checked` confirms that both the
restored and fresh-prefill speculative paths ran; `spec.exact` requires both
greedy token streams to match AR. Prompts of at least 512 tokens also exercise
disk persistence.

## Versioning

Ember uses calendar versions in `YEAR.MONTH.DAY` form without zero-padding.
The root [`VERSION`](VERSION) file is authoritative, Git release tags add a `v`
prefix, and published container tags omit it. For example, source version
`2026.8.9` uses Git tag `v2026.8.9` and container tag `2026.8.9`. See
[`CHANGELOG.md`](CHANGELOG.md) for release notes.

## Memory

The affine model is **85.3 GiB** on a **125 GiB** box (~89 GB resident) — roughly
**36 GiB of headroom**, enough to absorb concurrent large-context load without the
swap-thrash the prior ~96 GiB (Q3-down) build hit. Combined with background-gating
and ember's lean footprint (8 vs lucebox's 32 snapshot slots; ~28 MB process RSS;
no leaks), memory is no longer the operational limiter it once was. The remaining
pressure is transient (e.g. streaming the model file off disk while serving it).
